#!/usr/bin/env python3
"""Produce a complete stability study from two commits.

The baseline commit has sequential accumulation; the measured commit has
balanced accumulation. Both are checked out as clean detached worktrees under
--work-dir and built there. The study directory receives the published files
at its top level and the raw results, corpora and logs under raw/.

Every step records its inputs and outputs in study.json. Rerunning the same
command skips each measurement step whose recorded inputs and outputs are
unchanged, so an interrupted study resumes where it stopped. The key figures,
the README and SHA256SUMS are regenerated on every run.

Requires NumPy, gmpy2 and Matplotlib in the Python that runs this script.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROFILES: dict[str, dict[str, Any]] = {
    "full": {
        "accuracy": ["--profile", "full", "--real-max", "2048", "--complex-max", "512"],
        "speed": ["--blocks", "12", "--minimum-ms", "20", "--real-dimensions", "16", "64", "256", "1024",
                  "--complex-dimensions", "16", "64", "256"],
        "cc-speed": ["--blocks", "12", "--minimum-ms", "20", "--trials", "32",
                     "--dimensions", "16", "64", "128", "256", "512", "1024"],
        "block-size": ["--blocks", "4", "--minimum-ms", "10", "--trials", "16",
                       "--real-dimensions", "64", "256", "1024", "--complex-dimensions", "64", "256"],
        "plots": ["--blocks", "12", "--minimum-ms", "20", "--trials", "32", "--real-max", "1024",
                  "--complex-max", "512"],
    },
    "smoke": {
        "accuracy": ["--profile", "smoke", "--real-max", "32", "--complex-max", "16"],
        "speed": ["--blocks", "2", "--minimum-ms", "1", "--real-dimensions", "16", "64",
                  "--complex-dimensions", "16"],
        "cc-speed": ["--blocks", "12", "--minimum-ms", "0.5", "--trials", "4", "--dimensions", "16", "64"],
        "block-size": ["--blocks", "3", "--minimum-ms", "1", "--trials", "4",
                       "--real-dimensions", "64", "--complex-dimensions", "64"],
        "plots": ["--blocks", "4", "--minimum-ms", "1", "--trials", "4", "--real-max", "64",
                  "--complex-max", "32"],
    },
}
BUILDS = {
    "baseline-simd": ("baseline", None),
    "baseline-scalar": ("baseline", "FASTCD_DISABLE_SIMD=1"),
    "balanced-simd": ("measured", None),
    "balanced-scalar": ("measured", "FASTCD_DISABLE_SIMD=1"),
    "block-4": ("measured", "FASTCD_SUM_BLOCK=4"),
    "block-16": ("measured", "FASTCD_SUM_BLOCK=16"),
}
FIGURES = [f"{name}.{extension}" for name in (
    "relative-error-real", "relative-error-complex", "speed-real-single", "speed-real-batch",
    "speed-complex-single", "speed-complex-batch") for extension in ("svg", "pdf", "png")]


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*arguments: str, cwd: Path) -> str:
    return subprocess.check_output(["git", *arguments], cwd=cwd, text=True).strip()


def source_hashes(root: Path) -> dict[str, str]:
    """SHA-256 of the library sources, with the same file set as stability.py."""
    files = sorted([*root.glob("src/**/*.c"), *root.glob("src/**/*.h"), *root.glob("include/**/*.h")])
    return {p.relative_to(root).as_posix(): sha256(p) for p in files}


class Study:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.repository = Path(git("rev-parse", "--show-toplevel", cwd=Path.cwd()))
        self.output = args.output.resolve()
        self.raw = self.output / "raw"
        self.logs = self.raw / "logs"
        self.work = args.work_dir.resolve()
        self.logs.mkdir(parents=True, exist_ok=True)
        self.baseline = git("rev-parse", "--verify", args.baseline_commit + "^{commit}", cwd=self.repository)
        self.commit = git("rev-parse", "--verify", args.commit + "^{commit}", cwd=self.repository)
        self.sources = {"baseline": self.work / "baseline-source", "measured": self.work / "measured-source"}
        self.tools = self.sources["measured"] if args.tools is None else args.tools.resolve()
        self.parameters = PROFILES[args.profile]
        self.record_file = self.output / "study.json"
        if self.record_file.exists():
            self.record = json.loads(self.record_file.read_text())
            for key, value in (("baseline_commit", self.baseline), ("commit", self.commit), ("profile", args.profile)):
                if self.record[key] != value:
                    raise SystemExit(f"{self.record_file} belongs to a study with a different {key}")
        else:
            self.record = {"schema": "fastcd-stability-study-1", "date": dt.date.today().isoformat(),
                           "profile": args.profile, "baseline_commit": self.baseline, "commit": self.commit,
                           "steps": {}}
        self.record.update({
            "command": [sys.executable, *sys.argv],
            "command_line": shlex.join(["python3", *sys.argv]),
            "tools": "measured commit" if args.tools is None else str(args.tools),
            "timing_cpu": args.cpu, "accuracy_cpu": args.accuracy_cpu,
        })

    # ---------- bookkeeping ----------

    def relative(self, path: Path) -> str:
        for base in (self.output, self.repository):
            try:
                return path.resolve().relative_to(base).as_posix()
            except ValueError:
                continue
        return str(path)

    def log(self, message: str) -> None:
        line = f"[{now()}] {message}"
        print(line, flush=True)
        with (self.logs / "study.log").open("a") as log:
            log.write(line + "\n")

    def save(self) -> None:
        self.record_file.write_text(json.dumps(self.record, indent=2, allow_nan=False) + "\n")

    def hashes(self, paths: list[Path]) -> dict[str, str] | None:
        if not all(p.is_file() for p in paths):
            return None
        return {self.relative(p): sha256(p) for p in paths}

    def complete(self, name: str, inputs: list[Path], outputs: list[Path]) -> bool:
        step = self.record["steps"].get(name)
        return bool(step) and step["inputs"] == self.hashes(inputs) and step["outputs"] == self.hashes(outputs)

    def run(self, name: str, command: list[str], cpu: int | None = None, cwd: Path | None = None) -> None:
        self.log(f"{name}: {shlex.join(command)}")
        preexec = (lambda: os.sched_setaffinity(0, {cpu})) if cpu is not None else None
        with (self.logs / f"{name}.log").open("a") as log:
            log.write(f"[{now()}] {shlex.join(command)}\n")
            log.flush()
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=cwd, preexec_fn=preexec)

    def step(self, name: str, command: list[str], inputs: list[Path], outputs: list[Path],
             cpu: int | None = None) -> None:
        if self.complete(name, inputs, outputs):
            self.log(f"{name}: complete, skipped")
            return
        started, clock = now(), time.monotonic()
        self.run(name, command, cpu)
        recorded = self.hashes(outputs)
        if recorded is None:
            missing = [self.relative(p) for p in outputs if not p.is_file()]
            raise SystemExit(f"{name} did not produce {missing}")
        self.record["steps"][name] = {"started": started, "finished": now(),
                                      "seconds": round(time.monotonic() - clock, 1),
                                      "command": command, "inputs": self.hashes(inputs), "outputs": recorded}
        self.save()

    def tool(self, name: str) -> list[str]:
        return [sys.executable, str(self.tools / "benchmarks" / name)]

    # ---------- sources and builds ----------

    def worktrees(self) -> None:
        for role, commit in (("baseline", self.baseline), ("measured", self.commit)):
            path = self.sources[role]
            if path.exists():
                if git("rev-parse", "HEAD", cwd=path) != commit or git("status", "--porcelain", cwd=path):
                    raise SystemExit(f"{path} is not a clean checkout of {commit}; remove it and rerun")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            self.log(f"worktree {role}: {commit}")
            subprocess.run(["git", "worktree", "add", "--detach", str(path), commit], cwd=self.repository,
                           check=True, capture_output=True)
        if self.args.tools is not None and not (self.tools / "benchmarks" / "stability.py").exists():
            raise SystemExit(f"--tools {self.tools} has no benchmarks/stability.py")

    def build_directory(self, name: str) -> Path:
        return self.work / "builds" / name

    def library(self, name: str) -> Path:
        candidates = [p for p in self.build_directory(name).glob("libfastcd.so.*") if p.is_file() and not p.is_symlink()]
        if len(candidates) != 1:
            raise SystemExit(f"expected one versioned library in {self.build_directory(name)}")
        return candidates[0]

    def builds(self) -> dict[str, Any]:
        records = {}
        for name, (role, definitions) in BUILDS.items():
            directory = self.build_directory(name)
            configure = ["cmake", "-S", str(self.sources[role]), "-B", str(directory), "-DCMAKE_BUILD_TYPE=Release",
                         "-DBUILD_SHARED_LIBS=ON", "-DBUILD_TESTING=OFF", "-DFASTCD_BUILD_BENCHMARK=ON",
                         "-DFASTCD_WARNINGS_AS_ERRORS=ON"]
            if definitions:
                configure.append(f"-DFASTCD_TUNING_DEFINITIONS={definitions}")
            build = ["cmake", "--build", str(directory), "--parallel", str(self.args.jobs)]
            stamp = directory / "stability-build.json"
            expected = {"commit": self.baseline if role == "baseline" else self.commit, "configure": configure}
            if not (stamp.exists() and json.loads(stamp.read_text()) == expected):
                started, clock = now(), time.monotonic()
                self.run(f"build-{name}", configure)
                self.run(f"build-{name}", build)
                stamp.write_text(json.dumps(expected, indent=2) + "\n")
                self.record["steps"][f"build-{name}"] = {"started": started, "finished": now(),
                                                         "seconds": round(time.monotonic() - clock, 1),
                                                         "command": [configure, build], "inputs": None, "outputs": None}
                self.save()
            else:
                self.log(f"build-{name}: complete, skipped")
            files = sorted(p for p in directory.iterdir() if p.is_file() and not p.is_symlink()
                           and (p.name in ("fastcd_benchmark", "fastcd_domains_benchmark")
                                or p.name.startswith("libfastcd.so.")))
            records[name] = {"source": role, "commit": expected["commit"], "tuning_definitions": definitions,
                             "directory": self.relative(directory),
                             "configure_command": configure, "build_command": build,
                             "build_record": (directory / "fastcd-build-Release.txt").read_text(),
                             "sha256": {p.name: sha256(p) for p in files}}
        return records

    # ---------- the study ----------

    def execute(self) -> None:
        self.worktrees()
        self.record["environment"] = environment()
        self.record["source"] = {role: {"commit": self.baseline if role == "baseline" else self.commit,
                                        "files_sha256": source_hashes(path)}
                                 for role, path in self.sources.items()}
        self.save()
        builds = self.builds()
        (self.output / "builds.json").write_text(json.dumps({"schema": "fastcd-stability-builds-1",
                                                             "builds": builds}, indent=2) + "\n")
        out, raw, p = self.output, self.raw, self.parameters
        directory = self.build_directory
        base, balanced, scalar = self.library("baseline-simd"), self.library("balanced-simd"), self.library("balanced-scalar")
        accuracy, corpus = raw / "accuracy.json", raw / "accuracy-corpus.npz"
        accuracy_cpu = [] if self.args.accuracy_cpu is None else ["--cpu", str(self.args.accuracy_cpu)]
        timing_cpu = ["--cpu", str(self.args.cpu)]
        self.step("accuracy", [*self.tool("stability.py"), "--baseline", str(base), "--modified", str(balanced),
                               "--scalar-modified", str(scalar), "--baseline-source", str(self.sources["baseline"]),
                               "--modified-source", str(self.sources["measured"]), *p["accuracy"],
                               "--precision", "256", "--seed", "20261105", "--output", str(accuracy), *accuracy_cpu],
                  [base, balanced, scalar], [accuracy, corpus])
        analysis = [accuracy, corpus, base, balanced]
        self.step("accuracy-tables", [*self.tool("stability_report.py"), str(accuracy),
                                      "--tables", str(out / "accuracy-tables.md"),
                                      "--csv", str(out / "accuracy-summary.csv")],
                  [accuracy], [out / "accuracy-tables.md", out / "accuracy-summary.csv"])
        self.step("ties", [*self.tool("stability_ties.py"), "--accuracy", str(accuracy), "--sequential", str(base),
                           "--balanced", str(balanced), "--output", str(out / "ties.json")],
                  analysis, [out / "ties.json"], self.args.accuracy_cpu)
        self.step("agreement", [*self.tool("stability_agreement.py"), "--accuracy", str(accuracy),
                                "--simd", str(balanced), "--scalar", str(scalar), "--output", str(out / "agreement.json")],
                  [*analysis, scalar], [out / "agreement.json"], self.args.accuracy_cpu)
        self.step("cc-accuracy", [*self.tool("stability_cc_accuracy.py"), "--accuracy", str(accuracy),
                                  "--baseline", str(base), "--measured", str(balanced),
                                  "--output", str(raw / "cc-accuracy.json"),
                                  "--summary", str(out / "cc-accuracy-summary.json")],
                  analysis, [raw / "cc-accuracy.json", out / "cc-accuracy-summary.json"], self.args.accuracy_cpu)
        timed = [self.library(name) for name in ("baseline-simd", "baseline-scalar", "balanced-simd", "balanced-scalar")]
        self.step("speed", [*self.tool("stability_speed.py"),
                            "--sequential", str(directory("baseline-simd")),
                            "--sequential-scalar", str(directory("baseline-scalar")),
                            "--balanced", str(directory("balanced-simd")),
                            "--balanced-scalar", str(directory("balanced-scalar")),
                            "--output", str(raw / "speed.json"), "--corpora", str(raw / "speed-corpora"),
                            *p["speed"], *timing_cpu],
                  timed, [raw / "speed.json"])
        self.step("speed-tables", [*self.tool("stability_speed_report.py"), str(raw / "speed.json"),
                                   "--tables", str(out / "speed-tables.md"), "--csv", str(out / "speed-summary.csv")],
                  [raw / "speed.json"], [out / "speed-tables.md", out / "speed-summary.csv"])
        self.step("cc-speed", [*self.tool("stability_cc_performance.py"),
                               "--baseline", str(directory("baseline-simd")),
                               "--baseline-scalar", str(directory("baseline-scalar")),
                               "--measured", str(directory("balanced-simd")),
                               "--measured-scalar", str(directory("balanced-scalar")),
                               "--output", str(raw / "cc-speed.json"), "--corpora", str(raw / "cc-speed-corpora"),
                               "--tables", str(out / "cc-speed-tables.md"), "--csv", str(out / "cc-speed-summary.csv"),
                               *p["cc-speed"], *timing_cpu],
                  timed, [raw / "cc-speed.json", out / "cc-speed-tables.md", out / "cc-speed-summary.csv"])
        blocks = [self.library("block-4"), balanced, self.library("block-16")]
        self.step("block-size", [*self.tool("stability_block_size.py"),
                                 "--build", f"4={directory('block-4')}", "--build", f"8={directory('balanced-simd')}",
                                 "--build", f"16={directory('block-16')}",
                                 "--output", str(raw / "block-size.json"), "--corpora", str(raw / "block-size-corpora"),
                                 "--tables", str(out / "block-size-tables.md"),
                                 "--csv", str(out / "block-size-summary.csv"), *p["block-size"], *timing_cpu],
                  blocks, [raw / "block-size.json", out / "block-size-tables.md", out / "block-size-summary.csv"])
        self.step("plots", [*self.tool("stability_plots.py"), "--accuracy", str(accuracy),
                            "--build-dir", str(directory("balanced-simd")), "--output", str(out), "--raw", str(raw),
                            "--source", str(self.sources["measured"]), "--measure",
                            "--real-algorithms", "quasilinear", "direct", "cariow_cariowa",
                            *p["plots"], *timing_cpu],
                  [accuracy, balanced],
                  [raw / "plot-timings.json", out / "PLOTS.md", out / "plot-data.json", out / "plot-data.csv",
                   *(out / name for name in FIGURES)])
        self.finish()

    def finish(self) -> None:
        archive = [*self.tool("stability_archive.py")]
        self.run("key-figures", [*archive, "key-figures", str(self.output)])
        if self.args.document:
            self.update_document(self.args.document.resolve())
        self.record["finished"] = now()
        self.save()
        self.run("readme", [*archive, "readme", str(self.output)])
        self.log(f"study complete: {self.output}; writing SHA256SUMS")
        # SHA256SUMS covers raw/logs too, so nothing may write there afterwards
        listed = subprocess.run([*archive, "checksums", str(self.output)], check=True, capture_output=True, text=True)
        print(listed.stdout.strip(), flush=True)

    def update_document(self, document: Path) -> None:
        sys.path.insert(0, str(self.tools / "benchmarks"))
        import stability_archive as archive

        text = document.read_text()
        for label, name in archive.SECTION_FILES.items():
            text = archive.replace_section(text, label, (self.output / name).read_text())
        document.write_text(text)
        self.log(f"document: updated {self.relative(document)}")

    def cleanup(self) -> None:
        """Remove the worktrees and builds; prints only, since the study log is sealed."""
        for path in self.sources.values():
            if path.exists():
                print(f"cleanup: removing worktree {path}", flush=True)
                subprocess.run(["git", "worktree", "remove", "--force", str(path)], cwd=self.repository, check=True)
        if (self.work / "builds").exists():
            print(f"cleanup: removing {self.work / 'builds'}", flush=True)
            shutil.rmtree(self.work / "builds")


def environment() -> dict[str, Any]:
    import gmpy2
    import matplotlib
    import numpy

    cpu_model = None
    if Path("/proc/cpuinfo").exists():
        cpu_model = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                          if line.startswith("model name")), None)
    compiler = os.environ.get("CC", "cc")
    try:
        identity = subprocess.run([compiler, "--version"], capture_output=True, text=True,
                                  check=True).stdout.splitlines()[0].strip()
    except (OSError, subprocess.CalledProcessError, IndexError):
        identity = None
    try:
        blas = numpy.show_config(mode="dicts")["Build Dependencies"]["blas"]
        numpy_blas = " ".join(str(blas[key]) for key in ("name", "version") if blas.get(key))
    except (KeyError, TypeError):
        numpy_blas = None
    return {"python": sys.version, "platform": platform.platform(), "cpu_model": cpu_model or platform.processor(),
            "compiler": identity, "numpy": numpy.__version__, "numpy_blas": numpy_blas, "gmpy2": gmpy2.version(),
            "mpfr": gmpy2.mpfr_version(), "matplotlib": matplotlib.__version__}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-commit", required=True, help="commit with sequential accumulation")
    parser.add_argument("--commit", default="HEAD", help="measured commit (default: HEAD)")
    parser.add_argument("--output", type=Path, required=True, help="study directory, e.g. benchmark-results/stability-YYYYMMDD")
    parser.add_argument("--work-dir", type=Path, default=Path("build/stability-work"),
                        help="directory for the worktrees and builds (default: build/stability-work)")
    parser.add_argument("--cpu", type=int, required=True, help="CPU for every timing step")
    parser.add_argument("--accuracy-cpu", type=int, help="CPU for the accuracy steps (default: not pinned)")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="full")
    parser.add_argument("--tools", type=Path,
                        help="tree whose benchmarks/ scripts run the study (default: the measured commit's worktree)")
    parser.add_argument("--document", type=Path, help="docs/stability.md, to update its delimited tables")
    parser.add_argument("--jobs", type=int, default=8, help="parallel build jobs")
    parser.add_argument("--cleanup", action="store_true",
                        help="remove the worktrees and builds after the study completes")
    args = parser.parse_args()
    if args.cpu == args.accuracy_cpu:
        parser.error("use different CPUs for timings and the accuracy steps")
    study = Study(args)
    try:
        study.execute()
    except subprocess.CalledProcessError as error:
        study.log(f"failed: {shlex.join(map(str, error.cmd))} (exit {error.returncode}); see {study.logs}")
        raise SystemExit(1) from error
    if args.cleanup:
        study.cleanup()


if __name__ == "__main__":
    main()
