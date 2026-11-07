"""Layout, checksums, key figures and README of a stability study directory.

Published files sit at the top of the study directory and are distributed with
the repository. Raw results, corpora and logs sit under raw/, stay on the
machine that ran the study, and are excluded from Git and the source
distributions. SHA256SUMS lists both. This module uses only the standard
library, so tools/check_stability_archives.py can import it without NumPy.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

RAW = "raw"
ACCURACY_SECTION = "STABILITY ACCURACY"
SPEED_SECTION = "STABILITY SPEED PERFORMANCE"
SECTION_FILES = {ACCURACY_SECTION: "accuracy-tables.md", SPEED_SECTION: "speed-tables.md"}

PUBLISHED = {
    "README.md": "This description, generated from the results.",
    "SHA256SUMS": "SHA-256 of every file in the study, including the raw files.",
    "study.json": "The two commits, the command, the host and software, the source hashes of the measured commit, and every step with its duration.",
    "builds.json": "Configure and build commands, build records, and the SHA-256 of every library and benchmark runner.",
    "key-figures.json": "The figures quoted in docs/stability.md, computed from the results.",
    "accuracy-tables.md": "The accuracy tables shown in docs/stability.md.",
    "accuracy-summary.csv": "Error statistics for every domain, dimension, family and algorithm.",
    "ties.json": "Exact comparison of the cases whose sequential and balanced errors round to the same binary64 value.",
    "agreement.json": "Bitwise comparison of scalar, SIMD, single and batched products on every stored operand pair.",
    "cc-accuracy-summary.json": "Cariow–Cariowa errors of the two builds: counts of improved, worsened and tied cases, and statistics by dimension and family.",
    "speed-tables.md": "The timing table shown in docs/stability.md.",
    "speed-summary.csv": "Medians and paired ratios of the sequential and balanced timings.",
    "cc-speed-tables.md": "Cariow–Cariowa timings of the two builds, with the direct control.",
    "cc-speed-summary.csv": "Medians and paired ratios of the Cariow–Cariowa, quasilinear and direct timings of the two builds.",
    "block-size-tables.md": "Quasilinear timings for balanced-accumulation block sizes 4, 8 and 16.",
    "block-size-summary.csv": "Medians and paired ratios of the block-size timings.",
    "PLOTS.md": "Description of the six charts and the command that drew them.",
    "plot-data.json": "Every plotted value, with sample keys, counts and bootstrap intervals.",
    "plot-data.csv": "Every plotted value in tabular form.",
}
FIGURES = ("relative-error-real", "relative-error-complex", "speed-real-single", "speed-real-batch",
           "speed-complex-single", "speed-complex-batch")
FIGURE_FORMATS = ("svg", "pdf", "png")
STEPS = {
    "build-baseline-simd": "Build the baseline library.",
    "build-baseline-scalar": "Build the baseline library without the explicit SIMD kernels.",
    "build-balanced-simd": "Build the measured library.",
    "build-balanced-scalar": "Build the measured library without the explicit SIMD kernels.",
    "build-block-4": "Build the measured library with block size 4.",
    "build-block-16": "Build the measured library with block size 16.",
    "accuracy": "Measure every error against the exact product.",
    "accuracy-tables": "Tabulate the accuracy results.",
    "ties": "Compare exactly the cases whose stored errors are equal.",
    "agreement": "Compare scalar, SIMD, single and batched products bit for bit.",
    "cc-accuracy": "Compare the Cariow–Cariowa errors of the two builds.",
    "speed": "Time sequential and balanced accumulation.",
    "speed-tables": "Tabulate the timings.",
    "cc-speed": "Time the Cariow–Cariowa products of the two builds.",
    "block-size": "Time block sizes 4, 8 and 16.",
    "plots": "Measure and draw the six charts.",
}
RAW_FILES = {
    "accuracy.json": "Every case of the accuracy study: operand hashes, seeds, and the error measures of quasilinear multiplication with sequential and with balanced accumulation, direct multiplication and Cariow–Cariowa multiplication.",
    "accuracy-corpus.npz": "The stored operands of the accuracy study.",
    "cc-accuracy.json": "Every compared Cariow–Cariowa case.",
    "speed.json": "Every native timing observation of the sequential and balanced builds.",
    "speed-corpora/": "Operands of the sequential and balanced timings.",
    "cc-speed.json": "Every native timing observation of the Cariow–Cariowa comparison.",
    "cc-speed-corpora/": "Operands of the Cariow–Cariowa timings.",
    "block-size.json": "Every native timing observation of the block-size comparison.",
    "block-size-corpora/": "Operands of the block-size timings.",
    "plot-timings.json": "Every native timing observation behind the speed charts.",
    "plot-corpora/": "Operands of the chart timings.",
    "logs/": "The study log and the output of every step and build.",
}


def published_names() -> list[str]:
    figures = [f"{name}.{extension}" for name in FIGURES for extension in FIGURE_FORMATS]
    return [*PUBLISHED, *figures]


# ---------- Delimited sections of docs/stability.md ----------

def markers(label: str) -> tuple[str, str]:
    return f"<!-- BEGIN {label} -->", f"<!-- END {label} -->"


def extract_section(text: str, label: str) -> str:
    """Return the text between a pair of section markers, normalized."""
    start, end = markers(label)
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"document needs exactly one {start} and one {end}")
    body = text.split(start, 1)[1].split(end, 1)[0]
    return body.strip("\n") + "\n"


def replace_section(text: str, label: str, body: str) -> str:
    start, end = markers(label)
    extract_section(text, label)
    before, rest = text.split(start, 1)
    after = rest.split(end, 1)[1]
    return f"{before}{start}\n\n{body.strip(chr(10))}\n\n{end}{after}"


# ---------- Checksums ----------

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_checksums(directory: Path) -> int:
    """Write SHA256SUMS for every other file in the study directory."""
    files = sorted(p for p in directory.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    lines = [f"{sha256(p)}  {p.relative_to(directory).as_posix()}\n" for p in files]
    (directory / "SHA256SUMS").write_text("".join(lines))
    return len(lines)


def read_checksums(directory: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or len(digest) != 64 or name.startswith("/") or ".." in Path(name).parts:
            raise ValueError(f"malformed SHA256SUMS line in {directory}: {line!r}")
        if name in entries:
            raise ValueError(f"duplicate SHA256SUMS entry in {directory}: {name}")
        entries[name] = digest
    return entries


def is_raw(name: str) -> bool:
    return name == RAW or name.startswith(RAW + "/")


# ---------- Key figures ----------

def _median(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def _case(row: dict[str, Any]) -> dict[str, Any]:
    return {"key": row["key"], "domain": row["domain"], "dimension": row["dimension"], "family": row["family"]}


def accuracy_figures(document: dict[str, Any]) -> dict[str, Any]:
    samples = document["samples"]
    names = {"S": "original_quasilinear", "B": "modified_quasilinear", "D": "direct"}

    def eta(row: dict[str, Any], label: str) -> float | None:
        value: float | None = row["algorithms"][names[label]]["normalized"]
        return value

    figures: dict[str, Any] = {"pairs": len(samples)}
    for domain in ("real", "complex"):
        rows = [r for r in samples if r["domain"] == domain]
        if not rows:
            continue
        dense = [r for r in rows if r["family"] in ("gaussian", "uniform")]
        ratios = []
        for n in sorted({r["dimension"] for r in dense}):
            group = [r for r in dense if r["dimension"] == n]
            ratios.append({"dimension": n, "median_eta_B_over_D":
                           _median([eta(r, "B") for r in group]) / _median([eta(r, "D") for r in group])})
        largest = {label: max((r for r in rows if eta(r, label) is not None), key=lambda r, label=label: eta(r, label))
                   for label in names}
        pooled = [r for r in dense if r["dimension"] >= 32]
        worse = [r for r in rows if not r["improved"] and not r["tied"]]
        by_difference = max(worse, key=lambda r: eta(r, "B") - eta(r, "S"), default=None)
        size = min(512, max(r["dimension"] for r in rows))

        def family_medians(family: str, n: int = size, group: list[dict[str, Any]] = rows) -> dict[str, Any] | None:
            selected = [r for r in group if r["family"] == family and r["dimension"] == n]
            if not selected:
                return None
            return {"dimension": n, "samples": len(selected),
                    **{f"median_eta_{label}": _median([eta(r, label) for r in selected]) for label in names},
                    **{f"median_relative_{label}": _median([r["algorithms"][names[label]]["relative"] for r in selected])
                       for label in names},
                    **{f"median_chi_{label}": _median([r["algorithms"][names[label]]["chi"] for r in selected])
                       for label in names}}

        figures[domain] = {
            "pairs": len(rows), "families": len({r["family"] for r in rows}),
            "dense_median_eta_B_over_D": ratios,
            "dense_median_eta_B_over_D_maximum": max(ratios, key=lambda r: r["median_eta_B_over_D"]) if ratios else None,
            "largest_eta": {label: {**_case(row), "eta": eta(row, label)} for label, row in largest.items()},
            "dense_from_32": {"pairs": len(pooled),
                              "B_better": sum(r["improved"] for r in pooled) / len(pooled) if pooled else None,
                              "B_worse": sum(not r["improved"] and not r["tied"] for r in pooled) / len(pooled) if pooled else None},
            "largest_increase_B_over_S": None if by_difference is None else
            {**_case(by_difference), "eta_S": eta(by_difference, "S"), "eta_B": eta(by_difference, "B")},
            "long_sum_cancellation": family_medians("long_sum_cancellation"),
            "repeated_blocks": family_medians("repeated_blocks"),
            "near_zero_divisor_52": family_medians("near_zero_divisor_52"),
        }
    return figures


def speed_figures(result: dict[str, Any]) -> dict[str, Any]:
    rows = result["summary"]

    def span(selected: list[dict[str, Any]], key: str) -> list[float] | None:
        values = [r[key] for r in selected]
        return [min(values), max(values)] if values else None

    def pick(domain: str, backend: str, minimum: int = 0, n: int | None = None) -> list[dict[str, Any]]:
        return [r for r in rows if r["domain"] == domain and r["backend"] == backend
                and r["dimension"] >= minimum and (n is None or r["dimension"] == n)]

    dimensions = sorted({r["dimension"] for r in rows if r["domain"] == "complex"})
    largest_real = max((r["dimension"] for r in rows if r["domain"] == "real"), default=None)
    return {
        "ratios": [{key: r[key] for key in ("domain", "backend", "mode", "dimension", "balanced_over_sequential",
                                           "balanced_over_sequential_95ci", "direct_control")} for r in rows],
        "real_simd_from_64": span(pick("real", "simd", 64), "balanced_over_sequential"),
        "real_scalar_from_64": span(pick("real", "scalar", 64), "balanced_over_sequential"),
        "complex_by_dimension": {str(n): span([r for r in rows if r["domain"] == "complex" and r["dimension"] == n],
                                              "balanced_over_sequential") for n in dimensions},
        "direct_control_real": span([r for r in rows if r["domain"] == "real"], "direct_control"),
        "direct_control_complex": span([r for r in rows if r["domain"] == "complex"], "direct_control"),
        "direct_over_balanced_real_largest": None if largest_real is None else
        {"dimension": largest_real, **{r["mode"]: r["direct_over_balanced"] for r in pick("real", "simd", n=largest_real)}},
    }


def cc_speed_figures(result: dict[str, Any]) -> dict[str, Any]:
    rows = [r for r in result["summary"] if r["algorithm"] == "cariow_cariowa"]
    controls = [r["measured_over_baseline"] for r in result["summary"] if r["algorithm"] == "direct"]
    return {
        "ratios": [{key: r[key] for key in ("backend", "mode", "dimension", "measured_over_baseline",
                                           "measured_over_baseline_95ci")} for r in rows],
        "direct_control": [min(controls), max(controls)] if controls else None,
    }


def block_size_figures(result: dict[str, Any]) -> dict[str, Any]:
    figures: dict[str, Any] = {}
    for size in result["block_sizes"]:
        if size == result["reference_block_size"]:
            continue
        key = f"ratio_{size}_over_{result['reference_block_size']}"
        for domain in ("real", "complex"):
            values = [r[key] for r in result["summary"] if r["domain"] == domain]
            if values:
                figures[f"{key}_{domain}"] = [min(values), max(values)]
        figures[key] = [{"domain": r["domain"], "mode": r["mode"], "dimension": r["dimension"], "ratio": r[key]}
                        for r in result["summary"]]
    return figures


def key_figures(directory: Path) -> dict[str, Any]:
    """Collect the figures quoted in docs/stability.md from a finished study."""
    def load(name: str) -> dict[str, Any]:
        data: dict[str, Any] = json.loads((directory / name).read_text())
        return data

    cc_accuracy = load("cc-accuracy-summary.json")
    ties = load("ties.json")
    agreement = load("agreement.json")
    return {
        "schema": "fastcd-stability-key-figures-1",
        "accuracy": accuracy_figures(load(f"{RAW}/accuracy.json")),
        "ties": {key: ties[key] for key in ("recorded_ties", "identical_outputs", "equal_exact_errors",
                                            "balanced_smaller", "sequential_smaller", "unresolved_by_family")},
        "cc_accuracy": {key: cc_accuracy[key] for key in ("aggregate", "dense_by_dimension", "largest_baseline",
                                                          "largest_measured", "largest_increase")},
        "speed": speed_figures(load(f"{RAW}/speed.json")),
        "cc_speed": cc_speed_figures(load(f"{RAW}/cc-speed.json")),
        "block_size": block_size_figures(load(f"{RAW}/block-size.json")),
        "agreement": {"counts": agreement["counts"], "mismatches": len(agreement["mismatches"])},
    }


# ---------- README ----------

def _duration(seconds: float) -> str:
    minutes, rest = divmod(round(seconds), 60)
    return f"{minutes} min {rest:02d} s" if minutes else f"{rest} s"


def _build_line(record: str, key: str) -> str:
    for line in record.splitlines():
        if line.startswith(key + "="):
            return line.split("=", 1)[1]
    return ""


def readme(directory: Path) -> str:
    """Generate the README of a finished study directory."""
    study = json.loads((directory / "study.json").read_text())
    builds = json.loads((directory / "builds.json").read_text())
    environment = study["environment"]
    record = builds["builds"]["balanced-simd"]["build_record"]
    flags = " ".join(part for part in (_build_line(record, "configuration_flags"), _build_line(record, "options")) if part)
    baseline, measured = study["baseline_commit"], study["commit"]
    lines = [f"# Stability study, {study['date']}", ""]
    if study["profile"] != "full":
        lines += [f"This is a **{study['profile']}** run, made to test the tooling. Its numbers are not results.", ""]
    lines += [
        "This directory holds the measurements behind "
        "[docs/stability.md](../../docs/stability.md), which defines the error measures "
        "and interprets the results.", "",
        "## What was compared", "",
        f"- **Sequential accumulation:** commit `{baseline}`, the baseline.",
        f"- **Balanced accumulation:** commit `{measured}`, the measured commit.", "",
        "Errors are measured against the exact product. Both commits contain the same direct "
        "multiplication, which is measured alongside for comparison, and the accuracy study "
        "checks that the two builds give bit-for-bit identical direct products on every stored "
        "operand pair. Direct timings are therefore the control for noise between the two "
        "builds. The study also compares the Cariow–Cariowa correction sums of the two commits "
        "and times the balanced-accumulation block sizes 4, 8 and 16 at the measured commit.", "",
        "## Host and software", "",
        "| | |", "|:---|:---|",
        f"| CPU | {environment['cpu_model']} |",
        f"| Operating system | {environment['platform']} |",
        f"| Compiler | {environment.get('compiler') or _build_line(record, 'version')} (`{_build_line(record, 'compiler')}`) |",
        f"| Library flags | `{flags}` |",
        f"| Python | {environment['python'].split()[0]} |",
        f"| NumPy, gmpy2, MPFR | {environment['numpy']}, {environment['gmpy2']}, {environment['mpfr']} |",
        f"| Matplotlib | {environment['matplotlib']} |",
        f"| Logical CPU for the timings | {study['timing_cpu']} |",
        f"| Logical CPU for the accuracy study | {study['accuracy_cpu'] if study['accuracy_cpu'] is not None else 'not pinned'} |",
        "",
        "## How it was produced", "",
        "`benchmarks/stability_study.py` built both commits from clean detached worktrees "
        "and ran every step below with "
        + ("the scripts of the measured commit." if study["tools"] == "measured commit"
           else f"the scripts in `{study['tools']}` rather than those of the measured commit."), "",
        "```sh", study["command_line"], "```", "",
        "| Step | What it does | Duration |", "|:---|:---|---:|",
    ]
    for name, step in study["steps"].items():
        lines.append(f"| {name} | {STEPS.get(name, '')} | {_duration(step['seconds'])} |")
    lines += ["", "## Published files", "",
              "These files are part of the repository and the source distributions.", "",
              "| File | Contents |", "|:---|:---|"]
    for name, description in PUBLISHED.items():
        # README.md and SHA256SUMS are written after this text is generated.
        if name in ("README.md", "SHA256SUMS") or (directory / name).exists():
            lines.append(f"| `{name}` | {description} |")
    if all((directory / f"{name}.{extension}").exists() for name in FIGURES for extension in FIGURE_FORMATS):
        lines.append("| `relative-error-*.{svg,pdf,png}`, `speed-*.{svg,pdf,png}` | "
                     "The six charts described in PLOTS.md, in three formats. |")
    lines += ["", "## Raw files", "",
              f"Files under `{RAW}/` stay on the machine that ran the study. They are not "
              "included in Git or in the source distributions, but SHA256SUMS lists them.", "",
              "| Path | Contents |", "|:---|:---|"]
    for name, description in RAW_FILES.items():
        if (directory / RAW / name.rstrip("/")).exists():
            lines.append(f"| `{RAW}/{name}` | {description} |")
    lines += ["", "## Verification", "",
              "From the repository root:", "",
              "```sh", "python3 tools/check_stability_archives.py", "```", "",
              "The checker compares every published file with SHA256SUMS, checks that the "
              "tables in docs/stability.md are the tables in this directory, and, when the git "
              "history is available, checks that the measured commit contains exactly the "
              "source files recorded in study.json. With `--full` it also requires the raw "
              "files, checks their hashes and regenerates the tables from the raw results.", ""]
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("key-figures", "readme", "checksums"))
    parser.add_argument("directory", type=Path, help="study directory")
    args = parser.parse_args()
    if args.action == "key-figures":
        figures = key_figures(args.directory)
        (args.directory / "key-figures.json").write_text(json.dumps(figures, indent=2, allow_nan=False) + "\n")
    elif args.action == "readme":
        (args.directory / "README.md").write_text(readme(args.directory))
    else:
        print(f"{write_checksums(args.directory)} files listed in {args.directory / 'SHA256SUMS'}")


if __name__ == "__main__":
    main()
