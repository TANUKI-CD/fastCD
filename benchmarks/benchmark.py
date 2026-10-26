#!/usr/bin/env python3
"""Compare fastCD's three multiplication algorithms and report the crossover.

The script generates (or reuses) one deterministic corpus of random operand
pairs per dimension, then drives the C runner ``fastcd_benchmark`` in fresh
subprocesses.  Every algorithm is timed in two modes over the same pairs:
``batch`` (one validated batch call per corpus pass, which uses the paired
SIMD kernels where the build supports them) and ``single`` (one validated
single-product call per pair).  Each block runs every (algorithm, mode) task
once in a rotated order, so every task occupies every position equally often.
Block observations are summarized with means, two-sided 95% Student-t
intervals, paired ratios against the quasilinear algorithm, and the paired
single/batch ratio of each algorithm.  All pairs are verified against the
independent oracle in the first block of each task.  Complete order balance
requires a block count divisible by the task count (six where all three
algorithms are available, four above the Cariow--Cariowa limit); the default
of 24 blocks satisfies both, and multiples of 12 keep that property.

Example (from a configured build directory ``build``):

    python3 benchmarks/benchmark.py --runner build/fastcd_benchmark \
        --output benchmark-results/results-current.json --cpu 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ALGORITHMS = ("direct", "cariow_cariowa", "quasilinear")
MODES = ("batch", "single")
BASELINE = "quasilinear"
CC_MAX_DIMENSION = 1024
# Cariow--Cariowa is timed at every dimension the library supports by default.
# Above 64 it runs the table-driven kernel, whose time reflects the gathered
# sign corrections rather than the operation count (see docs/benchmark.md);
# --max-cc-dimension restricts the timed range when that is not wanted.
CC_TIMING_CAP = CC_MAX_DIMENSION

# Two-sided 95% Student-t critical values t(0.975, df) for df = 1..30.  Larger
# degrees of freedom deliberately use the df = 30 value (slightly conservative).
T95 = (
    float("nan"), 12.7062047364, 4.30265272975, 3.18244630528, 2.77644510520,
    2.57058183564, 2.44691185114, 2.36462425101, 2.30600413503, 2.26215716285,
    2.22813885196, 2.20098516008, 2.17881282966, 2.16036865646, 2.14478668792,
    2.13144954556, 2.11990529922, 2.10981557783, 2.10092204024, 2.09302405441,
    2.08596344727, 2.07961384473, 2.07387306790, 2.06865761042, 2.06389856163,
    2.05953855275, 2.05552943864, 2.05183051648, 2.04840714180, 2.04522964213,
    2.04227245630,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runner", type=Path, required=True,
                        help="path to the built fastcd_benchmark executable")
    parser.add_argument("--build-metadata", type=Path,
                        help="fastcd-build-CONFIG.txt (default: beside the runner)")
    parser.add_argument("--output", type=Path, required=True,
                        help="JSON result path; a Markdown table is written next to it")
    parser.add_argument("--dimensions", default="2,4,8,16,32,64,128,256,512,1024",
                        help="comma-separated powers of two")
    parser.add_argument("--trials", type=int, default=10000,
                        help="multiplication pairs per dimension")
    parser.add_argument("--blocks", type=int, default=24,
                        help="subprocess blocks; each yields one observation per task "
                             "(multiples of 12 keep the task order fully balanced)")
    parser.add_argument("--modes", default=",".join(MODES),
                        help="comma-separated subset of batch,single")
    parser.add_argument("--warmup", type=int, default=2, help="untimed corpus passes")
    parser.add_argument("--min-time-ms", type=float, default=20.0,
                        help="minimum duration of one observation")
    parser.add_argument("--seed", type=int, default=20260906, help="master corpus seed")
    parser.add_argument("--norm-min", type=float, default=0.5)
    parser.add_argument("--norm-max", type=float, default=2.0)
    parser.add_argument("--corpora", type=Path, default=Path("benchmark-corpora"),
                        help="directory holding corpus-nN.f64 files and manifest.json")
    parser.add_argument("--max-direct-dimension", type=int, default=1024,
                        help="largest dimension at which the quadratic direct product is timed")
    parser.add_argument("--max-cc-dimension", type=int, default=CC_TIMING_CAP,
                        help="largest dimension at which Cariow--Cariowa is timed "
                             "(default: every dimension the library supports)")
    parser.add_argument("--verify-trials", type=int, default=None,
                        help="pairs verified in the first block (default: all)")
    parser.add_argument("--cpu", type=int, default=None,
                        help="logical CPU (Linux default: first allowed CPU)")
    parser.add_argument("--subprocess-timeout", type=float, default=3600.0,
                        help="seconds allowed for one runner invocation")
    arguments = parser.parse_args()
    if arguments.trials < 1 or arguments.blocks < 1 or arguments.warmup < 0:
        parser.error("trials and blocks must be positive; warmup must be nonnegative")
    if not math.isfinite(arguments.min_time_ms) or arguments.min_time_ms < 0:
        parser.error("min-time-ms must be finite and nonnegative")
    if not (0 < arguments.norm_min <= arguments.norm_max < math.inf):
        parser.error("norm bounds must be finite and 0 < norm-min <= norm-max")
    if arguments.verify_trials is not None and not 1 <= arguments.verify_trials <= arguments.trials:
        parser.error("verify-trials must be between 1 and trials")
    modes = arguments.modes.split(",")
    if any(mode not in MODES for mode in modes):
        parser.error("modes must be a comma-separated subset of batch,single")
    arguments.modes = tuple(mode for mode in MODES if mode in modes)
    if arguments.output.suffix.lower() == ".md":
        parser.error("output must not use a .md suffix; JSON and Markdown "
                     "paths must differ")
    return arguments


def cpu_state(cpu: int | None) -> dict[str, Any]:
    """Record frequency-scaling and SMT settings that affect timing, when exposed."""

    base = Path("/sys/devices/system/cpu")
    files: dict[str, Path | None] = {
        "scaling_governor": (base / f"cpu{cpu}" / "cpufreq" / "scaling_governor"
                             if cpu is not None else None),
        "intel_pstate_no_turbo": base / "intel_pstate" / "no_turbo",
        "cpufreq_boost": base / "cpufreq" / "boost",
        "smt_control": base / "smt" / "control",
    }
    state: dict[str, Any] = {}
    for key, path in files.items():
        try:
            state[key] = None if path is None else path.read_text().strip()
        except OSError:
            state[key] = None
    return state


def cpu_name() -> str:
    """Return the CPU model name, using /proc/cpuinfo on Linux."""

    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def dimension_seed(master: int, dimension: int) -> list[int]:
    return [master, dimension]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(root: Path) -> dict[str, Any]:
    """Return the checked-out commit and whether tracked files differ from it.

    Every field is None when *root* is not inside a git checkout or git is
    unavailable, so that runs from an extracted source archive still work.
    """

    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(["git", "-C", str(root), *args],
                                       capture_output=True, text=True, check=True)
        except (OSError, subprocess.CalledProcessError):
            return None
        return completed.stdout.strip()

    commit = run("rev-parse", "HEAD")
    if not commit:
        return {"commit": None, "dirty": None, "describe": None}
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit,
            "dirty": None if status is None else bool(status),
            "describe": run("describe", "--tags", "--always", "--dirty")}


def git_summary(state: dict[str, Any]) -> str:
    """Describe a git state in one clause for the report header."""

    if not state.get("commit"):
        return "Commit: unknown (not a git checkout)"
    tree = ("with uncommitted changes to tracked files" if state["dirty"]
            else "clean tree")
    return f"Commit: {state['commit']} ({tree})"


def provenance(arguments: argparse.Namespace) -> dict[str, Any]:
    """Bind the executable and build options to the source tree being tested."""

    runner = arguments.runner.resolve()
    metadata = arguments.build_metadata
    if metadata is None:
        candidates = list(runner.parent.glob("fastcd-build-*.txt"))
        if len(candidates) != 1:
            raise SystemExit("rebuild the runner with CMake and supply --build-metadata "
                             "if its build record is not unique or is in another directory")
        metadata = candidates[0]
    record = metadata.read_text()
    build_id = run_json([str(runner), "--metadata"])["build_id"]
    if record.splitlines()[0] != f"build_id={build_id}":
        raise SystemExit("runner and build record differ; rebuild the runner")
    root = Path(__file__).resolve().parents[1]
    for line in record.split("source_sha256:\n", 1)[1].splitlines():
        expected, name = line.split("  ", 1)
        if sha256_file(root / name) != expected:
            raise SystemExit(f"source changed since configuration: {name}; rebuild the runner")
    return {"runner": str(runner), "runner_sha256": sha256_file(runner),
            "build_record": record, "build_record_sha256": sha256_file(metadata)}


def generate_corpus(path: Path, dimension: int, trials: int, seed: list[int],
                    norm_min: float, norm_max: float) -> dict[str, Any]:
    """Write native float64 [3][trials][dimension]: operands x, y, and a spare."""

    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    values = rng.uniform(-1.0, 1.0, size=(3, trials, dimension))
    norms = np.linalg.norm(values, axis=-1)
    for operand, trial in np.argwhere(norms == 0.0):
        values[operand, trial, 0] = 1.0
        norms[operand, trial] = 1.0
    radii = rng.uniform(norm_min, norm_max, size=(3, trials))
    values /= norms[..., np.newaxis]
    values *= radii[..., np.newaxis]
    np.ascontiguousarray(values, dtype=np.float64).tofile(path)
    return {"file": path.name, "dimension": dimension, "trials": trials, "seed": seed,
            "sha256": sha256_file(path), "bytes": path.stat().st_size,
            "layout": "native float64 [operand=3][trial][coordinate], C order",
            "direction": "independent Uniform[-1,1) coordinates normalized to unit norm",
            "radius": f"Uniform[{norm_min},{norm_max})", "rng": "NumPy Generator(PCG64)"}


def prepare_corpora(arguments: argparse.Namespace, dimensions: list[int]) -> dict[str, Any]:
    directory = arguments.corpora
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    protocol = {"generator": "PCG64 SeedSequence([seed, dimension]); normalize then scale",
                "trials": arguments.trials, "seed": arguments.seed,
                "norm_min": arguments.norm_min, "norm_max": arguments.norm_max,
                "byteorder": sys.byteorder}
    manifest: dict[str, Any] = {"protocol": protocol, "corpora": {}}
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("protocol") == protocol:
            manifest = existing
    for dimension in dimensions:
        path = directory / f"corpus-n{dimension}.f64"
        entry = manifest["corpora"].get(str(dimension))
        expected_bytes = 3 * arguments.trials * dimension * 8
        if entry and path.exists() and path.stat().st_size == expected_bytes \
                and sha256_file(path) == entry["sha256"]:
            continue
        print(f"generating corpus N={dimension}", file=sys.stderr)
        manifest["corpora"][str(dimension)] = generate_corpus(
            path, dimension, arguments.trials, dimension_seed(arguments.seed, dimension),
            arguments.norm_min, arguments.norm_max)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def run_json(command: list[str], timeout: float | None = None) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True,
                                   timeout=timeout)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"runner exceeded {timeout} s: {' '.join(command)}") from error
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"runner produced no JSON: {' '.join(command)}\n"
                           f"{completed.stderr}") from error
    if completed.returncode != 0 and result.get("status") not in ("verification_failure",):
        raise RuntimeError(f"runner failed: {' '.join(command)}\n{completed.stderr}")
    return result


def available_algorithms(dimension: int, arguments: argparse.Namespace,
                         cc_limit: int = CC_MAX_DIMENSION) -> list[str]:
    """Return the algorithms timed at *dimension*; cc_limit comes from the runner."""

    names = []
    for name in ALGORITHMS:
        if name == "cariow_cariowa" and dimension > min(cc_limit,
                                                        arguments.max_cc_dimension):
            continue
        if name == "direct" and dimension > arguments.max_direct_dimension:
            continue
        names.append(name)
    return names


def absent_status(name: str, dimension: int, arguments: argparse.Namespace,
                  cc_limit: int) -> tuple[str, str]:
    """Return the status and reason of an algorithm that was not timed.

    ``unavailable`` means the runner does not support the algorithm at this
    dimension; ``not_timed`` means it is available but above a timing cap.
    """

    if name == "cariow_cariowa" and dimension > cc_limit:
        return "unavailable", f"not supported above dimension {cc_limit}"
    if name == "cariow_cariowa":
        return "not_timed", ("available, but above the Cariow--Cariowa timing cap "
                             "(--max-cc-dimension)")
    return "not_timed", "available, but above the direct timing cap (--max-direct-dimension)"


def absent_label(result: dict[str, Any]) -> str:
    """Render the table cell of an algorithm that was not timed."""

    return "not timed" if result["status"] == "not_timed" else "unavailable"


def summarize(values: list[float]) -> dict[str, Any]:
    count = len(values)
    mean = math.fsum(values) / count
    if count > 1:
        variance = math.fsum((v - mean) ** 2 for v in values) / (count - 1)
        sd = math.sqrt(variance)
        se = sd / math.sqrt(count)
        half = T95[min(count - 1, 30)] * se
        ci = [mean - half, mean + half]
    else:
        sd = se = None
        ci = [None, None]
    ordered = sorted(values)
    return {"count": count, "mean": mean, "sd": sd, "se": se, "ci95": ci,
            "median": ordered[count // 2] if count % 2 else
            0.5 * (ordered[count // 2 - 1] + ordered[count // 2]),
            "min": ordered[0], "max": ordered[-1]}


def paired_ratio(competitor: list[float], baseline: list[float]) -> dict[str, Any]:
    logs = [math.log(c / b) for c, b in zip(competitor, baseline, strict=True)]
    stats = summarize(logs)
    wins = sum(1 for c, b in zip(competitor, baseline, strict=True) if c < b)
    losses = sum(1 for c, b in zip(competitor, baseline, strict=True) if c > b)
    return {"definition": "competitor_ns / baseline_ns per block, geometric mean",
            "paired_blocks": stats["count"],
            "geometric_mean": math.exp(stats["mean"]),
            "ci95": [None if v is None else math.exp(v) for v in stats["ci95"]],
            "competitor_faster_blocks": wins, "competitor_slower_blocks": losses}


def format_ci(summary: dict[str, Any]) -> str:
    lo, hi = summary["ci95"]
    if lo is None:
        return f"{summary['mean']:.2f}"
    return f"{summary['mean']:.2f} [{lo:.2f}, {hi:.2f}]"


def fastest_label(results: dict[str, Any]) -> str:
    """Name the lowest verified mean, or a tie when the runner-up is not separable.

    Two algorithms tie when the 95% interval of their paired per-block ratio
    contains one, so the label never claims a winner the data cannot support.
    """

    verified = sorted((r for r in results.values() if r["status"] == "ok"),
                      key=lambda r: r["summary"]["mean"])
    if not verified:
        return "none verified"
    if len(verified) == 1:
        return str(verified[0]["algorithm"])
    ratio = paired_ratio(verified[0]["observations_ns_per_product"],
                         verified[1]["observations_ns_per_product"])
    lo, hi = ratio["ci95"]
    if lo is not None and lo <= 1.0 <= hi:
        return f"tie ({verified[0]['algorithm']}, {verified[1]['algorithm']})"
    return str(verified[0]["algorithm"])


def mode_table(document: dict[str, Any], mode: str) -> list[str]:
    """Render the timing table of one mode with its fastest and default columns."""

    default_key = "default_batch_algorithm" if mode == "batch" else "default_algorithm"
    lines = ["| N | direct | cariow_cariowa | quasilinear | fastest mean | default |",
             "| ---: | --- | --- | --- | --- | --- |"]
    for entry in document["results"]:
        results = entry["modes"][mode]
        row = [str(entry["dimension"])]
        for name in ALGORITHMS:
            result = results[name]
            row.append(format_ci(result["summary"]) if result["status"] == "ok"
                       else absent_label(result))
        row.append(fastest_label(results))
        row.append(entry[default_key])
        lines.append("| " + " | ".join(row) + " |")
    return lines


def ratio_table(document: dict[str, Any], mode: str) -> list[str]:
    """Render the paired competitor/quasilinear ratios of one mode."""

    lines = ["| N | direct / quasilinear | cariow_cariowa / quasilinear |", "| ---: | --- | --- |"]
    for entry in document["results"]:
        cells = []
        for name in ("direct", "cariow_cariowa"):
            ratio = entry["ratios_vs_quasilinear"][mode].get(name)
            if ratio is None:
                cells.append(absent_label(entry["modes"][mode][name]))
            else:
                lo, hi = ratio["ci95"]
                interval = "" if lo is None else f" [{lo:.3f}, {hi:.3f}]"
                cells.append(f"{ratio['geometric_mean']:.3f}{interval} "
                             f"({ratio['competitor_faster_blocks']}/{ratio['paired_blocks']} blocks faster)")
        lines.append(f"| {entry['dimension']} | " + " | ".join(cells) + " |")
    return lines


def timing_cap_notes(document: dict[str, Any]) -> list[str]:
    """Explain algorithms that the library supports but the run did not time."""

    protocol = document["protocol"]
    dimensions = protocol["dimensions"]
    cc_limit = int(document["runner_metadata"].get("cariow_cariowa_max_dimension",
                                                    CC_MAX_DIMENSION))
    notes = []
    cc_cap = min(cc_limit, protocol["max_cc_dimension"])
    if any(d > cc_cap for d in dimensions) and cc_limit > cc_cap:
        notes.append(f"Cariow–Cariowa was timed only through dimension {cc_cap}, although "
                     f"the library supports it through {cc_limit}. Above 64 its kernel is "
                     "table-driven and this implementation does not vectorize its sparse "
                     "sign corrections within a product, so timings there reflect its "
                     "access pattern as well as its arithmetic count; the "
                     "algorithm is not recommended above 64. See docs/benchmark.md.")
    if any(d > protocol["max_direct_dimension"] for d in dimensions):
        notes.append(f"Direct was timed only through dimension "
                     f"{protocol['max_direct_dimension']} (--max-direct-dimension).")
    return [" ".join(notes), ""] if notes else []


def write_markdown(path: Path, document: dict[str, Any]) -> None:
    modes = document["protocol"]["modes"]
    lines = ["# fastCD multiplication benchmark", "",
             f"Host: {document['host']['processor']} ({document['host']['machine']}); "
             f"{document['runner_metadata']['compiler']['name']} "
             f"{document['runner_metadata']['compiler']['version']}; "
             f"{document['protocol']['blocks']} blocks of {document['protocol']['trials']} pairs "
             f"per dimension; each cell is the mean over blocks of ns per product with a "
             f"two-sided 95% Student-t interval.", "",
             "Batch mode times one validated batch call per corpus pass (two products per "
             "SIMD instruction where the build supports it); single mode times one validated "
             "single-product call per pair.  Python object overhead is excluded. Verification "
             "uses an independent float64 oracle, not an exact product. Fastest names "
             "the lowest verified mean, or a tie when the paired-ratio interval of "
             "the two lowest means contains one; default is the library's automatic "
             "selection for that mode.", "",
             f"Seed: {document['protocol']['seed']}; CPU: {document['protocol']['cpu']}; "
             f"status: {document['status']}. "
             f"{git_summary(document['provenance'].get('git', {}))}. Build options, "
             "source and executable hashes, corpus hashes, and raw observations are in "
             "the accompanying JSON.", ""]
    lines += timing_cap_notes(document)
    for mode in modes:
        lines += [f"## {mode.capitalize()} products (ns per product)", ""]
        lines += mode_table(document, mode) + [""]
    for mode in modes:
        lines += [f"## Paired ratios against quasilinear, {mode} mode", "",
                  "Geometric mean of per-block ratios (competitor / quasilinear) with 95% interval; "
                  "values above 1 mean quasilinear was faster.", ""]
        lines += ratio_table(document, mode) + [""]
    if len(modes) == 2:
        lines += ["## Single-product time relative to batched time", "",
                  "Geometric mean of per-block ratios (single / batch) for each algorithm; "
                  "values above 1 mean batching was faster.", "",
                  "| N | direct | cariow_cariowa | quasilinear |", "| ---: | --- | --- | --- |"]
        for entry in document["results"]:
            cells = []
            for name in ALGORITHMS:
                ratio = entry["single_over_batch"].get(name)
                if ratio is None:
                    cells.append(absent_label(entry["modes"][modes[0]][name]))
                else:
                    lo, hi = ratio["ci95"]
                    interval = "" if lo is None else f" [{lo:.3f}, {hi:.3f}]"
                    cells.append(f"{ratio['geometric_mean']:.3f}{interval}")
            lines.append(f"| {entry['dimension']} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Verification against the defining recursion", "",
              "Maximum norm-scaled coordinate error |actual - oracle| / (||x|| ||y||) over the "
              "verified pairs, and mismatches against the tolerance 128 eps N, for each mode.", "",
              "| N | direct | cariow_cariowa | quasilinear |", "| ---: | --- | --- | --- |"]
    for entry in document["results"]:
        cells = []
        for name in ALGORITHMS:
            parts = []
            for mode in modes:
                result = entry["modes"][mode][name]
                if result["status"] in ("unavailable", "not_timed"):
                    parts.append(f"{mode}: {absent_label(result)}")
                else:
                    v = result["verification"]
                    parts.append(f"{mode}: {v['max_norm_scaled_error']:.2e} "
                                 f"({v['mismatched_trials']}/{v['verified_trials']})")
            cells.append("; ".join(parts))
        lines.append(f"| {entry['dimension']} | " + " | ".join(cells) + " |")
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> int:
    arguments = parse_arguments()
    dimensions = sorted({int(v) for v in arguments.dimensions.split(",")})
    for dimension in dimensions:
        if dimension <= 0 or dimension & (dimension - 1):
            raise SystemExit(f"dimension {dimension} is not a power of two")
    if arguments.cpu is None and hasattr(os, "sched_getaffinity"):
        arguments.cpu = min(os.sched_getaffinity(0))
    if arguments.cpu is not None:
        if not hasattr(os, "sched_setaffinity"):
            raise SystemExit("CPU pinning requires Linux")
        os.sched_setaffinity(0, {arguments.cpu})
    if arguments.output.exists() or arguments.output.with_suffix(".md").exists():
        raise SystemExit(f"refusing to overwrite {arguments.output}")
    identity = provenance(arguments)
    git = git_state(Path(__file__).resolve().parents[1])
    if git["dirty"]:
        print("note: tracked files have uncommitted changes; the recorded commit "
              "does not describe the timed sources exactly", file=sys.stderr)
    runner_metadata = run_json([str(arguments.runner), "--metadata"],
                               arguments.subprocess_timeout)
    cc_limit = int(runner_metadata.get("cariow_cariowa_max_dimension",
                                       CC_MAX_DIMENSION))
    partial_path = arguments.output.with_name(arguments.output.stem + ".partial.json")
    manifest = prepare_corpora(arguments, dimensions)
    verify = arguments.trials if arguments.verify_trials is None else arguments.verify_trials
    rng = random.Random(arguments.seed)
    results_list: list[dict[str, Any]] = []
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for dimension in dimensions:
        names = available_algorithms(dimension, arguments, cc_limit)
        tasks = [(name, mode) for mode in arguments.modes for name in names]
        base_order = list(tasks)
        rng.shuffle(base_order)
        observations: dict[tuple[str, str], list[float]] = {task: [] for task in tasks}
        raw: dict[tuple[str, str], list[dict[str, Any]]] = {task: [] for task in tasks}
        verification: dict[tuple[str, str], Any] = {}
        workspace: dict[str, int] = {}
        defaults: dict[str, str] = {}
        for block in range(arguments.blocks):
            shift = block % len(tasks)
            order = base_order[shift:] + base_order[:shift]
            for name, mode in order:
                command = [str(arguments.runner), "--corpus",
                           str(arguments.corpora / f"corpus-n{dimension}.f64"),
                           "--dimension", str(dimension), "--trials", str(arguments.trials),
                           "--algorithm", name, "--mode", mode, "--repeats", "1",
                           "--warmup", str(arguments.warmup),
                           "--min-time-ms", repr(arguments.min_time_ms),
                           "--verify-trials", str(verify if block == 0 else 0)]
                result = run_json(command, arguments.subprocess_timeout)
                observations[(name, mode)].append(result["observations"][0]["ns_per_operation"])
                raw[(name, mode)].append({"block": block, "order": order.index((name, mode)),
                                          "passes": result["corpus_passes_per_observation"],
                                          **result["observations"][0]})
                workspace[name] = result["workspace_doubles"]
                defaults["default_algorithm"] = result["default_algorithm"]
                defaults["default_batch_algorithm"] = result["default_batch_algorithm"]
                if block == 0:
                    verification[(name, mode)] = result["verification"]
            print(f"N={dimension} block {block + 1}/{arguments.blocks} done", file=sys.stderr)
        entry: dict[str, Any] = {"dimension": dimension, **defaults, "modes": {},
                                 "ratios_vs_quasilinear": {}, "single_over_batch": {}}
        for mode in arguments.modes:
            results: dict[str, Any] = {}
            for name in ALGORITHMS:
                if name not in names:
                    status, reason = absent_status(name, dimension, arguments, cc_limit)
                    results[name] = {"algorithm": name, "status": status,
                                     "reason": reason}
                    continue
                task = (name, mode)
                checksums_match = len({item["checksum"] for item in raw[task]}) == 1
                valid = verification[task]["mismatched_trials"] == 0 and checksums_match
                results[name] = {
                    "algorithm": name, "mode": mode,
                    "status": "ok" if valid else "verification_failure",
                    "workspace_doubles": workspace[name],
                    "observations_ns_per_product": observations[task],
                    "observations": raw[task], "checksums_match": checksums_match,
                    "summary": summarize(observations[task]), "verification": verification[task]}
            entry["modes"][mode] = results
            entry["ratios_vs_quasilinear"][mode] = {}
            for name in names:
                if name != BASELINE and results[name]["status"] == "ok" \
                        and results[BASELINE]["status"] == "ok":
                    entry["ratios_vs_quasilinear"][mode][name] = paired_ratio(
                        observations[(name, mode)], observations[(BASELINE, mode)])
        if len(arguments.modes) == 2:
            for name in names:
                if all(entry["modes"][mode][name]["status"] == "ok" for mode in MODES):
                    entry["single_over_batch"][name] = paired_ratio(
                        observations[(name, "single")], observations[(name, "batch")])
        results_list.append(entry)
        # Keep finished dimensions on disk so an interrupted run is not a total loss.
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path.write_text(json.dumps(
            {"schema": "fastcd-benchmark-results-2-partial", "results": results_list},
            indent=2, allow_nan=False))
    document = {
        "schema": "fastcd-benchmark-results-2",
        "status": "verification_failure" if any(
            r["status"] == "verification_failure" for entry in results_list
            for mode_results in entry["modes"].values() for r in mode_results.values()) else "ok",
        "started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol": {"dimensions": dimensions, "trials": arguments.trials, "blocks": arguments.blocks,
                     "warmup": arguments.warmup, "min_time_ms": arguments.min_time_ms,
                     "seed": arguments.seed, "norm_min": arguments.norm_min,
                     "norm_max": arguments.norm_max, "verify_trials": verify,
                     "max_direct_dimension": arguments.max_direct_dimension,
                     "max_cc_dimension": arguments.max_cc_dimension,
                     "cariow_cariowa_supported_dimension": cc_limit,
                     "modes": list(arguments.modes),
                     "cpu": arguments.cpu, "method_order": "shuffled (algorithm, mode) order rotated per block"},
        "host": {"machine": platform.machine(), "processor": cpu_name(),
                 "system": platform.platform(), "python": sys.version.split()[0],
                 "numpy": np.__version__, "cpu_state": cpu_state(arguments.cpu)},
        "runner_metadata": runner_metadata, "provenance": {**identity, "git": git},
        "corpora": manifest, "results": results_list}
    if provenance(arguments) != identity:
        raise SystemExit("executable or build record changed during the run; discard timings")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(document, indent=2, allow_nan=False))
    write_markdown(arguments.output.with_suffix(".md"), document)
    partial_path.unlink(missing_ok=True)
    print(f"wrote {arguments.output} and {arguments.output.with_suffix('.md')}")
    return 0 if document["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
