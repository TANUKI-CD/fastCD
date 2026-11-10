"""Time Cariow–Cariowa products of the baseline and the measured build in pairs.

Quasilinear and direct products are timed in the same blocks. Direct
arithmetic is identical in both builds, so its ratio is the control for noise
between builds. Quasilinear arithmetic differs between the baseline and the
measured build, so its ratio is reported in the CSV but is not a control.
"""

from __future__ import annotations

import argparse
import csv
import io
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from stability_timing import (
    BOOTSTRAP,
    CORPUS_DISTRIBUTION,
    build_record,
    corpus,
    paired,
    pin,
    real_task,
    write_json,
)

ALGORITHMS = ("cariow_cariowa", "quasilinear", "direct")
VERSIONS = ("baseline", "measured")


def summarize(values: dict[tuple[str, str, str], list[float]], mode: str, algorithm: str) -> dict[str, Any]:
    baseline, measured = (values[mode, algorithm, version] for version in VERSIONS)
    comparison = paired(baseline, measured)
    return {"baseline_ns": statistics.median(baseline), "measured_ns": statistics.median(measured),
            "measured_over_baseline": comparison["ratio"], "measured_over_baseline_95ci": comparison["ratio_95ci"]}


def render(result: dict[str, Any]) -> str:
    """Markdown table of Cariow–Cariowa ratios with the direct control."""
    if not result["completed"]:
        raise ValueError("Cariow–Cariowa timings did not finish; refusing a partial table")
    rows = {(r["backend"], r["mode"], r["dimension"], r["algorithm"]): r for r in result["summary"]}
    lines = [f"Cariow–Cariowa products, {result['blocks']} paired blocks pinned to logical CPU "
             f"{', '.join(map(str, result['affinity']))}. The baseline and measured columns are the "
             "builds of the baseline and measured commits; times are medians in ns per product, and "
             "brackets are paired bootstrap 95% intervals. The direct control is the measured / "
             "baseline ratio of direct timings, whose arithmetic is identical in both builds.", "",
             "| kernels | mode | N | baseline ns | measured ns | measured / baseline [95% CI] | direct control |",
             "|:---|:---|---:|---:|---:|:---|---:|"]
    for (backend, mode, n, algorithm), row in rows.items():
        if algorithm != "cariow_cariowa":
            continue
        low, high = row["measured_over_baseline_95ci"]
        control = rows[backend, mode, n, "direct"]["measured_over_baseline"]
        lines.append(f"| {backend} | {mode} | {n} | {row['baseline_ns']:.5g} | {row['measured_ns']:.5g} | "
                     f"{row['measured_over_baseline']:.4g} [{low:.4g}, {high:.4g}] | {control:.4g} |")
    return "\n".join(lines) + "\n"


def summary_csv(result: dict[str, Any]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(result["summary"][0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(result["summary"])
    return output.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for version in VERSIONS:
        parser.add_argument("--" + version, type=Path, required=True, help=f"{version} build directory")
        parser.add_argument("--" + version + "-scalar", type=Path, required=True,
                            help=f"{version} build directory without explicit SIMD kernels")
    parser.add_argument("--output", type=Path, required=True, help="JSON results file")
    parser.add_argument("--corpora", type=Path, required=True, help="directory for the operand corpora")
    parser.add_argument("--tables", type=Path, help="Markdown table to write when the timings finish")
    parser.add_argument("--csv", type=Path, help="CSV summary to write when the timings finish")
    parser.add_argument("--dimensions", type=int, nargs="+", default=[16, 64, 128, 256, 512, 1024])
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--minimum-ms", type=float, default=20)
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--cpu", type=int)
    args = parser.parse_args()
    if args.blocks < 12 or args.blocks % 12:
        parser.error("use a positive multiple of twelve blocks, so every task position sees both build orders")
    if any(n < 2 or n > 1024 or n & (n - 1) for n in args.dimensions):
        parser.error("dimensions must be powers of two from 2 through 1024")
    affinity = pin(args.cpu)
    builds = {backend: {version: getattr(args, version + ("_scalar" if backend == "scalar" else "")).resolve()
                        for version in VERSIONS} for backend in ("simd", "scalar")}
    result: dict[str, Any] = {
        "schema": "fastcd-cc-performance-2", "completed": False,
        "command": [sys.executable, *sys.argv], "python": sys.version, "affinity": affinity,
        "blocks": args.blocks, "minimum_ms": args.minimum_ms, "trials": args.trials, "warmup_passes": 2,
        "version_order": "baseline/measured/measured/baseline over four blocks; the six algorithm/mode tasks rotate "
                         "every block, so each task position sees both build orders across twelve blocks",
        "bootstrap": BOOTSTRAP, "corpus_distribution": CORPUS_DISTRIBUTION,
        "builds": {backend: {version: build_record(directory) for version, directory in versions.items()}
                   for backend, versions in builds.items()},
        "corpora": [], "raw": [], "summary": [],
    }
    write_json(args.output, result)
    try:
        tasks = [(mode, algorithm) for mode in ("single", "batch") for algorithm in ALGORITHMS]
        for n in args.dimensions:
            path = args.corpora / f"real-{n}.bin"
            seed = 2026110500 + n
            result["corpora"].append({"dimension": n, "seed": seed, "trials": args.trials, "file": path.name,
                                      "sha256": corpus(path, n, args.trials, "real", seed)})
            for backend, versions in builds.items():
                values: dict[tuple[str, str, str], list[float]] = {}
                for block in range(args.blocks):
                    order = VERSIONS if block % 4 in {0, 3} else VERSIONS[::-1]
                    shift = block % len(tasks)
                    for version in order:
                        for mode, algorithm in tasks[shift:] + tasks[:shift]:
                            timing, raw = real_task(versions[version], path, n, args.trials, mode, algorithm,
                                                    args.minimum_ms)
                            result["raw"].append({"dimension": n, "backend": backend, "block": block,
                                                  "version": version, "native": raw})
                            values.setdefault((mode, algorithm, version), []).append(timing)
                for mode in ("single", "batch"):
                    for algorithm in ALGORITHMS:
                        result["summary"].append({"backend": backend, "dimension": n, "mode": mode,
                                                  "algorithm": algorithm, **summarize(values, mode, algorithm)})
                write_json(args.output, result)
                print(f"timed N={n} {backend}", flush=True)
        result["completed"] = True
    except Exception as error:
        result["failure"] = {"type": type(error).__name__, "message": str(error)}
        if isinstance(error, subprocess.CalledProcessError):
            result["failure"].update(stdout=error.stdout, stderr=error.stderr)
        raise
    finally:
        write_json(args.output, result)
    if args.tables:
        args.tables.write_text(render(result))
    if args.csv:
        args.csv.write_text(summary_csv(result))


if __name__ == "__main__":
    main()
