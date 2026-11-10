"""Time sequential against balanced accumulation in paired blocks.

Each build is supplied twice: with the explicit SIMD kernels and as a scalar
build (FASTCD_DISABLE_SIMD=1). In every paired block both builds time the same
corpus; the order of the two builds alternates between blocks, and the order
of the real algorithm/mode tasks rotates every two blocks.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from stability_timing import (
    BOOTSTRAP,
    CORPUS_DISTRIBUTION,
    build_record,
    complex_task,
    corpus,
    paired,
    pin,
    real_task,
    write_json,
)

VERSIONS = ("sequential", "balanced")
REAL_TASKS = [(mode, algorithm) for mode in ("single", "batch") for algorithm in ("direct", "quasilinear")]


def summarize(values: dict[tuple[str, str, str], list[float]], mode: str) -> dict[str, Any]:
    """Medians per build and the paired ratios balanced / sequential."""
    quasilinear = paired(values[mode, "quasilinear", "sequential"], values[mode, "quasilinear", "balanced"])
    control = paired(values[mode, "direct", "sequential"], values[mode, "direct", "balanced"])
    summary: dict[str, Any] = {
        f"{version}_{algorithm}_ns": statistics.median(values[mode, algorithm, version])
        for version in VERSIONS for algorithm in ("quasilinear", "direct")
    }
    summary.update({
        "balanced_over_sequential": quasilinear["ratio"],
        "balanced_over_sequential_95ci": quasilinear["ratio_95ci"],
        "direct_control": control["ratio"],
        "direct_control_95ci": control["ratio_95ci"],
        "direct_over_balanced": statistics.median([
            direct / ql for direct, ql in zip(
                values[mode, "direct", "balanced"], values[mode, "quasilinear", "balanced"], strict=True,
            )
        ]),
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for version in VERSIONS:
        parser.add_argument("--" + version, type=Path, required=True, help=f"{version} build directory")
        parser.add_argument("--" + version + "-scalar", type=Path, required=True,
                            help=f"{version} build directory without explicit SIMD kernels")
    parser.add_argument("--output", type=Path, required=True, help="JSON results file")
    parser.add_argument("--corpora", type=Path, required=True, help="directory for the operand corpora")
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--minimum-ms", type=float, default=20)
    parser.add_argument("--real-dimensions", type=int, nargs="+", default=[16, 64, 256, 1024])
    parser.add_argument("--complex-dimensions", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--cpu", type=int)
    args = parser.parse_args()
    if args.blocks < 2 or args.blocks % 2:
        parser.error("use a positive even number of paired blocks, so both build orders occur equally often")
    if args.minimum_ms <= 0:
        parser.error("the minimum observation time must be positive")
    affinity = pin(args.cpu)
    builds = {
        backend: {version: getattr(args, version + ("_scalar" if backend == "scalar" else "")).resolve()
                  for version in VERSIONS}
        for backend in ("simd", "scalar")
    }
    result: dict[str, Any] = {
        "schema": "fastcd-stability-speed-2", "completed": False,
        "command": [sys.executable, *sys.argv], "python": sys.version, "platform": platform.platform(),
        "cpu": subprocess.check_output(["lscpu", "--json"], text=True) if sys.platform == "linux" else platform.processor(),
        "affinity": affinity, "blocks": args.blocks, "minimum_ms": args.minimum_ms, "warmup_passes": 2,
        "version_order": "sequential then balanced in even blocks, balanced then sequential in odd blocks",
        "bootstrap": BOOTSTRAP, "corpus_distribution": CORPUS_DISTRIBUTION,
        "dimensions": {"real": args.real_dimensions, "complex": args.complex_dimensions},
        "builds": {backend: {version: build_record(directory) for version, directory in versions.items()}
                   for backend, versions in builds.items()},
        "corpora": [], "raw": [], "summary": [],
    }
    write_json(args.output, result)
    try:
        for domain, dimensions in (("real", args.real_dimensions), ("complex", args.complex_dimensions)):
            for n in dimensions:
                trials = min(32, max(3, 2048 // n))
                seed = 2026110500 + n + (100000 if domain == "complex" else 0)
                path = args.corpora / f"{domain}-{n}.bin"
                result["corpora"].append({"domain": domain, "dimension": n, "trials": trials, "seed": seed,
                                          "file": path.name, "sha256": corpus(path, n, trials, domain, seed)})
                for backend, versions in builds.items():
                    values: dict[tuple[str, str, str], list[float]] = {}
                    for block in range(args.blocks):
                        order = VERSIONS if block % 2 == 0 else VERSIONS[::-1]
                        for version in order:
                            if domain == "complex":
                                times, raw = complex_task(versions[version], path, n, trials, args.minimum_ms)
                                result["raw"].append({"domain": domain, "dimension": n, "backend": backend,
                                                      "block": block, "version": version, "native": raw})
                                for (mode, algorithm), timing in times.items():
                                    values.setdefault((mode, algorithm, version), []).append(timing)
                                continue
                            # Both build orders occur at every rotation of the task order
                            shift = (block // 2) % len(REAL_TASKS)
                            for mode, algorithm in REAL_TASKS[shift:] + REAL_TASKS[:shift]:
                                timing, raw = real_task(versions[version], path, n, trials, mode, algorithm,
                                                        args.minimum_ms)
                                result["raw"].append({"domain": domain, "dimension": n, "backend": backend,
                                                      "block": block, "version": version, "native": raw})
                                values.setdefault((mode, algorithm, version), []).append(timing)
                    for mode in ("single", "batch"):
                        result["summary"].append({"domain": domain, "dimension": n, "backend": backend,
                                                  "mode": mode, **summarize(values, mode)})
                    write_json(args.output, result)
                    print(f"timed {domain} N={n} {backend}", flush=True)
        result["completed"] = True
    except Exception as error:
        result["failure"] = {"type": type(error).__name__, "message": str(error)}
        if isinstance(error, subprocess.CalledProcessError):
            result["failure"].update(stdout=error.stdout, stderr=error.stderr)
        raise
    finally:
        write_json(args.output, result)


if __name__ == "__main__":
    main()
