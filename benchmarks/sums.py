#!/usr/bin/env python3
"""Benchmark fused sums of products against a batch of products and dgemm.

Every task computes the sum of the products of each sample of ``count``
real-coefficient elements: a batch of products followed by a sum, the fused
sum of records, the fused sum of coordinate planes with the direct and with
the quasilinear algorithm, and, when ``--blas`` names a BLAS library, the
matrix of real cross moments by one ``dgemm`` followed by its contraction
with the basis signs.

The default protocol uses 8,192 deterministic elements in samples of 1,024,
60 balanced blocks, two warmup passes and at least 20 ms per observation.
Each block runs at its own stack offset within one runner process. The first
sample of every cell is verified against a long double evaluation, and all
samples for agreement between the tasks, outside timing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from . import benchmark as real_benchmark
except ImportError:  # Support direct execution from this directory.
    import benchmark as real_benchmark

TASKS = ("products", "records", "planes_direct", "planes_quasilinear", "dgemm")
# Each ratio divides the first task's time by the second's
RATIOS = (
    ("products", "records"),
    ("products", "planes"),
    ("dgemm", "records"),
    ("dgemm", "planes"),
    ("planes_direct", "planes_quasilinear"),
)


def corpus(path: Path, n: int, trials: int, seed: int) -> dict[str, Any]:
    """Write the x records then the y records, with a bound manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    entropy = [seed, n, trials]
    config = {
        "dimension": n,
        "trials": trials,
        "seed": entropy,
        "dtype": "<f8",
        "generator": "PCG64 standard normal",
    }
    manifest_path = path.with_suffix(".json")
    if path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest["config"] != config
            or manifest["sha256"] != real_benchmark.sha256_file(path)
            or path.stat().st_size != 2 * trials * n * 8
        ):
            raise ValueError(f"existing corpus or manifest differs: {path}")
        return manifest
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))
    with path.open("wb") as output:
        for _ in range(2):
            rng.standard_normal((trials, n)).astype("<f8").tofile(output)
    manifest = {
        "config": config,
        "sha256": real_benchmark.sha256_file(path),
        "bytes": path.stat().st_size,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def summary(run: dict[str, Any]) -> dict[str, Any]:
    """Summarize the tasks per element and form the paired ratios.

    ``planes`` stands for the plane task of the algorithm that automatic
    selection uses at this dimension.
    """
    names = [t for t in TASKS if any(o["task"] == t for o in run["observations"])]
    values = {
        t: [
            o["elapsed_ns"] / o["operations"]
            for o in sorted(run["observations"], key=lambda o: o["block"])
            if o["task"] == t
        ]
        for t in names
    }
    selected = f"planes_{run['default_sum_algorithm']}"
    if selected in values:
        values["planes"] = values[selected]
    return {
        "ns_per_element": {t: real_benchmark.summarize(v) for t, v in values.items()},
        "ratios": {
            f"{a}/{b}": real_benchmark.paired_ratio(values[a], values[b])
            for a, b in RATIOS
            if a in values and b in values
        },
    }


def _time(stats: dict[str, Any], task: str) -> str:
    return f"{stats[task]['mean']:.2f}" if task in stats else ""


def _ratio(ratios: dict[str, Any], name: str) -> str:
    if name not in ratios:
        return ""
    lo, hi = ratios[name]["ci95"]
    return f"{ratios[name]['geometric_mean']:.2f} ({lo:.2f}–{hi:.2f})"


def report(path: Path, document: dict[str, Any]) -> None:
    configuration = document["configuration"]
    blas = document["blas"]
    exact = sum(run["exact_samples"] for run in document["runs"])
    samples = sum(run["samples"] for run in document["runs"])
    lines = [
        "# Fused-sum benchmark",
        "",
        f"Measured {configuration['trials']:,} elements per cell in samples of "
        f"the stated count, {configuration['blocks']} blocks at separate stack "
        f"offsets, on {document['cpu']}.",
        "",
        "Times are means in ns per element. `products` is one batch of products "
        "followed by a sum; `records` and `planes` are fused sums with the "
        "automatic algorithm; `dgemm` is one matrix product and the contraction "
        "with the basis signs"
        + (f", with {blas['name']}." if blas else "; no BLAS library was given."),
        "",
        "| N | Count | Products | Records | Planes, direct | Planes, quasilinear "
        "| dgemm |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in document["runs"]:
        stats = run["summary"]["ns_per_element"]
        lines.append(
            f"| {run['dimension']} | {run['count']} | "
            + " | ".join(_time(stats, t) for t in TASKS)
            + " |"
        )
    lines += [
        "",
        "Ratios are paired geometric means with 95% Student-t intervals. Each "
        "divides the time of the first task by that of the second, so values "
        "above 1 favor the second. In the last column, values above 1 favor "
        "quasilinear.",
        "",
        "| N | Count | Products/records | Products/planes | dgemm/records "
        "| dgemm/planes | Direct/quasilinear (planes) |",
        "|---:|---:|---|---|---|---|---|",
    ]
    for run in document["runs"]:
        ratios = run["summary"]["ratios"]
        lines.append(
            f"| {run['dimension']} | {run['count']} | "
            + " | ".join(_ratio(ratios, f"{a}/{b}") for a, b in RATIOS)
            + " |"
        )
    lines += [
        "",
        f"A long double contraction of the cross moments verified {exact} of "
        f"{samples} {'sample' if samples == 1 else 'samples'}. Every sample "
        "was checked for agreement between the tasks, outside timing. The two "
        "fused layouts returned identical bits wherever they ran the same algorithm.",
        "",
        "Raw observations, corpus hashes, the runner, build and source hashes "
        "and the BLAS library are recorded in the adjacent JSON.",
        "",
    ]
    path.write_text("\n".join(lines))


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-metadata", type=Path)
    parser.add_argument("--corpora", type=Path, default=Path("benchmark-corpora/sums"))
    parser.add_argument("--dimensions", default="2,4,8,16,32,64,128,256,512,1024")
    parser.add_argument(
        "--counts", default="1024", help="comma-separated elements per sample"
    )
    parser.add_argument(
        "--trials", type=int, default=8192, help="elements per cell, in whole samples"
    )
    parser.add_argument("--blocks", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--min-time-ms", type=float, default=20)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--verify-samples", type=int, default=1)
    parser.add_argument(
        "--max-direct-dimension",
        type=int,
        default=64,
        help="largest dimension at which the quadratic direct sum is timed",
    )
    parser.add_argument(
        "--blas", type=Path, help="shared BLAS library providing cblas_dgemm"
    )
    parser.add_argument("--blas-symbol", default="cblas_dgemm")
    parser.add_argument(
        "--blas-ilp64", action="store_true", help="the library takes 64-bit integers"
    )
    parser.add_argument(
        "--blas-name", help="name of the BLAS library for the report (default: file)"
    )
    parser.add_argument("--subprocess-timeout", type=float, default=3600)
    args = parser.parse_args(argv)
    if args.output.suffix.lower() == ".md":
        parser.error("output and Markdown report paths must differ")
    if (
        args.trials < 1
        or args.blocks < 2
        or args.warmup < 0
        or args.verify_samples < 0
        or args.max_direct_dimension < 1
        or not math.isfinite(args.min_time_ms)
        or args.min_time_ms < 0
        or args.seed < 0
        or not math.isfinite(args.subprocess_timeout)
        or args.subprocess_timeout <= 0
    ):
        parser.error("positive trials and at least two blocks are required")
    try:
        args.dimension_list = sorted({int(v) for v in args.dimensions.split(",")})
        args.count_list = sorted({int(v) for v in args.counts.split(",")})
    except ValueError:
        parser.error("dimensions and counts must be integers")
    if any(n < 1 or n & (n - 1) for n in args.dimension_list):
        parser.error("dimensions must be positive powers of two")
    if any(c < 1 or args.trials % c for c in args.count_list):
        parser.error("every count must be positive and divide trials")
    if args.blas is not None and not args.blas.is_file():
        parser.error(f"BLAS library is missing: {args.blas}")
    if args.max_direct_dimension < 8:
        parser.error(
            "max-direct-dimension must be at least 8, through which the direct "
            "sum is the automatic algorithm"
        )
    for n in args.dimension_list:
        tasks = 3 + (n <= args.max_direct_dimension) + (args.blas is not None)
        if args.blocks % tasks:
            parser.error(
                f"block count must be a multiple of the {tasks} tasks at N={n}; "
                "60 balances every combination"
            )
    for path in (
        args.output,
        args.output.with_suffix(".md"),
        args.output.with_name(args.output.stem + ".partial.json"),
    ):
        if path.exists():
            parser.error(f"refusing to overwrite {path}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    if args.cpu is None and hasattr(os, "sched_getaffinity"):
        args.cpu = min(os.sched_getaffinity(0))
    if args.cpu is not None:
        if not hasattr(os, "sched_setaffinity"):
            raise ValueError("CPU affinity is unavailable on this platform")
        os.sched_setaffinity(0, {args.cpu})
    # One thread for every BLAS implementation the runner may load
    for name in ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    blas = None
    if args.blas is not None:
        args.blas = args.blas.resolve()
        # A dispatching library loads its backends from its own directory
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
            filter(None, [str(args.blas.parent), os.environ.get("LD_LIBRARY_PATH")])
        )
        blas = {
            "name": args.blas_name or args.blas.name,
            "path": str(args.blas),
            "sha256": real_benchmark.sha256_file(args.blas),
            "symbol": args.blas_symbol,
            "ilp64": args.blas_ilp64,
            "threads": 1,
        }
    args.runner = args.runner.resolve()
    metadata = real_benchmark.run_json([str(args.runner), "--metadata"])
    if metadata.get("schema") != "fastcd-sums-1":
        raise ValueError("runner must name fastcd_sums_benchmark")
    root = Path(__file__).resolve().parents[1]
    document: dict[str, Any] = {
        "schema": "fastcd-sums-report-1",
        "cpu": real_benchmark.cpu_name(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "configuration": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "blas": blas,
        "git": real_benchmark.git_state(root),
        "metadata": metadata,
        "provenance": real_benchmark.provenance(args),
        "cpu_state_before": real_benchmark.cpu_state(args.cpu),
        "runs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(args.output.stem + ".partial.json")
    for n in args.dimension_list:
        path = args.corpora / f"n{n}-t{args.trials}-s{args.seed}.bin"
        manifest = corpus(path, n, args.trials, args.seed)
        for count in args.count_list:
            print(f"N={n}, count={count}: verifying and timing", flush=True)
            start = time.monotonic()
            command = [
                str(args.runner),
                "--corpus",
                str(path.resolve()),
                "--dimension",
                str(n),
                "--count",
                str(count),
                "--trials",
                str(args.trials),
                "--blocks",
                str(args.blocks),
                "--warmup",
                str(args.warmup),
                "--min-time-ms",
                f"{args.min_time_ms:g}",
                "--verify-samples",
                str(args.verify_samples),
                "--max-direct-dimension",
                str(args.max_direct_dimension),
            ]
            if blas is not None:
                command += [
                    "--blas",
                    blas["path"],
                    "--blas-symbol",
                    blas["symbol"],
                    "--blas-ilp64",
                    str(int(blas["ilp64"])),
                ]
            run = real_benchmark.run_json(command, args.subprocess_timeout)
            if run["verified_samples"] != args.trials // count:
                raise RuntimeError("runner did not verify the complete corpus")
            run.update(
                {
                    "summary": summary(run),
                    "corpus": manifest,
                    "wall_seconds": time.monotonic() - start,
                }
            )
            document["runs"].append(run)
            partial.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    document["cpu_state_after"] = real_benchmark.cpu_state(args.cpu)
    if real_benchmark.provenance(args) != document["provenance"]:
        raise ValueError("build changed during the run; discard timings")
    args.output.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    report(args.output.with_suffix(".md"), document)
    partial.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
