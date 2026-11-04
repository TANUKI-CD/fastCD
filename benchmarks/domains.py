#!/usr/bin/env python3
"""Benchmark direct/quasilinear CD arithmetic over odd finite fields and C.

The default protocol uses 10,000 deterministic pairs, 24 balanced blocks,
two warmup passes and at least 20 ms per observation. All pairs are verified
outside timing. Whole-corpus replay, allocation exclusions and statistical
summaries match the existing real benchmark. No Cariow--Cariowa is measured.
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

DEFAULT_FIELDS = "3:1,5:1,13:1,3:5,5:2,13:5,65521:1,65521:2,2147483647:1,0:2"


def corpus(
    path: Path, p: int, d: int, n: int, trials: int, seed: int
) -> dict[str, Any]:
    """Write x then y in native canonical format, with a bound manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(np.uint32 if p else np.complex128)
    entropy = [seed, p, d, n, trials]
    config = {
        "char": p,
        "deg": d,
        "dimension": n,
        "trials": trials,
        "seed": entropy,
        "dtype": dtype.str,
        "generator": "PCG64",
    }
    manifest_path = path.with_suffix(".json")
    if path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest["config"] != config
            or manifest["sha256"] != real_benchmark.sha256_file(path)
            or path.stat().st_size != 2 * trials * n * (d * 4 if p else 16)
        ):
            raise ValueError(f"existing corpus or manifest differs: {path}")
        return manifest
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))
    with path.open("wb") as output:
        for _ in range(2):
            if p:
                values = rng.integers(0, p, (trials, n, d), dtype=np.uint32)
            else:
                values = rng.standard_normal((trials, n)) + 1j * rng.standard_normal(
                    (trials, n)
                )
            np.asarray(values, dtype=dtype).tofile(output)
    manifest = {
        "config": config,
        "sha256": real_benchmark.sha256_file(path),
        "bytes": path.stat().st_size,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def summary(run: dict[str, Any]) -> dict[str, Any]:
    """Summarize paired blocks separately by mode and algorithm."""
    result: dict[str, Any] = {}
    for mode in ("single", "batch"):
        observations = [o for o in run["observations"] if o["mode"] == mode]
        if not observations:
            continue
        names = sorted({o["algorithm"] for o in observations})
        values = {
            a: [
                o["elapsed_ns"] / o["operations"]
                for o in observations
                if o["algorithm"] == a
            ]
            for a in names
        }
        result[mode] = {
            "ns_per_product": {
                a: real_benchmark.summarize(v) for a, v in values.items()
            },
            "ratios_to_quasilinear": {
                a: real_benchmark.paired_ratio(v, values["quasilinear"])
                for a, v in values.items()
                if a != "quasilinear"
            },
        }
    return result


def report(path: Path, document: dict[str, Any]) -> None:
    lines = [
        "# Complex and finite-field benchmark",
        "",
        f"Measured {document['configuration']['trials']:,} pairs per cell, "
        f"{document['configuration']['blocks']} blocks, on {document['cpu']}.",
        "",
        "Times are means in ns/product. Ratios are paired geometric means "
        "with 95% Student-t intervals; values above 1 favor quasilinear.",
        "",
        "| Field | N | Mode | Direct | Quasilinear | Direct/QL (95% CI) |",
        "|---|---:|---|---:|---:|---|",
        "",
    ]
    for run in document["runs"]:
        field = f"F({run['char']}^{run['deg']})" if run["char"] else "C"
        for mode, stats in run["summary"].items():
            means = stats["ns_per_product"]
            ratio = stats["ratios_to_quasilinear"]["direct"]
            lo, hi = ratio["ci95"]
            lines.append(
                f"| {field} | {run['dimension']} | {mode} | "
                f"{means['direct']['mean']:.2f} | {means['quasilinear']['mean']:.2f} | "
                f"{ratio['geometric_mean']:.3f} ({lo:.3f}–{hi:.3f}) |"
            )
    lines += [
        "",
        "All corpus pairs were verified against an independent schoolbook "
        "polynomial / defining-basis oracle outside timing. Complex checks use "
        "long-double accumulation and a norm-scaled error bound.",
        "",
        "Raw observations, corpus hashes, selected backend, exact modulus, "
        "runner/build/source hashes and optional baseline measurements are "
        "recorded in the adjacent JSON. Finite AUTO remains quasilinear; "
        "these machine-specific measurements do not change the real defaults.",
        "",
    ]
    path.write_text("\n".join(lines))


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-metadata", type=Path)
    parser.add_argument(
        "--corpora", type=Path, default=Path("benchmark-corpora/domains")
    )
    parser.add_argument("--fields", default=DEFAULT_FIELDS)
    parser.add_argument("--dimensions", default="2,4,8,16,32,64,128,256,512,1024")
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--blocks", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--min-time-ms", type=float, default=20)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--modes", default="batch,single")
    parser.add_argument("--subprocess-timeout", type=float, default=3600)
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="also measure the three-real-product baseline (complex fields only)",
    )
    args = parser.parse_args(argv)
    if args.output.suffix.lower() == ".md":
        parser.error("output and Markdown report paths must differ")
    if (
        args.trials < 1
        or args.blocks < 2
        or args.warmup < 0
        or not math.isfinite(args.min_time_ms)
        or args.min_time_ms < 0
        or args.seed < 0
        or not math.isfinite(args.subprocess_timeout)
        or args.subprocess_timeout <= 0
    ):
        parser.error("positive trials and a balanced block count are required")
    try:
        args.dimension_list = sorted({int(v) for v in args.dimensions.split(",")})
        args.field_pairs = [
            tuple(map(int, v.split(":"))) for v in args.fields.split(",")
        ]
    except ValueError:
        parser.error("dimensions must be integers and fields must be char:deg pairs")
    if any(n < 1 or n & (n - 1) for n in args.dimension_list):
        parser.error("dimensions must be positive powers of two")
    if any(
        len(f) != 2
        or f[0] < 0
        or f[1] < 1
        or f[1] > 12
        or (f[0] == 0 and f[1] != 2)
        or (f[0] and (f[0] < 3 or f[0] > 2147483647 or f[0] % 2 == 0))
        for f in args.field_pairs
    ):
        parser.error("fields are char:deg pairs (C is 0:2)")
    modes = args.modes.split(",")
    if any(mode not in ("single", "batch") for mode in modes):
        parser.error("modes must be a comma-separated subset of batch,single")
    args.modes = tuple(m for m in ("batch", "single") if m in modes)
    if args.blocks % ((3 if args.baseline else 2) * len(args.modes)):
        parser.error("block count must balance all algorithm/mode tasks")
    if args.baseline and any(p for p, _ in args.field_pairs):
        parser.error(
            "--baseline requires --fields 0:2; finite comparisons use the flint protocol"
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
    args.runner = args.runner.resolve()
    metadata = real_benchmark.run_json([str(args.runner), "--metadata"])
    root = Path(__file__).resolve().parents[1]
    document = {
        "schema": "fastcd-domains-report-1",
        "cpu": real_benchmark.cpu_name(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "configuration": vars(args).copy(),
        "git": real_benchmark.git_state(root),
        "metadata": metadata,
        "provenance": real_benchmark.provenance(args),
        "cpu_state_before": real_benchmark.cpu_state(args.cpu),
        "runs": [],
    }
    document["configuration"] = {
        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(args.output.stem + ".partial.json")
    for p, d in args.field_pairs:
        for n in args.dimension_list:
            path = args.corpora / f"p{p}-d{d}-n{n}-t{args.trials}-s{args.seed}.bin"
            manifest = corpus(path, p, d, n, args.trials, args.seed)
            print(f"char={p}, deg={d}, N={n}: verifying and timing", flush=True)
            start = time.monotonic()
            command = [
                str(args.runner),
                "--corpus",
                str(path.resolve()),
                "--char",
                str(p),
                "--deg",
                str(d),
                "--dimension",
                str(n),
                "--trials",
                str(args.trials),
                "--blocks",
                str(args.blocks),
                "--warmup",
                str(args.warmup),
                "--min-time-ms",
                f"{args.min_time_ms:g}",
                "--mode",
                "both" if len(args.modes) == 2 else args.modes[0],
                "--baseline",
                str(int(args.baseline)),
            ]
            run = real_benchmark.run_json(command, args.subprocess_timeout)
            if run["verified"] != args.trials:
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
