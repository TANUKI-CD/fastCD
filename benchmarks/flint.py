#!/usr/bin/env python3
"""Compare fastCD finite-field arithmetic with FLINT (Fast Library for Number Theory).

Scalar multiplication uses the normal public fastCD and FLINT coefficient APIs.
Complete CD products compare fastCD's quasilinear dispatcher with the maintained
CD recurrence instantiated using FLINT coefficients. Each field uses identical
polynomial-basis inputs and an explicit common defining polynomial. Results are
reported separately for these two workloads; no Cariow--Cariowa is measured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from . import benchmark as common
    from . import domains
except ImportError:
    import benchmark as common
    import domains

DEFAULT_FIELDS = ",".join(
    f for f in domains.DEFAULT_FIELDS.split(",") if not f.startswith("0:")
)


def parse_fields(text: str) -> list[tuple[int, int]]:
    """Reject invalid fields before producing a corpus or starting a run."""
    try:
        fields = [tuple(map(int, f.split(":"))) for f in text.split(",")]
    except ValueError as error:
        raise ValueError("fields must be comma-separated char:deg pairs") from error
    for field in fields:
        if len(field) != 2:
            raise ValueError("fields must be comma-separated char:deg pairs")
        p, d = field
        if not 3 <= p <= 2147483647 or p % 2 == 0 or not 1 <= d <= 12:
            raise ValueError(
                "FLINT fields require an odd prime <= 2147483647 and deg 1..12"
            )
        if any(p % divisor == 0 for divisor in range(3, math.isqrt(p) + 1, 2)):
            raise ValueError(f"characteristic {p} is not prime")
    if len(fields) != len(set(fields)):
        raise ValueError("fields must not contain duplicates")
    return [(p, d) for p, d in fields]


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-metadata", type=Path)
    parser.add_argument("--corpora", type=Path, default=Path("benchmark-corpora/flint"))
    parser.add_argument("--fields", default=DEFAULT_FIELDS)
    parser.add_argument("--dimensions", default="2,4,8,16,32,64,128,256,512,1024")
    parser.add_argument(
        "--workloads", default="scalar,cd", help="comma-separated subset of scalar,cd"
    )
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--blocks", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--min-time-ms", type=float, default=20)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--subprocess-timeout", type=float, default=3600)
    args = parser.parse_args(argv)
    if args.output.suffix.lower() == ".md":
        parser.error("output and Markdown report paths must differ")
    if (
        args.trials < 1
        or args.blocks < 2
        or args.blocks % 2
        or args.warmup < 0
        or not math.isfinite(args.min_time_ms)
        or args.min_time_ms < 0
        or args.seed < 0
        or not math.isfinite(args.subprocess_timeout)
        or args.subprocess_timeout <= 0
    ):
        parser.error(
            "positive trials/timeout, an even positive block count and nonnegative timing options are required"
        )
    try:
        args.field_pairs = parse_fields(args.fields)
        args.dimension_list = sorted({int(n) for n in args.dimensions.split(",")})
        if any(n < 1 or n > 1024 or n & (n - 1) for n in args.dimension_list):
            raise ValueError("dimensions must be powers of two from 1 through 1024")
        workloads = args.workloads.split(",")
        if any(w not in ("scalar", "cd") for w in workloads):
            raise ValueError("workloads must be a comma-separated subset of scalar,cd")
        args.workloads = tuple(w for w in ("scalar", "cd") if w in workloads)
    except ValueError as error:
        parser.error(str(error))
    return args


def dependency_files(runner: Path) -> dict[str, str]:
    """Hash resolved shared libraries when the platform exposes their paths."""
    command = (
        ["otool", "-L", str(runner)]
        if sys.platform == "darwin"
        else ["ldd", str(runner)]
    )
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return {}
    files = {}
    for line in result.stdout.splitlines():
        words = line.strip().split()
        candidates = words[2:3] if len(words) > 2 and words[1] == "=>" else words[:1]
        for word in candidates:
            path = Path(word)
            if (
                path.is_absolute()
                and path.is_file()
                and path.resolve() != runner.resolve()
            ):
                files[str(path.resolve())] = common.sha256_file(path)
    return dict(sorted(files.items()))


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Bind the runner, configured dependency and actual loaded library files."""
    for path in (
        args.output,
        args.output.with_suffix(".md"),
        args.output.with_name(args.output.stem + ".partial.json"),
    ):
        if path.exists():
            raise ValueError(f"refusing to overwrite {path}")
    args.runner = args.runner.resolve()
    if args.build_metadata is None:
        candidates = list(args.runner.parent.glob("fastcd-flint-build-*.txt"))
        if len(candidates) != 1:
            raise ValueError(
                "rebuild with FASTCD_BUILD_FLINT_BENCHMARK=ON; supply --build-metadata if needed"
            )
        args.build_metadata = candidates[0]
    metadata = common.run_json(
        [str(args.runner), "--metadata"], args.subprocess_timeout
    )
    if metadata.get("schema") != "fastcd-flint-1":
        raise ValueError("--runner must name fastcd_flint_benchmark")
    if metadata["flint_compile_version"] != metadata["flint_runtime_version"]:
        raise ValueError(
            "FLINT compile/runtime versions differ; rebuild with a consistent installation"
        )
    identity = common.provenance(args)
    record = dict(
        line.split("=", 1)
        for line in identity["build_record"].splitlines()
        if "=" in line
    )
    library = Path(record["flint_library"])
    if common.sha256_file(library) != record["flint_library_sha256"]:
        raise ValueError(
            "FLINT library changed since configuration; rebuild the runner"
        )
    headers = Path(record["flint_include"]) / "flint"
    digest = hashlib.sha256(
        "".join(
            f"{common.sha256_file(h)}  {h.name}\n" for h in sorted(headers.glob("*.h"))
        ).encode()
    ).hexdigest()
    if digest != record["flint_headers_sha256"]:
        raise ValueError(
            "FLINT headers changed since configuration; rebuild the runner"
        )
    dependencies = dependency_files(args.runner)
    if any(
        "libflint" in Path(name).name and checksum != record["flint_library_sha256"]
        for name, checksum in dependencies.items()
    ):
        raise ValueError("loaded FLINT library differs from the configured library")
    return {
        "metadata": metadata,
        "provenance": identity,
        "runtime_dependencies_sha256": dependencies,
    }


def summarize(run: dict[str, Any], blocks: int) -> dict[str, Any]:
    """Pair by block identity, checking verification and identical workloads."""
    values: dict[str, list[float]] = {}
    all_checksums = set()
    for backend in ("fastcd", "flint"):
        samples = sorted(
            (o for o in run["observations"] if o["backend"] == backend),
            key=lambda o: o["block"],
        )
        if [o["block"] for o in samples] != list(range(blocks)):
            raise ValueError("missing or duplicate paired observation blocks")
        if any(
            o["verified_pairs"] != run["trials"]
            or o["passes"] != run["passes"]
            or o["operations"] != run["trials"] * run["passes"]
            or o["elapsed_ns"] <= 0
            for o in samples
        ):
            raise ValueError("unverified or unmatched timed corpus replay")
        all_checksums.update(o["checksum"] for o in samples)
        values[backend] = [o["elapsed_ns"] / o["operations"] for o in samples]
    if len(run["observations"]) != 2 * blocks or len(all_checksums) != 1:
        raise ValueError("backend checksums or observation counts differ")
    if run["verified"] != run["trials"]:
        raise ValueError("runner did not independently verify the full corpus")
    return {
        "ns_per_product": {b: common.summarize(v) for b, v in values.items()},
        "flint_over_fastcd": common.paired_ratio(values["flint"], values["fastcd"]),
    }


def report(path: Path, document: dict[str, Any]) -> None:
    metadata = document["metadata"]
    lines = [
        "# Finite-field comparison with FLINT",
        "",
        "FLINT (Fast Library for Number Theory) is an independent C library. "
        "This experiment compares its nmod_mul (prime fields) and fq_nmod_mul "
        "(polynomial-basis extensions) with fastCD's public arithmetic.",
        "",
        f"FLINT {metadata['flint_runtime_version']}; one arithmetic thread; "
        f"{document['configuration']['trials']:,} pairs per cell; "
        f"{document['configuration']['blocks']} paired blocks "
        "at separate stack offsets; "
        f"{document['configuration']['warmup']} warmup passes; "
        f"{document['configuration']['min_time_ms']:g} ms calibration target; {document['cpu']}.",
        "",
        "Times are means in ns/product. FLINT/fastCD is the geometric mean "
        "of paired block ratios with a 95% Student-t interval. Values above "
        "1 favor fastCD. Scalar and complete CD products are separate workloads.",
        "",
    ]
    for kind, title in (
        ("scalar", "Scalar field multiplication"),
        ("cd", "Complete CD multiplication"),
    ):
        runs = [r for r in document["runs"] if r["kind"] == kind]
        if not runs:
            continue
        lines += [
            f"## {title}",
            "",
            "| Field | N | fastCD | FLINT | FLINT/fastCD (95% CI) |",
            "|---|---:|---:|---:|---| ",
        ]
        for run in runs:
            means = run["summary"]["ns_per_product"]
            ratio = run["summary"]["flint_over_fastcd"]
            lo, hi = ratio["ci95"]
            lines.append(
                f"| F({run['char']}^{run['deg']}) | {run['dimension']} | "
                f"{means['fastcd']['mean']:.2f} | {means['flint']['mean']:.2f} | "
                f"{ratio['geometric_mean']:.3f} ({lo:.3f}–{hi:.3f}) |"
            )
        lines.append("")
    lines += [
        "CD results compare fastCD's public quasilinear dispatcher with the "
        "same maintained algebraic recurrence using FLINT coefficients. FLINT "
        "does not provide the CD algorithm. Native degree specialization, "
        "packed SIMD backends, element layouts and normalization costs are "
        "part of this implementation comparison; scalar results measure "
        "coefficient multiplication separately.",
        "",
        "Both backends use identical operands and defining polynomials, "
        "preallocated outputs/scratch and a common calibrated pass count. "
        "Setup and conversions are excluded; work inside FLINT's arithmetic "
        "routines is included. Every input pair and every actual final timed "
        "buffer were checked against an independent polynomial/basis oracle.",
        "",
        "The adjacent JSON records exact moduli, corpora, raw observations, "
        "verification counts, versions, build/source hashes and resolved shared "
        "library hashes where available. This fixed selection of fields is "
        "not a guarantee for other fields, representations or machines.",
        "",
    ]
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    identity = preflight(args)
    if args.cpu is None and hasattr(os, "sched_getaffinity"):
        args.cpu = min(os.sched_getaffinity(0))
    if args.cpu is not None:
        if not hasattr(os, "sched_setaffinity"):
            raise ValueError("CPU affinity is unavailable on this platform")
        os.sched_setaffinity(0, {args.cpu})
    document: dict[str, Any] = {
        "schema": "fastcd-flint-report-1",
        "status": "incomplete",
        "cpu": common.cpu_name(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "git": common.git_state(Path(__file__).resolve().parents[1]),
        "configuration": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        **identity,
        "cpu_state_before": common.cpu_state(args.cpu),
        "runs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(args.output.stem + ".partial.json")
    partial.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    for p, d in args.field_pairs:
        for kind in args.workloads:
            dimensions = [1] if kind == "scalar" else args.dimension_list
            for n in dimensions:
                path = args.corpora / f"p{p}-d{d}-n{n}-t{args.trials}-s{args.seed}.bin"
                manifest = domains.corpus(path, p, d, n, args.trials, args.seed)
                print(f"FLINT {kind}: char={p}, deg={d}, N={n}", flush=True)
                command = [
                    str(args.runner),
                    "--corpus",
                    str(path.resolve()),
                    "--kind",
                    kind,
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
                    "--seed",
                    str(args.seed),
                ]
                started = time.monotonic()
                run = common.run_json(command, args.subprocess_timeout)
                run.update(
                    {
                        "summary": summarize(run, args.blocks),
                        "corpus": manifest,
                        "wall_seconds": time.monotonic() - started,
                    }
                )
                document["runs"].append(run)
                partial.write_text(
                    json.dumps(document, indent=2, allow_nan=False) + "\n"
                )
    # Reject a run whose executable, source, build record or dependencies
    # changed while observations were being collected
    if (
        common.provenance(args) != identity["provenance"]
        or dependency_files(args.runner) != identity["runtime_dependencies_sha256"]
    ):
        raise ValueError(
            "build or runtime dependencies changed during the run; discard timings"
        )
    document.update({"status": "ok", "cpu_state_after": common.cpu_state(args.cpu)})
    args.output.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    report(args.output.with_suffix(".md"), document)
    partial.unlink()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError) as error:
        raise SystemExit(str(error)) from None
