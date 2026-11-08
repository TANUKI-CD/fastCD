"""Shared helpers for the stability timing drivers.

Every timed pass runs in the native C benchmark runners. Python writes the
operand corpora, chooses the process order, and summarizes paired blocks.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import struct
import subprocess
from pathlib import Path
from typing import Any

RUNNERS = ("fastcd_benchmark", "fastcd_domains_benchmark")
BOOTSTRAP = "4000 paired-block resamples, seed 71193, percentile 95% interval of the median ratio"
CORPUS_DISTRIBUTION = "uniform integers in [-2^20, 2^20) divided by 2^20, from Python random.Random"


def quantile(values: list[float], probability: float) -> float:
    """Interpolate a sample quantile without an additional dependency."""
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def interval(ratios: list[float]) -> list[float]:
    """Deterministic paired bootstrap 95% interval for the median ratio."""
    rng = random.Random(71193)
    medians = [
        statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(4000)
    ]
    return [quantile(medians, 0.025), quantile(medians, 0.975)]


def paired(before: list[float], after: list[float]) -> dict[str, Any]:
    """Median of the per-block ratios after/before, with its bootstrap interval."""
    ratios = [b / a for a, b in zip(before, after, strict=True)]
    return {"ratio": statistics.median(ratios), "ratio_95ci": interval(ratios)}


def invoke(command: list[str]) -> dict[str, Any]:
    """Run one verified native timing task and keep its complete JSON output."""
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    result: dict[str, Any] = json.loads(completed.stdout)
    if result.get("status", "ok") != "ok":
        raise RuntimeError(f"native verification failed: {result}")
    return result


def corpus(path: Path, n: int, trials: int, domain: str, seed: int) -> str:
    """Write seeded uniform dyadic operands and return their SHA-256."""
    rng = random.Random(seed)
    # The real runner reads x, y, z records and the complex runner reads x, y as pairs
    length = (3 if domain == "real" else 4) * n * trials
    values = [(rng.randrange(-1048576, 1048576) / 1048576) for _ in range(length)]
    payload = struct.pack(f"={length}d", *values)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def real_task(
    directory: Path, path: Path, n: int, trials: int, mode: str,
    algorithm: str, minimum_ms: float,
) -> tuple[float, dict[str, Any]]:
    """Time one real algorithm in one independently calibrated observation."""
    result = invoke([
        str(directory / "fastcd_benchmark"), "--corpus", str(path),
        "--dimension", str(n), "--trials", str(trials), "--mode", mode,
        "--algorithm", algorithm, "--repeats", "1", "--warmup", "2",
        "--min-time-ms", str(minimum_ms), "--verify-trials", str(trials),
    ])
    return float(result["observations"][0]["ns_per_operation"]), result


def complex_task(
    directory: Path, path: Path, n: int, trials: int, minimum_ms: float,
    inner_blocks: int = 2,
) -> tuple[dict[tuple[str, str], float], dict[str, Any]]:
    """Time complex direct and quasilinear products, single and batched.

    The native runner rotates its four algorithm/mode tasks; each value is the
    median over its inner observations, in ns per product.
    """
    result = invoke([
        str(directory / "fastcd_domains_benchmark"), "--corpus", str(path),
        "--dimension", str(n), "--trials", str(trials), "--mode", "both",
        "--char", "0", "--deg", "2", "--blocks", str(inner_blocks), "--warmup", "2",
        "--min-time-ms", str(minimum_ms),
    ])
    groups: dict[tuple[str, str], list[float]] = {}
    for observation in result["observations"]:
        key = observation["mode"], observation["algorithm"]
        groups.setdefault(key, []).append(
            observation["elapsed_ns"] / observation["operations"]
        )
    return {key: statistics.median(x) for key, x in groups.items()}, result


def library_file(directory: Path) -> Path:
    """Return the versioned shared library in a build directory."""
    candidates = sorted(
        path for path in directory.glob("libfastcd.so.*")
        if path.is_file() and not path.is_symlink()
    )
    if len(candidates) != 1:
        raise ValueError(f"expected one versioned libfastcd.so in {directory}")
    return candidates[0]


def build_record(directory: Path) -> dict[str, Any]:
    """Identify the runners, the shared library and the recorded compiler inputs."""
    return {
        "directory": str(directory),
        "build_record": (directory / "fastcd-build-Release.txt").read_text(),
        "sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.iterdir())
            if path.is_file() and not path.is_symlink()
            and (path.name in RUNNERS or path.name.startswith("libfastcd.so."))
        },
    }


def pin(cpu: int | None) -> list[int] | None:
    """Pin this process, and the runners it starts, to one CPU."""
    if not hasattr(os, "sched_setaffinity"):
        return None
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    return sorted(os.sched_getaffinity(0))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
