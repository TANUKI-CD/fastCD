"""Corpus-integrity and paired-block regression checks for field benchmarks."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import domains as driver
import numpy as np
import pytest


@pytest.mark.parametrize("p,d", [(13, 5), (2147483647, 1), (0, 2)])
def test_corpus_reuse_integrity_and_exact_shape(tmp_path, p, d):
    path = tmp_path / "corpus.bin"
    first = driver.corpus(path, p, d, 8, 7, 17)
    assert driver.corpus(path, p, d, 8, 7, 17) == first
    data = np.fromfile(path, dtype=np.uint32 if p else np.complex128)
    assert data.size == 2 * 7 * 8 * (d if p else 1)
    if p:
        assert (data < p).all()
    with path.open("r+b") as output:
        output.write(b"BAD!")
    with pytest.raises(ValueError, match="differs"):
        driver.corpus(path, p, d, 8, 7, 17)


def test_paired_statistics_keep_modes_and_blocks_separate():
    run = {"observations": []}
    for block in range(6):
        for mode in ("single", "batch"):
            for algorithm, time in (("direct", 200), ("quasilinear", 100)):
                run["observations"].append(
                    {
                        "block": block,
                        "mode": mode,
                        "algorithm": algorithm,
                        "elapsed_ns": time * (block + 1),
                        "operations": block + 1,
                    }
                )
    result = driver.summary(run)
    for mode in result.values():
        assert mode["ns_per_product"]["direct"]["mean"] == 200
        ratio = mode["ratios_to_quasilinear"]["direct"]
        assert ratio["geometric_mean"] == pytest.approx(2)
        assert ratio["ci95"] == pytest.approx([2, 2])


@pytest.mark.parametrize("p,d", [(13, 5), (0, 2)])
@pytest.mark.parametrize("n", [2, 4, 8, 32, 256])
def test_native_runner_verifies_and_balances_modes(tmp_path, p, d, n):
    root = Path(__file__).resolve().parents[1]
    runner = Path(
        os.environ.get(
            "FASTCD_DOMAINS_RUNNER", root / "build/domains/fastcd_domains_benchmark"
        )
    )
    if not runner.is_file():
        if "FASTCD_DOMAINS_RUNNER" in os.environ:
            pytest.fail(f"FASTCD_DOMAINS_RUNNER is not a file: {runner}")
        pytest.skip("native domain runner is not built here")
    runner = runner.resolve()
    path = tmp_path / "pairs.bin"
    driver.corpus(path, p, d, n, 5, 17)
    command = [
        str(runner),
        "--corpus",
        str(path),
        "--char",
        str(p),
        "--deg",
        str(d),
        "--dimension",
        str(n),
        "--trials",
        "5",
        "--blocks",
        "4",
        "--warmup",
        "0",
        "--min-time-ms",
        "0",
        "--mode",
        "both",
    ]
    run = json.loads(subprocess.check_output(command, text=True))
    assert run["verified"] == 5 and len(run["observations"]) == 16
    assert run["default_algorithm"] == ("direct" if n <= 4 else "quasilinear")
    assert run["backend"] == ("finite_shared_dispatch" if p else "complex_quasilinear")
    for block in range(4):
        current = [o for o in run["observations"] if o["block"] == block]
        assert {(o["algorithm"], o["mode"]) for o in current} == {
            (a, m) for a in ("direct", "quasilinear") for m in ("single", "batch")
        }
    if p:
        assert len({o["checksum"] for o in run["observations"]}) == 1
