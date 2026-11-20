"""Corpus-integrity, paired-statistics and runner checks for the sum benchmark."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import sums as driver


def test_corpus_reuse_integrity_and_exact_shape(tmp_path):
    path = tmp_path / "corpus.bin"
    first = driver.corpus(path, 8, 16, 17)
    assert driver.corpus(path, 8, 16, 17) == first
    assert np.fromfile(path, dtype="<f8").size == 2 * 16 * 8
    with path.open("r+b") as output:
        output.write(b"BAD!")
    with pytest.raises(ValueError, match="differs"):
        driver.corpus(path, 8, 16, 17)


def test_paired_statistics_follow_the_automatic_plane_algorithm():
    times = {"products": 800, "records": 200, "planes_direct": 100,
             "planes_quasilinear": 50, "dgemm": 400}
    run = {"default_sum_algorithm": "quasilinear", "observations": []}
    for block in range(5):
        for task, time in times.items():
            run["observations"].append(
                {
                    "block": block,
                    "task": task,
                    "elapsed_ns": time * (block + 1),
                    "operations": block + 1,
                }
            )
    result = driver.summary(run)
    assert result["ns_per_element"]["planes"]["mean"] == 50
    expected = {
        "products/records": 4,
        "products/planes": 16,
        "dgemm/records": 2,
        "dgemm/planes": 8,
        "planes_direct/planes_quasilinear": 2,
    }
    for name, value in expected.items():
        ratio = result["ratios"][name]
        assert ratio["geometric_mean"] == pytest.approx(value)
        assert ratio["ci95"] == pytest.approx([value, value])
    del times["dgemm"], times["planes_direct"]
    run["observations"] = [o for o in run["observations"] if o["task"] in times]
    assert set(driver.summary(run)["ratios"]) == {
        "products/records", "products/planes"
    }
    # Without the plane task of the automatic algorithm there is no planes entry
    run["default_sum_algorithm"] = "direct"
    result = driver.summary(run)
    assert "planes" not in result["ns_per_element"]
    assert set(result["ratios"]) == {"products/records"}


def test_options_that_cannot_be_reported_are_rejected(tmp_path):
    base = ["--runner", str(tmp_path / "runner"), "--output", str(tmp_path / "o.json")]
    with pytest.raises(SystemExit):
        driver.parse_arguments([*base, "--max-direct-dimension", "4"])
    with pytest.raises(SystemExit):
        driver.parse_arguments([*base, "--counts", "1000"])
    with pytest.raises(SystemExit):
        driver.parse_arguments([*base, "--blocks", "50"])
    assert driver.parse_arguments(base).seed == 20260906


def test_report_states_what_was_verified(tmp_path):
    stats = {"mean": 1.0}
    ratio = {"geometric_mean": 2.0, "ci95": [1.5, 2.5]}
    document = {
        "cpu": "test", "blas": None,
        "configuration": {"trials": 64, "blocks": 60, "verify_samples": 0},
        "runs": [{
            "dimension": 8, "count": 64, "exact_samples": 0, "samples": 1,
            "summary": {"ns_per_element": {"products": stats, "records": stats},
                        "ratios": {"products/records": ratio}},
        }],
    }
    driver.report(tmp_path / "r.md", document)
    assert "verified 0 of 1 sample." in (tmp_path / "r.md").read_text()
    document["configuration"]["verify_samples"] = 8
    document["runs"][0]["exact_samples"] = 1
    document["runs"].append(dict(document["runs"][0], count=4, samples=16, exact_samples=8))
    driver.report(tmp_path / "r.md", document)
    text = (tmp_path / "r.md").read_text()
    assert "verified 9 of 17 samples" in text


@pytest.fixture
def runner():
    root = Path(__file__).resolve().parents[1]
    runner = Path(
        os.environ.get("FASTCD_SUMS_RUNNER", root / "build/fastcd_sums_benchmark")
    )
    if not runner.is_file():
        if "FASTCD_SUMS_RUNNER" in os.environ:
            pytest.fail(f"FASTCD_SUMS_RUNNER is not a file: {runner}")
        pytest.skip("native sum runner is not built here")
    return str(runner.resolve())


@pytest.mark.parametrize("option,value", [
    ("--verify-samples", "-1"), ("--warmup", " -1"), ("--blocks", "-2"),
])
def test_native_runner_rejects_negative_counts(runner, option, value):
    result = subprocess.run([runner, option, value], capture_output=True, text=True)
    assert result.returncode != 0
    assert "invalid number" in result.stderr


@pytest.mark.parametrize("n,count", [(1, 8), (2, 5), (8, 16), (16, 24), (64, 8), (128, 8)])
def test_native_runner_verifies_and_balances_tasks(tmp_path, runner, n, count):
    path = tmp_path / "elements.bin"
    trials = 3 * count
    driver.corpus(path, n, trials, 17)
    command = [
        runner,
        "--corpus",
        str(path),
        "--dimension",
        str(n),
        "--count",
        str(count),
        "--trials",
        str(trials),
        "--blocks",
        "12",
        "--warmup",
        "0",
        "--min-time-ms",
        "0",
        "--verify-samples",
        "3",
    ]
    run = json.loads(subprocess.check_output(command, text=True))
    tasks = {"products", "records", "planes_quasilinear"}
    if n <= 64:
        tasks.add("planes_direct")
    assert run["verified_samples"] == 3 and run["exact_samples"] == 3
    assert run["default_sum_algorithm"] == ("direct" if n <= 8 else "quasilinear")
    assert len(run["observations"]) == 12 * len(tasks)
    for block in range(12):
        current = [o for o in run["observations"] if o["block"] == block]
        assert {o["task"] for o in current} == tasks
    # The fused layouts agree bit for bit when they run the same algorithm
    selected = f"planes_{run['default_sum_algorithm']}"
    sums = {o["task"]: o["checksum"] for o in run["observations"]}
    assert sums["records"] == sums[selected]
