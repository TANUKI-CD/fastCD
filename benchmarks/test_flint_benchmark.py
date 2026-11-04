"""Protocol compatibility, paired verification and native FLINT integration."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import benchmark as legacy
import domains
import flint as driver
import numpy as np
import protocols
import pytest


def test_legacy_default_still_selects_only_real_protocol(tmp_path):
    command = [
        "benchmark.py",
        "--runner",
        "unused",
        "--output",
        str(tmp_path / "real.json"),
    ]
    for extra in ([], ["--protocol", "real"]):
        with mock.patch.object(sys, "argv", command + extra):
            args = legacy.parse_arguments()
            assert args.protocol == "real"
            assert args.trials == 10000 and args.blocks == 24
            assert args.modes == ("batch", "single")
            with (
                mock.patch.object(
                    legacy, "provenance", side_effect=RuntimeError("real path")
                ),
                mock.patch.object(protocols, "run") as alternate,
            ):
                with pytest.raises(RuntimeError, match="real path"):
                    legacy.main()
                alternate.assert_not_called()


@pytest.mark.parametrize(
    "fields", ["0:2", "2:1", "9:2", "13:13", "13", "13:1,13:1", ""]
)
def test_invalid_fields_fail_before_corpus_generation(fields):
    with pytest.raises(ValueError):
        driver.parse_fields(fields)


def observations():
    samples = []
    for block in range(4):
        for backend, ns in (("fastcd", 100), ("flint", 200)):
            samples.append(
                {
                    "block": block,
                    "backend": backend,
                    "elapsed_ns": ns,
                    "operations": 10,
                    "passes": 2,
                    "verified_pairs": 5,
                    "checksum": "same",
                }
            )
    return {"trials": 5, "passes": 2, "verified": 5, "observations": samples}


def test_statistics_pair_by_block_and_define_ratio_direction():
    run = observations()
    run["observations"].reverse()
    result = driver.summarize(run, 4)
    assert result["ns_per_product"]["fastcd"]["mean"] == 10
    assert result["flint_over_fastcd"]["geometric_mean"] == pytest.approx(2)
    assert result["flint_over_fastcd"]["ci95"] == pytest.approx([2, 2])


@pytest.mark.parametrize(
    "fault", ["block", "passes", "verification", "checksum", "extra"]
)
def test_statistics_reject_incomplete_or_unmatched_observations(fault):
    run = observations()
    if fault == "block":
        run["observations"][0]["block"] = 1
    elif fault == "passes":
        run["observations"][0]["passes"] = 3
    elif fault == "verification":
        run["observations"][0]["verified_pairs"] = 0
    elif fault == "checksum":
        run["observations"][0]["checksum"] = "different"
    else:
        run["observations"].append(run["observations"][0].copy())
    with pytest.raises(ValueError):
        driver.summarize(run, 4)


def test_all_missing_flint_fails_before_any_benchmark_is_launched(tmp_path):
    for name in ("fastcd_benchmark", "fastcd_domains_benchmark"):
        (tmp_path / name).touch()
    command = [
        "benchmark.py",
        "--runner",
        str(tmp_path / "fastcd_benchmark"),
        "--protocol",
        "all",
        "--output",
        str(tmp_path / "result.json"),
    ]
    with mock.patch.object(sys, "argv", command):
        args = legacy.parse_arguments()
    with (
        mock.patch.object(legacy, "provenance", return_value={}),
        mock.patch.object(
            legacy, "run_json", return_value={"schema": "fastcd-domains-1"}
        ),
        mock.patch.object(protocols.subprocess, "run") as launched,
    ):
        with pytest.raises(ValueError, match="FLINT_BENCHMARK=ON"):
            protocols.run(args)
        launched.assert_not_called()
    assert not args.output.exists()


@pytest.fixture
def runner():
    root = Path(__file__).resolve().parents[1]
    path = Path(
        os.environ.get(
            "FASTCD_FLINT_RUNNER", root / "build/flint/fastcd_flint_benchmark"
        )
    )
    if not path.is_file():
        pytest.skip("optional FLINT benchmark is not built")
    return path.resolve()


@pytest.mark.parametrize(
    "p,d,n",
    [
        (7, 1, 1),
        (13, 5, 1),
        (3, 3, 32),
        (13, 5, 64),
        (5, 2, 1024),
        (2147483647, 12, 64),
    ],
)
def test_native_comparison_checks_every_timed_buffer(tmp_path, runner, p, d, n):
    path = tmp_path / "pairs.bin"
    domains.corpus(path, p, d, n, 5, 17)
    command = [
        str(runner),
        "--corpus",
        str(path),
        "--kind",
        "scalar" if n == 1 else "cd",
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
        "0.01",
    ]
    result = json.loads(subprocess.check_output(command, text=True))
    assert result["verified"] == 5
    assert result["min_time_ns"] == 10000
    assert len(result["modulus"]) == d + 1
    driver.summarize(result, 4)
    for backend in ("fastcd", "flint"):
        positions = [
            o["order"] for o in result["observations"] if o["backend"] == backend
        ]
        assert positions.count(0) == positions.count(1) == 2
    # A shared NumPy corpus containing a noncanonical residue is rejected
    # before either backend enters its arithmetic or produces observations.
    with path.open("r+b") as output:
        np.asarray([p], dtype=np.uint32).tofile(output)
    failed = subprocess.run(command, capture_output=True, text=True)
    assert failed.returncode != 0 and failed.stdout == ""


def test_all_writes_separate_public_reports_and_matching_corpora(tmp_path, runner):
    real_runner = runner.with_name("fastcd_benchmark" + runner.suffix)
    if not real_runner.exists():
        pytest.skip("other protocol runners are not built")
    output = tmp_path / "all.json"
    command = [
        sys.executable,
        str(Path(legacy.__file__)),
        "--protocol",
        "all",
        "--runner",
        str(real_runner),
        "--output",
        str(output),
        "--fields",
        "13:2, 00:2",
        "--dimensions",
        "2,4",
        "--trials",
        "5",
        "--blocks",
        "4",
        "--warmup",
        "0",
        "--min-time-ms",
        "0",
        "--corpora",
        str(tmp_path / "corpora"),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    suite = json.loads(output.read_text())
    assert suite["status"] == "ok"
    assert [r["protocol"] for r in suite["reports"]] == ["real", "domains", "flint"]
    for record in suite["reports"]:
        assert (tmp_path / record["markdown"]).is_file()
        assert legacy.sha256_file(tmp_path / record["json"]) == record["sha256"]
    real = json.loads((tmp_path / "all.real.json").read_text())
    assert real["schema"] == "fastcd-benchmark-results-2"
    assert real["protocol"]["modes"] == ["batch", "single"]
    domain = json.loads((tmp_path / "all.domains.json").read_text())
    flint = json.loads((tmp_path / "all.flint.json").read_text())
    for measured in flint["runs"]:
        if measured["kind"] == "cd":
            corresponding = next(
                r
                for r in domain["runs"]
                if r["char"] == 13 and r["dimension"] == measured["dimension"]
            )
            assert measured["modulus"] == corresponding["modulus"]
            assert measured["corpus"]["sha256"] == corresponding["corpus"]["sha256"]
    # Repeating a command must preserve the existing complete experiment.
    original = output.read_bytes()
    repeated = subprocess.run(command, capture_output=True, text=True)
    assert repeated.returncode != 0 and output.read_bytes() == original
