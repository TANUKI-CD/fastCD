"""Fast checks of the stability study tools that need no native build."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import stability_archive as archive
import stability_block_size as block_size
import stability_cc_performance as cc_performance
import stability_speed as speed
import stability_speed_report as speed_report
from stability_timing import interval, paired


def test_paired_ratio_is_the_median_of_block_ratios():
    result = paired([10.0, 10.0, 10.0], [11.0, 12.0, 13.0])
    assert result["ratio"] == pytest.approx(1.2)
    low, high = result["ratio_95ci"]
    assert 1.1 <= low <= result["ratio"] <= high <= 1.3
    assert interval([1.0, 1.0, 1.0]) == [1.0, 1.0]


def speed_result() -> dict:
    values = {}
    for version, factor in (("sequential", 1.0), ("balanced", 1.05)):
        for algorithm in ("quasilinear", "direct"):
            base = 100.0 if algorithm == "quasilinear" else 400.0
            values["single", algorithm, version] = [base * (factor if algorithm == "quasilinear" else 1.0)] * 4
    summary = {"domain": "real", "dimension": 64, "backend": "simd", "mode": "single",
               **speed.summarize(values, "single")}
    return {"completed": True, "blocks": 4, "affinity": [3], "summary": [summary]}


def test_speed_summary_and_table_compare_balanced_with_sequential():
    data = speed_result()
    row = data["summary"][0]
    assert row["balanced_over_sequential"] == pytest.approx(1.05)
    assert row["direct_control"] == pytest.approx(1.0)
    assert row["direct_over_balanced"] == pytest.approx(400 / 105)
    table = speed_report.render(data)
    assert "| real | simd | single | 64 | 100 | 105 | 1.05 [1.05, 1.05] | 400 | 3.81 | 1 |" in table
    assert "pinned to CPU 3" in table
    assert speed_report.summary_csv(data).splitlines()[0].startswith("domain,backend,mode,dimension")


def test_partial_timings_are_refused():
    with pytest.raises(ValueError, match="did not finish"):
        speed_report.render({"completed": False})


def test_cc_table_uses_direct_as_the_only_control():
    summary = []
    for algorithm, ratio in (("cariow_cariowa", 1.02), ("quasilinear", 0.9), ("direct", 1.01)):
        values = {("single", algorithm, "baseline"): [10.0] * 12, ("single", algorithm, "measured"): [10.0 * ratio] * 12}
        summary.append({"backend": "simd", "dimension": 512, "mode": "single", "algorithm": algorithm,
                        **cc_performance.summarize(values, "single", algorithm)})
    table = cc_performance.render({"completed": True, "blocks": 12, "affinity": [3], "summary": summary})
    rows = [line for line in table.splitlines() if line.startswith("| simd")]
    assert rows == ["| simd | single | 512 | 10 | 10.2 | 1.02 [1.02, 1.02] | 1.01 |"]


def test_block_size_ratios_compare_with_block_eight():
    values = {4: [11.0] * 3, 8: [10.0] * 3, 16: [9.5] * 3}
    summary = block_size.summarize(values)
    assert summary["ratio_4_over_8"] == pytest.approx(1.1)
    assert summary["ratio_16_over_8"] == pytest.approx(0.95)
    data = {"completed": True, "blocks": 3, "affinity": [3], "block_sizes": [4, 8, 16],
            "reference_block_size": 8, "summary": [{"domain": "real", "dimension": 64, "mode": "single", **summary}]}
    assert "| real | single | 64 | 11 | 10 | 9.5 | 1.1 [1.1, 1.1] | 0.95 [0.95, 0.95] |" in block_size.render(data)
    assert block_size.parse_build("4=/tmp/b")[0] == 4


def test_sections_are_replaced_and_extracted_exactly():
    text = "a\n<!-- BEGIN STABILITY ACCURACY -->\nold\n<!-- END STABILITY ACCURACY -->\nb\n"
    updated = archive.replace_section(text, archive.ACCURACY_SECTION, "new table\n")
    assert updated.startswith("a\n<!-- BEGIN STABILITY ACCURACY -->\n\nnew table\n\n<!-- END STABILITY ACCURACY -->")
    assert updated.endswith("\nb\n")
    assert archive.extract_section(updated, archive.ACCURACY_SECTION) == "new table\n"
    with pytest.raises(ValueError, match="exactly one"):
        archive.extract_section("no markers", archive.SPEED_SECTION)


def test_checksums_cover_raw_files_and_round_trip(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "speed.json").write_text("{}\n")
    (tmp_path / "README.md").write_text("# Study\n")
    assert archive.write_checksums(tmp_path) == 2
    entries = archive.read_checksums(tmp_path)
    assert sorted(entries) == ["README.md", "raw/speed.json"]
    assert archive.is_raw("raw/speed.json") and not archive.is_raw("README.md")


def write_study(directory: Path, profile: str = "full") -> None:
    environment = {"cpu_model": "Test CPU", "platform": "Linux", "python": "3.14.8 (main)", "numpy": "2.5.4",
                   "gmpy2": "2.3.2", "mpfr": "MPFR 4.2.2", "matplotlib": "3.11.2"}
    study = {"date": "2099-01-01", "profile": profile, "baseline_commit": "a" * 40, "commit": "b" * 40,
             "environment": environment, "timing_cpu": 3, "accuracy_cpu": 4, "tools": "measured commit",
             "command_line": "python3 benchmarks/stability_study.py --cpu 3",
             "steps": {"accuracy": {"seconds": 301.2}, "speed": {"seconds": 12.0}}}
    record = "compiler=/usr/bin/cc\nversion=16.2.1\nconfiguration_flags=-O3 -DNDEBUG\noptions=-fno-fast-math\n"
    (directory / "study.json").write_text(json.dumps(study))
    (directory / "builds.json").write_text(json.dumps({"builds": {"balanced-simd": {"build_record": record}}}))


def test_readme_describes_the_comparison_files_and_verification(tmp_path):
    write_study(tmp_path)
    (tmp_path / "accuracy-tables.md").write_text("| x |\n")
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "accuracy.json").write_text("{}\n")
    text = archive.readme(tmp_path)
    assert text.startswith("# Stability study, 2099-01-01\n")
    assert f"commit `{'a' * 40}`" in text and f"commit `{'b' * 40}`" in text
    assert "`-O3 -DNDEBUG -fno-fast-math`" in text
    assert "| accuracy | Measure every error against the exact product. | 5 min 01 s |" in text
    assert "| `accuracy-tables.md` |" in text and "| `raw/accuracy.json` |" in text
    assert "python3 tools/check_stability_archives.py" in text
    assert "smoke" not in text
    lowered = text.lower()
    for word in ("proof", "original", "previous", "unchanged", "retain"):
        assert word not in lowered


def test_readme_marks_a_smoke_run(tmp_path):
    write_study(tmp_path, "smoke")
    assert "**smoke** run, made to test the tooling" in archive.readme(tmp_path)


def test_speed_key_figures_summarize_ratio_ranges():
    rows = []
    for domain, backend, n, ratio, control in (("real", "simd", 64, 1.03, 1.01), ("real", "simd", 1024, 1.01, 0.99),
                                               ("real", "scalar", 64, 1.10, 1.0), ("complex", "simd", 64, 1.07, 1.06)):
        rows.append({"domain": domain, "backend": backend, "mode": "single", "dimension": n,
                     "balanced_over_sequential": ratio, "balanced_over_sequential_95ci": [ratio, ratio],
                     "direct_control": control, "direct_over_balanced": 19.0})
    figures = archive.speed_figures({"summary": rows})
    assert figures["real_simd_from_64"] == [1.01, 1.03]
    assert figures["real_scalar_from_64"] == [1.10, 1.10]
    assert figures["complex_by_dimension"] == {"64": [1.07, 1.07]}
    assert figures["direct_control_real"] == [0.99, 1.01]
    assert figures["direct_over_balanced_real_largest"] == {"dimension": 1024, "single": 19.0}


class FakeLibrary:
    """Returns stored outputs instead of calling a native library."""

    def __init__(self, outputs: dict[bytes, np.ndarray]):
        self.outputs = outputs

    def multiply(self, x, y, algorithm):
        return self.outputs[x.tobytes()]


def test_ties_are_resolved_by_exact_squared_errors():
    pytest.importorskip("gmpy2")
    import stability_ties as ties

    x, y = np.array([1.0, 0.0]), np.array([1.0, 0.0])   # exact product (1, 0)
    corpus = {"real_2_a_0_x": x, "real_2_a_0_y": y, "real_2_b_0_x": 2 * x, "real_2_b_0_y": y}
    sequential = FakeLibrary({x.tobytes(): np.array([1.0, 0.0]), (2 * x).tobytes(): np.array([2.0 + 2**-50, 0.0])})
    balanced = FakeLibrary({x.tobytes(): np.array([1.0, 0.0]), (2 * x).tobytes(): np.array([2.0, 0.0])})
    rows = [{"key": "real_2_a_0", "tied": True, "domain": "real", "dimension": 2, "family": "a", "seed": 1},
            {"key": "real_2_b_0", "tied": True, "domain": "real", "dimension": 2, "family": "b", "seed": 2},
            {"key": "real_2_c_0", "tied": False}]
    result = ties.resolve({"samples": rows}, corpus, sequential, balanced)
    assert result["recorded_ties"] == 2
    assert result["identical_outputs"] == 1
    assert result["balanced_smaller"] == 1 and result["sequential_smaller"] == 0
    assert result["records"][0]["smaller_error"] == "balanced"


def test_cc_cases_are_classified_and_recorded_metrics_are_rechecked():
    pytest.importorskip("gmpy2")
    import stability as study
    import stability_cc_accuracy as cc_accuracy

    x, y = np.array([0.5, 0.25]), np.array([0.75, -0.5])
    reference = study.exact_product(x, y)
    exact = np.array([0.5 * 0.75 + 0.25 * 0.5, -0.5 * 0.5 + 0.25 * 0.75])
    worse = exact + np.array([2.0**-40, 0.0])
    after = study.metric(exact, reference, x, y, 256)
    document = {"samples": [{"key": "real_2_gaussian_0", "dimension": 2, "family": "gaussian", "seed": 1,
                             "operands_sha256": "0", "algorithms": {"cariow_cariowa": after}}]}
    corpus = {"real_2_gaussian_0_x": x, "real_2_gaussian_0_y": y}
    result = cc_accuracy.compare(document, corpus, FakeLibrary({x.tobytes(): worse}),
                                 FakeLibrary({x.tobytes(): exact}), 256)
    assert result["samples"][0]["status"] == "improved"
    assert result["summary"]["aggregate"]["improved"] == 1
    assert result["summary"]["dense_by_dimension"][0]["dimension"] == 2
    tampered = {"samples": [{**document["samples"][0], "algorithms": {"cariow_cariowa": {**after, "absolute": 1.0}}}]}
    with pytest.raises(ValueError, match="differs from the accuracy study"):
        cc_accuracy.compare(tampered, corpus, FakeLibrary({x.tobytes(): worse}), FakeLibrary({x.tobytes(): exact}), 256)
