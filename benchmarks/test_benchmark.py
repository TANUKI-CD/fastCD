"""Regression tests for benchmark command-line validation."""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    from . import benchmark as driver
except ImportError:  # Support unittest discovery from this directory.
    import benchmark as driver


class BenchmarkArgumentsTests(unittest.TestCase):
    def parse(self, *arguments: str) -> argparse.Namespace:
        command = ["benchmark.py", "--runner", "unused-runner", *arguments]
        with mock.patch.object(sys, "argv", command):
            return driver.parse_arguments()

    def test_valid_modes_preserve_canonical_order(self) -> None:
        for modes, expected in (
            ("batch", ("batch",)),
            ("single", ("single",)),
            ("batch,single", ("batch", "single")),
            ("single,batch", ("batch", "single")),
            ("batch,batch", ("batch",)),
        ):
            with self.subTest(modes=modes):
                arguments = self.parse("--output", "run.json", "--modes", modes)
                self.assertEqual(arguments.modes, expected)

    def test_unknown_or_empty_modes_are_rejected(self) -> None:
        for modes in ("batch,sinlge", "unknown", "", "batch,", ",single"):
            with self.subTest(modes=modes), contextlib.redirect_stderr(
                io.StringIO()
            ) as error:
                with self.assertRaises(SystemExit) as raised:
                    self.parse("--output", "run.json", "--modes", modes)
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("modes", error.getvalue())

    def test_result_paths_have_distinct_markdown_paths(self) -> None:
        for output in ("run.json", "run.JSON", "run", "run.txt"):
            with self.subTest(output=output):
                arguments = self.parse("--output", output)
                self.assertNotEqual(
                    arguments.output, arguments.output.with_suffix(".md")
                )
                self.assertEqual(arguments.modes, ("batch", "single"))

    def test_invalid_output_is_rejected_before_benchmark_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for name in ("run.md", "run.MD"):
                output = Path(directory) / name
                command = [
                    "benchmark.py", "--runner", "unused-runner",
                    "--output", str(output),
                ]
                with self.subTest(output=name), mock.patch.object(
                    sys, "argv", command
                ), mock.patch.object(driver, "provenance") as provenance:
                    with contextlib.redirect_stderr(io.StringIO()) as error:
                        with self.assertRaises(SystemExit) as raised:
                            driver.main()
                    self.assertEqual(raised.exception.code, 2)
                    self.assertIn(".md", error.getvalue())
                    provenance.assert_not_called()
                    self.assertEqual(list(Path(directory).iterdir()), [])


class BenchmarkStatisticsTests(unittest.TestCase):
    def test_summarize_reports_moments_and_interval(self) -> None:
        summary = driver.summarize([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        self.assertEqual(summary["count"], 8)
        self.assertAlmostEqual(summary["mean"], 5.0)
        self.assertAlmostEqual(summary["median"], 4.5)
        self.assertAlmostEqual(summary["sd"], 2.138089935299395)
        half = driver.T95[7] * summary["sd"] / 8 ** 0.5
        self.assertAlmostEqual(summary["ci95"][0], 5.0 - half)
        self.assertAlmostEqual(summary["ci95"][1], 5.0 + half)
        self.assertEqual((summary["min"], summary["max"]), (2.0, 9.0))

    def test_t95_table_is_decreasing_and_ends_near_normal(self) -> None:
        values = driver.T95[1:]
        self.assertEqual(len(values), 30)
        self.assertTrue(all(a > b for a, b in zip(values, values[1:], strict=False)))
        self.assertAlmostEqual(values[0], 12.7062047364)
        self.assertAlmostEqual(values[-1], 2.0422724563)

    def test_paired_ratio_is_a_geometric_mean_with_win_counts(self) -> None:
        ratio = driver.paired_ratio([2.0, 2.0, 8.0], [1.0, 1.0, 2.0])
        self.assertAlmostEqual(ratio["geometric_mean"], (2.0 * 2.0 * 4.0) ** (1 / 3))
        self.assertEqual(ratio["paired_blocks"], 3)
        self.assertEqual(ratio["competitor_faster_blocks"], 0)
        self.assertEqual(ratio["competitor_slower_blocks"], 3)

    def test_fastest_label_marks_inseparable_means_as_ties(self) -> None:
        def result(name: str, observations: list[float]) -> dict[str, object]:
            return {"algorithm": name, "status": "ok",
                    "summary": driver.summarize(observations),
                    "observations_ns_per_product": observations}

        clear = {"direct": result("direct", [10.0, 10.1, 9.9, 10.0]),
                 "quasilinear": result("quasilinear", [5.0, 5.1, 4.9, 5.0]),
                 "cariow_cariowa": {"algorithm": "cariow_cariowa",
                                    "status": "unavailable"}}
        self.assertEqual(driver.fastest_label(clear), "quasilinear")
        tied = {"direct": result("direct", [5.0, 5.2, 4.8, 5.1]),
                "quasilinear": result("quasilinear", [5.1, 4.9, 5.2, 4.9]),
                "cariow_cariowa": {"algorithm": "cariow_cariowa",
                                   "status": "unavailable"}}
        self.assertTrue(driver.fastest_label(tied).startswith("tie ("))
        self.assertEqual(driver.fastest_label({"direct": clear["direct"]}), "direct")
        self.assertEqual(driver.fastest_label({}), "none verified")

    def test_available_algorithms_honor_runner_and_direct_limits(self) -> None:
        arguments = argparse.Namespace(max_direct_dimension=64, max_cc_dimension=1024)
        self.assertEqual(driver.available_algorithms(64, arguments, 64),
                         ["direct", "cariow_cariowa", "quasilinear"])
        self.assertEqual(driver.available_algorithms(128, arguments, 64),
                         ["quasilinear"])
        self.assertEqual(driver.available_algorithms(128, arguments, 128),
                         ["cariow_cariowa", "quasilinear"])

    def test_cariow_cariowa_timing_cap_is_distinct_from_support(self) -> None:
        self.assertEqual(driver.CC_TIMING_CAP, driver.CC_MAX_DIMENSION)
        arguments = argparse.Namespace(max_direct_dimension=1024, max_cc_dimension=64)
        self.assertEqual(driver.available_algorithms(64, arguments, 1024),
                         ["direct", "cariow_cariowa", "quasilinear"])
        self.assertEqual(driver.available_algorithms(128, arguments, 1024),
                         ["direct", "quasilinear"])
        self.assertEqual(driver.absent_status("cariow_cariowa", 128, arguments, 1024)[0],
                         "not_timed")
        self.assertEqual(driver.absent_status("cariow_cariowa", 2048, arguments, 1024)[0],
                         "unavailable")
        self.assertEqual(driver.absent_label({"status": "not_timed"}), "not timed")
        self.assertEqual(driver.absent_label({"status": "unavailable"}), "unavailable")

    def test_git_state_reports_commit_or_none(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outside = driver.git_state(Path(directory))
        self.assertEqual(outside, {"commit": None, "dirty": None, "describe": None})
        self.assertEqual(driver.git_summary(outside),
                         "Commit: unknown (not a git checkout)")
        inside = driver.git_state(Path(driver.__file__).resolve().parents[1])
        if inside["commit"] is not None:
            self.assertRegex(inside["commit"], r"^[0-9a-f]{40}$")
            self.assertIn(inside["dirty"], (True, False))
            self.assertTrue(driver.git_summary(inside).startswith(
                f"Commit: {inside['commit']} ("))

    def test_cpu_state_reports_every_key(self) -> None:
        state = driver.cpu_state(None)
        self.assertEqual(set(state), {"scaling_governor", "intel_pstate_no_turbo",
                                      "cpufreq_boost", "smt_control"})
        self.assertIsNone(state["scaling_governor"])


if __name__ == "__main__":
    unittest.main()
