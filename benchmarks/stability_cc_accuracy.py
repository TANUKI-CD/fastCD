"""Compare Cariow–Cariowa errors between the baseline and the measured library.

stability.py records Cariow–Cariowa errors for the measured library on every
real case through dimension 1024. This script evaluates the baseline library on
the same stored operands, computes the same error measures, and classifies
each case as improved, worsened or tied. Equal binary64 norms are resolved by
comparing the outputs and, when they differ, their exact squared errors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import stability as study
from stability_ties import checked_library

CARIOW_CARIOWA = 2
DENSE = ("gaussian", "uniform")


def classify(baseline: dict[str, Any], measured: dict[str, Any], first: np.ndarray,
             second: np.ndarray, reference: tuple[list[int], list[int], int]) -> tuple[str, str]:
    if measured["absolute"] < baseline["absolute"]:
        return "improved", "binary64 norms"
    if measured["absolute"] > baseline["absolute"]:
        return "worsened", "binary64 norms"
    if first.tobytes() == second.tobytes():
        return "tied", "identical outputs"
    old, new = study.squared_errors([first, second], reference)
    return ("improved" if new < old else "worsened" if new > old else "tied"), "exact squared errors"


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts and error statistics of a group of compared cases."""
    def values(side: str, field: str) -> list[float]:
        return [row[side][field] for row in rows if row[side][field] is not None]

    output: dict[str, Any] = {"samples": len(rows)}
    for status in ("improved", "worsened", "tied"):
        output[status] = sum(row["status"] == status for row in rows)
    for side in ("baseline", "measured"):
        eta, relative = values(side, "normalized"), values(side, "relative")
        output[side] = {
            "median_normalized": statistics.median(eta) if eta else None,
            "p99_normalized": float(np.percentile(eta, 99)) if eta else None,
            "maximum_normalized": max(eta) if eta else None,
            "median_relative": statistics.median(relative) if relative else None,
            "maximum_relative": max(relative) if relative else None,
            "median_scalar": statistics.median(values(side, "scalar")) if rows else None,
        }
    return output


def extreme(rows: list[dict[str, Any]], key: Any) -> dict[str, Any] | None:
    if not rows:
        return None
    row = max(rows, key=key)
    return {"key": row["key"], "dimension": row["dimension"], "family": row["family"],
            "baseline_normalized": row["baseline"]["normalized"], "measured_normalized": row["measured"]["normalized"]}


def compare(document: dict[str, Any], corpus: Any, baseline: study.Library,
            measured: study.Library, precision: int) -> dict[str, Any]:
    samples = []
    for row in document["samples"]:
        recorded = row["algorithms"].get("cariow_cariowa")
        if recorded is None:
            continue
        x, y = corpus[row["key"] + "_x"], corpus[row["key"] + "_y"]
        reference = study.exact_product(x, y)
        first = baseline.multiply(x, y, CARIOW_CARIOWA)
        second = measured.multiply(x, y, CARIOW_CARIOWA)
        baseline_error = study.metric(first, reference, x, y, precision)
        recomputed = study.metric(second, reference, x, y, precision)
        if recomputed != recorded:
            raise ValueError(f"measured Cariow–Cariowa output differs from the accuracy study: {row['key']}")
        status, method = classify(baseline_error, recorded, first, second, reference)
        samples.append({"key": row["key"], "dimension": row["dimension"], "family": row["family"],
                        "seed": row["seed"], "operands_sha256": row["operands_sha256"],
                        "baseline": baseline_error, "measured": recorded, "status": status, "decided_by": method,
                        "outputs_identical": first.tobytes() == second.tobytes()})
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in samples:
        groups[row["dimension"], row["family"]].append(row)
    worsened = [row for row in samples if row["status"] == "worsened"]
    summary = {
        "aggregate": aggregate(samples),
        "dense_by_dimension": [
            {"dimension": n, **aggregate([r for r in samples if r["dimension"] == n and r["family"] in DENSE])}
            for n in sorted({r["dimension"] for r in samples})
        ],
        "by_dimension_and_family": [{"dimension": n, "family": family, **aggregate(rows)}
                                    for (n, family), rows in sorted(groups.items())],
        "largest_baseline": extreme([r for r in samples if r["baseline"]["normalized"] is not None],
                                  lambda r: r["baseline"]["normalized"]),
        "largest_measured": extreme([r for r in samples if r["measured"]["normalized"] is not None],
                                 lambda r: r["measured"]["normalized"]),
        "largest_increase": extreme(worsened, lambda r: r["measured"]["normalized"] - r["baseline"]["normalized"]),
        "decided_by_exact_squared_errors": sum(r["decided_by"] == "exact squared errors" for r in samples),
    }
    return {"summary": summary, "samples": samples}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accuracy", type=Path, required=True, help="stability.py JSON results")
    parser.add_argument("--baseline", type=Path, required=True, help="baseline shared library")
    parser.add_argument("--measured", type=Path, required=True, help="measured shared library")
    parser.add_argument("--output", type=Path, required=True, help="JSON file with every compared case")
    parser.add_argument("--summary", type=Path, required=True, help="JSON file with the summaries only")
    args = parser.parse_args()
    document = json.loads(args.accuracy.read_text())
    if not document.get("completed"):
        raise SystemExit("the accuracy study did not finish")
    baseline = checked_library(args.baseline, document["baseline_library"])
    measured = checked_library(args.measured, document["modified_library"])
    with np.load(args.accuracy.with_name(document["corpus"]["path"])) as corpus:
        result = compare(document, corpus, baseline, measured, document["precision_bits"])
    header = {"schema": "fastcd-stability-cc-accuracy-1", "completed": True,
              "command": [sys.executable, *sys.argv],
              "accuracy_sha256": hashlib.sha256(args.accuracy.read_bytes()).hexdigest(),
              "baseline_library_sha256": document["baseline_library"]["sha256"],
              "measured_library_sha256": document["modified_library"]["sha256"]}
    args.output.write_text(json.dumps({**header, **result}, indent=2, allow_nan=False) + "\n")
    args.summary.write_text(json.dumps({**header, **result["summary"]}, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
