"""Resolve the recorded ties of an accuracy study exactly.

stability.py records a case as tied when the binary64 values of E_abs for
quasilinear multiplication with sequential (S) and balanced (B) accumulation
are equal. For each tied case this script recomputes both products. Identical
outputs are exact ties; otherwise the exact squared errors decide which error
is smaller, even when both round to the same binary64 norm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import stability as study

QUASILINEAR = 3


def checked_library(path: Path, record: dict[str, Any]) -> study.Library:
    """Load a library only if it is the one the accuracy study measured."""
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError(f"{path} is not the library recorded by the accuracy study")
    return study.Library(path)


def resolve(document: dict[str, Any], corpus: Any, sequential: study.Library,
            balanced: study.Library) -> dict[str, Any]:
    counts = {"recorded_ties": 0, "identical_outputs": 0, "equal_exact_errors": 0,
              "balanced_smaller": 0, "sequential_smaller": 0}
    records = []
    for row in document["samples"]:
        if not row["tied"]:
            continue
        counts["recorded_ties"] += 1
        x, y = corpus[row["key"] + "_x"], corpus[row["key"] + "_y"]
        first = sequential.multiply(x, y, QUASILINEAR)
        second = balanced.multiply(x, y, QUASILINEAR)
        if first.tobytes() == second.tobytes():
            counts["identical_outputs"] += 1
            counts["equal_exact_errors"] += 1
            continue
        old, new = study.squared_errors([first, second], study.exact_product(x, y))
        if old == new:
            counts["equal_exact_errors"] += 1
            continue
        smaller = "balanced" if new < old else "sequential"
        counts[smaller + "_smaller"] += 1
        records.append({"key": row["key"], "domain": row["domain"], "dimension": row["dimension"],
                        "family": row["family"], "seed": row["seed"], "smaller_error": smaller})
    if counts["recorded_ties"] != counts["equal_exact_errors"] + len(records):
        raise AssertionError("tie counts do not add up")
    families: dict[str, int] = {}
    for record in records:
        families[record["family"]] = families.get(record["family"], 0) + 1
    return {**counts, "unresolved_by_family": dict(sorted(families.items())), "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accuracy", type=Path, required=True, help="stability.py JSON results")
    parser.add_argument("--sequential", type=Path, required=True, help="baseline shared library")
    parser.add_argument("--balanced", type=Path, required=True, help="measured shared library")
    parser.add_argument("--output", type=Path, required=True, help="JSON file to write")
    args = parser.parse_args()
    document = json.loads(args.accuracy.read_text())
    if not document.get("completed"):
        raise SystemExit("the accuracy study did not finish")
    sequential = checked_library(args.sequential, document["baseline_library"])
    balanced = checked_library(args.balanced, document["modified_library"])
    with np.load(args.accuracy.with_name(document["corpus"]["path"])) as corpus:
        result = resolve(document, corpus, sequential, balanced)
    output = {"schema": "fastcd-stability-ties-1", "completed": True,
              "command": [sys.executable, *sys.argv],
              "method": "exact squared Euclidean errors on a common dyadic scale",
              "accuracy_sha256": hashlib.sha256(args.accuracy.read_bytes()).hexdigest(), **result}
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
