"""Check that scalar, SIMD, single and batched products agree bit for bit.

For every stored operand pair of an accuracy study, quasilinear products are
computed four ways: single products and one batch per domain and dimension,
each with the SIMD build and with the scalar build (FASTCD_DISABLE_SIMD=1).
Real Cariow–Cariowa products through dimension 1024 are checked the same way.
The three other outputs are compared with the SIMD single product.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import stability as study
from stability_ties import checked_library

ALGORITHMS = {"quasilinear": 3, "cariow_cariowa": 2}


def check(document: dict[str, Any], corpus: Any, simd: study.Library, scalar: study.Library) -> dict[str, Any]:
    groups: dict[tuple[str, int], list[str]] = defaultdict(list)
    for row in document["samples"]:
        groups[row["domain"], row["dimension"]].append(row["key"])
    counts: dict[str, dict[str, int]] = {}
    mismatches: list[dict[str, Any]] = []
    for name, algorithm in ALGORITHMS.items():
        cases = comparisons = 0
        for (domain, n), keys in sorted(groups.items()):
            if name == "cariow_cariowa" and (domain != "real" or n > 1024):
                continue
            xs = np.stack([corpus[key + "_x"] for key in keys])
            ys = np.stack([corpus[key + "_y"] for key in keys])
            outputs = {
                "simd_batch": simd.multiply_batch(xs, ys, algorithm),
                "scalar_batch": scalar.multiply_batch(xs, ys, algorithm),
            }
            for index, key in enumerate(keys):
                reference = simd.multiply(xs[index], ys[index], algorithm).tobytes()
                candidates = {"scalar_single": scalar.multiply(xs[index], ys[index], algorithm),
                              "simd_batch": outputs["simd_batch"][index],
                              "scalar_batch": outputs["scalar_batch"][index]}
                for label, values in candidates.items():
                    comparisons += 1
                    if values.tobytes() != reference:
                        mismatches.append({"algorithm": name, "key": key, "output": label})
                cases += 1
        counts[name] = {"cases": cases, "comparisons": comparisons}
    return {"counts": counts, "mismatches": mismatches,
            "batch_sizes": {f"{domain}-{n}": len(keys) for (domain, n), keys in sorted(groups.items())}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accuracy", type=Path, required=True, help="stability.py JSON results")
    parser.add_argument("--simd", type=Path, required=True, help="measured shared library")
    parser.add_argument("--scalar", type=Path, required=True, help="measured scalar shared library")
    parser.add_argument("--output", type=Path, required=True, help="JSON file to write")
    args = parser.parse_args()
    document = json.loads(args.accuracy.read_text())
    if not document.get("completed"):
        raise SystemExit("the accuracy study did not finish")
    simd = checked_library(args.simd, document["modified_library"])
    if document.get("scalar_modified_library"):
        scalar = checked_library(args.scalar, document["scalar_modified_library"])
    else:
        scalar = study.Library(args.scalar)
    with np.load(args.accuracy.with_name(document["corpus"]["path"])) as corpus:
        result = check(document, corpus, simd, scalar)
    output = {"schema": "fastcd-stability-agreement-1", "completed": True,
              "command": [sys.executable, *sys.argv],
              "accuracy_sha256": hashlib.sha256(args.accuracy.read_bytes()).hexdigest(),
              "simd_library_sha256": hashlib.sha256(args.simd.read_bytes()).hexdigest(),
              "scalar_library_sha256": hashlib.sha256(args.scalar.read_bytes()).hexdigest(), **result}
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    if result["mismatches"]:
        raise SystemExit(f"{len(result['mismatches'])} outputs differ; see {args.output}")


if __name__ == "__main__":
    main()
