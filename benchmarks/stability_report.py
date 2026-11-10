#!/usr/bin/env python3
"""Render the accuracy tables and CSV summary from stability.py results.

S is quasilinear multiplication with sequential accumulation (the baseline
library), B is quasilinear multiplication with balanced accumulation (the
measured library) and D is direct multiplication. The same tables are written
to the archive (--tables) and, with --document, into the delimited accuracy
section of docs/stability.md.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
from typing import Any

import stability_archive as archive

ALGORITHMS = ("original_quasilinear", "modified_quasilinear", "direct")
FOOTER = (
    "The archived CSV summary also gives, for every family, the count, the "
    "median, mean, p95, p99 and maximum of eta, the relative and scalar errors, the "
    "paired improvement fractions and descriptive Wilson 95% intervals. Groups with "
    "fewer than 100 samples have an empirical p99 close to their maximum and large "
    "sampling uncertainty."
)


def fmt(value: Any) -> str:
    return "—" if value is None else f"{value:.3g}"


def median(values: list[float]) -> float:
    import numpy as np

    return float(np.median(values))


def med(rows: list[dict[str, Any]], algorithm: str, key: str) -> float:
    return median([r["algorithms"][algorithm][key] for r in rows])


def share(rows: list[dict[str, Any]], key: str) -> str:
    return f"{sum(r[key] for r in rows) / len(rows):.0%}"


def render(document: dict[str, Any]) -> str:
    """Return the accuracy tables, without section markers."""
    import numpy as np

    if not document.get("completed"):
        raise ValueError("accuracy study did not finish; refusing a partial table")
    lines = [
        f"{len(document['samples']):,} paired cases from the {document['profile']} profile. "
        "S is quasilinear multiplication with sequential accumulation, B is quasilinear "
        "multiplication with balanced accumulation, and D is direct multiplication; all "
        "three use the same stored operands. Dense rows pool the Gaussian and uniform "
        "families. The scalar error is the absolute error of coordinate 0, and "
        "\"B better / tied\" compares the stored `E_abs` of B with that of S. "
        "Empirical p99 values and maxima describe this corpus only.", "",
    ]
    for domain in ("real", "complex"):
        dense = [r for r in document["samples"] if r["domain"] == domain and r["family"] in ("gaussian", "uniform")]
        if not dense:
            continue
        lines.extend([f"**{domain.capitalize()} dense inputs**", "",
                      "| N | pairs | median eta S / B / D | p99 eta S / B / D | max eta S / B / D | median scalar error S / B / D | B better / tied |",
                      "|---:|---:|:---|:---|:---|:---|:---|"])
        for n in sorted({r["dimension"] for r in dense}):
            rows = [r for r in dense if r["dimension"] == n]
            values = []
            for key, operation in (("normalized", np.median), ("normalized", lambda a: np.percentile(a, 99)),
                                   ("normalized", max), ("scalar", np.median)):
                values.append(" / ".join(fmt(operation([r["algorithms"][a][key] for r in rows])) for a in ALGORITHMS))
            lines.append(f"| {n} | {len(rows)} | " + " | ".join(values) + f" | {share(rows, 'improved')} / {share(rows, 'tied')} |")
        lines.extend(["", f"**{domain.capitalize()} growth with dimension**, B on dense inputs", "",
                      "| N | median eta | eta / (sqrt(N) log2 N) | eta / N | median relative error | max relative error |",
                      "|---:|---:|---:|---:|---:|---:|"])
        for n in sorted({r["dimension"] for r in dense}):
            rows = [r for r in dense if r["dimension"] == n]
            eta = med(rows, "modified_quasilinear", "normalized")
            lines.append(f"| {n} | {fmt(eta)} | {fmt(eta / (math.sqrt(n) * math.log2(n)))} | {fmt(eta / n)} | "
                         f"{fmt(med(rows, 'modified_quasilinear', 'relative'))} | "
                         f"{fmt(max(r['algorithms']['modified_quasilinear']['relative'] for r in rows))} |")
        lines.append("")
        size = min(512, max(r["dimension"] for r in dense))
        selected = [r for r in document["samples"] if r["domain"] == domain and r["dimension"] == size]
        lines.extend([f"**{domain.capitalize()} families at N={size}**", "",
                      "| family | pairs | median eta S / B / D | max relative error B | B better / tied |",
                      "|:---|---:|:---|---:|:---|"])
        for family in dict.fromkeys(r["family"] for r in selected):
            rows = [r for r in selected if r["family"] == family]
            relative = [r["algorithms"]["modified_quasilinear"]["relative"] for r in rows
                        if r["algorithms"]["modified_quasilinear"]["relative"] is not None]
            values = " / ".join(fmt(med(rows, a, "normalized")) for a in ALGORITHMS)
            lines.append(f"| {family} | {len(rows)} | {values} | {fmt(max(relative) if relative else None)} | "
                         f"{share(rows, 'improved')} / {share(rows, 'tied')} |")
        lines.append("")
    lines.append(FOOTER)
    return "\n".join(lines) + "\n"


def summary_csv(document: dict[str, Any]) -> str:
    summaries = document["summaries"]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(summaries[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(summaries)
    return output.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="stability.py JSON results")
    parser.add_argument("--tables", type=Path, required=True, help="Markdown tables to write")
    parser.add_argument("--csv", type=Path, required=True, help="CSV summary to write")
    parser.add_argument("--document", type=Path, help="docs/stability.md, to update its accuracy section")
    args = parser.parse_args()
    document = json.loads(args.results.read_text())
    tables = render(document)
    args.tables.write_text(tables)
    args.csv.write_text(summary_csv(document))
    if args.document:
        text = args.document.read_text()
        args.document.write_text(archive.replace_section(text, archive.ACCURACY_SECTION, tables))


if __name__ == "__main__":
    main()
