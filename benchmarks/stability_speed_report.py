"""Render the sequential/balanced timing table from stability_speed.py results.

The same table is written to the archive (--tables) and, with --document,
into the delimited timing section of docs/stability.md.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any

import stability_archive as archive

FIELDS = (
    "domain", "backend", "mode", "dimension", "sequential_quasilinear_ns", "balanced_quasilinear_ns",
    "balanced_over_sequential", "balanced_over_sequential_95ci", "sequential_direct_ns",
    "balanced_direct_ns", "direct_control", "direct_control_95ci", "direct_over_balanced",
)


def render(data: dict[str, Any]) -> str:
    """Return the timing table, without section markers."""
    if not data["completed"]:
        raise ValueError("timing study did not finish; refusing a partial table")
    cpus = ", ".join(map(str, data["affinity"]))
    lines = [
        f"Paired timings on one host: {data['blocks']} blocks pinned to CPU {cpus}. "
        "Times are medians in ns per product, and brackets are paired bootstrap 95% "
        "intervals. The direct control is the ratio of direct timings between the two "
        "builds, whose direct arithmetic is identical, so it shows the noise between "
        "builds.", "",
        "| domain | kernels | mode | N | sequential ns | balanced ns | balanced / sequential [95% CI] | direct ns | direct / balanced | direct control |",
        "|:---|:---|:---|---:|---:|---:|:---|---:|---:|---:|",
    ]
    for row in data["summary"]:
        low, high = row["balanced_over_sequential_95ci"]
        lines.append(
            f"| {row['domain']} | {row['backend']} | {row['mode']} | {row['dimension']} | "
            f"{row['sequential_quasilinear_ns']:.4g} | {row['balanced_quasilinear_ns']:.4g} | "
            f"{row['balanced_over_sequential']:.4g} [{low:.4g}, {high:.4g}] | "
            f"{row['balanced_direct_ns']:.4g} | {row['direct_over_balanced']:.4g} | "
            f"{row['direct_control']:.4g} |"
        )
    return "\n".join(lines) + "\n"


def summary_csv(data: dict[str, Any]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(data["summary"])
    return output.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="stability_speed.py JSON results")
    parser.add_argument("--tables", type=Path, required=True, help="Markdown table to write")
    parser.add_argument("--csv", type=Path, required=True, help="CSV summary to write")
    parser.add_argument("--document", type=Path, help="docs/stability.md, to update its timing section")
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    table = render(data)
    args.tables.write_text(table)
    args.csv.write_text(summary_csv(data))
    if args.document:
        text = args.document.read_text()
        args.document.write_text(archive.replace_section(text, archive.SPEED_SECTION, table))


if __name__ == "__main__":
    main()
