"""Time quasilinear products for several balanced-accumulation block sizes.

Each build is the measured source configured with
-DFASTCD_TUNING_DEFINITIONS=FASTCD_SUM_BLOCK=B, except that the default block
size 8 needs no definition. The generated leaves always use blocks of 8, so
only the sums inside the recursion change. Paired blocks rotate the order of
the builds; ratios compare each block size with 8.
"""

from __future__ import annotations

import argparse
import csv
import io
import statistics
import sys
from pathlib import Path
from typing import Any

from stability_timing import (
    BOOTSTRAP,
    CORPUS_DISTRIBUTION,
    build_record,
    complex_task,
    corpus,
    paired,
    pin,
    real_task,
    write_json,
)

REFERENCE = 8


def parse_build(text: str) -> tuple[int, Path]:
    size, _, directory = text.partition("=")
    if not size.isdigit() or not directory:
        raise argparse.ArgumentTypeError("use SIZE=BUILD_DIRECTORY")
    return int(size), Path(directory).resolve()


def summarize(values: dict[int, list[float]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"median_ns": {str(size): statistics.median(v) for size, v in sorted(values.items())}}
    for size in sorted(values):
        if size != REFERENCE:
            comparison = paired(values[REFERENCE], values[size])
            summary[f"ratio_{size}_over_{REFERENCE}"] = comparison["ratio"]
            summary[f"ratio_{size}_over_{REFERENCE}_95ci"] = comparison["ratio_95ci"]
    return summary


def render(data: dict[str, Any]) -> str:
    """Markdown table of the block-size timings."""
    if not data["completed"]:
        raise ValueError("block-size timings did not finish; refusing a partial table")
    others = [size for size in data["block_sizes"] if size != REFERENCE]
    header = ["domain", "mode", "N", *(f"{size} ns" for size in data["block_sizes"]),
              *(f"{size} / {REFERENCE} [95% CI]" for size in others)]
    lines = [f"Quasilinear products, {data['blocks']} paired blocks pinned to CPU "
             f"{', '.join(map(str, data['affinity']))}; medians in ns per product.", "",
             "| " + " | ".join(header) + " |", "|:---|:---|" + "---:|" * (len(header) - 2)]
    for row in data["summary"]:
        cells = [row["domain"], row["mode"], str(row["dimension"]),
                 *(f"{row['median_ns'][str(size)]:.4g}" for size in data["block_sizes"])]
        for size in others:
            low, high = row[f"ratio_{size}_over_{REFERENCE}_95ci"]
            cells.append(f"{row[f'ratio_{size}_over_{REFERENCE}']:.4g} [{low:.4g}, {high:.4g}]")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def summary_csv(data: dict[str, Any]) -> str:
    rows = []
    for row in data["summary"]:
        flat = {key: value for key, value in row.items() if key != "median_ns"}
        flat.update({f"median_ns_{size}": value for size, value in row["median_ns"].items()})
        rows.append(flat)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=parse_build, action="append", required=True,
                        help="SIZE=BUILD_DIRECTORY; repeat for each block size, including 8")
    parser.add_argument("--output", type=Path, required=True, help="JSON results file")
    parser.add_argument("--corpora", type=Path, required=True, help="directory for the operand corpora")
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--minimum-ms", type=float, default=10)
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--real-dimensions", type=int, nargs="+", default=[64, 256, 1024])
    parser.add_argument("--complex-dimensions", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--tables", type=Path, help="Markdown table to write when the timings finish")
    parser.add_argument("--csv", type=Path, help="CSV summary to write when the timings finish")
    args = parser.parse_args()
    builds = dict(args.build)
    if len(builds) != len(args.build) or REFERENCE not in builds:
        parser.error(f"give each block size once, including {REFERENCE}")
    if args.blocks < 2 or args.minimum_ms <= 0 or args.trials < 1:
        parser.error("use at least two blocks, a positive minimum time and at least one trial")
    affinity = pin(args.cpu)
    sizes = sorted(builds)
    result: dict[str, Any] = {
        "schema": "fastcd-stability-block-size-1", "completed": False,
        "command": [sys.executable, *sys.argv], "affinity": affinity,
        "blocks": args.blocks, "minimum_ms": args.minimum_ms, "trials": args.trials, "warmup_passes": 2,
        "block_sizes": sizes, "reference_block_size": REFERENCE,
        "build_order": "the list of block sizes rotates by one position in each paired block",
        "bootstrap": BOOTSTRAP, "corpus_distribution": CORPUS_DISTRIBUTION,
        "builds": {str(size): build_record(directory) for size, directory in sorted(builds.items())},
        "corpora": [], "raw": [], "summary": [],
    }
    write_json(args.output, result)
    for domain, dimensions in (("real", args.real_dimensions), ("complex", args.complex_dimensions)):
        for n in dimensions:
            seed = 2026110500 + n + (100000 if domain == "complex" else 0)
            path = args.corpora / f"{domain}-{n}.bin"
            result["corpora"].append({"domain": domain, "dimension": n, "trials": args.trials, "seed": seed,
                                      "file": path.name, "sha256": corpus(path, n, args.trials, domain, seed)})
            values: dict[tuple[str, int], list[float]] = {}
            for block in range(args.blocks):
                shift = block % len(sizes)
                for size in sizes[shift:] + sizes[:shift]:
                    if domain == "real":
                        for mode in ("single", "batch"):
                            timing, raw = real_task(builds[size], path, n, args.trials, mode, "quasilinear",
                                                    args.minimum_ms)
                            result["raw"].append({"domain": domain, "dimension": n, "mode": mode, "block_size": size,
                                                  "block": block, "time": timing, "native": raw})
                            values.setdefault((mode, size), []).append(timing)
                    else:
                        times, raw = complex_task(builds[size], path, n, args.trials, args.minimum_ms)
                        for (mode, algorithm), timing in times.items():
                            if algorithm == "quasilinear":
                                result["raw"].append({"domain": domain, "dimension": n, "mode": mode,
                                                      "block_size": size, "block": block, "time": timing,
                                                      "native": raw})
                                values.setdefault((mode, size), []).append(timing)
            for mode in sorted({mode for mode, _ in values}):
                result["summary"].append({"domain": domain, "dimension": n, "mode": mode,
                                          **summarize({size: values[mode, size] for size in sizes})})
            write_json(args.output, result)
            print(f"timed {domain} N={n}", flush=True)
    result["completed"] = True
    write_json(args.output, result)
    if args.tables:
        args.tables.write_text(render(result))
    if args.csv:
        args.csv.write_text(summary_csv(result))


if __name__ == "__main__":
    main()
