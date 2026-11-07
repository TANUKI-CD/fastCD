"""Measure and draw the accuracy and speed charts of the measured build.

Matplotlib is needed only for this experiment tool. Timings come from the
native C benchmark runners; plotting and corpus input/output are not timed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import random
import shlex
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from stability_timing import corpus, invoke, library_file, quantile, real_task

ALGORITHMS = ("quasilinear", "direct")
LABELS = {"quasilinear": "Quasilinear", "direct": "Direct", "cariow_cariowa": "Cariow–Cariowa"}
COLORS = {"quasilinear": "#0072B2", "direct": "#D55E00", "cariow_cariowa": "#009E73"}
MARKERS = {"quasilinear": "o", "direct": "s", "cariow_cariowa": "^"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path, base: Path) -> str:
    """Path of a recorded file relative to the chart directory, when possible."""
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path)


def accuracy_series(document: dict[str, Any],
                    algorithms_by_domain: dict[str, list[str]] | None = None) -> list[dict[str, Any]]:
    """Pool paired dense samples before taking algorithm-specific medians."""
    if not document["completed"]:
        raise ValueError("accuracy study did not complete")
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    selected = algorithms_by_domain or {domain: list(ALGORITHMS) for domain in ["real", "complex"]}
    for row in document["samples"]:
        if row["family"] not in {"gaussian", "uniform"}:
            continue
        names = ["modified_quasilinear" if a == "quasilinear" else a
                 for a in selected[row["domain"]]]
        if any(name not in row["algorithms"] for name in names):
            continue
        metrics = [row["algorithms"][a] for a in names]
        if any(m["zero_product"] or m["relative"] is None for m in metrics):
            continue
        groups.setdefault((row["domain"], row["dimension"]), []).append(row)
    result = []
    for (domain, n), rows in sorted(groups.items()):
        for algorithm in selected[domain]:
            source = "modified_quasilinear" if algorithm == "quasilinear" else algorithm
            values = [r["algorithms"][source]["relative"] for r in rows]
            if any(not math.isfinite(v) or v < 0 for v in values):
                raise ValueError("nonfinite or negative relative error in plotted samples")
            result.append({
                "metric": "relative_error", "domain": domain, "dimension": n,
                "mode": None, "algorithm": algorithm, "samples": len(values),
                "value": statistics.median(values), "units": "dimensionless",
                "sample_keys": [r["key"] for r in rows],
                "families": {family: sum(r["family"] == family for r in rows)
                             for family in ["gaussian", "uniform"]},
            })
    return result


def timing_summary(values: list[float]) -> dict[str, Any]:
    """Use the median of measured block throughputs, with a bootstrap CI."""
    if not values or any(not math.isfinite(t) or t <= 0 for t in values):
        raise ValueError("timing blocks must have finite positive ns/product")
    throughputs = [1e9 / t for t in values]
    rng = random.Random(20261105)
    resampled = [statistics.median(rng.choices(throughputs, k=len(throughputs)))
                 for _ in range(4000)]
    return {"value": statistics.median(throughputs), "units": "products/second",
            "samples": len(values), "median_ns_per_product": statistics.median(values),
            "throughput_95ci": [quantile(resampled, 0.025), quantile(resampled, 0.975)],
            "block_ns_per_product": values}


def checked_build(directory: Path, source_root: Path) -> dict[str, Any]:
    """Reject a build whose recorded source files differ from source_root."""
    record = (directory / "fastcd-build-Release.txt").read_text()
    for line in record.split("source_sha256:\n", 1)[1].splitlines():
        if "  " not in line:
            continue
        digest, relative = line.split("  ", 1)
        if sha256(source_root / relative) != digest:
            raise ValueError(f"source changed since native build configuration: {relative}")
    return {"directory": str(directory.resolve()), "build_record": record,
            "files_sha256": {
                p.name: sha256(p) for p in directory.iterdir()
                if p.is_file() and not p.is_symlink() and
                (p.name in {"fastcd_benchmark", "fastcd_domains_benchmark"}
                 or p.name.startswith("libfastcd.so."))
            }}


def complex_observation(directory: Path, path: Path, n: int, trials: int,
                        minimum_ms: float) -> tuple[dict[tuple[str, str], float], dict[str, Any]]:
    """Balance all four algorithm/mode task positions within each native run."""
    raw = invoke([
        str(directory / "fastcd_domains_benchmark"), "--corpus", str(path),
        "--dimension", str(n), "--trials", str(trials), "--mode", "both",
        "--char", "0", "--deg", "2", "--blocks", "4", "--warmup", "2",
        "--min-time-ms", str(minimum_ms),
    ])
    groups: dict[tuple[str, str], list[float]] = {}
    for observation in raw["observations"]:
        key = observation["mode"], observation["algorithm"]
        groups.setdefault(key, []).append(observation["elapsed_ns"] / observation["operations"])
    return {key: statistics.median(values) for key, values in groups.items()}, raw


def measure(args: argparse.Namespace, accuracy: dict[str, Any] | None) -> dict[str, Any]:
    """Run one complete fresh campaign with matched dimensions and inputs."""
    build = checked_build(args.build_dir, args.source)
    library = library_file(args.build_dir)
    expected = accuracy["modified_library"]["sha256"] if accuracy else args.expected_library_sha256
    if not expected:
        raise ValueError("measure-only requires the expected final native library SHA256")
    if sha256(library) != expected:
        raise ValueError("the timed library is not the library of the accuracy study")
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {args.cpu})
    corpus_directory = args.raw / "plot-corpora"
    corpus_directory.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema": "fastcd-stability-chart-timings-1", "completed": False,
        "command": sys.argv, "python": sys.version, "platform": platform.platform(),
        "cpu": subprocess.check_output(["lscpu", "--json"], text=True),
        "affinity": sorted(os.sched_getaffinity(0)), "build": build, "library": library.name,
        "blocks": args.blocks, "minimum_ms": args.minimum_ms, "trials": args.trials,
        "expected_library_sha256": expected,
        "algorithms": {"real": args.real_algorithms, "complex": args.complex_algorithms},
        "complex_inner_blocks": 4,
        "protocol": "native real and complex benchmark runners; the real task order rotates in each paired block; the complex runner rotates its four tasks and makes four inner observations per outer block",
        "corpora": [], "raw": [], "summary": [],
    }
    destination = args.raw / "plot-timings.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    try:
        for domain, maximum in [("real", args.real_max), ("complex", args.complex_max)]:
            for level in range(1, maximum.bit_length()):
                n = 1 << level
                seed = 2026110500 + n + (100000 if domain == "complex" else 0)
                path = corpus_directory / f"{domain}-{n}.bin"
                result["corpora"].append({"domain": domain, "dimension": n, "seed": seed,
                                          "trials": args.trials, "file": path.name,
                                          "sha256": corpus(path, n, args.trials, domain, seed)})
                groups: dict[tuple[str, str], list[float]] = {}
                for block in range(args.blocks):
                    if domain == "complex":
                        values, raw = complex_observation(args.build_dir, path, n, args.trials, args.minimum_ms)
                        result["raw"].append({"domain": domain, "dimension": n,
                                              "block": block, "native": raw})
                        for key, value in values.items():
                            if key[1] in args.complex_algorithms:
                                groups.setdefault(key, []).append(value)
                    else:
                        tasks = [(m, a) for m in ["single", "batch"] for a in args.real_algorithms]
                        shift = block % len(tasks)
                        for mode, algorithm in tasks[shift:] + tasks[:shift]:
                            value, raw = real_task(args.build_dir, path, n, args.trials,
                                                   mode, algorithm, args.minimum_ms)
                            result["raw"].append({"domain": domain, "dimension": n,
                                                  "block": block, "native": raw})
                            groups.setdefault((mode, algorithm), []).append(value)
                for (mode, algorithm), values in sorted(groups.items()):
                    result["summary"].append({"metric": "speed", "domain": domain,
                                              "dimension": n, "mode": mode,
                                              "algorithm": algorithm, **timing_summary(values)})
                destination.write_text(json.dumps(result, indent=2) + "\n")
                print(f"measured {domain} N={n}", flush=True)
        result["completed"] = True
    except Exception as error:
        result["failure"] = {"type": type(error).__name__, "message": str(error)}
        if isinstance(error, subprocess.CalledProcessError):
            result["failure"].update(stdout=error.stdout, stderr=error.stderr)
        raise
    finally:
        destination.write_text(json.dumps(result, indent=2) + "\n")
    return result


def make_figure(rows: list[dict[str, Any]], domain: str, metric: str,
                mode: str | None = None, algorithms: tuple[str, ...] | list[str] = ALGORITHMS) -> Any:
    """Create one axes with requested measured curves and no annotations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    figure, axes = plt.subplots(figsize=(7.4, 4.6), layout="constrained")
    selected = [r for r in rows if r["domain"] == domain and r["metric"] == metric
                and r["mode"] == mode]
    dimensions = sorted({r["dimension"] for r in selected})
    if not dimensions:
        raise ValueError("no measured points for requested figure")
    for algorithm in algorithms:
        series = sorted((r for r in selected if r["algorithm"] == algorithm),
                        key=lambda r: r["dimension"])
        if [r["dimension"] for r in series] != dimensions:
            raise ValueError("selected algorithm dimensions do not match")
        axes.plot(dimensions, [r["value"] for r in series], label=LABELS[algorithm],
                  color=COLORS[algorithm], marker=MARKERS[algorithm], linewidth=1.8,
                  markersize=4.8)
    axes.set_xscale("log", base=2)
    axes.set_xticks(dimensions)
    axes.xaxis.set_major_formatter(ScalarFormatter())
    if all(r["value"] > 0 for r in selected):
        axes.set_yscale("log")
    axes.set_xlabel("Dimension N")
    if metric == "relative_error":
        axes.set_title(f"{domain.capitalize()} coefficients: relative forward error")
        axes.set_ylabel("Median relative forward error")
    else:
        axes.set_title(f"{domain.capitalize()} coefficients: {mode}-product speed")
        axes.set_ylabel("Median throughput (products/s)")
    axes.grid(True, which="major", color="#dddddd", linewidth=0.6)
    axes.legend(frameon=False)
    return figure


def render(args: argparse.Namespace, accuracy: dict[str, Any], timings: dict[str, Any]) -> None:
    if not timings["completed"]:
        raise ValueError("timing campaign did not complete; refusing partial plots")
    measured_library = timings["build"]["files_sha256"][timings["library"]]
    if measured_library != accuracy["modified_library"]["sha256"]:
        raise ValueError("accuracy and timing datasets use different native libraries")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["svg.hashsalt"] = "fastcd-stability"
    selected = {"real": args.real_algorithms, "complex": args.complex_algorithms}
    if timings["algorithms"] != selected:
        raise ValueError("selected algorithms differ from the measured timing campaign")
    rows = accuracy_series(accuracy, selected) + timings["summary"]
    timing_file = args.raw / "plot-timings.json"
    data = {"schema": "fastcd-stability-plot-data-2", "accuracy_source": relative(args.accuracy, args.output),
            "accuracy_sha256": sha256(args.accuracy), "timing_source": relative(timing_file, args.output),
            "timing_sha256": sha256(timing_file),
            "algorithm_mapping": {"quasilinear": "modified_quasilinear (balanced accumulation)", "direct": "direct",
                                  "cariow_cariowa": "cariow_cariowa"},
            "selected_algorithms": selected,
            "accuracy_filter": "pool Gaussian and uniform paired samples at each domain/N with all selected algorithms available; exclude zero reference products; retain zero error values; median of individual relative errors",
            "speed_statistic": "median of 1e9/(outer-block ns/product), giving products/second; complex ns/product per outer block is median of four native inner observations",
            "confidence_intervals": "speed medians: 4000 bootstrap resamples of the outer blocks, seed 20261105; the intervals are in the JSON and CSV files, not drawn on the charts",
            "environment": {"python": sys.version, "matplotlib": matplotlib.__version__},
            "tool_sha256": sha256(Path(__file__)), "rows": rows, "figures": []}
    for domain in ["real", "complex"]:
        specs = [("relative_error", None, f"relative-error-{domain}"),
                 ("speed", "single", f"speed-{domain}-single"),
                 ("speed", "batch", f"speed-{domain}-batch")]
        for metric, mode, name in specs:
            figure = make_figure(rows, domain, metric, mode, selected[domain])
            for extension in ["svg", "pdf", "png"]:
                metadata = {"Creator": "fastCD stability plots"} if extension != "png" else {"Software": "fastCD stability plots"}
                if extension == "pdf":
                    metadata.update(CreationDate=None, ModDate=None)
                if extension == "svg":
                    metadata.update(Date=None)
                figure.savefig(args.output / f"{name}.{extension}", dpi=200, metadata=metadata)
            axes = figure.axes[0]
            data["figures"].append({"name": name, "domain": domain, "metric": metric,
                                    "mode": mode, "xscale": axes.get_xscale(),
                                    "yscale": axes.get_yscale(), "series": selected[domain],
                                    "files": {ext: sha256(args.output / f"{name}.{ext}")
                                              for ext in ["svg", "pdf", "png"]}})
            plt.close(figure)
    (args.output / "plot-data.json").write_text(json.dumps(data, indent=2) + "\n")
    output = io.StringIO()
    fields = ["metric", "domain", "dimension", "mode", "algorithm", "samples", "value", "units",
              "throughput_95ci_low", "throughput_95ci_high"]
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        interval = row.get("throughput_95ci", [None, None])
        writer.writerow({**row, "throughput_95ci_low": interval[0], "throughput_95ci_high": interval[1]})
    titles = {"relative-error-real": "Relative error, real coefficients",
              "relative-error-complex": "Relative error, complex coefficients",
              "speed-real-single": "Speed of single products, real coefficients",
              "speed-real-batch": "Speed of batched products, real coefficients",
              "speed-complex-single": "Speed of single products, complex coefficients",
              "speed-complex-batch": "Speed of batched products, complex coefficients"}
    figure_index = "\n".join([
        "| Chart | PNG | SVG | PDF |", "|:---|:---|:---|:---|",
        *(f"| {titles.get(figure['name'], figure['name'])} | "
          f"[PNG]({figure['name']}.png) | [SVG]({figure['name']}.svg) | "
          f"[PDF]({figure['name']}.pdf) |" for figure in data["figures"]),
    ])
    (args.output / "plot-data.csv").write_text(output.getvalue())
    def repository_relative(token: str) -> str:
        path = Path(token)
        if path.is_absolute() and path.is_relative_to(Path.cwd()):
            return str(path.relative_to(Path.cwd()))
        return token
    command = shlex.join(["python3", *(repository_relative(token) for token in sys.argv)])

    def names(algorithms: list[str]) -> str:
        words = [LABELS[a] if a == "cariow_cariowa" else LABELS[a].lower() for a in algorithms]
        return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]
    real, complex_ = names(selected["real"]), names(selected["complex"])
    (args.output / "PLOTS.md").write_text(
        "# Accuracy and speed charts\n\n"
        f"The six charts compare {real} multiplication for real coefficients, and "
        f"{complex_} multiplication for complex coefficients, at the same dimensions. "
        "Quasilinear multiplication uses balanced accumulation.\n\n"
        f"{figure_index}\n\n"
        "## Accuracy\n\n"
        "Each point is the median relative error `E_rel` (defined in docs/stability.md) "
        "of the Gaussian and uniform cases of the accuracy study at one dimension. "
        "Products whose exact value is zero are left out; outputs "
        "with zero error are kept. Complex errors use the Euclidean norm of the "
        "coefficients. The sample keys and counts behind every point are in "
        "plot-data.json.\n\n"
        "## Speed\n\n"
        f"Each point is the median throughput, in products per second, over "
        f"{timings['blocks']} timing blocks run with the native benchmark runners on "
        f"logical CPU {', '.join(map(str, timings['affinity']))}. In every block, each "
        f"algorithm is timed on the same {timings['trials']} stored operand pairs per "
        f"dimension, and every timing lasts at least {timings['minimum_ms']:g} ms. For real "
        "coefficients the order of the algorithms changes from block to block. For complex "
        "coefficients, one runner process per block times single and batched products of "
        "both algorithms in rotating order, and a block's value is the median of four "
        "timings. The runners check every timed product. Bootstrap 95% intervals are in "
        "plot-data.json and plot-data.csv.\n\n"
        "## Axes\n\n"
        "N is on a base-two logarithmic axis. The vertical axis is logarithmic when "
        "every plotted median is positive and linear otherwise. Medians do not "
        "describe worst-case errors, and timings on this host do not predict timings "
        "on other machines.\n\n"
        "## Reproduction\n\n"
        "From the repository root of the machine that ran the study:\n\n"
        "```sh\n" + command + "\n```\n\n"
        "Without --measure, the same command redraws the charts from the recorded "
        "timings.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accuracy", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="directory for the charts and plot data")
    parser.add_argument("--raw", type=Path, help="directory for the timing results and corpora (default: --output)")
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1],
                        help="source tree of the build, checked against its build record")
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--measure-only", action="store_true",
                        help="acquire timings while the independent accuracy study runs")
    parser.add_argument("--expected-library-sha256",
                        help="required by --measure-only; later rendering rechecks accuracy library identity")
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--minimum-ms", type=float, default=20)
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--real-max", type=int, default=1024)
    parser.add_argument("--complex-max", type=int, default=512)
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--real-algorithms", nargs="+", choices=list(LABELS), default=list(ALGORITHMS))
    parser.add_argument("--complex-algorithms", nargs="+", choices=list(ALGORITHMS), default=list(ALGORITHMS))
    args = parser.parse_args()
    if args.blocks < 4 or args.trials < 1 or not math.isfinite(args.minimum_ms) or args.minimum_ms <= 0:
        parser.error("require at least four blocks, positive trials and finite positive duration")
    for maximum in [args.real_max, args.complex_max]:
        if maximum < 2 or maximum & (maximum - 1):
            parser.error("maximum dimensions must be powers of two at least two")
    for algorithms in [args.real_algorithms, args.complex_algorithms]:
        if len(algorithms) != len(set(algorithms)):
            parser.error("algorithm lists must not contain duplicates")
    if "cariow_cariowa" in args.real_algorithms and args.real_max > 1024:
        parser.error("Cariow–Cariowa supports real dimensions only through 1024")
    args.output.mkdir(parents=True, exist_ok=True)
    args.raw = args.raw or args.output
    args.raw.mkdir(parents=True, exist_ok=True)
    if args.measure_only:
        measure(args, None)
        return
    accuracy = json.loads(args.accuracy.read_text())
    timings = (measure(args, accuracy) if args.measure else
               json.loads((args.raw / "plot-timings.json").read_text()))
    render(args, accuracy, timings)


if __name__ == "__main__":
    main()
