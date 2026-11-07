"""Scientific data filtering, throughput units and the requested chart contract."""

from __future__ import annotations

import hashlib

import pytest

pytest.importorskip("matplotlib")
import stability_plots as plots  # noqa: E402


def sample(key, family, value, zero=False):
    return {"key": key, "family": family, "domain": "real", "dimension": 8,
            "algorithms": {name: {"relative": value, "zero_product": zero}
                           for name in ["modified_quasilinear", "direct"]}}


def test_accuracy_pools_individual_dense_samples_and_retains_exact_results():
    document = {"completed": True, "samples": [
        sample("g1", "gaussian", 0), sample("g2", "gaussian", 2),
        sample("u1", "uniform", 10), sample("sparse", "sparse_basis", 100),
        sample("zero", "gaussian", None, True),
    ]}
    rows = plots.accuracy_series(document)
    assert len(rows) == 2
    assert all(row["value"] == 2 and row["samples"] == 3 for row in rows)
    assert all(row["sample_keys"] == ["g1", "g2", "u1"] for row in rows)


def test_speed_is_median_reciprocal_time_in_products_per_second():
    summary = plots.timing_summary([1, 2, 8, 16])
    assert summary["value"] == 312_500_000
    assert summary["median_ns_per_product"] == 5
    assert summary["units"] == "products/second"
    assert summary["throughput_95ci"][0] <= summary["value"] <= summary["throughput_95ci"][1]


def test_optional_real_algorithm_uses_its_own_metrics_and_common_dimensions():
    first = sample("all", "gaussian", 2)
    first["algorithms"]["cariow_cariowa"] = {"relative": 9, "zero_product": False}
    second = sample("unsupported", "uniform", 4)
    second["dimension"] = 2048
    rows = plots.accuracy_series({"completed": True, "samples": [first, second]},
                                 {"real": ["quasilinear", "direct", "cariow_cariowa"],
                                  "complex": ["quasilinear", "direct"]})
    assert [row["dimension"] for row in rows] == [8, 8, 8]
    assert [row["value"] for row in rows] == [2, 2, 9]


@pytest.mark.parametrize("domain", ["real", "complex"])
@pytest.mark.parametrize("metric,mode", [("relative_error", None), ("speed", "single"), ("speed", "batch")])
def test_each_chart_has_two_measured_series_and_no_notes(domain, metric, mode):
    import matplotlib.pyplot as plt

    rows = [{"domain": domain, "metric": metric, "mode": mode,
             "algorithm": algorithm, "dimension": n, "value": n * (1 if algorithm == "direct" else 2)}
            for algorithm in plots.ALGORITHMS for n in [2, 4, 8]]
    figure = plots.make_figure(rows, domain, metric, mode)
    axes = figure.axes[0]
    assert len(figure.axes) == 1 and len(axes.lines) == 2
    assert [line.get_label() for line in axes.lines] == ["Quasilinear", "Direct"]
    assert not figure.texts and not axes.texts
    assert axes.get_title() and axes.get_xlabel() == "Dimension N" and axes.get_ylabel()
    assert axes.get_xscale() == axes.get_yscale() == "log"
    assert axes.xaxis._scale.base == 2
    assert [text.get_text() for text in axes.get_legend().get_texts()] == ["Quasilinear", "Direct"]
    for line, algorithm in zip(axes.lines, plots.ALGORITHMS, strict=True):
        selected = [row for row in rows if row["algorithm"] == algorithm]
        assert list(line.get_xdata()) == [row["dimension"] for row in selected]
        assert list(line.get_ydata()) == [row["value"] for row in selected]
    plt.close(figure)


def test_build_provenance_rejects_a_changed_kernel(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    kernel = source / "kernel.c"
    kernel.write_text("original kernel")
    build = tmp_path / "build"
    build.mkdir()
    digest = hashlib.sha256(kernel.read_bytes()).hexdigest()
    (build / "fastcd-build-Release.txt").write_text(f"source_sha256:\n{digest}  kernel.c\n")
    plots.checked_build(build, source)
    kernel.write_text("changed kernel")
    with pytest.raises(ValueError, match="source changed"):
        plots.checked_build(build, source)


@pytest.mark.parametrize("metric,mode", [("relative_error", None), ("speed", "single"), ("speed", "batch")])
def test_real_three_algorithm_figure_keeps_only_title_axes_and_key(metric, mode):
    import matplotlib.pyplot as plt

    algorithms = ["quasilinear", "direct", "cariow_cariowa"]
    rows = [{"domain": "real", "metric": metric, "mode": mode,
             "algorithm": algorithm, "dimension": n, "value": n * (index + 1)}
            for index, algorithm in enumerate(algorithms) for n in [2, 4, 8]]
    figure = plots.make_figure(rows, "real", metric, mode, algorithms)
    axes = figure.axes[0]
    labels = ["Quasilinear", "Direct", "Cariow–Cariowa"]
    assert len(figure.axes) == 1 and len(axes.lines) == 3
    assert [line.get_label() for line in axes.lines] == labels
    assert [text.get_text() for text in axes.get_legend().get_texts()] == labels
    assert axes.get_title() and axes.get_xlabel() == "Dimension N" and axes.get_ylabel()
    assert axes.get_xscale() == axes.get_yscale() == "log"
    assert not figure.texts and not axes.texts
    for line in axes.lines:
        assert list(line.get_xdata()) == [2, 4, 8]
    plt.close(figure)
