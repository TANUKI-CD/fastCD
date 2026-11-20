"""Reference and data checks for the accuracy study of the fused sums."""

from __future__ import annotations

import os
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import stability_sums as study


def test_sum_reference_matches_exact_integer_arithmetic():
    validation = study.validate_oracle()
    assert validation["cases"] == 6 * len(study.SUM_FAMILIES)
    assert validation["max_relative_remainder"] < 2.0**-100


def test_signs_and_integer_contraction_follow_the_doubling_formula():
    signs = study.sign_table(4)
    # Quaternions: e1 e2 = e3, e2 e1 = -e3, e1 e1 = -1
    assert signs[1, 2] == 1 and signs[2, 1] == -1 and signs[1, 1] == -1
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    y = np.array([[5.0, 6.0, 7.0, 8.0]])
    assert study.integer_sum(x, y) == [-60, 12, 30, 24]


def test_binary64_conversion_is_exact():
    values = np.array([[0.1, -3.0], [2.0**-40, 0.0]])
    integers, exponent = study.to_integers(values)
    for m, v in zip(integers.ravel(), values.ravel(), strict=True):
        assert Fraction(m) * Fraction(2) ** exponent == Fraction(float(v))


def test_integer_covariance_of_a_small_sample():
    x = np.array([[1.0, 0.0], [3.0, 2.0], [5.0, 4.0]])
    exact = study.integer_covariance(x, x, 1)
    # cov(x, x) is the total variance: its imaginary part vanishes
    assert exact == [Fraction(8), Fraction(0)]


def test_families_have_their_stated_structure():
    rng = np.random.default_rng(3)
    x, y = study.make_inputs("cancelling", 8, 64, rng)
    assert all(v == 0 for v in study.integer_sum(x, y))
    x, y = study.make_inputs("conjugate", 8, 16, rng)
    assert all(v == 0 for v in study.integer_sum(x, y)[1:])
    x, y = study.make_inputs("sparse", 16, 32, rng)
    assert (np.count_nonzero(x, axis=1) == 1).all()


def test_plan_and_summary_cover_every_method():
    cases = study.plan("quick")
    assert {c[0] for c in cases} == {"sum", "covariance"}
    assert max(c[2] for c in cases) == 64
    assert max(c[2] for c in study.plan("full")) == 1024
    rows = [
        {"kind": "sum", "family": "gaussian", "dimension": 8, "count": 1024,
         "layouts_agree": True, "eta": dict.fromkeys(study.SUM_METHODS, 0.5)},
        {"kind": "covariance", "family": "offset", "dimension": 8, "count": 1024,
         "eta": dict.fromkeys(study.COVARIANCE_METHODS, 2.0)},
    ]
    summary = study.summarize(rows)
    assert summary["sum_by_dimension"][0]["max"]["fused"] == 0.5
    assert summary["covariance"][0]["median"]["covariance"] == 2.0
    assert summary["layouts_agree"]


def test_library_results_are_within_a_few_units():
    path = os.environ.get("FASTCD_SHARED_LIBRARY")
    if path is None:
        pytest.skip("FASTCD_SHARED_LIBRARY does not name a shared build")
    library = study.Library(Path(path))
    row = study.sum_case(library, "gaussian", 16, 200, [1, 2, 3])
    assert row["layouts_agree"]
    assert all(0 <= v < 8 for v in row["eta"].values())
    row = study.covariance_case(library, "offset", 8, 128, [4, 5, 6])
    assert 0 <= row["eta"]["covariance"] < 8
