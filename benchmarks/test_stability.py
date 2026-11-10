"""Independent oracle and data-integrity checks for the optional stability study."""

from fractions import Fraction

import numpy as np
import pytest

pytest.importorskip("gmpy2")
import stability as study  # noqa: E402


def test_exact_signed_xor_matches_defining_rational_recursion():
    validation = study.validate_oracle()
    assert validation["basis_cases_against_rational_doubling"] == 341


def test_binary64_conversion_uses_exact_dyadics():
    values = np.array([0.1, -3.0, 2.0 ** -40, 0.0])
    integers, exponent = study.encode(values)
    assert [Fraction(a, 1 << exponent) for a in integers] == [Fraction(float(a)) for a in values]


def test_complex_oracle_is_bilinear_not_hermitian():
    x = np.array([1 + 2j, 3 - 4j])
    y = np.array([5 - 6j, -7 + 8j])
    reference = study.as_mp(study.exact_product(x, y))
    assert [complex(v) for v in reference] == [x[0] * y[0] - x[1] * y[1], x[0] * y[1] + x[1] * y[0]]


def test_zero_divisor_and_isotropic_have_different_meaning():
    x, y = study.zero_divisor()
    real, imag, _ = study.exact_product(x, y)
    assert not any(real) and not any(imag)
    x, y = study.make_inputs("complex", 16, "isotropic", 73)
    assert np.sum(x * x) == 0 and np.linalg.norm(x) > 0
    real, imag, _ = study.exact_product(x, y)
    assert any(real) or any(imag)


def test_metrics_mark_zero_products_and_exact_ties():
    x, y = study.zero_divisor()
    metric = study.metric(np.zeros(16), study.exact_product(x, y), x, y, 256)
    assert metric["zero_product"]
    assert metric["relative"] is None and metric["chi"] is None
    assert metric["absolute"] == metric["normalized"] == 0
    assert study.ratio(0, 0) == {"value": None, "zero_case": "both_zero"}
    assert study.ratio(1, 0) == {"value": None, "zero_case": "positive_over_zero"}


@pytest.mark.parametrize("domain,family", [("real", "gaussian"), ("real", "scalar_cancellation"), ("complex", "uniform"), ("complex", "correlated_minus")])
def test_seed_reproduces_stored_operands(domain, family):
    first = study.make_inputs(domain, 32, family, 20261105)
    second = study.make_inputs(domain, 32, family, 20261105)
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))


def test_operands_do_not_depend_on_blas(monkeypatch):
    expected = {domain: study.make_inputs(domain, 64, "gaussian", 20261105) for domain in ("real", "complex")}

    def unavailable(*args, **kwargs):
        raise AssertionError("operand generation must not call BLAS")

    monkeypatch.setattr(np.linalg, "norm", unavailable)
    monkeypatch.setattr(np, "dot", unavailable)
    for domain, (x, y) in expected.items():
        again = study.make_inputs(domain, 64, "gaussian", 20261105)
        assert again[0].tobytes() == x.tobytes() and again[1].tobytes() == y.tobytes()
        assert abs(study.euclidean_norm(x) - 1.0) <= 2.0 ** -52
