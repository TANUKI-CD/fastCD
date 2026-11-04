"""Regression coverage for domain API validation and field presentations."""

from __future__ import annotations

import os
import pickle
import subprocess
import sys

import numpy as np
import pyfastcd as fcd
import pytest


@pytest.fixture(params=[(0, 1), (0, 2), (7, 1), (7, 2)])
def algebra(request):
    p, d = request.param
    return fcd.Algebra(dimension=4, char=p, deg=d)


@pytest.mark.parametrize("bad", ["a", None, 5])
def test_named_structural_methods_reject_nonrecords(algebra, bad):
    x = algebra.one()
    X = algebra.array([x.coefficients, x.coefficients])
    for name in ("inner", "imag_cross", "commutator", "anticommutator"):
        with pytest.raises(TypeError):
            getattr(x, name)(bad)
        with pytest.raises(TypeError):
            getattr(X, name)(bad)
    for y, z in ((bad, x), (x, bad), (bad, bad)):
        with pytest.raises(TypeError):
            x.associator(y, z)
        with pytest.raises(TypeError):
            X.associator(y, z)
    with pytest.raises(TypeError):
        x.associator(X, x)
    with pytest.raises(TypeError):
        X.inner(x.coefficients)


def test_associator_arity_matches_real_methods(algebra):
    x = algebra.one()
    for operands in ((), (x,), (x, x, x)):
        with pytest.raises(TypeError):
            x.associator(*operands)


def test_algebra_structural_methods_require_complete_records(algebra):
    x = algebra.one()
    for operation in (
        lambda: algebra.left_basis_multiply(1, 5),
        lambda: algebra.right_basis_multiply(5, 1),
        lambda: algebra.left_operator_commutator(x, 5, x),
        lambda: algebra.right_operator_commutator(x, x, 5),
    ):
        with pytest.raises((TypeError, ValueError)):
            operation()
    assert algebra.left_basis_multiply(1, x.coefficients)[()] == algebra.basis(1)
    assert algebra.right_basis_multiply(x.coefficients, 1)[()] == algebra.basis(1)
    assert algebra.left_operator_commutator(x, x.coefficients, x)[()] == algebra.zero()
    assert algebra.multiply(2, 3) == algebra.scalar(6)


def test_operators_preserve_reflected_dispatch(algebra):
    class Reflected:
        def __radd__(self, left):
            return "reflected"

    x = algebra.one()
    X = algebra.array([x.coefficients, x.coefficients])
    assert x + Reflected() == "reflected"
    assert X + Reflected() == "reflected"
    assert x.__add__(object()) is NotImplemented
    assert X.__mul__(x.coefficients) is NotImplemented
    with pytest.raises(TypeError):
        x * object()


@pytest.mark.parametrize("modulus", [(0, 1), (3, 1), (1, 8), (-20, -6)])
def test_prime_field_linear_presentations_have_one_parent(modulus):
    F = fcd.FiniteField(char=7)
    explicit = fcd.FiniteField.from_modulus(char=7, modulus=modulus)
    assert explicit is F and explicit.modulus == (0, 1)
    assert explicit(4) == F(4) and explicit(4) + F(2) == F(6)
    assert fcd.Algebra(dimension=4, field=explicit) is fcd.Algebra(dimension=4, char=7)
    assert pickle.loads(pickle.dumps(explicit)) is F
    for invalid in ((3, 2), (3, 7)):
        with pytest.raises(ValueError):
            fcd.FiniteField.from_modulus(char=7, modulus=invalid)


def test_field_repr_is_independent_of_unpickle_and_construction_order():
    F = fcd.FiniteField(char=13, deg=5)
    script = f"""
import pickle
import pyfastcd as fcd
restored = pickle.loads(bytes.fromhex({pickle.dumps(F).hex()!r}))
assert repr(restored) == 'FiniteField(char=13, deg=5)'
assert restored is fcd.FiniteField(char=13, deg=5)
explicit = fcd.FiniteField.from_modulus(char=7, modulus=(1, 0, 1))
before = repr(explicit)
default = fcd.FiniteField(char=7, deg=2)
assert repr(explicit) == before
if explicit is default:
    assert before == 'FiniteField(char=7, deg=2)'
else:
    assert before == 'FiniteField.from_modulus(char=7, modulus=(1, 0, 1))'
"""
    subprocess.run([sys.executable, "-c", script], check=True, env=os.environ.copy())


@pytest.mark.parametrize("degree", [1, 2, 5])
def test_sequences_of_field_scalars_construct_records_and_arrays(degree):
    F = fcd.FiniteField(char=7, deg=degree)
    A = fcd.Algebra(dimension=4, field=F)
    scalars = [F(i) for i in range(4)]
    expected = (
        A([0, 1, 2, 3])
        if degree == 1
        else A([[i] + [0] * (degree - 1) for i in range(4)])
    )
    assert A(scalars) == expected
    assert A.array(scalars)[()] == expected
    assert A.array([scalars, tuple(scalars)])[1] == expected
    assert fcd.cd_element(scalars, field=F) == expected
    field_array = F.array(scalars)
    for i, scalar in enumerate(scalars):
        assert field_array[i] == scalar
    objects = np.empty(4, dtype=object)
    for i, scalar in enumerate(scalars):
        objects[i] = scalar
    assert A(objects) == expected
    with pytest.raises(ValueError):
        A.asarray(objects, copy=False)
    with pytest.raises(ValueError, match="different algebras"):
        A([fcd.FiniteField(char=5).one()] * 4)


def test_cd_element_keeps_exact_mixed_large_integers():
    values = [2**63 + 1, -1, 2**160, -(2**127)]
    A = fcd.Algebra(dimension=4, char=13)
    assert fcd.cd_element(values, char=13) == A(values)


@pytest.mark.parametrize("p,d", [(0, 2), (7, 1), (7, 2)])
def test_unknown_algorithms_are_distinct_from_unavailable_algorithms(p, d):
    A = fcd.Algebra(dimension=4, char=p, deg=d)
    with pytest.raises(ValueError, match="unknown multiplication algorithm"):
        A.multiply(A.one(), A.one(), algorithm="Direct")
    with pytest.raises(ValueError, match="unavailable"):
        A.multiply(A.one(), A.one(), algorithm="cariow_cariowa")
