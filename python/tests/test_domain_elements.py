"""Element arithmetic over C and finite fields against array arithmetic.

Element operations and ElementArray operations reach the kernels through
different native entry points. These tests require the two to agree record
for record, bit for bit, and pin the scalar, index, power, hash and storage
rules that the Element entry points implement.
"""

from __future__ import annotations

import gc
import sys

import numpy as np
import pyfastcd as fcd
import pytest

PARENTS = {
    "C1": lambda: fcd.Algebra(dimension=1, char=0, deg=2),
    "C8": lambda: fcd.Algebra(dimension=8, char=0, deg=2),
    "C32": lambda: fcd.Algebra(dimension=32, char=0, deg=2),
    "F7_1": lambda: fcd.Algebra(dimension=1, char=7),
    "F7_8": lambda: fcd.Algebra(dimension=8, char=7),
    "F7_64": lambda: fcd.Algebra(dimension=64, char=7),
    "F13e5_4": lambda: fcd.Algebra(dimension=4, char=13, deg=5),
    "FM31_8": lambda: fcd.Algebra(dimension=8, char=2**31 - 1),
    "F7": lambda: fcd.FiniteField(char=7),
    "F13e5": lambda: fcd.FiniteField(char=13, deg=5),
}


@pytest.fixture(params=sorted(PARENTS))
def parent(request):
    return PARENTS[request.param]()


def records(parent, rng, batch=()):
    """Return random native coefficient storage with a leading batch shape."""

    shape = batch + parent.coefficient_shape
    if parent.char:
        return rng.integers(0, parent.char, size=shape, dtype=np.uint32)
    return rng.integers(-4, 5, size=shape) + 1j * rng.integers(-4, 5, size=shape)


def invertible(parent, rng):
    """Return a random element with nonzero quadratic norm."""

    while True:
        x = parent(records(parent, rng))
        if x.quadratic_norm():
            return x


def raw(value):
    """Return the exact bytes of an element, a record or a reduction."""

    if isinstance(value, complex):
        return np.asarray(value, dtype=np.complex128).tobytes()
    return np.ascontiguousarray(np.asarray(value)).tobytes()


def outcome(function):
    """Return a result's bytes, or the type of the exception it raises."""

    try:
        return raw(function())
    except (TypeError, ValueError, ZeroDivisionError, OverflowError) as error:
        return type(error)


def test_operations_match_array_records(parent):
    rng = np.random.default_rng(11)
    for _ in range(6):
        x, y, z = (invertible(parent, rng) for _ in range(3))
        X, Y, Z = (parent.array([np.asarray(v)]) for v in (x, y, z))
        pairs = [
            (x + y, X + Y),
            (x - y, X - Y),
            (x * y, X * Y),
            (-x, -X),
            (+x, +X),
            (x.copy(), X.copy()),
            (x.conjugate(), X.conjugate()),
            (x.imag_part(), X.imag_part()),
            (x.imag, X.imag),
            (x.square(), X.square()),
            (x.reciprocal(), X.reciprocal()),
            (x.quadratic_norm(), X.quadratic_norm()),
            (x.norm_squared(), X.norm_squared()),
            (x.inner(y), X.inner(Y)),
            (x.imag_cross(y), X.imag_cross(Y)),
            (x.commutator(y), X.commutator(Y)),
            (x.anticommutator(y), X.anticommutator(Y)),
            (x.associator(y, z), X.associator(Y, Z)),
            (parent.multiply(x, y), parent.multiply(X, Y)),
        ]
        if isinstance(parent, fcd.Algebra):
            index = int(rng.integers(0, parent.dimension))
            pairs += [
                (
                    parent.multiply(x, y, algorithm=name),
                    parent.multiply(X, Y, algorithm=name),
                )
                for name in ("direct", "quasilinear")
            ]
            pairs += [
                (
                    parent.left_basis_multiply(index, x),
                    parent.left_basis_multiply(index, X),
                ),
                (
                    parent.right_basis_multiply(x, index),
                    parent.right_basis_multiply(X, index),
                ),
                (
                    parent.left_operator_commutator(x, y, z),
                    parent.left_operator_commutator(X, Y, Z),
                ),
                (
                    parent.right_operator_commutator(x, y, z),
                    parent.right_operator_commutator(X, Y, Z),
                ),
            ]
        for single, batch in pairs:
            assert not isinstance(single, fcd.ElementArray)
            assert raw(single) == raw(np.asarray(batch)[0])
        assert x.real == X.real[0]
        assert (x == y) == bool((X == Y)[0])
        assert (x != y) == bool((X != Y)[0])
        assert x == x.copy() and not x != x.copy()
        assert bool(x) and not bool(parent.zero())


def test_scalar_operands_match_array_records(parent):
    rng = np.random.default_rng(12)
    x = invertible(parent, rng)
    X = parent.array([np.asarray(x)])
    if parent.char:
        p = parent.char
        scalars = [0, 1, 2, -1, p - 1, p, p + 3, -p, 2**63 - 1, 2**63, -(2**70), True]
        scalars += [parent.field(records(parent.field, rng)), parent.field(0)]
    else:
        scalars = [0, 1, -3, True, 2.5, -0.0, 1 - 2j, 0j, float("inf"), 2**70]
        scalars += [complex(float("nan"), 1.0), 10**400]
    for s in scalars:
        for single, batch in (
            (lambda s=s: x * s, lambda s=s: (X * s)[0]),
            (lambda s=s: s * x, lambda s=s: (s * X)[0]),
            (lambda s=s: x / s, lambda s=s: (X / s)[0]),
            (lambda s=s: x + s, lambda s=s: (X + s)[0]),
            (lambda s=s: s + x, lambda s=s: (s + X)[0]),
            (lambda s=s: x - s, lambda s=s: (X - s)[0]),
            (lambda s=s: s - x, lambda s=s: (s - X)[0]),
            (lambda s=s: parent.multiply(x, s), lambda s=s: parent.multiply(X, s)[0]),
            (lambda s=s: parent.multiply(s, x), lambda s=s: parent.multiply(s, X)[0]),
        ):
            assert outcome(single) == outcome(batch)
    for zero in (0, parent.zero() if not isinstance(parent, fcd.Algebra) else 0):
        with pytest.raises(ZeroDivisionError, match="zero quadratic norm or scalar"):
            x / zero


def test_integer_scalars_reduce_modulo_the_characteristic():
    F = fcd.FiniteField(char=13, deg=2)
    A = fcd.Algebra(dimension=2, field=F)
    x = A([[1, 2], [3, 4]])
    assert x * 14 == x
    assert x * -12 == x
    assert x * (13**40 + 1) == x
    assert x + 13 == x
    assert x * True == x
    assert (x * 2).coefficients.tolist() == [[2, 4], [6, 8]]
    assert (x + 5).coefficients.tolist() == [[6, 2], [3, 4]]
    assert (5 - x).coefficients.tolist() == [[4, 11], [10, 9]]
    assert F([3, 4]) * 14 == F([3, 4])
    assert F(5) + F(9) == F(1)


def test_field_scalars_combine_only_within_one_presentation():
    default = fcd.FiniteField(char=5, deg=2)
    other = fcd.FiniteField.from_modulus(char=5, modulus=(2, 0, 1))
    assert default is not other
    A = fcd.Algebra(dimension=4, field=default)
    x = A(np.arange(8, dtype=np.uint32).reshape(4, 2) % 5)
    c = default([2, 3])
    assert c * x == x * c == A.scalar(c) * x
    assert (x / c) * c == x
    assert x + c == x + A.scalar(c)
    assert c - x == A.scalar(c) - x
    for operation in (
        lambda: x * other([2, 3]),
        lambda: other([2, 3]) * x,
        lambda: x + other([2, 3]),
        lambda: c * other([2, 3]),
        lambda: x * fcd.FiniteField(char=5)(2),
    ):
        with pytest.raises(ValueError, match="different algebras"):
            operation()
    # Division accepts only a scalar of the coefficient field as divisor
    for operation in (lambda: x / other([2, 3]), lambda: x / x, lambda: c / x):
        with pytest.raises(TypeError):
            operation()


def test_powers_match_array_powers_and_written_products(parent):
    rng = np.random.default_rng(13)
    x = invertible(parent, rng)
    X = parent.array([np.asarray(x)])
    for exponent in list(range(-6, 12)) + [31, -17, 100]:
        assert raw(x**exponent) == raw(np.asarray(X**exponent)[0])
    assert x**0 == parent.one()
    assert x**1 == x and x**1 is not x
    assert x**2 == x.square()
    assert x**3 == (parent.one() * x) * x.square()
    assert x**-1 == x.reciprocal()
    assert x**-2 == x.reciprocal().square()
    assert parent.zero() ** 0 == parent.one()
    with pytest.raises(ZeroDivisionError):
        parent.zero() ** -1
    with pytest.raises(TypeError):
        pow(x, 2, 5)
    with pytest.raises(TypeError):
        x**1.0
    if parent.char:
        # Exponents beyond the machine range take the same written order
        huge = 2**70 + 3
        assert raw(x**huge) == raw(np.asarray(X**huge)[0])


def test_hash_agrees_with_equality(parent):
    rng = np.random.default_rng(14)
    x = parent(records(parent, rng))
    assert hash(x) == hash(x.copy()) == hash(parent(np.asarray(x)))
    assert len({x, x.copy(), parent(np.asarray(x))}) == 1
    assert parent.scalar(3) == 3 and hash(parent.scalar(3)) == hash(3)
    assert hash(parent.zero()) == hash(0)
    if parent.char:
        field = parent.field
        c = field(records(field, rng))
        assert parent.scalar(c) == c and hash(parent.scalar(c)) == hash(c)
        assert parent.scalar(parent.char + 3) == 3
        assert hash(parent.scalar(parent.char + 3)) == hash(3)
    else:
        z = 1.5 - 2j
        assert parent.scalar(z) == z and hash(parent.scalar(z)) == hash(z)
        assert hash(parent.scalar(2.0)) == hash(2)
        values = np.zeros(parent.dimension, dtype=complex)
        values[0] = complex("nan")
        assert hash(parent(values)) == hash(parent(values))


def test_coefficient_access(parent):
    rng = np.random.default_rng(15)
    x = parent(records(parent, rng))
    values = np.asarray(x)
    if not isinstance(parent, fcd.Algebra):
        assert x.dimension == 1 and x.level == 0
        assert x.real == x and x.real is not x
        with pytest.raises(TypeError):
            len(x)
        with pytest.raises(TypeError):
            x[0]
        return
    n = parent.dimension
    assert len(x) == x.dimension == n
    assert x.level == parent.level
    assert x.shape == parent.coefficient_shape
    assert x.dtype == parent.dtype
    for index in range(-n, n):
        expected = (
            parent.field(values[index]) if parent.char else complex(values[index])
        )
        assert x[index] == expected
        assert x[np.int64(index)] == expected
    assert x.real == x[0]
    assert list(x) == [x[i] for i in range(n)]
    assert x[:] == tuple(x)
    for index in (n, -n - 1, 2**63 - 1, 2**63, -(2**63) - 1, 2**70, -(2**70)):
        with pytest.raises(IndexError, match="element index is out of range"):
            x[index]
    for index in (True, 1.0, "1", None):
        with pytest.raises(TypeError):
            x[index]


@pytest.mark.parametrize(
    "selectors",
    [{}, {"char": 0, "deg": 2}, {"char": 7}, {"char": 13, "deg": 5}],
    ids=["R", "C", "F7", "F13e5"],
)
@pytest.mark.parametrize("batch", [(3,), (1,), (2, 3), (40,)])
def test_one_record_broadcasts_over_a_batch(selectors, batch):
    A = fcd.Algebra(dimension=8, **selectors)
    rng = np.random.default_rng(16)

    def storage(shape):
        if A.char:
            return rng.integers(0, A.char, size=shape + A.coefficient_shape)
        values = rng.integers(-4, 5, size=shape + A.coefficient_shape)
        if selectors:
            return values + 1j * rng.integers(-4, 5, size=values.shape)
        return values.astype(np.float64)

    X = A.array(storage(batch))
    Y = A.array(storage(batch))
    e = A(storage(()))
    flat_x = [A(v) for v in np.asarray(X).reshape((-1,) + A.coefficient_shape)]
    flat_y = [A(v) for v in np.asarray(Y).reshape((-1,) + A.coefficient_shape)]

    def expect(results):
        return np.stack([np.asarray(v) for v in results]).reshape(X.shape)

    def same(array, results):
        np.testing.assert_array_equal(np.asarray(array), expect(results))

    same(X * e, [x * e for x in flat_x])
    same(e * X, [e * x for x in flat_x])
    same(X + e, [x + e for x in flat_x])
    same(e - X, [e - x for x in flat_x])
    same(X.commutator(e), [x.commutator(e) for x in flat_x])
    pairs = list(zip(flat_x, flat_y, strict=True))
    same(X.associator(e, Y), [x.associator(e, y) for x, y in pairs])
    same(X.associator(Y, e), [x.associator(y, e) for x, y in pairs])
    same(A.multiply(X, np.asarray(e)), [x * e for x in flat_x])
    same(
        A.left_operator_commutator(e, X, e),
        [A.left_operator_commutator(e, x, e) for x in flat_x],
    )
    out = A.array(storage(batch))
    assert A.multiply(e, X, out=out) is out
    same(out, [e * x for x in flat_x])
    inner = X.inner(e)
    expected = [x.inner(e) for x in flat_x]
    if A.char:
        tail = A.field.coefficient_shape
        np.testing.assert_array_equal(
            np.asarray(inner).reshape((-1,) + tail),
            np.stack([np.asarray(v) for v in expected]).reshape((-1,) + tail),
        )
    else:
        np.testing.assert_array_equal(np.asarray(inner).reshape(-1), expected)


@pytest.mark.parametrize(
    "selectors",
    [{}, {"char": 0, "deg": 2}, {"char": 7}, {"char": 13, "deg": 5}],
    ids=["R", "C", "F7", "F13e5"],
)
def test_strided_storage_matches_contiguous_storage(selectors):
    A = fcd.Algebra(dimension=4, **selectors)
    rng = np.random.default_rng(17)
    tail = A.coefficient_shape
    if A.char:
        block = rng.integers(0, A.char, size=(6, 5) + tail).astype(np.uint32)
    elif selectors:
        block = rng.integers(-4, 5, size=(6, 5) + tail) + 1j
    else:
        block = rng.integers(-4, 5, size=(6, 5) + tail).astype(np.float64)
    one = block[0, 0]
    layouts = [
        block[::2],
        block[::-1, ::-1],
        block[1:4, ::2],
        np.swapaxes(block, 0, 1),
        np.broadcast_to(one, (3, 2) + tail),
        np.broadcast_to(block[:1], (4, 5) + tail),
    ]
    for view in layouts:
        strided = A.asarray(view)
        contiguous = A.array(np.ascontiguousarray(view))
        assert np.shares_memory(view, strided.coefficients)
        other = A.array(np.ascontiguousarray(view)[::-1])
        for operation in (
            lambda v, w: v * w,
            lambda v, w: w * v,
            lambda v, w: v + w,
            lambda v, w: -v,
            lambda v, w: v.conjugate(),
            lambda v, w: v.square(),
            lambda v, w: v.quadratic_norm(),
            lambda v, w: v.inner(w),
            lambda v, w: v.commutator(w),
            lambda v, w: v.associator(w, v),
            lambda v, w: A.left_basis_multiply(2, v),
            lambda v, w: v * 3,
        ):
            np.testing.assert_array_equal(
                np.asarray(operation(strided, other)),
                np.asarray(operation(contiguous, other)),
            )


@pytest.mark.parametrize("position", [0, 17, 39])
def test_noncanonical_residues_are_found_anywhere_in_a_batch(position):
    A = fcd.Algebra(dimension=8, char=7)
    rng = np.random.default_rng(18)
    good = A.array(rng.integers(0, 7, size=(40, 8)))
    storage = rng.integers(0, 7, size=(40, 8)).astype(np.uint32)
    bad = A.asarray(storage)
    storage[position, 5] = 7
    out = A.array(np.zeros((40, 8), dtype=np.uint32))
    before = out.coefficients.copy()
    for operation in (
        lambda: bad * good,
        lambda: good * bad,
        lambda: bad + good,
        lambda: -bad,
        lambda: bad.quadratic_norm(),
        lambda: bad * 2,
        lambda: A.multiply(good, bad, out=out),
        lambda: bad[::-1] * good,
    ):
        with pytest.raises(ValueError, match="canonical residues"):
            operation()
    np.testing.assert_array_equal(out.coefficients, before)
    # Indexing and copying reduce the stored word instead
    element = bad[position]
    assert np.asarray(element)[5] == 0
    assert bad.copy().coefficients[position, 5] == 0
    assert element * A.one() == element
    np.testing.assert_array_equal(
        np.asarray(good * element),
        np.asarray(good * A.array(np.asarray(element)[None])),
    )


@pytest.mark.parametrize(
    "selectors", [{"char": 0, "deg": 2}, {"char": 7}], ids=["C", "F7"]
)
def test_shared_storage_rules_hold_for_contiguous_batches(selectors):
    A = fcd.Algebra(dimension=4, **selectors)
    rng = np.random.default_rng(19)
    if A.char:
        storage = rng.integers(0, A.char, size=(6, 4)).astype(np.uint32)
    else:
        storage = (rng.integers(-4, 5, size=(6, 4)) + 1j).astype(np.complex128)
    X = A.asarray(storage)
    records_ = [A(v) for v in storage]
    # Identical or disjoint records are valid inputs
    for left, right, expected in (
        (X, X, [x * x for x in records_]),
        (
            X[:-1],
            X[1:],
            [x * y for x, y in zip(records_[:-1], records_[1:], strict=True)],
        ),
        (X, X[:1], [x * records_[0] for x in records_]),
        (
            X,
            X[::-1],
            [x * y for x, y in zip(records_, records_[::-1], strict=True)],
        ),
    ):
        np.testing.assert_array_equal(
            np.asarray(left * right), np.stack([np.asarray(v) for v in expected])
        )
    # An output may not share storage with an input
    with pytest.raises(ValueError):
        A.multiply(X, X, out=X)
    with pytest.raises(ValueError):
        A.multiply(X[:-1], X[:-1], out=X[1:])
    # Records that overlap without coinciding are rejected
    flat = storage.reshape(-1)
    shifted = A.asarray(flat[2:18].reshape(4, 4))
    with pytest.raises(ValueError, match="overlapping"):
        X[:4] * shifted


def test_element_operations_release_their_references(parent):
    rng = np.random.default_rng(20)
    x, y = invertible(parent, rng), invertible(parent, rng)
    owners = [parent] + ([parent.field] if parent.char else [])

    def exercise():
        for _ in range(200):
            x * y, x + y, -x, x.conjugate(), x.reciprocal(), x.inner(y)
            x.quadratic_norm(), x * 2, 2 * x, x / 3, x + 1, x**5, x**-2
            hash(x), x == y, bool(x), x.real, x.copy(), x.associator(y, x)
            if isinstance(parent, fcd.Algebra):
                x[0], len(x), parent.multiply(x, y), parent.multiply(x, 2)

    exercise()
    gc.collect()
    before = [sys.getrefcount(owner) for owner in owners]
    exercise()
    gc.collect()
    assert [sys.getrefcount(owner) for owner in owners] == before
