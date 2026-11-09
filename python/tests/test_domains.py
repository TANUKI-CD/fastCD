"""Independent exact arithmetic and unified-public-type regression tests."""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pyfastcd as fcd
import pytest


@pytest.mark.parametrize("p,d", [(3, 1), (3, 5), (5, 2), (13, 5), (65521, 2)])
@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 128, 256])
def test_finite_default_selection(p, d, n):
    A = fcd.Algebra(dimension=n, char=p, deg=d)
    expected = "direct" if n <= 4 else "quasilinear"
    assert A.default_algorithm == expected
    assert A.default_batch_algorithm == expected
    rng = np.random.default_rng(p + d + n)
    shape = (n, d) if d > 1 else (n,)
    x = A(rng.integers(0, p, size=shape, dtype=np.uint32))
    y = A(rng.integers(0, p, size=shape, dtype=np.uint32))
    np.testing.assert_array_equal(
        A.multiply(x, y, algorithm="auto").coefficients,
        A.multiply(x, y, algorithm=expected).coefficients,
    )


@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 128])
def test_complex_default_selection(n):
    A = fcd.Algebra(dimension=n, char=0, deg=2)
    expected = "direct" if n <= 4 else "quasilinear"
    assert A.default_algorithm == expected
    assert A.default_batch_algorithm == expected
    rng = np.random.default_rng(n)
    x = A(rng.normal(size=n) + 1j * rng.normal(size=n))
    y = A(rng.normal(size=n) + 1j * rng.normal(size=n))
    np.testing.assert_array_equal(
        A.multiply(x, y, algorithm="auto").coefficients,
        A.multiply(x, y, algorithm=expected).coefficients,
    )


def field_mul(a, b, p, modulus):
    d = len(modulus) - 1
    t = [0] * (2 * d - 1)
    for i in range(d):
        for j in range(d):
            t[i + j] = (t[i + j] + int(a[i]) * int(b[j])) % p
    for k in range(2 * d - 2, d - 1, -1):
        for j in range(d):
            t[k - d + j] = (t[k - d + j] - t[k] * modulus[j]) % p
    return tuple(t[:d])


def reference(x, y, p=0, modulus=None):
    if p:

        def add(a, b):
            return tuple((u + v) % p for u, v in zip(a, b, strict=True))

        def sub(a, b):
            return tuple((u - v) % p for u, v in zip(a, b, strict=True))

        def neg(a):
            return tuple(-u % p for u in a)

        def mul(a, b):
            return field_mul(a, b, p, modulus)
    else:

        def add(a, b):
            return a + b

        def sub(a, b):
            return a - b

        def neg(a):
            return -a

        def mul(a, b):
            return a * b

    def cd(a, b):
        if len(a) == 1:
            return [mul(a[0], b[0])]
        h = len(a) // 2
        u, v = a[:h], a[h:]
        s, t = b[:h], b[h:]

        def conj(z):
            return [z[0]] + [neg(q) for q in z[1:]]

        first = [sub(i, j) for i, j in zip(cd(u, s), cd(conj(t), v), strict=True)]
        second = [add(i, j) for i, j in zip(cd(t, u), cd(v, conj(s)), strict=True)]
        return first + second

    return cd(x, y)


@pytest.mark.parametrize("p", [3, 5, 13, 65521, 2147483647])
@pytest.mark.parametrize("d", [1, 2, 5, 12])
def test_fields_and_scalar_arithmetic(p, d):
    F = fcd.FiniteField(char=p, deg=d)
    assert F is fcd.FiniteField(char=p, deg=d)
    assert F.order == p**d and F.modulus[-1] == 1
    assert fcd.FiniteField.from_modulus(char=p, modulus=F.modulus) is F
    with pytest.raises(AttributeError):
        F.char = 7
    with pytest.raises(AttributeError):
        del F.modulus
    rng = np.random.default_rng(p + d)
    a = rng.integers(0, p, size=d, dtype=np.uint32)
    b = rng.integers(0, p, size=d, dtype=np.uint32)
    x = F(a if d > 1 else int(a[0]))
    y = F(b if d > 1 else int(b[0]))
    assert type(x) is fcd.Element
    expected = np.array(field_mul(a, b, p, F.modulus), dtype=np.uint32)
    np.testing.assert_array_equal((x * y).coefficients.reshape(d), expected)
    assert x * x.reciprocal() == 1
    assert F(-1) == p - 1 and F(p + 1) == 1 and F(1) != p + 1
    assert hash(F(1)) == hash(1)
    assert pickle.loads(pickle.dumps(F)) is F
    assert pickle.loads(pickle.dumps(x)) == x
    assert copy.copy(x) == x and copy.deepcopy(x) == x
    assert not x.coefficients.flags.writeable
    with pytest.raises(ValueError):
        x.coefficients.setflags(write=True)


@pytest.mark.parametrize(
    "p,d", [(3, 1), (3, 5), (5, 2), (13, 5), (65521, 2), (2147483647, 1)]
)
@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32])
def test_finite_cd_against_independent_reference(p, d, n):
    A = fcd.Algebra(dimension=n, char=p, deg=d)
    F = A.field
    assert type(A) is fcd.Algebra and type(A.one()) is fcd.Element
    assert A is fcd.Algebra(dimension=n, field=F)
    rng = np.random.default_rng(p + n + d)
    x = rng.integers(0, p, size=(n, d), dtype=np.uint32)
    y = rng.integers(0, p, size=(n, d), dtype=np.uint32)
    a = A(x if d > 1 else x[:, 0])
    b = A(y if d > 1 else y[:, 0])
    expected = np.array(
        reference(
            [tuple(map(int, r)) for r in x],
            [tuple(map(int, r)) for r in y],
            p,
            F.modulus,
        ),
        dtype=np.uint32,
    )
    for algorithm in ("direct", "quasilinear", "auto"):
        result = A.multiply(a, b, algorithm=algorithm)
        np.testing.assert_array_equal(result.coefficients.reshape(n, d), expected)
    assert a.square() == a * a
    assert A(a.coefficients) == a
    assert a * a.conjugate() == A.scalar(a.quadratic_norm())
    assert a.norm_squared() == a.quadratic_norm()
    assert a.inner(a) == a.quadratic_norm()
    assert a.commutator(b) == a * b - b * a
    assert 2 * a.imag_cross(b) == a.commutator(b)
    assert a.anticommutator(b) == a * b + b * a
    assert a.associator(b, a) == (a * b) * a - a * (b * a)
    assert pickle.loads(pickle.dumps(A)) is A
    assert pickle.loads(pickle.dumps(a)) == a
    assert A() == 0


@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 64, 128])
def test_complex_cd(n):
    A = fcd.Algebra(dimension=n, char=0, deg=2)
    rng = np.random.default_rng(n)
    x = rng.normal(size=n) + 1j * rng.normal(size=n)
    y = rng.normal(size=n) + 1j * rng.normal(size=n)
    expected = np.array(reference(list(x), list(y)))
    a, b = A(x), A(y)
    assert type(A) is fcd.Algebra and type(a) is fcd.Element
    assert a.dtype == np.dtype("complex128")
    assert a.real == x[0]
    for algorithm in ("direct", "quasilinear", "auto"):
        np.testing.assert_allclose(
            A.multiply(a, b, algorithm=algorithm).coefficients,
            expected,
            atol=1e-10,
            rtol=1e-12,
        )
    np.testing.assert_allclose(a.quadratic_norm(), sum(x * x), atol=1e-10)
    np.testing.assert_allclose(a.conjugate().coefficients, np.r_[x[:1], -x[1:]])
    np.testing.assert_allclose(
        (a * a.conjugate()).coefficients,
        A.scalar(a.quadratic_norm()).coefficients,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        a.reciprocal().coefficients,
        x * np.r_[1, -np.ones(n - 1)] / sum(x * x),
        atol=1e-10,
    )
    assert pickle.loads(pickle.dumps(a)) == a


@pytest.mark.parametrize("p,d", [(0, 2), (3, 1), (3, 5), (13, 2)])
def test_batches_and_protocols(p, d):
    A = fcd.Algebra(dimension=8, char=p, deg=d)
    shape = A.coefficient_shape
    values = np.ones((3, 2) + shape, dtype=A.dtype)
    X = A.asarray(values, copy=False)
    Y = X[::-1, :1]
    assert type(X) is fcd.ElementArray
    assert X.batch_shape == (3, 2) and Y.batch_shape == (3, 1)
    assert type(X[0, 0]) is fcd.Element
    out = np.empty_like(values)
    result = A.multiply(X, Y, out=out, algorithm="quasilinear")
    assert np.shares_memory(result.coefficients, out)
    for i in range(3):
        for j in range(2):
            assert result[i, j] == X[i, j] * Y[i, 0]
    assert np.multiply(X, Y, out=out) is out
    X[0, 1] = A.one()
    assert X[0, 1] == 1
    assert X.copy()[0, 1] == 1
    assert pickle.loads(pickle.dumps(X))[0, 1] == 1
    np.testing.assert_array_equal(np.asarray(X), values)
    if p:
        bad = values.copy()
        bad.flat[-1] = p
        shared = A.asarray(values, copy=False)
        values.flat[-1] = p
        saved = out.copy()
        with pytest.raises(ValueError):
            A.multiply(shared, shared, out=out)
        np.testing.assert_array_equal(out, saved)
        values.flat[-1] = 1
        with pytest.raises(ValueError):
            A.asarray(bad, copy=False)
    assert A.asarray(np.ones((0,) + shape, dtype=A.dtype)).batch_shape == (0,)


def test_quadratic_norm_and_zero_divisors():
    R = fcd.Algebra(dimension=2)
    assert R([3, 4]).quadratic_norm() == 25
    assert R([3, 4]).norm() == 5
    assert R.asarray([[3, 4]]).quadratic_norm()[0] == 25
    for A, coefs in [
        (fcd.Algebra(dimension=2, char=5), [1, 2]),
        (fcd.Algebra(dimension=2, deg=2), [1, 1j]),
    ]:
        x = A(coefs)
        assert bool(x) and x.quadratic_norm() == 0
        with pytest.raises(ZeroDivisionError):
            x.reciprocal()
        with pytest.raises(ZeroDivisionError):
            x**-1
        with pytest.raises(TypeError):
            x.norm()


def test_selection_validation_and_legacy_defaults():
    assert fcd.Algebra(dimension=8) is fcd.Algebra(level=3, char=0, deg=1)
    with pytest.raises(TypeError):
        fcd.Algebra(dimension=8)(np.ones(8, dtype=complex))
    for p, d in [(2, 1), (9, 1), (-3, 1), (13, 0), (13, 13), (0, 3)]:
        with pytest.raises(ValueError):
            fcd.Algebra(dimension=8, char=p, deg=d)
    for key in ("char", "deg"):
        with pytest.raises(TypeError):
            fcd.Algebra(dimension=8, **{key: True})
    A = fcd.Algebra(dimension=8, char=3, deg=2)
    with pytest.raises(TypeError):
        A(np.ones((8, 2), dtype=float))
    with pytest.raises(ValueError):
        A.multiply(A.one(), A.one(), algorithm="cariow_cariowa")
    with pytest.raises(ValueError):
        A.one() + fcd.Algebra(dimension=8).one()
    assert fcd.cd_element(A.one()) == A.one()
    assert fcd.cd_element(A.one().coefficients, field=A.field) == A.one()


def test_v110_real_pickle_fixtures():
    import base64
    import json
    from pathlib import Path

    fixture = json.loads(
        (Path(__file__).parent / "data/v1.1.0-real-pickles.json").read_text()
    )
    assert fixture["version"] == "1.1.0"
    for record in fixture["records"]:
        A, x, a = pickle.loads(base64.b64decode(record["payload"]))
        n = record["dimension"]
        assert A is fcd.Algebra(dimension=n) and A.char == 0 and A.deg == 1
        assert type(x) is fcd.Element and type(a) is fcd.ElementArray
        np.testing.assert_array_equal(x.coefficients, np.arange(n) - 0.5)
        np.testing.assert_array_equal(
            a.coefficients, np.arange(6 * n).reshape(6, n)[::-2]
        )
        assert x.algebra is A and a.algebra is A


@pytest.mark.parametrize("p,d", [(5, 1), (13, 5)])
def test_coefficient_field_arrays_and_batch_keys(p, d):
    A = fcd.Algebra(dimension=4, char=p, deg=d)
    shape = (3, 2) + A.coefficient_shape
    a = A.array(np.arange(np.prod(shape)).reshape(shape))
    scalars = a.real
    assert type(scalars) is fcd.ElementArray and scalars.algebra is A.field
    np.testing.assert_array_equal(
        (a * scalars).coefficients, (scalars * a).coefficients
    )
    np.testing.assert_array_equal((a / A.field.one()).coefficients, a.coefficients)
    assert (a + scalars - scalars == a).all()
    for key in (
        np.array([[True, False], [False, True], [True, False]]),
        (np.array([True, False, True]), Ellipsis),
        (None, Ellipsis, 1),
        (True, Ellipsis),
        (False,),
        ([2, 0], [1, 0]),
    ):
        selected = a[key]
        indices = np.arange(6).reshape(3, 2)[key]
        expected = a.coefficients.reshape((6,) + A.coefficient_shape)[indices]
        np.testing.assert_array_equal(selected.coefficients, expected)
        assert selected.algebra is A
    b = a.copy()
    mask = np.array([[True, False], [False, True], [True, False]])
    b[mask] = A.one()
    np.testing.assert_array_equal(
        b[mask].coefficients, np.broadcast_to(A.one().coefficients, b[mask].shape)
    )


def test_complex_scalar_comparison_hash_and_conjugation():
    from decimal import Decimal
    from fractions import Fraction

    A = fcd.Algebra(level=2, char=0, deg=2)
    x = A.scalar(2**53)
    assert x == 2**53 and x != 2**53 + 1
    assert hash(x) == hash(2**53)
    assert x != Decimal(2**53 + 1) and x != Fraction(2**53 + 1)
    assert x != np.int64(2**53 + 1)
    nan = A.scalar(complex(float("nan"), 1))
    assert hash(nan) == hash(nan) and nan != nan
    assert A.scalar(1j).conjugate() == 1j
    array = A.array(np.broadcast_to(x.coefficients, (3, A.dimension)))
    assert (array == 2**53).all() and not (array == 2**53 + 1).any()


@pytest.mark.parametrize("p,d", [(0, 2), (3, 5), (65521, 2)])
def test_layout_views_canonical_validation_and_failure_atomicity(p, d):
    A = fcd.Algebra(dimension=8, char=p, deg=d)
    shape = (4, 3) + A.coefficient_shape
    source = np.zeros(shape, dtype=A.dtype)
    source[...] = A.one().coefficients
    a = A.asarray(source[::-1, ::-1], copy=False)
    assert np.shares_memory(a.coefficients, source)
    assert a.batch_shape == (4, 3)
    np.testing.assert_array_equal((a * a).coefficients, a.coefficients)
    empty = A.asarray(np.empty((0,) + A.coefficient_shape, dtype=A.dtype), copy=False)
    assert (empty * empty).shape == empty.shape
    b = A.array(source)
    out = np.full(shape, 2, dtype=A.dtype)
    before = out.copy()
    with pytest.raises(ValueError):
        A.multiply(a, b, out=source)
    np.testing.assert_array_equal(out, before)
    if p:
        b.coefficients[-1].flat[-1] = p
        with pytest.raises(ValueError, match="canonical"):
            A.multiply(a, b, out=out)
        np.testing.assert_array_equal(out, before)
    b = A.array(source)
    b.coefficients[-1] = 0
    with pytest.raises(ZeroDivisionError):
        b.reciprocal()


def test_exact_mixed_integer_coercion_and_complex_real_scalars():
    from decimal import Decimal
    from fractions import Fraction

    A = fcd.Algebra(dimension=2, char=13)
    assert A([2**63 + 1, -1]) == A([(2**63 + 1) % 13, 12])
    assert A([2**160, -(2**127)]) == A([pow(2, 160, 13), -pow(2, 127, 13)])
    C = fcd.Algebra(dimension=2, char=0, deg=2)
    for scalar in (Decimal("0.5"), Fraction(1, 2), np.float64(0.5)):
        assert C.scalar(scalar) == 0.5 and C.one() * scalar == C.scalar(0.5)


@pytest.mark.parametrize("p,d", [(0, 2), (13, 5)])
def test_array_keyword_signatures_and_raw_record_returns(p, d):
    A = fcd.Algebra(dimension=4, char=p, deg=d)
    x = A.array(np.broadcast_to(A.one().coefficients, (3,) + A.coefficient_shape))
    y = A.array(np.broadcast_to(A.basis(1).coefficients, x.shape))
    np.testing.assert_array_equal(
        x.inner(other=y).coefficients if p else x.inner(other=y),
        x.inner(y).coefficients if p else x.inner(y),
    )
    assert (x.associator(y=y, z=x) == x.associator(y, x)).all()
    with pytest.raises(TypeError):
        x.conjugate(x)
    with pytest.raises(TypeError):
        x.norm_squared(unexpected=True)
    assert (
        type(A.multiply(A.one().coefficients, A.one().coefficients)) is fcd.ElementArray
    )
    assert type(A.left_basis_multiply(0, A.one().coefficients)) is fcd.ElementArray
    assert type(A.multiply(A.one(), A.one())) is fcd.Element
    assert type(A.multiply(1, 1)) is fcd.Element
    assert list(x)[0] == A.one()
    assert copy.deepcopy(x).algebra is A and pickle.loads(pickle.dumps(x)).algebra is A


@pytest.mark.parametrize("p,d", [(0, 2), (5, 1), (13, 5)])
def test_scalar_embedding_rejects_complete_cd_records(p, d):
    A = fcd.Algebra(dimension=4, char=p, deg=d)
    for value in (A.one(), A.one().coefficients, A.one().coefficients.tolist()):
        with pytest.raises(TypeError, match="coefficient field"):
            A.scalar(value)
    value = A.field.one() if p else 2j
    assert A.scalar(value).real == value


@pytest.mark.parametrize("d", [1, 5])
def test_matching_field_scalars_embed_in_equality_and_hashes(d):
    F = fcd.FiniteField(char=13, deg=d)
    A = fcd.Algebra(dimension=4, field=F)
    c = F(3) if d == 1 else F([3, 2, 1, 0, 0])
    x = A.scalar(c)
    assert x == c and c == x and hash(x) == hash(c)
    assert {x: 17}[c] == 17 and {c: 19}[x] == 19
    assert x + A.basis(1) != c
    array = A.array(np.broadcast_to(x.coefficients, (2, 3) + A.coefficient_shape))
    assert (array == c).all() and (c == array).all()
    assert (array == array.real).all() and (array.real == array).all()


@pytest.mark.parametrize("p,d", [(0, 2), (13, 5)])
def test_numpy_output_selects_containing_parent_and_declines_raw_arrays(p, d):
    A = fcd.Algebra(dimension=4, char=p, deg=d)
    values = np.broadcast_to(A.one().coefficients, (3,) + A.coefficient_shape)
    x = A.array(values)
    scalar = A.field(3) if p else 3j
    target = np.empty(x.shape, dtype=A.dtype)
    assert np.multiply(x, scalar, out=target) is target
    np.testing.assert_array_equal(target, (x * scalar).coefficients)
    assert np.multiply(scalar, x, out=target) is target
    np.testing.assert_array_equal(target, (scalar * x).coefficients)
    before = target.copy()
    with pytest.raises(TypeError):
        np.multiply(x, values, out=target)
    np.testing.assert_array_equal(target, before)
    with pytest.raises(TypeError):
        x * values


def test_real_element_retains_v110_allocation_layout():
    import struct

    basic = object.__basicsize__ + 4 * struct.calcsize("n")
    assert fcd.Element.__basicsize__ == basic
    assert fcd.Element.__itemsize__ == 8
    for n in (1, 2, 4, 16, 1024):
        x = fcd.Algebra(dimension=n).one()
        assert x.__sizeof__() == basic + 8 * n


def test_field_buffer_contiguity_requests():
    import ctypes

    class Buffer(ctypes.Structure):
        _fields_ = [
            ("buf", ctypes.c_void_p),
            ("obj", ctypes.c_void_p),
            ("len", ctypes.c_ssize_t),
            ("itemsize", ctypes.c_ssize_t),
            ("readonly", ctypes.c_int),
            ("ndim", ctypes.c_int),
            ("format", ctypes.c_char_p),
            ("shape", ctypes.POINTER(ctypes.c_ssize_t)),
            ("strides", ctypes.POINTER(ctypes.c_ssize_t)),
            ("suboffsets", ctypes.POINTER(ctypes.c_ssize_t)),
            ("internal", ctypes.c_void_p),
        ]

    get = ctypes.pythonapi["PyObject_GetBuffer"]
    get.argtypes = [ctypes.py_object, ctypes.POINTER(Buffer), ctypes.c_int]
    get.restype = ctypes.c_int
    release = ctypes.pythonapi["PyBuffer_Release"]
    release.argtypes = [ctypes.POINTER(Buffer)]
    release.restype = None
    F = fcd.FiniteField(char=13, deg=5)
    for x in (
        F.one(),
        fcd.Algebra(dimension=1, field=F).one(),
        fcd.Algebra(dimension=4, char=0, deg=2).one(),
    ):
        view = Buffer()
        assert get(x, ctypes.byref(view), 0x58 | 4) == 0  # F_CONTIGUOUS | FORMAT
        assert view.readonly and view.len == x.coefficients.nbytes
        release(ctypes.byref(view))
    x = fcd.Algebra(dimension=4, field=F).one()
    view = Buffer()
    with pytest.raises(BufferError, match="Fortran"):
        get(x, ctypes.byref(view), 0x58 | 4)
    assert get(x, ctypes.byref(view), 0x38 | 4) == 0  # C_CONTIGUOUS | FORMAT
    release(ctypes.byref(view))
