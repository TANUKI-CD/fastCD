r"""Independent Cayley--Dickson reference arithmetic for Python tests.

This module deliberately does not import :mod:`pyfastcd`.  It evaluates the
defining doubling identity directly, so an error in the production basis-sign
logic or in either optimized multiplication schedule cannot make the oracle
agree with the implementation by construction.

The convention used by fastCD is

.. math::

   (a,b)(c,d) = (ac - \overline{d}b,\; da + b\overline{c}).

The routines here favor clarity over speed and are intended only for tests.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _coefficients(value: ArrayLike) -> FloatArray:
    """Return *value* as a validated one-dimensional ``float64`` array."""

    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError("reference operands must be one-dimensional")
    dimension = result.size
    if dimension == 0 or dimension & (dimension - 1):
        raise ValueError("reference dimension must be a positive power of two")
    return result


def conjugate(value: ArrayLike) -> FloatArray:
    """Return the standard Cayley--Dickson conjugate of *value*."""

    result = _coefficients(value).copy()
    result[1:] *= -1.0
    return result


def multiply(left: ArrayLike, right: ArrayLike) -> FloatArray:
    """Multiply two equal-dimensional operands by defining recursion."""

    x = _coefficients(left)
    y = _coefficients(right)
    if x.shape != y.shape:
        raise ValueError("reference operands must have the same dimension")
    return _multiply_recursive(x, y)


def _multiply_recursive(left: FloatArray, right: FloatArray) -> FloatArray:
    """Evaluate the defining four-product recursion on validated operands."""

    dimension = left.size
    if dimension == 1:
        return np.array([left[0] * right[0]], dtype=np.float64)

    half = dimension // 2
    a, b = left[:half], left[half:]
    c, d = right[:half], right[half:]
    first = _multiply_recursive(a, c) - _multiply_recursive(conjugate(d), b)
    second = _multiply_recursive(d, a) + _multiply_recursive(b, conjugate(c))
    return np.concatenate((first, second))


def multiply_batch(left: ArrayLike, right: ArrayLike) -> FloatArray:
    """Multiply matching batches whose final axes hold coefficients."""

    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.shape != y.shape or x.ndim < 1:
        raise ValueError("reference batches must have matching shapes")
    records = [
        multiply(a, b)
        for a, b in zip(
            x.reshape(-1, x.shape[-1]),
            y.reshape(-1, y.shape[-1]),
            strict=True,
        )
    ]
    return np.asarray(records, dtype=np.float64).reshape(x.shape)


def basis_product(dimension: int, left_index: int, right_index: int) -> tuple[int, int]:
    """Return ``(index, sign)`` for a standard-basis product."""

    if dimension <= 0 or dimension & (dimension - 1):
        raise ValueError("reference dimension must be a positive power of two")
    if not (0 <= left_index < dimension and 0 <= right_index < dimension):
        raise IndexError("reference basis index is out of range")
    left = np.zeros(dimension, dtype=np.float64)
    right = np.zeros(dimension, dtype=np.float64)
    left[left_index] = 1.0
    right[right_index] = 1.0
    product = multiply(left, right)
    nonzero = np.flatnonzero(product)
    if nonzero.size != 1:
        raise AssertionError("a reference basis product was not monomial")
    index = int(nonzero[0])
    return index, int(product[index])


def imag_part(value: ArrayLike) -> FloatArray:
    """Return *value* with its scalar coordinate set to zero."""

    result = _coefficients(value).copy()
    result[0] = 0.0
    return result


def imag_cross(left: ArrayLike, right: ArrayLike) -> FloatArray:
    """Return one half of the commutator of the imaginary parts."""

    x = imag_part(left)
    y = imag_part(right)
    return 0.5 * (multiply(x, y) - multiply(y, x))


def commutator(left: ArrayLike, right: ArrayLike) -> FloatArray:
    """Return ``left*right - right*left``."""

    return multiply(left, right) - multiply(right, left)


def anticommutator(left: ArrayLike, right: ArrayLike) -> FloatArray:
    """Return ``left*right + right*left``."""

    return multiply(left, right) + multiply(right, left)


def associator(x: ArrayLike, y: ArrayLike, z: ArrayLike) -> FloatArray:
    """Return ``(x*y)*z - x*(y*z)`` with the displayed parentheses."""

    return multiply(multiply(x, y), z) - multiply(x, multiply(y, z))


def left_operator_commutator(
    a: ArrayLike, b: ArrayLike, x: ArrayLike
) -> FloatArray:
    """Return ``a*(b*x) - b*(a*x)``."""

    return multiply(a, multiply(b, x)) - multiply(b, multiply(a, x))


def right_operator_commutator(
    a: ArrayLike, b: ArrayLike, x: ArrayLike
) -> FloatArray:
    """Return ``(x*b)*a - (x*a)*b``."""

    return multiply(multiply(x, b), a) - multiply(multiply(x, a), b)


def left_associated_power(value: ArrayLike, exponent: int) -> FloatArray:
    """Return a simple repeated-product reference for nonnegative powers."""

    base = _coefficients(value)
    if exponent < 0:
        raise ValueError("reference exponent must be nonnegative")
    result = np.zeros_like(base)
    result[0] = 1.0
    for _ in range(exponent):
        result = multiply(result, base)
    return result


def as_tuples(values: Sequence[ArrayLike]) -> tuple[tuple[float, ...], ...]:
    """Return coefficient records as immutable tuples for test diagnostics."""

    return tuple(tuple(float(x) for x in _coefficients(value)) for value in values)
