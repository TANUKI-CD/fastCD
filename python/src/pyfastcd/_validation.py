"""Private validation helpers for :mod:`pyfastcd`."""

from __future__ import annotations

import operator
import struct
import sys
from typing import Any

# Messages shared verbatim with the C extension (python/src/pyfastcd/_core.c)
# so that both layers report identical conditions identically.
DIFFERENT_ALGEBRA_MESSAGE = "operands belong to different algebras"
COMPLEX_SCALAR_MESSAGE = "complex scalars are not supported"
MODULAR_POWER_MESSAGE = (
    "modular exponentiation is not defined for Cayley--Dickson elements"
)
INTEGER_EXPONENT_MESSAGE = "Cayley--Dickson powers require an integer exponent"

_MAX_ELEMENT_DIMENSION = sys.maxsize // struct.calcsize("d")


def _index(value: Any, name: str) -> int:
    """Return *value* as a Python integer suitable for an index."""

    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


def _dimension_and_level(
    *, dimension: Any | None, level: Any | None
) -> tuple[int, int]:
    """Validate an exclusive dimension/level specification."""

    if (dimension is None) == (level is None):
        raise TypeError("specify exactly one of dimension or level")
    if level is not None:
        normalized_level = _index(level, "level")
        if normalized_level < 0:
            raise ValueError("level must be nonnegative")
        if normalized_level >= sys.maxsize.bit_length():
            raise OverflowError("level is too large for a Python element")
        normalized_dimension = 1 << normalized_level
        if normalized_dimension > _MAX_ELEMENT_DIMENSION:
            raise OverflowError("level is too large for Python Element storage")
        return normalized_dimension, normalized_level

    normalized_dimension = _index(dimension, "dimension")
    if normalized_dimension <= 0 or (
        normalized_dimension & (normalized_dimension - 1)
    ):
        raise ValueError("dimension must be a positive power of two")
    if normalized_dimension > _MAX_ELEMENT_DIMENSION:
        raise OverflowError("dimension is too large for Python Element storage")
    return normalized_dimension, normalized_dimension.bit_length() - 1


def _basis_index(value: Any, dimension: int) -> int:
    """Validate a standard-basis index."""

    normalized = _index(value, "basis index")
    if normalized < 0 or normalized >= dimension:
        raise IndexError("basis index is out of range")
    return normalized
