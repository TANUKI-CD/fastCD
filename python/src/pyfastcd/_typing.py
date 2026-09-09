"""Private type aliases shared by the Python and native interfaces.

The concrete scalar union intentionally avoids :class:`typing.SupportsFloat`:
NumPy's complex scalar types satisfy that protocol even though fastCD rejects
complex operands.  The ndarray arm expresses the supported real dtypes, but
its zero-dimensional runtime requirement cannot be represented reliably by
current NumPy factory return annotations.
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from typing import Any, TypeAlias

import numpy as np

_RealNumericDType: TypeAlias = np.bool_ | np.integer[Any] | np.floating[Any]
_RealScalarArray: TypeAlias = np.ndarray[
    tuple[()], np.dtype[_RealNumericDType]
]
_RealScalar: TypeAlias = (
    bool
    | int
    | float
    | Decimal
    | Fraction
    | np.bool_
    | np.integer[Any]
    | np.floating[Any]
    | _RealScalarArray
)
