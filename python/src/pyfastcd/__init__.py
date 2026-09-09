"""Fast, matrix-free arithmetic over the real Cayley--Dickson tower.

``pyfastcd`` provides immutable scalar elements and NumPy-backed element
arrays with native ``float64`` coefficients.  Arithmetic is delegated to the
compiled fastCD kernels; the Python API does not construct multiplication or
operator matrices.

The conventional abbreviation is ``fcd``::

    import pyfastcd as fcd

    H = fcd.Algebra(dimension=4)
    a = H([1, 2, 3, 4])
    b = a ** 2

Use ``**`` for powers.  In ordinary Python, ``^`` is bitwise XOR and is not
overridden as exponentiation.

Public API
----------
Algebra
    Immutable, dimension-cached parent and primary constructor.
Element
    Immutable scalar element backed directly by native coefficient storage.
ElementArray
    NumPy-backed batch with leading-axis broadcasting.
BasisProduct
    Signed standard-basis multiplication result.
cd_element
    Convenience constructor that infers a power-of-two dimension.

Notes
-----
The package intentionally exposes mathematical values rather than native
status codes or workspace buffers.  All coefficients are real ``float64``;
complex coefficient arrays and arbitrary NumPy object arrays are unsupported.
"""

from ._algebra import Algebra, BasisProduct, cd_element
from ._array import ElementArray
from ._core import Element, _version

__version__ = _version()

__all__ = [
    "__version__",
    "Algebra",
    "BasisProduct",
    "Element",
    "ElementArray",
    "cd_element",
]
