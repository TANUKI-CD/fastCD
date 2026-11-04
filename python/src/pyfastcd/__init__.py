"""Fast, matrix-free Cayley--Dickson arithmetic over R, C and finite fields.

``pyfastcd`` provides immutable scalar elements and NumPy-backed element
arrays. Coefficients are real ``float64`` by default; ``char`` and ``deg``
select complex or finite-field coefficients.  Arithmetic is delegated to the
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
FiniteField
    Immutable finite coefficient field in a recorded polynomial basis.
BasisProduct
    Signed standard-basis multiplication result.
cd_element
    Convenience constructor that infers a power-of-two dimension.

Notes
-----
The package intentionally exposes mathematical values rather than native
status codes or workspace buffers. CD conjugation fixes coefficients;
``quadratic_norm()`` returns their canonical field-valued quadratic form.
Real parents reject complex coefficients.
"""

from ._algebra import Algebra, BasisProduct, cd_element
from ._array import ElementArray
from ._core import Element, _version
from ._field import FiniteField

__version__ = _version()

__all__ = [
    "__version__",
    "Algebra",
    "BasisProduct",
    "Element",
    "ElementArray",
    "cd_element",
    "FiniteField",
]
