"""Static API regressions; checked by mypy, never executed by pytest."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from pyfastcd import Algebra, Element, ElementArray, FiniteField, cd_element
from typing_extensions import assert_type

H = Algebra(dimension=4)
x = H([1, 2, 3, 4])
assert_type(H, Algebra)
assert_type(x, Element)
assert_type(x.coefficients, NDArray[np.float64])
assert_type(x.real, float)
assert_type(x.shape, tuple[int])
assert_type(x.__reduce__(), tuple[Algebra, tuple[tuple[float, ...]]])
assert_type(x.inner(x), float)
assert_type(x.norm_squared(), float)
assert_type(x.quadratic_norm(), float)
assert_type(x.norm(), float)
assert_type(x.algebra, Algebra)
assert_type(x.to_numpy(copy=False), NDArray[np.float64])
a = H.array([[1, 2, 3, 4]])
assert_type(a, ElementArray)
assert_type(a.real, NDArray[np.float64])
assert_type(a.norm_squared(), NDArray[np.float64])
assert_type(cd_element([1, 2]), Element)

C = Algebra(dimension=4, char=0, deg=2)
z = C([1j, 2, 3, 4])
assert_type(z.coefficients, NDArray[np.complex128])
assert_type(z.real, complex)
assert_type(z.inner(z), complex)
assert_type(z.quadratic_norm(), complex)
assert_type(C.array([[1j, 2, 3, 4]]).quadratic_norm(), NDArray[np.complex128])
assert_type(cd_element(z).quadratic_norm(), complex)

F = FiniteField(char=13, deg=5)
s = F.one()
assert_type(s.coefficients, NDArray[np.uint32])
assert_type(s.algebra, FiniteField)
A = Algebra(dimension=4, field=F)
f = A.one()
assert_type(f.coefficients, NDArray[np.uint32])
assert_type(f.quadratic_norm(), Element[np.uint32, Any, FiniteField, tuple[int, ...]])
assert_type(A.array([[s.coefficients] * 4]).quadratic_norm().algebra, FiniteField)

assert_type(f.shape, tuple[int, ...])

assert_type((f * s).coefficients, NDArray[np.uint32])
