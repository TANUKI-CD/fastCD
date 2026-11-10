# Python quick start

To install `pyfastcd` from the source checkout, use the Python interpreter
that will run your code. You will need Python 3.10 or newer, NumPy 1.26 or
newer, a C11 compiler, Python development headers, and CMake 3.18 or newer.
Run the following command:

```sh
python -m pip install .
```

## Algebras and elements

```python
import numpy as np
import pyfastcd as fcd

O = fcd.Algebra(dimension=8)        # octonions; same object as Algebra(level=3)
a = O([1, 2, 0, 0, 0, 0, 0, 0])     # exactly N coefficients, copied to float64
b = O.basis(2)                      # e_2
one = O.one()
three = O(3)                        # the scalar 3 * e_0
i = fcd.cd_element([0, 1])          # dimension inferred from the length

product = a * b                     # ordered Cayley–Dickson product
scaled = 0.5 * a
shifted = a - 3                     # subtract 3 * e_0
quotient = a / 2                    # real scalar division only
power = a ** 5                      # ** is exponentiation; ^ is not
square = a.square()                 # O(N), from the scalar and imaginary parts

conjugate = a.conjugate()
dot = a.inner(b)
norm = a.norm()                     # also abs(a)
unit = a.normalized()
inverse = a.reciprocal()            # conjugate(a) / Q(a)
comm = a.commutator(b)

e1, e4 = O.basis(1), O.basis(4)
assoc = e1.associator(b, e4)        # (e_1 e_2) e_4 - e_1 (e_2 e_4) = 2 e_7
```

An `Element` is immutable and hashable, so it can be a dictionary key or a
set member. You can read its coefficients as a NumPy view through
`a.coefficients` or `np.asarray(a)`, but you cannot change them.

From dimension 8 onward multiplication is not associative, so parentheses
matter, as the nonzero associator above shows. The octonions are still
alternative, so `a * (a.reciprocal() * b)` recovers `b` up to rounding. From
dimension 16 onward this can fail, and `reciprocal()` cannot be used to
solve general equations `a x = b`.

## Batches

For a batch, the last axis holds the coefficients of each element, while the
leading axes follow NumPy's broadcasting rules.

```python
rng = np.random.default_rng(1234)
x = O.asarray(rng.normal(size=(1000, 8)))   # shares compatible float64 storage
y = O.array(rng.normal(size=(1, 8)))        # always copies
z = x * y                                   # shape (1000, 8)
norms = z.norm()                            # shape (1000,)
first = z[0]                                # an Element copy
z[1] = one                                  # assign one element

out = np.empty((1000, 8))
result = O.multiply(x, y, out=out)          # reuse output storage
assert np.shares_memory(result.coefficients, out)
```

The array passed as `out` must have the exact shape of the result and use
native `float64` coefficients. Its storage must be writable, aligned, and
C-contiguous, and it must not overlap an input. Methods such as
`conjugate`, `square`, and `commutator` also work on arrays and allocate
storage for their results.

## Choosing the multiplication algorithm

Ordinary multiplication uses the default algorithm for the dimension, chosen
from the library benchmark. To choose an algorithm yourself, call
`Algebra.multiply`:

```python
print(O.default_algorithm)        # single products: 'direct' at dimension 8
print(O.default_batch_algorithm)  # batches of two or more: 'direct' here too
p1 = O.multiply(a, b, algorithm="direct")
p2 = O.multiply(a, b, algorithm="cariow_cariowa")  # dims 1..1024; slow above 64
p3 = O.multiply(x, y, algorithm="quasilinear", out=out)
```

The accepted names are `"auto"`, `"direct"`, `"cariow_cariowa"`, and
`"quasilinear"`; `"cariow_cariowa"` is available only for real
coefficients. Asking for an algorithm that is unavailable for the field or
the dimension raises a `ValueError`.

## Basis and operator actions

```python
p = O.basis_product(1, 2)                  # BasisProduct(index=3, sign=1)
assert O.basis(1) * O.basis(2) == p.sign * O.basis(p.index)
left = O.left_basis_multiply(2, a)         # e_2 * a
right = O.right_basis_multiply(a, 2)       # a * e_2
lop = O.left_operator_commutator(e1, b, e4)   # e_1 (e_2 e_4) - e_2 (e_1 e_4)
rop = O.right_operator_commutator(e1, b, e4)  # (e_4 e_2) e_1 - (e_4 e_1) e_2
```

Here `lop` is `-2 e_7`, while `(e_1 e_2 - e_2 e_1) e_4` is `2 e_7`: in a
nonassociative algebra, the operator commutator `a (b x) - b (a x)` is not
left multiplication by `ab - ba`.

## NumPy interoperability

`Element` and `ElementArray` work with the common NumPy ufuncs, such as
`np.add`, `np.multiply`, and `np.conjugate`; the
[API reference](python-api.md#numpy-interfaces) lists them. `np.multiply`
also accepts `out=` and returns the ndarray you passed. NumPy arithmetic on
the coefficient arrays themselves, such as `a.coefficients`, acts on
individual coordinates as usual.

An `Element` exposes a read-only buffer, so you can use `memoryview(a)`.
`ElementArray` also supports `__array__`, DLPack export and import from
host-accessible producers, pickling, and `copy`. See the
[Python API reference](python-api.md) for more information on these
interfaces.

## Explicit coefficient fields

Everything above used real coefficients. For complex coefficients or an
odd-characteristic finite field, select the coefficient field explicitly:

```python
C = fcd.Algebra(dimension=4, char=0, deg=2)
x = C([1 + 2j, 3j, 4, 0])
q = x.quadratic_norm()        # complex, sum(x_i**2)

F = fcd.FiniteField(char=13, deg=5)
A = fcd.Algebra(dimension=4, field=F)
a = A.scalar(F([1, 2, 0, 0, 0])) + A.basis(1)
b = A.basis(2)
c = A.multiply(a, b, algorithm="quasilinear")
assert a * a.conjugate() == A.scalar(a.quadratic_norm())
```

Equivalently, `Algebra(dimension=4, char=13, deg=5)` uses the default
irreducible polynomial for F_(13^5). The Cayley–Dickson dimension and the
extension degree are independent: an element of this algebra is stored as a
`(4, 5)` array of `uint32` residues, and only leading batch axes broadcast.
`quadratic_norm()` returns a field element; `norm()`, `abs()` and
`normalized()` are defined only over R. Named methods such as `inner` and
`commutator` require complete elements, so write `a.inner(A.scalar(2))`
rather than `a.inner(2)`. See the [API reference](python-api.md#finitefield)
for polynomial coordinates, copy rules and field-scalar arrays.
