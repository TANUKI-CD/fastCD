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

O = fcd.Algebra(dimension=8)        # octonions; fcd.Algebra(level=3) is the same object
a = O([1, 2, 0, 0, 0, 0, 0, 0])     # exactly N coefficients, copied to float64
b = O.basis(2)                      # e_2
one = O.one()
three = O(3)                        # the scalar 3 * e_0
i = fcd.cd_element([0, 1])          # dimension inferred from the length

product = a * b                     # ordered Cayley-Dickson product
scaled = 0.5 * a
shifted = a - 3                     # subtract 3 * e_0
quotient = a / 2                    # real scalar division only
power = a ** 5                      # ** is exponentiation; ^ is not
square = a.square()                 # linear-time specialized square

conjugate = a.conjugate()
dot = a.inner(b)
norm = a.norm()                     # also abs(a)
unit = a.normalized()
inverse = a.reciprocal()            # conjugate(a) / <a, a>
comm = a.commutator(b)
assoc = a.associator(b, one)        # (a b) one - a (b one)
```

An `Element` is immutable and hashable, so it can be a dictionary key or a set
member. You can read its coefficients
as a NumPy view through `a.coefficients` or `np.asarray(a)`, but you cannot
change them. From dimension 8 onward, parentheses matter when multiplying
elements. At dimension 8 the octonions are still alternative, so
`a * (a.reciprocal() * b)` recovers `b` up to rounding, but from dimension 16
onward `reciprocal()` cannot be used to solve general equations `a x = b`.

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
z[1] = one                                  # assign a record

out = np.empty((1000, 8))
result = O.multiply(x, y, out=out)          # reuse output storage
assert np.shares_memory(result.coefficients, out)
```

The array passed as `out` must have the exact shape of the result and use
native `float64` coefficients. Its storage must be writable and C-contiguous,
and it must not overlap an input. Methods such as `conjugate`, `square`, and
`commutator` also work on arrays and allocate storage for their results.

## Choosing the multiplication algorithm

Ordinary multiplication uses the default algorithm selected for the
dimension based on the library benchmark. To choose an algorithm yourself,
call `Algebra.multiply`:

```python
print(O.default_algorithm)                       # single products, e.g. 'direct'
print(O.default_batch_algorithm)                 # batches of two or more records
p1 = O.multiply(a, b, algorithm="direct")
p2 = O.multiply(a, b, algorithm="cariow_cariowa") # dimensions 1..1024; slow above 64
p3 = O.multiply(x, y, algorithm="quasilinear", out=out)
```

The accepted names are `"auto"`, `"direct"`, `"cariow_cariowa"`, and
`"quasilinear"`. Asking for an algorithm that does not support the dimension
raises a `ValueError`.

## Basis and operator actions

```python
p = O.basis_product(1, 2)                   # BasisProduct(index=3, sign=1)
assert O.basis(1) * O.basis(2) == p.sign * O.basis(p.index)
left = O.left_basis_multiply(2, a)          # e_2 * a
right = O.right_basis_multiply(a, 2)        # a * e_2
lop = O.left_operator_commutator(a, b, one) # a (b one) - b (a one)
rop = O.right_operator_commutator(a, b, one)
```

## NumPy interoperability

Both classes support `np.add`, `np.subtract`, `np.multiply`, `np.true_divide`
by a real scalar, `np.power` with an integer exponent, `np.negative`,
`np.positive`, `np.square`, `np.conjugate`, `np.absolute`, `np.equal`, and
`np.not_equal`, and `np.multiply` also accepts `out=`, returning the ndarray
you passed. If
you apply NumPy arithmetic directly
to the coefficient arrays, it acts on individual coordinates as usual.

An `Element` exposes a read-only buffer, so you can use `memoryview(a)`.
`ElementArray` also supports `__array__`, DLPack export and import from
host-accessible producers, pickling, and `copy`. See the
[Python API reference](python-api.md) for more information on these interfaces.
