# Python quick start

pyfastcd is the Python interface to fastCD. It represents an element of the
Cayley-Dickson algebra of dimension `N = 2^n` as an array of `N` coefficients,
real by default, whose coordinate 0 holds the scalar part, and it computes
products and the other operations of the algebra in compiled code.

This guide introduces the basics of how to use pyfastcd one idea at a time, from
installation and the products of basis elements through general elements,
batches, and other coefficient fields to sums of products. Each example builds
on the examples that come before it, and shows its results in comments. Users of
pyfastcd are recommended to start here and read this guide in its entirety,
running the examples for themselves before using the package in production. The
[Python API reference](python-api.md) gives the complete rules for everything
shown here and should be read for a more detailed reference if more information
is needed than what this quick start guide can provide.

## Installation

pyfastcd requires CPython 3.10 or newer and NumPy 1.26 or newer. To install it
from PyPI, run:

```sh
pip install pyfastcd
```

To build and install it from a source checkout instead, you also need a C11
compiler, the Python development headers, and CMake 3.18 or newer. Run the
following command from the repository root, with the Python interpreter that
will run your code:

```sh
python -m pip install .
```

## Basis elements and their products

An `Algebra` fixes the dimension, and its `basis` method returns the standard
basis elements `e_0 = 1, e_1, ..., e_(N-1)`. The product of two basis elements
is again a basis element, up to sign, so multiplying them is the quickest way to
see how Cayley-Dickson multiplication behaves:

```python
import numpy as np
import pyfastcd as fcd

O = fcd.Algebra(dimension=8)        # the octonions
e1, e2, e4 = O.basis(1), O.basis(2), O.basis(4)

e1 * e2                             # e_3
e2 * e1                             # -e_3
e1 * e1                             # -1
(e1 * e2) * e4                      # e_7
e1 * (e2 * e4)                      # -e_7
O.basis_product(1, 2)               # BasisProduct(index=3, sign=1)
```

The product is not commutative, since `e_1 e_2` and `e_2 e_1` differ in sign,
and every basis element other than `e_0` squares to `-1`. From dimension 8 on
the product is not associative either, so `(e_1 e_2) e_4` and `e_1 (e_2 e_4)`
differ, and parentheses matter in every product of three or more factors.
`basis_product` returns the index and the sign of a basis product without
forming any element, here `e_1 e_2 = +e_3`, and [Basis
products](mathematics.md#basis-products) gives the rule that produces these
signs.

## General elements

A general element is a linear combination of basis elements. Calling the algebra
with exactly `N` coefficients creates one, and calling it with a scalar creates
that multiple of `e_0`:

```python
a = O([1, 2, 0, 0, 0, 0, 0, 0])     # 1 + 2 e_1
three = O(3)                        # 3
one = O.one()                       # 1
i = fcd.cd_element([0, 1])          # e_1, dimension 2 from the length

a.coefficients                      # array([1., 2., 0., 0., 0., 0., 0., 0.])
a[1]                                # 2.0
```

An `Element` is immutable and hashable, so it can serve as a dictionary key or a
set member. Its coefficients can be read, through indexing or through the
read-only NumPy view `coefficients`, but not changed; an operation on elements
always returns a new element.

## Arithmetic and structure

Elements combine with the usual operators, and a scalar combines with an element
as the corresponding multiple of `e_0`:

```python
b = O.basis(2) + 3                  # 3 + e_2

a * b                               # 3 + 6 e_1 + e_2 + 2 e_3
a + b                               # 4 + 2 e_1 + e_2
0.5 * a                             # 0.5 + e_1
a - 1                               # 2 e_1
a / 2                               # 0.5 + e_1
a ** 2                              # -3 + 4 e_1
a.square()                          # -3 + 4 e_1, computed with O(N) operations

a.conjugate()                       # 1 - 2 e_1
a.inner(b)                          # 3.0
a.norm()                            # 2.23606797749979, that is, sqrt(5)
a.reciprocal()                      # 0.2 - 0.4 e_1
a * a.reciprocal()                  # 1
```

The conjugate negates every coordinate except the scalar one, `inner` is the
inner product of the coordinate vectors, and the reciprocal is
`conjugate(a) / Q(a)`, where `Q(a)` is the squared norm. Because the octonions
are alternative, the reciprocal undoes multiplication even without
associativity: `a.reciprocal() * (a * x)` equals `x` up to rounding for every
octonion `x`, so it solves equations `a x = c`. From dimension 16 on this can
fail, because the algebras have zero divisors, nonzero elements whose product is
zero:

```python
x = O([0, 1, 2, 3, 4, 5, 6, 7])
np.allclose(a.reciprocal() * (a * x), x)    # True

S = fcd.Algebra(dimension=16)               # the sedenions
u = S.basis(1) + S.basis(10)
v = S.basis(4) - S.basis(15)
u * v                                       # 0, although neither factor is zero
```

Here `u.reciprocal() * (u * v)` is zero rather than `v`, so the reciprocal no
longer solves equations in general.

## Batches

An `ElementArray` holds a batch of elements in one NumPy array, whose last axis
holds the coefficients of each element and whose leading axes follow NumPy's
broadcasting rules:

```python
rng = np.random.default_rng(1234)
X = O.asarray(rng.normal(size=(1000, 8)))   # 1000 octonions, storage shared
Y = O.array(rng.normal(size=(1, 8)))        # one octonion, copied
Z = X * Y                                   # 1000 products, of shape (1000, 8)
Z.norm().shape                              # (1000,)
Z[0]                                        # an Element copy of record 0
Z[1] = one                                  # assign one record

out = np.empty((1000, 8))
O.multiply(X, Y, out=out)                   # writes the products into out
```

Since `Y` has a leading axis of length 1, it multiplies every element of `X`.
`asarray` shares the storage of a compatible array instead of copying it, while
`array` always copies, and the operations on a batch run in compiled code over
the whole array. Methods such as `norm`, `conjugate`, and `commutator` work on
arrays as they do on elements, and `out` lets a product reuse existing storage,
provided it has exactly the shape and dtype of the result; [Return values and
output storage](python-api.md#return-values-and-output-storage) gives the
remaining conditions.

## Choosing the multiplication algorithm

fastCD has three multiplication algorithms, direct, Cariow-Cariowa, and
quasilinear, and `*` uses the default for the dimension, which can differ
between single products and batches:

```python
O.default_algorithm                 # 'direct', for single products
O.default_batch_algorithm           # 'direct', for batches
p = O.multiply(a, b, algorithm="quasilinear")
q = O.multiply(a, b, algorithm="cariow_cariowa")
p == q                              # True
```

The defaults come from benchmarks of single products and of batches, which [How
the results set the defaults](benchmark.md#how-the-results-set-the-defaults)
reports. `Algebra.multiply` takes the algorithm explicitly: `"auto"`,
`"direct"`, `"quasilinear"`, or, for real coefficients only and through
dimension 1024, `"cariow_cariowa"`. Every algorithm computes the same product,
as the last line shows for these small integer coordinates, but in floating
point their results can differ in the last bits. A name that is unavailable for
the field or the dimension raises `ValueError`.

## Other coefficient fields

Everything above uses real coefficients, which an algebra has unless its
coefficient field is selected explicitly. `char=0, deg=2` selects complex
coefficients:

```python
C = fcd.Algebra(dimension=4, char=0, deg=2)  # complex quaternions
w = C([1 + 2j, 3j, 4, 0])                   # (1 + 2i) + 3i e_1 + 4 e_2
w.quadratic_norm()                          # (4+4j)
w.conjugate()                               # (1 + 2i) - 3i e_1 - 4 e_2
```

Neither the quadratic form `Q(w) = sum w_k^2` nor the Cayley-Dickson conjugate
conjugates the complex coefficients, so over C the form can vanish on a nonzero
element, which then has no reciprocal.

`char=p` selects a finite field of odd characteristic `p`, and `deg=d` its
extension of degree `d`, whose elements are polynomials in `t` modulo an
irreducible polynomial that pyfastcd chooses. A `FiniteField` object represents
the field itself and creates its scalars:

```python
F = fcd.FiniteField(char=13, deg=5)          # the field of 13^5 elements
A = fcd.Algebra(dimension=4, field=F)        # also char=13, deg=5
c = F([1, 2, 0, 0, 0])                      # 1 + 2t
g = A.scalar(c) + A.basis(1)                # (1 + 2t) + e_1
g.coefficients.shape                        # (4, 5)
g.quadratic_norm()                          # 2 + 4t + 4t^2, a scalar of F
g * g.conjugate() == A.scalar(g.quadratic_norm())   # True
```

Each finite-field coefficient is stored as its `d` polynomial coordinates, so an
element of this algebra is a `(4, 5)` array of `uint32` residues, and its
arithmetic is exact. `quadratic_norm()` returns a scalar of the coefficient
field, a complex number over C and a field element over a finite field, whereas
`norm()`, `abs()`, and `normalized()` exist only over R, because a finite field
has no ordering or positive square root and fastCD provides no Hermitian norm
over C.

## Operator actions

Some operations act on an element by multiplying it on one side, either by a
basis element or twice in succession:

```python
O.left_basis_multiply(2, a)                 # e_2 a = e_2 - 2 e_3
O.right_basis_multiply(a, 2)                # a e_2 = e_2 + 2 e_3
O.left_operator_commutator(e1, e2, e4)      # -2 e_7
(e1 * e2 - e2 * e1) * e4                    # 2 e_7
```

Multiplying by a basis element only permutes the coordinates of an element and
changes some of their signs, which `left_basis_multiply` and
`right_basis_multiply` do without a full product. The operator commutator
`a (b x) - b (a x)` compares left multiplication by `b` and then `a` with the
reverse order; in an associative algebra it equals `(ab - ba) x`, but in the
octonions the two differ, as the last two lines show.
`right_operator_commutator` computes `(x b) a - (x a) b` in the same way.

## Sums of products and covariance

Sums of products and covariances are available for real coefficients only, and
they are experimental features of fastCD: they may not be fully optimized or
ready for production use, and the code implementing them may change over time,
but their API is expected to remain the same. Both return one element computed
from two samples of elements:

```python
s = O.asarray(rng.normal(size=(1000, 8)))
t = O.asarray(rng.normal(size=(1000, 8)))

total = O.multiply_sum(s, t)        # s[0] * t[0] + ... + s[999] * t[999]
cov = O.covariance(s, t)            # covariance, divisor 999
```

`multiply_sum` returns the sum of the products of corresponding elements without
storing the products, because it accumulates the multiplications of the
algorithm over all the pairs and applies the rest of the algorithm once (see
[Fused sums of products](mathematics.md#fused-sums-of-products)); its result can
therefore differ in the last bits from the products added up one by one.
`covariance` returns `sum((s[p] - mean(s)) * conjugate(t[p] - mean(t)))` divided
by `count - ddof`, with `ddof=1` by default, and the scalar coordinate of the
result is the sum of the covariances of the coordinates.

At dimension 2 an element `[re, im]` is a complex number, and `covariance` is
the ordinary covariance of complex samples:

```python
zs = rng.normal(size=500) + 1j * rng.normal(size=500)
ws = rng.normal(size=500) + 1j * rng.normal(size=500)
D = fcd.Algebra(dimension=2)
cz = D.covariance(np.column_stack((zs.real, zs.imag)),
                  np.column_stack((ws.real, ws.imag)))
complex(cz[0], cz[1])               # np.cov(zs, ws)[0, 1], up to rounding
```

## Working with NumPy

Elements and arrays work with the common NumPy functions, which give them the
meaning of the operations of the algebra:

```python
np.multiply(a, b) == a * b          # True
np.conjugate(Z).shape               # (1000, 8)
np.asarray(a)                       # read-only array([1., 2., 0., ...])
Z.coefficients.mean(axis=0)         # the mean of each coordinate over the batch
```

NumPy ufuncs such as `np.add`, `np.multiply`, and `np.conjugate` act on elements
and arrays as the corresponding operations of the algebra, while NumPy functions
applied to the coefficient arrays, such as `Z.coefficients.mean(axis=0)`, act on
individual coordinates as usual. Arrays can also be pickled, copied, and
exchanged with other array libraries through DLPack; [Arrays, NumPy, and
DLPack](python-api.md#arrays-numpy-and-dlpack) lists these interfaces.

## Where to go next

- The [Python API reference](python-api.md) documents every class, method, and
  rule.
- [Conventions and results](mathematics.md) describes the algebra, the
  multiplication algorithms, and their operation counts.
- The [benchmark](benchmark.md) and the [stability study](stability.md) report
  the speed and the accuracy of the algorithms.
