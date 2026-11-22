# Python API reference

pyfastcd is the Python interface to fastCD. It provides NumPy-compatible classes
for Cayley-Dickson elements with real, complex, or finite-field coefficients,
whose operations call the compiled C library; errors are reported as Python
exceptions, and the workspaces that the C library needs are managed internally,
so Python code never allocates them.

This page documents every public name of pyfastcd and the rules that govern
them, while the [Python quick start](quickstart-python.md) introduces them by
example and is the place to begin. The public names are `Algebra`,
`BasisProduct`, `Element`, `ElementArray`, `FiniteField`, `cd_element`, and
`__version__`, and modules whose names begin with an underscore are private.
After the dependencies and the terms used throughout, the page documents the
module functions, then the classes in the order a program meets them, `Algebra`,
`FiniteField`, `BasisProduct`, and `Element` and `ElementArray`, and finally the
storage of arrays and their NumPy and DLPack interfaces, exceptions, threads,
versioning, and typing.

---

## Dependencies

pyfastcd depends on:

- CPython 3.10 or newer;
- NumPy 1.26 or newer, or NumPy 2.1 or newer to import a generic DLPack producer
  with `copy=False`.

The compiled extension contains the fastCD C library, so pyfastcd needs no
separate installation of fastCD.

---

## Terms and notation

The rest of this page uses the following terms.

**Coefficient field and selectors.** Every algebra has its coefficients in R, C,
or a finite field F_(p^d) (see [The algebra](mathematics.md#the-algebra)). Three
keyword arguments, the *selectors* `char`, `deg`, and `field`, choose the field:

| Selectors | Coefficients | `dtype` | `coefficient_shape` |
|---|---|---|---|
| omitted, or `char=0, deg=1` | R | `float64` | `(N,)` |
| `char=0, deg=2` | C | `complex128` | `(N,)` |
| `char=p, deg=1` | F_p | `uint32` | `(N,)` |
| `char=p, deg=d`, `d > 1` | F_(p^d) | `uint32` | `(N, d)` |
| `field=F` | the `FiniteField` `F` | `uint32` | `(N,)` or `(N, F.deg)` |

- `char` is `0` or an odd prime at most `2**31-1`.
- `deg` is `1` (R) or `2` (C) when `char=0`, and `1` to `12` for a finite field.
  It is the degree of the coefficient field and is independent of the
  Cayley-Dickson `level`.
- `field` is a `FiniteField`, passed on its own: combining it with a nondefault
  `char` or `deg` raises `TypeError`.
- Characteristic 2, negative or composite characteristics, and unsupported
  degrees raise `ValueError`. Boolean values, including `numpy.bool_`, are
  rejected for `char` and `deg`, as they are for `dimension` and `level`.

Without selectors the coefficients are real `float64` values, so complex and
finite-field coefficients are always selected explicitly.

**Parent.** Every element and array belongs to a *parent*: the `Algebra` that
created it, or the `FiniteField` of a coefficient-field scalar, which
`x.algebra` returns. Operations combine only objects with the same parent, with
one exception: over a finite field, scalars and scalar arrays of the coefficient
field can scale and divide the elements and arrays of the algebra. Nothing
converts between parents implicitly.

**Record.** A *record* is the coefficient storage of one Cayley-Dickson element,
and its shape is the parent's `coefficient_shape`: `N` coefficients, followed
over F_(p^d) by an axis of `d` polynomial coordinates. An array of elements has
shape `batch_shape + coefficient_shape`, where the *batch shape* consists of the
leading axes.

**Scalars.** A *coefficient-field scalar* is a value of the coefficient field.
For a real parent it may be a `bool`, `int`, `float`, `Decimal`, `Fraction`,
real NumPy scalar, or real zero-dimensional ndarray, while complex values, text,
and arrays with an object dtype are rejected; an object with a custom
`__float__` method can also be used in construction and arithmetic. Over C a
scalar may also be complex. Over a finite field a scalar is an integer, which is
reduced modulo `char` however large it is, or a scalar of the same field, and
floats are rejected even when they are integral. Construction and arithmetic
round real scalars to binary64, although equality compares exact values (see
[Equality and hashing](#equality-and-hashing)).

Below, `A` is an algebra, `N = A.dimension`, `x`, `y`, and `z` are elements, `X`
is an element array, and `r` is a coefficient-field scalar.

---

## Module contents

- `__version__: str`

  The version of pyfastcd, which is also the version of the C library it
  contains.

- `cd_element(coefficients, *, char=0, deg=1, field=None) -> Element`

  Construct one element from a single record, inferring `N` from its length.

  - **Parameters:** `coefficients` is an ordered sequence or array, not an
    iterator, with the shape given in the selector table and a length that is a
    positive power of two. The selectors choose the coefficient field as for
    `Algebra`.
  - **Returns:** a new `Element` of the algebra of dimension `N` over that
    field.
  - **Notes:** An `Element` argument is copied and keeps its parent, and
    selectors, if given, must describe that parent, so that other selectors
    raise `ValueError`. An invalid shape or length raises `ValueError`, and for
    a real parent, complex, object-valued, or nonnumeric arrays raise
    `TypeError`.

---

## Algebra

### Constructor

```python
Algebra(*, dimension=None, level=None, char=0, deg=1, field=None)
```

Create the algebra of dimension `N = 2**level` over the coefficient field that
the selectors choose. Exactly one of `dimension`, a positive power of two, and
`level`, the doubling level, must be supplied.

Algebras are immutable, cannot be subclassed, and are cached by dimension and
coefficient field, so `Algebra(level=3) is Algebra(dimension=8)`.

The selectors make a difference that is easy to overlook at dimension 2:
`Algebra(dimension=2)` is the complex numbers, built from two **real**
coefficients, whereas `Algebra(dimension=2, char=0, deg=2)` has two **complex**
coefficients and is a different algebra, isomorphic to the product C x C.

### Properties

- `dimension: int`

  The number of coefficients, `N`.

- `level: int`

  The doubling level, `log2(N)`.

- `dtype`

  The storage dtype from the selector table.

- `char: int`, `deg: int`

  The characteristic and the degree of the coefficient field.

- `field: FiniteField | None`

  The finite coefficient field, or `None` over R and C.

- `coefficient_shape: tuple[int, ...]`

  The shape of one record.

- `default_algorithm: str`

  The algorithm that `"auto"` uses for a single product, that is, for two
  `Element` operands or a batch of one record. For real coefficients this is the
  choice of the C library's `fastcd_default_algorithm`, and over C and finite
  fields it is `"direct"` for `N <= 4` and `"quasilinear"` otherwise; [How the
  results set the defaults](benchmark.md#how-the-results-set-the-defaults)
  explains these choices.

- `default_batch_algorithm: str`

  The algorithm that `"auto"` uses for batches of two or more records: the C
  library's `fastcd_default_batch_algorithm` for real coefficients, and the same
  choice as `default_algorithm` over C and finite fields.

- `default_sum_algorithm: str`

  The algorithm that `"auto"` uses in `multiply_sum`: `"direct"` through `N = 8`
  and `"quasilinear"` above. Real coefficients only.

### Methods

The methods fall into four groups: constructing elements and arrays,
multiplying, summing products, and acting with basis elements and operators.
Every method that accepts `out=` follows the rules of [Return values and output
storage](#return-values-and-output-storage).

- `A(value=0) -> Element`

  Create an element of `A`.

  - **Parameters:** `value` is a coefficient-field scalar, which is placed in
    coordinate zero; an `Element` with parent `A`, which is copied; or exactly
    `N` coefficients from a sequence, iterator, or buffer.
  - **Returns:** a new `Element` with parent `A`.
  - **Notes:** Coefficients are never padded or truncated, and unordered inputs
    such as `set` and `dict` raise `TypeError`. Each coefficient follows the
    rules for [scalars](#terms-and-notation): over C it may be complex, and over
    a finite field it is an integer, reduced modulo `char` before conversion to
    `uint32`, or a scalar of the same field.

- `scalar(r) -> Element`

  Return the scalar element `r * e_0`.

- `zero() -> Element`

  Return the additive identity.

- `one() -> Element`

  Return the multiplicative identity.

- `basis(index) -> Element`

  Return `e_index`, raising `IndexError` if the index is outside `range(N)`.

- `basis_product(left_index, right_index) -> BasisProduct`

  Return the `BasisProduct` that describes `e_i e_j`, where `i` is `left_index`
  and `j` is `right_index`.

- `array(values) -> ElementArray`

  Copy `values` into new writable, aligned, C-contiguous storage with dtype
  `A.dtype`.

- `asarray(values, *, copy=None) -> ElementArray`

  Create an array that shares the storage of `values` where possible.

  - **Parameters:** `values` is array-like with the shape
    `batch_shape + coefficient_shape`. `copy=None` shares compatible storage and
    copies otherwise, `copy=True` always copies, and `copy=False` requires
    sharing.
  - **Returns:** an `ElementArray` with parent `A`.
  - **Notes:** If `copy=False` and sharing is impossible, `ValueError` is
    raised. [Storage](#storage) gives the conditions for sharing.

- `multiply(x, y, *, out=None, algorithm="auto") -> Element | ElementArray`

  Multiply elements or arrays, broadcasting their leading axes.

  - **Parameters:** `x` and `y` are elements, arrays, or coefficient-field
    scalars, where a scalar operand scales every coordinate of the other.
    `algorithm` is `"auto"`, `"direct"`, `"quasilinear"`, or, for real
    coefficients only, `"cariow_cariowa"`.
  - **Returns:** the products, as described in [Return values and output
    storage](#return-values-and-output-storage).
  - **Notes:** `"auto"` uses `default_algorithm` for single products and
    `default_batch_algorithm` for batches of two or more records. Names are
    case-sensitive: an unknown name, or one unavailable for this dimension or
    field, raises `ValueError`, and a name that is not a string raises
    `TypeError`.

- `multiply_sum(x, y, *, algorithm="auto", planes=False) -> Element`

  Return the sum of the products of corresponding records,
  `x[0] * y[0] + x[1] * y[1] + ...`, computed as a [fused
  sum](mathematics.md#fused-sums-of-products), which stores no product. This
  method is experimental: it may not be fully optimized or ready for production
  use, and the code implementing it may change over time, but its API is
  expected to remain the same. It is available for real coefficients only.

  - **Parameters:** `x` and `y` have the same batch shape, and every batch axis
    is summed; either may instead hold a single record, which then multiplies
    every record of the other. With `planes=True`, both are arrays of shape
    `(N, count)` whose row `i` holds coordinate `i` of every record, which is
    the layout the kernels work in. `algorithm` is `"auto"`, `"direct"`, or
    `"quasilinear"`.
  - **Returns:** one `Element`, the sum.
  - **Notes:** Batch shapes are compared as given, so `(2, 3, N)` and
    `(3, 2, N)` raise `ValueError`. Operands without records give the zero
    element, also against a single record. A two-dimensional array of aligned
    `float64` records with contiguous coordinates, such as `a[::2]`, is read in
    place, and other layouts are copied as needed. The result rounds differently
    from `x * y` summed over its records, and parents other than real ones raise
    `TypeError`.

- `covariance(x, y, *, ddof=1) -> Element`

  Return the covariance of two samples, `sum((x[p] - mean(x)) * conjugate(y[p] -
  mean(y))) / (count - ddof)` over all records `p`. Like `multiply_sum`, this
  method is experimental, with the same expectations, and available for real
  coefficients only.

  - **Parameters:** `x` and `y` are samples with the same batch shape, and
    `ddof` is the delta degrees of freedom, with `ddof=1` giving the unbiased
    estimate.
  - **Returns:** one `Element`, whose scalar coordinate is the sum over the
    coordinates of their real covariances.
  - **Notes:** A count that does not exceed `ddof`, or a negative `ddof`, raises
    `ValueError`. Unlike `multiply_sum`, `covariance` does not repeat a single
    record, so a single record against a sample raises `ValueError`. The mean is
    held in two parts to reduce rounding error in the centering, and samples of
    extreme magnitude are scaled internally (see
    [Covariance](mathematics.md#covariance)).
  - **Example:** At dimension 2 this is the covariance of complex samples stored
    as `[real, imag]`. For complex arrays `z` and `w`,
    `Algebra(dimension=2).covariance(np.column_stack((z.real, z.imag)),
    np.column_stack((w.real, w.imag)), ddof=ddof)` has the coordinates of
    `numpy.cov(z, w, ddof=ddof)[0, 1]`. The parent with `complex128`
    coefficients, `Algebra(dimension=1, char=0, deg=2)`, is a different algebra
    and raises `TypeError`.

- `left_basis_multiply(index, x, *, out=None)`

  Compute `e_index * x`, a signed permutation of the coordinates of `x`.

- `right_basis_multiply(x, index, *, out=None)`

  Compute `x * e_index`, a signed permutation of the coordinates of `x`.

- `left_operator_commutator(a, b, x, *, out=None)`

  Compute `a (b x) - b (a x)`.

- `right_operator_commutator(a, b, x, *, out=None)`

  Compute `(x b) a - (x a) b`.

### Return values and output storage

A method that accepts `out=` returns an `Element` when `out` is omitted and
every operand is an `Element`, where for `multiply` an operand may also be a
coefficient-field scalar. Otherwise it returns an `ElementArray`: `out` itself
when `out` is an `ElementArray`, a new wrapper around `out` when it is an
ndarray, and a new array when `out` is omitted.

The array passed as `out` must have exactly the broadcast result shape and the
parent's dtype, and its storage must be writable, aligned, and C-contiguous and
must not overlap any input. A wrong dtype raises `TypeError` for a real parent
and `ValueError` for complex and finite-field parents.

---

## FiniteField

A `FiniteField` is the coefficient field F_(p^d), with `char = p` an odd prime
and `deg = d` in the ranges given under [Terms and
notation](#terms-and-notation). It serves two purposes: it selects the
coefficients of an algebra through the selector `field=F`, and it is the parent
of its own scalars, the field elements that methods such as `quadratic_norm`
return over a finite field.

### Constructor

```python
FiniteField(*, char, deg=1)
FiniteField.from_modulus(*, char, modulus)
```

Fields are immutable, hashable, and cached. An extension field is represented by
polynomials modulo a monic irreducible polynomial of degree `d`, which this page
calls its *presentation*. `FiniteField(char=p, deg=d)` chooses the presentation
deterministically, while `from_modulus` takes it explicitly: `modulus` is an
ordered sequence of integers, constant coefficient first, whose entries are
reduced modulo `char` and whose leading coefficient must be one. Its degree is
`len(modulus) - 1`, so `deg` is not passed as well, and a reducible polynomial
raises `ValueError`.

The identity of a field includes its presentation, not just its order, so two
presentations of F_(p^d) are different parents and do not convert into each
other. At degree one the polynomial plays no part in the arithmetic, and every
monic linear modulus gives the same cached prime field, whose `modulus` is
`(0, 1)`; for example, `FiniteField.from_modulus(char=7, modulus=(3, 1))` is the
same object as `FiniteField(char=7)`.

### Properties

- `char: int`, `deg: int`

  The characteristic `p` and the degree `d`.

- `order: int`

  The number of elements, `char**deg`, as a Python integer.

- `modulus: tuple[int, ...]`

  The presentation, constant coefficient first, as an immutable tuple.

- `dtype`

  `uint32`, the storage dtype of the residues.

- `coefficient_shape: tuple[int, ...]`

  `()` at degree 1 and `(deg,)` otherwise, the shape of one field scalar.

- `dimension`, `level`, `field`

  `1`, `0`, and the field itself. As the parent of its own scalars, a field
  behaves like a one-dimensional algebra, so `F.dimension == 1`, `F.level == 0`,
  and `F.field is F`.

### Methods

- `F(value=0)`, `F.scalar(value)`, `F.zero()`, `F.one()`

  Return field scalars, as `Element` objects with `element.algebra is F`.

- `F.array(values)`, `F.asarray(values, *, copy=None)`

  Return `ElementArray` objects of field scalars with parent `F`, following the
  rules of `Algebra.array` and `Algebra.asarray`. A field array has shape
  `batch_shape` at degree 1 and `batch_shape + (deg,)` for an extension.

- `F.multiply(x, y, *, out=None)`

  Multiply field scalars or arrays, broadcasting their leading axes, with the
  output rules of `Algebra.multiply`.

### Field scalars and polynomial coordinates

The polynomial coordinates of a field scalar are explicit. For `deg > 1`,
`F([a0, ..., a_(deg-1)])` is the residue class of `a0 + a1*t + ...` modulo
`F.modulus`, while at degree one the scalar is written `F(a0)`. An integer
embeds in the prime subfield, and no single integer encodes an extension
element. Field scalars have no length and no indexing, so their polynomial
coordinates are read through `.coefficients`. Lists and nested lists of scalars
of `F` can construct Cayley-Dickson elements and arrays over `F`, each scalar
supplying one coefficient:

```python
import pyfastcd as fcd

F = fcd.FiniteField(char=13, deg=5)
A = fcd.Algebra(level=4, field=F)
c = F([1, 2, 0, 0, 0])
x = A.scalar(c) + A.basis(3)
q = x.quadratic_norm()       # Element with q.algebra is F
assert x * x.conjugate() == A.scalar(q)
```

A pickled field records its presentation, so an unpickled field is the same
field even when its polynomial was chosen automatically.

---

## BasisProduct

### Constructor

```python
BasisProduct(index: int, sign: int)
```

A `BasisProduct` is an immutable, hashable dataclass that compares by value and
unpacks as `(index, sign)`. For the result of `A.basis_product(i, j)`,
`e_i e_j = sign * e_index`.

### Properties

- `index: int`

  The index of the product, `i ^ j`.

- `sign: int`

  The sign of the product, `+1` or `-1`.

---

## Element and ElementArray

An `Element` is one Cayley-Dickson element or field scalar, and an
`ElementArray` is a batch of them in one NumPy array. The two classes share
their properties, methods, and operators, so this section documents them
together and notes where they differ.

### Construction

An `Element` is created with `A(...)` or `F(...)`, and an `ElementArray` with
the parent's `array` or `asarray` method; constructing either class directly
raises `TypeError`.

Elements are immutable and hashable: equal elements hash equal, and an element
equal to a real scalar hashes like that scalar. Arrays are unhashable and are
writable exactly when their NumPy storage is.

### Properties

- `algebra`

  The parent `Algebra` or `FiniteField`.

- `dimension`, `level`, `dtype`

  The dimension, doubling level, and coefficient dtype of the parent.

- `shape`

  `parent.coefficient_shape` for an element, and
  `batch_shape + parent.coefficient_shape` for an array.

- `batch_shape`

  The leading axes before the record, available on `ElementArray` only. The
  polynomial axis of an extension field is part of the record, not a batch axis.

- `real`

  Coordinate zero: a `float` over R, a `complex` over C, and a field `Element`
  over a finite field. Real and complex arrays return an ndarray view, and
  finite-field arrays an `ElementArray` view whose parent is the field. For a
  field scalar, `.real` is an equal copy of the scalar.

- `imag`

  A new element or array with coordinate zero set to zero, the same as
  `imag_part()`.

- `coefficients`

  A NumPy view of the coefficient storage, read-only for an element and sharing
  the data of an array.

### Methods

These methods allocate new results, except `to_numpy(copy=False)`, which returns
a view.

`Element` methods take `Element` operands only, passed positionally, while
`ElementArray` methods accept `Element` or `ElementArray` operands and
broadcast, so an element is combined with an array by calling the method on the
array or by using an operator. Unlike the operators, methods do not embed
scalars: every operand must be a complete element or array with the same parent,
and any other operand raises `TypeError` or `ValueError`, so a scalar enters a
method as `A.scalar(r)`.

Over a finite field, `inner` and `quadratic_norm` return scalars of the field
`F`, and every other method returns a result with the receiver's parent.

- `copy()`

  Return an independent copy.

- `conjugate()`

  Return `(x_0, -x_1, ..., -x_(N-1))`. This involution fixes every coefficient,
  so it does not conjugate complex coefficients or apply the Frobenius map over
  a finite field.

- `imag_part()`

  Return a copy with coordinate zero set to zero, the same as `imag`.

- `inner(other)`

  Return the coefficient pairing `sum_i x_i other_i`, which is the Euclidean
  inner product over R and is bilinear, without conjugation, over C and finite
  fields. Its return types follow the table under `quadratic_norm()`.

- `quadratic_norm()`

  Return the quadratic form `Q(x) = sum_i x_i**2` in the coefficient field, for
  which `x * x.conjugate() == A.scalar(Q(x))` in exact arithmetic. Over R it
  uses scaled accumulation to avoid needless intermediate overflow and
  underflow. For a field scalar `c` it returns `c*c`, which is not the norm from
  an extension field to its prime subfield.

  | Coefficient field | Element result | ElementArray result |
  |---|---|---|
  | R | `float` | `float64` ndarray |
  | C | `complex` | `complex128` ndarray |
  | finite field F | `Element`, parent F | `ElementArray`, parent F |

- `norm_squared()`

  The same as `quadratic_norm()`, so over C it is not `sum(abs(x_i)**2)`.

- `norm()`

  Return the Euclidean norm `sqrt(Q(x))`, using scaled accumulation. `norm()`,
  `abs(x)`, and `normalized()` exist only over R and raise `TypeError` for other
  parents, where `quadratic_norm()` takes their place.

- `normalized()`

  Return `x / norm(x)`, raising `ZeroDivisionError` for an exactly zero input.

- `reciprocal()`

  Return `conjugate(x) / Q(x)`, raising `ZeroDivisionError` when `Q(x) == 0`,
  including for the nonzero elements with `Q(x) == 0` that exist over C and
  finite fields. Batch reciprocals and divisions check every denominator before
  producing any result.

- `square()`

  Return `x^2`, computed with `O(N)` operations from the scalar and imaginary
  parts (see [Derived operations](mathematics.md#derived-operations)).

- `imag_cross(other)`

  Return the alternating product `(XY - YX) / 2` of the imaginary parts `X` of
  `x` and `Y` of `other` (see [Derived
  operations](mathematics.md#derived-operations)). The scalar parts of the
  inputs are ignored, and the result has scalar part zero.

- `commutator(other)`

  Return `x*other - other*x`.

- `anticommutator(other)`

  Return `x*other + other*x`, computed with `O(N)` operations from the scalar
  and imaginary parts.

- `associator(y, z)`

  Return `(x y) z - x (y z)`.

- `to_numpy(*, copy=True)`

  Return the coefficient ndarray, where `copy` must be a Python `bool`.

### Operators

Operators accept elements and arrays with the same parent, and coefficient-field
scalars, and over a finite field an array of field scalars can also scale or
divide. The results are new objects, and arrays broadcast their leading axes.

- `+x`, `-x`

  Copy an element, or negate its coordinates.

- `x + y`, `x - y`, `x + r`, `r - x`, ...

  Add or subtract coordinates, embedding scalars as `r * e_0`.

- `x * y`

  Compute the ordered product with the automatic algorithm.

- `r * x`, `x * r`

  Scale every coordinate.

- `x / r`

  Divide by a coefficient-field scalar, raising `ZeroDivisionError` for division
  by zero.

- `x ** k`

  Compute an integer power by binary exponentiation, using the reciprocal for
  negative powers. A real `Element` limits the exponent to the signed 64-bit
  range, while complex and finite-field elements, and all arrays, accept any
  Python integer.

- `abs(x)`

  Return the Euclidean norm over R; other parents raise `TypeError`.

- `x == y`, `x != y`

  Compare whole elements exactly, as described under [Equality and
  hashing](#equality-and-hashing). For arrays the result is a Boolean array over
  the broadcast batch shape.

### Equality and hashing

Equality compares whole elements exactly, and its rules for the different kinds
of operand keep it consistent with hashing:

- **Real scalars.** Construction and arithmetic round scalars to binary64, but
  equality compares exact numeric values, including those of real
  zero-dimensional ndarrays. An element containing `0.1` is therefore unequal to
  `Decimal("0.1")` and `Fraction(1, 10)`, and an element containing `2**53` is
  unequal to the integer `2**53 + 1`. Over C, comparison with a scalar is exact
  in the same way, and comparing with an object that only defines `__float__`
  requires that object to implement comparison as well.
- **Finite-field integers.** Over a finite field, integers are not reduced
  modulo `char` for comparison, so `F(1) != char + 1`, as hashing requires. A
  field scalar equals the element that embeds it in an algebra over the same
  field, and fields with different presentations compare unequal.
- **Other fastCD parents.** A fastCD object with a different parent compares
  unequal everywhere: `False` for an element, and an all-`False` array of the
  broadcast batch shape for an array. Incompatible batch shapes raise
  `ValueError`.
- **Other Python objects.** For any other operand, the fastCD comparison returns
  `NotImplemented`, so that Python tries the comparison of the other operand;
  with ordinary objects such as `str`, `complex`, `None`, and `list`, `==` gives
  `False` and `!=` gives `True`.
- **NumPy operands.** Comparison through NumPy with an unsupported operand, such
  as a nonscalar ndarray or, over R and finite fields, a complex NumPy scalar,
  raises `TypeError`.
- **Exceptions of the scalar.** Comparison with a scalar can raise that scalar's
  own exceptions; for example, comparing with `Decimal("sNaN")` raises
  `decimal.InvalidOperation` when that decimal trap is enabled.

### Indexing and iteration

- `len(x)`

  Return `N` for a Cayley-Dickson element; a coefficient-field scalar has no
  length.

- Iterating over or indexing an element yields its coefficients: `float` over R,
  `complex` over C, and field-scalar `Element` values over a finite field. A
  slice returns a tuple of coefficients, and a field scalar itself cannot be
  indexed, so its polynomial coordinates are read through `.coefficients`.
- Iterating over or indexing an array selects records along the batch axes.
  `X[i]` returns an `Element` copy when it selects one record and an
  `ElementArray` slice otherwise, and, as in NumPy, an array with an empty batch
  shape has no length and cannot be iterated.
- `X[key] = value` assigns a scalar, an object with the same parent, or a
  coefficient array.
- `bool(x)` is false only for a zero element, while `bool(X)` raises
  `ValueError`.

---

## Arrays, NumPy, and DLPack

### Storage

- **Shape.** Coefficients have shape `batch_shape + coefficient_shape`, and only
  the leading batch axes broadcast, neither the Cayley-Dickson axis nor the
  polynomial axis. Empty batch axes are supported over every field.
- **Zero-copy sharing.** `A.asarray` shares storage only when it has the
  parent's dtype, is aligned, and has coefficient axes with these strides:
  - Over R, the last axis has a stride of one coefficient. At dimension 1 a zero
    stride is also allowed, and empty C-contiguous arrays are exempt.
  - Over C and finite fields, the coefficient axes, `(N,)` or `(N, deg)` for an
    extension, have C-order strides, and axes of length one and empty arrays are
    exempt.
  - Leading batch axes may be strided, reversed, or broadcast, provided
    alignment is preserved.
  - Over a finite field, `copy=False` also requires canonical residues in
    `[0, char)`.
- **Copying.** Numeric input that cannot be shared is converted and copied,
  unless `copy=False` forbids it. When finite-field coefficients are copied,
  integers are reduced modulo `char` before being stored as `uint32`.
- **Checks.** Because the storage of an array can change after the array is
  created, every arithmetic operation that reads a finite-field array checks
  again, before writing any output, that the storage holds canonical residues.
  An `Element` is immutable and always holds canonical residues. Comparison,
  indexing, and copying do not check: `==` compares the stored words, and copies
  reduce them modulo `char`.
- **Overlap.** Inputs are only read and may share storage, as in
  `X[:-1] * X[1:]` or `X * X[:1]`, as long as each pair of corresponding
  elements is either identical or disjoint. Native operations reject partially
  overlapping elements with `ValueError`, while real array addition,
  subtraction, and equality are performed by NumPy and accept them.

### NumPy interfaces

- `Element` exposes a read-only buffer, which NumPy can read directly.
- `ElementArray.__array__` returns a view unless a change of dtype or
  `copy=True` requires a copy.
- Both classes support `np.add`, `np.subtract`, `np.multiply`, `np.true_divide`
  by a coefficient-field scalar (or, over a finite field, an array of field
  scalars), `np.power` with an integer exponent, `np.negative`, `np.positive`,
  `np.square`, `np.conjugate`, `np.absolute`, `np.equal`, and `np.not_equal`.
  Only `np.multiply` accepts `out=`, and with an ndarray `out` it returns that
  ndarray, whereas `A.multiply` returns a new `ElementArray` wrapper.
- Other ufuncs, and ufunc methods and options such as `reduce`, `accumulate`,
  and `where=`, raise `TypeError`.
- Functions that are not ufuncs, such as `np.mean` and `np.stack`, read the
  coefficients through `__array__` or the buffer and follow ordinary NumPy
  semantics, so `np.mean(x)` returns a NumPy scalar when every axis is reduced,
  and `np.stack` returns a plain ndarray. Reductions that NumPy implements as
  ufunc methods, such as `np.sum`, `np.prod`, and `np.max`, raise `TypeError`
  and are applied to `.coefficients` instead.

### DLPack interfaces

- `ElementArray.__dlpack__(*, stream=None, max_version=None, dl_device=None,
  copy=None)`

  Export the coefficient storage through DLPack.

- `ElementArray.__dlpack_device__()`

  Report the DLPack device.

- `A.array` and `A.asarray` import host-accessible DLPack producers through
  `np.from_dlpack`. Requiring `copy=False` for a generic DLPack producer needs
  NumPy 2.1 or newer, and older supported NumPy versions raise `ValueError` for
  that request.

### Copying and serialization

Pickling, `copy.copy`, and `copy.deepcopy` create independent coefficient
storage.

---

## Exceptions

pyfastcd reports errors with standard Python exceptions:

- `TypeError`

  An unsupported operand, an unsupported dtype, including a wrong `out` dtype
  for a real parent, an invalid type of element or basis index, direct
  construction of a class, a copy flag that is not a `bool`, or an algorithm
  name that is not a string.

- `ValueError`

  An invalid dimension, mismatched parents, a failure of shape or broadcasting,
  an incompatible layout or writeability, overlapping buffers, an unavailable
  algorithm, a wrong `out` dtype for a complex or finite-field parent, or the
  truth value of an array.

- `IndexError`

  A basis or coefficient index outside the valid range, or an invalid index into
  an array, which follows NumPy.

- `ZeroDivisionError`

  Division by a zero scalar, normalizing a zero real element, or the reciprocal
  or a negative power of an element with `Q(x) == 0`, including the nonzero such
  elements over C and finite fields.

- `OverflowError`

  A dimension, scalar, or storage size beyond the limits of the implementation,
  or a real `Element` exponent outside the signed 64-bit range.

- `MemoryError`, `BufferError`, `RuntimeError`

  A failed allocation, an unsatisfiable buffer or DLPack request, or an
  unexpected native failure.

### Floating-point conditions

Native operations raise no exception for NaN, infinity, or overflow. NumPy's
error settings apply only where NumPy itself performs the arithmetic: in the
coordinate operations on real `ElementArray` operands, which are addition,
subtraction, scaling, scalar division, and equality, and in the comparison of a
complex element or array with a scalar, which NumPy carries out exactly.
Arithmetic on complex and finite-field arrays, and all `Element` arithmetic, is
native and never raises on these conditions.

---

## Threads

Algebras, fields, and elements are immutable and can be shared between threads.
The storage of an array must not be modified while an operation that uses it is
running, because large batch operations release the global interpreter lock
(GIL), so other threads can run during them.

Free-threaded CPython builds are not supported: the extension does not declare
free-threading compatibility, so importing `pyfastcd` on such an interpreter
re-enables the GIL.

---

## Versioning and stability

pyfastcd follows semantic versioning, and its version string is the version of
the fastCD C library it wraps. The public surface consists of the names in
`pyfastcd.__all__` together with the members documented on this page, while
modules and names with a leading underscore are private and may change without
notice. Removing public API, or changing a calling contract incompatibly,
requires a major version and a prior deprecation note in `CHANGELOG.md`.

Calls written for version 1.0.0 behave the same way in every later 1.x release,
with the same classes, argument rules, exceptions, and results. Two kinds of
difference are permitted, and either may come with a minor or patch release that
improves performance or correctness:

- floating-point results may round differently, even when an algorithm is named
  explicitly, so results are not guaranteed to be bitwise identical across
  versions;
- `"auto"` may choose a different algorithm.

`CHANGELOG.md` records every numerical change of this kind.

---

## Typing

pyfastcd ships type stubs. Without type arguments, the annotations `Algebra`,
`Element`, and `ElementArray` mean real coefficients, with `float64` storage and
`float` scalars. A type checker infers complex or finite-field types from
literal `char=0, deg=2` or `field=F` arguments, but when `char` and `deg` are
variables it infers dynamic types, because their values select the field only at
run time. `Algebra[...]`, `Element[...]`, and `ElementArray[...]` may be written
in annotations, but the type arguments have no effect at run time, where only
`char`, `deg`, and `field` select the coefficients.
