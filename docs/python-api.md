# Python API reference

`pyfastcd` is the Python interface to fastCD. Its public names are
`Algebra`, `BasisProduct`, `Element`, `ElementArray`, `FiniteField`,
`cd_element`, and `__version__`; modules whose names begin with an
underscore are private. Errors are reported as Python exceptions, and Python
callers never manage workspaces.

## Terms and notation

**Coefficient field and selectors.** Every algebra has its coefficients in
R, C, or a finite field F_(p^d) (see
[mathematics.md](mathematics.md#the-algebra)). Three keyword arguments, the
*selectors* `char`, `deg`, and `field`, choose the field:

| Selectors | Coefficients | `dtype` | `coefficient_shape` |
|---|---|---|---|
| omitted, or `char=0, deg=1` | R | `float64` | `(N,)` |
| `char=0, deg=2` | C | `complex128` | `(N,)` |
| `char=p, deg=1` | F_p | `uint32` | `(N,)` |
| `char=p, deg=d`, `d > 1` | F_(p^d) | `uint32` | `(N, d)` |
| `field=F` | the `FiniteField` `F` | `uint32` | `(N,)` or `(N, F.deg)` |

- `char` is `0` or an odd prime at most `2**31-1`.
- `deg` is `1` (R) or `2` (C) when `char=0`, and `1` to `12` for a finite
  field. It is the degree of the coefficient field and is independent of the
  Cayley–Dickson `level`.
- `field` is a `FiniteField`. Pass it on its own: combining it with a
  nondefault `char` or `deg` raises `TypeError`.
- Characteristic 2, negative or composite characteristics, and unsupported
  degrees raise `ValueError`. Boolean values, including `numpy.bool_`, are
  rejected for `char` and `deg`, as they are for `dimension` and `level`.

With no selectors, coefficients are real `float64` values. Complex and
finite-field coefficients are always selected explicitly.

**Parent.** Every element and array belongs to a *parent*: the `Algebra` that
created it, or the `FiniteField` of a coefficient-field scalar. `x.algebra`
returns it. Operations combine only objects with the same parent, and
nothing converts between parents implicitly.

**Record.** A *record* is the coefficient storage of one Cayley–Dickson
element. Its shape is the parent's `coefficient_shape`: `N` coefficients,
followed over F_(p^d) by an axis of `d` polynomial coordinates. An array of
elements has shape `batch_shape + coefficient_shape`.

Below, `A` is an algebra, `N = A.dimension`, `x`, `y`, and `z` are elements,
`X` is an element array, and `r` is a coefficient-field scalar.

## Module contents

- `__version__: str`

  The version of the library.

- `cd_element(coefficients, *, char=0, deg=1, field=None) -> Element`

  Construct one element from a single record, inferring `N` from its
  length, which must be a positive power of two. The input is an ordered
  sequence or array, not an iterator, with the shape given in the selector
  table. An `Element` argument is copied and keeps its parent unless
  selectors are given. An invalid shape or length raises `ValueError`; for a
  real parent, complex, object-valued, or nonnumeric arrays raise
  `TypeError`.

### Real scalars

For a real parent, scalars may be `bool`, `int`, `float`, `Decimal`,
`Fraction`, real NumPy scalars, or real zero-dimensional ndarrays. Complex
values, text, and arrays with an object dtype are rejected. An object with a
custom `__float__` method can be used in construction and arithmetic.

Construction and arithmetic round scalars to binary64, but equality compares
exact numeric values. An element containing `0.1` is therefore unequal to
`Decimal("0.1")` and `Fraction(1, 10)`, and an element containing `2**53` is
unequal to the integer `2**53 + 1`. The same rules apply to `ElementArray`.
Comparing with an object that only defines `__float__` also requires that
object to implement comparison.

---

## Algebra

### Constructor

```python
Algebra(*, dimension=None, level=None, char=0, deg=1, field=None)
```

Create the algebra of dimension `N = 2**level` over the coefficient field
chosen by the selectors. Supply exactly one of `dimension`, a positive power
of two, and `level`, the doubling level. Algebras are immutable, cannot be
subclassed, and are cached by dimension and coefficient field, so
`Algebra(level=3) is Algebra(dimension=8)`.

`Algebra(dimension=2)` is the complex numbers, built from two **real**
coefficients. `Algebra(dimension=2, char=0, deg=2)` has two **complex**
coefficients; it is a different algebra, isomorphic to C ⊕ C.

### Properties

- `dimension: int`

  The number of coefficients, `N`.

- `level: int`

  The doubling level, `log2(N)`.

- `dtype`

  The storage dtype from the selector table.

- `char: int`, `deg: int`

  The characteristic and degree of the coefficient field.

- `field: FiniteField | None`

  The finite coefficient field; `None` over R and C.

- `coefficient_shape: tuple[int, ...]`

  The shape of one record.

- `default_algorithm: str`

  The algorithm that `"auto"` uses for a single product: two `Element`
  operands, or a batch of one record. For real coefficients this is the
  choice of the C library's `fastcd_default_algorithm` (see
  [Automatic algorithm selection](mathematics.md#automatic-algorithm-selection)).
  Over C it is `"quasilinear"`; over finite fields it is `"direct"` for
  `N <= 4` and `"quasilinear"` otherwise.

- `default_batch_algorithm: str`

  The algorithm that `"auto"` uses for batches of two or more records: the
  C library's `fastcd_default_batch_algorithm` for real coefficients, and
  the same choice as `default_algorithm` over C and finite fields.

### Methods

- `A(value=0) -> Element`

  Create an element from `value`, which is one of:

  - a coefficient scalar, placed in coordinate zero;
  - an `Element` with parent `A`, which is copied;
  - exactly `N` coefficients from a sequence, iterator, or buffer.
    Coefficients are never padded or truncated, and unordered inputs such
    as `set` and `dict` raise `TypeError`.

  Over C, coefficients may be complex numbers. Over a finite field they are
  integers or scalars of the same field; integers, however large, are
  reduced modulo `char` before conversion to `uint32`, and floats are
  rejected even when they are integral.

- `scalar(r) -> Element`

  Return the scalar element `r * e_0`.

- `zero() -> Element`

  Return the additive identity.

- `one() -> Element`

  Return the multiplicative identity.

- `basis(index) -> Element`

  Return `e_index`. Raise `IndexError` if the index is outside `range(N)`.

- `basis_product(i, j) -> BasisProduct`

  Return the `BasisProduct` describing `e_i e_j`.

- `array(values) -> ElementArray`

  Copy `values` into new writable, aligned, C-contiguous storage with
  dtype `A.dtype`.

- `asarray(values, *, copy=None) -> ElementArray`

  Share compatible storage when `copy=None`, always copy when `copy=True`,
  and require zero-copy sharing when `copy=False`. If zero-copy sharing is
  required but impossible, raise `ValueError`. [Storage](#storage) gives
  the conditions for sharing.

- `multiply(x, y, *, out=None, algorithm="auto") -> Element | ElementArray`

  Multiply elements or arrays, broadcasting leading axes; a coefficient
  scalar operand scales every coordinate. `algorithm` is `"auto"`,
  `"direct"`, `"quasilinear"`, or, for real coefficients only,
  `"cariow_cariowa"`. `"auto"` uses `default_algorithm` for single products
  and `default_batch_algorithm` for batches of two or more records. Names
  are case-sensitive. An unknown name, or one unavailable for this
  dimension or field, raises `ValueError`; a name that is not a string
  raises `TypeError`.

- `left_basis_multiply(index, x, *, out=None)`

  Compute `e_index * x` as a signed permutation.

- `right_basis_multiply(x, index, *, out=None)`

  Compute `x * e_index` as a signed permutation.

- `left_operator_commutator(a, b, x, *, out=None)`

  Compute `a (b x) - b (a x)`.

- `right_operator_commutator(a, b, x, *, out=None)`

  Compute `(x b) a - (x a) b`.

### Return values and output storage

A method that accepts `out=` returns:

- an `Element` when `out` is omitted and every operand is an `Element` (for
  `multiply`, an operand may also be a coefficient scalar);
- otherwise an `ElementArray`: `out` itself when it is an `ElementArray`, a
  new wrapper around `out` when it is an ndarray, and a new array when
  `out` is omitted.

`out` must have exactly the broadcast result shape and the parent's dtype.
Its storage must be writable, aligned, and C-contiguous, and must not
overlap any input.

---

## FiniteField

```python
FiniteField(*, char, deg=1)
FiniteField.from_modulus(*, char, modulus)
```

A `FiniteField` is the coefficient field F_(p^d), with `char = p` an odd
prime and `deg = d` in the ranges given under
[Terms and notation](#terms-and-notation). Fields are immutable, hashable,
and cached.

An extension field is represented by polynomials modulo a monic irreducible
polynomial of degree `d`, called its *presentation* here.
`FiniteField(char=p, deg=d)` chooses the presentation deterministically.
`from_modulus` takes it explicitly: `modulus` is an ordered integer
sequence, constant coefficient first, whose entries are reduced modulo
`char` and whose leading coefficient must be one. The degree is
`len(modulus)-1`, so do not also pass `deg`. A reducible polynomial raises
`ValueError`.

A field's identity includes its presentation, not just its order: two
presentations of F_(p^d) are different parents and do not convert into each
other. At degree one the polynomial plays no part in arithmetic, so every
monic linear modulus gives the same cached prime field, whose `modulus` is
`(0, 1)`. For example, `FiniteField.from_modulus(char=7, modulus=(3, 1))` is
the same object as `FiniteField(char=7)`.

A field has the read-only attributes `char`, `deg`, `order` (`char**deg`, a
Python integer), `modulus` (an immutable tuple), `dtype` (`uint32`), and
`coefficient_shape` (`()` at degree 1, otherwise `(deg,)`). As the parent of
its own scalars, a field behaves like a one-dimensional algebra:
`F.dimension == 1`, `F.level == 0`, and `F.field is F`.

- `F(value=0)`, `F.scalar(value)`, `F.zero()`, `F.one()` return `Element`
  objects with `element.algebra is F`.
- `F.array(values)` and `F.asarray(values, *, copy=None)` return
  `ElementArray` objects with parent `F`. A field array has shape
  `batch_shape` at degree 1 and `batch_shape + (deg,)` for an extension.
- `F.multiply(x, y, *, out=None)` multiplies field scalars or arrays,
  broadcasting leading axes, with the same output rules as
  `Algebra.multiply`.

Polynomial coordinates are explicit: for `deg > 1`,
`F([a0, ..., a_(deg-1)])` is the residue class of `a0 + a1*t + ...` modulo
`F.modulus`; at degree one, write `F(a0)`. An integer embeds in the
prime subfield; there is no encoding of an extension element as a single
integer. Field scalars have no length or indexing; use `.coefficients` for
their polynomial coordinates. Lists and nested lists of scalars of `F` can
construct Cayley–Dickson elements and arrays over `F`, each scalar
supplying one coefficient.

```python
import pyfastcd as fcd

F = fcd.FiniteField(char=13, deg=5)
A = fcd.Algebra(level=4, field=F)
c = F([1, 2, 0, 0, 0])
x = A.scalar(c) + A.basis(3)
q = x.quadratic_norm()       # Element with q.algebra is F
assert x * x.conjugate() == A.scalar(q)
```

Pickles record the presentation, so an unpickled field is the same field
even when its polynomial was chosen automatically.

---

## BasisProduct

### Constructor

```python
BasisProduct(index: int, sign: int)
```

An immutable, hashable dataclass that compares by value and unpacks as
`(index, sign)`. For the result of `A.basis_product(i, j)`,
`e_i e_j = sign * e_index`.

### Properties

- `index: int`

  The destination basis index, `i ^ j`.

- `sign: int`

  The sign of the basis product, either `+1` or `-1`.

---

## Element and ElementArray

### Construction

Create an `Element` with `A(...)` or `F(...)` and an `ElementArray` with the
parent's `array` or `asarray` method; constructing either class directly
raises `TypeError`. Elements are immutable and hashable: equal elements hash
equal, and an element equal to a real scalar hashes like that scalar. Arrays
are unhashable and are writable exactly when their NumPy storage is.

### Properties

- `algebra`

  The parent `Algebra` or `FiniteField`.

- `dimension`, `level`, `dtype`

  The dimension, doubling level, and coefficient dtype of the parent.

- `shape`

  `parent.coefficient_shape` for an element, or
  `batch_shape + parent.coefficient_shape` for an array.

- `batch_shape`

  The leading axes before the record. Available on `ElementArray` only. The
  polynomial axis of an extension field is part of the record, not a batch
  axis.

- `real`

  Coordinate zero: a `float` over R, a `complex` over C, or a field
  `Element` over a finite field. Real and complex arrays return an ndarray
  view, and finite-field arrays an `ElementArray` view whose parent is the
  field. For a field scalar, `.real` is an equal copy of the scalar.

- `imag`

  A new element or array with coordinate zero set to zero; the same as
  `imag_part()`.

- `coefficients`

  A NumPy view of the coefficient storage. The view of an element is
  read-only; the view of an array shares its data.

### Methods

These methods allocate new results, except `to_numpy(copy=False)`, which
returns a view. `Element` methods take `Element` operands only, positionally.
`ElementArray` methods accept `Element` or `ElementArray` operands and
broadcast, so to combine an element with an array, call the method on the
array or use an operator.

Unlike the operators, methods do not embed scalars: every operand must be a
complete element or array with the same parent, and any other operand
raises `TypeError` or `ValueError`. Use `A.scalar(r)` for a scalar element.
Over a finite field, `inner` and `quadratic_norm` return scalars of the
field `F`; every other result has the receiver's parent.

- `copy()`

  Return an independent copy.

- `conjugate()`

  Return `(x_0, -x_1, ..., -x_(N-1))`. This involution fixes every
  coefficient: it does not conjugate complex coefficients or apply the
  Frobenius map over a finite field.

- `imag_part()`

  Return a copy with coordinate zero set to zero; the same as `imag`.

- `inner(other)`

  Return the coefficient pairing `sum_i x_i other_i`. Over R this is the
  Euclidean inner product; over C and finite fields it is bilinear, without
  conjugation. Return types follow the table under `quadratic_norm()`.

- `quadratic_norm()`

  Return the quadratic form `Q(x) = sum_i x_i**2` in the coefficient field;
  in exact arithmetic, `x * x.conjugate() == A.scalar(Q(x))`. Over R it uses
  scaled accumulation to avoid needless intermediate overflow and underflow.
  For a field scalar `c` it returns `c*c`, which is not the norm from an
  extension field to its prime subfield.

  | Coefficient field | Element result | ElementArray result |
  |---|---|---|
  | R | `float` | `float64` ndarray |
  | C | `complex` | `complex128` ndarray |
  | finite field F | `Element`, parent F | `ElementArray`, parent F |

- `norm_squared()`

  The same as `quadratic_norm()`. Over C it is not `sum(abs(x_i)**2)`.

- `norm()`

  Return the Euclidean norm `sqrt(Q(x))`, using scaled accumulation.
  `norm()`, `abs(x)`, and `normalized()` exist only over R and raise
  `TypeError` for other parents; use `quadratic_norm()` there.

- `normalized()`

  Return `x / norm(x)`. Raise `ZeroDivisionError` for an exactly zero input.

- `reciprocal()`

  Return `conjugate(x) / Q(x)`. Raise `ZeroDivisionError` when `Q(x) == 0`,
  including for the nonzero elements with `Q(x) == 0` that exist over C and
  finite fields. Batch reciprocals and divisions check every denominator
  before producing any result.

- `square()`

  Return `x^2`, computed with `O(N)` operations from the scalar and
  imaginary parts (see
  [Derived operations](mathematics.md#derived-operations)).

- `imag_cross(other)`

  Return the alternating product `(XY - YX) / 2` of the imaginary parts `X`
  of `x` and `Y` of `other` (see
  [Derived operations](mathematics.md#derived-operations)). The scalar parts
  of the inputs are ignored, and the result has scalar part zero.

- `commutator(other)`

  Return `x*other - other*x`.

- `anticommutator(other)`

  Return `x*other + other*x`, computed with `O(N)` operations from the
  scalar and imaginary parts.

- `associator(y, z)`

  Return `(x y) z - x (y z)`.

- `to_numpy(*, copy=True)`

  Return the coefficient ndarray. `copy` must be a Python `bool`.

### Operators

Operators accept elements and arrays with the same parent, and
coefficient-field scalars. Results are new objects, and arrays broadcast
their leading axes.

- `+x`, `-x`

  Copy an element or negate its coordinates.

- `x + y`, `x - y`, `x + r`, `r - x`, ...

  Add or subtract coordinates, embedding scalars as `r * e_0`.

- `x * y`

  Compute the ordered product with the automatic algorithm.

- `r * x`, `x * r`

  Scale every coordinate.

- `x / r`

  Divide by a coefficient-field scalar. Division by zero raises
  `ZeroDivisionError`.

- `x ** k`

  Compute an integer power by binary exponentiation; negative powers use
  the reciprocal. A real `Element` limits the exponent to the signed 64-bit
  range; complex and finite-field elements, and all arrays, accept any
  Python integer.

- `abs(x)`

  Return the Euclidean norm over R; other parents raise `TypeError`.

- `x == y`, `x != y`

  Compare whole elements exactly. For arrays the result is a Boolean array
  over the broadcast batch shape.

  - Real scalars, including real zero-dimensional ndarrays, compare as
    described under [Real scalars](#real-scalars). Over C, comparison with
    a scalar is exact in the same way.
  - Over a finite field, integers are not reduced modulo `char` for
    comparison, so `F(1) != char + 1`; this keeps equality consistent with
    hashing. A field scalar equals the element that embeds it in an algebra
    over the same field, and fields with different presentations compare
    unequal.
  - A fastCD object with a different parent compares unequal everywhere:
    `False` for an element, and an all-`False` array of the broadcast batch
    shape for an array. Incompatible batch shapes raise `ValueError`.
  - For any other operand, the fastCD comparison returns `NotImplemented`,
    so Python tries the other operand's comparison. With ordinary objects
    such as `str`, `complex`, `None`, and `list`, `==` gives `False` and
    `!=` gives `True`.
  - Through NumPy, comparison with an unsupported operand, such as a
    nonscalar ndarray or a complex NumPy scalar, raises `TypeError`.
  - Comparison with a scalar can raise that scalar's own exceptions; for
    example, comparing with `Decimal("sNaN")` raises
    `decimal.InvalidOperation` when that decimal trap is enabled.

### Indexing and iteration

- `len(x)`

  Return `N` for a Cayley–Dickson element. A coefficient-field scalar has no
  length.

- Iterating over or indexing an element yields its coefficients: `float`
  over R, `complex` over C, and field-scalar `Element` values over a finite
  field. A slice returns a tuple of coefficients. A field scalar itself
  cannot be indexed; use `.coefficients` for its polynomial coordinates.
- Iterating over or indexing an array selects records along the batch axes.
  `X[i]` returns an `Element` copy when it selects one record and an
  `ElementArray` slice otherwise. As in NumPy, an array with an empty batch
  shape has no length and cannot be iterated.
- `X[key] = value` assigns a scalar, an object with the same parent, or a
  coefficient array.
- `bool(x)` is false only for a zero element. `bool(X)` raises `ValueError`.

---

## Arrays, NumPy, and DLPack

### Storage

- **Shape.** Coefficients have shape `batch_shape + coefficient_shape`. Only
  the leading batch axes broadcast; neither the Cayley–Dickson axis nor the
  polynomial axis does. Empty batch axes are supported over every field.
- **Zero-copy sharing.** `A.asarray` shares storage only when it has the
  parent's dtype and is aligned, and its coefficient axes meet these
  stride rules:
  - Over R, the last axis has a stride of one coefficient. At dimension 1 a
    zero stride is also allowed, and empty C-contiguous arrays are exempt.
  - Over C and finite fields, the coefficient axes (`(N,)`, or `(N, deg)`
    for an extension) have C-order strides. Axes of length one and empty
    arrays are exempt.
  - Leading batch axes may be strided, reversed, or broadcast, provided
    alignment is preserved.
  - Over a finite field, `copy=False` also requires canonical residues in
    `[0, char)`.
- **Copying.** Numeric input that cannot be shared is converted and copied,
  unless `copy=False` forbids it. When finite-field coefficients are copied,
  integers are reduced modulo `char` before being stored as `uint32`.
- **Checks.** Before writing any output, every operation checks again that
  finite-field storage holds canonical residues.
- **Overlap.** Inputs are read-only and may share storage, as in
  `X[:-1] * X[1:]` or `X * X[:1]`, as long as each pair of corresponding
  elements is either identical or disjoint. Partially overlapping elements
  raise `ValueError`.

### NumPy interfaces

- `Element` exposes a read-only buffer, which NumPy can read directly.
- `ElementArray.__array__` returns a view unless a dtype change or
  `copy=True` requires a copy.
- Both classes support `np.add`, `np.subtract`, `np.multiply`,
  `np.true_divide` by a real scalar, `np.power` with an integer exponent,
  `np.negative`, `np.positive`, `np.square`, `np.conjugate`,
  `np.absolute`, `np.equal`, and `np.not_equal`. Only `np.multiply` accepts
  `out=`; with an ndarray `out` it returns that ndarray, whereas
  `A.multiply` returns a new `ElementArray` wrapper.
- Other ufuncs, and ufunc methods and options such as `reduce`,
  `accumulate`, and `where=`, raise `TypeError`.
- Functions that are not ufuncs, such as `np.mean` and `np.stack`, read the
  coefficients through `__array__` or the buffer and follow ordinary NumPy
  semantics: `np.mean(x)` returns a NumPy scalar when every axis is reduced,
  and `np.stack` returns a plain ndarray.

### DLPack interfaces

- `ElementArray.__dlpack__(*, stream=None, max_version=None, dl_device=None, copy=None)`

  Export the coefficient storage through DLPack.

- `ElementArray.__dlpack_device__()`

  Report the DLPack device.

- `A.array` and `A.asarray` import host-accessible DLPack producers through
  `np.from_dlpack`. Requiring `copy=False` for a generic DLPack producer
  needs NumPy 2.1 or newer; older supported NumPy versions raise
  `ValueError` for that request.

### Copying and serialization

Pickling, `copy.copy`, and `copy.deepcopy` create independent coefficient
storage.

---

## Exceptions

- `TypeError`

  An unsupported operand or dtype, an invalid index type for an element or
  basis index (array indexing follows NumPy and raises `IndexError`),
  direct class construction, a non-`bool` copy flag, or a non-string
  algorithm name.

- `ValueError`

  An invalid dimension, mismatched parents, a shape or broadcast failure,
  incompatible layout or writeability, overlapping buffers, an unavailable
  algorithm, or the truth value of an array.

- `IndexError`

  A basis or coefficient index outside the valid range.

- `ZeroDivisionError`

  Division by a zero scalar, normalizing a zero real element, or the
  reciprocal or a negative power of an element with `Q(x) == 0`, including
  nonzero such elements over C and finite fields.

- `OverflowError`

  A dimension, scalar, or storage size beyond the implementation limits, or
  a real `Element` exponent outside the signed 64-bit range.

- `MemoryError`, `BufferError`, `RuntimeError`

  Allocation failure, an unsatisfiable buffer or DLPack request, or an
  unexpected native failure.

Native operations do not raise on NaN, infinity, or overflow. NumPy's error
settings apply to the coordinate operations that NumPy itself performs on
`ElementArray` operands: addition, subtraction, scaling, scalar division,
and equality. `Element` arithmetic is native and never raises on these
conditions.

---

## Threads

Algebras, fields, and elements are immutable and can be shared between
threads. Do not modify an array's storage while an operation that uses it is
running: large batch operations release the global interpreter lock (GIL),
so other threads can run during them.

Free-threaded CPython builds are not supported. The extension does not
declare free-threading compatibility, so importing `pyfastcd` on such an
interpreter re-enables the GIL.

---

## Versioning and stability

pyfastcd follows semantic versioning, and its version string is the version
of the fastCD C library it wraps. The public surface is the names in
`pyfastcd.__all__` together with the members documented on this page;
modules and names with a leading underscore are private and may change
without notice. Removing public API or changing a calling contract
incompatibly requires a major version and a prior deprecation note in
`CHANGELOG.md`.

Calls written for version 1.0.0 behave the same way in every later 1.x
release: the same classes, argument rules, exceptions, and results. Only two
kinds of difference are permitted, and both may come with a minor or patch
release that improves performance or correctness:

- floating-point results may round differently, even when an algorithm is
  named explicitly, so results are not guaranteed bitwise identical across
  versions;
- `"auto"` may choose a different algorithm.

Numerical changes of this kind are recorded in `CHANGELOG.md`.

---

## Typing

pyfastcd ships type stubs. Without type arguments, `Algebra`, `Element`, and
`ElementArray` annotations mean real coefficients: `float64` storage and
`float` scalars. A type checker infers complex or finite-field types from
literal `char=0, deg=2` or `field=F` arguments. When `char` and `deg` are
variables, it infers dynamic types, because their values select the field
only at run time. `Algebra[...]`, `Element[...]`, and `ElementArray[...]`
may be written in annotations; the type arguments have no effect at run
time, where only `char`, `deg`, and `field` select coefficients.
