# Python API reference

## pyfastcd

The public names are `Algebra`, `BasisProduct`, `Element`, `ElementArray`,
`FiniteField`, `cd_element`, and `__version__`. Modules whose names begin with an underscore
are private. Python callers use exceptions rather than native status codes
and do not manage workspaces.

Below, `A` is an algebra, `N = A.dimension`, `x, y, z` are elements, `X` is
an element array, and `r` is a coefficient-field scalar. With no `char`,
`deg` or `field` argument, every constructor uses real `float64`
coefficients; complex and finite-field coefficients are selected explicitly,
as described under `Algebra` and `FiniteField`.

### Attributes

- `__version__: str`

  The version of the library.

### Functions

- `cd_element(coefficients, *, char=0, deg=1, field=None) -> Element`

  Construct one element, inferring its dimension from its coefficient record.
  Input is an ordered sequence or array, not an iterator. The shape is `(N,)`
  over R, C and prime fields, or `(N, deg)` over finite extensions; `N` must
  be a positive power of two. Invalid shape or length raises `ValueError`;
  complex, object-valued, or nonnumeric coefficient arrays raise `TypeError`
  for a real parent. An `Element` argument is copied, keeping its parent
  when no selectors are supplied. The selectors have the same meaning as
  for `Algebra`. Exactly one record is required.

### Scalars

For a real parent, real scalars are `bool`, `int`, `float`, `Decimal`, `Fraction`, real
NumPy scalars, and real zero-dimensional ndarrays. Complex values, text, and
arrays with an object dtype are rejected.

Construction and arithmetic convert scalars to binary64. Equality compares
their numeric values without that rounding: an element containing `0.1` is
unequal to `Decimal("0.1")` or `Fraction(1, 10)`, and an element containing
`2**53` is unequal to the integer `2**53 + 1`. The same rules apply to
`ElementArray`. A custom `__float__` method permits conversion and arithmetic;
numeric equality also requires the object's comparison protocol.

---

## Algebra

### Constructor

```python
Algebra(*, dimension=None, level=None, char=0, deg=1, field=None)
```

Create an immutable algebra, cached by dimension and coefficient field.
Algebras cannot be subclassed. With `char`, `deg` and `field` omitted, the
coefficients are real; a real algebra never accepts complex coefficients
implicitly.

**Parameters**

- `dimension`: a positive power of two.
- `level`: the doubling level, where `dimension = 2**level`.

- `char`: `0` (default), or an odd prime `3 <= char <= 2**31-1`.
- `deg`: extension degree (default `1`). For `char=0`, `1` means R and `2`
  means C. For finite fields, degrees `1..12` are supported.
- `field`: a `FiniteField`. Supply it with the default
  `char=0, deg=1` selectors; conflicting nondefault selectors raise `TypeError`.

Supply exactly one of `dimension` and `level`. Boolean values, including
`numpy.bool_`, are rejected for dimension, level, char, and deg. `char` must
be prime when positive; characteristic 2, negative characteristic, composite
characteristic, and unsupported degrees raise `ValueError`. `deg` describes
the coefficient field, independently of the CD doubling `level`.

| Selectors | Coefficients | `dtype` | `coefficient_shape` |
|---|---|---|---|
| omitted, or `char=0, deg=1` | R | `float64` | `(N,)` |
| `char=0, deg=2` | C | `complex128` | `(N,)` |
| `char=p, deg=1` | F_p | `uint32` | `(N,)` |
| `char=p, deg=d`, `d > 1` | F_(p^d) | `uint32` | `(N, d)` |

`Algebra(dimension=2)` is the complex numbers built from two **real**
coefficients. `Algebra(dimension=2, char=0, deg=2)` has two **complex**
coefficients and is a different algebra, isomorphic to C ⊕ C. Elements
never convert between parents implicitly.

### Properties

- `dimension: int`

  The number of coefficients over the selected field, `N`.

- `level: int`

  The doubling level, `log2(N)`.

- `dtype`

  The native storage dtype in the table above.

- `char: int`, `deg: int`

  The coefficient characteristic and extension degree.

- `field: FiniteField | None`

  The finite coefficient parent; `None` for R and C.

- `coefficient_shape: tuple[int, ...]`

  The complete trailing storage axes of one algebra element.

- `default_algorithm: str`

  The algorithm selected by `"auto"` for two `Element` operands or a
  one-record batch. The choice is shared with C.

- `default_batch_algorithm: str`

  The algorithm selected by `"auto"` for batches of two or more records.
  The choice is shared with C.

### Methods

- `A(value=0) -> Element`

  Embed a coefficient scalar in coordinate zero, copy a same-parent `Element`,
  or copy exactly `N` coefficients from a sequence, iterator, or buffer.
  Unordered inputs such as `set` and `dict` raise `TypeError`. Coefficients
  are never padded or truncated. Complex parents accept complex numeric
  coefficients; finite parents accept integer coordinates and matching
  coefficient-field scalars. Finite integers are reduced modulo `char`
  before conversion to `uint32`, including arbitrarily large Python integers.
  Floats are rejected by finite constructors, including integral floats.

- `scalar(r) -> Element`

  Return the scalar element `r * e_0`.

- `zero() -> Element`

  Return the additive identity.

- `one() -> Element`

  Return the multiplicative identity.

- `basis(index) -> Element`

  Return `e_index`. Raise `IndexError` if the index is outside `range(N)`.

- `basis_product(i, j) -> BasisProduct`

  Return the signed basis product without using a table.

- `array(values) -> ElementArray`

  Copy values into writable, aligned, C-contiguous storage with `A.dtype`.

- `asarray(values, *, copy=None) -> ElementArray`

  Share compatible storage when `copy=None`, always copy when `copy=True`,
  and require zero-copy sharing when `copy=False`. If zero-copy sharing is
  required but impossible, raise `ValueError`.

- `multiply(x, y, *, out=None, algorithm="auto") -> Element | ElementArray`

  Real parents support `"auto"`, `"direct"`, `"cariow_cariowa"`, and
  `"quasilinear"`. Complex/finite parents support `"auto"`, `"direct"`,
  and `"quasilinear"`; `"auto"` currently selects quasilinear for both
  singles and batches. Cariow–Cariowa is unavailable over C and finite fields.
  Automatic selection uses `default_algorithm` for single products and
  `default_batch_algorithm` for batches of two or more records. Coefficient scalars
  scale coordinates, and leading axes broadcast. Raise `ValueError` for an
  unknown algorithm name or one unavailable at `N`, or `TypeError` if the
  name is not a string. Names are case-sensitive.

- `left_basis_multiply(index, x, *, out=None)`

  Compute `e_index * x` as a signed permutation.

- `right_basis_multiply(x, index, *, out=None)`

  Compute `x * e_index` as a signed permutation.

- `left_operator_commutator(a, b, x, *, out=None)`

  Compute `a (b x) - b (a x)`.

- `right_operator_commutator(a, b, x, *, out=None)`

  Compute `(x b) a - (x a) b`.

**Returns and output storage**

Methods accepting `out=` return an `Element` when every operand is an
`Element` (or, for `multiply` only, a coefficient scalar) and no `out` is given;
otherwise they return an
`ElementArray`, the same wrapper when `out` is an `ElementArray`, or a new
wrapper around a supplied ndarray. `out` must have the exact broadcast result
shape, the parent's native dtype, writable aligned C-contiguous storage, and no
overlap with any input.

---

## FiniteField

```python
FiniteField(*, char, deg=1)
FiniteField.from_modulus(*, char, modulus)
```

Construct an immutable, hashable, cached coefficient parent. `char` is an odd
prime at most `2**31-1`; `deg` lies in `1..12`. An extension field F_(p^d)
is represented as polynomials modulo a monic irreducible polynomial of
degree `d`, called its presentation here. The default constructor chooses
that polynomial deterministically. Field identity includes the polynomial,
not just the field order, so two presentations of isomorphic extension
fields do not coerce into each other. At degree one the polynomial plays no
part in arithmetic, so every monic linear modulus gives the same cached
prime field, whose recorded modulus is `(0, 1)`.

`from_modulus` constructs a field from an explicit polynomial. `modulus` is
an ordered integer sequence, constant coefficient first; entries are reduced
modulo `char`, and the leading coefficient must be one. Its degree is
`len(modulus)-1`; do not also pass `deg`. A reducible polynomial raises
`ValueError`. For example, `FiniteField.from_modulus(char=7, modulus=(3, 1))`
is the same object as `FiniteField(char=7)`.

Read-only metadata: `char`, `deg`, `order` (`char**deg`, a Python integer),
`modulus` (an immutable tuple), `dtype` (`uint32`), and `coefficient_shape`
(`()` at degree 1, otherwise `(deg,)`). `dimension=1`, `level=0`, and
`field is self` describe scalars over this field.

- `F(value=0)`, `F.scalar(value)`, `F.zero()`, `F.one()` return `Element`
  objects with `element.algebra is F`.
- `F.array(values)` and `F.asarray(values, *, copy=None)` return
  `ElementArray` objects with parent `F`. Field arrays have shape `batch_shape`
  at degree 1, or `batch_shape + (deg,)` for extensions.
- `F.multiply(x, y, *, out=None)` multiplies field scalars/arrays with leading
  batch broadcasting and the same output contract as `Algebra.multiply`.

Polynomial coordinates are explicit: for `deg > 1`, `F([a0, ..., a_(deg-1)])`
represents the residue class of `a0 + a1*t + ...` modulo `F.modulus`. At
degree one use `F(a0)`. An integer embeds in the prime subfield; there is no
packed-integer encoding of extension elements. Field scalars have no length
or indexing; use `.coefficients` for their polynomial coordinates. Lists and
nested lists of field scalars from `F` can construct CD elements and arrays
over `F`, each scalar supplying one coefficient.

```python
import pyfastcd as fcd

F = fcd.FiniteField(char=13, deg=5)
A = fcd.Algebra(level=4, field=F)
c = F([1, 2, 0, 0, 0])
x = A.scalar(c) + A.basis(3)
q = x.quadratic_norm()       # Element with q.algebra is F
assert x * x.conjugate() == A.scalar(q)
```

Pickles record the polynomial, so an unpickled field is the same field even
when it was chosen automatically. Field objects are immutable and can be
shared between algebras and threads. Do not modify an array's storage while
an operation on it is running; large batches release the GIL.

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

Create an `Element` with `A(...)` or `F(...)` and an `ElementArray` with
the parent's `array` or `asarray` method. Direct class construction raises `TypeError`. Elements are
immutable and hashable: equal elements hash equal, and an element equal to a
real scalar hashes like that scalar. Arrays are unhashable and inherit
writeability from their NumPy storage.

### Properties

- `algebra`

  The parent `Algebra` or `FiniteField`.

- `dimension`, `level`, `dtype`

  The dimension, doubling level, and coefficient dtype of the parent.

- `shape`

  `parent.coefficient_shape` for an element, or
  `batch_shape + parent.coefficient_shape` for an array.

- `batch_shape`

  The leading dimensions before the complete coefficient record. Available
  on `ElementArray` only. An extension's polynomial axis is not a batch axis.

- `real`

  Coordinate zero as a `float` over R, `complex` over C, or field `Element`
  over finite fields. Real/complex arrays expose an ndarray view; finite
  arrays expose a same-field `ElementArray` view. For field scalars,
  `.real` is the scalar itself (copied for an immutable element).

- `imag`

  A new element or array with coordinate zero cleared.

- `coefficients`

  A NumPy view of the coefficient storage. The element view is read-only;
  the array view shares its underlying data.

### Methods

These methods allocate new results, except when a NumPy view is requested
with `to_numpy(copy=False)`. `Element` methods take positional `Element`
operands only, while `ElementArray` methods accept `Element` or
`ElementArray` operands and broadcast; to combine an element with an array,
call the method on the array or use an operator.

Methods that take another record require it to be a complete element or
array from the same parent; unlike the operators, they do not embed scalars,
and an unsupported operand raises `TypeError` or `ValueError`. Use
`A.scalar(r)` when a scalar element is intended. `inner` and
`quadratic_norm` over a finite field return field scalars whose parent is
`F`; every other result has the receiver's parent.

- `copy()`

  Return an independent copy.

- `conjugate()`

  Return `(x_0, -x_1, ..., -x_(N-1))`. This CD involution fixes every
  coefficient, including complex coefficients; it does not apply complex
  conjugation or finite-field Frobenius to those coefficients.

- `imag_part()`

  Return a copy with coordinate zero cleared.

- `inner(other)`

  Return the bilinear coefficient pairing `sum_i x_i other_i`: the Euclidean
  inner product over R, and the bilinear form without conjugation over C
  and finite fields. Return types follow the `quadratic_norm` table.

- `quadratic_norm()`

  Return the canonical field-valued quadratic form `Q(x) = sum_i x_i**2`,
  satisfying `x * x.conjugate() == A.scalar(Q(x))` in exact arithmetic.
  For a field scalar `c`, this is `c*c`.

  This is the CD quadratic form over F. It is distinct from the relative
  finite-field norm from an extension to its prime subfield.

  | Coefficient field | Element result | ElementArray result |
  |---|---|---|
  | R | `float` | `float64` ndarray |
  | C | `complex` | `complex128` ndarray |
  | finite field F | `Element`, parent F | `ElementArray`, parent F |

- `norm_squared()`

  An alias for `quadratic_norm()`. Over R it uses scaled accumulation. Over
  C this is not `sum(abs(x_i)**2)`.

- `norm()`

  Return the real Euclidean norm using scaled accumulation.
  `norm()`, `abs(x)`, and `normalized()` require a real parent. They raise
  `TypeError` for complex/finite parents; use `quadratic_norm()` there.

- `normalized()`

  Return `x / norm(x)`. Raise `ZeroDivisionError` for an exactly zero input.

- `reciprocal()`

  Return `conjugate(x) / Q(x)`. Raise `ZeroDivisionError` when `Q(x) == 0`,
  including nonzero isotropic elements over C and finite fields. Batch
  reciprocals and division check all denominators before producing results.

- `square()`

  Return `x^2` using the linear-time formula.

- `imag_cross(other)`

  Return `(uv - vu) / 2` for the imaginary parts. Input scalars are ignored,
  and the output scalar is zero.

- `commutator(other)`

  Return `x*other - other*x`.

- `anticommutator(other)`

  Return `x*other + other*x` using the linear-time formula.

- `associator(y, z)`

  Return `(x y) z - x (y z)`.

- `to_numpy(*, copy=True)`

  Return the coefficient ndarray. The `copy` argument must be a Python
  `bool`.

### Operators

Operators accept same-parent elements/arrays and coefficient-field scalars. Results
are new objects, and arrays broadcast their leading axes.

- `+x`, `-x`

  Copy an element or negate its coordinates.

- `x + y`, `x - y`, `x + r`, `r - x`, ...

  Add or subtract coordinates, embedding scalars as `r * e_0`.

- `x * y`

  Compute the ordered product with the automatic algorithm.

- `r * x`, `x * r`

  Scale every coordinate.

- `x / r`

  Divide by a coefficient-field scalar. A zero divisor raises `ZeroDivisionError`.

- `x ** k`

  Compute an integer power by binary exponentiation. Negative powers use
  the reciprocal. A real `Element` limits the exponent to the signed 64-bit
  range; complex and finite-field elements and all arrays accept any Python
  integer.

- `abs(x)`

  Return the real Euclidean coefficient norm; other parents raise `TypeError`.

- `x == y`, `x != y`

  Compare whole elements exactly, returning a Boolean array for batches.
  Supported real scalars include real zero-dimensional ndarrays. An
  unsupported operand makes the fastCD comparison method return
  `NotImplemented`, allowing reflected comparison. With ordinary Python
  operands such as `str`, `complex`, `None`, and `list`, `==` is `False`
  and `!=` is `True`. A fastCD object of another algebra compares unequal
  everywhere, giving `False` for an element and an all-`False` array of the
  broadcast batch shape for an
  array. Incompatible batch shapes raise `ValueError`.

  Over C, comparison with a scalar is exact in the same way. Over a finite
  field, integers are not reduced modulo `char` for comparison, so
  `F(1) != char + 1`; this keeps equality consistent with hashing. A field
  scalar compares equal to the element that embeds it in a CD algebra over
  the same field, and fields with different presentations compare unequal.

  NumPy comparison dispatch with an unsupported operand, such as a
  nonscalar ndarray or a complex NumPy scalar, raises `TypeError`. Scalar
  comparisons can also propagate exceptions from the operand's comparison
  protocol; for example, comparing with `Decimal("sNaN")` raises
  `decimal.InvalidOperation` when that decimal trap is enabled.

### Indexing and iteration

- `len(x)`

  Return `N` for a CD element. A coefficient-field scalar has no length.

- Element iteration and indexing yield coefficients as floats; a slice
  returns a tuple of floats for the default real parent. Complex parents
  yield `complex` values; finite CD parents yield coefficient-field
  `Element` values. A field scalar itself has no sequence indexing: use
  `.coefficients` for its polynomial coordinates.
- Array iteration and indexing select complete records along batch axes.
  `X[i]` returns an `Element` copy for a single record and an `ElementArray`
  slice otherwise. As in NumPy, an array with an empty batch shape has no
  length and cannot be iterated.
- `X[key] = value` assigns a scalar, a same-parent object, or a coefficient
  array.
- `bool(x)` is false only for a zero element. `bool(X)` raises `ValueError`.

---

## Arrays, NumPy, and DLPack

### Storage

- Coefficients have shape `batch_shape + parent.coefficient_shape`. Only
  leading batch axes broadcast; neither the CD nor the polynomial axis does.
- Zero-copy sharing through `A.asarray` requires the parent's native dtype and aligned
  coefficient storage. Over R the final stride must be one coefficient,
  except that dimension 1 also permits zero stride and empty C-contiguous
  arrays need no final-stride check. Over C and finite fields, non-singleton
  trailing coefficient axes must have C-order strides; singleton axes and
  empty arrays need no trailing-stride check. Leading axes may be strided,
  reversed, or broadcast when alignment is preserved. Numeric inputs with an incompatible dtype or
  layout are converted and copied unless `copy=False` forbids it. Finite
  extension records require contiguous trailing `(N, deg)` axes; `copy=False`
  additionally requires canonical residues in `[0, char)`. When copying,
  integer coordinates are reduced modulo `char` before being stored as
  `uint32`. Every operation re-checks that finite storage still holds
  canonical residues before it writes any output. Empty leading batch axes
  are supported for every coefficient field.
- Inputs are read-only and may share storage, as in `X[:-1] * X[1:]` or
  `X * X[:1]`, as long as each pair of corresponding elements is identical
  or disjoint. Partially overlapping elements raise `ValueError`.

### NumPy interfaces

- `Element` exposes a read-only buffer, which NumPy can read directly.
- `ElementArray.__array__` returns a view unless a dtype change or
  `copy=True` requires a copy.
- Both classes support `np.add`, `np.subtract`, `np.multiply`, `np.true_divide`
  by a real scalar, `np.power` with an integer exponent, `np.negative`,
  `np.positive`, `np.square`, `np.conjugate`, `np.absolute`, `np.equal`, and
  `np.not_equal`.
  Only `np.multiply` accepts `out=`; with an ndarray `out` it returns that
  ndarray, whereas `A.multiply` returns a new `ElementArray` wrapper.
- Other ufuncs and methods, including `reduce`, `accumulate`, and `where=`,
  return `NotImplemented`.
- Functions such as `np.mean` and `np.stack`, which are not ufuncs,
  read coefficients through `__array__` or the buffer and use ordinary NumPy
  array semantics. `np.mean(x)` returns a NumPy scalar when all axes are
  reduced; `np.stack` returns a plain ndarray.

### DLPack interfaces

- `ElementArray.__dlpack__(*, stream=None, max_version=None, dl_device=None, copy=None)`

  Export the coefficient storage through DLPack.

- `ElementArray.__dlpack_device__()`

  Report the DLPack device.

- `A.array` and `A.asarray` import host-accessible DLPack producers through
  `np.from_dlpack`. Requiring `copy=False` for a generic DLPack producer
  needs NumPy 2.1 or newer; older supported NumPy versions raise `ValueError`
  for that request.

### Copying and serialization

Pickling, `copy.copy`, and `copy.deepcopy` create independent coefficient
storage.

---

## Exceptions

- `TypeError`

  Unsupported operand or dtype, an invalid index type for an element or
  basis index (array indexing follows NumPy and raises `IndexError`), direct
  class construction, a non-`bool` copy flag, or a non-string algorithm name.

- `ValueError`

  Invalid dimension, mismatched parents, a shape or broadcast failure,
  incompatible layout or writeability, overlapping buffers, an unavailable
  algorithm, or ambiguous array truth.

- `IndexError`

  A basis or coefficient index outside the valid range.

- `ZeroDivisionError`

  A zero scalar divisor, real normalization of zero, or a reciprocal/negative
  power whose quadratic norm is zero. Nonzero isotropic inputs are included.

- `OverflowError`

  A dimension, scalar, or storage size beyond the implementation limits, or
  a real `Element` exponent outside the signed 64-bit range.

- `MemoryError`, `BufferError`, `RuntimeError`

  Allocation failure, an unsatisfiable buffer or DLPack request, or an
  unexpected native failure.

Native operations do not raise on NaN, infinity, or overflow. NumPy's error
settings apply to the coordinate operations it performs on `ElementArray`
operands: addition, subtraction, scaling, scalar division, and equality.
`Element` arithmetic is native and never raises on these conditions.

Free-threaded CPython builds are not supported. The extension does not
declare free-threading compatibility, so importing `pyfastcd` on such an
interpreter re-enables the global interpreter lock.

---

## Versioning and stability

pyfastcd follows semantic versioning, and its version string is the version of
the fastCD C library it wraps. The public surface is the names in
`pyfastcd.__all__` together with the members documented on this page; modules
and names with a leading underscore are private and may change without notice.
Public API removals and incompatible calling-contract changes require a major
version change and a prior deprecation note in `CHANGELOG.md`.

Calls written for version 1.0.0 behave the same way in every later 1.x
release: the same classes, argument rules, exceptions and results. The only
permitted differences are floating-point rounding and automatic algorithm
choices: AUTO choices and internal arithmetic schedules may change in minor
or patch releases to improve performance or correctness, and floating-point
results are not guaranteed bitwise identical across versions, even when the
same algorithm is named explicitly. Relevant numerical changes are recorded
in `CHANGELOG.md`. Workspace lengths are implementation details obtained
through the C query functions.

## Typing

The type stubs give bare `Algebra`, `Element` and `ElementArray` annotations
real coefficient and result types. Explicit `char=0, deg=2` or `field=F`
arguments infer complex or finite-field types; integer `char` and `deg`
variables infer dynamic types, because a variable characteristic can select
different fields. `Algebra[...]`, `Element[...]` and `ElementArray[...]` may
be used in annotations; subscripting has no runtime effect, and only `char`,
`deg` and `field` select coefficients.
