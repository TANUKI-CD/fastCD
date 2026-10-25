# Python API reference

## pyfastcd

The public names are `Algebra`, `BasisProduct`, `Element`, `ElementArray`,
`cd_element`, and `__version__`. Modules whose names begin with an underscore
are private. Python callers use exceptions rather than native status codes
and do not manage workspaces.

Below, `A` is an algebra, `N = A.dimension`, `x, y, z` are elements, `X` is
an element array, and `r` is a real scalar.

### Attributes

- `__version__: str`

  The version of the library.

### Functions

- `cd_element(coefficients) -> Element`

  Construct an element, inferring its dimension from the length of the
  coefficient vector. The input must be a one-dimensional sequence or array,
  not an iterator, and its length must be a positive power of two; other
  inputs raise `ValueError`.

### Scalars

Real scalars are `bool`, `int`, `float`, `Decimal`, `Fraction`, real
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
Algebra(*, dimension=None, level=None)
```

Create an immutable algebra, cached by dimension. Algebras cannot be
subclassed.

**Parameters**

- `dimension`: a positive power of two.
- `level`: the doubling level, where `dimension = 2**level`.

Supply exactly one parameter. Boolean values are rejected.

### Properties

- `dimension: int`

  The number of real coefficients, `N`.

- `level: int`

  The doubling level, `log2(N)`.

- `dtype`

  The coefficient dtype, `numpy.dtype('float64')`.

- `default_algorithm: str`

  The algorithm selected by `"auto"` for two `Element` operands or a
  one-record batch. The choice is shared with C.

- `default_batch_algorithm: str`

  The algorithm selected by `"auto"` for batches of two or more records.
  The choice is shared with C.

### Methods

- `A(value=0.0) -> Element`

  Embed a real scalar in coordinate zero, copy a same-parent `Element`, or
  copy exactly `N` real coefficients from a sequence, iterator, or buffer.
  Unordered inputs such as `set` and `dict` raise `TypeError`. Coefficients
  are never padded or truncated.

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

  Copy values into writable, aligned, C-contiguous `float64` storage.

- `asarray(values, *, copy=None) -> ElementArray`

  Share compatible storage when `copy=None`, always copy when `copy=True`,
  and require zero-copy sharing when `copy=False`. If zero-copy sharing is
  required but impossible, raise `ValueError`.

- `multiply(x, y, *, out=None, algorithm="auto") -> Element | ElementArray`

  Multiply with `"auto"`, `"direct"`, `"cariow_cariowa"`, or `"quasilinear"`.
  Automatic selection uses `default_algorithm` for single products and
  `default_batch_algorithm` for batches of two or more records. Real scalars
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
`Element` (or, for `multiply` only, a real scalar) and no `out` is given;
otherwise they return an
`ElementArray`, the same wrapper when `out` is an `ElementArray`, or a new
wrapper around a supplied ndarray. `out` must have the exact broadcast result
shape, native `float64`, writable aligned C-contiguous storage, and no
overlap with any input.

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

Create an `Element` with `A(...)` and an `ElementArray` with `A.array` or
`A.asarray`. Direct class construction raises `TypeError`. Elements are
immutable and hashable: equal elements hash equal, and an element equal to a
real scalar hashes like that scalar. Arrays are unhashable and inherit
writeability from their NumPy storage.

### Properties

- `algebra`

  The parent algebra.

- `dimension`, `level`, `dtype`

  The dimension, doubling level, and coefficient dtype of the parent.

- `shape`

  `(N,)` for an element, or `batch_shape + (N,)` for an array.

- `batch_shape`

  The leading dimensions, `shape[:-1]`. Available on `ElementArray` only.

- `real`

  Coordinate zero as a `float` for an element, or a view with shape
  `batch_shape` for an array.

- `imag`

  A new element or array with coordinate zero cleared.

- `coefficients`

  A NumPy view of the coefficient storage. The element view is read-only;
  the array view shares its underlying data.

### Methods

These methods allocate new results. Element-valued results have the same
class as the receiver. `Element` methods take positional `Element` operands
only, while `ElementArray` methods accept `Element` or `ElementArray`
operands and broadcast; to combine an element with an array, call the method
on the array or use an operator.

- `copy()`

  Return an independent copy.

- `conjugate()`

  Return `(x_0, -x_1, ..., -x_(N-1))`.

- `imag_part()`

  Return a copy with coordinate zero cleared.

- `inner(other)`

  Return the Euclidean coefficient inner product as a `float` or a
  `float64` array.

- `norm_squared()`

  Return `<x,x>` using scaled evaluation.

- `norm()`

  Return the square root of `<x,x>` using scaled evaluation.

- `normalized()`

  Return `x / norm(x)`. Raise `ZeroDivisionError` for an exactly zero input.

- `reciprocal()`

  Return `conjugate(x) / <x,x>`. Raise `ZeroDivisionError` for an exactly
  zero input.

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

Operators accept same-parent elements or arrays and real scalars. Results
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

  Divide by a real scalar. A zero divisor raises `ZeroDivisionError`.

- `x ** k`

  Compute an integer power by binary exponentiation. Negative powers use
  the reciprocal.

- `abs(x)`

  Return the Euclidean coefficient norm.

- `x == y`, `x != y`

  Compare whole elements exactly, returning a Boolean array for batches.
  An operand that is neither a same-parent fastCD object nor a real scalar,
  such as a `str`, `complex`, `None`, or `list`, returns `NotImplemented`,
  so `==` is `False` and `!=` is `True` rather than raising. A fastCD object
  of another algebra compares unequal everywhere, giving `False` for an
  element and an all-`False` array of the broadcast batch shape for an
  array, as NumPy does for incomparable operands. Two cases raise: a raw
  `ndarray` operand is routed through `np.equal` and raises `TypeError`, and
  a non-broadcastable shape raises `ValueError`.

### Indexing and iteration

- `len(x)`

  Return `N` for an element.

- Element iteration and indexing yield coefficients as floats; a slice
  returns a tuple of floats.
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

- Coefficients have shape `batch_shape + (N,)`. Only leading axes broadcast.
- Zero-copy sharing through `A.asarray` requires native `float64`, NumPy
  alignment, and a unit final stride. Other inputs are converted and copied.
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
- NumPy functions that are not ufuncs, such as `np.mean` and `np.stack`,
  read coefficients through `__array__` or the buffer and act coordinatewise,
  returning plain ndarrays rather than fastCD objects.

### DLPack interfaces

- `ElementArray.__dlpack__(*, stream=None, max_version=None, dl_device=None, copy=None)`

  Export the coefficient storage through DLPack.

- `ElementArray.__dlpack_device__()`

  Report the DLPack device.

- `A.array` and `A.asarray` import host-accessible DLPack producers through
  `np.from_dlpack`.

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

  A zero scalar divisor, or normalization, a reciprocal, or a negative
  power of a zero element.

- `OverflowError`

  A dimension, scalar, or storage size beyond the implementation limits, or
  an `Element` exponent outside the signed 64-bit range.

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
the fastCD C library it wraps. The public surface is the six names in
`pyfastcd.__all__` together with the members documented on this page; modules
and names with a leading underscore are private and may change without notice.
Public API removals and incompatible calling-contract changes require a major
version change and a prior deprecation note in `CHANGELOG.md`.

AUTO algorithm choices and internal arithmetic schedules may change in minor
or patch releases to improve performance or correctness. Workspace lengths
are implementation details obtained through the C query functions.
Floating-point results are not guaranteed bitwise identical across versions,
even when selecting the same named algorithm explicitly. Relevant numerical
changes are recorded in `CHANGELOG.md`.
