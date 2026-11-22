# C API reference

fastCD's C library multiplies Cayley-Dickson elements and computes the other
operations of the algebras over the real and complex numbers and over finite
fields of odd characteristic. Its interface is declared in three public headers:
`<fastcd/fastcd.h>` for real coefficients, together with the status codes, the
algorithm selector, and the workspace type that every function uses, and
`<fastcd/complex.h>` and `<fastcd/finite.h>` for complex and finite-field
coefficients, which both include `<fastcd/arithmetic.h>`. The headers are C11
and carry C++ linkage guards.

Every function follows the same conventions: it reports success or failure
through its return value and writes its outputs only on success, and it never
allocates memory on the heap or keeps a pointer after it returns, so storage for
temporary values comes from the caller as a *workspace*, whose length a query
function reports. This page documents every function and the rules that govern
them, while the [C quick start](quickstart-c.md) introduces them by example and
is the place to begin. The section on `fastcd.h` covers linking, the types, the
calling contract, and the compatibility guarantee before the functions
themselves, and the section on arithmetic across coefficient fields covers the
complex and finite-field interface.

## fastcd.h

### Linking

A CMake project links the target `fastcd::fastcd`, which supplies the include
path, the library, and its dependencies, while a program linked by hand uses
`-lfastcd -lm`.

Every public function is declared with the attribute `FASTCD_API`, which is
empty for the static library and marks the functions for import from the shared
library otherwise. The CMake target defines `FASTCD_SHARED` for the consumers of
a shared library, and the library build itself defines
`FASTCD_BUILDING_LIBRARY`, so only a program that links the shared library
without CMake defines `FASTCD_SHARED` itself.

---

### Types and constants

```c
typedef enum {
    FASTCD_SUCCESS = 0,
    FASTCD_INVALID_DIMENSION,      /* zero or not a power of two */
    FASTCD_INVALID_LEVEL,          /* 2^level does not fit size_t */
    FASTCD_INVALID_BASIS_INDEX,    /* index >= dimension */
    FASTCD_INSUFFICIENT_WORKSPACE, /* non-null scratch shorter than required */
    FASTCD_ZERO_NORM,              /* division by zero in normalize,
                                      reciprocal or divide */
    FASTCD_OVERLAPPING_BUFFERS,    /* aliasing contract violated */
    FASTCD_INVALID_STRIDE,         /* batch stride smaller than dimension */
    FASTCD_SIZE_OVERFLOW,          /* an extent does not fit size_t */
    FASTCD_NULL_POINTER,           /* required pointer is null */
    FASTCD_UNSUPPORTED_ALGORITHM,  /* unknown algorithm or operation, or an
                                      algorithm unavailable for the call */
    FASTCD_INVALID_COUNT           /* too few elements for the operation */
} FastCDStatus;

typedef enum {
    FASTCD_ALGORITHM_AUTO = 0,
    FASTCD_ALGORITHM_DIRECT,
    FASTCD_ALGORITHM_CARIOW_CARIOWA,
    FASTCD_ALGORITHM_QUASILINEAR
} FastCDAlgorithm;

#define FASTCD_CARIOW_CARIOWA_MAX_DIMENSION ((size_t)1024)

typedef struct FastCDWorkspaceF64 {
    double* data;
    size_t length;
} FastCDWorkspaceF64;

typedef struct FastCDBasisProduct {
    size_t index;
    int sign;
} FastCDBasisProduct;

typedef struct FastCDSumMapF64 {
    const double* x_shift;
    const double* x_scale;
    const double* y_shift;
    const double* y_scale;
} FastCDSumMapF64;
```

Each type has one role, described in the section named here:

- `FastCDStatus` is the result of every arithmetic function and of every query
  with an output parameter, as described under [Calling
  contract](#calling-contract).
- `FastCDAlgorithm` selects a multiplication algorithm (see [Algorithm
  selection](#algorithm-selection)), and `FASTCD_CARIOW_CARIOWA_MAX_DIMENSION`
  is the largest dimension at which Cariow-Cariowa is available.
- `FastCDWorkspaceF64` describes caller storage for temporary values (see
  [Workspaces](#calling-contract) under the calling contract).
- `FastCDBasisProduct` holds the index and the sign of a product of basis
  elements (see [Indexed basis operations](#indexed-basis-operations)).
- `FastCDSumMapF64` is an affine map applied to records as a fused sum reads
  them (see [Fused sums](#fused-sums)).

---

### Calling contract

The following rules apply to every function in `fastcd.h` unless its entry says
otherwise.

**Parameters**

- `dimension`, written as `N` below, is a nonzero power of two. An element is
  stored as a *record* of `N` contiguous `double` coefficients in the standard
  coordinates, with the scalar part in coordinate zero, and [Conventions and
  results](mathematics.md) gives the multiplication convention and the basis
  signs.
- Lengths and strides count `double` elements.
- Inputs, outputs, and workspaces belong to the caller: no function allocates on
  the heap or retains a pointer, and the three string queries return immutable
  static storage.

**Returns**

- Arithmetic functions and queries with output parameters return a
  `FastCDStatus`. An error leaves every output unchanged, and when several
  arguments are invalid, which error is reported is unspecified.
- Every function that takes arrays can return `FASTCD_INVALID_DIMENSION`,
  `FASTCD_SIZE_OVERFLOW`, or `FASTCD_NULL_POINTER`, and functions with array
  inputs also check for overlap. Further errors are listed with the functions
  that return them.
- NaN, infinity, signed zero, overflow, underflow, and cancellation are
  numerical results, not errors, and different algorithms and specialized
  formulas can round differently without returning an error.

**Aliasing**

An exact alias has the same starting pointer as the array it aliases and covers
a complete record, and partial overlap is rejected with
`FASTCD_OVERLAPPING_BUFFERS`. Read-only inputs may be the same complete array,
and the two inputs of a batch may share storage under the rule given with the
batch functions. Outputs must be disjoint from every input, except in these
cases:

- `fastcd_add_f64` and `fastcd_sub_f64` allow `out == x`, `out == y`, or both.
- `fastcd_neg_f64`, `fastcd_scale_f64`, `fastcd_conjugate_f64`,
  `fastcd_imag_part_f64`, `fastcd_normalize_f64`, `fastcd_reciprocal_f64`, and
  `fastcd_square_f64` allow `out == x`.

The one-value outputs of `fastcd_inner_f64`, `fastcd_norm_squared_f64`, and
`fastcd_norm_f64`, and the outputs of `fastcd_basis_lmul_f64` and
`fastcd_basis_rmul_f64`, must also be disjoint from the inputs.

**Workspaces**

Because the library never allocates, a function that needs temporary storage
takes a workspace: the caller obtains the required length from the matching
query, allocates that many `double` elements, and describes the storage with a
`FastCDWorkspaceF64`.

- When the required length is zero, `workspace` may be `NULL`.
- Otherwise a null descriptor or data pointer returns `FASTCD_NULL_POINTER`, and
  too little storage returns `FASTCD_INSUFFICIENT_WORKSPACE`.
- The queried prefix of the storage must be disjoint from every array, and the
  descriptor object itself must not overlap an input or output, a precondition
  that the library does not check.
- The contents of the workspace have no meaning after the call.

**Threads**

Calls can share read-only inputs across threads as long as their outputs and
workspaces are disjoint. Each batch call runs on one thread and reuses one
workspace for all of its products.

**SIMD kernels**

On GCC and Clang targets with SSE2 or AArch64 SIMD, a batch of two or more
products evaluates two products at a time, one in each SIMD lane, at these
dimensions:

- quasilinear multiplication and Cariow-Cariowa at dimensions 16 through 512;
- direct multiplication at dimensions 64 through 512.

These paired kernels copy each pair of operands into an interleaved layout in a
fixed 40 KiB buffer on the stack, so caller storage needs no extra alignment,
but a batch call uses that much stack. When a batch has an odd number of
products, the last one is computed as a single product, and at other dimensions,
and with other compilers or targets, batches compute one product at a time with
the single-product kernels and the caller's workspace.

On the same targets, a single quasilinear product above dimension 16 places the
two recursive calls of its top level in the two SIMD lanes (see [How quasilinear
multiplication works](mathematics.md#how-quasilinear-multiplication-works)) and
continues through the paired kernel, using the caller's workspace, whose query
reports the larger requirement of this form. Single direct and Cariow-Cariowa
products use scalar kernels, which the compiler may vectorize.

The scalar and SIMD kernels perform the same operations in the same order, so
they produce identical finite results for the same algorithm, although NaN
payloads may differ.

---

### Compatibility

Every later 1.x release keeps the calling contract of version 1.0.0. Code
written against 1.0.0 compiles against any later 1.x release, and each of its
calls keeps:

- the same parameters, with the same meaning for each of them;
- the same status code for each error condition;
- the same aliasing and workspace rules;
- the same mathematical result, up to floating-point rounding.

A release may change:

- the algorithm that `AUTO` selects at a given dimension;
- workspace lengths, which programs therefore obtain from the query functions at
  run time rather than hard-coding them;
- the arithmetic schedule of an algorithm, and with it the last bits of a
  floating-point result, even when the algorithm is selected explicitly.

`CHANGELOG.md` records every change that affects numerical results.

---

### Library information

- `const char* fastcd_version_string(void)`

  Return the version string of the linked library, such as `"1.5.1"`.

- `const char* fastcd_status_string(FastCDStatus status)`

  Return a fixed English message for a status, or `"Unknown fastCD status"` for
  an unknown value.

- `const char* fastcd_algorithm_string(FastCDAlgorithm algorithm)`

  Return `"auto"`, `"direct"`, `"cariow_cariowa"`, or `"quasilinear"`, or
  `"unknown"` for an unknown value.

---

### Dimension utilities

- `bool fastcd_valid_dimension(size_t dimension)`

  Return whether `dimension` is a nonzero power of two, in constant time.

- `FastCDStatus fastcd_dimension(size_t level, size_t* dimension)`

  Write `1 << level` to `dimension`, or return `FASTCD_INVALID_LEVEL` if the
  shift would overflow.

- `FastCDStatus fastcd_level(size_t dimension, size_t* level)`

  Write `log2(dimension)` to `level`, or return `FASTCD_INVALID_DIMENSION` if
  `dimension` is zero or not a power of two.

---

### Algorithm selection

`FASTCD_ALGORITHM_AUTO` chooses an algorithm by dimension, and in this release
it uses these rules, whose basis in measurements is described under [How the
results set the defaults](benchmark.md#how-the-results-set-the-defaults):

- Single products use direct multiplication at `N = 4, 8, 16` and quasilinear
  multiplication at every other dimension. This rule also applies to batches of
  one product and to the products inside `fastcd_associator_f64` and the two
  operator commutators.
- Batches of two or more products use direct multiplication at `N = 8`,
  Cariow-Cariowa at `N = 16`, and quasilinear multiplication at every other
  dimension. Builds without the two-lane SIMD kernels use the single-product
  rule for batches.
- [Fused sums](#fused-sums) use direct multiplication through `N = 8` and
  quasilinear multiplication above.

`fastcd_imag_cross_f64` and `fastcd_commutator_f64` always use the quasilinear
recursion, and the following functions report the choices of `AUTO`:

- `FastCDStatus fastcd_default_algorithm(size_t dimension, FastCDAlgorithm*
  algorithm)`

  Write the concrete algorithm that `AUTO` selects for a single product.

- `FastCDStatus fastcd_default_batch_algorithm(size_t dimension,
  FastCDAlgorithm* algorithm)`

  Write the concrete algorithm that `AUTO` selects for a batch of at least two
  products.

- `FastCDStatus fastcd_default_sum_algorithm(size_t dimension, FastCDAlgorithm*
  algorithm)`

  Write the concrete algorithm that `AUTO` selects for a fused sum.

- `bool fastcd_algorithm_supports_dimension(FastCDAlgorithm algorithm, size_t
  dimension)`

  Return `false` for an invalid dimension, an unknown algorithm, or
  Cariow-Cariowa above dimension 1024, and `true` otherwise.

---

### Workspace queries

Each query writes the length, in doubles, of the workspace that one function or
group of functions needs, and it can return `FASTCD_NULL_POINTER`,
`FASTCD_INVALID_DIMENSION`, `FASTCD_SIZE_OVERFLOW`, and, where an algorithm is
involved, `FASTCD_UNSUPPORTED_ALGORITHM`. The lengths can change between
releases (see [Compatibility](#compatibility)), so programs obtain them from the
queries rather than from the formulas below, which describe the default build of
this release.

The lengths do not depend on the number of products in a batch, and those of the
products are `O(N)` doubles:

- Direct multiplication needs nothing for `N <= 16` and `2N - 32` doubles above
  that.
- Quasilinear multiplication needs nothing for `N <= 16`, and above that it
  needs `5N - 63` doubles on targets with the two-lane SIMD kernels and
  `2N - 32` otherwise.
- Cariow-Cariowa needs nothing through `N = 64`, where it uses straight-line
  kernels, and `2N` doubles above that, where its table-driven kernel runs.
- The associator and the operator commutators need `2N` doubles more than their
  products, for two intermediate elements.
- A fused sum needs one length for both layouts and both algorithms, which grows
  like `N log N`: 1,487 doubles at `N = 8`, 2,991 at 16, 13,551 at 64, and
  60,399 at 256. The covariance needs `13N` doubles more.

- `FastCDStatus fastcd_mul_workspace_size(size_t dimension, size_t* length)`

  Size the workspace for `fastcd_mul_f64` and `fastcd_mul_batch_f64`. The length
  covers both automatic selections, for single products and for batches.

- `FastCDStatus fastcd_mul_algorithm_workspace_size(size_t dimension,
  FastCDAlgorithm algorithm, size_t* length)`

  Size the workspace for `fastcd_mul_with_algorithm_f64` and
  `fastcd_mul_batch_with_algorithm_f64`.

- `FastCDStatus fastcd_mul_sum_workspace_size(size_t dimension, size_t* length)`

  Size the workspace for the four `fastcd_mul_sum` functions.

- `FastCDStatus fastcd_covariance_workspace_size(size_t dimension, size_t*
  length)`

  Size the workspace for `fastcd_covariance_f64`.

- `FastCDStatus fastcd_imag_cross_workspace_size(size_t dimension, size_t*
  length)`

  Size the workspace for `fastcd_imag_cross_f64` and `fastcd_commutator_f64`.

- `FastCDStatus fastcd_associator_workspace_size(size_t dimension, size_t*
  length)`

  Size the workspace for `fastcd_associator_f64`.

- `FastCDStatus fastcd_lop_commutator_workspace_size(size_t dimension, size_t*
  length)`

  Size the workspace for `fastcd_lop_commutator_apply_f64`.

- `FastCDStatus fastcd_rop_commutator_workspace_size(size_t dimension, size_t*
  length)`

  Size the workspace for `fastcd_rop_commutator_apply_f64`.

---

### Element construction

- `FastCDStatus fastcd_scalar_f64(double* out, size_t dimension, double value)`

  Set `out[0] = value` and every other coordinate to `+0.0`.

- `FastCDStatus fastcd_basis_f64(double* out, size_t dimension, size_t index)`

  Set `out[index] = 1.0` and every other coordinate to `+0.0`, or return
  `FASTCD_INVALID_BASIS_INDEX` when `index >= N`.

---

### Coordinate arithmetic

These functions act on each coordinate separately: they take `O(N)` time, use no
workspace, and perform one `double` operation per coordinate.

- `FastCDStatus fastcd_add_f64(double* out, const double* x, const double* y,
  size_t dimension)`

  Set `out[i] = x[i] + y[i]`.

- `FastCDStatus fastcd_sub_f64(double* out, const double* x, const double* y,
  size_t dimension)`

  Set `out[i] = x[i] - y[i]`.

- `FastCDStatus fastcd_neg_f64(double* out, const double* x, size_t dimension)`

  Set `out[i] = -x[i]`.

- `FastCDStatus fastcd_scale_f64(double* out, const double* x, double scalar,
  size_t dimension)`

  Set `out[i] = scalar * x[i]`.

---

### Structural operations

These functions compute the conjugate, the imaginary part, the inner product,
the norms, and the reciprocal; they take `O(N)` time and use no workspace.

**Numerical behavior**

- Norms return NaN if any coordinate is NaN, otherwise `+inf` if any coordinate
  is infinite, and `+0.0` for an all-zero input.
- Normalization and reciprocals return `FASTCD_ZERO_NORM` only for an exactly
  zero finite input, including signed zeros.
- For normalization and reciprocals, any NaN makes every output NaN, and with an
  infinity present but no NaN, infinite coordinates become NaN and finite
  coordinates become signed zeros.

**Functions**

- `FastCDStatus fastcd_conjugate_f64(double* out, const double* x, size_t
  dimension)`

  Set `out` to `(x[0], -x[1], ..., -x[N-1])`.

- `FastCDStatus fastcd_imag_part_f64(double* out, const double* x, size_t
  dimension)`

  Set `out` to `(+0.0, x[1], ..., x[N-1])`.

- `FastCDStatus fastcd_inner_f64(double* value, const double* x, const double*
  y, size_t dimension)`

  Write `sum_i x[i] y[i]` to `value`, accumulating in coordinate order without
  scaling.

- `FastCDStatus fastcd_norm_squared_f64(double* value, const double* x, size_t
  dimension)`

  Write `sum_i x[i]^2` to `value`, using scaled accumulation.

- `FastCDStatus fastcd_norm_f64(double* value, const double* x, size_t
  dimension)`

  Write the Euclidean norm to `value`, using scaled accumulation.

- `FastCDStatus fastcd_normalize_f64(double* out, const double* x, size_t
  dimension)`

  Set `out` to `x / norm(x)`, using scaled division.

- `FastCDStatus fastcd_reciprocal_f64(double* out, const double* x, size_t
  dimension)`

  Set `out` to `conjugate(x) / norm_squared(x)`, using scaled division. This is
  the two-sided reciprocal `x^(-1)`, not a general division operation.

---

### Multiplication

- `FastCDStatus fastcd_mul_f64(double* out, const double* x, const double* y,
  size_t dimension, const FastCDWorkspaceF64* workspace)`

  Set `out` to `xy`, using the automatic algorithm. A NaN coordinate in either
  operand makes every output coordinate NaN, and an infinite coordinate makes
  every output coordinate non-finite, where which coordinates become NaN rather
  than infinite depends on the algorithm.

- `FastCDStatus fastcd_mul_with_algorithm_f64(double* out, const double* x,
  const double* y, size_t dimension, FastCDAlgorithm algorithm, const
  FastCDWorkspaceF64* workspace)`

  Set `out` to `xy`, using `algorithm`, which may be `AUTO`. An unknown or
  unavailable algorithm returns `FASTCD_UNSUPPORTED_ALGORITHM`.

- `FastCDStatus fastcd_square_f64(double* out, const double* x, size_t
  dimension)`

  Set `out` to `x^2 = (alpha^2 - <X, X>) + 2 alpha X`, where `alpha = x[0]` is
  the scalar part and `X` the imaginary part, in `O(N)` time. If a finite input
  overflows this formula, the affected coordinate is recomputed with scaling.

- `FastCDStatus fastcd_mul_batch_f64(double* out, size_t out_stride, const
  double* x, size_t x_stride, const double* y, size_t y_stride, size_t count,
  size_t dimension, const FastCDWorkspaceF64* workspace)`

  Multiply `count` pairs, using the automatic algorithm.

- `FastCDStatus fastcd_mul_batch_with_algorithm_f64(double* out, size_t
  out_stride, const double* x, size_t x_stride, const double* y, size_t
  y_stride, size_t count, size_t dimension, FastCDAlgorithm algorithm, const
  FastCDWorkspaceF64* workspace)`

  Multiply `count` pairs, using the selected `algorithm`.

**Batch parameters**

The two batch functions read `count` records from each input and write `count`
records, placed according to their strides:

- `count == 0` validates only the dimension and the algorithm, so every pointer
  may be null and nothing is written.
- `count == 1` ignores all strides.
- For `count > 1`, `out_stride` must be at least `N`, and each input stride must
  be zero, to repeat one record throughout the batch, or at least `N`.

Each array spans `(count - 1) * stride + N` doubles, or `N` doubles for a
repeated input, and the overflow checks and the output overlap check cover the
whole span, padding included. The two inputs may share storage, for example when
one repeats a record of the other or when both are windows of one array, as long
as every pair of records with the same index is identical or disjoint, and a
partially overlapping pair of records is rejected. Output padding is left
unchanged, and the one workspace is reused for each product in turn.

---

### Fused sums

A fused sum is `x_0 y_0 + x_1 y_1 + ... + x_(count-1) y_(count-1)`, computed
without forming the products: every multiplication of the algorithm is
accumulated over the elements, and the algorithm's output additions are applied
once, to the accumulated values (see [Fused sums of
products](mathematics.md#fused-sums-of-products)). Its result therefore rounds
differently from a batch of products followed by a sum. Fused sums and the
covariance are available for real coefficients only, and they are experimental
features: they may not be fully optimized or ready for production use, and the
code implementing them may change over time, but their API is expected to remain
the same.

- `FastCDStatus fastcd_mul_sum_f64(double* out, const double* x, size_t
  x_stride, const double* y, size_t y_stride, size_t count, size_t dimension,
  const FastCDSumMapF64* map, const FastCDWorkspaceF64* workspace)`

  Set `out` to the sum of the products of `count` pairs of records, using the
  automatic algorithm. The strides follow the rules of the batch functions, so
  each is zero, to repeat one record, or at least `N`, and both are ignored when
  `count == 1`. `map` may be `NULL`.

- `FastCDStatus fastcd_mul_sum_with_algorithm_f64(double* out, const double* x,
  size_t x_stride, const double* y, size_t y_stride, size_t count, size_t
  dimension, const FastCDSumMapF64* map, FastCDAlgorithm algorithm, const
  FastCDWorkspaceF64* workspace)`

  Compute the same sum using `algorithm`, which is `AUTO`, direct, or
  quasilinear; Cariow-Cariowa returns `FASTCD_UNSUPPORTED_ALGORITHM`.

- `FastCDStatus fastcd_mul_sum_planes_f64(double* out, const double* x, const
  double* y, size_t pitch, size_t count, size_t dimension, const
  FastCDWorkspaceF64* workspace)`

  Compute the same sum of elements stored as *coordinate planes*, in which
  coordinate `i` of element `p` is `x[i * pitch + p]`, so that each coordinate
  of the elements forms one contiguous row. `pitch` must be at least `count`,
  and a smaller value returns `FASTCD_INVALID_STRIDE`.

- `FastCDStatus fastcd_mul_sum_planes_with_algorithm_f64(double* out, const
  double* x, const double* y, size_t pitch, size_t count, size_t dimension,
  FastCDAlgorithm algorithm, const FastCDWorkspaceF64* workspace)`

  Compute the sum of planes using `algorithm`.

- `FastCDStatus fastcd_covariance_f64(double* out, const double* x, size_t
  x_stride, const double* y, size_t y_stride, size_t count, size_t dimension,
  size_t ddof, const FastCDWorkspaceF64* workspace)`

  Set `out` to the covariance of two samples of `count` records,

  ```text
  (1 / (count - ddof)) sum_p (x_p - mean(x)) conjugate(y_p - mean(y)),
  ```

  an element of the algebra whose scalar coordinate is the sum over the
  coordinates of their real covariances, with `ddof = 1` giving the unbiased
  estimate. `count <= ddof` returns `FASTCD_INVALID_COUNT`, including a count of
  zero. The means are subtracted and the conjugate is taken as the records are
  read, and the sum is a fused sum with the automatic algorithm.

  Each mean is held in two parts, the first record and the mean of the
  differences from it, and subtracting both reduces the rounding error of the
  centering. A finite sample whose largest magnitude among its varying
  coordinates lies outside `[2^-257, 2^256)` is scaled by a power of two, and
  the result is scaled back, so small coordinates can underflow during scaling.

  At dimension 2, with records `[real, imag]`, the result is the ordinary
  covariance of two complex samples, `E[(z - mean(z)) conjugate(w - mean(w))]`
  with the chosen divisor, while the complex-coefficient functions of
  [Arithmetic across coefficient fields](#arithmetic-across-coefficient-fields)
  have no covariance.

**Maps**

A `FastCDSumMapF64` applies an affine map to every record as it is read:
coordinate `i` of an element of `x` enters the sum as
`(value - x_shift[i]) * x_scale[i]`, and likewise for `y`. Each member is an
array of `N` doubles or `NULL`, where a null shift subtracts nothing and a null
scale multiplies by nothing. Centering a sample, scaling its coordinates, and
conjugating an operand, with the scales `1, -1, ..., -1`, are all maps, so a sum
of products of transformed records needs no copy of the data. The arrays must be
disjoint from the output and the workspace.

**Parameters and layouts**

- For the four sum functions, `count == 0` writes the zero element, and the
  inputs and the workspace are not read and may be null.
- The output must be disjoint from the inputs and the workspace, while the
  inputs are only read and may overlap each other in any way, including
  `x == y`.
- If a coordinate that is read is NaN or infinite, which output coordinates are
  non-finite is unspecified.
- Both layouts cut the elements into the same blocks and return the same bits
  for the same elements, the record functions rewriting each block of records
  into planes inside the workspace.

Whether a fused sum is faster than a batch of products followed by a sum, or
than a matrix product of the samples, depends on the dimension, the number of
elements, the layout, and the processor, and the
[benchmark](benchmark.md#fused-sums) reports measurements.

**Kernels**

The library chooses the kernel of the fused sums at run time. GCC and Clang
builds for x86 contain one kernel that holds one coordinate of eight elements in
a 512-bit register and one that holds it in two 256-bit registers, and they
choose between them by the processor's AVX-512, AVX2, and FMA features, while
other builds use a portable kernel. All kernels perform the same operations in
the same order, eight elements at a time, with every accumulation a
multiply-add, so the two x86 kernels return identical bits, and so does the
portable kernel on targets with a fused multiply-add instruction, such as
AArch64. On other targets the portable kernel rounds each product and addition
separately, and its results differ in the last bits.

---

### Noncommutative and nonassociative products

These functions compute the products that measure the failure of commutativity
and associativity, with the formulas of [Derived
operations](mathematics.md#derived-operations), where `x = alpha + X` and
`y = beta + Y`. Their outputs must be disjoint from their inputs, although
read-only inputs may coincide. The alternating product and the commutator are
exactly zero at `N = 1, 2` and use the workspace from
`fastcd_imag_cross_workspace_size`, while the associator uses its own composite
workspace query, even at `N = 1`.

- `FastCDStatus fastcd_imag_cross_f64(double* out, const double* x, const
  double* y, size_t dimension, const FastCDWorkspaceF64* workspace)`

  Compute the alternating product `X cross Y = (XY - YX)/2` of the imaginary
  parts of `x` and `y` with the quasilinear recursion. The scalar parts of `x`
  and `y` are ignored, and `out[0] = +0.0`.

- `FastCDStatus fastcd_commutator_f64(double* out, const double* x, const
  double* y, size_t dimension, const FastCDWorkspaceF64* workspace)`

  Compute `xy - yx = 2 (X cross Y)`, with `out[0] = +0.0`.

- `FastCDStatus fastcd_anticommutator_f64(double* out, const double* x, const
  double* y, size_t dimension)`

  Compute `xy + yx = 2 (alpha beta - <X, Y>) + 2 (alpha Y + beta X)` in `O(N)`
  time, with no workspace.

- `FastCDStatus fastcd_associator_f64(double* out, const double* x, const
  double* y, const double* z, size_t dimension, const FastCDWorkspaceF64*
  workspace)`

  Compute `(xy)z - x(yz)` through four automatic products.

---

### Indexed basis operations

These functions work with basis elements by index, using the basis signs
`sigma_N` of [Basis products](mathematics.md#basis-products).

- `FastCDStatus fastcd_basis_product(FastCDBasisProduct* product, size_t
  dimension, size_t left_index, size_t right_index)`

  Write the signed basis product `e_left_index e_right_index = product->sign *
  e_(product->index)`, where `product->index = left_index ^ right_index` and
  `product->sign = sigma_N(left_index, right_index)`, with `O(log N)` integer
  work. The errors are `FASTCD_NULL_POINTER`, `FASTCD_INVALID_DIMENSION`, and
  `FASTCD_INVALID_BASIS_INDEX`.

- `FastCDStatus fastcd_basis_lmul_f64(double* out, size_t basis_index, const
  double* x, size_t dimension)`

  Compute `e_basis_index x`, a signed permutation of `x`: `out[basis_index ^ i]
  = sigma_N(basis_index, i) x[i]`.

- `FastCDStatus fastcd_basis_rmul_f64(double* out, const double* x, size_t
  basis_index, size_t dimension)`

  Compute `x e_basis_index`: `out[i ^ basis_index] = sigma_N(i, basis_index)
  x[i]`.

---

### Matrix-free operator commutators

An operator commutator compares two orders of multiplication on one side without
forming the matrices of the multiplication operators. Each function uses four
automatic products and one subtraction, and it needs the workspace of its own
composite query, even at `N = 1`.

- `FastCDStatus fastcd_lop_commutator_apply_f64(double* out, const double* a,
  const double* b, const double* x, size_t dimension, const FastCDWorkspaceF64*
  workspace)`

  Compute `[L_a, L_b](x) = a(bx) - b(ax)`.

- `FastCDStatus fastcd_rop_commutator_apply_f64(double* out, const double* a,
  const double* b, const double* x, size_t dimension, const FastCDWorkspaceF64*
  workspace)`

  Compute `[R_a, R_b](x) = (xb)a - (xa)b`.

---

## Arithmetic across coefficient fields

Complex arithmetic is declared in `<fastcd/complex.h>` and finite-field
arithmetic in `<fastcd/finite.h>`. Both include `<fastcd/arithmetic.h>`, which
declares the operation selector `FastCDOperation` and
`fastcd_quadratic_norm_f64`, and like `fastcd.h` their functions allocate no
heap storage and retain no caller pointers.

A record is stored the same way over every field, as the `N` coefficients in the
standard coordinates, scalar first and contiguous, and only what one coefficient
occupies depends on the field:

| Field | One coefficient | One record |
| --- | --- | --- |
| R | one `double` | `N` doubles |
| C | one `FastCDComplexF64`, two doubles, real part first | `N` complex values, `2N` doubles |
| F_(p^d) | `d` canonical `uint32_t` residues, constant term first | `N * d` words |

A canonical residue is an integer in `0, ..., p - 1`.

### Rules shared by complex and finite-field functions

**Algorithms**

The functions accept `FASTCD_ALGORITHM_AUTO`, `FASTCD_ALGORITHM_DIRECT`, and
`FASTCD_ALGORITHM_QUASILINEAR`. Over C and over finite fields, `AUTO` does not
use the real rules but selects direct multiplication for `N <= 4` and
quasilinear multiplication otherwise, and Cariow-Cariowa returns
`FASTCD_UNSUPPORTED_ALGORITHM` over C and `FASTCD_FF_UNSUPPORTED_ALGORITHM` over
finite fields.

**Aliasing**

- An exact `out == x` is allowed for add, subtract, negate, scale, divide,
  conjugate, imaginary part, reciprocal, and square, and add and subtract also
  allow `out == y`.
- Every other output must be disjoint from the inputs, including the
  one-coefficient outputs of the inner product and the quadratic norm and the
  outputs of basis multiplication.
- `x` and `y` must be the same complete record or disjoint. For scale and
  divide, `y` is one coefficient and may point into `x`, but the output must not
  overlap it.
- The active prefix of the workspace must be disjoint from every operand, from
  the output, and, over finite fields, from the plan.

The functions check these rules before writing anything and report a violation
with `FASTCD_OVERLAPPING_BUFFERS` or `FASTCD_FF_OVERLAPPING_BUFFERS`, and a
partial overlap between `x` and `y` is rejected in the same way. The checks of
the composite operations differ between the two fields. Over C, the third
operand `z` is not checked against `x` or `y`. Over finite fields, each product
inside a composite operation checks its own two factors, so `z` must be the same
record as `y` or disjoint from it in the associator, and the same as or disjoint
from both `x` and `y` in the operator commutators.

**Preconditions the library does not check**

Buffer lengths, alignment, initialized and unmodified field contexts and plans,
and canonical residues are the caller's responsibility.

**Errors and threads**

A nonzero status leaves every output unchanged. Field contexts and plans are
read-only during a call, so concurrent calls may share them, while each call
needs its own output and workspace. Inputs, contexts, and plans must not change
while a call that reads them is running, including from another thread.

### Record operations

The functions `fastcd_apply_c128` and `fastcd_apply_ff` apply one operation to
records, selected by a `FastCDOperation`:

| Operation | Operands besides `x` | Output | Workspace |
|---|---|---|---|
| `FASTCD_OP_MULTIPLY` | `y` | record | product |
| `FASTCD_OP_ADD`, `FASTCD_OP_SUBTRACT` | `y` | record | none |
| `FASTCD_OP_NEGATE`, `FASTCD_OP_CONJUGATE`, `FASTCD_OP_IMAG_PART` | none | record | none |
| `FASTCD_OP_SCALE`, `FASTCD_OP_DIVIDE` | one coefficient `y` | record | none |
| `FASTCD_OP_INNER` | `y` | one coefficient | none |
| `FASTCD_OP_QUADRATIC_NORM` | none | one coefficient | none |
| `FASTCD_OP_RECIPROCAL`, `FASTCD_OP_SQUARE` | none | record | none |
| `FASTCD_OP_IMAG_CROSS`, `FASTCD_OP_COMMUTATOR` | `y` | record | product |
| `FASTCD_OP_ANTICOMMUTATOR` | `y` | record | none |
| `FASTCD_OP_ASSOCIATOR`, `FASTCD_OP_LEFT_OPERATOR_COMMUTATOR`, `FASTCD_OP_RIGHT_OPERATOR_COMMUTATOR` | `y`, `z` | record | product and two records |
| `FASTCD_OP_BASIS_LEFT`, `FASTCD_OP_BASIS_RIGHT` | `basis_index` | record | none |

Each operation computes the same formula as the corresponding real function
above, `FASTCD_OP_IMAG_CROSS` being the alternating product, and in the three
composite operations `z` is the third operand:

- `FASTCD_OP_ASSOCIATOR` computes `(xy)z - x(yz)`.
- `FASTCD_OP_LEFT_OPERATOR_COMMUTATOR` computes `[L_x, L_y](z) = x(yz) - y(xz)`.
- `FASTCD_OP_RIGHT_OPERATOR_COMMUTATOR` computes
  `[R_x, R_y](z) = (zy)x - (zx)y`.

Operand pointers that an operation does not use may be null, and `basis_index`
is read only by the two basis operations. The `algorithm` argument selects how
records are multiplied in multiplication, the alternating product, the
commutator, and the composite operations, but every operation checks it, so
Cariow-Cariowa or an unknown value is rejected even by operations that do not
multiply records. An operation outside `FastCDOperation` returns
`FASTCD_UNSUPPORTED_ALGORITHM` or `FASTCD_FF_UNSUPPORTED_ALGORITHM`, and
`FASTCD_OP_DIVIDE` by a zero coefficient returns `FASTCD_ZERO_NORM` or
`FASTCD_FF_ZERO_NORM`.

### Quadratic norm

```c
FastCDStatus fastcd_quadratic_norm_f64(double* value,
        const double* x, size_t dimension);
FastCDStatus fastcd_quadratic_norm_c128(FastCDComplexF64* value,
        const FastCDComplexF64* x, size_t dimension);
FastCDFFStatus fastcd_quadratic_norm_ff(uint32_t* value,
        const uint32_t* x, const FastCDPlanFF* plan);
```

Each function writes `Q(x) = sum_i x_i^2`, computed in the coefficient field
without complex conjugation, as one coefficient: one `double`, one
`FastCDComplexF64`, or `d` residues. The real function is an alias for
`fastcd_norm_squared_f64`, with the same scaled accumulation and validation.

Over C and finite fields, `Q(x)` can be zero for a nonzero `x` (see [The
quadratic form](mathematics.md#the-quadratic-form)). Such an element has no
reciprocal, and `FASTCD_OP_RECIPROCAL` returns `FASTCD_ZERO_NORM` or
`FASTCD_FF_ZERO_NORM` for it, as it does for zero.

### Complex coefficients

```c
typedef struct FastCDComplexF64 { double real, imag; } FastCDComplexF64;

FastCDStatus fastcd_workspace_size_c128(size_t dimension,
        FastCDOperation operation, size_t* length);
FastCDStatus fastcd_apply_c128(FastCDComplexF64* out,
        const FastCDComplexF64* x, const FastCDComplexF64* y,
        const FastCDComplexF64* z, size_t dimension, FastCDOperation operation,
        size_t basis_index, FastCDAlgorithm algorithm,
        const FastCDWorkspaceF64* workspace);
FastCDStatus fastcd_mul_with_algorithm_c128(FastCDComplexF64* out,
        const FastCDComplexF64* x, const FastCDComplexF64* y, size_t dimension,
        FastCDAlgorithm algorithm, const FastCDWorkspaceF64* workspace);
FastCDStatus fastcd_mul_c128(FastCDComplexF64* out,
        const FastCDComplexF64* x, const FastCDComplexF64* y, size_t dimension,
        const FastCDWorkspaceF64* workspace);
FastCDStatus fastcd_mul_batch_c128(FastCDComplexF64* out, size_t out_stride,
        const FastCDComplexF64* x, size_t x_stride,
        const FastCDComplexF64* y, size_t y_stride, size_t count,
        size_t dimension, FastCDAlgorithm algorithm,
        const FastCDWorkspaceF64* workspace);
```

**Parameters**

- `dimension` is `N`, the number of complex coefficients in a record, and batch
  strides also count `FastCDComplexF64` values.
- Workspace lengths count **doubles**, as in the real interface, and the
  workspace is described by a `FastCDWorkspaceF64`.
- Complex values are passed as the two-double struct, never as C99
  `double _Complex`.

**Functions**

- `fastcd_workspace_size_c128` writes the number of doubles that `operation`
  needs, a length that covers both direct and quasilinear evaluation and is zero
  for operations that do not multiply records. A batch needs the
  `FASTCD_OP_MULTIPLY` length.
- `fastcd_apply_c128` applies one record operation.
- `fastcd_mul_with_algorithm_c128` multiplies two records with the given
  algorithm, and `fastcd_mul_c128` multiplies them with `AUTO`.
- `fastcd_mul_batch_c128` multiplies `count` pairs of records.

**Numerical behavior**

NaN, infinity, and overflow are numerical results, not errors.
`FASTCD_OP_CONJUGATE` negates coordinates `1` through `N - 1` and does not
conjugate the complex coefficients. The quadratic norm, the reciprocal, and the
square accumulate `Q(x)` and the pairing in plain complex arithmetic, without
the scaling that the real functions use, so they can overflow where their real
counterparts do not.

**Batches**

- `count == 0` validates only the dimension and the algorithm, so every pointer
  may be null and nothing is written.
- `count == 1` ignores all strides.
- For `count > 1`, `out_stride` must be at least `N`, each input stride must be
  zero, to repeat one record, or at least `N`, and the workspace is required.

All sizes and overlaps are checked before the first product is written: the
output span and the active workspace must be disjoint from both input spans, and
records with the same index in the two inputs must be identical or disjoint.

### Finite fields

Finite-field arithmetic needs two objects that the real and complex functions do
not, a field context, which describes F_(p^d), and a plan, which fixes the
Cayley-Dickson dimension over that field, and both are set up once and then
shared by every call. The arithmetic runs in kernels specialized by the
extension degree: on x86 the library contains SSE4.1, AVX2, and AVX-512 versions
and chooses among them at run time by the features of the processor, the
characteristic, the degree, and the dimension, while a portable kernel serves
other targets.

#### Field contexts

```c
FastCDFFStatus fastcd_field_init(FastCDField* field, uint32_t prime,
        uint32_t degree, const uint32_t* modulus);
FastCDFFStatus fastcd_ff_field_init(FastCDField* field, uint32_t prime,
        uint32_t degree, const uint32_t* modulus);
FastCDFFStatus fastcd_ff_field_init_trusted(FastCDField* field, uint32_t prime,
        uint32_t degree, const uint32_t* modulus);
```

A `FastCDField` describes F_(p^d). Its public members are `prime`, `degree`, and
`modulus[13]`, and the remaining members hold precomputed reduction data.
`FASTCD_FF_MAX_DEGREE` is 12, and an initialized context is treated as
immutable. The three constructors differ in how much they check:

- `fastcd_field_init` accepts odd primes `3` through `INT32_MAX` and degrees `1`
  through `12`. With a null `modulus` it chooses a deterministic monic
  irreducible polynomial and records it in `field->modulus`, and an explicit
  `modulus` has exactly `degree + 1` canonical residues, constant term first,
  with leading coefficient one. Setup checks that `prime` is prime and that the
  polynomial is irreducible, and leaves `field` unchanged on failure.
  `prime == 2` returns `FASTCD_FF_CHARACTERISTIC_TWO`.
- `fastcd_ff_field_init` is the lower-level constructor. It also accepts
  `prime == 2`, for scalar arithmetic only, since plans reject characteristic 2,
  and it requires an explicit polynomial for `degree > 1`, while at degree one a
  null `modulus` selects `t`.
- `fastcd_ff_field_init_trusted` makes the same checks of range and shape but
  skips the primality and irreducibility checks, so it is meant only for
  parameters that the caller has already validated. It has no Python
  counterpart.

Invalid parameters return `FASTCD_FF_INVALID_FIELD`, and `fastcd_ff_field_init`
returns `FASTCD_FF_NULL_POINTER` for `degree > 1` without a `modulus`.

#### Plans

```c
FastCDFFStatus fastcd_ff_plan_init(FastCDPlanFF* plan,
        const FastCDField* field, size_t level);
FastCDFFStatus fastcd_ff_cd_workspace_size(const FastCDPlanFF* plan,
        size_t* words);
FastCDFFStatus fastcd_apply_workspace_size_ff(const FastCDPlanFF* plan,
        FastCDOperation operation, size_t* workspace_words);
```

A `FastCDPlanFF` fixes the Cayley-Dickson level over one field.
`fastcd_ff_plan_init` copies the field into the plan and records `level`,
`dimension = 2^level`, the record length `words = dimension * degree`, the
product workspace length `workspace_words`, and the inverses of 2 and of
`dimension` modulo `p`. A plan contains no pointers and is treated as immutable.
Level zero is allowed, and its algebra is the coefficient field itself. Plans
reject characteristic 2 with `FASTCD_FF_CHARACTERISTIC_TWO`, and sizes that do
not fit `size_t` with `FASTCD_FF_SIZE_OVERFLOW`.

Finite-field workspaces are arrays of `uint32_t`, so their lengths count words:

- `fastcd_ff_cd_workspace_size` writes the workspace that every product function
  needs, `fastcd_mul_ff`, `fastcd_mul_with_algorithm_ff`, `fastcd_ff_cd_mul`,
  and `fastcd_mul_batch_ff`, with either algorithm. The value equals
  `plan->workspace_words`.
- `fastcd_apply_workspace_size_ff` writes the workspace that `fastcd_apply_ff`
  needs for one operation: the product workspace for multiplication, the
  alternating product, and the commutator, that plus two records for the
  composite operations, and zero for the others.

A larger buffer may be supplied, in which case only the required prefix is used
and checked for overlap.

#### Scalar arithmetic

```c
void fastcd_ff_field_add(uint32_t* out, const uint32_t* x,
        const uint32_t* y, const FastCDField* field);
void fastcd_ff_field_sub(uint32_t* out, const uint32_t* x,
        const uint32_t* y, const FastCDField* field);
void fastcd_ff_field_mul(uint32_t* out, const uint32_t* x,
        const uint32_t* y, const FastCDField* field);
void fastcd_field_neg(uint32_t* out, const uint32_t* x,
        const FastCDField* field);
FastCDFFStatus fastcd_field_inv(uint32_t* out, const uint32_t* x,
        const FastCDField* field);
uint32_t fastcd_ff_prime_mul(uint32_t a, uint32_t b, const FastCDField* field);
uint32_t fastcd_ff_field_reduce64(uint64_t x, const FastCDField* field);
```

These functions act on single elements of F_(p^d), each `degree` canonical words
long, and they have no Python counterparts.

- Add, subtract, multiply, and negate need no workspace and return nothing.
  `out` may equal `x` or `y` exactly, and otherwise the buffers must be
  disjoint; buffer ranges and residues are not checked.
- `fastcd_field_inv` computes `x^(p^d - 2)`, the inverse of a nonzero `x`, and
  for zero returns `FASTCD_FF_ZERO_NORM` and leaves `out` unchanged. `out == x`
  is allowed.
- `fastcd_ff_prime_mul` multiplies two residues modulo the prime `p`. An
  extension context may be passed, but the result is a product in F_p, not a
  product of extension elements.
- `fastcd_ff_field_reduce64` reduces any `uint64_t` modulo `p`.

`fastcd_ff_field_mul`, `fastcd_ff_prime_mul`, and `fastcd_ff_field_reduce64` are
inline functions in the header. For `degree > 1`, `fastcd_ff_field_mul` calls
the exported `fastcd_ff_field_mul_extension`, which applications should not call
directly. Defining `FASTCD_FF_FIELD_EXTERN_ONLY` before including
`<fastcd/finite.h>` makes all three calls go to exported out-of-line versions
instead, and the header defines `FASTCD_FF_HAVE_INLINE_FIELD_API` to 1 so that
code can test for the inline functions.

#### Cayley-Dickson operations

```c
FastCDFFStatus fastcd_mul_with_algorithm_ff(uint32_t* out, const uint32_t* x,
        const uint32_t* y, const FastCDPlanFF* plan, FastCDAlgorithm algorithm,
        uint32_t* workspace, size_t workspace_words);
FastCDFFStatus fastcd_mul_ff(uint32_t* out, const uint32_t* x,
        const uint32_t* y, const FastCDPlanFF* plan, uint32_t* workspace,
        size_t workspace_words);
FastCDFFStatus fastcd_ff_cd_mul(uint32_t* out, const uint32_t* x,
        const uint32_t* y, const FastCDPlanFF* plan, uint32_t* workspace,
        size_t workspace_words);
FastCDFFStatus fastcd_apply_ff(uint32_t* out, const uint32_t* x,
        const uint32_t* y, const uint32_t* z, const FastCDPlanFF* plan,
        FastCDOperation operation, size_t basis_index,
        FastCDAlgorithm algorithm, uint32_t* workspace, size_t workspace_words);
FastCDFFStatus fastcd_mul_batch_ff(uint32_t* out, size_t out_stride,
        const uint32_t* x, size_t x_stride, const uint32_t* y, size_t y_stride,
        size_t count, const FastCDPlanFF* plan, FastCDAlgorithm algorithm,
        uint32_t* workspace, size_t workspace_words);
const char* fastcd_ff_status_string(FastCDFFStatus status);
```

**Parameters**

- A record has `plan->words` words, and lengths and strides count `uint32_t`
  words.
- `workspace_words` is the length of the supplied workspace; [Plans](#plans)
  gives the required length.

**Functions**

- `fastcd_mul_with_algorithm_ff` multiplies two records with the given
  algorithm, `fastcd_mul_ff` with `AUTO`, and `fastcd_ff_cd_mul` always with
  quasilinear multiplication.
- `fastcd_apply_ff` applies one [record operation](#record-operations).
- `fastcd_mul_batch_ff` multiplies `count` pairs of records under the rules for
  complex batches, with `plan->words` in place of `N`; with `count == 0` it
  checks only that the plan is non-null and the algorithm is supported.
- `fastcd_ff_status_string` returns a fixed English message for a status.

**Statuses**

`FastCDFFStatus` is a separate enumeration from `FastCDStatus`. Its values, in
declaration order, are `FASTCD_FF_SUCCESS`, `FASTCD_FF_INVALID_LEVEL`,
`FASTCD_FF_INVALID_FIELD`, `FASTCD_FF_CHARACTERISTIC_TWO`,
`FASTCD_FF_INSUFFICIENT_WORKSPACE`, `FASTCD_FF_OVERLAPPING_BUFFERS`,
`FASTCD_FF_SIZE_OVERFLOW`, `FASTCD_FF_NULL_POINTER`,
`FASTCD_FF_UNSUPPORTED_ALGORITHM`, `FASTCD_FF_ZERO_NORM`,
`FASTCD_FF_INVALID_BASIS_INDEX`, and `FASTCD_FF_INVALID_STRIDE`, of which
`FASTCD_FF_INVALID_LEVEL` is declared but not currently returned, because a
level whose dimension does not fit `size_t` returns `FASTCD_FF_SIZE_OVERFLOW`.
