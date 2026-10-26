# C API reference

## fastcd.h

The public interface is declared in `<fastcd/fastcd.h>`, which includes C++
linkage guards.

### Linking

- Link the CMake target `fastcd::fastcd`, or use `-lfastcd -lm` when linking
  by hand.
- Shared-library consumers built without CMake define `FASTCD_SHARED`.
  Only the library build defines `FASTCD_BUILDING_LIBRARY`.

---

### Types and constants

```c
typedef enum {
    FASTCD_SUCCESS = 0,
    FASTCD_INVALID_DIMENSION,      /* zero or not a power of two */
    FASTCD_INVALID_LEVEL,          /* 2^level does not fit size_t */
    FASTCD_INVALID_BASIS_INDEX,    /* index >= dimension */
    FASTCD_INSUFFICIENT_WORKSPACE, /* non-null scratch shorter than required */
    FASTCD_ZERO_NORM,              /* normalize/reciprocal of exact zero */
    FASTCD_OVERLAPPING_BUFFERS,    /* aliasing contract violated */
    FASTCD_INVALID_STRIDE,         /* batch stride smaller than dimension */
    FASTCD_SIZE_OVERFLOW,          /* an extent does not fit size_t */
    FASTCD_NULL_POINTER,           /* required pointer is null */
    FASTCD_UNSUPPORTED_ALGORITHM   /* unknown algorithm, or CC above 1024 */
} FastCDStatus;

typedef enum {
    FASTCD_ALGORITHM_AUTO = 0,
    FASTCD_ALGORITHM_DIRECT,
    FASTCD_ALGORITHM_CARIOW_CARIOWA,
    FASTCD_ALGORITHM_QUASILINEAR
} FastCDAlgorithm;

#define FASTCD_CARIOW_CARIOWA_MAX_DIMENSION ((size_t)1024)

typedef struct FastCDWorkspaceF64 { double* data; size_t length; } FastCDWorkspaceF64;
typedef struct FastCDBasisProduct { size_t index; int sign; } FastCDBasisProduct;
```

---

### Calling contract

**Parameters**

- `dimension` (written as `N` below): a nonzero power of two. An element
  contains `N` contiguous `double` coefficients in the recursive standard
  basis, with the scalar in coordinate zero. See [conventions and
  results](mathematics.md) for the multiplication convention and basis signs.
- Lengths and strides count `double` elements.
- Inputs, outputs, and workspaces belong to the caller. No function
  allocates on the heap or retains a pointer. The three string queries
  return immutable static storage.

**Returns**

- Arithmetic functions and queries with output parameters return
  `FastCDStatus`. An error leaves every output unchanged. When several
  arguments are invalid, the reported error is unspecified.
- Array functions can return `FASTCD_INVALID_DIMENSION`,
  `FASTCD_SIZE_OVERFLOW`, or `FASTCD_NULL_POINTER`. Functions with array
  inputs also check overlap. Additional errors appear with the relevant
  functions below.
- NaN, infinity, signed zero, overflow, underflow, and cancellation are
  numerical results. Different algorithms and specialized formulas can
  round differently without returning an error.

AUTO choices, workspace lengths, and arithmetic schedules may change in minor
or patch releases while preserving the public calling contracts. Obtain
workspace lengths from the query functions. Floating-point results are not
guaranteed bitwise identical across versions, even for the same explicitly
selected algorithm. Relevant numerical changes are recorded in `CHANGELOG.md`.

**Aliasing**

An exact alias has the same starting pointer and covers a complete element.
Partial overlap is rejected with `FASTCD_OVERLAPPING_BUFFERS`. Read-only
inputs may be the same complete array, and the two inputs of a batch may
share storage under the record rule given with the batch functions. Outputs
must be disjoint from every input, except for these cases:

- `fastcd_add_f64` and `fastcd_sub_f64` allow `out == x`, `out == y`, or both.
- `fastcd_neg_f64`, `fastcd_scale_f64`, `fastcd_conjugate_f64`,
  `fastcd_imag_part_f64`, `fastcd_normalize_f64`, `fastcd_reciprocal_f64`, and
  `fastcd_square_f64` allow `out == x`.

Scalar results and basis permutations must also be disjoint from inputs.

**Workspaces**

Use the matching query to obtain the required length, allocate that many
`double` elements, and describe the storage with `FastCDWorkspaceF64`.

- When the required length is zero, `workspace` may be `NULL`.
- Otherwise, a null descriptor or data pointer returns `FASTCD_NULL_POINTER`,
  and too little storage returns `FASTCD_INSUFFICIENT_WORKSPACE`.
- The queried prefix must be disjoint from every array. The descriptor
  object itself must not overlap an input or output; this precondition is
  not checked by the library.
- Scratch contents have no meaning after the call.

**Threading and SIMD**

Calls can share read-only inputs across threads when their outputs and
workspaces are disjoint. Each batch uses one thread and one workspace.

On GCC/Clang targets with SSE2 or AArch64 SIMD, batches of two or more
products use two products per instruction at these dimensions:

- Quasilinear and Cariow–Cariowa: dimension 16 through 512.
- Direct: dimension 64 through 512.

On the same targets, a single quasilinear product above dimension 16
evaluates the two children of the top complex node in the two SIMD lanes
and continues through the paired kernel, using the caller's workspace; the
workspace query reports the larger requirement of that form. Single direct
and Cariow–Cariowa products use scalar kernels, which the compiler may
vectorize.

The scalar and SIMD kernels share an arithmetic schedule and produce
identical finite results for the same algorithm; NaN payloads may differ.
Odd tails use a scalar kernel. Packing needs no extra alignment of caller
storage and uses a fixed 40 KiB stack array. Outside these thresholds,
or on other compiler/target combinations, batches use the scalar kernels
and the caller's workspace.

---

### Library information

- `const char* fastcd_version_string(void)`

  Return the library version string, currently `"1.1.0"`.

- `const char* fastcd_status_string(FastCDStatus status)`

  Return a fixed English status message. Unknown values return `"Unknown fastCD
  status"`.

- `const char* fastcd_algorithm_string(FastCDAlgorithm algorithm)`

  Return `"auto"`, `"direct"`, `"cariow_cariowa"`, or `"quasilinear"`. Unknown values
  return `"unknown"`.

---

### Dimension utilities

- `bool fastcd_valid_dimension(size_t dimension)`

  Return whether `dimension` is a nonzero power of two. This check takes constant time.

- `FastCDStatus fastcd_dimension(size_t level, size_t* dimension)`

  Write `1 << level` to `dimension`. Return `FASTCD_INVALID_LEVEL` if the shift would
  overflow.

- `FastCDStatus fastcd_level(size_t dimension, size_t* level)`

  Write `log2(dimension)` to `level`.

---

### Algorithm selection

`FASTCD_ALGORITHM_AUTO` uses two tables selected from the benchmark:

- Single products use direct at `N = 8, 16` and quasilinear elsewhere.
  This table also serves one-record batches and operator routines.
- Batches of two or more products use direct at `N = 8`, Cariow–Cariowa at
  `N = 16`, and quasilinear elsewhere. Builds without SIMD pairing use the
  single-product table.

**Functions**

- `FastCDStatus fastcd_default_algorithm(size_t dimension, FastCDAlgorithm* algorithm)`

  Write the concrete algorithm selected by `AUTO` for a single product.

- `FastCDStatus fastcd_default_batch_algorithm(size_t dimension, FastCDAlgorithm*
  algorithm)`

  Write the concrete algorithm selected by `AUTO` for a batch of at least two products.

- `bool fastcd_algorithm_supports_dimension(FastCDAlgorithm algorithm, size_t dimension)`

  Return `false` for an invalid dimension, an unknown algorithm, or Cariow–Cariowa above
  dimension 1024; otherwise return `true`.

---

### Workspace queries

Each query writes a scratch length in doubles. Errors are
`FASTCD_NULL_POINTER`, `FASTCD_INVALID_DIMENSION`, `FASTCD_SIZE_OVERFLOW`,
and, where an algorithm is involved, `FASTCD_UNSUPPORTED_ALGORITHM`.

Lengths are `O(N)` doubles and do not depend on the batch count.
With the default leaf cutoffs, lengths are zero for Cariow–Cariowa through
`N = 64` and `2N` above that, where its table-driven kernel runs.
Direct needs zero doubles for `N <= 16` and `2N - 32` above that.
Quasilinear needs zero for `N <= 16`;
above that it needs `5N - 63` on targets with the two-lane SIMD kernels and
`2N - 32` otherwise. Composite operations add `2N`. Use the queries in your
code, as these sizes describe the current implementation;
[mathematics.md](mathematics.md) explains where the terms come from.

- `FastCDStatus fastcd_mul_workspace_size(size_t dimension, size_t* length)`

  Size the workspace for `fastcd_mul_f64` and `fastcd_mul_batch_f64`, covering the
  larger of the two automatic selections' needs.

- `FastCDStatus fastcd_mul_algorithm_workspace_size(size_t dimension, FastCDAlgorithm
  algorithm, size_t* length)`

  Size the workspace for `fastcd_mul_with_algorithm_f64` and
  `fastcd_mul_batch_with_algorithm_f64`.

- `FastCDStatus fastcd_imag_cross_workspace_size(size_t dimension, size_t* length)`

  Size the workspace for `fastcd_imag_cross_f64` and `fastcd_commutator_f64`.

- `FastCDStatus fastcd_associator_workspace_size(size_t dimension, size_t* length)`

  Size the workspace for `fastcd_associator_f64`.

- `FastCDStatus fastcd_lop_commutator_workspace_size(size_t dimension, size_t* length)`

  Size the workspace for `fastcd_lop_commutator_apply_f64`.

- `FastCDStatus fastcd_rop_commutator_workspace_size(size_t dimension, size_t* length)`

  Size the workspace for `fastcd_rop_commutator_apply_f64`.

---

### Element construction

- `FastCDStatus fastcd_scalar_f64(double* out, size_t dimension, double value)`

  Set `out[0] = value` and every other coordinate to `+0.0`.

- `FastCDStatus fastcd_basis_f64(double* out, size_t dimension, size_t index)`

  Set `out[index] = 1.0` and every other coordinate to `+0.0`. Return
  `FASTCD_INVALID_BASIS_INDEX` when `index >= N`.

---

### Coordinate arithmetic

These functions take `O(N)` time, use no workspace, and perform one
`double` operation per coordinate.

- `FastCDStatus fastcd_add_f64(double* out, const double* x, const double* y, size_t
  dimension)`

  Set `out[i] = x[i] + y[i]`.

- `FastCDStatus fastcd_sub_f64(double* out, const double* x, const double* y, size_t
  dimension)`

  Set `out[i] = x[i] - y[i]`.

- `FastCDStatus fastcd_neg_f64(double* out, const double* x, size_t dimension)`

  Set `out[i] = -x[i]`.

- `FastCDStatus fastcd_scale_f64(double* out, const double* x, double scalar, size_t
  dimension)`

  Set `out[i] = scalar * x[i]`.

---

### Structural operations

These functions take `O(N)` time and use no workspace.

**Numerical behavior**

- Norms return NaN if any coordinate is NaN, otherwise `+inf` if any is
  infinite, and `+0.0` for an all-zero input.
- Normalization and reciprocals return `FASTCD_ZERO_NORM` only for an
  exactly zero finite input, including signed zeros.
- For normalization and reciprocals, any NaN makes every output NaN. With
  infinity present and no NaN, infinite coordinates become NaN and finite
  coordinates become signed zero.

**Functions**

- `FastCDStatus fastcd_conjugate_f64(double* out, const double* x, size_t dimension)`

  Set `out` to `(x[0], -x[1], ..., -x[N-1])`.

- `FastCDStatus fastcd_imag_part_f64(double* out, const double* x, size_t dimension)`

  Set `out` to `(+0.0, x[1], ..., x[N-1])`.

- `FastCDStatus fastcd_inner_f64(double* value, const double* x, const double* y, size_t
  dimension)`

  Write `sum_i x[i] y[i]` to `value`, accumulating in coordinate order without scaling.

- `FastCDStatus fastcd_norm_squared_f64(double* value, const double* x, size_t dimension)`

  Write `sum_i x[i]^2` to `value` using scaled accumulation.

- `FastCDStatus fastcd_norm_f64(double* value, const double* x, size_t dimension)`

  Write the Euclidean norm to `value` using scaled accumulation.

- `FastCDStatus fastcd_normalize_f64(double* out, const double* x, size_t dimension)`

  Set `out` to `x / norm(x)` using scaled division.

- `FastCDStatus fastcd_reciprocal_f64(double* out, const double* x, size_t dimension)`

  Set `out` to `conjugate(x) / norm_squared(x)` using scaled division. This is the
  canonical reciprocal, not a general division operation.

---

### Multiplication

- `FastCDStatus fastcd_mul_f64(double* out, const double* x, const double* y, size_t
  dimension, const FastCDWorkspaceF64* workspace)`

  Set `out` to `xy` using the automatic algorithm. A NaN coordinate in either
  operand makes every output coordinate NaN, and an infinite coordinate makes
  every output coordinate non-finite; which coordinates become NaN rather
  than infinite depends on the algorithm.

- `FastCDStatus fastcd_mul_with_algorithm_f64(double* out, const double* x, const double*
  y, size_t dimension, FastCDAlgorithm algorithm, const FastCDWorkspaceF64* workspace)`

  Set `out` to `xy` using `algorithm`, which may be `AUTO`. An unknown or unavailable
  algorithm returns `FASTCD_UNSUPPORTED_ALGORITHM`.

- `FastCDStatus fastcd_square_f64(double* out, const double* x, size_t dimension)`

  Set `out` to `x^2 = (alpha^2 - <u,u>) + 2 alpha u` in `O(N)` time. If a finite input
  overflows the plain scalar formula, recompute that coordinate with scaling.

- `FastCDStatus fastcd_mul_batch_f64(double* out, size_t out_stride, const double* x,
  size_t x_stride, const double* y, size_t y_stride, size_t count, size_t dimension,
  const FastCDWorkspaceF64* workspace)`

  Multiply `count` pairs using the automatic algorithm.

- `FastCDStatus fastcd_mul_batch_with_algorithm_f64(double* out, size_t out_stride, const
  double* x, size_t x_stride, const double* y, size_t y_stride, size_t count, size_t
  dimension, FastCDAlgorithm algorithm, const FastCDWorkspaceF64* workspace)`

  Multiply `count` pairs using the selected `algorithm`.

**Batch parameters**

- `count == 0`: validate only the dimension and algorithm. Every pointer
  may be null, and nothing is written.
- `count == 1`: ignore all strides.
- `count > 1`: `out_stride` must be at least `N`. Each input stride must
  be zero, to broadcast one element, or at least `N`.

Each span covers `(count - 1) * stride + N` doubles, or `N` doubles for a
broadcast input. Overflow checks and the output overlap check cover the whole
span, including padding. The two inputs may share storage, for example when
one broadcasts a record of the other or both are windows of one array, as
long as every pair of same-index records is identical or disjoint; a
partially overlapping record pair is rejected. Output padding is unchanged,
and one workspace is reused serially.

---

### Noncommutative and nonassociative products

Outputs must be disjoint from inputs, although read-only inputs may
coincide. Cross products and commutators are exactly zero at `N = 1, 2` and
use the workspace from `fastcd_imag_cross_workspace_size`. The associator
uses its own composite workspace query, even at `N = 1`.

- `FastCDStatus fastcd_imag_cross_f64(double* out, const double* x, const double* y, size_t
  dimension, const FastCDWorkspaceF64* workspace)`

  Compute `(uv - vu)/2` from the imaginary parts of `x` and `y` using one quasilinear
  alternating product. Ignore input scalars and set `out[0] = +0.0`.

- `FastCDStatus fastcd_commutator_f64(double* out, const double* x, const double* y, size_t
  dimension, const FastCDWorkspaceF64* workspace)`

  Compute `xy - yx` as twice the imaginary cross product, with `out[0] = +0.0`.

- `FastCDStatus fastcd_anticommutator_f64(double* out, const double* x, const double* y,
  size_t dimension)`

  Compute `xy + yx` using the `O(N)` scalar–imaginary formula, with no workspace.

- `FastCDStatus fastcd_associator_f64(double* out, const double* x, const double* y, const
  double* z, size_t dimension, const FastCDWorkspaceF64* workspace)`

  Compute `(xy)z - x(yz)` through four automatic products.

---

### Indexed basis operations

- `FastCDStatus fastcd_basis_product(FastCDBasisProduct* product, size_t dimension, size_t
  left_index, size_t right_index)`

  Write the signed basis product `e_left_index e_right_index = product->sign *
  e_(product->index)`, where `product->index = left_index ^ right_index`. This takes
  `O(log N)` integer work. Errors are `FASTCD_NULL_POINTER`, `FASTCD_INVALID_DIMENSION`,
  and `FASTCD_INVALID_BASIS_INDEX`.

- `FastCDStatus fastcd_basis_lmul_f64(double* out, size_t basis_index, const double* x,
  size_t dimension)`

  Compute `e_basis_index x` as a signed permutation: `out[basis_index ^ i] =
  sign(basis_index, i) x[i]`.

- `FastCDStatus fastcd_basis_rmul_f64(double* out, const double* x, size_t basis_index,
  size_t dimension)`

  Compute `x e_basis_index` with the same permutation and the sign `sign(i,
  basis_index)`.

---

### Matrix-free operator commutators

Each function uses four automatic products and one subtraction. Query the
matching composite workspace size; workspace is required even at `N = 1`.

- `FastCDStatus fastcd_lop_commutator_apply_f64(double* out, const double* a, const double*
  b, const double* x, size_t dimension, const FastCDWorkspaceF64* workspace)`

  Compute `[L_a, L_b](x) = a(bx) - b(ax)`.

- `FastCDStatus fastcd_rop_commutator_apply_f64(double* out, const double* a, const double*
  b, const double* x, size_t dimension, const FastCDWorkspaceF64* workspace)`

  Compute `[R_a, R_b](x) = (xb)a - (xa)b`.
