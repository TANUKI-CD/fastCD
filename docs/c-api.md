# C API reference

## fastcd.h

The public interface is declared in `<fastcd/fastcd.h>`, which includes C++
linkage guards.

### Linking

- Link the CMake target `fastcd::fastcd`, or use `-lfastcd -lm` when linking
  by hand.
- Shared-library consumers built without CMake define `FASTCD_SHARED`.
  Only the library build defines `FASTCD_BUILDING_LIBRARY`. Together they
  set `FASTCD_API`, the export and import attribute on every public
  function.

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
    FASTCD_UNSUPPORTED_ALGORITHM   /* unknown algorithm or operation, or an
                                      algorithm unavailable for the call */
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

- `dimension` (written as `N` below): a nonzero power of two. An element is
  stored as a *record*: `N` contiguous `double` coefficients in the
  recursive standard basis, with the scalar in coordinate zero. See
  [Conventions and results](mathematics.md) for the multiplication
  convention and basis signs.
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

**Aliasing**

An exact alias has the same starting pointer and covers a complete record.
Partial overlap is rejected with `FASTCD_OVERLAPPING_BUFFERS`. Read-only
inputs may be the same complete array, and the two inputs of a batch may
share storage under the rule given with the batch functions. Outputs must
be disjoint from every input, except for these cases:

- `fastcd_add_f64` and `fastcd_sub_f64` allow `out == x`, `out == y`, or both.
- `fastcd_neg_f64`, `fastcd_scale_f64`, `fastcd_conjugate_f64`,
  `fastcd_imag_part_f64`, `fastcd_normalize_f64`, `fastcd_reciprocal_f64`, and
  `fastcd_square_f64` allow `out == x`.

The one-value outputs of `fastcd_inner_f64`, `fastcd_norm_squared_f64` and
`fastcd_norm_f64`, and the outputs of `fastcd_basis_lmul_f64` and
`fastcd_basis_rmul_f64`, must also be disjoint from the inputs.

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
workspaces are disjoint. Each batch call runs on one thread and reuses one
workspace for all its products.

On GCC and Clang targets with SSE2 or AArch64 SIMD, a batch of two or more
products evaluates two products at a time, one in each SIMD lane, at these
dimensions:

- Quasilinear and Cariow–Cariowa: dimension 16 through 512.
- Direct: dimension 64 through 512.

These paired kernels copy each pair of operands into an interleaved layout
in a fixed 40 KiB stack buffer, so caller storage needs no extra alignment.
When a batch has an odd number of products, the last one is computed as a
single product. At other dimensions, and with other compilers or targets,
batches compute one product at a time with the single-product kernels and
the caller's workspace.

On the same targets, a single quasilinear product above dimension 16 places
the two recursive calls of `C_(n-1)` (see
[How quasilinear multiplication works](mathematics.md#how-quasilinear-multiplication-works))
in the two SIMD lanes and continues through the paired kernel, using the
caller's workspace; the workspace query reports the larger requirement of
that form. Single direct and Cariow–Cariowa products use scalar kernels,
which the compiler may vectorize.

The scalar and SIMD kernels perform the same operations in the same order,
so they produce identical finite results for the same algorithm; NaN
payloads may differ.

---

### Compatibility

Every later 1.x release keeps the calling contract of version 1.0.0. Code
written against 1.0.0 compiles against any later 1.x release, and each of
its calls keeps:

- the same parameters and the same meaning for each of them;
- the same status code for each error condition;
- the same aliasing and workspace rules;
- the same mathematical result, up to floating-point rounding.

A release may change:

- the algorithm that `AUTO` selects at a given dimension;
- workspace lengths, so obtain them from the query functions at run time
  rather than hard-coding them;
- the arithmetic schedule of an algorithm, and therefore the last bits of a
  floating-point result, even when the algorithm is selected explicitly.

`CHANGELOG.md` records every change that affects numerical results.

---

### Library information

- `const char* fastcd_version_string(void)`

  Return the version string of the linked library, such as `"1.4.0"`.

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

  Write `log2(dimension)` to `level`. Return `FASTCD_INVALID_DIMENSION` if
  `dimension` is zero or not a power of two.

---

### Algorithm selection

`FASTCD_ALGORITHM_AUTO` chooses from two tables, set from the single-core
benchmark reported in the README:

- Single products use direct at `N = 4, 8, 16` and quasilinear at every
  other dimension. This table also applies to batches of one product and to the
  products inside `fastcd_associator_f64` and the two operator commutators.
- Batches of two or more products use direct at `N = 8`, Cariow–Cariowa at
  `N = 16`, and quasilinear at every other dimension. Builds without the
  two-lane SIMD kernels use the single-product table for batches.

`fastcd_imag_cross_f64` and `fastcd_commutator_f64` always use the
quasilinear recursion.

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

Lengths are `O(N)` doubles and do not depend on the batch count. With the
default leaf cutoffs:

- Direct needs nothing for `N <= 16` and `2N - 32` doubles above that.
- Quasilinear needs nothing for `N <= 16`. Above that it needs `5N - 63`
  doubles on targets with the two-lane SIMD kernels and `2N - 32` otherwise.
- Cariow–Cariowa needs nothing through `N = 64`, where it uses straight-line
  kernels, and `2N` doubles above that, where its table-driven kernel runs.
- The associator and the operator commutators need `2N` doubles more than
  their products.

These lengths can change between releases (see [Compatibility](#compatibility)),
so obtain them from the queries. [Workspace](mathematics.md#workspace)
explains where the terms come from.

- `FastCDStatus fastcd_mul_workspace_size(size_t dimension, size_t* length)`

  Size the workspace for `fastcd_mul_f64` and `fastcd_mul_batch_f64`. The length covers
  both automatic selections, for single products and for batches.

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
  two-sided reciprocal `x^(-1)`, not a general division operation.

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

  Set `out` to `x^2 = (alpha^2 - <X, X>) + 2 alpha X`, where `alpha = x[0]` is the
  scalar part and `X` the imaginary part, in `O(N)` time. If a finite input
  overflows this formula, recompute the affected coordinate with scaling.

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
  be zero, to broadcast one record, or at least `N`.

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
coincide. The alternating product and the commutator are exactly zero at
`N = 1, 2` and use the workspace from `fastcd_imag_cross_workspace_size`. The
associator uses its own composite workspace query, even at `N = 1`. The
formulas are those of [Derived operations](mathematics.md#derived-operations),
with `x = alpha + X` and `y = beta + Y`.

- `FastCDStatus fastcd_imag_cross_f64(double* out, const double* x, const double* y, size_t
  dimension, const FastCDWorkspaceF64* workspace)`

  Compute the alternating product `X cross Y = (XY - YX)/2` of the imaginary parts of
  `x` and `y` with the quasilinear recursion. The scalar parts of `x` and `y` are
  ignored, and `out[0] = +0.0`.

- `FastCDStatus fastcd_commutator_f64(double* out, const double* x, const double* y, size_t
  dimension, const FastCDWorkspaceF64* workspace)`

  Compute `xy - yx = 2 (X cross Y)`, with `out[0] = +0.0`.

- `FastCDStatus fastcd_anticommutator_f64(double* out, const double* x, const double* y,
  size_t dimension)`

  Compute `xy + yx = 2 (alpha beta - <X, Y>) + 2 (alpha Y + beta X)` in `O(N)` time,
  with no workspace.

- `FastCDStatus fastcd_associator_f64(double* out, const double* x, const double* y, const
  double* z, size_t dimension, const FastCDWorkspaceF64* workspace)`

  Compute `(xy)z - x(yz)` through four automatic products.

---

### Indexed basis operations

These functions use the basis signs `sigma_N` of
[Basis products](mathematics.md#basis-products).

- `FastCDStatus fastcd_basis_product(FastCDBasisProduct* product, size_t dimension, size_t
  left_index, size_t right_index)`

  Write the signed basis product `e_left_index e_right_index = product->sign *
  e_(product->index)`, where `product->index = left_index ^ right_index` and
  `product->sign = sigma_N(left_index, right_index)`. This takes `O(log N)` integer work.
  Errors are `FASTCD_NULL_POINTER`, `FASTCD_INVALID_DIMENSION`, and
  `FASTCD_INVALID_BASIS_INDEX`.

- `FastCDStatus fastcd_basis_lmul_f64(double* out, size_t basis_index, const double* x,
  size_t dimension)`

  Compute `e_basis_index x`, a signed permutation of `x`:
  `out[basis_index ^ i] = sigma_N(basis_index, i) x[i]`.

- `FastCDStatus fastcd_basis_rmul_f64(double* out, const double* x, size_t basis_index,
  size_t dimension)`

  Compute `x e_basis_index`: `out[i ^ basis_index] = sigma_N(i, basis_index) x[i]`.

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

---

## Arithmetic across coefficient fields

Complex and finite-field arithmetic is declared in `<fastcd/complex.h>` and
`<fastcd/finite.h>`. Both include `<fastcd/arithmetic.h>`, which declares the
operation selector `FastCDOperation` and `fastcd_quadratic_norm_f64`. These
headers are C11 with C++ linkage guards, and their functions allocate no
heap storage and retain no caller pointers.

A record is stored the same way over every field: the `N` coefficients in
the recursive standard basis, scalar first, contiguously. What one
coefficient occupies depends on the field:

| Field | One coefficient | One record |
| --- | --- | --- |
| R | one `double` | `N` doubles |
| C | one `FastCDComplexF64`, two doubles, real part first | `N` complex values, `2N` doubles |
| F_(p^d) | `d` canonical `uint32_t` residues, constant term first | `N * d` words |

A canonical residue is an integer in `0, ..., p - 1`.

### Rules shared by complex and finite-field functions

**Algorithms**

The functions accept `FASTCD_ALGORITHM_AUTO`, `FASTCD_ALGORITHM_DIRECT` and
`FASTCD_ALGORITHM_QUASILINEAR`. `AUTO` does not use the real tables: over C
and over finite fields it selects direct for `N <= 4` and quasilinear
otherwise. Cariow–Cariowa returns
`FASTCD_UNSUPPORTED_ALGORITHM` (complex) or
`FASTCD_FF_UNSUPPORTED_ALGORITHM` (finite).

**Aliasing**

- Exact `out == x` is allowed for add, subtract, negate, scale, divide,
  conjugate, imaginary part, reciprocal and square. Add and subtract also
  allow `out == y`.
- Every other output must be disjoint from the inputs. This includes the
  one-coefficient outputs of the inner product and the quadratic norm, and
  the outputs of basis multiplication.
- `x` and `y` must be the same complete record or disjoint. For scale and
  divide, `y` is one coefficient and may point into `x`, but the output must
  not overlap it.
- The active workspace prefix must be disjoint from every operand, from the
  output and, over finite fields, from the plan.

The functions check these rules before writing anything and report a
violation with `FASTCD_OVERLAPPING_BUFFERS` or
`FASTCD_FF_OVERLAPPING_BUFFERS`. A partial overlap between `x` and `y` is
rejected the same way. Over C, the third operand `z` of the composite
operations is not checked against `x` or `y`. Over finite fields, each
product inside a composite operation checks its own two factors, so `z` must
be the same record as `y` or disjoint from it in the associator, and the
same as or disjoint from both `x` and `y` in the operator commutators.

**Preconditions the library does not check**

Buffer lengths, alignment, initialized and unmodified field contexts and
plans, and canonical residues are the caller's responsibility.

**Errors and threads**

A nonzero status leaves every output unchanged. Field contexts and plans
are read-only during a call, so concurrent calls may share them; each call
needs its own output and workspace. Inputs, contexts and plans must not
change while a call that reads them is running, including from another
thread.

### Record operations

`FastCDOperation` selects the operation applied by `fastcd_apply_c128` and
`fastcd_apply_ff`:

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
above; `FASTCD_OP_IMAG_CROSS` is the alternating product. In the three
composite operations, `z` is the third operand:

- `FASTCD_OP_ASSOCIATOR` computes `(xy)z - x(yz)`.
- `FASTCD_OP_LEFT_OPERATOR_COMMUTATOR` computes `[L_x, L_y](z) = x(yz) - y(xz)`.
- `FASTCD_OP_RIGHT_OPERATOR_COMMUTATOR` computes `[R_x, R_y](z) = (zy)x - (zx)y`.

Operand pointers that an operation does not use may be null, and
`basis_index` is read only by the two basis operations. The `algorithm`
argument selects how records are multiplied in multiplication, the
alternating product, the commutator and the composite operations. Every
operation checks it, so Cariow–Cariowa or an unknown value is rejected even
by operations that do not multiply records. An operation outside
`FastCDOperation` returns `FASTCD_UNSUPPORTED_ALGORITHM` or
`FASTCD_FF_UNSUPPORTED_ALGORITHM`, and `FASTCD_OP_DIVIDE` by a zero
coefficient returns `FASTCD_ZERO_NORM` or `FASTCD_FF_ZERO_NORM`.

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
`fastcd_norm_squared_f64`, with the same scaled accumulation and
validation.

Over C and finite fields, `Q(x)` can be zero for a nonzero `x` (see
[The quadratic form](mathematics.md#the-quadratic-form)). The reciprocal of
such an element does not exist, and `FASTCD_OP_RECIPROCAL` returns
`FASTCD_ZERO_NORM` or `FASTCD_FF_ZERO_NORM` for it, as it does for zero.

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

- `dimension` is `N`, the number of complex coefficients in a record.
  Batch strides also count `FastCDComplexF64` values.
- Workspace lengths count **doubles**, as in the real API, and the workspace
  is described by `FastCDWorkspaceF64`.
- Complex values are passed as the two-double struct, never as C99
  `double _Complex`.

**Functions**

- `fastcd_workspace_size_c128` writes the number of doubles that `operation`
  needs. The length covers both direct and quasilinear evaluation, and it is
  zero for operations that do not multiply records. A batch needs the
  `FASTCD_OP_MULTIPLY` length.
- `fastcd_apply_c128` applies one record operation.
- `fastcd_mul_with_algorithm_c128` multiplies two records with the given
  algorithm, and `fastcd_mul_c128` multiplies them with `AUTO`.
- `fastcd_mul_batch_c128` multiplies `count` pairs of records.

**Numerical behavior**

NaN, infinity and overflow are numerical results, not errors.
`FASTCD_OP_CONJUGATE` negates coordinates `1` through `N - 1` and does not
conjugate the complex coefficients. The quadratic norm, the reciprocal and
the square accumulate `Q(x)` and the pairing in plain complex arithmetic,
without the scaling the real functions use, so they can overflow where
their real counterparts do not.

**Batches**

- `count == 0`: validate only the dimension and algorithm. Every pointer
  may be null, and nothing is written.
- `count == 1`: ignore all strides.
- `count > 1`: `out_stride` must be at least `N`, and each input stride must
  be zero, to broadcast one record, or at least `N`. The workspace is
  required.

All sizes and overlaps are checked before the first product is written.
The output span and the active workspace must be disjoint from both input
spans, and same-index input records must be identical or disjoint.

### Finite fields

#### Field contexts

```c
FastCDFFStatus fastcd_field_init(FastCDField* field, uint32_t prime,
        uint32_t degree, const uint32_t* modulus);
FastCDFFStatus fastcd_ff_field_init(FastCDField* field, uint32_t prime,
        uint32_t degree, const uint32_t* modulus);
FastCDFFStatus fastcd_ff_field_init_trusted(FastCDField* field, uint32_t prime,
        uint32_t degree, const uint32_t* modulus);
```

A `FastCDField` describes F_(p^d). Its public members are `prime`,
`degree` and `modulus[13]`; the remaining members hold precomputed
reduction data. `FASTCD_FF_MAX_DEGREE` is 12. Treat an initialized context
as immutable.

- `fastcd_field_init` accepts odd primes `3` through `INT32_MAX` and degrees
  `1` through `12`. With a null `modulus` it chooses a deterministic monic
  irreducible polynomial and records it in `field->modulus`. An explicit
  `modulus` has exactly `degree + 1` canonical residues, constant term first,
  with leading coefficient one. Setup checks that `prime` is prime and that
  the polynomial is irreducible, and leaves `field` unchanged on failure.
  `prime == 2` returns `FASTCD_FF_CHARACTERISTIC_TWO`.
- `fastcd_ff_field_init` is the lower-level constructor. It also accepts
  `prime == 2`, for scalar arithmetic only, since plans reject
  characteristic 2. It requires an explicit polynomial for `degree > 1`; at
  degree one a null `modulus` selects `t`.
- `fastcd_ff_field_init_trusted` makes the same range and shape checks but
  skips the primality and irreducibility checks. Use it only for parameters
  you have already validated.

Invalid parameters return `FASTCD_FF_INVALID_FIELD`, and
`fastcd_ff_field_init` returns `FASTCD_FF_NULL_POINTER` for `degree > 1`
without a `modulus`.

Neither the trusted constructor nor the scalar operations below are
available from Python.

#### Plans

```c
FastCDFFStatus fastcd_ff_plan_init(FastCDPlanFF* plan,
        const FastCDField* field, size_t level);
FastCDFFStatus fastcd_ff_cd_workspace_size(const FastCDPlanFF* plan,
        size_t* words);
FastCDFFStatus fastcd_apply_workspace_size_ff(const FastCDPlanFF* plan,
        FastCDOperation operation, size_t* workspace_words);
```

A `FastCDPlanFF` fixes the Cayley–Dickson level over one field.
`fastcd_ff_plan_init` copies the field into the plan and records `level`,
`dimension = 2^level`, the record length `words = dimension * degree`, the
product workspace length `workspace_words`, and the inverses of 2 and of
`dimension` modulo `p`. A plan contains no pointers; treat it as immutable.
Level zero is allowed, and its algebra is the coefficient field itself.
Plans reject characteristic 2 with `FASTCD_FF_CHARACTERISTIC_TWO`, and sizes
that do not fit `size_t` with `FASTCD_FF_SIZE_OVERFLOW`.

Finite-field workspaces are arrays of `uint32_t`, and their lengths count
words:

- `fastcd_ff_cd_workspace_size` writes the workspace needed by every product
  function: `fastcd_mul_ff`, `fastcd_mul_with_algorithm_ff`,
  `fastcd_ff_cd_mul` and `fastcd_mul_batch_ff`, with either algorithm. The
  value equals `plan->workspace_words`.
- `fastcd_apply_workspace_size_ff` writes the workspace needed by
  `fastcd_apply_ff` for one operation: the product workspace for
  multiplication, the alternating product and the commutator, that plus two
  records for the composite operations, and zero for the others.

A larger buffer may be supplied; only the required prefix is used and
checked for overlap.

#### Scalar arithmetic

```c
void fastcd_ff_field_add(uint32_t* out, const uint32_t* x,
        const uint32_t* y, const FastCDField* field);
void fastcd_ff_field_sub(uint32_t* out, const uint32_t* x,
        const uint32_t* y, const FastCDField* field);
void fastcd_ff_field_mul(uint32_t* out, const uint32_t* x,
        const uint32_t* y, const FastCDField* field);
void fastcd_field_neg(uint32_t* out, const uint32_t* x, const FastCDField* field);
FastCDFFStatus fastcd_field_inv(uint32_t* out, const uint32_t* x,
        const FastCDField* field);
uint32_t fastcd_ff_prime_mul(uint32_t a, uint32_t b, const FastCDField* field);
uint32_t fastcd_ff_field_reduce64(uint64_t x, const FastCDField* field);
```

These functions act on single elements of F_(p^d), each `degree` canonical
words long.

- Add, subtract, multiply and negate need no workspace and return nothing.
  `out` may equal `x` or `y` exactly; otherwise the buffers must be
  disjoint. They do not check buffer ranges or residues.
- `fastcd_field_inv` computes `x^(p^d - 2)`, the inverse of a nonzero `x`.
  For zero it returns `FASTCD_FF_ZERO_NORM` and leaves `out` unchanged.
  `out == x` is allowed.
- `fastcd_ff_prime_mul` multiplies two residues modulo the prime `p`. An
  extension context may be passed, but the result is a product in F_p, not
  of extension elements.
- `fastcd_ff_field_reduce64` reduces any `uint64_t` modulo `p`.

`fastcd_ff_field_mul`, `fastcd_ff_prime_mul` and `fastcd_ff_field_reduce64`
are inline functions in the header. For `degree > 1`, `fastcd_ff_field_mul`
calls the exported `fastcd_ff_field_mul_extension`, which applications should
not call directly. Define `FASTCD_FF_FIELD_EXTERN_ONLY` before including
`<fastcd/finite.h>` to call exported out-of-line versions of all three
instead. The header defines `FASTCD_FF_HAVE_INLINE_FIELD_API` to 1 so that
code can test for these inline functions.

#### Cayley–Dickson operations

```c
FastCDFFStatus fastcd_mul_with_algorithm_ff(uint32_t* out, const uint32_t* x,
        const uint32_t* y, const FastCDPlanFF* plan, FastCDAlgorithm algorithm,
        uint32_t* workspace, size_t workspace_words);
FastCDFFStatus fastcd_mul_ff(uint32_t* out, const uint32_t* x, const uint32_t* y,
        const FastCDPlanFF* plan, uint32_t* workspace, size_t workspace_words);
FastCDFFStatus fastcd_ff_cd_mul(uint32_t* out, const uint32_t* x, const uint32_t* y,
        const FastCDPlanFF* plan, uint32_t* workspace, size_t workspace_words);
FastCDFFStatus fastcd_apply_ff(uint32_t* out, const uint32_t* x,
        const uint32_t* y, const uint32_t* z, const FastCDPlanFF* plan,
        FastCDOperation operation, size_t basis_index, FastCDAlgorithm algorithm,
        uint32_t* workspace, size_t workspace_words);
FastCDFFStatus fastcd_mul_batch_ff(uint32_t* out, size_t out_stride,
        const uint32_t* x, size_t x_stride, const uint32_t* y, size_t y_stride,
        size_t count, const FastCDPlanFF* plan, FastCDAlgorithm algorithm,
        uint32_t* workspace, size_t workspace_words);
const char* fastcd_ff_status_string(FastCDFFStatus status);
```

**Parameters**

- A record has `plan->words` words. Lengths and strides count `uint32_t`
  words.
- `workspace_words` is the length of the supplied workspace; see
  [Plans](#plans) for the required length.

**Functions**

- `fastcd_mul_with_algorithm_ff` multiplies two records with the given
  algorithm, `fastcd_mul_ff` with `AUTO`, and `fastcd_ff_cd_mul` always with
  quasilinear.
- `fastcd_apply_ff` applies one [record operation](#record-operations).
- `fastcd_mul_batch_ff` multiplies `count` pairs of records under the
  complex batch rules, with `plan->words` in place of `N`. With
  `count == 0` it checks only that the plan is non-null and the algorithm
  is supported.
- `fastcd_ff_status_string` returns a fixed English message for a status.

**Statuses**

`FastCDFFStatus` is a separate enum from `FastCDStatus`. Its values, in
declaration order, are `FASTCD_FF_SUCCESS`, `FASTCD_FF_INVALID_LEVEL`,
`FASTCD_FF_INVALID_FIELD`, `FASTCD_FF_CHARACTERISTIC_TWO`,
`FASTCD_FF_INSUFFICIENT_WORKSPACE`, `FASTCD_FF_OVERLAPPING_BUFFERS`,
`FASTCD_FF_SIZE_OVERFLOW`, `FASTCD_FF_NULL_POINTER`,
`FASTCD_FF_UNSUPPORTED_ALGORITHM`, `FASTCD_FF_ZERO_NORM`,
`FASTCD_FF_INVALID_BASIS_INDEX`, and `FASTCD_FF_INVALID_STRIDE`.
`FASTCD_FF_INVALID_LEVEL` is not currently returned: a level whose
dimension does not fit `size_t` returns `FASTCD_FF_SIZE_OVERFLOW`.
