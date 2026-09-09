#ifndef FASTCD_H
#define FASTCD_H

#include <stdbool.h>
#include <stddef.h>

/*
 * FASTCD_SHARED and FASTCD_BUILDING_LIBRARY are supplied by the fastCD CMake
 * target.  Static-library users need no import or export annotation.
 */
#if defined(FASTCD_SHARED)
#  if defined(_WIN32) || defined(__CYGWIN__)
#    if defined(FASTCD_BUILDING_LIBRARY)
#      define FASTCD_API __declspec(dllexport)
#    else
#      define FASTCD_API __declspec(dllimport)
#    endif
#  elif defined(__GNUC__) || defined(__clang__)
#    define FASTCD_API __attribute__((visibility("default")))
#  else
#    define FASTCD_API
#  endif
#else
#  define FASTCD_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * fastCD implements the standard real Cayley--Dickson tower
 *
 *     A_0 = R,                 A_n = A_{n-1} x A_{n-1} (ordered pairs),
 *     (a,b)(c,d) = (ac - conjugate(d)b, da + b conjugate(c)).
 *
 * An element is a contiguous array of `dimension` real coefficients in the
 * recursive standard basis.  The scalar coefficient is stored at index zero,
 * and every valid dimension is a nonzero power of two.
 *
 * All array storage belongs to the caller.  No function retains an input,
 * output, or workspace pointer after returning.  Lengths and strides count
 * `double` elements, not bytes.  Exact in-place aliases are supported only
 * where a declaration says so.  Writable outputs and the queried active
 * prefix of a workspace must otherwise be disjoint from every array.  Two
 * read-only operands may be the same complete array, and the inputs of a
 * batch may share storage when every pair of same-index records is identical
 * or disjoint; distinct partially overlapping ranges are otherwise rejected.
 * A function returning an error leaves every output value unchanged.
 */

/* ---------- Definitions of enums and structs ---------- */

typedef enum {
    FASTCD_SUCCESS = 0,
    FASTCD_INVALID_DIMENSION,
    FASTCD_INVALID_LEVEL,
    FASTCD_INVALID_BASIS_INDEX,
    FASTCD_INSUFFICIENT_WORKSPACE,
    FASTCD_ZERO_NORM,
    FASTCD_OVERLAPPING_BUFFERS,
    FASTCD_INVALID_STRIDE,
    FASTCD_SIZE_OVERFLOW,
    FASTCD_NULL_POINTER,
    FASTCD_UNSUPPORTED_ALGORITHM
} FastCDStatus;

/*
 * Multiplication algorithms.  DIRECT evaluates the defining doubling
 * recursion with N^2 real multiplications.  CARIOW_CARIOWA uses the
 * transform-and-correction kernels with N(N-1)/2 + 2 multiplications,
 * available through dimension FASTCD_CARIOW_CARIOWA_MAX_DIMENSION and not
 * recommended above 64.  QUASILINEAR uses O(N log N) real
 * arithmetic at every dimension.  AUTO resolves per dimension from two
 * benchmark-derived tables, one for single products and one for batches of
 * at least two products; query them with fastcd_default_algorithm and
 * fastcd_default_batch_algorithm.
 */
typedef enum {
    FASTCD_ALGORITHM_AUTO = 0,
    FASTCD_ALGORITHM_DIRECT,
    FASTCD_ALGORITHM_CARIOW_CARIOWA,
    FASTCD_ALGORITHM_QUASILINEAR
} FastCDAlgorithm;

#define FASTCD_CARIOW_CARIOWA_MAX_DIMENSION ((size_t)1024)

typedef struct FastCDWorkspaceF64 {
    double* data;
    size_t length; // number of available double elements
} FastCDWorkspaceF64;

/* A workspace descriptor object must not overlap any input or output array. */

typedef struct FastCDBasisProduct {
    size_t index;
    int sign; // either +1 or -1
} FastCDBasisProduct;

/* ---------- Library information and errors ---------- */

// Return the semantic version of the linked fastCD library
FASTCD_API const char* fastcd_version_string(void);

// Return a static description of a FastCDStatus value
FASTCD_API const char* fastcd_status_string(FastCDStatus status);

// Return the static lowercase name of a FastCDAlgorithm value
FASTCD_API const char* fastcd_algorithm_string(FastCDAlgorithm algorithm);

/* ---------- Dimension utilities ---------- */

// Check whether dimension is a nonzero power of two
FASTCD_API bool fastcd_valid_dimension(size_t dimension);

// Compute dimension = 2^level with overflow checking
FASTCD_API FastCDStatus fastcd_dimension(size_t level, size_t* dimension);

// Compute the unique level satisfying dimension = 2^level
FASTCD_API FastCDStatus fastcd_level(size_t dimension, size_t* level);

/* ---------- Algorithm selection ---------- */

// Write the algorithm that FASTCD_ALGORITHM_AUTO selects for one product
FASTCD_API FastCDStatus fastcd_default_algorithm(size_t dimension,
        FastCDAlgorithm* algorithm);

// Write the algorithm that FASTCD_ALGORITHM_AUTO selects for a batch of at
// least two products (fastcd_mul_batch_f64 with count >= 2); it equals the
// single-product choice wherever the build cannot pair products
FASTCD_API FastCDStatus fastcd_default_batch_algorithm(size_t dimension,
        FastCDAlgorithm* algorithm);

// Check whether an algorithm can multiply elements of this dimension
FASTCD_API bool fastcd_algorithm_supports_dimension(FastCDAlgorithm algorithm,
        size_t dimension);

/* ---------- Workspace queries ---------- */

// Return the workspace length required by fastcd_mul_f64 and
// fastcd_mul_batch_f64 (it covers both automatic selections)
FASTCD_API FastCDStatus fastcd_mul_workspace_size(size_t dimension,
        size_t* length);

// Return the workspace length required by the explicit-algorithm products
FASTCD_API FastCDStatus fastcd_mul_algorithm_workspace_size(size_t dimension,
        FastCDAlgorithm algorithm, size_t* length);

// Return the workspace length required by fastcd_imag_cross_f64 and
// fastcd_commutator_f64
FASTCD_API FastCDStatus fastcd_imag_cross_workspace_size(size_t dimension,
        size_t* length);

// Return the workspace length required by fastcd_associator_f64
FASTCD_API FastCDStatus fastcd_associator_workspace_size(size_t dimension,
        size_t* length);

// Return the workspace length required by fastcd_lop_commutator_apply_f64
FASTCD_API FastCDStatus fastcd_lop_commutator_workspace_size(size_t dimension,
        size_t* length);

// Return the workspace length required by fastcd_rop_commutator_apply_f64
FASTCD_API FastCDStatus fastcd_rop_commutator_workspace_size(size_t dimension,
        size_t* length);

/* ---------- Element construction ---------- */

// Set out to the scalar element value * 1
FASTCD_API FastCDStatus fastcd_scalar_f64(double* out, size_t dimension,
        double value);

// Set out to the standard basis element e_index
FASTCD_API FastCDStatus fastcd_basis_f64(double* out, size_t dimension,
        size_t index);

/* ---------- Coordinate arithmetic ---------- */

// Set out = x + y; out may equal x or y exactly
FASTCD_API FastCDStatus fastcd_add_f64(double* out, const double* x,
        const double* y, size_t dimension);

// Set out = x - y; out may equal x or y exactly
FASTCD_API FastCDStatus fastcd_sub_f64(double* out, const double* x,
        const double* y, size_t dimension);

// Set out = -x; out may equal x exactly
FASTCD_API FastCDStatus fastcd_neg_f64(double* out, const double* x,
        size_t dimension);

// Set out = scalar * x; out may equal x exactly
FASTCD_API FastCDStatus fastcd_scale_f64(double* out, const double* x,
        double scalar, size_t dimension);

/* ---------- Structural operations ---------- */

/*
 * Norm-related operations propagate IEEE NaN and infinity values.  Normalize
 * and reciprocal return FASTCD_ZERO_NORM for an exactly all-zero finite input.
 */

// Set out to the standard Cayley--Dickson conjugate of x; out may equal x
FASTCD_API FastCDStatus fastcd_conjugate_f64(double* out, const double* x,
        size_t dimension);

// Set out to the imaginary part of x; out may equal x exactly
FASTCD_API FastCDStatus fastcd_imag_part_f64(double* out, const double* x,
        size_t dimension);

// Compute the Euclidean inner product using sequential double accumulation
FASTCD_API FastCDStatus fastcd_inner_f64(double* value, const double* x,
        const double* y, size_t dimension);

// Compute x conjugate(x) = sum_i x_i^2 with scaled accumulation
FASTCD_API FastCDStatus fastcd_norm_squared_f64(double* value,
        const double* x, size_t dimension);

// Compute the Euclidean norm of x using scaled accumulation
FASTCD_API FastCDStatus fastcd_norm_f64(double* value, const double* x,
        size_t dimension);

// Set out = x / norm(x); out may equal x exactly
FASTCD_API FastCDStatus fastcd_normalize_f64(double* out, const double* x,
        size_t dimension);

// Set out = conjugate(x) / norm_squared(x); out may equal x exactly
FASTCD_API FastCDStatus fastcd_reciprocal_f64(double* out, const double* x,
        size_t dimension);

/* ---------- Cayley--Dickson multiplication ---------- */

/*
 * Set out = xy with the automatically selected algorithm.  The output and the
 * queried active workspace prefix must be disjoint from all active arrays.
 * x and y may be the same complete array.  The workspace may be NULL only
 * when the corresponding workspace query returns zero.  A NaN coordinate in
 * either operand makes every output coordinate NaN, and an infinite
 * coordinate makes every output coordinate non-finite; which of those become
 * NaN rather than infinite depends on the algorithm.
 */
FASTCD_API FastCDStatus fastcd_mul_f64(double* out, const double* x,
        const double* y, size_t dimension,
        const FastCDWorkspaceF64* workspace);

// Set out = xy with an explicitly selected algorithm; see
// fastcd_mul_algorithm_workspace_size for its scratch requirement
FASTCD_API FastCDStatus fastcd_mul_with_algorithm_f64(double* out,
        const double* x, const double* y, size_t dimension,
        FastCDAlgorithm algorithm, const FastCDWorkspaceF64* workspace);

// Set out = x^2 using the linear-time quadratic identity; out may equal x
FASTCD_API FastCDStatus fastcd_square_f64(double* out, const double* x,
        size_t dimension);

/*
 * Multiply count pairs of contiguous elements.  Strides are measured between
 * element starts in doubles.  An input stride of zero broadcasts that input.
 * For count greater than one, nonzero strides must be at least dimension and
 * out_stride must be nonzero.  When count is one, all strides are ignored;
 * when count is zero, only dimension and algorithm are validated and every
 * pointer may be NULL.  The two inputs may share storage, as when one
 * broadcasts a record of the other, provided no pair of same-index records
 * partially overlaps.  One multiplication workspace is reused across the
 * batch.
 */
FASTCD_API FastCDStatus fastcd_mul_batch_f64(double* out, size_t out_stride,
        const double* x, size_t x_stride, const double* y, size_t y_stride,
        size_t count, size_t dimension, const FastCDWorkspaceF64* workspace);

// Multiply a batch with an explicitly selected algorithm
FASTCD_API FastCDStatus fastcd_mul_batch_with_algorithm_f64(double* out,
        size_t out_stride, const double* x, size_t x_stride, const double* y,
        size_t y_stride, size_t count, size_t dimension,
        FastCDAlgorithm algorithm, const FastCDWorkspaceF64* workspace);

/* ---------- Noncommutative and nonassociative products ---------- */

/*
 * Set out = Im(x) cross Im(y), where cross is one half of the commutator on
 * imaginary inputs.  The scalar coefficients of x and y are ignored and the
 * scalar coefficient of out is zero.
 */
FASTCD_API FastCDStatus fastcd_imag_cross_f64(double* out, const double* x,
        const double* y, size_t dimension,
        const FastCDWorkspaceF64* workspace);

// Set out = xy - yx using one alternating product
FASTCD_API FastCDStatus fastcd_commutator_f64(double* out, const double* x,
        const double* y, size_t dimension,
        const FastCDWorkspaceF64* workspace);

// Set out = xy + yx using the linear-time scalar--imaginary formula
FASTCD_API FastCDStatus fastcd_anticommutator_f64(double* out,
        const double* x, const double* y, size_t dimension);

// Set out = (xy)z - x(yz)
FASTCD_API FastCDStatus fastcd_associator_f64(double* out, const double* x,
        const double* y, const double* z, size_t dimension,
        const FastCDWorkspaceF64* workspace);

/* ---------- Indexed basis operations ---------- */

// Compute e_left_index * e_right_index = sign * e_index
FASTCD_API FastCDStatus fastcd_basis_product(FastCDBasisProduct* product,
        size_t dimension, size_t left_index, size_t right_index);

// Set out = e_basis_index * x using a signed permutation of coordinates
FASTCD_API FastCDStatus fastcd_basis_lmul_f64(double* out, size_t basis_index,
        const double* x, size_t dimension);

// Set out = x * e_basis_index using a signed permutation of coordinates
FASTCD_API FastCDStatus fastcd_basis_rmul_f64(double* out, const double* x,
        size_t basis_index, size_t dimension);

/* ---------- Matrix-free operator commutators ---------- */

// Set out = [L_a,L_b](x) = a(bx) - b(ax)
FASTCD_API FastCDStatus fastcd_lop_commutator_apply_f64(double* out,
        const double* a, const double* b, const double* x,
        size_t dimension, const FastCDWorkspaceF64* workspace);

// Set out = [R_a,R_b](x) = (xb)a - (xa)b
FASTCD_API FastCDStatus fastcd_rop_commutator_apply_f64(double* out,
        const double* a, const double* b, const double* x,
        size_t dimension, const FastCDWorkspaceF64* workspace);

#ifdef __cplusplus
}
#endif

#endif
