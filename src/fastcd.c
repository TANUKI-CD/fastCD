#include <limits.h>
#include <math.h>
#include <stdint.h>

#include "fastcd/fastcd.h"
#include "fastcd_internal.h"
#include "pair.h"

#define FASTCD_VERSION "1.0.0"

/* ---------- Definitions of private structs ---------- */

typedef struct FastCDScaledSumSquares {
    double scale;
    double sum_squares;
    bool has_infinity;
    bool has_nan;
} FastCDScaledSumSquares;

/* ---------- Checked size and memory helpers ---------- */

// Multiply two size_t values when the product is representable
static bool checked_size_mul(size_t x, size_t y, size_t* product) {
    if (product == NULL || (x != 0 && y > SIZE_MAX / x)) return false;
    *product = x * y;
    return true;
}

// Add two size_t values when the sum is representable
static bool checked_size_add(size_t x, size_t y, size_t* sum) {
    if (sum == NULL || x > SIZE_MAX - y) return false;
    *sum = x + y;
    return true;
}

// Return the byte extent of an array of doubles
static FastCDStatus double_array_bytes(size_t length, size_t* bytes) {
    if (bytes == NULL) return FASTCD_NULL_POINTER;
    if (!checked_size_mul(length, sizeof(double), bytes)) {
        return FASTCD_SIZE_OVERFLOW;
    }
    return FASTCD_SUCCESS;
}

// Return whether two nonempty byte ranges overlap
static bool ranges_overlap(const void* x, size_t x_bytes, const void* y,
        size_t y_bytes) {
    uintptr_t x_address;
    uintptr_t y_address;

    if (x == NULL || y == NULL || x_bytes == 0 || y_bytes == 0) return false;
    x_address = (uintptr_t)x;
    y_address = (uintptr_t)y;
    if (x_address <= y_address) return y_address - x_address < x_bytes;
    return x_address - y_address < y_bytes;
}

// Check output/input overlap while optionally permitting an exact alias
static bool invalid_output_overlap(const void* out, size_t out_bytes,
        const void* input, size_t input_bytes, bool allow_exact) {
    if (out == input) return !allow_exact;
    return ranges_overlap(out, out_bytes, input, input_bytes);
}

// Return whether two inputs overlap without being the same complete range
static bool partial_input_overlap(const void* x, size_t x_bytes,
        const void* y, size_t y_bytes) {
    if (x == y && x_bytes == y_bytes) return false;
    return ranges_overlap(x, x_bytes, y, y_bytes);
}

// Validate one element dimension and return its byte extent
static FastCDStatus validate_element_extent(size_t dimension, size_t* bytes) {
    if (!fastcd_valid_dimension(dimension)) return FASTCD_INVALID_DIMENSION;
    return double_array_bytes(dimension, bytes);
}

// Validate a supplied workspace and all ranges with which it must be disjoint
static FastCDStatus validate_workspace(const FastCDWorkspaceF64* workspace,
        size_t required, const void* out, size_t out_bytes,
        const void* x, size_t x_bytes, const void* y, size_t y_bytes,
        const void* z, size_t z_bytes) {
    size_t workspace_bytes;
    FastCDStatus status;

    if (required == 0) return FASTCD_SUCCESS;
    if (workspace == NULL || workspace->data == NULL) {
        return FASTCD_NULL_POINTER;
    }
    if (workspace->length < required) return FASTCD_INSUFFICIENT_WORKSPACE;
    status = double_array_bytes(required, &workspace_bytes);
    if (status != FASTCD_SUCCESS) return status;
    if (ranges_overlap(workspace->data, workspace_bytes, out, out_bytes)
            || ranges_overlap(workspace->data, workspace_bytes, x, x_bytes)
            || ranges_overlap(workspace->data, workspace_bytes, y, y_bytes)
            || ranges_overlap(workspace->data, workspace_bytes, z, z_bytes)) {
        return FASTCD_OVERLAPPING_BUFFERS;
    }
    return FASTCD_SUCCESS;
}

/* ---------- Basis and numerical helpers ---------- */

// Compute the multiplication sign of two valid standard-basis indices
static int basis_sign_unchecked(size_t dimension, size_t left, size_t right) {
    int sign = 1;

    while (dimension > 1) {
        size_t half = dimension >> 1;

        if (left < half) {
            if (right >= half) {
                size_t temporary = left;
                left = right - half;
                right = temporary;
            }
        } else if (right < half) {
            left -= half;
            if (right != 0) sign = -sign;
        } else {
            size_t left_low = left - half;
            size_t right_low = right - half;

            if (right_low == 0) sign = -sign;
            left = right_low;
            right = left_low;
        }
        dimension = half;
    }
    return sign;
}

// Compute scaled sum-of-squares data without overflowing finite intermediates
static FastCDScaledSumSquares scaled_sum_squares(const double* x,
        size_t dimension) {
    FastCDScaledSumSquares result = {0.0, 1.0, false, false};

    for (size_t i = 0; i < dimension; i++) {
        double absolute = fabs(x[i]);

        if (isnan(absolute)) {
            result.has_nan = true;
        } else if (isinf(absolute)) {
            result.has_infinity = true;
        } else if (absolute != 0.0) {
            if (result.scale < absolute) {
                double ratio = result.scale / absolute;
                result.sum_squares = 1.0 + result.sum_squares * ratio * ratio;
                result.scale = absolute;
            } else {
                double ratio = absolute / result.scale;
                result.sum_squares += ratio * ratio;
            }
        }
    }
    return result;
}

// Convert scaled sum-of-squares data to a Euclidean norm
static double norm_from_scaled(FastCDScaledSumSquares scaled) {
    if (scaled.has_nan) return NAN;
    if (scaled.has_infinity) return INFINITY;
    if (scaled.scale == 0.0) return 0.0;
    return scaled.scale * sqrt(scaled.sum_squares);
}

// Convert scaled sum-of-squares data to a squared Euclidean norm
static double norm_squared_from_scaled(FastCDScaledSumSquares scaled) {
    if (scaled.has_nan) return NAN;
    if (scaled.has_infinity) return INFINITY;
    if (scaled.scale == 0.0) return 0.0;
    if (scaled.scale >= 1.0) {
        return (scaled.scale * scaled.scale) * scaled.sum_squares;
    }
    return scaled.scale * (scaled.scale * scaled.sum_squares);
}

// Compute 2xy while avoiding a needlessly overflowing or underflowing double
static double twice_product(double x, double y) {
    if (fabs(x) <= fabs(y)) return (x + x) * y;
    return x * (y + y);
}

// Recompute the scalar part of x^2 after finite squared terms overflow
static double scaled_square_scalar(const double* x, size_t dimension) {
    double scale = 0.0;
    double normalized;

    for (size_t i = 0; i < dimension; i++) {
        double absolute = fabs(x[i]);
        if (scale < absolute) scale = absolute;
    }
    if (scale == 0.0) return 0.0;

    normalized = (x[0] / scale) * (x[0] / scale);
    for (size_t i = 1; i < dimension; i++) {
        double coordinate = x[i] / scale;
        normalized -= coordinate * coordinate;
    }
    return (normalized * scale) * scale;
}

/* ---------- Algorithm selection ---------- */

// Return whether a value names one of the public algorithms
static bool valid_algorithm(FastCDAlgorithm algorithm) {
    return algorithm == FASTCD_ALGORITHM_AUTO
        || algorithm == FASTCD_ALGORITHM_DIRECT
        || algorithm == FASTCD_ALGORITHM_CARIOW_CARIOWA
        || algorithm == FASTCD_ALGORITHM_QUASILINEAR;
}

/*
 * Algorithms selected by FASTCD_ALGORITHM_AUTO for dimension 2^level.  The
 * last entry of each table applies to every larger dimension.  Single
 * products and batches of at least two products have separate tables
 * because batches run the two-product SIMD kernels (where the build and
 * dimension allow it), which change the ranking below dimension 32.  Both
 * tables come from benchmarks/benchmark.py (batch and single modes) and are
 * the only place to edit when a new benchmark run reorders the algorithms.
 */
static const FastCDAlgorithm auto_single_by_level[] = {
    FASTCD_ALGORITHM_QUASILINEAR,   /* 1 */
    FASTCD_ALGORITHM_QUASILINEAR,   /* 2 */
    FASTCD_ALGORITHM_DIRECT,        /* 4 */
    FASTCD_ALGORITHM_DIRECT,        /* 8 */
    FASTCD_ALGORITHM_DIRECT,        /* 16 */
    FASTCD_ALGORITHM_QUASILINEAR    /* 32 and above */
};

static const FastCDAlgorithm auto_batch_by_level[] = {
    FASTCD_ALGORITHM_QUASILINEAR,   /* 1 */
    FASTCD_ALGORITHM_QUASILINEAR,   /* 2 */
    FASTCD_ALGORITHM_QUASILINEAR,   /* 4 */
    FASTCD_ALGORITHM_DIRECT,        /* 8 */
    FASTCD_ALGORITHM_CARIOW_CARIOWA,/* 16 */
    FASTCD_ALGORITHM_QUASILINEAR    /* 32 and above */
};

// Look up a per-level table for a valid dimension
static FastCDAlgorithm table_algorithm(const FastCDAlgorithm* table,
        size_t entries, size_t dimension) {
    size_t level = 0;

    while (dimension > 1 && level + 1 < entries) {
        dimension >>= 1;
        level++;
    }
    return table[level];
}

// Return the algorithm that AUTO selects for one product at a valid dimension
static FastCDAlgorithm default_algorithm_unchecked(size_t dimension) {
    return table_algorithm(auto_single_by_level,
            sizeof(auto_single_by_level) / sizeof(auto_single_by_level[0]),
            dimension);
}

// Return the algorithm that AUTO selects for a batch at a valid dimension
static FastCDAlgorithm default_batch_algorithm_unchecked(size_t dimension) {
    /* The batch table describes the paired SIMD kernels; a build or dimension
     * without pairing runs batches with the scalar kernels, whose ranking is
     * the single-product one. */
    if (!FASTCD_HAVE_PAIR || dimension > FASTCD_PAIR_MAX_DIMENSION) {
        return default_algorithm_unchecked(dimension);
    }
    return table_algorithm(auto_batch_by_level,
            sizeof(auto_batch_by_level) / sizeof(auto_batch_by_level[0]),
            dimension);
}

// Return whether a resolved algorithm supports a valid dimension
static bool supports_unchecked(FastCDAlgorithm algorithm, size_t dimension) {
    if (algorithm == FASTCD_ALGORITHM_CARIOW_CARIOWA) {
        return fastcd_cc_supported(dimension);
    }
    return valid_algorithm(algorithm);
}

// Resolve AUTO and reject algorithms unavailable at this valid dimension
static FastCDStatus resolve_algorithm(size_t dimension,
        FastCDAlgorithm algorithm, bool batch, FastCDAlgorithm* effective) {
    if (!valid_algorithm(algorithm)) return FASTCD_UNSUPPORTED_ALGORITHM;
    if (algorithm == FASTCD_ALGORITHM_AUTO) {
        algorithm = batch ? default_batch_algorithm_unchecked(dimension)
            : default_algorithm_unchecked(dimension);
    }
    if (!supports_unchecked(algorithm, dimension)) {
        return FASTCD_UNSUPPORTED_ALGORITHM;
    }
    *effective = algorithm;
    return FASTCD_SUCCESS;
}

// Return the scratch length of a resolved algorithm at a valid dimension
static size_t algorithm_workspace_unchecked(size_t dimension,
        FastCDAlgorithm effective) {
    switch (effective) {
        case FASTCD_ALGORITHM_DIRECT:
            return fastcd_direct_workspace(dimension);
        case FASTCD_ALGORITHM_QUASILINEAR:
            return fastcd_quasilinear_workspace(dimension);
        case FASTCD_ALGORITHM_CARIOW_CARIOWA:
            return fastcd_cc_workspace(dimension);
        case FASTCD_ALGORITHM_AUTO:
        default:
            return 0;
    }
}

// Return the scratch length of the imaginary cross product
static size_t cross_workspace_unchecked(size_t dimension) {
    return fastcd_quasilinear_workspace(dimension);
}

// Multiply one validated pair with a resolved algorithm
static void mul_unchecked(double* restrict out, const double* x,
        const double* y, size_t dimension, FastCDAlgorithm effective,
        double* restrict workspace) {
    switch (effective) {
        case FASTCD_ALGORITHM_DIRECT:
            fastcd_direct_mul(out, x, y, dimension, workspace);
            return;
        case FASTCD_ALGORITHM_QUASILINEAR:
            fastcd_quasilinear_mul(out, x, y, dimension, workspace);
            return;
        case FASTCD_ALGORITHM_CARIOW_CARIOWA:
        case FASTCD_ALGORITHM_AUTO:
        default:
            fastcd_cc_mul(out, x, y, dimension, workspace);
            return;
    }
}

// Multiply a validated batch with a resolved algorithm
static void mul_batch_unchecked(double* restrict out, size_t out_stride,
        const double* x, size_t x_stride, const double* y, size_t y_stride,
        size_t count, size_t dimension, FastCDAlgorithm effective,
        double* restrict workspace) {
    switch (effective) {
        case FASTCD_ALGORITHM_DIRECT:
            fastcd_direct_mul_batch(out, out_stride, x, x_stride, y, y_stride,
                    count, dimension, workspace);
            return;
        case FASTCD_ALGORITHM_QUASILINEAR:
            fastcd_quasilinear_mul_batch(out, out_stride, x, x_stride, y,
                    y_stride, count, dimension, workspace);
            return;
        case FASTCD_ALGORITHM_CARIOW_CARIOWA:
        case FASTCD_ALGORITHM_AUTO:
        default:
            fastcd_cc_mul_batch(out, out_stride, x, x_stride, y, y_stride,
                    count, dimension, workspace);
            return;
    }
}

// Evaluate the imaginary cross product after validation
static void cross_unchecked(double* restrict out, const double* x,
        const double* y, size_t dimension, double* restrict workspace) {
    fastcd_quasilinear_cross(out, x, y, dimension, workspace);
}

/* ---------- Library information and errors ---------- */

// Return the linked fastCD semantic version string
const char* fastcd_version_string(void) {
    return FASTCD_VERSION;
}

// Return a stable message for one public status code
const char* fastcd_status_string(FastCDStatus status) {
    switch (status) {
        case FASTCD_SUCCESS:
            return "Success";
        case FASTCD_INVALID_DIMENSION:
            return "Invalid Cayley--Dickson dimension";
        case FASTCD_INVALID_LEVEL:
            return "Invalid Cayley--Dickson level";
        case FASTCD_INVALID_BASIS_INDEX:
            return "Invalid standard-basis index";
        case FASTCD_INSUFFICIENT_WORKSPACE:
            return "Insufficient multiplication workspace";
        case FASTCD_ZERO_NORM:
            return "Element has zero Euclidean norm";
        case FASTCD_OVERLAPPING_BUFFERS:
            return "Unsupported overlapping buffers";
        case FASTCD_INVALID_STRIDE:
            return "Invalid batch stride";
        case FASTCD_SIZE_OVERFLOW:
            return "Size calculation overflow";
        case FASTCD_NULL_POINTER:
            return "Required pointer is null";
        case FASTCD_UNSUPPORTED_ALGORITHM:
            return "Algorithm is unknown or unavailable at this dimension";
        default:
            return "Unknown fastCD status";
    }
}

// Return the static lowercase name of a FastCDAlgorithm value
const char* fastcd_algorithm_string(FastCDAlgorithm algorithm) {
    switch (algorithm) {
        case FASTCD_ALGORITHM_AUTO:
            return "auto";
        case FASTCD_ALGORITHM_DIRECT:
            return "direct";
        case FASTCD_ALGORITHM_CARIOW_CARIOWA:
            return "cariow_cariowa";
        case FASTCD_ALGORITHM_QUASILINEAR:
            return "quasilinear";
        default:
            return "unknown";
    }
}

/* ---------- Dimension utilities ---------- */

// Return whether a dimension is a nonzero power of two
bool fastcd_valid_dimension(size_t dimension) {
    return dimension != 0 && (dimension & (dimension - 1)) == 0;
}

// Compute dimension as two to a checked Cayley--Dickson level
FastCDStatus fastcd_dimension(size_t level, size_t* dimension) {
    if (dimension == NULL) return FASTCD_NULL_POINTER;
    if (level >= sizeof(size_t) * CHAR_BIT) return FASTCD_INVALID_LEVEL;
    *dimension = (size_t)1 << level;
    return FASTCD_SUCCESS;
}

// Recover the unique doubling level of a valid dimension
FastCDStatus fastcd_level(size_t dimension, size_t* level) {
    size_t result = 0;

    if (level == NULL) return FASTCD_NULL_POINTER;
    if (!fastcd_valid_dimension(dimension)) return FASTCD_INVALID_DIMENSION;
    while (dimension > 1) {
        dimension >>= 1;
        result++;
    }
    *level = result;
    return FASTCD_SUCCESS;
}

/* ---------- Algorithm selection ---------- */

// Write the algorithm that FASTCD_ALGORITHM_AUTO selects at this dimension
FastCDStatus fastcd_default_algorithm(size_t dimension,
        FastCDAlgorithm* algorithm) {
    if (algorithm == NULL) return FASTCD_NULL_POINTER;
    if (!fastcd_valid_dimension(dimension)) return FASTCD_INVALID_DIMENSION;
    *algorithm = default_algorithm_unchecked(dimension);
    return FASTCD_SUCCESS;
}

// Write the algorithm that FASTCD_ALGORITHM_AUTO selects for a batch
FastCDStatus fastcd_default_batch_algorithm(size_t dimension,
        FastCDAlgorithm* algorithm) {
    if (algorithm == NULL) return FASTCD_NULL_POINTER;
    if (!fastcd_valid_dimension(dimension)) return FASTCD_INVALID_DIMENSION;
    *algorithm = default_batch_algorithm_unchecked(dimension);
    return FASTCD_SUCCESS;
}

// Check whether an algorithm can multiply elements of this dimension
bool fastcd_algorithm_supports_dimension(FastCDAlgorithm algorithm,
        size_t dimension) {
    FastCDAlgorithm effective;

    if (!fastcd_valid_dimension(dimension)) return false;
    return resolve_algorithm(dimension, algorithm, false, &effective)
        == FASTCD_SUCCESS;
}

/* ---------- Workspace queries ---------- */

// Validate a dimension and algorithm, then compute the multiplication scratch
static FastCDStatus mul_workspace_query(size_t dimension,
        FastCDAlgorithm algorithm, size_t* length, FastCDAlgorithm* effective) {
    size_t bytes;
    size_t result;
    FastCDStatus status = validate_element_extent(dimension, &bytes);

    if (status != FASTCD_SUCCESS) return status;
    status = resolve_algorithm(dimension, algorithm, false, effective);
    if (status != FASTCD_SUCCESS) return status;
    result = algorithm_workspace_unchecked(dimension, *effective);
    if (algorithm == FASTCD_ALGORITHM_AUTO) {
        /* One query serves both single products and batches. */
        size_t batch_result = algorithm_workspace_unchecked(dimension,
                default_batch_algorithm_unchecked(dimension));

        if (batch_result > result) result = batch_result;
    }
    status = double_array_bytes(result, &bytes);
    if (status != FASTCD_SUCCESS) return status;
    *length = result;
    return FASTCD_SUCCESS;
}

// Return the workspace length required by the explicit-algorithm products
FastCDStatus fastcd_mul_algorithm_workspace_size(size_t dimension,
        FastCDAlgorithm algorithm, size_t* length) {
    FastCDAlgorithm effective;
    size_t result;
    FastCDStatus status;

    if (length == NULL) return FASTCD_NULL_POINTER;
    status = mul_workspace_query(dimension, algorithm, &result, &effective);
    if (status != FASTCD_SUCCESS) return status;
    *length = result;
    return FASTCD_SUCCESS;
}

// Return the real scratch length required by automatic multiplication
FastCDStatus fastcd_mul_workspace_size(size_t dimension, size_t* length) {
    return fastcd_mul_algorithm_workspace_size(dimension,
            FASTCD_ALGORITHM_AUTO, length);
}

// Return the scratch length shared by imaginary cross and commutator calls
FastCDStatus fastcd_imag_cross_workspace_size(size_t dimension,
        size_t* length) {
    size_t bytes;
    size_t result;
    FastCDStatus status;

    if (length == NULL) return FASTCD_NULL_POINTER;
    status = validate_element_extent(dimension, &bytes);
    if (status != FASTCD_SUCCESS) return status;
    result = cross_workspace_unchecked(dimension);
    status = double_array_bytes(result, &bytes);
    if (status != FASTCD_SUCCESS) return status;
    *length = result;
    return FASTCD_SUCCESS;
}

// Return two element buffers plus the automatic multiplication workspace
static FastCDStatus composite_workspace_size(size_t dimension,
        size_t* length) {
    size_t multiplication_length;
    size_t two_elements;
    size_t result;
    size_t bytes;
    FastCDStatus status;

    if (length == NULL) return FASTCD_NULL_POINTER;
    status = fastcd_mul_workspace_size(dimension, &multiplication_length);
    if (status != FASTCD_SUCCESS) return status;
    if (!checked_size_mul(dimension, 2, &two_elements)
            || !checked_size_add(two_elements, multiplication_length,
                &result)) {
        return FASTCD_SIZE_OVERFLOW;
    }
    status = double_array_bytes(result, &bytes);
    if (status != FASTCD_SUCCESS) return status;
    *length = result;
    return FASTCD_SUCCESS;
}

// Return the workspace required to evaluate one associator
FastCDStatus fastcd_associator_workspace_size(size_t dimension,
        size_t* length) {
    return composite_workspace_size(dimension, length);
}

// Return the workspace required by a left operator commutator
FastCDStatus fastcd_lop_commutator_workspace_size(size_t dimension,
        size_t* length) {
    return composite_workspace_size(dimension, length);
}

// Return the workspace required by a right operator commutator
FastCDStatus fastcd_rop_commutator_workspace_size(size_t dimension,
        size_t* length) {
    return composite_workspace_size(dimension, length);
}

/* ---------- Element construction ---------- */

// Embed one real scalar in coordinate zero
FastCDStatus fastcd_scalar_f64(double* out, size_t dimension, double value) {
    size_t bytes;
    FastCDStatus status = validate_element_extent(dimension, &bytes);

    if (status != FASTCD_SUCCESS) return status;
    if (out == NULL) return FASTCD_NULL_POINTER;
    out[0] = value;
    for (size_t i = 1; i < dimension; i++) out[i] = 0.0;
    return FASTCD_SUCCESS;
}

// Set one output coefficient to form a standard basis element
FastCDStatus fastcd_basis_f64(double* out, size_t dimension, size_t index) {
    size_t bytes;
    FastCDStatus status = validate_element_extent(dimension, &bytes);

    if (status != FASTCD_SUCCESS) return status;
    if (out == NULL) return FASTCD_NULL_POINTER;
    if (index >= dimension) return FASTCD_INVALID_BASIS_INDEX;
    for (size_t i = 0; i < dimension; i++) out[i] = 0.0;
    out[index] = 1.0;
    return FASTCD_SUCCESS;
}

/* ---------- Coordinate arithmetic ---------- */

// Validate one coordinatewise binary operation and its aliasing contract
static FastCDStatus validate_binary_coordinate_args(double* out,
        const double* x, const double* y, size_t dimension) {
    size_t bytes;
    FastCDStatus status = validate_element_extent(dimension, &bytes);

    if (status != FASTCD_SUCCESS) return status;
    if (out == NULL || x == NULL || y == NULL) return FASTCD_NULL_POINTER;
    if (invalid_output_overlap(out, bytes, x, bytes, true)
            || invalid_output_overlap(out, bytes, y, bytes, true)
            || partial_input_overlap(x, bytes, y, bytes)) {
        return FASTCD_OVERLAPPING_BUFFERS;
    }
    return FASTCD_SUCCESS;
}

// Validate one coordinatewise unary operation and its aliasing contract
static FastCDStatus validate_unary_coordinate_args(double* out,
        const double* x, size_t dimension) {
    size_t bytes;
    FastCDStatus status = validate_element_extent(dimension, &bytes);

    if (status != FASTCD_SUCCESS) return status;
    if (out == NULL || x == NULL) return FASTCD_NULL_POINTER;
    if (invalid_output_overlap(out, bytes, x, bytes, true)) {
        return FASTCD_OVERLAPPING_BUFFERS;
    }
    return FASTCD_SUCCESS;
}

// Add two coefficient arrays while allowing exact output aliasing
FastCDStatus fastcd_add_f64(double* out, const double* x, const double* y,
        size_t dimension) {
    FastCDStatus status = validate_binary_coordinate_args(out, x, y,
            dimension);

    if (status != FASTCD_SUCCESS) return status;
    for (size_t i = 0; i < dimension; i++) out[i] = x[i] + y[i];
    return FASTCD_SUCCESS;
}

// Subtract two coefficient arrays while allowing exact output aliasing
FastCDStatus fastcd_sub_f64(double* out, const double* x, const double* y,
        size_t dimension) {
    FastCDStatus status = validate_binary_coordinate_args(out, x, y,
            dimension);

    if (status != FASTCD_SUCCESS) return status;
    for (size_t i = 0; i < dimension; i++) out[i] = x[i] - y[i];
    return FASTCD_SUCCESS;
}

// Negate one coefficient array while allowing exact output aliasing
FastCDStatus fastcd_neg_f64(double* out, const double* x, size_t dimension) {
    FastCDStatus status = validate_unary_coordinate_args(out, x, dimension);

    if (status != FASTCD_SUCCESS) return status;
    for (size_t i = 0; i < dimension; i++) out[i] = -x[i];
    return FASTCD_SUCCESS;
}

// Scale every coefficient by one real value with checked output aliasing
FastCDStatus fastcd_scale_f64(double* out, const double* x, double scalar,
        size_t dimension) {
    FastCDStatus status = validate_unary_coordinate_args(out, x, dimension);

    if (status != FASTCD_SUCCESS) return status;
    for (size_t i = 0; i < dimension; i++) out[i] = scalar * x[i];
    return FASTCD_SUCCESS;
}

/* ---------- Structural operations ---------- */

// Conjugate one element while allowing exact output aliasing
FastCDStatus fastcd_conjugate_f64(double* out, const double* x,
        size_t dimension) {
    FastCDStatus status = validate_unary_coordinate_args(out, x, dimension);

    if (status != FASTCD_SUCCESS) return status;
    out[0] = x[0];
    for (size_t i = 1; i < dimension; i++) out[i] = -x[i];
    return FASTCD_SUCCESS;
}

// Copy the imaginary part while setting the scalar coordinate to zero
FastCDStatus fastcd_imag_part_f64(double* out, const double* x,
        size_t dimension) {
    FastCDStatus status = validate_unary_coordinate_args(out, x, dimension);

    if (status != FASTCD_SUCCESS) return status;
    out[0] = 0.0;
    for (size_t i = 1; i < dimension; i++) out[i] = x[i];
    return FASTCD_SUCCESS;
}

// Compute the Euclidean coefficient inner product of two elements
FastCDStatus fastcd_inner_f64(double* value, const double* x, const double* y,
        size_t dimension) {
    size_t bytes;
    double result;
    FastCDStatus status = validate_element_extent(dimension, &bytes);

    if (status != FASTCD_SUCCESS) return status;
    if (value == NULL || x == NULL || y == NULL) return FASTCD_NULL_POINTER;
    if (ranges_overlap(value, sizeof(double), x, bytes)
            || ranges_overlap(value, sizeof(double), y, bytes)
            || partial_input_overlap(x, bytes, y, bytes)) {
        return FASTCD_OVERLAPPING_BUFFERS;
    }
    result = x[0] * y[0];
    for (size_t i = 1; i < dimension; i++) result += x[i] * y[i];
    *value = result;
    return FASTCD_SUCCESS;
}

// Compute the squared Euclidean coefficient norm with scaled accumulation
FastCDStatus fastcd_norm_squared_f64(double* value, const double* x,
        size_t dimension) {
    size_t bytes;
    FastCDStatus status = validate_element_extent(dimension, &bytes);

    if (status != FASTCD_SUCCESS) return status;
    if (value == NULL || x == NULL) return FASTCD_NULL_POINTER;
    if (ranges_overlap(value, sizeof(double), x, bytes)) {
        return FASTCD_OVERLAPPING_BUFFERS;
    }
    *value = norm_squared_from_scaled(scaled_sum_squares(x, dimension));
    return FASTCD_SUCCESS;
}

// Compute the Euclidean coefficient norm with scaled accumulation
FastCDStatus fastcd_norm_f64(double* value, const double* x,
        size_t dimension) {
    size_t bytes;
    FastCDStatus status = validate_element_extent(dimension, &bytes);

    if (status != FASTCD_SUCCESS) return status;
    if (value == NULL || x == NULL) return FASTCD_NULL_POINTER;
    if (ranges_overlap(value, sizeof(double), x, bytes)) {
        return FASTCD_OVERLAPPING_BUFFERS;
    }
    *value = norm_from_scaled(scaled_sum_squares(x, dimension));
    return FASTCD_SUCCESS;
}

// Normalize one element by its Euclidean coefficient norm
FastCDStatus fastcd_normalize_f64(double* out, const double* x,
        size_t dimension) {
    FastCDScaledSumSquares scaled;
    FastCDStatus status = validate_unary_coordinate_args(out, x, dimension);

    if (status != FASTCD_SUCCESS) return status;
    scaled = scaled_sum_squares(x, dimension);
    if (!scaled.has_nan && !scaled.has_infinity && scaled.scale == 0.0) {
        return FASTCD_ZERO_NORM;
    }
    if (scaled.has_nan || scaled.has_infinity) {
        double norm = norm_from_scaled(scaled);
        for (size_t i = 0; i < dimension; i++) out[i] = x[i] / norm;
    } else {
        double root = sqrt(scaled.sum_squares);
        for (size_t i = 0; i < dimension; i++) {
            out[i] = (x[i] / scaled.scale) / root;
        }
    }
    return FASTCD_SUCCESS;
}

// Compute the canonical conjugate-over-norm-squared reciprocal
FastCDStatus fastcd_reciprocal_f64(double* out, const double* x,
        size_t dimension) {
    FastCDScaledSumSquares scaled;
    FastCDStatus status = validate_unary_coordinate_args(out, x, dimension);

    if (status != FASTCD_SUCCESS) return status;
    scaled = scaled_sum_squares(x, dimension);
    if (!scaled.has_nan && !scaled.has_infinity && scaled.scale == 0.0) {
        return FASTCD_ZERO_NORM;
    }
    if (scaled.has_nan || scaled.has_infinity) {
        double denominator = norm_squared_from_scaled(scaled);
        for (size_t i = 0; i < dimension; i++) {
            double coordinate = i == 0 ? x[i] : -x[i];
            out[i] = coordinate / denominator;
        }
    } else {
        for (size_t i = 0; i < dimension; i++) {
            double coordinate;

            if (scaled.scale < 1.0) {
                coordinate = (x[i] / scaled.scale)
                    / (scaled.scale * scaled.sum_squares);
            } else {
                coordinate = ((x[i] / scaled.scale) / scaled.sum_squares)
                    / scaled.scale;
            }
            out[i] = i == 0 ? coordinate : -coordinate;
        }
    }
    return FASTCD_SUCCESS;
}

/* ---------- Cayley--Dickson multiplication ---------- */

// Validate a multiplication call against an already computed workspace size
static FastCDStatus validate_mul_args(double* out, const double* x,
        const double* y, size_t dimension, size_t required_workspace,
        const FastCDWorkspaceF64* workspace) {
    size_t bytes;
    FastCDStatus status = validate_element_extent(dimension, &bytes);

    if (status != FASTCD_SUCCESS) return status;
    if (out == NULL || x == NULL || y == NULL) return FASTCD_NULL_POINTER;
    if (invalid_output_overlap(out, bytes, x, bytes, false)
            || invalid_output_overlap(out, bytes, y, bytes, false)
            || partial_input_overlap(x, bytes, y, bytes)) {
        return FASTCD_OVERLAPPING_BUFFERS;
    }
    return validate_workspace(workspace, required_workspace, out, bytes,
            x, bytes, y, bytes, NULL, 0);
}

// Set out = xy with an explicitly selected algorithm
FastCDStatus fastcd_mul_with_algorithm_f64(double* out, const double* x,
        const double* y, size_t dimension, FastCDAlgorithm algorithm,
        const FastCDWorkspaceF64* workspace) {
    FastCDAlgorithm effective;
    size_t required;
    FastCDStatus status = mul_workspace_query(dimension, algorithm,
            &required, &effective);

    if (status != FASTCD_SUCCESS) return status;
    status = validate_mul_args(out, x, y, dimension, required, workspace);
    if (status != FASTCD_SUCCESS) return status;
    mul_unchecked(out, x, y, dimension, effective,
            required == 0 ? NULL : workspace->data);
    return FASTCD_SUCCESS;
}

// Multiply two elements through the automatically selected algorithm
FastCDStatus fastcd_mul_f64(double* out, const double* x, const double* y,
        size_t dimension, const FastCDWorkspaceF64* workspace) {
    return fastcd_mul_with_algorithm_f64(out, x, y, dimension,
            FASTCD_ALGORITHM_AUTO, workspace);
}

// Square one element with the specialized linear-time formula
FastCDStatus fastcd_square_f64(double* out, const double* x,
        size_t dimension) {
    double alpha;
    double alpha_squared;
    double imaginary_norm_squared = 0.0;
    double scalar;
    bool finite_input = true;
    FastCDStatus status = validate_unary_coordinate_args(out, x, dimension);

    if (status != FASTCD_SUCCESS) return status;
    alpha = x[0];
    if (!isfinite(alpha)) finite_input = false;
    if (dimension > 1) {
        imaginary_norm_squared = x[1] * x[1];
        if (!isfinite(x[1])) finite_input = false;
        for (size_t i = 2; i < dimension; i++) {
            imaginary_norm_squared += x[i] * x[i];
            if (!isfinite(x[i])) finite_input = false;
        }
    }
    alpha_squared = alpha * alpha;
    if (finite_input && (!isfinite(alpha_squared)
                || !isfinite(imaginary_norm_squared))) {
        scalar = scaled_square_scalar(x, dimension);
    } else {
        scalar = alpha_squared - imaginary_norm_squared;
    }
    for (size_t i = 1; i < dimension; i++) {
        out[i] = twice_product(alpha, x[i]);
    }
    out[0] = scalar;
    return FASTCD_SUCCESS;
}

// Compute a strided batch span in double elements
static FastCDStatus batch_span(size_t dimension, size_t stride, size_t count,
        size_t* span) {
    size_t offset;

    if (count <= 1 || stride == 0) {
        *span = dimension;
        return FASTCD_SUCCESS;
    }
    if (!checked_size_mul(count - 1, stride, &offset)
            || !checked_size_add(offset, dimension, span)) {
        return FASTCD_SIZE_OVERFLOW;
    }
    return FASTCD_SUCCESS;
}

// Return whether any pair of same-index batch input records partially overlaps
static bool batch_records_conflict(const double* x, size_t x_stride,
        const double* y, size_t y_stride, size_t count, size_t element_bytes) {
    for (size_t i = 0; i < count; i++) {
        if (partial_input_overlap(x + i * x_stride, element_bytes,
                    y + i * y_stride, element_bytes)) {
            return true;
        }
    }
    return false;
}

// Multiply a batch with an explicitly selected algorithm
FastCDStatus fastcd_mul_batch_with_algorithm_f64(double* out,
        size_t out_stride, const double* x, size_t x_stride, const double* y,
        size_t y_stride, size_t count, size_t dimension,
        FastCDAlgorithm algorithm, const FastCDWorkspaceF64* workspace) {
    FastCDAlgorithm effective;
    size_t out_span;
    size_t x_span;
    size_t y_span;
    size_t out_bytes;
    size_t x_bytes;
    size_t y_bytes;
    size_t element_bytes;
    size_t required;
    FastCDStatus status;

    if (count == 0) {
        if (!fastcd_valid_dimension(dimension)) {
            return FASTCD_INVALID_DIMENSION;
        }
        return resolve_algorithm(dimension, algorithm, false, &effective);
    }
    status = mul_workspace_query(dimension, algorithm, &required, &effective);
    if (status != FASTCD_SUCCESS) return status;
    if (count >= 2) {
        status = resolve_algorithm(dimension, algorithm, true, &effective);
        if (status != FASTCD_SUCCESS) return status;
    }
    if (out == NULL || x == NULL || y == NULL) return FASTCD_NULL_POINTER;
    if (count > 1 && (out_stride < dimension
                || (x_stride != 0 && x_stride < dimension)
                || (y_stride != 0 && y_stride < dimension))) {
        return FASTCD_INVALID_STRIDE;
    }
    if (count == 1) {
        out_stride = 0;
        x_stride = 0;
        y_stride = 0;
    }
    status = batch_span(dimension, out_stride, count, &out_span);
    if (status != FASTCD_SUCCESS) return status;
    status = batch_span(dimension, x_stride, count, &x_span);
    if (status != FASTCD_SUCCESS) return status;
    status = batch_span(dimension, y_stride, count, &y_span);
    if (status != FASTCD_SUCCESS) return status;
    status = double_array_bytes(out_span, &out_bytes);
    if (status != FASTCD_SUCCESS) return status;
    status = double_array_bytes(x_span, &x_bytes);
    if (status != FASTCD_SUCCESS) return status;
    status = double_array_bytes(y_span, &y_bytes);
    if (status != FASTCD_SUCCESS) return status;
    status = double_array_bytes(dimension, &element_bytes);
    if (status != FASTCD_SUCCESS) return status;
    /* Read-only inputs may share storage, as when one input broadcasts a
     * record of the other or both are windows of one array.  Only a pair of
     * same-index records that partially overlap is rejected. */
    if (ranges_overlap(out, out_bytes, x, x_bytes)
            || ranges_overlap(out, out_bytes, y, y_bytes)
            || (ranges_overlap(x, x_bytes, y, y_bytes)
                && batch_records_conflict(x, x_stride, y, y_stride, count,
                    element_bytes))) {
        return FASTCD_OVERLAPPING_BUFFERS;
    }
    status = validate_workspace(workspace, required, out, out_bytes,
            x, x_bytes, y, y_bytes, NULL, 0);
    if (status != FASTCD_SUCCESS) return status;
    mul_batch_unchecked(out, out_stride, x, x_stride, y, y_stride, count,
            dimension, effective, required == 0 ? NULL : workspace->data);
    return FASTCD_SUCCESS;
}

// Validate once and multiply a strided batch with automatic selection
FastCDStatus fastcd_mul_batch_f64(double* out, size_t out_stride,
        const double* x, size_t x_stride, const double* y, size_t y_stride,
        size_t count, size_t dimension, const FastCDWorkspaceF64* workspace) {
    return fastcd_mul_batch_with_algorithm_f64(out, out_stride, x, x_stride,
            y, y_stride, count, dimension, FASTCD_ALGORITHM_AUTO, workspace);
}

/* ---------- Noncommutative and nonassociative products ---------- */

// Validate one imaginary-cross or commutator operation
static FastCDStatus validate_cross_args(double* out, const double* x,
        const double* y, size_t dimension,
        const FastCDWorkspaceF64* workspace, size_t* required) {
    size_t bytes;
    size_t workspace_bytes;
    FastCDStatus status = validate_element_extent(dimension, &bytes);

    if (status != FASTCD_SUCCESS) return status;
    if (out == NULL || x == NULL || y == NULL) return FASTCD_NULL_POINTER;
    if (ranges_overlap(out, bytes, x, bytes)
            || ranges_overlap(out, bytes, y, bytes)
            || partial_input_overlap(x, bytes, y, bytes)) {
        return FASTCD_OVERLAPPING_BUFFERS;
    }
    *required = cross_workspace_unchecked(dimension);
    status = double_array_bytes(*required, &workspace_bytes);
    if (status != FASTCD_SUCCESS) return status;
    return validate_workspace(workspace, *required, out, bytes, x, bytes,
            y, bytes, NULL, 0);
}

// Compute the generalized imaginary cross product
FastCDStatus fastcd_imag_cross_f64(double* out, const double* x,
        const double* y, size_t dimension,
        const FastCDWorkspaceF64* workspace) {
    size_t required;
    FastCDStatus status = validate_cross_args(out, x, y, dimension, workspace,
            &required);

    if (status != FASTCD_SUCCESS) return status;
    cross_unchecked(out, x, y, dimension,
            required == 0 ? NULL : workspace->data);
    return FASTCD_SUCCESS;
}

// Compute the commutator through its imaginary alternating product
FastCDStatus fastcd_commutator_f64(double* out, const double* x,
        const double* y, size_t dimension,
        const FastCDWorkspaceF64* workspace) {
    size_t required;
    FastCDStatus status = validate_cross_args(out, x, y, dimension, workspace,
            &required);

    if (status != FASTCD_SUCCESS) return status;
    cross_unchecked(out, x, y, dimension,
            required == 0 ? NULL : workspace->data);
    for (size_t i = 1; i < dimension; i++) out[i] += out[i];
    out[0] = 0.0;
    return FASTCD_SUCCESS;
}

// Compute the anticommutator from its symmetric closed form
FastCDStatus fastcd_anticommutator_f64(double* out, const double* x,
        const double* y, size_t dimension) {
    size_t bytes;
    double scalar;
    FastCDStatus status = validate_element_extent(dimension, &bytes);

    if (status != FASTCD_SUCCESS) return status;
    if (out == NULL || x == NULL || y == NULL) return FASTCD_NULL_POINTER;
    if (ranges_overlap(out, bytes, x, bytes)
            || ranges_overlap(out, bytes, y, bytes)
            || partial_input_overlap(x, bytes, y, bytes)) {
        return FASTCD_OVERLAPPING_BUFFERS;
    }
    scalar = x[0] * y[0];
    for (size_t i = 1; i < dimension; i++) scalar -= x[i] * y[i];
    out[0] = scalar + scalar;
    for (size_t i = 1; i < dimension; i++) {
        double vector = x[0] * y[i] + y[0] * x[i];
        out[i] = vector + vector;
    }
    return FASTCD_SUCCESS;
}

// Validate a composite operation with two scratch elements and multiplication
static FastCDStatus validate_composite_args(double* out, const double* x,
        const double* y, const double* z, size_t dimension,
        const FastCDWorkspaceF64* workspace, size_t* required) {
    size_t bytes;
    FastCDStatus status = validate_element_extent(dimension, &bytes);

    if (status != FASTCD_SUCCESS) return status;
    if (out == NULL || x == NULL || y == NULL || z == NULL) {
        return FASTCD_NULL_POINTER;
    }
    if (ranges_overlap(out, bytes, x, bytes)
            || ranges_overlap(out, bytes, y, bytes)
            || ranges_overlap(out, bytes, z, bytes)
            || partial_input_overlap(x, bytes, y, bytes)
            || partial_input_overlap(x, bytes, z, bytes)
            || partial_input_overlap(y, bytes, z, bytes)) {
        return FASTCD_OVERLAPPING_BUFFERS;
    }
    status = composite_workspace_size(dimension, required);
    if (status != FASTCD_SUCCESS) return status;
    return validate_workspace(workspace, *required, out, bytes, x, bytes,
            y, bytes, z, bytes);
}

// Multiply within a validated composite workspace using automatic selection
static void composite_mul(double* restrict out, const double* x,
        const double* y, size_t dimension, double* restrict workspace) {
    FastCDAlgorithm effective = default_algorithm_unchecked(dimension);

    mul_unchecked(out, x, y, dimension, effective, workspace);
}

// Compute the associator (xy)z - x(yz) with automatic multiplication
FastCDStatus fastcd_associator_f64(double* out, const double* x,
        const double* y, const double* z, size_t dimension,
        const FastCDWorkspaceF64* workspace) {
    size_t required;
    double* xy;
    double* yz;
    double* scratch;
    FastCDStatus status = validate_composite_args(out, x, y, z, dimension,
            workspace, &required);

    if (status != FASTCD_SUCCESS) return status;
    xy = workspace->data;
    yz = workspace->data + dimension;
    scratch = workspace->data + 2 * dimension;
    composite_mul(xy, x, y, dimension, scratch);
    composite_mul(yz, y, z, dimension, scratch);
    composite_mul(out, xy, z, dimension, scratch);
    composite_mul(xy, x, yz, dimension, scratch);
    for (size_t i = 0; i < dimension; i++) out[i] -= xy[i];
    return FASTCD_SUCCESS;
}

/* ---------- Indexed basis operations ---------- */

// Return the standard-basis index and sign of a basis product
FastCDStatus fastcd_basis_product(FastCDBasisProduct* product,
        size_t dimension, size_t left_index, size_t right_index) {
    FastCDBasisProduct result;

    if (product == NULL) return FASTCD_NULL_POINTER;
    if (!fastcd_valid_dimension(dimension)) return FASTCD_INVALID_DIMENSION;
    if (left_index >= dimension || right_index >= dimension) {
        return FASTCD_INVALID_BASIS_INDEX;
    }
    result.index = left_index ^ right_index;
    result.sign = basis_sign_unchecked(dimension, left_index, right_index);
    *product = result;
    return FASTCD_SUCCESS;
}

// Apply left multiplication by one standard basis element
FastCDStatus fastcd_basis_lmul_f64(double* out, size_t basis_index,
        const double* x, size_t dimension) {
    size_t bytes;
    FastCDStatus status = validate_element_extent(dimension, &bytes);

    if (status != FASTCD_SUCCESS) return status;
    if (out == NULL || x == NULL) return FASTCD_NULL_POINTER;
    if (basis_index >= dimension) return FASTCD_INVALID_BASIS_INDEX;
    if (ranges_overlap(out, bytes, x, bytes)) return FASTCD_OVERLAPPING_BUFFERS;
    for (size_t i = 0; i < dimension; i++) {
        int sign = basis_sign_unchecked(dimension, basis_index, i);
        out[basis_index ^ i] = sign > 0 ? x[i] : -x[i];
    }
    return FASTCD_SUCCESS;
}

// Apply right multiplication by one standard basis element
FastCDStatus fastcd_basis_rmul_f64(double* out, const double* x,
        size_t basis_index, size_t dimension) {
    size_t bytes;
    FastCDStatus status = validate_element_extent(dimension, &bytes);

    if (status != FASTCD_SUCCESS) return status;
    if (out == NULL || x == NULL) return FASTCD_NULL_POINTER;
    if (basis_index >= dimension) return FASTCD_INVALID_BASIS_INDEX;
    if (ranges_overlap(out, bytes, x, bytes)) return FASTCD_OVERLAPPING_BUFFERS;
    for (size_t i = 0; i < dimension; i++) {
        int sign = basis_sign_unchecked(dimension, i, basis_index);
        out[i ^ basis_index] = sign > 0 ? x[i] : -x[i];
    }
    return FASTCD_SUCCESS;
}

/* ---------- Matrix-free operator commutators ---------- */

// Apply the left multiplication-operator commutator a(bx) - b(ax)
FastCDStatus fastcd_lop_commutator_apply_f64(double* out, const double* a,
        const double* b, const double* x, size_t dimension,
        const FastCDWorkspaceF64* workspace) {
    size_t required;
    double* inner;
    double* second;
    double* scratch;
    FastCDStatus status = validate_composite_args(out, a, b, x, dimension,
            workspace, &required);

    if (status != FASTCD_SUCCESS) return status;
    inner = workspace->data;
    second = workspace->data + dimension;
    scratch = workspace->data + 2 * dimension;
    composite_mul(inner, b, x, dimension, scratch);
    composite_mul(out, a, inner, dimension, scratch);
    composite_mul(inner, a, x, dimension, scratch);
    composite_mul(second, b, inner, dimension, scratch);
    for (size_t i = 0; i < dimension; i++) out[i] -= second[i];
    return FASTCD_SUCCESS;
}

// Apply the right multiplication-operator commutator (xb)a - (xa)b
FastCDStatus fastcd_rop_commutator_apply_f64(double* out, const double* a,
        const double* b, const double* x, size_t dimension,
        const FastCDWorkspaceF64* workspace) {
    size_t required;
    double* inner;
    double* second;
    double* scratch;
    FastCDStatus status = validate_composite_args(out, a, b, x, dimension,
            workspace, &required);

    if (status != FASTCD_SUCCESS) return status;
    inner = workspace->data;
    second = workspace->data + dimension;
    scratch = workspace->data + 2 * dimension;
    composite_mul(inner, x, b, dimension, scratch);
    composite_mul(out, inner, a, dimension, scratch);
    composite_mul(inner, x, a, dimension, scratch);
    composite_mul(second, inner, b, dimension, scratch);
    for (size_t i = 0; i < dimension; i++) out[i] -= second[i];
    return FASTCD_SUCCESS;
}
