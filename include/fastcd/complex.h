#ifndef FASTCD_COMPLEX_H
#define FASTCD_COMPLEX_H
#include "arithmetic.h"
#include "fastcd.h"
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Standard CD arithmetic over complex coefficients. The portable complex128
 * coefficient holds two binary64 components. The CD involution fixes the
 * coefficients and negates only the imaginary CD coordinates.
 *
 * Dimensions and batch strides count complex coefficients, and workspace
 * lengths count doubles, as in FastCDWorkspaceF64. All buffers are
 * caller-owned, and arithmetic allocates no memory. An initialized workspace
 * may be reused for sequential calls, and concurrent calls need separate
 * writable storage.
 *
 * Coordinatewise operations, reciprocal and square accept an exact out == x
 * alias, and addition and subtraction also accept out == y. Products,
 * reductions, basis operations and composites need a disjoint output. Inputs
 * may be the same complete record, and other input overlaps are rejected with
 * FASTCD_OVERLAPPING_BUFFERS. Active workspace must be disjoint from every
 * operand and the output. Reported errors leave the output unchanged. Record
 * lengths and alignment are preconditions.
 */
typedef struct FastCDComplexF64 {
    double real, imag;
} FastCDComplexF64;

// Query scratch for one operation, returning its length in doubles
FASTCD_API FastCDStatus fastcd_workspace_size_c128(
        size_t dimension, FastCDOperation operation, size_t* length);
// Apply an operation to one record; reductions write one complex coefficient
FASTCD_API FastCDStatus fastcd_apply_c128(FastCDComplexF64* out,
        const FastCDComplexF64* x, const FastCDComplexF64* y, const FastCDComplexF64* z,
        size_t dimension, FastCDOperation operation, size_t basis_index,
        FastCDAlgorithm algorithm, const FastCDWorkspaceF64* workspace);

// Multiply using AUTO, DIRECT, or QUASILINEAR; other algorithms are rejected
FASTCD_API FastCDStatus fastcd_mul_with_algorithm_c128(FastCDComplexF64* out,
        const FastCDComplexF64* x, const FastCDComplexF64* y, size_t dimension,
        FastCDAlgorithm algorithm, const FastCDWorkspaceF64* workspace);

// Multiply using the complex default, independently of the real AUTO tables
FASTCD_API FastCDStatus fastcd_mul_c128(FastCDComplexF64* out,
        const FastCDComplexF64* x, const FastCDComplexF64* y, size_t dimension,
        const FastCDWorkspaceF64* workspace);

// Multiply count records; zero input strides broadcast one complete record
FASTCD_API FastCDStatus fastcd_mul_batch_c128(FastCDComplexF64* out, size_t out_stride,
        const FastCDComplexF64* x, size_t x_stride, const FastCDComplexF64* y,
        size_t y_stride, size_t count, size_t dimension, FastCDAlgorithm algorithm,
        const FastCDWorkspaceF64* workspace);

// Return Q(x)=sum_i x_i^2, without complex conjugation or a Hermitian absolute value
FASTCD_API FastCDStatus fastcd_quadratic_norm_c128(
        FastCDComplexF64* value, const FastCDComplexF64* x, size_t dimension);
#ifdef __cplusplus
}
#endif
#endif
