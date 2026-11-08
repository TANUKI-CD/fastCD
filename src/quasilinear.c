#include <stdint.h>

#include "fastcd_internal.h"
#include "pair.h"

/*
 * Quasilinear Cayley--Dickson multiplication.
 *
 * For x = alpha + X and y = beta + Y with imaginary X and Y, the product is
 * reconstructed from the alternating product X x Y and linear terms. The
 * alternating product comes from a recursion over complex-coefficient
 * alternating products of half the dimension. The one-call root makes one
 * call to C_(n-1), and above the base cases each call to C_k makes two
 * recursive calls of dimension K/2 plus O(K) work, giving O(N log N) real
 * arithmetic and O(N) scratch.
 *
 * The dimension-4 complex base case is the fused nonrecursive map, and the
 * dimension-8 base case uses four sparse three-dimensional cross products.
 * Larger complex calls use the fused map and two recursive calls. Complex
 * calls through max(8, FASTCD_QL_COMPLEX_LEAF) and complete products through
 * FASTCD_QL_FULL_LEAF run generated straight-line code, and the dimension-8
 * base case applies even with an unrolling cutoff of 4. The default cutoffs
 * come from benchmarks, and docs/benchmark.md identifies the measured source
 * snapshots and arithmetic schedules.
 *
 * Where the two-lane SIMD type exists, a single product above the full leaf
 * runs quasilinear_simd.h, which evaluates the two recursive calls of the top
 * complex call as the two lanes of the paired kernel. The top call does its
 * own O(K) work two coordinates per instruction, with the same arithmetic per
 * coordinate, so the result is bitwise identical to the scalar recursion and
 * to batches.
 */

#ifndef FASTCD_QL_COMPLEX_LEAF
#define FASTCD_QL_COMPLEX_LEAF 8
#endif
#ifndef FASTCD_QL_FULL_LEAF
#define FASTCD_QL_FULL_LEAF 16
#endif
#if FASTCD_QL_COMPLEX_LEAF != 4 && FASTCD_QL_COMPLEX_LEAF != 8 && FASTCD_QL_COMPLEX_LEAF != 16
#error "FASTCD_QL_COMPLEX_LEAF must be 4, 8, or 16"
#endif
#if FASTCD_QL_FULL_LEAF != 8 && FASTCD_QL_FULL_LEAF != 16 && FASTCD_QL_FULL_LEAF != 32
#error "FASTCD_QL_FULL_LEAF must be 8, 16, or 32"
#endif

#define FASTCD_REAL double
#define FASTCD_ZERO 0.0
#define FASTCD_NAME(name) fastcd_##name
#include "generated/quasilinear_kernels.h"
#include "quasilinear_impl.h"
#undef FASTCD_NAME
#undef FASTCD_ZERO
#undef FASTCD_REAL

#if FASTCD_HAVE_PAIR
#define FASTCD_REAL FastCDPair
#define FASTCD_ZERO ((FastCDPair){0.0, 0.0})
#define FASTCD_NAME(name) fastcd_pair_##name
#include "generated/quasilinear_kernels.h"
#include "quasilinear_impl.h"
#undef FASTCD_NAME
#undef FASTCD_ZERO
#undef FASTCD_REAL
#include "quasilinear_simd.h"
#endif

// Evaluate one complete quasilinear product without validation
void fastcd_quasilinear_mul(double* restrict out, const double* x,
        const double* y, size_t dimension, double* restrict workspace) {
#if FASTCD_HAVE_PAIR
    if (dimension > FASTCD_QL_FULL_LEAF) {
        ql_simd_cross_recursive(out, x, y, dimension, workspace);
        ql_simd_reconstruct(out, x, y, dimension);
        return;
    }
#endif
    fastcd_ql_mul_recursive(out, x, y, dimension, workspace);
}

// Compute the alternating product with the quasilinear recursion, without validation
void fastcd_quasilinear_cross(double* restrict out, const double* x,
        const double* y, size_t dimension, double* restrict workspace) {
#if FASTCD_HAVE_PAIR
    if (dimension > FASTCD_QL_FULL_LEAF) {
        ql_simd_cross_recursive(out, x, y, dimension, workspace);
        return;
    }
#endif
    fastcd_ql_cross_recursive(out, x, y, dimension, workspace);
}

// Evaluate a validated strided batch of quasilinear products
void fastcd_quasilinear_mul_batch(double* restrict out, size_t out_stride,
        const double* x, size_t x_stride, const double* y, size_t y_stride,
        size_t count, size_t dimension, double* restrict workspace) {
#if FASTCD_HAVE_PAIR
    if (count >= 2 && dimension >= 16
            && dimension <= FASTCD_PAIR_MAX_DIMENSION) {
        size_t paired = fastcd_pair_batch(out, out_stride, x, x_stride, y,
                y_stride, count, dimension, fastcd_pair_ql_mul_recursive);

        if (paired != count) {
            fastcd_quasilinear_mul(out + paired * out_stride,
                    x + paired * x_stride, y + paired * y_stride, dimension,
                    workspace);
        }
        return;
    }
#endif
#define FASTCD_QL_BATCH(kernel) \
    for (size_t i = 0; i < count; i++) { \
        kernel(out + i * out_stride, x + i * x_stride, y + i * y_stride); \
    } \
    return
    switch (dimension) {
        case 1: FASTCD_QL_BATCH(fastcd_ql_mul_1);
        case 2: FASTCD_QL_BATCH(fastcd_ql_mul_2);
        case 4: FASTCD_QL_BATCH(fastcd_ql_mul_4);
        case 8: FASTCD_QL_BATCH(fastcd_ql_mul_8);
#if FASTCD_QL_FULL_LEAF >= 16
        case 16: FASTCD_QL_BATCH(fastcd_ql_mul_16);
#endif
#if FASTCD_QL_FULL_LEAF >= 32
        case 32: FASTCD_QL_BATCH(fastcd_ql_mul_32);
#endif
        default:
            break;
    }
#undef FASTCD_QL_BATCH
    for (size_t i = 0; i < count; i++) {
        fastcd_quasilinear_mul(out + i * out_stride, x + i * x_stride,
                y + i * y_stride, dimension, workspace);
    }
}

// Return the scratch length of one call to the complex kernel
static size_t complex_workspace(size_t dimension) {
    size_t total = 0;

    while (dimension > 8 && dimension > FASTCD_QL_COMPLEX_LEAF) {
        total += 2 * dimension;
        dimension >>= 1;
    }
    return total;
}

#if FASTCD_HAVE_PAIR
// Return the scratch length of the SIMD top call in doubles: an alignment
// word, six pair planes, and the scratch of the paired recursive calls
static size_t simd_complex_workspace(size_t dimension) {
    if (dimension <= 8 || dimension <= FASTCD_QL_COMPLEX_LEAF) return 0;
    return 1 + 6 * dimension + 2 * complex_workspace(dimension >> 1);
}
#endif

// Return the scratch length used by the quasilinear product and alternating product
size_t fastcd_quasilinear_workspace(size_t dimension) {
    if (dimension <= FASTCD_QL_FULL_LEAF) return 0;
#if FASTCD_HAVE_PAIR
    return simd_complex_workspace(dimension >> 1);
#else
    return complex_workspace(dimension >> 1);
#endif
}
