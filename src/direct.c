#include "fastcd_internal.h"
#include "pair.h"

/*
 * Direct multiplication: the defining Cayley--Dickson doubling recursion
 *
 *     (a,b)(c,d) = (ac - conjugate(d)b, da + b conjugate(c)),
 *
 * evaluated with four half-dimensional products per level and N^2 real
 * multiplications in total.  Products of dimension at most
 * FASTCD_DIRECT_LEAF are generated straight-line code with hardcoded basis
 * signs; above that, one level of the recursion uses N scratch doubles for
 * one conjugated operand and one partial product.
 */

#ifndef FASTCD_DIRECT_LEAF
#define FASTCD_DIRECT_LEAF 16
#endif
#if FASTCD_DIRECT_LEAF != 8 && FASTCD_DIRECT_LEAF != 16 && FASTCD_DIRECT_LEAF != 32
#error "FASTCD_DIRECT_LEAF must be 8, 16, or 32"
#endif

#define FASTCD_REAL double
#define FASTCD_ZERO 0.0
#define FASTCD_NAME(name) fastcd_##name
#include "generated/direct_kernels.h"
#include "direct_impl.h"
#undef FASTCD_NAME
#undef FASTCD_ZERO
#undef FASTCD_REAL

#if FASTCD_HAVE_PAIR
#define FASTCD_REAL FastCDPair
#define FASTCD_ZERO ((FastCDPair){0.0, 0.0})
#define FASTCD_NAME(name) fastcd_pair_##name
#include "generated/direct_kernels.h"
#include "direct_impl.h"
#undef FASTCD_NAME
#undef FASTCD_ZERO
#undef FASTCD_REAL
#endif

// Evaluate one direct product after validation
void fastcd_direct_mul(double* restrict out, const double* x, const double* y,
        size_t dimension, double* restrict workspace) {
    fastcd_direct_recursive(out, x, y, dimension, workspace);
}

#if FASTCD_HAVE_PAIR
// Keep SIMD packing storage out of the scalar batch function's stack frame
static __attribute__((noinline)) void direct_pair_batch(
        double* restrict out, size_t out_stride,
        const double* x, size_t x_stride, const double* y, size_t y_stride,
        size_t count, size_t dimension, double* restrict workspace) {
    size_t paired = fastcd_pair_batch(out, out_stride, x, x_stride, y,
            y_stride, count, dimension, fastcd_pair_direct_recursive);

    if (paired != count) {
        fastcd_direct_mul(out + paired * out_stride, x + paired * x_stride,
                y + paired * y_stride, dimension, workspace);
    }
}
#endif

// Keep the scalar loop's register allocation independent of SIMD dispatch
#if FASTCD_HAVE_PAIR
__attribute__((noinline))
#endif
static void direct_scalar_batch(double* restrict out, size_t out_stride,
        const double* x, size_t x_stride, const double* y, size_t y_stride,
        size_t count, size_t dimension, double* restrict workspace) {
#define FASTCD_DIRECT_BATCH(kernel) \
    for (size_t i = 0; i < count; i++) { \
        kernel(out + i * out_stride, x + i * x_stride, y + i * y_stride); \
    } \
    return
    switch (dimension) {
        case 1: FASTCD_DIRECT_BATCH(fastcd_direct_mul_1);
        case 2: FASTCD_DIRECT_BATCH(fastcd_direct_mul_2);
        case 4: FASTCD_DIRECT_BATCH(fastcd_direct_mul_4);
        case 8: FASTCD_DIRECT_BATCH(fastcd_direct_mul_8);
#if FASTCD_DIRECT_LEAF >= 16
        case 16: FASTCD_DIRECT_BATCH(fastcd_direct_mul_16);
#endif
#if FASTCD_DIRECT_LEAF >= 32
        case 32: FASTCD_DIRECT_BATCH(fastcd_direct_mul_32);
#endif
        default: break;
    }
#undef FASTCD_DIRECT_BATCH
    for (size_t i = 0; i < count; i++) {
        fastcd_direct_recursive(out + i * out_stride, x + i * x_stride,
                y + i * y_stride, dimension, workspace);
    }
}

// Evaluate a validated strided batch of direct products
void fastcd_direct_mul_batch(double* restrict out, size_t out_stride,
        const double* x, size_t x_stride, const double* y, size_t y_stride,
        size_t count, size_t dimension, double* restrict workspace) {
#if FASTCD_HAVE_PAIR
    if (count >= 2 && dimension >= 64
            && dimension <= FASTCD_PAIR_MAX_DIMENSION) {
        direct_pair_batch(out, out_stride, x, x_stride, y, y_stride, count,
                dimension, workspace);
        return;
    }
#endif
    direct_scalar_batch(out, out_stride, x, x_stride, y, y_stride, count,
            dimension, workspace);
}

// Return the scratch length used by the direct product
size_t fastcd_direct_workspace(size_t dimension) {
    size_t total = 0;

    while (dimension > FASTCD_DIRECT_LEAF) {
        total += dimension;
        dimension >>= 1;
    }
    return total;
}
