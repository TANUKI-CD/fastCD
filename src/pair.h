#ifndef FASTCD_PAIR_H
#define FASTCD_PAIR_H

#include <stddef.h>

/* Two independent products share each SIMD instruction.  Other targets use
 * the same scalar kernels.  No extra alignment is required of caller data. */
#if !defined(FASTCD_DISABLE_SIMD) && (defined(__GNUC__) || defined(__clang__)) \
        && (defined(__SSE2__) || defined(__aarch64__))
#define FASTCD_HAVE_PAIR 1
typedef double FastCDPair __attribute__((vector_size(16)));

typedef void (*FastCDPairKernel)(FastCDPair* restrict, const FastCDPair*,
        const FastCDPair*, size_t, FastCDPair* restrict);

/* The packing storage is a fixed array of 5 * FASTCD_PAIR_MAX_DIMENSION pairs
 * (40 KiB), which keeps batches safe on small thread stacks without relying
 * on variable-length arrays; pairing stopped paying off near dimension 1024
 * in measurements anyway.  Above this limit the ordinary recursive kernel
 * uses caller-owned scratch. */
#define FASTCD_PAIR_MAX_DIMENSION ((size_t)512)

/* The helper must be inlined into each batch function: with a fixed 40 KiB
 * array the compiler otherwise outlines it and calls the kernel through a
 * generic entry per pair, which measured 30% slower at dimension 16. */
// Pack and evaluate the even prefix of a validated batch at a bounded dimension
static inline __attribute__((always_inline)) size_t fastcd_pair_batch(
        double* restrict out, size_t out_stride,
        const double* x, size_t x_stride, const double* y, size_t y_stride,
        size_t count, size_t dimension, FastCDPairKernel kernel) {
    FastCDPair storage[5 * FASTCD_PAIR_MAX_DIMENSION];
    FastCDPair* a = storage;
    FastCDPair* b = storage + dimension;
    FastCDPair* c = storage + 2 * dimension;
    FastCDPair* scratch = storage + 3 * dimension;
    size_t paired = count & ~(size_t)1;

    for (size_t i = 0; i < paired; i += 2) {
        for (size_t k = 0; k < dimension; k++) {
            a[k] = (FastCDPair){x[i * x_stride + k],
                x[(i + 1) * x_stride + k]};
            b[k] = (FastCDPair){y[i * y_stride + k],
                y[(i + 1) * y_stride + k]};
        }
        kernel(c, a, b, dimension, scratch);
        for (size_t k = 0; k < dimension; k++) {
            out[i * out_stride + k] = c[k][0];
            out[(i + 1) * out_stride + k] = c[k][1];
        }
    }
    return paired;
}
#else
#define FASTCD_HAVE_PAIR 0
/* No pairing: batches use the scalar kernels and the single-product table. */
#define FASTCD_PAIR_MAX_DIMENSION ((size_t)0)
#endif

#endif
