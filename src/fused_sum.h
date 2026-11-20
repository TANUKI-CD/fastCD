#ifndef FASTCD_FUSED_SUM_H
#define FASTCD_FUSED_SUM_H

#include <stdbool.h>
#include <stddef.h>

/*
 * Private interface of the fused sums of products. A fused sum accumulates
 * every bilinear product of a multiplication algorithm over the elements and
 * applies the algorithm's output additions once, to the accumulated sums.
 * Nothing in this header is part of the installed C API.
 */

// Elements accumulated before the accumulators are next written to memory
#define FASTCD_FUSED_CHUNK ((size_t)64)

/* Doubles between the rows of the operand block. The eight spare doubles keep
 * rows that are a power of two apart from sharing a cache set. */
#define FASTCD_FUSED_PITCH (FASTCD_FUSED_CHUNK + 8)

/* One sum, already validated. The operands are either records, element p of x
 * at x + p * x_stride, or planes, coordinate i of element p at
 * x[i * pitch + p]. A coordinate i of a record operand is read as
 * (value - shift[i]) * scale[i]; each of the four arrays may be NULL. */
typedef struct FastCDSumJob {
    const double* x;
    const double* y;
    size_t x_stride;
    size_t y_stride;
    size_t pitch;
    size_t count;
    size_t dimension;
    bool planes;
    bool quasilinear;
    const double* x_shift;
    const double* x_scale;
    const double* y_shift;
    const double* y_scale;
} FastCDSumJob;

// Eight-double accumulators of the complex kernel of dimension K
static inline size_t fastcd_sum_node_vectors(size_t K) {
    if (K <= 2) return 0;
    if (K == 4) return 9;
    if (K == 8) return 21;
    return 3 + 6 * (K / 2) + 2 * fastcd_sum_node_vectors(K / 2);
}

// Eight-double accumulators of a whole sum of dimension N
static inline size_t fastcd_sum_vectors(size_t N) {
    return N + fastcd_sum_node_vectors(N / 2);
}

/* Doubles of scratch used by a kernel: the accumulators, one block of 2N
 * operand rows, and 3N eight-double temporaries for the output additions. */
static inline size_t fastcd_sum_kernel_doubles(size_t N) {
    return 8 * fastcd_sum_vectors(N) + 2 * N * FASTCD_FUSED_PITCH + 24 * N;
}

// Sign s with e_i e_j = s e_(i xor j) in the algebra of dimension n
static inline int fastcd_sum_sign(size_t n, size_t i, size_t j) {
    int s = 1;

    while (n > 1) {
        const size_t half = n / 2;
        if (i < half && j < half) {
            // sigma_N(i, j) = sigma_M(i, j)
        } else if (i < half) {
            const size_t t = j - half;
            j = i;
            i = t;
        } else if (j < half) {
            i -= half;
            if (j != 0) s = -s;
        } else {
            const size_t t = j - half;
            j = i - half;
            i = t;
            // sigma_N(M + i, M + j) = -epsilon(j) sigma_M(j, i)
            if (i == 0) s = -s;
        }
        n = half;
    }
    return s;
}

/* The kernels. Each computes out[0..N) from a job; work holds
 * fastcd_sum_kernel_doubles(N) doubles and is 64-byte aligned. All kernels
 * perform the same operations in the same order. The x86 kernels, and the
 * portable kernel on a target with a fused multiply-add instruction, return
 * the same bits. */
void fastcd_fused_sum_portable(double* restrict out, const FastCDSumJob* job,
        double* restrict work);
#if defined(FASTCD_SUM_SIMD)
void fastcd_fused_sum_avx2(double* restrict out, const FastCDSumJob* job,
        double* restrict work);
void fastcd_fused_sum_avx512(double* restrict out, const FastCDSumJob* job,
        double* restrict work);
#endif

#endif
