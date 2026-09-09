#include <stdint.h>
#include <string.h>

#include "fastcd_internal.h"
#include "pair.h"
#include "generated/cariow_cariowa_masks.h"

#if defined(__GNUC__) || defined(__clang__)
#define FASTCD_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define FASTCD_NOINLINE __declspec(noinline)
#else
#define FASTCD_NOINLINE
#endif

/*
 * Cariow--Cariowa multiplication for dimensions 1 through 1024.
 *
 * With H the unnormalized Walsh--Hadamard transform, the unsigned XOR
 * convolution c_k = sum_i x_i y_(i xor k) equals N^-1 H((Hx) o (Hy)).  The
 * basis signs satisfy e_i e_j = sigma_N(i,j) e_(i xor j).  Set epsilon_0 = -1
 * and epsilon_k = 1 otherwise, and M_k = {i : epsilon_k sigma_N(i,i xor k) < 0}.
 * Then z_k = epsilon_k (c_k - 2 sum_{i in M_k} x_i y_(i xor k)).
 * For N >= 2, |M_0| = 1 and |M_k| = N/2 - 1 for k != 0: the N
 * transform-domain products plus the correction products total N(N-1)/2 + 2
 * general multiplications (excluding fixed power-of-two scalings).
 *
 * Through FASTCD_CC_LEAF (64 by default) the correction pattern is baked
 * into generated straight-line code with no runtime sign or mask evaluation
 * and no caller-provided scratch.  Above it the same operations run as
 * loops over generated sign-correction tables (cariow_cariowa_impl.h) with
 * 2N scratch coefficients and the correction terms of each output summed in
 * four interleaved partial sums, because the unrolled form grows with N^2
 * and is about 13 MB of code at dimension 1024.  The algorithm is not
 * recommended above 64: quasilinear is already twice as fast there, and the
 * gathered sign corrections cannot be vectorized within a product, which the
 * benchmark shows at every dimension (see docs/benchmark.md).
 *
 * A single product runs the scalar kernel.  Unlike the direct and
 * quasilinear schedules, the sign corrections cannot share SIMD lanes within
 * one product: adjacent outputs need different subsets of the terms (about
 * one quarter in common), so only batches of two or more products use the
 * paired kernel.  The optional FASTCD_CC_DOUBLING=1 build instead evaluates
 * every product of dimension 16 and above as the four half-dimensional
 * products of one Cayley--Dickson doubling step, two per SIMD lane.  That
 * keeps N(N-1)/2 - N/2 + 8 multiplications but roughly doubles the
 * Walsh--Hadamard additions, so it is a different arithmetic schedule from
 * the paper's fixed-dimension procedure and is off by default.  Single
 * products and batches agree bitwise in either configuration.
 */

#if FASTCD_CC_LEAF != 8 && FASTCD_CC_LEAF != 16 && FASTCD_CC_LEAF != 32 \
        && FASTCD_CC_LEAF != 64
#error "FASTCD_CC_LEAF must be 8, 16, 32, or 64"
#endif
#ifndef FASTCD_CC_DOUBLING
#define FASTCD_CC_DOUBLING 0
#endif
#if FASTCD_CC_DOUBLING && !FASTCD_HAVE_PAIR
#error "FASTCD_CC_DOUBLING requires the two-lane SIMD type"
#endif
#if FASTCD_CC_DOUBLING
static void cc_doubling_mul(double* restrict out, const double* x,
        const double* y, size_t dimension);
#endif
static FASTCD_NOINLINE void cc_table_single(double* restrict out,
        const double* x, const double* y, size_t dimension,
        double* restrict workspace);
static FASTCD_NOINLINE void cc_table_batch(double* restrict out,
        size_t out_stride, const double* x, size_t x_stride, const double* y,
        size_t y_stride, size_t count, size_t dimension,
        double* restrict workspace);

// Expand one row word into the positions of its set bits without branches
static inline size_t cc_expand_word(uint64_t bits, uint8_t* positions) {
    size_t count = 0;

    // Enumerate the set bits of this word of M_k, byte by byte.  The caller
    // adds the word's base index to recover each coefficient index i.
    for (size_t b = 0; b < 8; b++) {
        uint8_t byte = (uint8_t)(bits >> (8 * b));

        memcpy(positions + count, fastcd_cc_byte_positions[b][byte], 8);
        count += fastcd_cc_byte_popcount[byte];
    }
    return count;
}

// Return the sign-correction table of a dimension from 16 through 1024
static const uint64_t* cc_mask_table(size_t dimension) {
    // Row k encodes M_k; basis signs were evaluated by the generator.
    switch (dimension) {
        case 16: return &fastcd_cc_mask_16[0][0];
        case 32: return &fastcd_cc_mask_32[0][0];
        case 64: return &fastcd_cc_mask_64[0][0];
        case 128: return &fastcd_cc_mask_128[0][0];
        case 256: return &fastcd_cc_mask_256[0][0];
        case 512: return &fastcd_cc_mask_512[0][0];
        case 1024: return &fastcd_cc_mask_1024[0][0];
        default: return NULL;
    }
}

#define FASTCD_REAL double
#define FASTCD_ZERO 0.0
#define FASTCD_NAME(name) fastcd_##name
#include "generated/cariow_cariowa_kernels.h"
#include "cariow_cariowa_impl.h"
#undef FASTCD_NAME
#undef FASTCD_ZERO
#undef FASTCD_REAL

#if FASTCD_HAVE_PAIR
#define FASTCD_REAL FastCDPair
#define FASTCD_ZERO ((FastCDPair){0.0, 0.0})
#define FASTCD_NAME(name) fastcd_pair_##name
#include "generated/cariow_cariowa_kernels.h"
#include "cariow_cariowa_impl.h"
#undef FASTCD_NAME
#undef FASTCD_ZERO
#undef FASTCD_REAL
#endif

// Report whether a Cariow--Cariowa kernel exists for a valid dimension
bool fastcd_cc_supported(size_t dimension) {
    return dimension >= 1 && dimension <= FASTCD_CARIOW_CARIOWA_MAX_DIMENSION;
}

// Evaluate one Cariow--Cariowa product after validation
void fastcd_cc_mul(double* restrict out, const double* x, const double* y,
        size_t dimension, double* restrict workspace) {
#if FASTCD_CC_DOUBLING
    if (dimension >= 16 && dimension <= 64) {
        cc_doubling_mul(out, x, y, dimension);
        return;
    }
#endif
    if (dimension > FASTCD_CC_LEAF) {
        cc_table_single(out, x, y, dimension, workspace);
        return;
    }
    switch (dimension) {
        case 1: fastcd_cc_mul_1(out, x, y); return;
        case 2: fastcd_cc_mul_2(out, x, y); return;
        case 4: fastcd_cc_mul_4(out, x, y); return;
        case 8: fastcd_cc_mul_8(out, x, y); return;
        case 16: fastcd_cc_mul_16(out, x, y); return;
        case 32: fastcd_cc_mul_32(out, x, y); return;
        case 64: fastcd_cc_mul_64(out, x, y); return;
        default: return;
    }
}

#if FASTCD_HAVE_PAIR
// Evaluate two independent products in the lanes of the generated SIMD kernel
static void cc_pair_mul(FastCDPair* restrict out, const FastCDPair* x,
        const FastCDPair* y, size_t dimension, FastCDPair* restrict workspace) {
    (void)workspace;
    switch (dimension) {
        case 8: fastcd_pair_cc_mul_8(out, x, y); return;
        case 16: fastcd_pair_cc_mul_16(out, x, y); return;
        case 32: fastcd_pair_cc_mul_32(out, x, y); return;
        case 64: fastcd_pair_cc_mul_64(out, x, y); return;
        default: return;
    }
}

// Evaluate two independent table-driven products in the SIMD lanes
static void cc_pair_table_mul(FastCDPair* restrict out, const FastCDPair* x,
        const FastCDPair* y, size_t dimension, FastCDPair* restrict workspace) {
    fastcd_pair_cc_table_mul(out, x, y, dimension, cc_mask_table(dimension),
            workspace);
}
#endif

/* The table-driven paths live in their own non-inlined functions so that the
 * straight-line dispatch and batch loops below keep the code shape they had
 * before dimensions above FASTCD_CC_LEAF existed. */

// Evaluate one table-driven product above the straight-line dimensions
static FASTCD_NOINLINE void cc_table_single(double* restrict out,
        const double* x, const double* y, size_t dimension,
        double* restrict workspace) {
    fastcd_cc_table_mul(out, x, y, dimension, cc_mask_table(dimension),
            workspace);
}

// Evaluate a validated batch of table-driven products
static FASTCD_NOINLINE void cc_table_batch(double* restrict out,
        size_t out_stride, const double* x, size_t x_stride, const double* y,
        size_t y_stride, size_t count, size_t dimension,
        double* restrict workspace) {
    const uint64_t* masks = cc_mask_table(dimension);

#if FASTCD_HAVE_PAIR
    if (count >= 2 && dimension <= FASTCD_PAIR_MAX_DIMENSION) {
        size_t paired = fastcd_pair_batch(out, out_stride, x, x_stride, y,
                y_stride, count, dimension, cc_pair_table_mul);

        out += paired * out_stride;
        x += paired * x_stride;
        y += paired * y_stride;
        count -= paired;
    }
#endif
    for (size_t i = 0; i < count; i++) {
        fastcd_cc_table_mul(out + i * out_stride, x + i * x_stride,
                y + i * y_stride, dimension, masks, workspace);
    }
}

#if FASTCD_CC_DOUBLING
/*
 * One doubling step (a,b)(c,d) = (ac - conj(d) b, da + b conj(c)) with the
 * four half-dimensional Cariow--Cariowa products evaluated two per lane:
 * lanes (ac, da) in the first paired call and (conj(d) b, b conj(c)) in the
 * second.  The two calls read disjoint storage so that they can overlap in
 * the pipeline.  Storage is 3N pairs, at most 3 KiB at dimension 64.
 */
// Evaluate one product of dimension 16 to 64 through a doubling step
static void cc_doubling_mul(double* restrict out, const double* x,
        const double* y, size_t dimension) {
    FastCDPair storage[3 * 64];
    size_t half = dimension >> 1;
    FastCDPair* left_first = storage;
    FastCDPair* right_first = storage + half;
    FastCDPair* left_second = storage + 2 * half;
    FastCDPair* right_second = storage + 3 * half;
    FastCDPair* first = storage + 4 * half;
    FastCDPair* second = storage + 5 * half;
    const double* a = x;
    const double* b = x + half;
    const double* c = y;
    const double* d = y + half;

    // Optional outer doubling: x = (a,b), y = (c,d).  These are four
    // half-dimensional CC evaluations, not one full-dimensional convolution.
    // Pack scalar coordinates unchanged by conjugation.
    left_first[0] = (FastCDPair){a[0], d[0]};
    right_first[0] = (FastCDPair){c[0], a[0]};
    left_second[0] = (FastCDPair){d[0], b[0]};
    right_second[0] = (FastCDPair){b[0], c[0]};
    // Negate the imaginary coordinates of d and c for their CD conjugates.
    for (size_t i = 1; i < half; i++) {
        left_first[i] = (FastCDPair){a[i], d[i]};
        right_first[i] = (FastCDPair){c[i], a[i]};
        left_second[i] = (FastCDPair){-d[i], b[i]};
        right_second[i] = (FastCDPair){b[i], -c[i]};
    }
    // First call: lanes (ac, da).  Second: lanes (conjugate(d)b, b conjugate(c)).
    cc_pair_mul(first, left_first, right_first, half, NULL);
    cc_pair_mul(second, left_second, right_second, half, NULL);
    // Return (ac - conjugate(d)b, da + b conjugate(c)).
    for (size_t i = 0; i < half; i++) {
        out[i] = first[i][0] - second[i][0];
        out[half + i] = first[i][1] + second[i][1];
    }
}
#endif

// Evaluate a validated strided batch of Cariow--Cariowa products
void fastcd_cc_mul_batch(double* restrict out, size_t out_stride,
        const double* x, size_t x_stride, const double* y, size_t y_stride,
        size_t count, size_t dimension, double* restrict workspace) {
#if FASTCD_CC_DOUBLING
    /* Products follow the single-product schedule so batches match it. */
    if (dimension >= 16 && dimension <= 64) {
        for (size_t i = 0; i < count; i++) {
            cc_doubling_mul(out + i * out_stride, x + i * x_stride,
                    y + i * y_stride, dimension);
        }
        return;
    }
#endif
    if (dimension > FASTCD_CC_LEAF) {
        cc_table_batch(out, out_stride, x, x_stride, y, y_stride, count,
                dimension, workspace);
        return;
    }
    (void)workspace;
#if FASTCD_HAVE_PAIR
#define FASTCD_CC_PAIR_BATCH(n) \
    if (count >= 2) { \
        size_t paired = fastcd_pair_batch(out, out_stride, x, x_stride, y, \
                y_stride, count, n, cc_pair_mul); \
        if (paired == count) return; \
        out += paired * out_stride; \
        x += paired * x_stride; \
        y += paired * y_stride; \
        count = 1; \
    }
#else
#define FASTCD_CC_PAIR_BATCH(n)
#endif
#define FASTCD_CC_BATCH(kernel) \
    for (size_t i = 0; i < count; i++) { \
        kernel(out + i * out_stride, x + i * x_stride, y + i * y_stride); \
    } \
    return
    switch (dimension) {
        case 1: FASTCD_CC_BATCH(fastcd_cc_mul_1);
        case 2: FASTCD_CC_BATCH(fastcd_cc_mul_2);
        case 4: FASTCD_CC_BATCH(fastcd_cc_mul_4);
        case 8: FASTCD_CC_BATCH(fastcd_cc_mul_8);
        case 16:
            FASTCD_CC_PAIR_BATCH(16);
            FASTCD_CC_BATCH(fastcd_cc_mul_16);
        case 32:
            FASTCD_CC_PAIR_BATCH(32);
            FASTCD_CC_BATCH(fastcd_cc_mul_32);
        case 64:
            FASTCD_CC_PAIR_BATCH(64);
            FASTCD_CC_BATCH(fastcd_cc_mul_64);
        default: return;
    }
#undef FASTCD_CC_BATCH
#undef FASTCD_CC_PAIR_BATCH
}
