#include <immintrin.h>

#include "fused_sum.h"

/*
 * The AVX2 kernel of the fused sums: two 256-bit registers per vector. This
 * file is compiled with -mavx2 -mfma and is called only on processors that
 * report both features.
 */

typedef struct FastCDSumPair {
    __m256d lo;
    __m256d hi;
} FastCDSumPair;

// Build a vector from its two halves
static inline FastCDSumPair avx2_make(__m256d lo, __m256d hi) {
    FastCDSumPair r;
    r.lo = lo;
    r.hi = hi;
    return r;
}

#define FS_V FastCDSumPair
#define FS_NAME(name) avx2_##name
#define FS_PUBLIC fastcd_fused_sum_avx2
#define FS_ZERO avx2_make(_mm256_setzero_pd(), _mm256_setzero_pd())
#define FS_LOAD(p) avx2_make(_mm256_loadu_pd(p), _mm256_loadu_pd((p) + 4))
#define FS_STORE(p, v) avx2_store(p, v)
#define FS_SET1(s) avx2_make(_mm256_set1_pd(s), _mm256_set1_pd(s))
#define FS_ADD(a, b) avx2_add(a, b)
#define FS_SUB(a, b) avx2_sub(a, b)
#define FS_MUL(a, b) avx2_mul(a, b)
#define FS_FMA(a, b, c) avx2_fma(a, b, c)
#define FS_FNMA(a, b, c) avx2_fnma(a, b, c)
#define FS_LANES(v, d) avx2_store(d, v)
#define FS_TRANSPOSE 1

// Store eight doubles
static inline void avx2_store(double* p, FastCDSumPair v) {
    _mm256_storeu_pd(p, v.lo);
    _mm256_storeu_pd(p + 4, v.hi);
}

// Add two vectors
static inline FastCDSumPair avx2_add(FastCDSumPair a, FastCDSumPair b) {
    return avx2_make(_mm256_add_pd(a.lo, b.lo), _mm256_add_pd(a.hi, b.hi));
}

// Subtract two vectors
static inline FastCDSumPair avx2_sub(FastCDSumPair a, FastCDSumPair b) {
    return avx2_make(_mm256_sub_pd(a.lo, b.lo), _mm256_sub_pd(a.hi, b.hi));
}

// Multiply two vectors
static inline FastCDSumPair avx2_mul(FastCDSumPair a, FastCDSumPair b) {
    return avx2_make(_mm256_mul_pd(a.lo, b.lo), _mm256_mul_pd(a.hi, b.hi));
}

// Return a b + c with one rounding
static inline FastCDSumPair avx2_fma(FastCDSumPair a, FastCDSumPair b,
        FastCDSumPair c) {
    return avx2_make(_mm256_fmadd_pd(a.lo, b.lo, c.lo),
            _mm256_fmadd_pd(a.hi, b.hi, c.hi));
}

// Return c - a b with one rounding
static inline FastCDSumPair avx2_fnma(FastCDSumPair a, FastCDSumPair b,
        FastCDSumPair c) {
    return avx2_make(_mm256_fnmadd_pd(a.lo, b.lo, c.lo),
            _mm256_fnmadd_pd(a.hi, b.hi, c.hi));
}

// Transpose coordinates c..c+3 of four records into four registers
static inline void avx2_transpose4(__m256d* o, const double* r, size_t stride,
        size_t c) {
    const __m256d r0 = _mm256_loadu_pd(r + c);
    const __m256d r1 = _mm256_loadu_pd(r + stride + c);
    const __m256d r2 = _mm256_loadu_pd(r + 2 * stride + c);
    const __m256d r3 = _mm256_loadu_pd(r + 3 * stride + c);
    const __m256d lo01 = _mm256_unpacklo_pd(r0, r1), hi01 = _mm256_unpackhi_pd(r0, r1);
    const __m256d lo23 = _mm256_unpacklo_pd(r2, r3), hi23 = _mm256_unpackhi_pd(r2, r3);

    o[0] = _mm256_permute2f128_pd(lo01, lo23, 0x20);
    o[1] = _mm256_permute2f128_pd(hi01, hi23, 0x20);
    o[2] = _mm256_permute2f128_pd(lo01, lo23, 0x31);
    o[3] = _mm256_permute2f128_pd(hi01, hi23, 0x31);
}

/* Transpose coordinates c..c+3 of eight records, stride apart: o[k] receives
 * coordinate c + k with record l in lane l. */
static inline void avx2_transpose(FastCDSumPair* o, const double* r,
        size_t stride, size_t c) {
    __m256d lo[4];
    __m256d hi[4];

    avx2_transpose4(lo, r, stride, c);
    avx2_transpose4(hi, r + 4 * stride, stride, c);
    for (int k = 0; k < 4; k++) o[k] = avx2_make(lo[k], hi[k]);
}

#include "fused_sum_impl.h"
