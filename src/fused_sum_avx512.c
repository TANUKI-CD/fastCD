#include <immintrin.h>

#include "fused_sum.h"

/*
 * The AVX-512 kernel of the fused sums: one 512-bit register per vector.
 * This file is compiled with -mavx512f and is called only on processors that
 * report the feature.
 */

#define FS_V __m512d
#define FS_NAME(name) avx512_##name
#define FS_PUBLIC fastcd_fused_sum_avx512
#define FS_ZERO _mm512_setzero_pd()
#define FS_LOAD(p) _mm512_loadu_pd(p)
#define FS_STORE(p, v) _mm512_storeu_pd(p, v)
#define FS_SET1(s) _mm512_set1_pd(s)
#define FS_ADD(a, b) _mm512_add_pd(a, b)
#define FS_SUB(a, b) _mm512_sub_pd(a, b)
#define FS_MUL(a, b) _mm512_mul_pd(a, b)
#define FS_FMA(a, b, c) _mm512_fmadd_pd(a, b, c)
#define FS_FNMA(a, b, c) _mm512_fnmadd_pd(a, b, c)
#define FS_LANES(v, d) _mm512_storeu_pd(d, v)
#define FS_TRANSPOSE 1

// Load coordinates c..c+3 of the records p and q into one register
static inline __m512d avx512_pair(const double* p, const double* q) {
    return _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_loadu_pd(p)),
            _mm256_loadu_pd(q), 1);
}

/* Transpose coordinates c..c+3 of eight records, stride apart: o[k] receives
 * coordinate c + k with record l in lane l. */
static inline void avx512_transpose(__m512d* o, const double* r, size_t stride,
        size_t c) {
    const __m512d a = avx512_pair(r + c, r + 2 * stride + c);
    const __m512d b = avx512_pair(r + stride + c, r + 3 * stride + c);
    const __m512d e = avx512_pair(r + 4 * stride + c, r + 6 * stride + c);
    const __m512d f = avx512_pair(r + 5 * stride + c, r + 7 * stride + c);
    const __m512d ab_lo = _mm512_unpacklo_pd(a, b), ab_hi = _mm512_unpackhi_pd(a, b);
    const __m512d ef_lo = _mm512_unpacklo_pd(e, f), ef_hi = _mm512_unpackhi_pd(e, f);

    o[0] = _mm512_shuffle_f64x2(ab_lo, ef_lo, 0x88);
    o[1] = _mm512_shuffle_f64x2(ab_hi, ef_hi, 0x88);
    o[2] = _mm512_shuffle_f64x2(ab_lo, ef_lo, 0xdd);
    o[3] = _mm512_shuffle_f64x2(ab_hi, ef_hi, 0xdd);
}

#include "fused_sum_impl.h"
