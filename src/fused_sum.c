#include <math.h>
#include <stdint.h>

#include "fastcd_internal.h"
#include "fused_sum.h"

/*
 * The portable kernel of the fused sums and the selection of a kernel. The
 * portable vector is eight doubles. Where the target has a fused multiply-add
 * instruction (the compiler then defines FP_FAST_FMA), the kernel uses fma()
 * and returns the same bits as the x86 kernels. Elsewhere fma() is a library
 * call many times slower than the arithmetic it replaces, so the kernel
 * rounds the product and the addition separately; its results then differ
 * from those of the other kernels in the last bits.
 */

#if defined(FP_FAST_FMA) || defined(FASTCD_SUM_FORCE_FMA)
#  define PORTABLE_FMA(a, b, c) fma(a, b, c)
#  define PORTABLE_FNMA(a, b, c) fma(-(a), b, c)
#else
#  define PORTABLE_FMA(a, b, c) ((a) * (b) + (c))
#  define PORTABLE_FNMA(a, b, c) ((c) - (a) * (b))
#endif

typedef struct FastCDSumVector {
    double lane[8];
} FastCDSumVector;

// Return the zero vector
static inline FastCDSumVector portable_zero(void) {
    FastCDSumVector r;
    for (int k = 0; k < 8; k++) r.lane[k] = 0.0;
    return r;
}

// Load eight doubles
static inline FastCDSumVector portable_load(const double* p) {
    FastCDSumVector r;
    for (int k = 0; k < 8; k++) r.lane[k] = p[k];
    return r;
}

// Store eight doubles
static inline void portable_store(double* p, FastCDSumVector v) {
    for (int k = 0; k < 8; k++) p[k] = v.lane[k];
}

// Return the vector with every lane equal to s
static inline FastCDSumVector portable_set1(double s) {
    FastCDSumVector r;
    for (int k = 0; k < 8; k++) r.lane[k] = s;
    return r;
}

// Add two vectors
static inline FastCDSumVector portable_add(FastCDSumVector a, FastCDSumVector b) {
    FastCDSumVector r;
    for (int k = 0; k < 8; k++) r.lane[k] = a.lane[k] + b.lane[k];
    return r;
}

// Subtract two vectors
static inline FastCDSumVector portable_sub(FastCDSumVector a, FastCDSumVector b) {
    FastCDSumVector r;
    for (int k = 0; k < 8; k++) r.lane[k] = a.lane[k] - b.lane[k];
    return r;
}

// Multiply two vectors
static inline FastCDSumVector portable_mul(FastCDSumVector a, FastCDSumVector b) {
    FastCDSumVector r;
    for (int k = 0; k < 8; k++) r.lane[k] = a.lane[k] * b.lane[k];
    return r;
}

// Return a b + c, with one rounding where the target fuses the two
static inline FastCDSumVector portable_fma(FastCDSumVector a, FastCDSumVector b,
        FastCDSumVector c) {
    FastCDSumVector r;
    for (int k = 0; k < 8; k++) r.lane[k] = PORTABLE_FMA(a.lane[k], b.lane[k], c.lane[k]);
    return r;
}

// Return c - a b, with one rounding where the target fuses the two
static inline FastCDSumVector portable_fnma(FastCDSumVector a, FastCDSumVector b,
        FastCDSumVector c) {
    FastCDSumVector r;
    for (int k = 0; k < 8; k++) r.lane[k] = PORTABLE_FNMA(a.lane[k], b.lane[k], c.lane[k]);
    return r;
}

#define FS_V FastCDSumVector
#define FS_NAME(name) portable_##name
#define FS_PUBLIC fastcd_fused_sum_portable
#define FS_ZERO portable_zero()
#define FS_LOAD(p) portable_load(p)
#define FS_STORE(p, v) portable_store(p, v)
#define FS_SET1(s) portable_set1(s)
#define FS_ADD(a, b) portable_add(a, b)
#define FS_SUB(a, b) portable_sub(a, b)
#define FS_MUL(a, b) portable_mul(a, b)
#define FS_FMA(a, b, c) portable_fma(a, b, c)
#define FS_FNMA(a, b, c) portable_fnma(a, b, c)
#define FS_LANES(v, d) portable_store(d, v)

#include "fused_sum_impl.h"

/* ---------- Kernel selection ---------- */

int fastcd_fused_sum_kernels(void) {
#if defined(FASTCD_SUM_SIMD)
    if (__builtin_cpu_supports("avx512f")) return 3;
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) return 2;
#endif
    return 1;
}

size_t fastcd_fused_sum_workspace(size_t dimension) {
    // Seven doubles of slack align the kernel's storage to 64 bytes
    return fastcd_sum_kernel_doubles(dimension) + 7;
}

void fastcd_fused_sum(double* restrict out, const FastCDSumJob* job,
        double* restrict workspace) {
    const uintptr_t address = (uintptr_t)workspace;
    double* const work = workspace
        + ((64 - address % 64) % 64) / sizeof(double);
#if defined(FASTCD_SUM_SIMD)
    const int kernel = fastcd_fused_sum_kernels();

    if (kernel == 3) {
        fastcd_fused_sum_avx512(out, job, work);
        return;
    }
    if (kernel == 2) {
        fastcd_fused_sum_avx2(out, job, work);
        return;
    }
#endif
    fastcd_fused_sum_portable(out, job, work);
}
