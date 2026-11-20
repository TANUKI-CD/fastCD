/* Fused sums of products and the covariance. Operands that are small
 * multiples of 1/8 make every product, partial sum and halving exact, so the
 * sums are compared for equality with the summed products of the reference
 * recursion, for every kernel, layout and algorithm. */
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fastcd/fastcd.h"
#include "fastcd_internal.h"
#include "fused_sum.h"
#include "reference_cd.h"

#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #c); return 0; \
} } while (0)

#define GUARD 16

// Draw a multiple of 1/8 in [-4, 4) from a linear congruential generator
static double dyadic(uint64_t* state) {
    *state = *state * UINT64_C(6364136223846793005) + 1;
    return (double)((int)(*state >> 58) - 32) * 0.125;
}

// Draw a double in (-1, 1) with a full mantissa
static double uniform(uint64_t* state) {
    *state = *state * UINT64_C(6364136223846793005) + 1;
    return (double)(int64_t)(*state >> 11) / 4503599627370496.0 - 1.0;
}

// Allocate a zeroed array of doubles or stop the test
static double* make(size_t length) {
    double* p = calloc(length ? length : 1, sizeof(double));
    if (p == NULL) {
        fprintf(stderr, "out of memory\n");
        exit(2);
    }
    return p;
}

// Sum the reference products of count records, in long double
static void reference_sum(double* out, const double* x, size_t x_stride,
        const double* y, size_t y_stride, size_t count, size_t n) {
    long double* total = calloc(n, sizeof(long double));
    double* product = make(n);

    for (size_t p = 0; p < count; p++) {
        reference_cd_mul_signed_f64(product, x + p * x_stride, y + p * y_stride, n);
        for (size_t i = 0; i < n; i++) total[i] += product[i];
    }
    for (size_t i = 0; i < n; i++) out[i] = (double)total[i];
    free(total);
    free(product);
}

// A workspace of the queried length inside guard words, at a chosen offset
typedef struct {
    double* base;
    FastCDWorkspaceF64 w;
    size_t length;
} Guarded;

static Guarded guarded(size_t length, size_t offset) {
    Guarded g;
    g.base = make(length + 2 * GUARD + 8);
    g.length = length;
    for (size_t i = 0; i < length + 2 * GUARD + 8; i++) g.base[i] = -777.0;
    g.w.data = g.base + GUARD + offset;
    g.w.length = length;
    return g;
}

// Return whether the words around a guarded workspace are untouched
static int guards_intact(const Guarded* g) {
    for (const double* p = g->base; p < g->w.data; p++) if (*p != -777.0) return 0;
    for (const double* p = g->w.data + g->length; p < g->base + g->length
            + 2 * GUARD + 8; p++) if (*p != -777.0) return 0;
    return 1;
}

// Transpose count records with the given stride into planes pitch apart
static void to_planes(double* planes, size_t pitch, const double* records,
        size_t stride, size_t count, size_t n) {
    for (size_t p = 0; p < count; p++)
        for (size_t i = 0; i < n; i++) planes[i * pitch + p] = records[p * stride + i];
}

static const FastCDAlgorithm sum_algorithms[3] = {FASTCD_ALGORITHM_AUTO,
    FASTCD_ALGORITHM_DIRECT, FASTCD_ALGORITHM_QUASILINEAR};

/* Exact operands: every algorithm, both layouts and every kernel must return
 * exactly the summed reference products. */
static int exact_cases(void) {
    static const size_t counts[] = {0, 1, 2, 7, 8, 9, 63, 64, 65, 129, 200};
    uint64_t state = 2026;

    for (size_t n = 1; n <= 256; n *= 2) {
        size_t required;
        CHECK(fastcd_mul_sum_workspace_size(n, &required) == FASTCD_SUCCESS);
        CHECK(required == fastcd_fused_sum_workspace(n));
        for (size_t ci = 0; ci < sizeof counts / sizeof counts[0]; ci++) {
            const size_t count = counts[ci];
            const size_t xs = n + (ci % 3), ys = n + 2 * (ci % 2), pitch = count + ci;
            double* x = make(count * xs + n);
            double* y = make(count * ys + n);
            double* px = make(n * pitch + 8);
            double* py = make(n * pitch + 8);
            double* want = make(n);
            double* got = make(n);
            Guarded g = guarded(required, ci % 8);

            for (size_t i = 0; i < count * xs + n; i++) x[i] = dyadic(&state);
            for (size_t i = 0; i < count * ys + n; i++) y[i] = dyadic(&state);
            reference_sum(want, x, xs, y, ys, count, n);
            to_planes(px, pitch, x, xs, count, n);
            to_planes(py, pitch, y, ys, count, n);
            for (size_t a = 0; a < 3; a++) {
                memset(got, 0xff, n * sizeof(double));
                CHECK(fastcd_mul_sum_with_algorithm_f64(got, x, xs, y, ys, count,
                            n, NULL, sum_algorithms[a], &g.w) == FASTCD_SUCCESS);
                for (size_t i = 0; i < n; i++) CHECK(got[i] == want[i]);
                memset(got, 0xff, n * sizeof(double));
                CHECK(fastcd_mul_sum_planes_with_algorithm_f64(got, px, py, pitch,
                            count, n, sum_algorithms[a], &g.w) == FASTCD_SUCCESS);
                for (size_t i = 0; i < n; i++) CHECK(got[i] == want[i]);
                CHECK(guards_intact(&g));
            }
            free(x); free(y); free(px); free(py); free(want); free(got);
            free(g.base);
        }
    }
    return 1;
}

typedef void (*Kernel)(double* restrict, const FastCDSumJob*, double* restrict);

/*
 * The kernel source instantiated with a vector whose multiply-add rounds the
 * product and the sum separately. This is the arithmetic of the portable
 * kernel on targets without a fused multiply-add instruction; the test's
 * own portable kernel uses fma() wherever the x86 kernels are built.
 */
typedef struct { double lane[8]; } Plain;
static Plain plain_zero(void) { Plain r; memset(&r, 0, sizeof r); return r; }
static Plain plain_load(const double* p) { Plain r; memcpy(r.lane, p, sizeof r.lane); return r; }
static void plain_store(double* p, Plain v) { memcpy(p, v.lane, sizeof v.lane); }
static Plain plain_set1(double s) { Plain r; for (int k = 0; k < 8; k++) r.lane[k] = s; return r; }
static Plain plain_add(Plain a, Plain b) { for (int k = 0; k < 8; k++) a.lane[k] += b.lane[k]; return a; }
static Plain plain_sub(Plain a, Plain b) { for (int k = 0; k < 8; k++) a.lane[k] -= b.lane[k]; return a; }
static Plain plain_mul(Plain a, Plain b) { for (int k = 0; k < 8; k++) a.lane[k] *= b.lane[k]; return a; }
static Plain plain_fma(Plain a, Plain b, Plain c) {
    for (int k = 0; k < 8; k++) c.lane[k] = a.lane[k] * b.lane[k] + c.lane[k];
    return c;
}
static Plain plain_fnma(Plain a, Plain b, Plain c) {
    for (int k = 0; k < 8; k++) c.lane[k] = c.lane[k] - a.lane[k] * b.lane[k];
    return c;
}
static void plain_sum(double* restrict out, const FastCDSumJob* job,
        double* restrict work);
#define FS_V Plain
#define FS_NAME(name) plain_##name
#define FS_PUBLIC plain_sum
#define FS_ZERO plain_zero()
#define FS_LOAD(p) plain_load(p)
#define FS_STORE(p, v) plain_store(p, v)
#define FS_SET1(s) plain_set1(s)
#define FS_ADD(a, b) plain_add(a, b)
#define FS_SUB(a, b) plain_sub(a, b)
#define FS_MUL(a, b) plain_mul(a, b)
#define FS_FMA(a, b, c) plain_fma(a, b, c)
#define FS_FNMA(a, b, c) plain_fnma(a, b, c)
#define FS_LANES(v, d) plain_store(d, v)
#include "fused_sum_impl.h"

/* Inexact operands: the result is within a rounding bound of the reference.
 * The kernels with a fused multiply-add return the same bits in both
 * layouts, the public functions return the bits of the fused or of the
 * separately rounded model, whichever the library runs here, and that
 * second model is itself within the bound. */
static int rounded_cases(void) {
    Kernel kernels[3] = {fastcd_fused_sum_portable, NULL, NULL};
    int available = 1;
    uint64_t state = 77;

#if defined(FASTCD_TEST_SUM_SIMD)
    available = fastcd_fused_sum_kernels();
    kernels[1] = fastcd_fused_sum_avx2;
    kernels[2] = fastcd_fused_sum_avx512;
#endif
    for (size_t n = 1; n <= 512; n *= 2) {
        static const size_t counts[] = {5, 64, 203};
        for (size_t ci = 0; ci < 3; ci++) {
            const size_t count = counts[ci], pitch = count + 3;
            const size_t doubles = fastcd_sum_kernel_doubles(n);
            double* x = make(count * n);
            double* y = make(count * n);
            double* px = make(n * pitch);
            double* py = make(n * pitch);
            double* want = make(n);
            double* first = make(n);
            double* plain = make(n);
            double* got = make(n);
            double* raw = make(doubles + 16);
            double* work = raw + ((64 - (uintptr_t)raw % 64) % 64) / sizeof(double);
            double scale = 0.0;
            Guarded g;
            size_t required;

            for (size_t i = 0; i < count * n; i++) {
                x[i] = uniform(&state);
                y[i] = uniform(&state);
            }
            for (size_t p = 0; p < count; p++) {
                double nx = 0.0, ny = 0.0;
                for (size_t i = 0; i < n; i++) {
                    nx += x[p * n + i] * x[p * n + i];
                    ny += y[p * n + i] * y[p * n + i];
                }
                scale += sqrt(nx * ny);
            }
            reference_sum(want, x, n, y, n, count, n);
            to_planes(px, pitch, x, n, count, n);
            to_planes(py, pitch, y, n, count, n);
            CHECK(fastcd_mul_sum_workspace_size(n, &required) == FASTCD_SUCCESS);
            g = guarded(required, 3);
            for (int quasilinear = 0; quasilinear < 2; quasilinear++) {
                const FastCDAlgorithm algorithm = quasilinear
                    ? FASTCD_ALGORITHM_QUASILINEAR : FASTCD_ALGORITHM_DIRECT;
                FastCDSumJob job;
                memset(&job, 0, sizeof job);
                job.x = x; job.y = y; job.x_stride = n; job.y_stride = n;
                job.count = count; job.dimension = n;
                job.quasilinear = quasilinear && n > 1;
                for (int k = 0; k < available; k++) {
                    for (int planes = 0; planes < 2; planes++) {
                        job.planes = planes != 0;
                        job.x = planes ? px : x;
                        job.y = planes ? py : y;
                        job.pitch = pitch;
                        kernels[k](got, &job, work);
                        if (k == 0 && planes == 0) {
                            memcpy(first, got, n * sizeof(double));
                            for (size_t i = 0; i < n; i++) {
                                CHECK(fabs(got[i] - want[i])
                                        <= 64.0 * DBL_EPSILON * scale);
                            }
                        }
                        CHECK(memcmp(got, first, n * sizeof(double)) == 0);
                    }
                }
                job.planes = false;
                job.x = x;
                job.y = y;
                plain_sum(plain, &job, work);
                for (size_t i = 0; i < n; i++) {
                    CHECK(fabs(plain[i] - want[i]) <= 64.0 * DBL_EPSILON * scale);
                }
                job.planes = true;
                job.x = px;
                job.y = py;
                plain_sum(got, &job, work);
                CHECK(memcmp(got, plain, n * sizeof(double)) == 0);
                CHECK(fastcd_mul_sum_with_algorithm_f64(got, x, n, y, n, count, n,
                            NULL, algorithm, &g.w) == FASTCD_SUCCESS);
                CHECK(memcmp(got, first, n * sizeof(double)) == 0
                        || memcmp(got, plain, n * sizeof(double)) == 0);
                CHECK(fastcd_mul_sum_planes_with_algorithm_f64(first, px, py,
                            pitch, count, n, algorithm, &g.w) == FASTCD_SUCCESS);
                CHECK(memcmp(got, first, n * sizeof(double)) == 0);
            }
            CHECK(guards_intact(&g));
            free(x); free(y); free(px); free(py); free(want); free(first);
            free(plain); free(got); free(raw); free(g.base);
        }
    }
    return 1;
}

/* Maps and broadcasting: a mapped sum equals the sum of records mapped
 * beforehand, bit for bit, and a zero stride repeats one record. */
static int map_cases(void) {
    uint64_t state = 5;

    for (size_t n = 1; n <= 64; n *= 2) {
        const size_t count = 77;
        double* x = make(count * n);
        double* y = make(count * n);
        double* mx = make(count * n);
        double* my = make(count * n);
        double* maps = make(4 * n);
        double* want = make(n);
        double* got = make(n);
        FastCDSumMapF64 map;
        Guarded g;
        size_t required;

        CHECK(fastcd_mul_sum_workspace_size(n, &required) == FASTCD_SUCCESS);
        g = guarded(required, 1);
        for (size_t i = 0; i < count * n; i++) {
            x[i] = uniform(&state);
            y[i] = uniform(&state);
        }
        for (size_t i = 0; i < 4 * n; i++) maps[i] = uniform(&state);
        map.x_shift = maps; map.x_scale = maps + n;
        map.y_shift = maps + 2 * n; map.y_scale = maps + 3 * n;
        for (int variant = 0; variant < 3; variant++) {
            if (variant == 1) map.x_scale = NULL;
            if (variant == 2) { map.y_shift = NULL; map.x_shift = NULL; }
            for (size_t p = 0; p < count; p++)
                for (size_t i = 0; i < n; i++) {
                    double a = x[p * n + i], b = y[p * n + i];
                    if (map.x_shift) a = a - map.x_shift[i];
                    if (map.x_scale) a = a * map.x_scale[i];
                    if (map.y_shift) b = b - map.y_shift[i];
                    if (map.y_scale) b = b * map.y_scale[i];
                    mx[p * n + i] = a;
                    my[p * n + i] = b;
                }
            CHECK(fastcd_mul_sum_f64(want, mx, n, my, n, count, n, NULL, &g.w)
                    == FASTCD_SUCCESS);
            CHECK(fastcd_mul_sum_f64(got, x, n, y, n, count, n, &map, &g.w)
                    == FASTCD_SUCCESS);
            CHECK(memcmp(got, want, n * sizeof(double)) == 0);
        }
        // x broadcast: sum_p x_0 y_p
        for (size_t p = 0; p < count; p++) memcpy(mx + p * n, x, n * sizeof(double));
        CHECK(fastcd_mul_sum_f64(want, mx, n, y, n, count, n, NULL, &g.w)
                == FASTCD_SUCCESS);
        CHECK(fastcd_mul_sum_f64(got, x, 0, y, n, count, n, NULL, &g.w)
                == FASTCD_SUCCESS);
        CHECK(memcmp(got, want, n * sizeof(double)) == 0);
        // The operands may be the same array
        CHECK(fastcd_mul_sum_f64(got, x, n, x, n, count, n, NULL, &g.w)
                == FASTCD_SUCCESS);
        reference_sum(want, x, n, x, n, count, n);
        for (size_t i = 0; i < n; i++) {
            CHECK(fabs(got[i] - want[i]) <= 64.0 * DBL_EPSILON * (double)(count * n));
        }
        CHECK(guards_intact(&g));
        free(x); free(y); free(mx); free(my); free(maps); free(want); free(got);
        free(g.base);
    }
    return 1;
}

// Compare the covariance with a long double evaluation of its definition
static int covariance_cases(void) {
    uint64_t state = 91;

    for (size_t n = 1; n <= 128; n *= 2) {
        static const size_t counts[] = {2, 9, 150};
        for (size_t ci = 0; ci < 3; ci++) {
            const size_t count = counts[ci], xs = n + 1, ys = n;
            double* x = make(count * xs);
            double* y = make(count * ys);
            double* cx = make(n);
            double* cy = make(n);
            double* product = make(n);
            double* got = make(n);
            long double* mean = calloc(2 * n, sizeof(long double));
            long double* total = calloc(n, sizeof(long double));
            double scale = 0.0;
            Guarded g;
            size_t required;

            CHECK(fastcd_covariance_workspace_size(n, &required) == FASTCD_SUCCESS);
            g = guarded(required, 5);
            for (size_t i = 0; i < count * xs; i++) x[i] = 3.0 + uniform(&state);
            for (size_t i = 0; i < count * ys; i++) y[i] = uniform(&state) - 7.0;
            for (size_t p = 0; p < count; p++)
                for (size_t i = 0; i < n; i++) {
                    mean[i] += x[p * xs + i];
                    mean[n + i] += y[p * ys + i];
                }
            for (size_t p = 0; p < count; p++) {
                double nx = 0.0, ny = 0.0;
                for (size_t i = 0; i < n; i++) {
                    cx[i] = (double)(x[p * xs + i] - mean[i] / (long double)count);
                    cy[i] = (double)(y[p * ys + i] - mean[n + i] / (long double)count);
                    if (i > 0) cy[i] = -cy[i];
                    nx += cx[i] * cx[i];
                    ny += cy[i] * cy[i];
                }
                scale += sqrt(nx * ny);
                reference_cd_mul_signed_f64(product, cx, cy, n);
                for (size_t i = 0; i < n; i++) total[i] += product[i];
            }
            for (size_t ddof = 0; ddof < 2; ddof++) {
                CHECK(fastcd_covariance_f64(got, x, xs, y, ys, count, n, ddof, &g.w)
                        == FASTCD_SUCCESS);
                for (size_t i = 0; i < n; i++) {
                    const double want = (double)(total[i] / (long double)(count - ddof));
                    CHECK(fabs(got[i] - want) <= 256.0 * DBL_EPSILON
                            * (scale + (double)n) / (double)(count - ddof));
                }
            }
            CHECK(fastcd_covariance_f64(got, x, xs, y, ys, count, n, count, &g.w)
                    == FASTCD_INVALID_COUNT);
            CHECK(guards_intact(&g));
            free(x); free(y); free(cx); free(cy); free(product); free(got);
            free(mean); free(total); free(g.base);
        }
    }
    return 1;
}

/* Covariances whose means or magnitudes defeat a direct evaluation: a
 * translation by an exactly representable amount, and samples near the
 * limits of the exponent range. Each result is exact. */
static int covariance_extremes(void) {
    double data[4096];
    double out[4];
    FastCDWorkspaceF64 w = {data, 4096};
    static const double base_x[4] = {0, 0, 2, 2}, base_y[4] = {0, 2, 0, 2};
    static const double shifts[4] = {0.0, 9007199254740992.0, -9007199254740992.0,
        4503599627370496.0};

    // Every translation is exact, so the covariances are those of the base
    for (size_t k = 0; k < 4; k++) {
        double x[4], y[4];
        for (size_t p = 0; p < 4; p++) {
            x[p] = base_x[p] + shifts[k];
            y[p] = base_y[p] + shifts[k];
        }
        CHECK(fastcd_covariance_f64(out, x, 1, y, 1, 4, 1, 1, &w) == FASTCD_SUCCESS);
        CHECK(out[0] == 0.0);
        CHECK(fastcd_covariance_f64(out, x, 1, x, 1, 4, 1, 1, &w) == FASTCD_SUCCESS);
        CHECK(out[0] == 4.0 / 3.0);
    }
    {
        // Opposite values at the top of the range against the bottom
        const double x[2] = {0x1p1023, -0x1p1023}, y[2] = {0x1p-1022, -0x1p-1022};
        CHECK(fastcd_covariance_f64(out, x, 1, y, 1, 2, 1, 0, &w) == FASTCD_SUCCESS);
        CHECK(out[0] == 2.0);
        CHECK(fastcd_covariance_f64(out, y, 1, x, 1, 2, 1, 0, &w) == FASTCD_SUCCESS);
        CHECK(out[0] == 2.0);
    }
    {
        // A variance whose sum of squares overflows before the division
        const double v[4] = {0x1p511, -0x1p511, 0x1p511, -0x1p511};
        const double tiny[4] = {0x1p-600, -0x1p-600, 0x1p-600, -0x1p-600};
        CHECK(fastcd_covariance_f64(out, v, 1, v, 1, 4, 1, 0, &w) == FASTCD_SUCCESS);
        CHECK(out[0] == 0x1p1022);
        // One whose squares underflow to zero
        CHECK(fastcd_covariance_f64(out, tiny, 1, tiny, 1, 4, 1, 0, &w)
                == FASTCD_SUCCESS);
        CHECK(out[0] == 0.0 || out[0] == 0x1p-1074);
        CHECK(fastcd_covariance_f64(out, tiny, 1, v, 1, 4, 1, 0, &w) == FASTCD_SUCCESS);
        CHECK(out[0] == 0x1p-89);
    }
    {
        // Quaternion samples at 2^600: imaginary coordinates and the conjugate
        double x[8] = {1, 2, 3, 4, -1, -2, -3, -4}, y[8] = {4, 3, 2, 1, -4, -3, -2, -1};
        double want[4];
        CHECK(fastcd_covariance_f64(want, x, 4, y, 4, 2, 4, 0, &w) == FASTCD_SUCCESS);
        for (size_t i = 0; i < 8; i++) { x[i] *= 0x1p600; y[i] *= 0x1p-300; }
        CHECK(fastcd_covariance_f64(out, x, 4, y, 4, 2, 4, 0, &w) == FASTCD_SUCCESS);
        for (size_t i = 0; i < 4; i++) CHECK(out[i] == want[i] * 0x1p300);
    }
    {
        // A single record has zero population covariance
        const double x[2] = {3.0, 5.0};
        CHECK(fastcd_covariance_f64(out, x, 2, x, 2, 1, 2, 0, &w) == FASTCD_SUCCESS);
        CHECK(out[0] == 0.0 && out[1] == 0.0);
    }
    return 1;
}

// Check the argument validation of the sums
static int contract_cases(void) {
    double x[64], y[64], out[8], small[4];
    double* data = make(4096);
    FastCDWorkspaceF64 w = {data, 4096};
    FastCDWorkspaceF64 tiny = {small, 4};
    FastCDAlgorithm algorithm;
    FastCDSumMapF64 map = {NULL, NULL, NULL, NULL};
    size_t length;

    for (size_t i = 0; i < 64; i++) { x[i] = 1.0; y[i] = 2.0; }
    CHECK(fastcd_default_sum_algorithm(8, &algorithm) == FASTCD_SUCCESS
            && algorithm == FASTCD_ALGORITHM_DIRECT);
    CHECK(fastcd_default_sum_algorithm(16, &algorithm) == FASTCD_SUCCESS
            && algorithm == FASTCD_ALGORITHM_QUASILINEAR);
    CHECK(fastcd_default_sum_algorithm(3, &algorithm) == FASTCD_INVALID_DIMENSION);
    CHECK(fastcd_default_sum_algorithm(8, NULL) == FASTCD_NULL_POINTER);
    CHECK(fastcd_mul_sum_workspace_size(6, &length) == FASTCD_INVALID_DIMENSION);
    CHECK(fastcd_mul_sum_workspace_size(8, NULL) == FASTCD_NULL_POINTER);
    CHECK(fastcd_mul_sum_workspace_size((SIZE_MAX >> 1) + 1, &length)
            == FASTCD_SIZE_OVERFLOW);
    CHECK(fastcd_mul_sum_workspace_size(8, &length) == FASTCD_SUCCESS
            && length <= 4096);
    CHECK(fastcd_mul_sum_f64(out, x, 8, y, 8, 8, 8, NULL, &w) == FASTCD_SUCCESS);
    CHECK(fastcd_mul_sum_f64(out, x, 8, y, 8, 8, 5, NULL, &w)
            == FASTCD_INVALID_DIMENSION);
    CHECK(fastcd_mul_sum_f64(NULL, x, 8, y, 8, 8, 8, NULL, &w) == FASTCD_NULL_POINTER);
    CHECK(fastcd_mul_sum_f64(out, NULL, 8, y, 8, 8, 8, NULL, &w) == FASTCD_NULL_POINTER);
    CHECK(fastcd_mul_sum_f64(out, x, 8, y, 8, 8, 8, NULL, NULL) == FASTCD_NULL_POINTER);
    CHECK(fastcd_mul_sum_f64(out, x, 8, y, 8, 8, 8, NULL, &tiny)
            == FASTCD_INSUFFICIENT_WORKSPACE);
    CHECK(fastcd_mul_sum_f64(out, x, 7, y, 8, 8, 8, NULL, &w) == FASTCD_INVALID_STRIDE);
    CHECK(fastcd_mul_sum_f64(x + 8, x, 8, y, 8, 8, 8, NULL, &w)
            == FASTCD_OVERLAPPING_BUFFERS);
    CHECK(fastcd_mul_sum_f64(data, x, 8, y, 8, 8, 8, NULL, &w)
            == FASTCD_OVERLAPPING_BUFFERS);
    map.y_scale = out;
    CHECK(fastcd_mul_sum_f64(out, x, 8, y, 8, 8, 8, &map, &w)
            == FASTCD_OVERLAPPING_BUFFERS);
    CHECK(fastcd_mul_sum_with_algorithm_f64(out, x, 8, y, 8, 8, 8, NULL,
                FASTCD_ALGORITHM_CARIOW_CARIOWA, &w) == FASTCD_UNSUPPORTED_ALGORITHM);
    CHECK(fastcd_mul_sum_with_algorithm_f64(out, x, 8, y, 8, 8, 8, NULL,
                (FastCDAlgorithm)17, &w) == FASTCD_UNSUPPORTED_ALGORITHM);
    // A count of zero reads no input and needs no workspace
    out[3] = 5.0;
    CHECK(fastcd_mul_sum_f64(out, NULL, 0, NULL, 0, 0, 8, NULL, NULL) == FASTCD_SUCCESS);
    CHECK(out[3] == 0.0);
    CHECK(fastcd_mul_sum_planes_f64(out, x, y, 8, 8, 8, &w) == FASTCD_SUCCESS);
    CHECK(fastcd_mul_sum_planes_f64(out, x, y, 7, 8, 8, &w) == FASTCD_INVALID_STRIDE);
    CHECK(fastcd_mul_sum_planes_f64(out, x, NULL, 8, 8, 8, &w) == FASTCD_NULL_POINTER);
    CHECK(fastcd_mul_sum_planes_f64(x, x, y, 8, 8, 8, &w) == FASTCD_OVERLAPPING_BUFFERS);
    CHECK(fastcd_mul_sum_planes_with_algorithm_f64(out, x, y, 8, 8, 8,
                FASTCD_ALGORITHM_CARIOW_CARIOWA, &w) == FASTCD_UNSUPPORTED_ALGORITHM);
    CHECK(fastcd_covariance_workspace_size(8, &length) == FASTCD_SUCCESS
            && length <= 4096);
    CHECK(fastcd_covariance_f64(out, x, 8, y, 8, 8, 8, 1, &w) == FASTCD_SUCCESS);
    // Constant samples have zero covariance
    for (size_t i = 0; i < 8; i++) CHECK(out[i] == 0.0);
    CHECK(fastcd_covariance_f64(out, x, 8, y, 8, 1, 8, 1, &w) == FASTCD_INVALID_COUNT);
    CHECK(fastcd_covariance_f64(out, x, 8, y, 8, 8, 8, 1, &tiny)
            == FASTCD_INSUFFICIENT_WORKSPACE);
    CHECK(fastcd_covariance_f64(out, x, 8, y, 8, 8, 8, 1, NULL) == FASTCD_NULL_POINTER);
    CHECK(fastcd_covariance_f64(out, x, 3, y, 8, 8, 8, 1, &w) == FASTCD_INVALID_STRIDE);
    CHECK(strcmp(fastcd_status_string(FASTCD_INVALID_COUNT), "Invalid element count") == 0);
    free(data);
    return 1;
}

int main(void) {
    if (!exact_cases() || !rounded_cases() || !map_cases() || !covariance_cases()
            || !covariance_extremes() || !contract_cases()) {
        return 1;
    }
    printf("fused sum tests passed\n");
    return 0;
}
