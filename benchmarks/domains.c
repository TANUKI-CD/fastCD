#define _POSIX_C_SOURCE 200809L
#include "fastcd/complex.h"
#include "fastcd/finite.h"
#include "finite_backend.h"
#include "stack_offset.h"
#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(_WIN32)
#include <windows.h>
#endif
#ifndef FASTCD_BENCHMARK_BUILD_ID
#define FASTCD_BENCHMARK_BUILD_ID "unknown"
#endif

/* Complex and finite-field multiplication benchmark runner. Allocation, field
 * setup, corpus I/O and oracle verification all happen outside timing, and
 * each timed observation replays whole corpus passes. Each block rotates every
 * algorithm and mode task to balance their positions and runs at its own
 * stack offset (see stack_offset.h). */
typedef struct {
    uint32_t p, d;
    size_t n, trials, words, bytes, length;
    int single;
    void *x, *y, *out, *work;
    FastCDPlanFF plan;
    FastCDWorkspaceF64 complex_work;
    double* real_storage;
    FastCDWorkspaceF64 real_work;
} Task;

// Read a monotonic high-resolution clock without allocating or printing
static uint64_t now_ns(void) {
#if defined(_WIN32)
    LARGE_INTEGER value, frequency;
    QueryPerformanceCounter(&value);
    QueryPerformanceFrequency(&frequency);
    return (uint64_t)((long double)value.QuadPart * 1e9L / frequency.QuadPart);
#else
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) + (uint64_t)value.tv_nsec;
#endif
}

// Derive standard basis signs independently of every production kernel
static int basis_sign(size_t n, size_t i, size_t j) {
    if (n == 1) return 1;
    size_t h = n / 2;
    if (i < h && j < h) return basis_sign(h, i, j);
    if (i < h) return basis_sign(h, j - h, i);
    if (j < h) return (j ? -1 : 1) * basis_sign(h, i - h, j);
    return (j == h ? -1 : 1) * basis_sign(h, j - h, i - h);
}

// Use schoolbook polynomial multiplication and ordinary remainder arithmetic
static void scalar_reference(
        uint32_t* out, const uint32_t* x, const uint32_t* y, const FastCDField* f) {
    uint64_t a[23] = {0};
    size_t d = f->degree;
    for (size_t i = 0; i < d; i++)
        for (size_t j = 0; j < d; j++)
            a[i + j] = (a[i + j] + (uint64_t)x[i] * y[j]) % f->prime;
    for (size_t k = 2 * d - 1; k-- > d;)
        for (size_t j = 0; j < d; j++)
            a[k - d + j] = (a[k - d + j] + f->prime - a[k] * f->modulus[j] % f->prime) %
                    f->prime;
    for (size_t i = 0; i < d; i++) out[i] = (uint32_t)a[i];
}

// Replay one complete corpus pass using already allocated storage and scratch
static int run(Task* t, int alg) {
    FastCDAlgorithm a =
            alg == 0 ? FASTCD_ALGORITHM_DIRECT : FASTCD_ALGORITHM_QUASILINEAR;
    if (t->p) {
        if (!t->single)
            return fastcd_mul_batch_ff(t->out, t->words, t->x, t->words, t->y, t->words,
                           t->trials, &t->plan, a, t->work,
                           t->length) == FASTCD_FF_SUCCESS;
        for (size_t i = 0; i < t->trials; i++)
            if (fastcd_mul_with_algorithm_ff((uint32_t*)t->out + i * t->words,
                        (uint32_t*)t->x + i * t->words, (uint32_t*)t->y + i * t->words,
                        &t->plan, a, t->work, t->length) != FASTCD_FF_SUCCESS)
                return 0;
        return 1;
    }
    if (alg == 2) {
        // Three real CD products form an independent complex baseline, timed
        // with packing and reconstruction in preallocated buffers
        double *xr = t->real_storage, *xi = xr + t->n, *yr = xi + t->n, *yi = yr + t->n;
        double *sx = yi + t->n, *sy = sx + t->n, *u = sy + t->n, *v = u + t->n,
               *w = v + t->n;
        const FastCDComplexF64 *x = t->x, *y = t->y;
        FastCDComplexF64* out = t->out;
        for (size_t k = 0; k < t->trials; k++) {
            for (size_t i = 0; i < t->n; i++) {
                xr[i] = x[k * t->n + i].real;
                xi[i] = x[k * t->n + i].imag;
                yr[i] = y[k * t->n + i].real;
                yi[i] = y[k * t->n + i].imag;
                sx[i] = xr[i] + xi[i];
                sy[i] = yr[i] + yi[i];
            }
            if (fastcd_mul_with_algorithm_f64(
                        u, xr, yr, t->n, FASTCD_ALGORITHM_QUASILINEAR, &t->real_work) ||
                    fastcd_mul_with_algorithm_f64(v, xi, yi, t->n,
                            FASTCD_ALGORITHM_QUASILINEAR, &t->real_work) ||
                    fastcd_mul_with_algorithm_f64(w, sx, sy, t->n,
                            FASTCD_ALGORITHM_QUASILINEAR, &t->real_work))
                return 0;
            for (size_t i = 0; i < t->n; i++) {
                out[k * t->n + i].real = u[i] - v[i];
                out[k * t->n + i].imag = w[i] - u[i] - v[i];
            }
        }
        return 1;
    }
    if (!t->single)
        return fastcd_mul_batch_c128(t->out, t->n, t->x, t->n, t->y, t->n, t->trials,
                       t->n, a, &t->complex_work) == FASTCD_SUCCESS;
    for (size_t i = 0; i < t->trials; i++)
        if (fastcd_mul_with_algorithm_c128((FastCDComplexF64*)t->out + i * t->n,
                    (FastCDComplexF64*)t->x + i * t->n,
                    (FastCDComplexF64*)t->y + i * t->n, t->n, a,
                    &t->complex_work) != FASTCD_SUCCESS)
            return 0;
    return 1;
}

// One timed observation: a task's calibrated corpus passes and their elapsed time
typedef struct {
    Task* task;
    int algorithm;
    size_t passes;
    uint64_t elapsed;
} Observation;

// Time an observation's corpus passes, returning zero if any call fails
static int observe(void* context) {
    Observation* o = context;
    uint64_t start = now_ns();
    for (size_t r = 0; r < o->passes; r++)
        if (!run(o->task, o->algorithm)) return 0;
    o->elapsed = now_ns() - start;
    return 1;
}

// Hash output bytes after timing to keep results observable and record provenance
static uint64_t checksum(const Task* t) {
    const unsigned char* a = t->out;
    uint64_t h = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < t->trials * t->bytes; i++)
        h = (h ^ a[i]) * UINT64_C(1099511628211);
    return h;
}

// Check every pair in each requested implementation/mode against an independent oracle
static int verify(
        Task* t, int algorithms, int both, size_t* verified, long double* max_error) {
    if (t->n > SIZE_MAX / t->n) return 0;
    int8_t* signs = malloc(t->n * t->n);
    void* reference = calloc(t->trials, t->bytes);
    if (!signs || !reference) {
        free(signs);
        free(reference);
        return 0;
    }
    for (size_t i = 0; i < t->n; i++)
        for (size_t j = 0; j < t->n; j++)
            signs[i * t->n + j] = (int8_t)basis_sign(t->n, i, j);
    for (size_t trial = 0; trial < t->trials; trial++) {
        if (t->p) {
            const uint32_t *x = (uint32_t*)t->x + trial * t->words,
                           *y = (uint32_t*)t->y + trial * t->words;
            uint32_t *o = (uint32_t*)reference + trial * t->words, scalar[12];
            for (size_t i = 0; i < t->n; i++)
                for (size_t j = 0; j < t->n; j++) {
                    scalar_reference(
                            scalar, x + i * t->d, y + j * t->d, &t->plan.field);
                    for (size_t k = 0; k < t->d; k++) {
                        size_t pos = (i ^ j) * t->d + k;
                        o[pos] =
                                (uint32_t)(((uint64_t)o[pos] +
                                                   (signs[i * t->n + j] > 0 ? scalar[k]
                                                                            : t->p -
                                                                           scalar[k])) %
                                        t->p);
                    }
                }
        } else {
            const FastCDComplexF64 *x = (FastCDComplexF64*)t->x + trial * t->n,
                                   *y = (FastCDComplexF64*)t->y + trial * t->n;
            FastCDComplexF64* o = (FastCDComplexF64*)reference + trial * t->n;
            for (size_t k = 0; k < t->n; k++) {
                long double re = 0, im = 0;
                for (size_t i = 0; i < t->n; i++) {
                    size_t j = i ^ k;
                    long double sign = signs[i * t->n + j];
                    re += sign *
                            ((long double)x[i].real * y[j].real -
                                    (long double)x[i].imag * y[j].imag);
                    im += sign *
                            ((long double)x[i].real * y[j].imag +
                                    (long double)x[i].imag * y[j].real);
                }
                o[k].real = (double)re;
                o[k].imag = (double)im;
            }
        }
    }
    int ok = 1;
    int selected_mode = t->single;
    for (int task = 0; task < algorithms * (both ? 2 : 1) && ok; task++) {
        int alg = both ? task / 2 : task;
        t->single = both ? task % 2 : selected_mode;
        if (!run(t, alg)) {
            ok = 0;
            break;
        }
        if (t->p)
            ok = memcmp(t->out, reference, t->trials * t->bytes) == 0;
        else {
            const FastCDComplexF64 *o = t->out, *r = reference;
            for (size_t trial = 0; trial < t->trials && ok; trial++) {
                const FastCDComplexF64 *x = (FastCDComplexF64*)t->x + trial * t->n,
                                       *y = (FastCDComplexF64*)t->y + trial * t->n;
                long double nx = 0, ny = 0;
                for (size_t i = 0; i < t->n; i++) {
                    nx += (long double)x[i].real * x[i].real +
                            (long double)x[i].imag * x[i].imag;
                    ny += (long double)y[i].real * y[i].real +
                            (long double)y[i].imag * y[i].imag;
                }
                long double scale = sqrtl(nx * ny);
                for (size_t i = 0; i < t->n; i++) {
                    size_t k = trial * t->n + i;
                    long double error = hypotl((long double)o[k].real - r[k].real,
                            (long double)o[k].imag - r[k].imag);
                    long double relative = scale != 0.0L ? error / scale : error;
                    if (relative > *max_error) *max_error = relative;
                    if (!isfinite(error) ||
                            relative > 64 * DBL_EPSILON * (long double)t->n)
                        ok = 0;
                }
            }
        }
    }
    if (ok) *verified = t->trials;
    t->single = selected_mode;
    free(signs);
    free(reference);
    return ok;
}

// Parse a nonnegative option without narrowing or accepting signed spellings
static size_t parse(const char* value) {
    char* end;
    errno = 0;
    uintmax_t n = strtoumax(value, &end, 10);
    if (errno || *end || *value == '-' || *value == '+' || !value[0] || n > SIZE_MAX) {
        fprintf(stderr, "invalid integer\n");
        exit(EXIT_FAILURE);
    }
    return (size_t)n;
}

// Set up, verify, calibrate and time a single field/dimension corpus
int main(int argc, char** argv) {
    if (argc == 2 && strcmp(argv[1], "--metadata") == 0) {
        printf("{\"schema\":\"fastcd-domains-1\",\"build_id\":\"%s\",\"version\":\"%"
               "s\","
               "\"algorithms\":[\"direct\",\"quasilinear\"],\"clock\":\"monotonic\"}\n",
                FASTCD_BENCHMARK_BUILD_ID, fastcd_version_string());
        return EXIT_SUCCESS;
    }
    Task t = {0};
    t.d = 1;
    size_t blocks = 24, warmup = 2;
    uint64_t minimum = UINT64_C(20000000);
    const char* corpus = NULL;
    int baseline = 0, both = 1;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 == argc) {
            fprintf(stderr, "option needs value\n");
            return EXIT_FAILURE;
        }
        const char *key = argv[i], *v = argv[i + 1];
        if (!strcmp(key, "--corpus"))
            corpus = v;
        else if (!strcmp(key, "--mode")) {
            if (strcmp(v, "single") && strcmp(v, "batch") && strcmp(v, "both"))
                return EXIT_FAILURE;
            both = !strcmp(v, "both");
            t.single = !strcmp(v, "single");
        } else if (!strcmp(key, "--char")) {
            size_t p = parse(v);
            if (p > INT32_MAX) return EXIT_FAILURE;
            t.p = (uint32_t)p;
        } else if (!strcmp(key, "--deg")) {
            size_t d = parse(v);
            if (d > 12) return EXIT_FAILURE;
            t.d = (uint32_t)d;
        } else if (!strcmp(key, "--dimension"))
            t.n = parse(v);
        else if (!strcmp(key, "--trials"))
            t.trials = parse(v);
        else if (!strcmp(key, "--blocks"))
            blocks = parse(v);
        else if (!strcmp(key, "--warmup"))
            warmup = parse(v);
        else if (!strcmp(key, "--min-time-ms")) {
            char* end;
            errno = 0;
            double ms = strtod(v, &end);
            if (errno || end == v || *end || !isfinite(ms) || ms < 0 ||
                    ms > (double)(UINT64_MAX / 1000000))
                return EXIT_FAILURE;
            minimum = (uint64_t)(ms * 1e6);
        } else if (!strcmp(key, "--baseline"))
            baseline = parse(v) != 0;
        else {
            fprintf(stderr, "unknown option: %s\n", key);
            return EXIT_FAILURE;
        }
    }
    if (!corpus || !fastcd_valid_dimension(t.n) || !t.trials || blocks < 2 || !t.d ||
            (!t.p && t.d != 2) || (t.p && t.p < 3)) {
        fprintf(stderr, "invalid domain benchmark configuration\n");
        return EXIT_FAILURE;
    }
    size_t level;
    fastcd_level(t.n, &level);
    if (t.p) {
        FastCDField f;
        if (fastcd_field_init(&f, t.p, t.d, NULL) ||
                fastcd_ff_plan_init(&t.plan, &f, level))
            return EXIT_FAILURE;
        t.words = t.n * t.d;
        t.bytes = t.words * 4;
        t.length = t.plan.workspace_words;
        if (baseline) {
            fprintf(stderr,
                    "--baseline is available only for complex coefficients; "
                    "use the flint protocol for finite-field comparisons\n");
            return EXIT_FAILURE;
        }
    } else {
        t.words = t.n;
        t.bytes = t.n * sizeof(FastCDComplexF64);
        if (fastcd_workspace_size_c128(t.n, FASTCD_OP_MULTIPLY, &t.length))
            return EXIT_FAILURE;
    }
    size_t item = t.p ? 4 : 8;
    if (t.trials > SIZE_MAX / t.bytes || t.length > SIZE_MAX / item)
        return EXIT_FAILURE;
    // Corpus I/O and every scratch allocation precede verification and timing,
    // so no implementation allocates inside its timer
    t.x = malloc(t.trials * t.bytes);
    t.y = malloc(t.trials * t.bytes);
    t.out = malloc(t.trials * t.bytes);
    t.work = malloc(t.length * item);
    if (!t.x || !t.y || !t.out || !t.work) return EXIT_FAILURE;
    FILE* input = fopen(corpus, "rb");
    if (!input || fread(t.x, t.bytes, t.trials, input) != t.trials ||
            fread(t.y, t.bytes, t.trials, input) != t.trials || fgetc(input) != EOF)
        return EXIT_FAILURE;
    fclose(input);
    t.complex_work.data = t.work;
    t.complex_work.length = t.length;
    if (t.p) {
        for (size_t i = 0; i < t.trials * t.words; i++)
            if (((uint32_t*)t.x)[i] >= t.p || ((uint32_t*)t.y)[i] >= t.p)
                return EXIT_FAILURE;
    } else {
        size_t real_len;
        if (fastcd_mul_workspace_size(t.n, &real_len) ||
                t.n > (SIZE_MAX / sizeof(double) - real_len) / 9)
            return EXIT_FAILURE;
        t.real_storage = malloc((9 * t.n + real_len) * sizeof(double));
        if (!t.real_storage) return EXIT_FAILURE;
        t.real_work.data = t.real_storage + 9 * t.n;
        t.real_work.length = real_len;
    }
    int algorithms = baseline ? 3 : 2;
    size_t verified = 0;
    long double error = 0;
    if (!verify(&t, algorithms, both, &verified, &error)) {
        fprintf(stderr, "independent verification failed\n");
        return EXIT_FAILURE;
    }
    int tasks = algorithms * (both ? 2 : 1);
    int selected_mode = t.single;
    // Calibrate whole corpus passes independently for each algorithm and mode,
    // so every observation covers the same pairs
    size_t passes[6] = {1, 1, 1, 1, 1, 1};
    for (int task = 0; task < tasks; task++) {
        int a = both ? task / 2 : task;
        t.single = both ? task % 2 : selected_mode;
        for (size_t w = 0; w < warmup; w++)
            if (!run(&t, a)) return EXIT_FAILURE;
        for (;;) {
            uint64_t start = now_ns();
            for (size_t r = 0; r < passes[task]; r++)
                if (!run(&t, a)) return EXIT_FAILURE;
            uint64_t elapsed = now_ns() - start;
            if (elapsed >= minimum || !minimum) break;
            if (passes[task] > SIZE_MAX / 2 / t.trials) return EXIT_FAILURE;
            passes[task] *= 2;
        }
    }
    printf("{\"schema\":\"fastcd-domains-1\",\"char\":%" PRIu32 ",\"deg\":%" PRIu32
           ",\"dimension\":%zu,"
           "\"mode\":\"%s\",\"trials\":%zu,\"verified\":%zu,\"max_norm_scaled_error\":%"
           ".17g,"
           "\"modulus\":[",
            t.p, t.d, t.n, both ? "both" : (selected_mode ? "single" : "batch"),
            t.trials, verified, (double)error);
    if (t.p)
        for (size_t j = 0; j <= t.d; j++)
            printf("%s%" PRIu32, j ? "," : "", t.plan.field.modulus[j]);
    printf("],\"backend\":\"%s\",\"default_algorithm\":\"%s\","
           "\"warmup\":%zu,\"min_time_ns\":%" PRIu64 ",\"observations\":[",
            t.p ? "finite_shared_dispatch" : "complex_quasilinear",
            fastcd_algorithm_string(fastcd_domain_auto_algorithm(t.n)), warmup,
            minimum);
    const char* names[3] = {"direct", "quasilinear", "three_real"};
    // Rotate task positions between blocks and give each block its own stack
    // offset, with serialization, checksums and flushing after each timer and
    // never inside measured product passes
    for (size_t block = 0; block < blocks; block++)
        for (int k = 0; k < tasks; k++) {
            int task = (int)((block + (size_t)k) % (size_t)tasks);
            int a = both ? task / 2 : task;
            t.single = both ? task % 2 : selected_mode;
            size_t offset = fastcd_block_stack_offset(block, blocks);
            Observation o = {&t, a, passes[task], 0};
            if (!fastcd_at_stack_offset(offset, observe, &o)) return EXIT_FAILURE;
            printf("%s{\"block\":%zu,\"algorithm\":\"%s\",\"mode\":\"%s\",\"elapsed_"
                   "ns\":%" PRIu64 ",\"passes\":%zu,"
                   "\"operations\":%zu,\"stack_offset\":%zu,\"checksum\":\"%016" PRIx64
                   "\"}",
                    block || k ? "," : "", block, names[a],
                    t.single ? "single" : "batch", o.elapsed, passes[task],
                    passes[task] * t.trials, offset, checksum(&t));
            fflush(stdout);
        }
    puts("]}");
    free(t.x);
    free(t.y);
    free(t.out);
    free(t.work);
    free(t.real_storage);
    return EXIT_SUCCESS;
}
