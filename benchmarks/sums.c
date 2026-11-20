#define _POSIX_C_SOURCE 200809L
#include "fastcd/fastcd.h"
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
#else
#include <dlfcn.h>
#endif
#ifndef FASTCD_BENCHMARK_BUILD_ID
#define FASTCD_BENCHMARK_BUILD_ID "unknown"
#endif

/* Fused-sum benchmark runner. The corpus is a run of real-coefficient
 * elements cut into samples of `count` elements, and every task computes the
 * sum of products of each sample:
 *
 *   products            one batch of products, then a sum over the products
 *   records             fastcd_mul_sum_f64 on the records
 *   planes_direct       fastcd_mul_sum_planes_with_algorithm_f64, direct
 *   planes_quasilinear  the same, quasilinear
 *   dgemm               the matrix of real cross moments by one dgemm of an
 *                       optional BLAS library, then its signed contraction
 *
 * Allocation, corpus I/O, the transposition into planes and verification
 * happen outside timing, and each timed observation replays whole corpus
 * passes. Each block rotates the tasks to balance their positions and runs
 * at its own stack offset (see stack_offset.h). */

enum { PRODUCTS, RECORDS, PLANES_DIRECT, PLANES_QUASILINEAR, DGEMM, TASKS };
static const char* const task_names[TASKS] = {"products", "records",
    "planes_direct", "planes_quasilinear", "dgemm"};

typedef void (*Dgemm32)(int, int, int, int, int, int, double, const double*,
        int, const double*, int, double, double*, int);
typedef void (*Dgemm64)(int, int, int, int64_t, int64_t, int64_t, double,
        const double*, int64_t, const double*, int64_t, double, double*, int64_t);

typedef struct {
    size_t n, count, samples, pitch;
    double *x, *y, *px, *py, *products, *gram, *out, *scratch;
    signed char* sign;
    FastCDWorkspaceF64 mul_work, sum_work;
    Dgemm32 dgemm32;
    Dgemm64 dgemm64;
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

// Compute the sum of products of one sample with one task into out
static int sample_sum(const Task* t, int task, size_t s, double* out) {
    const size_t n = t->n, count = t->count;
    const double* x = t->x + s * count * n;
    const double* y = t->y + s * count * n;
    const double* px = t->px + s * n * t->pitch;
    const double* py = t->py + s * n * t->pitch;
    switch (task) {
        case PRODUCTS:
            if (fastcd_mul_batch_f64(t->products, n, x, n, y, n, count, n,
                        t->mul_work.length ? &t->mul_work : NULL))
                return 0;
            for (size_t i = 0; i < n; i++) out[i] = 0.0;
            for (size_t p = 0; p < count; p++)
                for (size_t i = 0; i < n; i++) out[i] += t->products[p * n + i];
            return 1;
        case RECORDS:
            return fastcd_mul_sum_f64(out, x, n, y, n, count, n, NULL,
                           &t->sum_work) == FASTCD_SUCCESS;
        case PLANES_DIRECT:
            return fastcd_mul_sum_planes_with_algorithm_f64(out, px, py, t->pitch,
                           count, n, FASTCD_ALGORITHM_DIRECT, &t->sum_work) ==
                   FASTCD_SUCCESS;
        case PLANES_QUASILINEAR:
            return fastcd_mul_sum_planes_with_algorithm_f64(out, px, py, t->pitch,
                           count, n, FASTCD_ALGORITHM_QUASILINEAR, &t->sum_work) ==
                   FASTCD_SUCCESS;
        default:
            // G = X^T Y for records X, Y; then out[i ^ j] += sign(i, j) G[i][j]
            if (t->dgemm64)
                t->dgemm64(101, 112, 111, (int64_t)n, (int64_t)n, (int64_t)count, 1.0,
                        x, (int64_t)n, y, (int64_t)n, 0.0, t->gram, (int64_t)n);
            else
                t->dgemm32(101, 112, 111, (int)n, (int)n, (int)count, 1.0, x, (int)n,
                        y, (int)n, 0.0, t->gram, (int)n);
            for (size_t i = 0; i < n; i++) out[i] = 0.0;
            for (size_t i = 0; i < n; i++) {
                const double* g = t->gram + i * n;
                const signed char* sg = t->sign + i * n;
                for (size_t j = 0; j < n; j++) out[i ^ j] += sg[j] * g[j];
            }
            return 1;
    }
}

// Run one corpus pass of a task, leaving the sums of every sample in t->out
static int run(const Task* t, int task) {
    for (size_t s = 0; s < t->samples; s++)
        if (!sample_sum(t, task, s, t->out + s * t->n)) return 0;
    return 1;
}

typedef struct {
    const Task* task;
    int id;
    size_t passes;
    uint64_t elapsed;
} Observation;

// Time an observation's corpus passes, returning zero if any call fails
static int observe(void* context) {
    Observation* o = context;
    uint64_t start = now_ns();
    for (size_t r = 0; r < o->passes; r++)
        if (!run(o->task, o->id)) return 0;
    o->elapsed = now_ns() - start;
    return 1;
}

// Hash output bytes after timing to keep results observable and record provenance
static uint64_t checksum(const Task* t) {
    uint64_t hash = UINT64_C(1469598103934665603);
    const unsigned char* bytes = (const unsigned char*)t->out;
    for (size_t i = 0; i < t->samples * t->n * sizeof(double); i++) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

/* Check every enabled task outside timing. The first `exact` samples are
 * compared with a long double contraction of the cross moments; on every
 * sample the two fused layouts must agree bit for bit when they run the same
 * algorithm, and every other task must agree with them within the bound. */
static int verify(const Task* t, const int* enabled, size_t exact, FastCDAlgorithm
        selected, long double* worst) {
    const size_t n = t->n, count = t->count;
    long double* reference = malloc(n * sizeof(*reference));
    double* first = malloc(n * sizeof(double));
    double* other = malloc(n * sizeof(double));
    int ok = reference && first && other;
    *worst = 0;
    for (size_t s = 0; ok && s < t->samples; s++) {
        const double* x = t->x + s * count * n;
        const double* y = t->y + s * count * n;
        long double scale = 0;
        for (size_t p = 0; p < count; p++) {
            long double nx = 0, ny = 0;
            for (size_t i = 0; i < n; i++) {
                nx += (long double)x[p * n + i] * x[p * n + i];
                ny += (long double)y[p * n + i] * y[p * n + i];
            }
            scale += sqrtl(nx * ny);
        }
        const long double bound = 64 * DBL_EPSILON * scale;
        ok = sample_sum(t, RECORDS, s, first);
        if (ok && s < exact) {
            for (size_t k = 0; k < n; k++) reference[k] = 0;
            for (size_t i = 0; i < n; i++)
                for (size_t j = 0; j < n; j++) {
                    long double g = 0;
                    for (size_t p = 0; p < count; p++)
                        g += (long double)x[p * n + i] * y[p * n + j];
                    reference[i ^ j] += t->sign[i * n + j] * g;
                }
            for (size_t k = 0; ok && k < n; k++) {
                long double error = fabsl(first[k] - reference[k]);
                if (scale > 0 && error / scale > *worst) *worst = error / scale;
                ok = error <= bound;
            }
        }
        for (int task = 0; ok && task < TASKS; task++) {
            if (!enabled[task] || task == RECORDS) continue;
            ok = sample_sum(t, task, s, other);
            const int same_schedule = (task == PLANES_DIRECT &&
                                              selected == FASTCD_ALGORITHM_DIRECT) ||
                                      (task == PLANES_QUASILINEAR &&
                                              selected == FASTCD_ALGORITHM_QUASILINEAR);
            if (ok && same_schedule)
                ok = memcmp(first, other, n * sizeof(double)) == 0;
            for (size_t k = 0; ok && k < n; k++)
                ok = fabsl((long double)first[k] - other[k]) <= 2 * bound;
        }
    }
    free(reference);
    free(first);
    free(other);
    return ok;
}

// Parse a nonnegative decimal size
static size_t parse(const char* value) {
    char* end;
    errno = 0;
    unsigned long long parsed = strtoull(value, &end, 10);
    if (errno || end == value || *end || parsed > SIZE_MAX) {
        fprintf(stderr, "invalid number: %s\n", value);
        exit(EXIT_FAILURE);
    }
    return (size_t)parsed;
}

int main(int argc, char** argv) {
    if (argc == 2 && strcmp(argv[1], "--metadata") == 0) {
        printf("{\"schema\":\"fastcd-sums-1\",\"build_id\":\"%s\",\"version\":\"%s\","
               "\"tasks\":[\"products\",\"records\",\"planes_direct\","
               "\"planes_quasilinear\",\"dgemm\"],\"clock\":\"monotonic\"}\n",
                FASTCD_BENCHMARK_BUILD_ID, fastcd_version_string());
        return EXIT_SUCCESS;
    }
    Task t = {0};
    size_t trials = 0, blocks = 60, warmup = 2, exact = 1, max_direct = 64;
    uint64_t minimum = UINT64_C(20000000);
    const char *corpus = NULL, *blas = NULL, *symbol = "cblas_dgemm";
    int ilp64 = 0;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 == argc) {
            fprintf(stderr, "option needs value\n");
            return EXIT_FAILURE;
        }
        const char *key = argv[i], *v = argv[i + 1];
        if (!strcmp(key, "--corpus"))
            corpus = v;
        else if (!strcmp(key, "--dimension"))
            t.n = parse(v);
        else if (!strcmp(key, "--count"))
            t.count = parse(v);
        else if (!strcmp(key, "--trials"))
            trials = parse(v);
        else if (!strcmp(key, "--blocks"))
            blocks = parse(v);
        else if (!strcmp(key, "--warmup"))
            warmup = parse(v);
        else if (!strcmp(key, "--verify-samples"))
            exact = parse(v);
        else if (!strcmp(key, "--max-direct-dimension"))
            max_direct = parse(v);
        else if (!strcmp(key, "--blas"))
            blas = v;
        else if (!strcmp(key, "--blas-symbol"))
            symbol = v;
        else if (!strcmp(key, "--blas-ilp64"))
            ilp64 = parse(v) != 0;
        else if (!strcmp(key, "--min-time-ms")) {
            char* end;
            errno = 0;
            double ms = strtod(v, &end);
            if (errno || end == v || *end || !isfinite(ms) || ms < 0 ||
                    ms > (double)(UINT64_MAX / 1000000))
                return EXIT_FAILURE;
            minimum = (uint64_t)(ms * 1e6);
        } else {
            fprintf(stderr, "unknown option: %s\n", key);
            return EXIT_FAILURE;
        }
    }
    if (!corpus || !fastcd_valid_dimension(t.n) || !t.count || !trials ||
            trials % t.count || blocks < 2 || t.n > INT32_MAX || t.count > INT32_MAX) {
        fprintf(stderr, "invalid sum benchmark configuration\n");
        return EXIT_FAILURE;
    }
    t.samples = trials / t.count;
    // Eight spare doubles keep rows a power of two apart out of one cache set
    t.pitch = t.count + 8;
    if (blas) {
#if defined(_WIN32)
        fprintf(stderr, "--blas is not available on this platform\n");
        return EXIT_FAILURE;
#else
        void* library = dlopen(blas, RTLD_NOW | RTLD_GLOBAL);
        void* entry = library ? dlsym(library, symbol) : NULL;
        if (!entry) {
            fprintf(stderr, "cannot load %s from %s: %s\n", symbol, blas, dlerror());
            return EXIT_FAILURE;
        }
        // POSIX guarantees that a symbol address converts to a function pointer
        if (ilp64)
            memcpy(&t.dgemm64, &entry, sizeof entry);
        else
            memcpy(&t.dgemm32, &entry, sizeof entry);
#endif
    }
    size_t mul_length, sum_length;
    FastCDAlgorithm selected;
    // Every allocation below is one of these extents in doubles or in bytes
    const size_t limit = SIZE_MAX / sizeof(double);
    if (fastcd_mul_workspace_size(t.n, &mul_length) ||
            fastcd_mul_sum_workspace_size(t.n, &sum_length) ||
            fastcd_default_sum_algorithm(t.n, &selected) || t.n > limit / t.n ||
            trials > limit / t.n || t.pitch > limit / t.n ||
            t.samples > limit / (t.n * t.pitch) || t.count > limit / t.n ||
            mul_length > limit - sum_length - 1) {
        fprintf(stderr, "sum benchmark configuration is too large\n");
        return EXIT_FAILURE;
    }
    // Corpus I/O and every allocation precede verification and timing, so no
    // task allocates inside its timer
    t.x = malloc(trials * t.n * sizeof(double));
    t.y = malloc(trials * t.n * sizeof(double));
    t.px = calloc(t.samples * t.n * t.pitch, sizeof(double));
    t.py = calloc(t.samples * t.n * t.pitch, sizeof(double));
    t.products = malloc(t.count * t.n * sizeof(double));
    t.gram = malloc(t.n * t.n * sizeof(double));
    t.out = malloc(t.samples * t.n * sizeof(double));
    t.scratch = malloc((mul_length + sum_length + 1) * sizeof(double));
    t.sign = malloc(t.n * t.n);
    if (!t.x || !t.y || !t.px || !t.py || !t.products || !t.gram || !t.out ||
            !t.scratch || !t.sign)
        return EXIT_FAILURE;
    t.mul_work.data = t.scratch;
    t.mul_work.length = mul_length;
    t.sum_work.data = t.scratch + mul_length;
    t.sum_work.length = sum_length;
    FILE* input = fopen(corpus, "rb");
    if (!input || fread(t.x, t.n * sizeof(double), trials, input) != trials ||
            fread(t.y, t.n * sizeof(double), trials, input) != trials ||
            fgetc(input) != EOF)
        return EXIT_FAILURE;
    fclose(input);
    for (size_t i = 0; i < t.n; i++)
        for (size_t j = 0; j < t.n; j++)
            t.sign[i * t.n + j] = (signed char)basis_sign(t.n, i, j);
    for (size_t s = 0; s < t.samples; s++)
        for (size_t p = 0; p < t.count; p++)
            for (size_t i = 0; i < t.n; i++) {
                t.px[(s * t.n + i) * t.pitch + p] = t.x[(s * t.count + p) * t.n + i];
                t.py[(s * t.n + i) * t.pitch + p] = t.y[(s * t.count + p) * t.n + i];
            }
    // The algorithm of automatic selection is timed at every dimension
    int enabled[TASKS] = {1, 1,
        t.n <= max_direct || selected == FASTCD_ALGORITHM_DIRECT, 1, blas != NULL};
    long double error = 0;
    if (!verify(&t, enabled, exact, selected, &error)) {
        fprintf(stderr, "independent verification failed\n");
        return EXIT_FAILURE;
    }
    int ids[TASKS];
    int tasks = 0;
    for (int task = 0; task < TASKS; task++)
        if (enabled[task]) ids[tasks++] = task;
    // Calibrate whole corpus passes independently for each task, so every
    // observation covers the same samples
    size_t passes[TASKS] = {1, 1, 1, 1, 1};
    for (int k = 0; k < tasks; k++) {
        for (size_t w = 0; w < warmup; w++)
            if (!run(&t, ids[k])) return EXIT_FAILURE;
        for (;;) {
            uint64_t start = now_ns();
            for (size_t r = 0; r < passes[k]; r++)
                if (!run(&t, ids[k])) return EXIT_FAILURE;
            uint64_t elapsed = now_ns() - start;
            if (elapsed >= minimum || !minimum) break;
            if (passes[k] > SIZE_MAX / 2 / trials) return EXIT_FAILURE;
            passes[k] *= 2;
        }
    }
    printf("{\"schema\":\"fastcd-sums-1\",\"dimension\":%zu,\"count\":%zu,"
           "\"trials\":%zu,\"samples\":%zu,\"pitch\":%zu,\"verified_samples\":%zu,"
           "\"exact_samples\":%zu,\"max_norm_scaled_error\":%.17g,"
           "\"default_sum_algorithm\":\"%s\",\"blas\":%s,\"warmup\":%zu,"
           "\"min_time_ns\":%" PRIu64 ",\"observations\":[",
            t.n, t.count, trials, t.samples, t.pitch, t.samples,
            exact < t.samples ? exact : t.samples, (double)error,
            fastcd_algorithm_string(selected), blas ? "true" : "false", warmup,
            minimum);
    // Rotate task positions between blocks and give each block its own stack
    // offset, with serialization, checksums and flushing after each timer and
    // never inside measured passes
    for (size_t block = 0; block < blocks; block++)
        for (int k = 0; k < tasks; k++) {
            int slot = (int)((block + (size_t)k) % (size_t)tasks);
            size_t offset = fastcd_block_stack_offset(block, blocks);
            Observation o = {&t, ids[slot], passes[slot], 0};
            if (!fastcd_at_stack_offset(offset, observe, &o)) return EXIT_FAILURE;
            printf("%s{\"block\":%zu,\"task\":\"%s\",\"elapsed_ns\":%" PRIu64
                   ",\"passes\":%zu,\"operations\":%zu,\"stack_offset\":%zu,"
                   "\"checksum\":\"%016" PRIx64 "\"}",
                    block || k ? "," : "", block, task_names[ids[slot]], o.elapsed,
                    passes[slot], passes[slot] * trials, offset, checksum(&t));
            fflush(stdout);
        }
    puts("]}");
    free(t.x);
    free(t.y);
    free(t.px);
    free(t.py);
    free(t.products);
    free(t.gram);
    free(t.out);
    free(t.scratch);
    free(t.sign);
    return EXIT_SUCCESS;
}
