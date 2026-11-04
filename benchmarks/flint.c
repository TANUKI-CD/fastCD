#if !defined(_WIN32)
#define _POSIX_C_SOURCE 200809L
#endif

#include "flint_schedule.h"
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(_WIN32)
#include <windows.h>
#endif

#ifndef FASTCD_BENCHMARK_BUILD_ID
#define FASTCD_BENCHMARK_BUILD_ID "unrecorded"
#endif

/* One field/workload per process. Both backends replay identical fixed
 * operands using the same calibrated number of complete corpus passes.
 * Setup, representation conversion, allocation, verification and checksums
 * stay outside timing. FLINT's own internal arithmetic work stays inside. */
typedef struct {
    FlintPlan flint;
    size_t trials, n, count, words, work_count, native_work_words, flint_count,
            initialized;
    int scalar, extension_initialized;
    uint32_t *storage, *x, *y, *out, *reference, *work;
    void *flint_storage, *fx, *fy, *fo, *fw;
} FlintTask;

/* ---------- Storage and independent verification ---------- */

// Check additions and products before allocating arrays or forming pointers.
static int task_init(
        FlintTask* t, uint32_t p, uint32_t d, size_t n, size_t trials, int scalar) {
    size_t level;
    if (!trials || !n || n > 1024 || fastcd_level(n, &level) ||
            fastcd_field_init(&t->flint.plan.field, p, d, NULL))
        return 0;
    FastCDField field = t->flint.plan.field;
    if (fastcd_ff_plan_init(&t->flint.plan, &field, level)) return 0;
    t->trials = trials;
    t->n = n;
    t->scalar = scalar;
    if (trials > SIZE_MAX / n) return 0;
    t->count = trials * n;
    if (t->count > SIZE_MAX / d) return 0;
    t->words = t->count * d;
    // Packed native scratch is measured in words and need not be divisible
    // by d. FLINT instead needs whole coefficient objects for the generic
    // recurrence; compute that bound independently of native dispatch.
    t->native_work_words = scalar ? 0 : t->flint.plan.workspace_words;
    t->work_count = scalar ? 0 : 32;
    if (!scalar)
        for (size_t k = n / 2; k >= 4; k /= 2)
            t->work_count += k == 4 ? 32 : 32 + 2 * k;
    t->flint.workspace_coefficients = t->work_count;
    size_t native_work = t->native_work_words;
    if (native_work > SIZE_MAX / sizeof(uint32_t) ||
            t->words > (SIZE_MAX / sizeof(uint32_t) - native_work) / 4)
        return 0;
    t->storage = calloc(4 * t->words + native_work, sizeof(uint32_t));
    if (!t->storage) return 0;
    t->x = t->storage;
    t->y = t->x + t->words;
    t->out = t->y + t->words;
    t->reference = t->out + t->words;
    t->work = t->reference + t->words;
    size_t item = d == 1 ? sizeof(ulong) : sizeof(fq_nmod_struct);
    if (t->work_count > SIZE_MAX / item ||
            t->count > (SIZE_MAX / item - t->work_count) / 3)
        return 0;
    t->flint_count = 3 * t->count + t->work_count;
    t->flint_storage = calloc(t->flint_count, item);
    if (!t->flint_storage) return 0;
    t->fx = t->flint_storage;
    t->fy = (unsigned char*)t->fx + t->count * item;
    t->fo = (unsigned char*)t->fy + t->count * item;
    t->fw = (unsigned char*)t->fo + t->count * item;
    nmod_init(&t->flint.prime, p);
    if (d > 1) {
        nmod_poly_t modulus;
        nmod_poly_init(modulus, p);
        for (uint32_t j = 0; j <= d; j++)
            nmod_poly_set_coeff_ui(modulus, (slong)j, field.modulus[j]);
        // An explicit polynomial prevents FLINT from choosing a different
        // presentation or silently benchmarking a different representation.
        fq_nmod_ctx_init_modulus(t->flint.extension, modulus, "a");
        nmod_poly_clear(modulus);
        t->extension_initialized = 1;
        for (size_t i = 0; i < t->flint_count; i++) {
            fq_nmod_struct* value = (fq_nmod_struct*)t->flint_storage + i;
            fq_nmod_init(value, t->flint.extension);
            nmod_poly_fit_length(value, (slong)(2 * d));
            t->initialized++;
        }
    }
    // The shared template also defines loop entries; this protocol calls
    // only its N-coefficient CD entry, including at the coefficient-field root.
    (void)bench_flint_prime_loop;
    (void)bench_flint_extension_loop;
    return 1;
}

// Every partially initialized state is safe to release after a setup error.
static void task_clear(FlintTask* t) {
    for (size_t i = 0; i < t->initialized; i++)
        fq_nmod_clear((fq_nmod_struct*)t->flint_storage + i, t->flint.extension);
    if (t->extension_initialized) fq_nmod_ctx_clear(t->flint.extension);
    free(t->flint_storage);
    free(t->storage);
}

// Convert once using public polynomial setters and preallocated capacities.
static int convert_inputs(FlintTask* t) {
    uint32_t d = t->flint.plan.field.degree, p = t->flint.plan.field.prime;
    for (size_t i = 0; i < t->words; i++)
        if (t->x[i] >= p || t->y[i] >= p) return 0;
    for (size_t i = 0; i < t->count; i++) {
        if (d == 1) {
            ((ulong*)t->fx)[i] = t->x[i];
            ((ulong*)t->fy)[i] = t->y[i];
        } else {
            fq_nmod_struct *x = (fq_nmod_struct*)t->fx + i,
                           *y = (fq_nmod_struct*)t->fy + i;
            for (uint32_t j = 0; j < d; j++) {
                nmod_poly_set_coeff_ui(x, (slong)j, t->x[i * d + j]);
                nmod_poly_set_coeff_ui(y, (slong)j, t->y[i * d + j]);
            }
        }
    }
    return 1;
}

// Derive each signed basis product from the defining CD doubling convention.
static int basis_sign(size_t n, size_t i, size_t j) {
    if (n == 1) return 1;
    size_t h = n / 2;
    if (i < h && j < h) return basis_sign(h, i, j);
    if (i < h) return basis_sign(h, j - h, i);
    if (j < h) return (j ? -1 : 1) * basis_sign(h, i - h, j);
    return (j == h ? -1 : 1) * basis_sign(h, j - h, i - h);
}

// Ordinary schoolbook multiplication and long division share no optimized
// scalar products, reduction helpers or evaluation maps with either backend.
static void reference_scalar(
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

// Verify every pair, without relying on agreement between the two timed paths.
static int make_reference(FlintTask* t) {
    const FastCDField* f = &t->flint.plan.field;
    size_t n = t->n, d = f->degree;
    int8_t* signs = malloc(n * n);
    if (!signs) return 0;
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++) signs[i * n + j] = (int8_t)basis_sign(n, i, j);
    memset(t->reference, 0, t->words * sizeof(uint32_t));
    for (size_t trial = 0; trial < t->trials; trial++) {
        const uint32_t *x = t->x + trial * n * d, *y = t->y + trial * n * d;
        uint32_t *out = t->reference + trial * n * d, scalar[12];
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < n; j++) {
                reference_scalar(scalar, x + i * d, y + j * d, f);
                for (size_t k = 0; k < d; k++) {
                    size_t pos = (i ^ j) * d + k;
                    uint32_t value =
                            signs[i * n + j] > 0 ? scalar[k] : f->prime - scalar[k];
                    out[pos] = (uint32_t)(((uint64_t)out[pos] + value) % f->prime);
                }
            }
    }
    free(signs);
    return 1;
}

/* ---------- Timed public calls ---------- */

// Place one compiler-only fence at each whole-corpus boundary for both
// backends. Repeated inline prime products must retain their output stores;
// this emits no hardware fence and adds no per-product synchronization.
static int replay(FlintTask* t, int flint, size_t passes) {
    size_t d = t->flint.plan.field.degree, stride = t->n * d;
    for (size_t r = 0; r < passes; r++) {
        if (!flint) {
            if (t->scalar) {
                for (size_t i = 0; i < t->trials; i++)
                    fastcd_ff_field_mul(t->out + i * d, t->x + i * d, t->y + i * d,
                            &t->flint.plan.field);
            } else {
                for (size_t i = 0; i < t->trials; i++)
                    if (fastcd_ff_cd_mul(t->out + i * stride, t->x + i * stride,
                                t->y + i * stride, &t->flint.plan, t->work,
                                t->native_work_words))
                        return 0;
            }
        } else if (t->scalar) {
            if (d == 1) {
                for (size_t i = 0; i < t->trials; i++)
                    ((ulong*)t->fo)[i] = nmod_mul(
                            ((ulong*)t->fx)[i], ((ulong*)t->fy)[i], t->flint.prime);
            } else {
                for (size_t i = 0; i < t->trials; i++)
                    fq_nmod_mul((fq_nmod_struct*)t->fo + i, (fq_nmod_struct*)t->fx + i,
                            (fq_nmod_struct*)t->fy + i, t->flint.extension);
            }
        } else {
            size_t item = d == 1 ? sizeof(ulong) : sizeof(fq_nmod_struct);
            for (size_t i = 0; i < t->trials; i++) {
                size_t offset = i * t->n * item;
                if (bench_flint_cd((unsigned char*)t->fo + offset,
                            (const unsigned char*)t->fx + offset,
                            (const unsigned char*)t->fy + offset, &t->flint, t->fw,
                            t->work_count))
                    return 0;
            }
        }
        atomic_signal_fence(memory_order_seq_cst);
    }
    return 1;
}

// Check the actual final timed output, and hash canonical uint32 coefficients
// identically despite the backends' different physical element layouts.
static int check_output(const FlintTask* t, int flint, uint64_t* checksum) {
    size_t d = t->flint.plan.field.degree;
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < t->words; i++) {
        ulong value = !flint ? t->out[i]
                : d == 1     ? ((const ulong*)t->fo)[i]
                         : nmod_poly_get_coeff_ui((const fq_nmod_struct*)t->fo + i / d,
                                   (slong)(i % d));
        if (value != t->reference[i]) {
            fprintf(stderr,
                    "%s output mismatch at word %zu: got %" PRIu64 ", expected %" PRIu32
                    "\n",
                    flint ? "FLINT" : "fastCD", i, (uint64_t)value, t->reference[i]);
            return 0;
        }
        hash = (hash ^ (uint32_t)value) * UINT64_C(1099511628211);
    }
    *checksum = hash;
    return 1;
}

// Use the same high-resolution monotonic wall clock as the real benchmark.
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

// Retain thread CPU time as a diagnostic where the platform exposes it.
static uint64_t thread_ns(void) {
#if defined(CLOCK_THREAD_CPUTIME_ID)
    struct timespec value;
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) == 0)
        return (uint64_t)value.tv_sec * UINT64_C(1000000000) + (uint64_t)value.tv_nsec;
#endif
    return 0;
}

// Increase a common pass count until even the faster backend reaches the
// target. Paired observations therefore contain exactly the same products.
static int calibrate(FlintTask* t, size_t warmup, uint64_t minimum, size_t* passes) {
    for (int b = 0; b < 2; b++)
        if (!replay(t, b, warmup)) return 0;
    for (*passes = 1;; *passes *= 2) {
        int enough = 1;
        for (int b = 0; b < 2; b++) {
            uint64_t start = now_ns();
            if (!replay(t, b, *passes)) return 0;
            if (now_ns() - start < minimum) enough = 0;
        }
        if (enough) return 1;
        if (*passes > SIZE_MAX / 2 / t->trials) return 0;
    }
}

/* ---------- CLI and executable correctness checks ---------- */

// Reject signed spellings, overflow and trailing characters before narrowing.
static int parse_size(const char* text, size_t* value) {
    char* end;
    if (!text[0] || text[0] == '-' || text[0] == '+') return 0;
    errno = 0;
    uintmax_t parsed = strtoumax(text, &end, 10);
    if (errno || *end || parsed > SIZE_MAX) return 0;
    *value = (size_t)parsed;
    return 1;
}

// Accept the same fractional millisecond targets as the real protocol.
static int parse_minimum(const char* text, uint64_t* value) {
    char* end;
    errno = 0;
    double ms = strtod(text, &end);
    if (errno || end == text || *end || !isfinite(ms) || ms < 0 ||
            ms > (double)(UINT64_MAX / 1000000))
        return 0;
    *value = (uint64_t)(ms * 1e6);
    return 1;
}

// Exercise all supported degrees, both small and large primes, zero/unit/
// boundary/dense operands, and every level where the local maps change shape.
static int self_test(void) {
    for (uint32_t group = 0; group < 3; group++) {
        uint32_t p = group == 0 ? 3 : group == 1 ? 13 : 2147483647;
        for (uint32_t d = 1; d <= 12; d++) {
            for (size_t n = 1; n <= 32; n *= 2) {
                FlintTask t = {0};
                if (!task_init(&t, p, d, n, 5, n == 1)) {
                    fprintf(stderr,
                            "FLINT self-test setup failed: p=%" PRIu32 ", d=%" PRIu32
                            ", N=%zu\n",
                            p, d, n);
                    task_clear(&t);
                    return 0;
                }
                for (size_t i = 0; i < t.words; i++) {
                    size_t trial = i / (n * d), j = i % (n * d);
                    t.x[i] = trial == 0  ? 0
                            : trial == 1 ? (uint32_t)(j == 0)
                            : trial == 2 ? p - 1
                                         : (uint32_t)((i * 17 + 3) % p);
                    t.y[i] = trial == 0  ? 0
                            : trial == 1 ? (uint32_t)(j == 0)
                            : trial == 2 ? p - 1
                                         : (uint32_t)((i * 31 + 7) % p);
                }
                uint64_t checksum;
                int ok = convert_inputs(&t) && make_reference(&t);
                for (int b = 0; b < 2 && ok; b++)
                    ok = replay(&t, b, 2) && check_output(&t, b, &checksum);
                if (n > 1 && ok) {
                    size_t item = d == 1 ? sizeof(ulong) : sizeof(fq_nmod_struct);
                    ok = bench_flint_cd(t.fx, t.fx, t.fy, &t.flint, t.fw,
                                 t.work_count) == FASTCD_FF_OVERLAPPING_BUFFERS &&
                            bench_flint_cd(t.fo, t.fx, t.fy, &t.flint, t.fw,
                                    t.work_count - 1) ==
                                    FASTCD_FF_INSUFFICIENT_WORKSPACE &&
                            bench_flint_cd(t.fo, t.fx, t.fy, &t.flint,
                                    (unsigned char*)t.fx + item,
                                    t.work_count) == FASTCD_FF_OVERLAPPING_BUFFERS;
                }
                task_clear(&t);
                if (!ok) {
                    fprintf(stderr,
                            "FLINT self-test failed: p=%" PRIu32 ", d=%" PRIu32
                            ", N=%zu\n",
                            p, d, n);
                    return 0;
                }
            }
        }
    }
    puts("FLINT agreement: all degrees 1..12, three primes, scalar/CD levels 0..5 passed");
    return 1;
}

// Run one independently verified field/workload and emit paired raw observations.
int main(int argc, char** argv) {
    flint_set_num_threads(1);
    if (argc == 2 && !strcmp(argv[1], "--metadata")) {
        printf("{\"schema\":\"fastcd-flint-1\",\"build_id\":\"%s\","
               "\"version\":\"%s\",\"flint_compile_version\":\"%s\","
               "\"flint_runtime_version\":\"%s\",\"flint_threads\":1,"
               "\"prime_api\":\"nmod_mul\",\"extension_api\":\"fq_nmod_mul\","
               "\"cd_schedule\":\"src/finite/schedule_impl.h\","
               "\"clock\":\"monotonic\"}\n",
                FASTCD_BENCHMARK_BUILD_ID, fastcd_version_string(), FLINT_VERSION,
                flint_version);
        return EXIT_SUCCESS;
    }
    if (argc == 2 && !strcmp(argv[1], "--self-test")) {
        int ok = self_test();
        flint_cleanup();
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    const char *path = NULL, *kind = "scalar";
    size_t p = 0, d = 1, n = 1, trials = 10000, blocks = 24, warmup = 2;
    uint64_t minimum = UINT64_C(20000000);
    size_t order_seed = 20261104;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 == argc) goto usage;
        const char *key = argv[i], *value = argv[i + 1];
        size_t* target = NULL;
        if (!strcmp(key, "--corpus"))
            path = value;
        else if (!strcmp(key, "--kind"))
            kind = value;
        else if (!strcmp(key, "--char"))
            target = &p;
        else if (!strcmp(key, "--deg"))
            target = &d;
        else if (!strcmp(key, "--dimension"))
            target = &n;
        else if (!strcmp(key, "--trials"))
            target = &trials;
        else if (!strcmp(key, "--blocks"))
            target = &blocks;
        else if (!strcmp(key, "--warmup"))
            target = &warmup;
        else if (!strcmp(key, "--min-time-ms")) {
            if (!parse_minimum(value, &minimum)) goto usage;
        } else if (!strcmp(key, "--seed"))
            target = &order_seed;
        else
            goto usage;
        if (target && !parse_size(value, target)) goto usage;
    }
    if (!path || p < 3 || p > INT32_MAX || !d || d > 12 || !trials || blocks < 2 ||
            blocks % 2 || (strcmp(kind, "scalar") && strcmp(kind, "cd")) ||
            (!strcmp(kind, "scalar") && n != 1))
        goto usage;
    FlintTask t = {0};
    int ok =
            task_init(&t, (uint32_t)p, (uint32_t)d, n, trials, !strcmp(kind, "scalar"));
    FILE* input = ok ? fopen(path, "rb") : NULL;
    if (!input || fread(t.x, sizeof(uint32_t), t.words, input) != t.words ||
            fread(t.y, sizeof(uint32_t), t.words, input) != t.words ||
            fgetc(input) != EOF)
        ok = 0;
    if (input) fclose(input);
    uint64_t checksum;
    ok = ok && convert_inputs(&t) && make_reference(&t);
    for (int b = 0; b < 2 && ok; b++)
        ok = replay(&t, b, 1) && check_output(&t, b, &checksum);
    size_t passes = 1;
    ok = ok && calibrate(&t, warmup, minimum, &passes);
    if (!ok) {
        fprintf(stderr,
                "invalid corpus/field, allocation failure, or independent verification failed\n");
        task_clear(&t);
        return EXIT_FAILURE;
    }
    printf("{\"schema\":\"fastcd-flint-1\",\"kind\":\"%s\",\"char\":%zu,"
           "\"deg\":%zu,\"dimension\":%zu,\"trials\":%zu,\"verified\":%zu,"
           "\"passes\":%zu,\"warmup\":%zu,\"min_time_ns\":%" PRIu64 ",\"modulus\":[",
            kind, p, d, n, trials, trials, passes, warmup, minimum);
    for (size_t j = 0; j <= d; j++)
        printf("%s%" PRIu32, j ? "," : "", t.flint.plan.field.modulus[j]);
    printf("],\"native_workspace_bytes\":%zu,\"flint_workspace_elements\":%zu,"
           "\"observations\":[",
            t.native_work_words * sizeof(uint32_t), t.work_count);
    // Alternate the first backend across paired blocks for balanced positions.
    // Every final output is checked after its timer, before accepting a sample.
    for (size_t block = 0; block < blocks && ok; block++)
        for (size_t order = 0; order < 2 && ok; order++) {
            int b = (int)((block + order + (order_seed & 1u)) & 1u);
            uint64_t start = now_ns(), cpu_start = thread_ns();
            ok = replay(&t, b, passes);
            uint64_t cpu = thread_ns() - cpu_start, elapsed = now_ns() - start;
            ok = ok && elapsed && check_output(&t, b, &checksum);
            if (!ok) break;
            printf("%s{\"block\":%zu,\"order\":%zu,\"backend\":\"%s\","
                   "\"elapsed_ns\":%" PRIu64 ",\"thread_cpu_ns\":%" PRIu64
                   ",\"passes\":%zu,\"operations\":%zu,\"verified_pairs\":%zu,"
                   "\"checksum\":\"%016" PRIx64 "\"}",
                    block || order ? "," : "", block, order, b ? "flint" : "fastcd",
                    elapsed, cpu, passes, passes * trials, trials, checksum);
            fflush(stdout);
        }
    if (ok)
        puts("]}");
    else
        fprintf(stderr,
                "timed output verification failed; no valid result document produced\n");
    task_clear(&t);
    flint_cleanup();
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
usage:
    fprintf(stderr,
            "usage: %s --metadata | --self-test | --corpus FILE --char P "
            "[--deg D] [--kind scalar|cd] [--dimension N] [--trials T] "
            "[--blocks B] [--warmup W] [--min-time-ms MS] [--seed S]\n",
            argv[0]);
    return EXIT_FAILURE;
}
