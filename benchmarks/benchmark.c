#if !defined(_WIN32)
#define _POSIX_C_SOURCE 200809L
#endif

#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#include <windows.h>
#endif

#include "fastcd/fastcd.h"
#include "reference_cd.h"

#ifndef FASTCD_BENCHMARK_BUILD_ID
#define FASTCD_BENCHMARK_BUILD_ID "unrecorded"
#endif

/*
 * fastCD multiplication benchmark runner.  Each invocation times one
 * algorithm at one dimension over one corpus and prints a single JSON
 * document.  Corpus I/O, allocation, warmup, calibration, and verification
 * are outside the recorded timing; each observation times whole passes over
 * the corpus, either as one validated batch call (mode batch) or as one
 * validated single-product call per pair (mode single).
 * benchmarks/benchmark.py drives this program and aggregates its output.
 */

typedef struct BenchmarkConfig {
    const char* corpus_path;
    size_t dimension;
    size_t trials;
    FastCDAlgorithm algorithm;
    const char* algorithm_name;
    size_t repeats;
    size_t warmup;
    double min_time_ms;
    size_t verify_trials;
    bool verify_explicit;
    bool single;              /* time one validated call per product */
} BenchmarkConfig;

typedef struct Observation {
    uint64_t elapsed_ns;
    uint64_t operations;
    uint64_t checksum;
} Observation;

typedef struct Verification {
    size_t verified;
    size_t mismatched;
    long double max_absolute_error;
    long double max_norm_scaled_error;
    bool finite;
} Verification;

/* ---------- Helpers ---------- */

// Print usage
static void print_usage(const char* program) {
    printf("Usage:\n"
           "  %s --metadata\n"
           "  %s --corpus FILE --dimension N --trials T --algorithm NAME\n"
           "     [--mode batch|single] [--repeats R] [--warmup W]"
           " [--min-time-ms X] [--verify-trials V]\n\n"
           "FILE holds 3*T*N native doubles: the x block, the y block, and an"
           " unused\nthird block.  NAME is direct, cariow_cariowa (alias cc),"
           " quasilinear, or auto.\nMode batch (default) times one validated"
           " batch call per corpus pass; mode single\ntimes T validated"
           " single-product calls per pass.  Unsupported dimensions print\na"
           " JSON document with status \"unavailable\".\n", program, program);
}

// Parse a nonnegative size
static bool parse_size(const char* text, size_t* value) {
    char* end = NULL;
    uintmax_t parsed;

    if (text == NULL || *text == '\0' || *text == '-' || *text == '+') {
        return false;
    }
    errno = 0;
    parsed = strtoumax(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed > SIZE_MAX) {
        return false;
    }
    *value = (size_t)parsed;
    return true;
}

// Parse a finite nonnegative double
static bool parse_double(const char* text, double* value) {
    char* end = NULL;
    double parsed;

    if (text == NULL || *text == '\0') return false;
    errno = 0;
    parsed = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !isfinite(parsed)
            || parsed < 0.0) {
        return false;
    }
    *value = parsed;
    return true;
}

// Resolve an algorithm name
static bool parse_algorithm(const char* text, FastCDAlgorithm* algorithm) {
    if (strcmp(text, "auto") == 0) {
        *algorithm = FASTCD_ALGORITHM_AUTO;
    } else if (strcmp(text, "direct") == 0) {
        *algorithm = FASTCD_ALGORITHM_DIRECT;
    } else if (strcmp(text, "cariow_cariowa") == 0 || strcmp(text, "cc") == 0) {
        *algorithm = FASTCD_ALGORITHM_CARIOW_CARIOWA;
    } else if (strcmp(text, "quasilinear") == 0) {
        *algorithm = FASTCD_ALGORITHM_QUASILINEAR;
    } else {
        return false;
    }
    return true;
}

// Read the monotonic clock in nanoseconds
static uint64_t now_ns(void) {
#if defined(_WIN32)
    LARGE_INTEGER counter;
    LARGE_INTEGER frequency;

    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (uint64_t)((long double)counter.QuadPart * 1e9L
            / (long double)frequency.QuadPart);
#else
    struct timespec value;
#if defined(CLOCK_MONOTONIC_RAW)
    clock_gettime(CLOCK_MONOTONIC_RAW, &value);
#else
    clock_gettime(CLOCK_MONOTONIC, &value);
#endif
    return (uint64_t)value.tv_sec * UINT64_C(1000000000)
        + (uint64_t)value.tv_nsec;
#endif
}

// Return the clock name recorded in metadata
static const char* clock_name(void) {
#if defined(_WIN32)
    return "QueryPerformanceCounter";
#elif defined(CLOCK_MONOTONIC_RAW)
    return "CLOCK_MONOTONIC_RAW";
#else
    return "CLOCK_MONOTONIC";
#endif
}

// Print a JSON string
static void print_json_string(const char* text) {
    putchar('"');
    for (const unsigned char* c = (const unsigned char*)text; *c; c++) {
        if (*c == '"' || *c == '\\') {
            putchar('\\');
            putchar((int)*c);
        } else if (*c < 0x20) {
            printf("\\u%04x", (unsigned)*c);
        } else {
            putchar((int)*c);
        }
    }
    putchar('"');
}

// Print compiler metadata as JSON
static void print_compiler_json(void) {
    printf("{\"name\":");
#if defined(__clang__)
    print_json_string("clang");
    printf(",\"version\":");
    print_json_string(__clang_version__);
#elif defined(__GNUC__)
    print_json_string("gcc");
    printf(",\"version\":");
    print_json_string(__VERSION__);
#elif defined(_MSC_VER)
    print_json_string("msvc");
    printf(",\"version\":\"%d\"", (int)_MSC_VER);
#else
    print_json_string("unknown");
    printf(",\"version\":\"unknown\"");
#endif
    printf(",\"pointer_bits\":%zu}", sizeof(void*) * 8u);
}

// Print a long double as JSON or null when nonfinite
static void print_json_long_double(long double value) {
    if (isfinite(value)) {
        printf("%.21Lg", value);
    } else {
        printf("null");
    }
}

/* ---------- Corpus and verification ---------- */

// Read exactly 3*trials*dimension doubles
static double* read_corpus(const BenchmarkConfig* config) {
    size_t elements;
    double* values;
    FILE* file;
    size_t read_count;

    if (config->trials > SIZE_MAX / 3 / config->dimension / sizeof(double)) {
        return NULL;
    }
    elements = 3 * config->trials * config->dimension;
    values = malloc(elements * sizeof(double));
    if (values == NULL) return NULL;
    file = fopen(config->corpus_path, "rb");
    if (file == NULL) {
        free(values);
        return NULL;
    }
    read_count = fread(values, sizeof(double), elements, file);
    if (read_count != elements || fgetc(file) != EOF) {
        fclose(file);
        free(values);
        return NULL;
    }
    fclose(file);
    return values;
}

// Compute a Euclidean norm in long double without squaring extremes
static long double operand_norm(const double* values, size_t dimension) {
    long double norm = 0.0L;

    for (size_t i = 0; i < dimension; i++) {
        norm = hypotl(norm, (long double)values[i]);
    }
    return norm;
}

// Compare timed outputs with the independent oracle
static bool verify(const BenchmarkConfig* config, const double* x,
        const double* y, const double* out, Verification* result) {
    const size_t dimension = config->dimension;
    const long double tolerance = 128.0L * (long double)DBL_EPSILON
        * (long double)dimension;
    double* expected = malloc(dimension * sizeof(double));

    memset(result, 0, sizeof(*result));
    result->finite = true;
    if (expected == NULL) return false;
    for (size_t trial = 0; trial < config->verify_trials; trial++) {
        const double* xi = x + trial * dimension;
        const double* yi = y + trial * dimension;
        const double* actual = out + trial * dimension;
        long double scale = operand_norm(xi, dimension) * operand_norm(yi, dimension);
        bool mismatch = false;

        if (reference_cd_mul_f64(expected, xi, yi, dimension)
                != FASTCD_SUCCESS) {
            free(expected);
            return false;
        }
        for (size_t k = 0; k < dimension; k++) {
            long double difference;
            long double scaled;

            if (!isfinite(actual[k]) || !isfinite(expected[k])) {
                result->finite = false;
                mismatch = true;
                continue;
            }
            difference = fabsl((long double)actual[k] - (long double)expected[k]);
            scaled = scale == 0.0L ? difference : difference / scale;
            if (difference > result->max_absolute_error) {
                result->max_absolute_error = difference;
            }
            if (scaled > result->max_norm_scaled_error) {
                result->max_norm_scaled_error = scaled;
            }
            if (scale == 0.0L ? difference != 0.0L : scaled > tolerance) {
                mismatch = true;
            }
        }
        result->verified++;
        if (mismatch) result->mismatched++;
    }
    free(expected);
    return true;
}

/* ---------- Timing ---------- */

typedef struct Task {
    const BenchmarkConfig* config;
    const double* x;
    const double* y;
    double* out;
    FastCDWorkspaceF64 workspace;
    const FastCDWorkspaceF64* workspace_pointer;
} Task;

// Run one pass over the corpus: one batch call, or one call per product
static FastCDStatus run_pass(const Task* task) {
    const size_t n = task->config->dimension;

    if (task->config->single) {
        for (size_t i = 0; i < task->config->trials; i++) {
            FastCDStatus status = fastcd_mul_with_algorithm_f64(
                    task->out + i * n, task->x + i * n, task->y + i * n, n,
                    task->config->algorithm, task->workspace_pointer);

            if (status != FASTCD_SUCCESS) return status;
        }
        return FASTCD_SUCCESS;
    }
    return fastcd_mul_batch_with_algorithm_f64(task->out, n, task->x, n,
            task->y, n, task->config->trials, n, task->config->algorithm,
            task->workspace_pointer);
}

// Time a number of passes
static bool time_passes(const Task* task, size_t passes, uint64_t* elapsed) {
    uint64_t start = now_ns();

    for (size_t pass = 0; pass < passes; pass++) {
        if (run_pass(task) != FASTCD_SUCCESS) return false;
    }
    *elapsed = now_ns() - start;
    return true;
}

// Fold the output bits into a checksum so results are observable
static uint64_t checksum(const Task* task) {
    uint64_t hash = UINT64_C(14695981039346656037);
    size_t elements = task->config->trials * task->config->dimension;

    for (size_t i = 0; i < elements; i++) {
        uint64_t bits;
        memcpy(&bits, &task->out[i], sizeof(bits));
        hash = (hash ^ bits) * UINT64_C(1099511628211);
    }
    return hash;
}

/* ---------- Reports ---------- */

// Print the shared metadata fields
static void print_metadata_fields(bool single) {
    printf("\"runner\":\"fastcd_benchmark\",\"library_version\":");
    print_json_string(fastcd_version_string());
    printf(",\"build_id\":");
    print_json_string(FASTCD_BENCHMARK_BUILD_ID);
    printf(",\"compiler\":");
    print_compiler_json();
    printf(",\"clock\":");
    print_json_string(clock_name());
    printf(",\"timing_boundary\":");
    print_json_string(single
            ? "one validated fastcd_mul_with_algorithm_f64 call per product "
              "per corpus pass; preallocated buffers"
            : "one validated fastcd_mul_batch_with_algorithm_f64 call per "
              "corpus pass; preallocated buffers");
    printf(",\"timing_excludes\":[\"corpus_io\",\"allocation\",\"warmup\","
           "\"calibration\",\"verification\",\"checksum\"]");
}

// Print the standalone metadata document
static void print_metadata(void) {
    printf("{\"schema\":\"fastcd-benchmark-1\",");
    print_metadata_fields(false);
    printf(",\"algorithms\":[\"direct\",\"cariow_cariowa\",\"quasilinear\","
           "\"auto\"],\"modes\":[\"batch\",\"single\"],"
           "\"cariow_cariowa_max_dimension\":%zu}\n",
            FASTCD_CARIOW_CARIOWA_MAX_DIMENSION);
}

// Print the common configuration fields
static void print_config(const BenchmarkConfig* config,
        FastCDAlgorithm resolved) {
    FastCDAlgorithm default_algorithm = FASTCD_ALGORITHM_AUTO;
    FastCDAlgorithm default_batch_algorithm = FASTCD_ALGORITHM_AUTO;

    printf("\"dimension\":%zu,\"trials\":%zu,\"mode\":",
            config->dimension, config->trials);
    print_json_string(config->single ? "single" : "batch");
    printf(",\"algorithm\":");
    print_json_string(config->algorithm_name);
    printf(",\"resolved_algorithm\":");
    print_json_string(fastcd_algorithm_string(resolved));
    fastcd_default_algorithm(config->dimension, &default_algorithm);
    fastcd_default_batch_algorithm(config->dimension, &default_batch_algorithm);
    printf(",\"default_algorithm\":");
    print_json_string(fastcd_algorithm_string(default_algorithm));
    printf(",\"default_batch_algorithm\":");
    print_json_string(fastcd_algorithm_string(default_batch_algorithm));
    printf(",\"repeats\":%zu,\"warmup\":%zu,\"min_time_ms\":%.17g,"
           "\"verify_trials\":%zu", config->repeats, config->warmup,
            config->min_time_ms, config->verify_trials);
}

/* ---------- Main ---------- */

int main(int argc, char** argv) {
    BenchmarkConfig config = {NULL, 0, 0, FASTCD_ALGORITHM_AUTO, "auto",
        5, 1, 20.0, 0, false, false};
    FastCDAlgorithm resolved = FASTCD_ALGORITHM_AUTO;
    Task task;
    Observation* observations = NULL;
    Verification verification;
    double* corpus = NULL;
    double* scratch = NULL;
    size_t workspace_length = 0;
    size_t passes = 1;
    uint64_t elapsed = 0;
    bool seen_corpus = false;
    bool seen_dimension = false;
    bool seen_trials = false;
    bool seen_algorithm = false;
    FastCDStatus status;

    if (argc == 2 && strcmp(argv[1], "--metadata") == 0) {
        print_metadata();
        return EXIT_SUCCESS;
    }
    if (argc == 2 && (strcmp(argv[1], "--help") == 0
                || strcmp(argv[1], "-h") == 0)) {
        print_usage(argv[0]);
        return EXIT_SUCCESS;
    }
    for (int index = 1; index < argc; index += 2) {
        const char* option = argv[index];
        const char* value = index + 1 < argc ? argv[index + 1] : NULL;
        bool ok = value != NULL;

        if (!ok) {
            fprintf(stderr, "%s requires a value\n", option);
            return EXIT_FAILURE;
        }
        if (strcmp(option, "--corpus") == 0) {
            config.corpus_path = value;
            seen_corpus = true;
        } else if (strcmp(option, "--dimension") == 0) {
            ok = parse_size(value, &config.dimension)
                && fastcd_valid_dimension(config.dimension);
            seen_dimension = true;
        } else if (strcmp(option, "--trials") == 0) {
            ok = parse_size(value, &config.trials) && config.trials > 0;
            seen_trials = true;
        } else if (strcmp(option, "--algorithm") == 0) {
            ok = parse_algorithm(value, &config.algorithm);
            config.algorithm_name = fastcd_algorithm_string(config.algorithm);
            seen_algorithm = true;
        } else if (strcmp(option, "--repeats") == 0) {
            ok = parse_size(value, &config.repeats) && config.repeats > 0;
        } else if (strcmp(option, "--warmup") == 0) {
            ok = parse_size(value, &config.warmup);
        } else if (strcmp(option, "--min-time-ms") == 0) {
            ok = parse_double(value, &config.min_time_ms);
        } else if (strcmp(option, "--verify-trials") == 0) {
            ok = parse_size(value, &config.verify_trials);
            config.verify_explicit = true;
        } else if (strcmp(option, "--mode") == 0) {
            if (strcmp(value, "batch") == 0) {
                config.single = false;
            } else if (strcmp(value, "single") == 0) {
                config.single = true;
            } else {
                ok = false;
            }
        } else {
            fprintf(stderr, "unknown option: %s\n", option);
            return EXIT_FAILURE;
        }
        if (!ok) {
            fprintf(stderr, "invalid value for %s\n", option);
            return EXIT_FAILURE;
        }
    }
    if (!seen_corpus || !seen_dimension || !seen_trials || !seen_algorithm) {
        fprintf(stderr, "--corpus, --dimension, --trials, and --algorithm are "
                "required\n");
        return EXIT_FAILURE;
    }
    if (!config.verify_explicit) config.verify_trials = config.trials;
    if (config.verify_trials > config.trials) config.verify_trials = config.trials;
    if (sizeof(double) != 8 || DBL_MANT_DIG != 53) {
        fprintf(stderr, "the corpus requires an IEEE binary64 double\n");
        return EXIT_FAILURE;
    }

    if (!fastcd_algorithm_supports_dimension(config.algorithm,
                config.dimension)) {
        printf("{\"schema\":\"fastcd-benchmark-1\",\"status\":\"unavailable\",");
        print_config(&config, config.algorithm);
        printf(",\"reason\":\"algorithm does not support this dimension\"}\n");
        return EXIT_SUCCESS;
    }
    if (config.algorithm != FASTCD_ALGORITHM_AUTO) {
        resolved = config.algorithm;
    } else if (config.single || config.trials < 2) {
        fastcd_default_algorithm(config.dimension, &resolved);
    } else {
        fastcd_default_batch_algorithm(config.dimension, &resolved);
    }

    corpus = read_corpus(&config);
    if (corpus == NULL) {
        fprintf(stderr, "failed to read an exact 3*trials*dimension corpus "
                "from %s\n", config.corpus_path);
        return EXIT_FAILURE;
    }
    task.config = &config;
    task.x = corpus;
    task.y = corpus + config.trials * config.dimension;
    task.out = malloc(config.trials * config.dimension * sizeof(double));
    status = fastcd_mul_algorithm_workspace_size(config.dimension,
            config.algorithm, &workspace_length);
    if (task.out == NULL || status != FASTCD_SUCCESS) {
        fprintf(stderr, "allocation or workspace query failed\n");
        free(task.out);
        free(corpus);
        return EXIT_FAILURE;
    }
    if (workspace_length != 0) {
        scratch = malloc(workspace_length * sizeof(double));
        if (scratch == NULL) {
            fprintf(stderr, "workspace allocation failed\n");
            free(task.out);
            free(corpus);
            return EXIT_FAILURE;
        }
    }
    task.workspace.data = scratch;
    task.workspace.length = workspace_length;
    task.workspace_pointer = workspace_length == 0 ? NULL : &task.workspace;

    /* Warmup, then calibrate the pass count to the minimum observation time. */
    for (size_t pass = 0; pass < config.warmup; pass++) {
        if (run_pass(&task) != FASTCD_SUCCESS) {
            fprintf(stderr, "warmup multiplication failed\n");
            goto fail;
        }
    }
    for (int attempt = 0; attempt < 40; attempt++) {
        double target = config.min_time_ms * 1e6;
        double proposed;

        if (!time_passes(&task, passes, &elapsed)) goto fail;
        if ((double)elapsed >= target) break;
        proposed = elapsed == 0 ? (double)passes * 10.0
            : ceil((double)passes * target / (double)elapsed * 1.1);
        if (proposed <= (double)passes) proposed = (double)passes + 1.0;
        if (proposed > 1e12) goto fail;
        passes = (size_t)proposed;
    }

    observations = calloc(config.repeats, sizeof(*observations));
    if (observations == NULL) goto fail;
    for (size_t repeat = 0; repeat < config.repeats; repeat++) {
        if (!time_passes(&task, passes, &observations[repeat].elapsed_ns)) {
            goto fail;
        }
        observations[repeat].operations = (uint64_t)passes
            * (uint64_t)config.trials;
        observations[repeat].checksum = checksum(&task);
    }
    if (!verify(&config, task.x, task.y, task.out, &verification)) goto fail;

    printf("{\"schema\":\"fastcd-benchmark-1\",\"status\":\"%s\",",
            verification.mismatched == 0 ? "ok" : "verification_failure");
    print_config(&config, resolved);
    putchar(',');
    print_metadata_fields(config.single);
    printf(",\"workspace_doubles\":%zu,\"corpus_passes_per_observation\":%zu,"
           "\"observations\":[", workspace_length, passes);
    for (size_t repeat = 0; repeat < config.repeats; repeat++) {
        if (repeat != 0) putchar(',');
        printf("{\"elapsed_ns\":%" PRIu64 ",\"operations\":%" PRIu64
               ",\"ns_per_operation\":%.17g,\"checksum\":%" PRIu64 "}",
                observations[repeat].elapsed_ns,
                observations[repeat].operations,
                (double)observations[repeat].elapsed_ns
                    / (double)observations[repeat].operations,
                observations[repeat].checksum);
    }
    printf("],\"verification\":{\"oracle\":\"independent defining "
           "recursion\",\"verified_trials\":%zu,\"mismatched_trials\":%zu,"
           "\"finite\":%s,\"tolerance\":\"128*DBL_EPSILON*dimension*norm(x)"
           "*norm(y)\",\"max_absolute_error\":",
            verification.verified, verification.mismatched,
            verification.finite ? "true" : "false");
    print_json_long_double(verification.max_absolute_error);
    printf(",\"max_norm_scaled_error\":");
    print_json_long_double(verification.max_norm_scaled_error);
    printf("}}\n");

    free(observations);
    free(scratch);
    free(task.out);
    free(corpus);
    return verification.mismatched == 0 ? EXIT_SUCCESS : EXIT_FAILURE;

fail:
    fprintf(stderr, "benchmark task failed\n");
    free(observations);
    free(scratch);
    free(task.out);
    free(corpus);
    return EXIT_FAILURE;
}
