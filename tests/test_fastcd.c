/*
 * fastCD automated test suite.
 *
 * Every arithmetic result is checked against tests/reference_cd.c, an
 * independent literal evaluation of the Cayley--Dickson doubling formula, and
 * every public routine is exercised through its documented validation,
 * aliasing, and workspace contracts.  "Exact" comparisons use ==, so they are
 * bitwise except that +0.0 and -0.0 compare equal (mathematically equivalent
 * operation orders legitimately differ in the sign of an exact zero); bitwise
 * comparisons are used where two library calls must agree identically.
 */

#include <ctype.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "fastcd/fastcd.h"
#include "reference_cd.h"

/* ---------- Tiny test harness ---------- */

#define FAILURE_REPORT_LIMIT ((size_t)200)

static size_t tests_run = 0;
static size_t tests_failed = 0;

#define CHECK(condition, ...) do {                                         \
    tests_run++;                                                           \
    if (!(condition)) {                                                    \
        tests_failed++;                                                    \
        if (tests_failed <= FAILURE_REPORT_LIMIT) {                        \
            fprintf(stderr, "  FAIL line %d: ", __LINE__);                \
            fprintf(stderr, __VA_ARGS__);                                  \
            fputc('\n', stderr);                                           \
        }                                                                  \
    }                                                                      \
} while (0)

/* ---------- Test helpers ---------- */

typedef FastCDStatus (*WorkspaceQuery)(size_t, size_t*);
typedef FastCDStatus (*BinaryOperation)(double*, const double*,
        const double*, size_t, const FastCDWorkspaceF64*);
typedef FastCDStatus (*TernaryOperation)(double*, const double*,
        const double*, const double*, size_t, const FastCDWorkspaceF64*);

static const double SENTINEL = 19.0;
static const double CANARY = 12345.5;
#define GUARD_WORDS ((size_t)4)

static const FastCDAlgorithm explicit_algorithms[] = {
    FASTCD_ALGORITHM_DIRECT, FASTCD_ALGORITHM_CARIOW_CARIOWA,
    FASTCD_ALGORITHM_QUASILINEAR
};
#define EXPLICIT_ALGORITHM_COUNT \
    (sizeof(explicit_algorithms) / sizeof(explicit_algorithms[0]))

// Fill an array with one value
static void fill_array(double* x, size_t count, double value) {
    for (size_t i = 0; i < count; i++) x[i] = value;
}

// Return whether every entry equals one value
static bool array_is(const double* x, size_t count, double value) {
    if (x == NULL) return false;
    for (size_t i = 0; i < count; i++) {
        if (x[i] != value) return false;
    }
    return true;
}

// Compare coordinate arrays exactly (+0.0 and -0.0 compare equal)
static bool arrays_exact(const double* x, const double* y, size_t dimension) {
    if (x == NULL || y == NULL) return false;
    for (size_t i = 0; i < dimension; i++) {
        if (x[i] != y[i]) return false;
    }
    return true;
}

// Compare coordinate arrays bit for bit
static bool arrays_bitwise(const double* x, const double* y,
        size_t dimension) {
    if (x == NULL || y == NULL) return false;
    return memcmp(x, y, dimension * sizeof(double)) == 0;
}

// Return whether every coordinate is finite
static bool arrays_finite(const double* x, size_t dimension) {
    for (size_t i = 0; i < dimension; i++) {
        if (!isfinite(x[i])) return false;
    }
    return true;
}

// Form the Euclidean norm in long double without squaring extreme values
static long double euclidean_norm(const double* x, size_t dimension) {
    long double norm = 0.0L;

    for (size_t i = 0; i < dimension; i++) {
        norm = hypotl(norm, (long double)x[i]);
    }
    return norm;
}

// Return the largest coordinate difference, or infinity for non-finite data
static long double max_abs_difference(const double* actual,
        const double* expected, size_t dimension) {
    long double worst = 0.0L;

    for (size_t i = 0; i < dimension; i++) {
        long double difference;

        if (!isfinite(actual[i]) || !isfinite(expected[i])) return HUGE_VALL;
        difference = fabsl((long double)actual[i] - (long double)expected[i]);
        if (difference > worst) worst = difference;
    }
    return worst;
}

// Return the norm-scaled tolerance 128 epsilon N scale
static long double scaled_tolerance(size_t dimension, long double scale) {
    return 128.0L * (long double)DBL_EPSILON * (long double)dimension * scale;
}

// Check a floating result under the norm-scaled tolerance; zero scale is exact
static bool within_tolerance(const double* actual, const double* expected,
        size_t dimension, long double scale) {
    long double difference = max_abs_difference(actual, expected, dimension);

    if (scale == 0.0L) return difference == 0.0L;
    return difference <= scaled_tolerance(dimension, scale);
}

// Allocate a block of contiguous arrays; NULL on failure or overflow
static double* allocate_block(size_t arrays, size_t dimension) {
    if (arrays == 0 || dimension == 0) return NULL;
    if (arrays > SIZE_MAX / sizeof(double) / dimension) return NULL;
    return malloc(arrays * dimension * sizeof(double));
}

// Deterministic xorshift64* generator so every run is reproducible
static uint64_t next_random_u64(void) {
    static uint64_t state = UINT64_C(0x4d595df4d0f33173);

    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return state * UINT64_C(2685821657736338717);
}

// Return a random integer-valued double in [-3,3]
static double random_small_integer(void) {
    return (double)((int)(next_random_u64() % UINT64_C(7)) - 3);
}

// Return a random double in [-1,1]
static double random_unit_interval(void) {
    uint64_t mantissa = next_random_u64() >> 11;
    double unit = (double)mantissa * (1.0 / 9007199254740992.0);
    return 2.0 * unit - 1.0;
}

// Fill an array with random small integers
static void fill_small_integers(double* x, size_t count) {
    for (size_t i = 0; i < count; i++) x[i] = random_small_integer();
}

// Fill an array with random values in [-1,1]
static void fill_unit_interval(double* x, size_t count) {
    for (size_t i = 0; i < count; i++) x[i] = random_unit_interval();
}

// Evaluate the independent oracle product
static bool oracle_mul(double* out, const double* x, const double* y,
        size_t dimension) {
    return reference_cd_mul_f64(out, x, y, dimension) == FASTCD_SUCCESS;
}

/* ---------- Workspace fixtures ---------- */

typedef struct GuardedWorkspace {
    double* storage;
    size_t required;
    FastCDWorkspaceF64 value;
} GuardedWorkspace;

// Allocate exactly the requested workspace length between canary words
static bool guarded_workspace_allocate(GuardedWorkspace* workspace,
        size_t required) {
    workspace->storage = NULL;
    workspace->required = required;
    workspace->value.data = NULL;
    workspace->value.length = 0;
    if (required > SIZE_MAX / sizeof(double) - 2 * GUARD_WORDS) return false;
    workspace->storage = malloc((required + 2 * GUARD_WORDS) * sizeof(double));
    if (workspace->storage == NULL) return false;
    fill_array(workspace->storage, required + 2 * GUARD_WORDS, CANARY);
    workspace->value.data = workspace->storage + GUARD_WORDS;
    workspace->value.length = required;
    return true;
}

// Reset every guard and scratch word to the canary value
static void guarded_workspace_arm(GuardedWorkspace* workspace) {
    if (workspace->storage != NULL) {
        fill_array(workspace->storage, workspace->required + 2 * GUARD_WORDS,
                CANARY);
    }
}

// Return whether the words outside the queried prefix still hold the canary
static bool guarded_workspace_intact(const GuardedWorkspace* workspace) {
    if (workspace->storage == NULL) return false;
    return array_is(workspace->storage, GUARD_WORDS, CANARY)
        && array_is(workspace->storage + GUARD_WORDS + workspace->required,
                GUARD_WORDS, CANARY);
}

// Release a guarded workspace
static void guarded_workspace_free(GuardedWorkspace* workspace) {
    free(workspace->storage);
    workspace->storage = NULL;
    workspace->required = 0;
    workspace->value.data = NULL;
    workspace->value.length = 0;
}

// Allocate the workspace one workspace query reports for a dimension
static bool workspace_for_query(GuardedWorkspace* workspace, size_t dimension,
        WorkspaceQuery query) {
    size_t required = 0;

    if (query(dimension, &required) != FASTCD_SUCCESS) {
        workspace->storage = NULL;
        workspace->required = 0;
        workspace->value.data = NULL;
        workspace->value.length = 0;
        return false;
    }
    return guarded_workspace_allocate(workspace, required);
}

// Allocate the workspace one algorithm needs at a dimension
static bool workspace_for_algorithm(GuardedWorkspace* workspace,
        size_t dimension, FastCDAlgorithm algorithm) {
    size_t required = 0;

    if (fastcd_mul_algorithm_workspace_size(dimension, algorithm, &required)
            != FASTCD_SUCCESS) {
        workspace->storage = NULL;
        workspace->required = 0;
        workspace->value.data = NULL;
        workspace->value.length = 0;
        return false;
    }
    return guarded_workspace_allocate(workspace, required);
}

// Multiply through the automatic algorithm with a freshly allocated workspace
static FastCDStatus mul_auto_allocated(double* out, const double* x,
        const double* y, size_t dimension) {
    GuardedWorkspace workspace;
    FastCDStatus status;

    if (!workspace_for_query(&workspace, dimension,
                fastcd_mul_workspace_size)) {
        return FASTCD_INSUFFICIENT_WORKSPACE;
    }
    status = fastcd_mul_f64(out, x, y, dimension, &workspace.value);
    guarded_workspace_free(&workspace);
    return status;
}

/* ---------- Multiplication routes ---------- */

typedef struct MulRoute {
    const char* name;
    FastCDAlgorithm algorithm;
    bool public_entry; // fastcd_mul_f64 / fastcd_mul_batch_f64
} MulRoute;

static const MulRoute mul_routes[] = {
    {"auto (public)", FASTCD_ALGORITHM_AUTO, true},
    {"auto (explicit)", FASTCD_ALGORITHM_AUTO, false},
    {"direct", FASTCD_ALGORITHM_DIRECT, false},
    {"cariow_cariowa", FASTCD_ALGORITHM_CARIOW_CARIOWA, false},
    {"quasilinear", FASTCD_ALGORITHM_QUASILINEAR, false}
};
#define MUL_ROUTE_COUNT (sizeof(mul_routes) / sizeof(mul_routes[0]))

// Multiply one pair through a route
static FastCDStatus mul_route_call(const MulRoute* route, double* out,
        const double* x, const double* y, size_t dimension,
        const FastCDWorkspaceF64* workspace) {
    if (route->public_entry) {
        return fastcd_mul_f64(out, x, y, dimension, workspace);
    }
    return fastcd_mul_with_algorithm_f64(out, x, y, dimension,
            route->algorithm, workspace);
}

// Multiply a batch through a route
static FastCDStatus mul_route_batch(const MulRoute* route, double* out,
        size_t out_stride, const double* x, size_t x_stride, const double* y,
        size_t y_stride, size_t count, size_t dimension,
        const FastCDWorkspaceF64* workspace) {
    if (route->public_entry) {
        return fastcd_mul_batch_f64(out, out_stride, x, x_stride, y, y_stride,
                count, dimension, workspace);
    }
    return fastcd_mul_batch_with_algorithm_f64(out, out_stride, x, x_stride,
            y, y_stride, count, dimension, route->algorithm, workspace);
}

// Wrap each explicit algorithm in the generic binary-operation signature
static FastCDStatus mul_direct(double* out, const double* x, const double* y,
        size_t dimension, const FastCDWorkspaceF64* workspace) {
    return fastcd_mul_with_algorithm_f64(out, x, y, dimension,
            FASTCD_ALGORITHM_DIRECT, workspace);
}

static FastCDStatus mul_cariow_cariowa(double* out, const double* x,
        const double* y, size_t dimension, const FastCDWorkspaceF64* workspace) {
    return fastcd_mul_with_algorithm_f64(out, x, y, dimension,
            FASTCD_ALGORITHM_CARIOW_CARIOWA, workspace);
}

static FastCDStatus mul_quasilinear(double* out, const double* x,
        const double* y, size_t dimension, const FastCDWorkspaceF64* workspace) {
    return fastcd_mul_with_algorithm_f64(out, x, y, dimension,
            FASTCD_ALGORITHM_QUASILINEAR, workspace);
}

static FastCDStatus mul_auto_explicit(double* out, const double* x,
        const double* y, size_t dimension, const FastCDWorkspaceF64* workspace) {
    return fastcd_mul_with_algorithm_f64(out, x, y, dimension,
            FASTCD_ALGORITHM_AUTO, workspace);
}

// Wrap each explicit algorithm's workspace query in the generic signature
static FastCDStatus query_direct(size_t dimension, size_t* length) {
    return fastcd_mul_algorithm_workspace_size(dimension,
            FASTCD_ALGORITHM_DIRECT, length);
}

static FastCDStatus query_cariow_cariowa(size_t dimension, size_t* length) {
    return fastcd_mul_algorithm_workspace_size(dimension,
            FASTCD_ALGORITHM_CARIOW_CARIOWA, length);
}

static FastCDStatus query_quasilinear(size_t dimension, size_t* length) {
    return fastcd_mul_algorithm_workspace_size(dimension,
            FASTCD_ALGORITHM_QUASILINEAR, length);
}

static FastCDStatus query_auto_explicit(size_t dimension, size_t* length) {
    return fastcd_mul_algorithm_workspace_size(dimension,
            FASTCD_ALGORITHM_AUTO, length);
}

/* ---------- Oracle compositions ---------- */

typedef enum RouteKind {
    ROUTE_MUL,
    ROUTE_CROSS,
    ROUTE_COMMUTATOR,
    ROUTE_ANTICOMMUTATOR,
    ROUTE_ASSOCIATOR,
    ROUTE_LOP,
    ROUTE_ROP
} RouteKind;

/*
 * Evaluate one operation from oracle products only.  For the ternary
 * operators the operands are named (x, y, z) = (a, b, x) of the library
 * declarations.  t1, t2, and t3 are scratch elements.
 */
// Compute the oracle reference of one route
static bool route_reference(RouteKind kind, double* expected, const double* x,
        const double* y, const double* z, size_t dimension, double* t1,
        double* t2, double* t3) {
    switch (kind) {
        case ROUTE_MUL:
            return oracle_mul(expected, x, y, dimension);
        case ROUTE_CROSS:
        case ROUTE_COMMUTATOR:
        case ROUTE_ANTICOMMUTATOR:
            if (!oracle_mul(t1, x, y, dimension)
                    || !oracle_mul(t2, y, x, dimension)) {
                return false;
            }
            for (size_t i = 0; i < dimension; i++) {
                if (kind == ROUTE_CROSS) {
                    expected[i] = 0.5 * (t1[i] - t2[i]);
                } else if (kind == ROUTE_COMMUTATOR) {
                    expected[i] = t1[i] - t2[i];
                } else {
                    expected[i] = t1[i] + t2[i];
                }
            }
            return true;
        case ROUTE_ASSOCIATOR:
            if (!oracle_mul(t1, x, y, dimension)
                    || !oracle_mul(t2, y, z, dimension)
                    || !oracle_mul(expected, t1, z, dimension)
                    || !oracle_mul(t3, x, t2, dimension)) {
                return false;
            }
            for (size_t i = 0; i < dimension; i++) expected[i] -= t3[i];
            return true;
        case ROUTE_LOP:
            if (!oracle_mul(t1, y, z, dimension)
                    || !oracle_mul(expected, x, t1, dimension)
                    || !oracle_mul(t2, x, z, dimension)
                    || !oracle_mul(t3, y, t2, dimension)) {
                return false;
            }
            for (size_t i = 0; i < dimension; i++) expected[i] -= t3[i];
            return true;
        case ROUTE_ROP:
            if (!oracle_mul(t1, z, y, dimension)
                    || !oracle_mul(expected, t1, x, dimension)
                    || !oracle_mul(t2, z, x, dimension)
                    || !oracle_mul(t3, t2, y, dimension)) {
                return false;
            }
            for (size_t i = 0; i < dimension; i++) expected[i] -= t3[i];
            return true;
        default:
            return false;
    }
}

/* ---------- Library information ---------- */

// Test version, status, and algorithm name strings
static void test_library_info(void) {
    const FastCDStatus statuses[] = {
        FASTCD_SUCCESS, FASTCD_INVALID_DIMENSION, FASTCD_INVALID_LEVEL,
        FASTCD_INVALID_BASIS_INDEX, FASTCD_INSUFFICIENT_WORKSPACE,
        FASTCD_ZERO_NORM, FASTCD_OVERLAPPING_BUFFERS, FASTCD_INVALID_STRIDE,
        FASTCD_SIZE_OVERFLOW, FASTCD_NULL_POINTER, FASTCD_UNSUPPORTED_ALGORITHM
    };
    const size_t status_count = sizeof(statuses) / sizeof(statuses[0]);
    const struct {
        FastCDAlgorithm algorithm;
        const char* name;
    } algorithm_names[] = {
        {FASTCD_ALGORITHM_AUTO, "auto"},
        {FASTCD_ALGORITHM_DIRECT, "direct"},
        {FASTCD_ALGORITHM_CARIOW_CARIOWA, "cariow_cariowa"},
        {FASTCD_ALGORITHM_QUASILINEAR, "quasilinear"}
    };
    const char* version = fastcd_version_string();
    const char* unknown_status = fastcd_status_string((FastCDStatus)999);
    const char* next_status = fastcd_status_string(
            (FastCDStatus)(FASTCD_UNSUPPORTED_ALGORITHM + 1));

    printf("\n=== library information ===\n");

    CHECK(version != NULL && version[0] != '\0', "version string is nonempty");
    CHECK(version != NULL && isdigit((unsigned char)version[0]) != 0
            && strchr(version, '.') != NULL,
            "version string looks like a semantic version");

    CHECK(unknown_status != NULL && unknown_status[0] != '\0',
            "out-of-range status has a description");
    CHECK(next_status != NULL && next_status[0] != '\0',
            "status past the last enumerator has a description");
    for (size_t i = 0; i < status_count; i++) {
        const char* text = fastcd_status_string(statuses[i]);

        CHECK(text != NULL && text[0] != '\0', "status %d has a description",
                (int)statuses[i]);
        CHECK(text != NULL && unknown_status != NULL
                && strcmp(text, unknown_status) != 0,
                "status %d is described distinctly from an unknown status",
                (int)statuses[i]);
        for (size_t j = 0; j < i; j++) {
            const char* other = fastcd_status_string(statuses[j]);

            CHECK(text != NULL && other != NULL && strcmp(text, other) != 0,
                    "statuses %d and %d have distinct descriptions",
                    (int)statuses[i], (int)statuses[j]);
        }
    }

    for (size_t i = 0; i < sizeof(algorithm_names) / sizeof(algorithm_names[0]);
            i++) {
        const char* text = fastcd_algorithm_string(algorithm_names[i].algorithm);
        bool lowercase = text != NULL;

        for (const char* c = text; lowercase && *c != '\0'; c++) {
            if (!islower((unsigned char)*c) && *c != '_') lowercase = false;
        }
        CHECK(text != NULL && strcmp(text, algorithm_names[i].name) == 0,
                "algorithm %d is named %s", (int)algorithm_names[i].algorithm,
                algorithm_names[i].name);
        CHECK(lowercase, "algorithm %d has a lowercase name",
                (int)algorithm_names[i].algorithm);
    }
    CHECK(strcmp(fastcd_algorithm_string((FastCDAlgorithm)7), "unknown") == 0,
            "algorithm value 7 is unknown");
    CHECK(strcmp(fastcd_algorithm_string((FastCDAlgorithm)4), "unknown") == 0,
            "algorithm value 4 is unknown");
    CHECK(strcmp(fastcd_algorithm_string((FastCDAlgorithm)1000), "unknown")
            == 0, "algorithm value 1000 is unknown");
}

/* ---------- Dimension utilities ---------- */

// Test power-of-two validation and dimension/level round trips
static void test_dimension_utilities(void) {
    const size_t bit_count = sizeof(size_t) * CHAR_BIT;
    size_t value = 777;

    printf("\n=== dimension utilities ===\n");

    CHECK(!fastcd_valid_dimension(0), "zero is not a valid dimension");
    CHECK(!fastcd_valid_dimension(3), "three is not a valid dimension");
    CHECK(!fastcd_valid_dimension(6), "six is not a valid dimension");
    CHECK(!fastcd_valid_dimension(12), "twelve is not a valid dimension");
    CHECK(!fastcd_valid_dimension(SIZE_MAX), "SIZE_MAX is not a valid dimension");
    CHECK(!fastcd_valid_dimension(((size_t)1 << (bit_count - 1)) + 1),
            "2^(bits-1)+1 is not a valid dimension");
    for (size_t level = 0; level < bit_count; level++) {
        size_t dimension = (size_t)1 << level;
        size_t recovered = SIZE_MAX;
        size_t constructed = 0;

        CHECK(fastcd_valid_dimension(dimension), "2^%zu is a valid dimension",
                level);
        CHECK(fastcd_dimension(level, &constructed) == FASTCD_SUCCESS
                && constructed == dimension, "dimension constructs 2^%zu",
                level);
        CHECK(fastcd_level(dimension, &recovered) == FASTCD_SUCCESS
                && recovered == level, "level recovers exponent %zu", level);
        if (dimension > 2) {
            CHECK(!fastcd_valid_dimension(dimension - 1)
                    && !fastcd_valid_dimension(dimension + 1),
                    "neighbors of 2^%zu are invalid", level);
        }
    }

    value = 777;
    CHECK(fastcd_level(3, &value) == FASTCD_INVALID_DIMENSION && value == 777,
            "level rejects a non-power-of-two without changing output");
    CHECK(fastcd_level(0, &value) == FASTCD_INVALID_DIMENSION && value == 777,
            "level rejects zero without changing output");
    CHECK(fastcd_level(1, NULL) == FASTCD_NULL_POINTER,
            "level rejects a null output");
    value = 777;
    CHECK(fastcd_dimension(bit_count, &value) == FASTCD_INVALID_LEVEL
            && value == 777,
            "dimension rejects an overflowing level without changing output");
    CHECK(fastcd_dimension(SIZE_MAX, &value) == FASTCD_INVALID_LEVEL
            && value == 777, "dimension rejects level SIZE_MAX");
    CHECK(fastcd_dimension(0, NULL) == FASTCD_NULL_POINTER,
            "dimension rejects a null output");
}

/* ---------- Algorithm selection ---------- */

// Test default-algorithm reporting, support queries, and unsupported errors
static void test_algorithm_selection(void) {
    const size_t max_level = 20;
    const FastCDAlgorithm invalid_algorithms[] = {
        (FastCDAlgorithm)4, (FastCDAlgorithm)7, (FastCDAlgorithm)1000
    };
    const size_t invalid_count =
        sizeof(invalid_algorithms) / sizeof(invalid_algorithms[0]);
    FastCDAlgorithm algorithm = (FastCDAlgorithm)77;
    FastCDAlgorithm batch_algorithm = (FastCDAlgorithm)77;
    size_t length = 777;

    printf("\n=== algorithm selection ===\n");

    CHECK(fastcd_default_algorithm(0, &algorithm) == FASTCD_INVALID_DIMENSION
            && algorithm == (FastCDAlgorithm)77,
            "default algorithm rejects dimension zero without writing");
    CHECK(fastcd_default_algorithm(3, &algorithm) == FASTCD_INVALID_DIMENSION
            && algorithm == (FastCDAlgorithm)77,
            "default algorithm rejects dimension three without writing");
    CHECK(fastcd_default_algorithm(8, NULL) == FASTCD_NULL_POINTER,
            "default algorithm rejects a null output");
    CHECK(fastcd_default_batch_algorithm(0, &batch_algorithm)
            == FASTCD_INVALID_DIMENSION
            && batch_algorithm == (FastCDAlgorithm)77,
            "batch default rejects dimension zero without writing");
    CHECK(fastcd_default_batch_algorithm(3, &batch_algorithm)
            == FASTCD_INVALID_DIMENSION
            && batch_algorithm == (FastCDAlgorithm)77,
            "batch default rejects dimension three without writing");
    CHECK(fastcd_default_batch_algorithm(8, NULL) == FASTCD_NULL_POINTER,
            "batch default rejects a null output");

    for (size_t level = 0; level <= max_level; level++) {
        size_t dimension = (size_t)1 << level;
        size_t auto_length = 777;
        size_t default_length = 778;
        size_t public_length = 779;
        size_t batch_length = 780;
        bool concrete;

        algorithm = (FastCDAlgorithm)77;
        CHECK(fastcd_default_algorithm(dimension, &algorithm)
                == FASTCD_SUCCESS, "default algorithm reports N=%zu",
                dimension);
        concrete = algorithm == FASTCD_ALGORITHM_DIRECT
            || algorithm == FASTCD_ALGORITHM_CARIOW_CARIOWA
            || algorithm == FASTCD_ALGORITHM_QUASILINEAR;
        CHECK(concrete, "default algorithm is a concrete algorithm at N=%zu",
                dimension);
        CHECK(fastcd_algorithm_supports_dimension(algorithm, dimension),
                "default algorithm supports its own dimension N=%zu",
                dimension);
        batch_algorithm = (FastCDAlgorithm)77;
        CHECK(fastcd_default_batch_algorithm(dimension, &batch_algorithm)
                == FASTCD_SUCCESS, "batch default algorithm reports N=%zu",
                dimension);
        CHECK((batch_algorithm == FASTCD_ALGORITHM_DIRECT
                    || batch_algorithm == FASTCD_ALGORITHM_CARIOW_CARIOWA
                    || batch_algorithm == FASTCD_ALGORITHM_QUASILINEAR)
                && fastcd_algorithm_supports_dimension(batch_algorithm,
                    dimension),
                "batch default is a concrete supported algorithm at N=%zu",
                dimension);
        CHECK(dimension < 64 || (algorithm == FASTCD_ALGORITHM_QUASILINEAR
                    && batch_algorithm == FASTCD_ALGORITHM_QUASILINEAR),
                "both automatic selections are quasilinear from N=64 (N=%zu)",
                dimension);
        {
            /* The documented tables (docs/c-api.md, "Algorithm selection").
             * A build without SIMD pairing serves batches from the single
             * table; that build is recognized by its batch choice at N=16. */
            static const FastCDAlgorithm documented_single[] = {
                FASTCD_ALGORITHM_QUASILINEAR, FASTCD_ALGORITHM_QUASILINEAR,
                FASTCD_ALGORITHM_DIRECT, FASTCD_ALGORITHM_DIRECT,
                FASTCD_ALGORITHM_DIRECT
            };
            static const FastCDAlgorithm documented_batch[] = {
                FASTCD_ALGORITHM_QUASILINEAR, FASTCD_ALGORITHM_QUASILINEAR,
                FASTCD_ALGORITHM_QUASILINEAR, FASTCD_ALGORITHM_DIRECT,
                FASTCD_ALGORITHM_CARIOW_CARIOWA
            };
            FastCDAlgorithm probe = FASTCD_ALGORITHM_AUTO;
            bool paired = fastcd_default_batch_algorithm(16, &probe)
                == FASTCD_SUCCESS && probe == FASTCD_ALGORITHM_CARIOW_CARIOWA;
            FastCDAlgorithm want_single = level < 5 ? documented_single[level]
                : FASTCD_ALGORITHM_QUASILINEAR;
            FastCDAlgorithm want_batch = !paired ? want_single
                : level < 5 ? documented_batch[level]
                : FASTCD_ALGORITHM_QUASILINEAR;

            CHECK(algorithm == want_single,
                    "single-product default follows the documented table N=%zu",
                    dimension);
            CHECK(batch_algorithm == want_batch,
                    "batch default follows the documented table N=%zu",
                    dimension);
        }
        CHECK(fastcd_algorithm_supports_dimension(FASTCD_ALGORITHM_AUTO,
                    dimension), "auto supports N=%zu", dimension);
        CHECK(fastcd_algorithm_supports_dimension(FASTCD_ALGORITHM_DIRECT,
                    dimension), "direct supports N=%zu", dimension);
        CHECK(fastcd_algorithm_supports_dimension(FASTCD_ALGORITHM_QUASILINEAR,
                    dimension), "quasilinear supports N=%zu", dimension);
        CHECK(fastcd_algorithm_supports_dimension(
                    FASTCD_ALGORITHM_CARIOW_CARIOWA, dimension)
                == (dimension <= FASTCD_CARIOW_CARIOWA_MAX_DIMENSION),
                "cariow_cariowa support matches its maximum dimension N=%zu",
                dimension);
        for (size_t i = 0; i < invalid_count; i++) {
            CHECK(!fastcd_algorithm_supports_dimension(invalid_algorithms[i],
                        dimension), "invalid algorithm %d is unsupported N=%zu",
                    (int)invalid_algorithms[i], dimension);
        }
        CHECK(fastcd_mul_workspace_size(dimension, &public_length)
                == FASTCD_SUCCESS
                && fastcd_mul_algorithm_workspace_size(dimension, algorithm,
                    &default_length) == FASTCD_SUCCESS
                && fastcd_mul_algorithm_workspace_size(dimension,
                    batch_algorithm, &batch_length) == FASTCD_SUCCESS
                && public_length == (default_length > batch_length
                    ? default_length : batch_length),
                "automatic workspace covers both default algorithms N=%zu",
                dimension);
        CHECK(fastcd_mul_algorithm_workspace_size(dimension,
                    FASTCD_ALGORITHM_AUTO, &auto_length) == FASTCD_SUCCESS
                && auto_length == public_length,
                "explicit AUTO workspace equals the public query N=%zu",
                dimension);
    }

    {
        const size_t invalid_dimensions[] = {0, 3, 6, 100};

        for (size_t d = 0; d < 4; d++) {
            CHECK(!fastcd_algorithm_supports_dimension(FASTCD_ALGORITHM_AUTO,
                        invalid_dimensions[d])
                    && !fastcd_algorithm_supports_dimension(
                        FASTCD_ALGORITHM_DIRECT, invalid_dimensions[d])
                    && !fastcd_algorithm_supports_dimension(
                        FASTCD_ALGORITHM_CARIOW_CARIOWA, invalid_dimensions[d])
                    && !fastcd_algorithm_supports_dimension(
                        FASTCD_ALGORITHM_QUASILINEAR, invalid_dimensions[d])
                    && !fastcd_algorithm_supports_dimension(
                        (FastCDAlgorithm)7, invalid_dimensions[d]),
                    "no algorithm supports invalid dimension %zu",
                    invalid_dimensions[d]);
        }
    }

    length = 777;
    CHECK(fastcd_mul_algorithm_workspace_size(
                2 * FASTCD_CARIOW_CARIOWA_MAX_DIMENSION,
                FASTCD_ALGORITHM_CARIOW_CARIOWA, &length)
            == FASTCD_UNSUPPORTED_ALGORITHM && length == 777,
            "cariow_cariowa workspace query is unsupported above its maximum");
    CHECK(fastcd_mul_algorithm_workspace_size(
                FASTCD_CARIOW_CARIOWA_MAX_DIMENSION,
                FASTCD_ALGORITHM_CARIOW_CARIOWA, &length)
            == FASTCD_SUCCESS
            && length == 2 * FASTCD_CARIOW_CARIOWA_MAX_DIMENSION,
            "cariow_cariowa table kernel needs 2N doubles at its maximum");
    length = 777;
    for (size_t i = 0; i < invalid_count; i++) {
        CHECK(fastcd_mul_algorithm_workspace_size(8, invalid_algorithms[i],
                    &length) == FASTCD_UNSUPPORTED_ALGORITHM && length == 777,
                "invalid algorithm %d workspace query is unsupported",
                (int)invalid_algorithms[i]);
    }
    CHECK(fastcd_mul_algorithm_workspace_size(8, FASTCD_ALGORITHM_DIRECT, NULL)
            == FASTCD_NULL_POINTER,
            "algorithm workspace query rejects a null length");
    CHECK(fastcd_mul_algorithm_workspace_size(3,
                FASTCD_ALGORITHM_CARIOW_CARIOWA, &length)
            == FASTCD_INVALID_DIMENSION && length == 777,
            "algorithm workspace query validates the dimension");
    CHECK(fastcd_mul_workspace_size(8, NULL) == FASTCD_NULL_POINTER,
            "public workspace query rejects a null length");

    {
        /* wide is twice the Cariow--Cariowa maximum; the scratch covers the
         * 2N table-kernel workspace at that maximum. */
        enum { wide = 2048, scratch_length = 2048 };
        double x[wide];
        double y[wide];
        double out[wide];
        double expected[wide];
        double scratch[scratch_length];
        FastCDWorkspaceF64 workspace = {scratch, scratch_length};

        CHECK(wide == 2 * FASTCD_CARIOW_CARIOWA_MAX_DIMENSION,
                "fixture spans twice the cariow_cariowa maximum dimension");
        fill_small_integers(x, wide);
        fill_small_integers(y, wide);
        fill_array(out, wide, SENTINEL);
        CHECK(fastcd_mul_with_algorithm_f64(out, x, y, wide,
                    FASTCD_ALGORITHM_CARIOW_CARIOWA, &workspace)
                == FASTCD_UNSUPPORTED_ALGORITHM && array_is(out, wide, SENTINEL),
                "cariow_cariowa product above its maximum is unsupported without writing");
        for (size_t i = 0; i < invalid_count; i++) {
            CHECK(fastcd_mul_with_algorithm_f64(out, x, y, 8,
                        invalid_algorithms[i], &workspace)
                    == FASTCD_UNSUPPORTED_ALGORITHM
                    && array_is(out, wide, SENTINEL),
                    "invalid algorithm %d product is unsupported without writing",
                    (int)invalid_algorithms[i]);
        }
        CHECK(fastcd_mul_batch_with_algorithm_f64(out, wide, x, wide, y, wide,
                    1, wide, FASTCD_ALGORITHM_CARIOW_CARIOWA, &workspace)
                == FASTCD_UNSUPPORTED_ALGORITHM && array_is(out, wide, SENTINEL),
                "cariow_cariowa batch above its maximum is unsupported without writing");
        CHECK(fastcd_mul_batch_with_algorithm_f64(NULL, 0, NULL, 0, NULL, 0, 0,
                    wide, FASTCD_ALGORITHM_CARIOW_CARIOWA, NULL)
                == FASTCD_UNSUPPORTED_ALGORITHM,
                "empty cariow_cariowa batch above its maximum is still unsupported");
        CHECK(fastcd_mul_batch_with_algorithm_f64(NULL, 0, NULL, 0, NULL, 0, 0,
                    8, (FastCDAlgorithm)7, NULL) == FASTCD_UNSUPPORTED_ALGORITHM,
                "empty batch rejects an invalid algorithm");
        CHECK(oracle_mul(expected, x, y, FASTCD_CARIOW_CARIOWA_MAX_DIMENSION)
                && fastcd_mul_with_algorithm_f64(out, x, y,
                    FASTCD_CARIOW_CARIOWA_MAX_DIMENSION,
                    FASTCD_ALGORITHM_CARIOW_CARIOWA, &workspace)
                == FASTCD_SUCCESS
                && arrays_exact(out, expected,
                    FASTCD_CARIOW_CARIOWA_MAX_DIMENSION),
                "cariow_cariowa product succeeds at its maximum dimension");
    }
}

/* ---------- Workspace contracts ---------- */

typedef struct WorkspaceRoute {
    const char* name;
    RouteKind kind;
    WorkspaceQuery query;
    BinaryOperation binary;
    TernaryOperation ternary;
    size_t max_dimension;
} WorkspaceRoute;

static const WorkspaceRoute workspace_routes[] = {
    {"mul auto (public)", ROUTE_MUL, fastcd_mul_workspace_size, fastcd_mul_f64,
        NULL, SIZE_MAX},
    {"mul auto (explicit)", ROUTE_MUL, query_auto_explicit, mul_auto_explicit,
        NULL, SIZE_MAX},
    {"mul direct", ROUTE_MUL, query_direct, mul_direct, NULL, SIZE_MAX},
    {"mul cariow_cariowa", ROUTE_MUL, query_cariow_cariowa, mul_cariow_cariowa,
        NULL, FASTCD_CARIOW_CARIOWA_MAX_DIMENSION},
    {"mul quasilinear", ROUTE_MUL, query_quasilinear, mul_quasilinear, NULL,
        SIZE_MAX},
    {"imag_cross", ROUTE_CROSS, fastcd_imag_cross_workspace_size,
        fastcd_imag_cross_f64, NULL, SIZE_MAX},
    {"commutator", ROUTE_COMMUTATOR, fastcd_imag_cross_workspace_size,
        fastcd_commutator_f64, NULL, SIZE_MAX},
    {"associator", ROUTE_ASSOCIATOR, fastcd_associator_workspace_size, NULL,
        fastcd_associator_f64, SIZE_MAX},
    {"lop_commutator", ROUTE_LOP, fastcd_lop_commutator_workspace_size, NULL,
        fastcd_lop_commutator_apply_f64, SIZE_MAX},
    {"rop_commutator", ROUTE_ROP, fastcd_rop_commutator_workspace_size, NULL,
        fastcd_rop_commutator_apply_f64, SIZE_MAX}
};
#define WORKSPACE_ROUTE_COUNT \
    (sizeof(workspace_routes) / sizeof(workspace_routes[0]))

// Invoke one workspace route with its natural operand list
static FastCDStatus route_call(const WorkspaceRoute* route, double* out,
        const double* x, const double* y, const double* z, size_t dimension,
        const FastCDWorkspaceF64* workspace) {
    if (route->binary != NULL) {
        return route->binary(out, x, y, dimension, workspace);
    }
    return route->ternary(out, x, y, z, dimension, workspace);
}

// Test every workspace-taking routine's query, length, alias, and canary rules
static void test_workspace_contracts(void) {
    const size_t dimensions[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
    const size_t dimension_count = sizeof(dimensions) / sizeof(dimensions[0]);
    size_t zero_required_cases = 0;
    size_t positive_required_cases = 0;

    printf("\n=== workspace contracts ===\n");

    for (size_t r = 0; r < WORKSPACE_ROUTE_COUNT; r++) {
        const WorkspaceRoute* route = &workspace_routes[r];
        size_t length = 777;

        CHECK(route->query(16, NULL) == FASTCD_NULL_POINTER,
                "%s query rejects a null length", route->name);
        CHECK(route->query(3, &length) == FASTCD_INVALID_DIMENSION
                && length == 777,
                "%s query preserves output on invalid dimension", route->name);
        CHECK(route->query(0, &length) == FASTCD_INVALID_DIMENSION
                && length == 777, "%s query rejects dimension zero",
                route->name);
        if (route->max_dimension < SIZE_MAX) {
            CHECK(route->query(2 * route->max_dimension, &length)
                    == FASTCD_UNSUPPORTED_ALGORITHM && length == 777,
                    "%s query is unsupported above its maximum dimension",
                    route->name);
        }

        for (size_t d = 0; d < dimension_count; d++) {
            size_t dimension = dimensions[d];
            size_t required = SIZE_MAX;
            double* block;
            double* x;
            double* y;
            double* z;
            double* out;
            double* first;
            double* expected;
            double* t1;
            double* t2;
            double* t3;
            double* storage;
            FastCDWorkspaceF64 exact;
            FastCDWorkspaceF64 longer;

            if (dimension > route->max_dimension) continue;
            block = allocate_block(9, dimension);
            CHECK(block != NULL, "allocate %s fixtures N=%zu", route->name,
                    dimension);
            if (block == NULL) continue;
            x = block;
            y = block + dimension;
            z = block + 2 * dimension;
            out = block + 3 * dimension;
            first = block + 4 * dimension;
            expected = block + 5 * dimension;
            t1 = block + 6 * dimension;
            t2 = block + 7 * dimension;
            t3 = block + 8 * dimension;
            fill_small_integers(x, dimension);
            fill_small_integers(y, dimension);
            fill_small_integers(z, dimension);

            CHECK(route->query(dimension, &required) == FASTCD_SUCCESS
                    && required <= SIZE_MAX / sizeof(double) - 2 * GUARD_WORDS,
                    "%s query succeeds N=%zu", route->name, dimension);
            if (required > SIZE_MAX / sizeof(double) - 2 * GUARD_WORDS) {
                free(block);
                continue;
            }
            storage = malloc((required + 2 * GUARD_WORDS) * sizeof(double));
            CHECK(storage != NULL, "allocate %s workspace N=%zu", route->name,
                    dimension);
            if (storage == NULL) {
                free(block);
                continue;
            }
            if (required == 0) zero_required_cases++;
            else positive_required_cases++;
            CHECK(route_reference(route->kind, expected, x, y, z, dimension,
                        t1, t2, t3), "%s oracle reference N=%zu", route->name,
                    dimension);

            exact.data = storage + GUARD_WORDS;
            exact.length = required;
            longer.data = storage + GUARD_WORDS;
            longer.length = required + GUARD_WORDS - 1;

            fill_array(storage, required + 2 * GUARD_WORDS, CANARY);
            fill_array(out, dimension, SENTINEL);
            CHECK(route_call(route, out, x, y, z, dimension, &exact)
                    == FASTCD_SUCCESS && arrays_exact(out, expected, dimension),
                    "%s exact-size workspace succeeds N=%zu", route->name,
                    dimension);
            CHECK(array_is(storage, GUARD_WORDS, CANARY)
                    && array_is(storage + GUARD_WORDS + required, GUARD_WORDS,
                        CANARY),
                    "%s exact-size call touches only the queried prefix N=%zu",
                    route->name, dimension);
            memcpy(first, out, dimension * sizeof(double));

            fill_array(storage, required + 2 * GUARD_WORDS, CANARY);
            fill_array(out, dimension, SENTINEL);
            CHECK(route_call(route, out, x, y, z, dimension, &longer)
                    == FASTCD_SUCCESS && arrays_bitwise(out, first, dimension),
                    "%s longer workspace gives identical results N=%zu",
                    route->name, dimension);
            CHECK(array_is(storage, GUARD_WORDS, CANARY)
                    && array_is(storage + GUARD_WORDS + required, GUARD_WORDS,
                        CANARY),
                    "%s leaves words beyond the queried prefix untouched N=%zu",
                    route->name, dimension);

            if (required > 0) {
                FastCDWorkspaceF64 short_workspace = {storage + GUARD_WORDS,
                    required - 1};
                FastCDWorkspaceF64 null_data = {NULL, required};
                FastCDWorkspaceF64 on_out = {out, required};
                FastCDWorkspaceF64 on_x = {x, required};
                FastCDWorkspaceF64 on_y = {y, required};
                FastCDWorkspaceF64 on_z = {z, required};
                FastCDWorkspaceF64 partial_x = {x + dimension - 1, required};

                fill_array(out, dimension, SENTINEL);
                CHECK(route_call(route, out, x, y, z, dimension,
                            &short_workspace) == FASTCD_INSUFFICIENT_WORKSPACE
                        && array_is(out, dimension, SENTINEL),
                        "%s one-short workspace fails atomically N=%zu",
                        route->name, dimension);
                CHECK(route_call(route, out, x, y, z, dimension, NULL)
                        == FASTCD_NULL_POINTER
                        && array_is(out, dimension, SENTINEL),
                        "%s null descriptor fails atomically N=%zu",
                        route->name, dimension);
                CHECK(route_call(route, out, x, y, z, dimension, &null_data)
                        == FASTCD_NULL_POINTER
                        && array_is(out, dimension, SENTINEL),
                        "%s null workspace data fails atomically N=%zu",
                        route->name, dimension);
                CHECK(route_call(route, out, x, y, z, dimension, &on_out)
                        == FASTCD_OVERLAPPING_BUFFERS
                        && array_is(out, dimension, SENTINEL),
                        "%s rejects a workspace overlapping the output N=%zu",
                        route->name, dimension);
                CHECK(route_call(route, out, x, y, z, dimension, &on_x)
                        == FASTCD_OVERLAPPING_BUFFERS
                        && array_is(out, dimension, SENTINEL),
                        "%s rejects a workspace overlapping x N=%zu",
                        route->name, dimension);
                CHECK(route_call(route, out, x, y, z, dimension, &on_y)
                        == FASTCD_OVERLAPPING_BUFFERS
                        && array_is(out, dimension, SENTINEL),
                        "%s rejects a workspace overlapping y N=%zu",
                        route->name, dimension);
                if (route->ternary != NULL) {
                    CHECK(route_call(route, out, x, y, z, dimension, &on_z)
                            == FASTCD_OVERLAPPING_BUFFERS
                            && array_is(out, dimension, SENTINEL),
                            "%s rejects a workspace overlapping z N=%zu",
                            route->name, dimension);
                }
                CHECK(route_call(route, out, x, y, z, dimension, &partial_x)
                        == FASTCD_OVERLAPPING_BUFFERS
                        && array_is(out, dimension, SENTINEL),
                        "%s rejects a workspace partially overlapping x N=%zu",
                        route->name, dimension);
            } else {
                FastCDWorkspaceF64 empty = {NULL, 0};

                fill_array(out, dimension, SENTINEL);
                CHECK(route_call(route, out, x, y, z, dimension, NULL)
                        == FASTCD_SUCCESS && arrays_bitwise(out, first, dimension),
                        "%s accepts a NULL workspace when none is required N=%zu",
                        route->name, dimension);
                fill_array(out, dimension, SENTINEL);
                CHECK(route_call(route, out, x, y, z, dimension, &empty)
                        == FASTCD_SUCCESS && arrays_bitwise(out, first, dimension),
                        "%s accepts an empty descriptor when none is required N=%zu",
                        route->name, dimension);
            }

            fill_array(out, dimension, SENTINEL);
            CHECK(route_call(route, out, x, y, z, 3, &exact)
                    == FASTCD_INVALID_DIMENSION
                    && array_is(out, dimension, SENTINEL),
                    "%s rejects dimension three atomically", route->name);
            CHECK(route_call(route, out, x, y, z, 0, &exact)
                    == FASTCD_INVALID_DIMENSION
                    && array_is(out, dimension, SENTINEL),
                    "%s rejects dimension zero atomically", route->name);
            CHECK(route_call(route, NULL, x, y, z, dimension, &exact)
                    == FASTCD_NULL_POINTER, "%s rejects a null output N=%zu",
                    route->name, dimension);
            CHECK(route_call(route, out, NULL, y, z, dimension, &exact)
                    == FASTCD_NULL_POINTER && array_is(out, dimension, SENTINEL),
                    "%s rejects a null first operand N=%zu", route->name,
                    dimension);
            CHECK(route_call(route, out, x, NULL, z, dimension, &exact)
                    == FASTCD_NULL_POINTER && array_is(out, dimension, SENTINEL),
                    "%s rejects a null second operand N=%zu", route->name,
                    dimension);
            if (route->ternary != NULL) {
                CHECK(route_call(route, out, x, y, NULL, dimension, &exact)
                        == FASTCD_NULL_POINTER
                        && array_is(out, dimension, SENTINEL),
                        "%s rejects a null third operand N=%zu", route->name,
                        dimension);
            }

            free(storage);
            free(block);
        }
    }
    CHECK(zero_required_cases > 0 && positive_required_cases > 0,
            "workspace sweep covered zero and positive requirements (%zu/%zu)",
            zero_required_cases, positive_required_cases);
}

/* ---------- Construction and coordinate arithmetic ---------- */

// Test construction, coordinate operations, and documented exact aliases
static void test_coordinate_operations(void) {
    const size_t dimension = 8;
    const double x[8] = {1, -2, 3, -4, 5, -6, 7, -8};
    const double y[8] = {-8, 7, -6, 5, -4, 3, -2, 1};
    const double sum[8] = {-7, 5, -3, 1, 1, -3, 5, -7};
    const double difference[8] = {9, -9, 9, -9, 9, -9, 9, -9};
    const double negative[8] = {-1, 2, -3, 4, -5, 6, -7, 8};
    const double scaled[8] = {2.5, -5, 7.5, -10, 12.5, -15, 17.5, -20};
    const double doubled[8] = {2, -4, 6, -8, 10, -12, 14, -16};
    double out[8];
    double alias[8];

    printf("\n=== construction and coordinate arithmetic ===\n");

    CHECK(fastcd_scalar_f64(out, dimension, 3.5) == FASTCD_SUCCESS
            && out[0] == 3.5, "scalar constructor sets the scalar coordinate");
    CHECK(array_is(out + 1, dimension - 1, 0.0),
            "scalar constructor zeros the imaginary coordinates");
    CHECK(fastcd_scalar_f64(out, 1, -2.0) == FASTCD_SUCCESS && out[0] == -2.0,
            "scalar constructor works at dimension one");

    for (size_t index = 0; index < dimension; index++) {
        bool correct;

        CHECK(fastcd_basis_f64(out, dimension, index) == FASTCD_SUCCESS,
                "basis constructor accepts index %zu", index);
        correct = true;
        for (size_t i = 0; i < dimension; i++) {
            if (out[i] != (i == index ? 1.0 : 0.0)) correct = false;
        }
        CHECK(correct, "basis e_%zu has the right coordinates", index);
    }
    fill_array(out, dimension, SENTINEL);
    CHECK(fastcd_basis_f64(out, dimension, dimension)
            == FASTCD_INVALID_BASIS_INDEX && array_is(out, dimension, SENTINEL),
            "basis constructor preserves output for an invalid index");
    CHECK(fastcd_basis_f64(out, dimension, SIZE_MAX)
            == FASTCD_INVALID_BASIS_INDEX && array_is(out, dimension, SENTINEL),
            "basis constructor preserves output for index SIZE_MAX");
    CHECK(fastcd_scalar_f64(NULL, dimension, 1.0) == FASTCD_NULL_POINTER,
            "scalar constructor rejects null output");
    CHECK(fastcd_basis_f64(NULL, dimension, 0) == FASTCD_NULL_POINTER,
            "basis constructor rejects null output");
    CHECK(fastcd_basis_f64(out, 3, 0) == FASTCD_INVALID_DIMENSION
            && array_is(out, dimension, SENTINEL),
            "basis constructor rejects invalid dimension");
    CHECK(fastcd_scalar_f64(out, 0, 1.0) == FASTCD_INVALID_DIMENSION
            && array_is(out, dimension, SENTINEL),
            "scalar constructor rejects dimension zero");

    CHECK(fastcd_add_f64(out, x, y, dimension) == FASTCD_SUCCESS
            && arrays_exact(out, sum, dimension), "coordinate addition");
    memcpy(alias, x, sizeof(alias));
    CHECK(fastcd_add_f64(alias, alias, y, dimension) == FASTCD_SUCCESS
            && arrays_exact(alias, sum, dimension),
            "coordinate addition supports out == x");
    memcpy(alias, y, sizeof(alias));
    CHECK(fastcd_add_f64(alias, x, alias, dimension) == FASTCD_SUCCESS
            && arrays_exact(alias, sum, dimension),
            "coordinate addition supports out == y");
    memcpy(alias, x, sizeof(alias));
    CHECK(fastcd_add_f64(alias, alias, alias, dimension) == FASTCD_SUCCESS
            && arrays_exact(alias, doubled, dimension),
            "coordinate addition supports out == x == y");
    CHECK(fastcd_add_f64(out, x, x, dimension) == FASTCD_SUCCESS
            && arrays_exact(out, doubled, dimension),
            "coordinate addition accepts identical read-only inputs");

    CHECK(fastcd_sub_f64(out, x, y, dimension) == FASTCD_SUCCESS
            && arrays_exact(out, difference, dimension),
            "coordinate subtraction");
    memcpy(alias, x, sizeof(alias));
    CHECK(fastcd_sub_f64(alias, alias, y, dimension) == FASTCD_SUCCESS
            && arrays_exact(alias, difference, dimension),
            "coordinate subtraction supports out == x");
    memcpy(alias, y, sizeof(alias));
    CHECK(fastcd_sub_f64(alias, x, alias, dimension) == FASTCD_SUCCESS
            && arrays_exact(alias, difference, dimension),
            "coordinate subtraction supports out == y");
    CHECK(fastcd_sub_f64(out, x, x, dimension) == FASTCD_SUCCESS
            && array_is(out, dimension, 0.0),
            "coordinate subtraction of identical inputs is zero");

    CHECK(fastcd_neg_f64(out, x, dimension) == FASTCD_SUCCESS
            && arrays_exact(out, negative, dimension), "coordinate negation");
    memcpy(alias, x, sizeof(alias));
    CHECK(fastcd_neg_f64(alias, alias, dimension) == FASTCD_SUCCESS
            && arrays_exact(alias, negative, dimension),
            "coordinate negation supports an exact alias");

    CHECK(fastcd_scale_f64(out, x, 2.5, dimension) == FASTCD_SUCCESS
            && arrays_exact(out, scaled, dimension), "coordinate scaling");
    memcpy(alias, x, sizeof(alias));
    CHECK(fastcd_scale_f64(alias, alias, 2.5, dimension) == FASTCD_SUCCESS
            && arrays_exact(alias, scaled, dimension),
            "coordinate scaling supports an exact alias");
    CHECK(fastcd_scale_f64(out, x, 0.0, dimension) == FASTCD_SUCCESS
            && array_is(out, dimension, 0.0), "scaling by zero gives zeros");

    fill_array(out, dimension, SENTINEL);
    CHECK(fastcd_add_f64(out, NULL, y, dimension) == FASTCD_NULL_POINTER
            && array_is(out, dimension, SENTINEL),
            "addition preserves output on null input");
    CHECK(fastcd_add_f64(out, x, NULL, dimension) == FASTCD_NULL_POINTER
            && array_is(out, dimension, SENTINEL),
            "addition preserves output on null second input");
    CHECK(fastcd_add_f64(NULL, x, y, dimension) == FASTCD_NULL_POINTER,
            "addition rejects null output");
    CHECK(fastcd_sub_f64(out, x, y, 3) == FASTCD_INVALID_DIMENSION
            && array_is(out, dimension, SENTINEL),
            "subtraction preserves output on invalid dimension");
    CHECK(fastcd_sub_f64(NULL, x, y, dimension) == FASTCD_NULL_POINTER,
            "subtraction rejects null output");
    CHECK(fastcd_neg_f64(NULL, x, dimension) == FASTCD_NULL_POINTER,
            "negation rejects null output");
    CHECK(fastcd_neg_f64(out, NULL, dimension) == FASTCD_NULL_POINTER
            && array_is(out, dimension, SENTINEL),
            "negation preserves output on null input");
    CHECK(fastcd_neg_f64(out, x, 0) == FASTCD_INVALID_DIMENSION
            && array_is(out, dimension, SENTINEL),
            "negation rejects dimension zero");
    CHECK(fastcd_scale_f64(out, NULL, 2.0, dimension) == FASTCD_NULL_POINTER
            && array_is(out, dimension, SENTINEL),
            "scaling preserves output on null input");
    CHECK(fastcd_scale_f64(out, x, 2.0, 6) == FASTCD_INVALID_DIMENSION
            && array_is(out, dimension, SENTINEL),
            "scaling rejects invalid dimension");
}

/* ---------- Overlap validation ---------- */

// Verify exact aliases, rejected partial overlaps, and identical inputs
static void test_overlap_validation(void) {
    const size_t dimension = 8;
    double storage[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    double separate[8] = {8, 7, 6, 5, 4, 3, 2, 1};
    double third[8] = {1, -1, 2, -2, 3, -3, 4, -4};
    double snapshot[9];
    double output[8];
    double expected[8];
    double t1[8];
    double t2[8];
    double t3[8];
    double value = SENTINEL;
    const BinaryOperation products[] = {
        fastcd_mul_f64, mul_auto_explicit, mul_direct, mul_cariow_cariowa,
        mul_quasilinear
    };
    const WorkspaceQuery product_queries[] = {
        fastcd_mul_workspace_size, query_auto_explicit, query_direct,
        query_cariow_cariowa, query_quasilinear
    };
    const char* product_names[] = {
        "auto (public) product", "auto (explicit) product", "direct product",
        "cariow_cariowa product", "quasilinear product"
    };
    const BinaryOperation crosses[] = {
        fastcd_imag_cross_f64, fastcd_commutator_f64
    };
    const RouteKind cross_kinds[] = {ROUTE_CROSS, ROUTE_COMMUTATOR};
    const char* cross_names[] = {"imaginary cross", "commutator"};
    const TernaryOperation ternaries[] = {
        fastcd_associator_f64, fastcd_lop_commutator_apply_f64,
        fastcd_rop_commutator_apply_f64
    };
    const WorkspaceQuery ternary_queries[] = {
        fastcd_associator_workspace_size, fastcd_lop_commutator_workspace_size,
        fastcd_rop_commutator_workspace_size
    };
    const RouteKind ternary_kinds[] = {ROUTE_ASSOCIATOR, ROUTE_LOP, ROUTE_ROP};
    const char* ternary_names[] = {
        "associator", "left-operator commutator", "right-operator commutator"
    };

    printf("\n=== overlap validation ===\n");

    memcpy(snapshot, storage, sizeof(storage));
    fill_array(output, dimension, SENTINEL);

    CHECK(fastcd_add_f64(storage + 1, storage, separate, dimension)
            == FASTCD_OVERLAPPING_BUFFERS && arrays_bitwise(storage, snapshot, 9),
            "addition rejects partial output overlap without writing");
    CHECK(fastcd_add_f64(output, storage, storage + 1, dimension)
            == FASTCD_OVERLAPPING_BUFFERS && array_is(output, dimension, SENTINEL),
            "addition rejects partially overlapping inputs");
    CHECK(fastcd_sub_f64(storage + 1, storage, separate, dimension)
            == FASTCD_OVERLAPPING_BUFFERS && arrays_bitwise(storage, snapshot, 9),
            "subtraction rejects partial output overlap without writing");
    CHECK(fastcd_sub_f64(output, storage + 1, storage, dimension)
            == FASTCD_OVERLAPPING_BUFFERS && array_is(output, dimension, SENTINEL),
            "subtraction rejects partially overlapping inputs");
    CHECK(fastcd_neg_f64(storage + 1, storage, dimension)
            == FASTCD_OVERLAPPING_BUFFERS && arrays_bitwise(storage, snapshot, 9),
            "negation rejects partial overlap without writing");
    CHECK(fastcd_scale_f64(storage + 1, storage, 2.0, dimension)
            == FASTCD_OVERLAPPING_BUFFERS && arrays_bitwise(storage, snapshot, 9),
            "scaling rejects partial overlap without writing");
    CHECK(fastcd_conjugate_f64(storage + 1, storage, dimension)
            == FASTCD_OVERLAPPING_BUFFERS && arrays_bitwise(storage, snapshot, 9),
            "conjugation rejects partial overlap without writing");
    CHECK(fastcd_imag_part_f64(storage + 1, storage, dimension)
            == FASTCD_OVERLAPPING_BUFFERS && arrays_bitwise(storage, snapshot, 9),
            "imaginary projection rejects partial overlap without writing");
    CHECK(fastcd_square_f64(storage + 1, storage, dimension)
            == FASTCD_OVERLAPPING_BUFFERS && arrays_bitwise(storage, snapshot, 9),
            "square rejects partial overlap without writing");
    CHECK(fastcd_normalize_f64(storage + 1, storage, dimension)
            == FASTCD_OVERLAPPING_BUFFERS && arrays_bitwise(storage, snapshot, 9),
            "normalization rejects partial overlap without writing");
    CHECK(fastcd_reciprocal_f64(storage + 1, storage, dimension)
            == FASTCD_OVERLAPPING_BUFFERS && arrays_bitwise(storage, snapshot, 9),
            "reciprocal rejects partial overlap without writing");
    CHECK(fastcd_square_f64(storage, storage + 1, dimension)
            == FASTCD_OVERLAPPING_BUFFERS && arrays_bitwise(storage, snapshot, 9),
            "square rejects partial overlap in the other direction");

    CHECK(fastcd_inner_f64(storage + 1, storage, separate, dimension)
            == FASTCD_OVERLAPPING_BUFFERS && arrays_bitwise(storage, snapshot, 9),
            "inner-product scalar output may not lie inside an input");
    CHECK(fastcd_inner_f64(storage + 1, separate, storage, dimension)
            == FASTCD_OVERLAPPING_BUFFERS && arrays_bitwise(storage, snapshot, 9),
            "inner-product scalar output may not lie inside the second input");
    CHECK(fastcd_inner_f64(&value, storage, storage + 1, dimension)
            == FASTCD_OVERLAPPING_BUFFERS && value == SENTINEL,
            "inner product rejects partially overlapping inputs");
    CHECK(fastcd_inner_f64(&value, storage, storage, dimension)
            == FASTCD_SUCCESS && value == 204.0,
            "inner product accepts identical inputs (sum of squares 1..8)");
    value = SENTINEL;
    CHECK(fastcd_norm_squared_f64(storage + 1, storage, dimension)
            == FASTCD_OVERLAPPING_BUFFERS && arrays_bitwise(storage, snapshot, 9),
            "squared-norm scalar output may not overlap its input");
    CHECK(fastcd_norm_f64(storage + 1, storage, dimension)
            == FASTCD_OVERLAPPING_BUFFERS && arrays_bitwise(storage, snapshot, 9),
            "norm scalar output may not overlap its input");
    CHECK(fastcd_norm_f64(storage, storage, dimension)
            == FASTCD_OVERLAPPING_BUFFERS && arrays_bitwise(storage, snapshot, 9),
            "norm scalar output may not be the input's first coordinate");

    CHECK(fastcd_anticommutator_f64(storage + 1, storage, separate, dimension)
            == FASTCD_OVERLAPPING_BUFFERS && arrays_bitwise(storage, snapshot, 9),
            "anticommutator rejects partial output overlap");
    CHECK(fastcd_anticommutator_f64(storage, storage, separate, dimension)
            == FASTCD_OVERLAPPING_BUFFERS && arrays_bitwise(storage, snapshot, 9),
            "anticommutator rejects out == x");
    CHECK(fastcd_anticommutator_f64(separate, storage, separate, dimension)
            == FASTCD_OVERLAPPING_BUFFERS, "anticommutator rejects out == y");
    CHECK(fastcd_anticommutator_f64(output, storage, storage + 1, dimension)
            == FASTCD_OVERLAPPING_BUFFERS && array_is(output, dimension, SENTINEL),
            "anticommutator rejects partially overlapping inputs");
    CHECK(route_reference(ROUTE_ANTICOMMUTATOR, expected, storage, storage,
                NULL, dimension, t1, t2, t3)
            && fastcd_anticommutator_f64(output, storage, storage, dimension)
            == FASTCD_SUCCESS && arrays_exact(output, expected, dimension),
            "anticommutator accepts identical inputs (2 x^2)");

    CHECK(fastcd_basis_lmul_f64(storage + 1, 1, storage, dimension)
            == FASTCD_OVERLAPPING_BUFFERS && arrays_bitwise(storage, snapshot, 9),
            "basis left action rejects partial overlap");
    CHECK(fastcd_basis_rmul_f64(storage + 1, storage, 1, dimension)
            == FASTCD_OVERLAPPING_BUFFERS && arrays_bitwise(storage, snapshot, 9),
            "basis right action rejects partial overlap");
    CHECK(fastcd_basis_lmul_f64(storage, 1, storage, dimension)
            == FASTCD_OVERLAPPING_BUFFERS && arrays_bitwise(storage, snapshot, 9),
            "basis left action rejects an exact alias");
    CHECK(fastcd_basis_rmul_f64(storage, storage, 1, dimension)
            == FASTCD_OVERLAPPING_BUFFERS && arrays_bitwise(storage, snapshot, 9),
            "basis right action rejects an exact alias");

    for (size_t p = 0; p < sizeof(products) / sizeof(products[0]); p++) {
        GuardedWorkspace workspace;

        if (!workspace_for_query(&workspace, dimension, product_queries[p])) {
            CHECK(false, "allocate %s workspace", product_names[p]);
            continue;
        }
        fill_array(output, dimension, SENTINEL);
        CHECK(products[p](storage + 1, storage, separate, dimension,
                    &workspace.value) == FASTCD_OVERLAPPING_BUFFERS
                && arrays_bitwise(storage, snapshot, 9),
                "%s rejects partial output/input overlap", product_names[p]);
        CHECK(products[p](output, storage, storage + 1, dimension,
                    &workspace.value) == FASTCD_OVERLAPPING_BUFFERS
                && array_is(output, dimension, SENTINEL),
                "%s rejects partially overlapping inputs", product_names[p]);
        CHECK(products[p](storage, storage, separate, dimension,
                    &workspace.value) == FASTCD_OVERLAPPING_BUFFERS
                && arrays_bitwise(storage, snapshot, 9),
                "%s rejects out == x", product_names[p]);
        CHECK(products[p](separate, storage, separate, dimension,
                    &workspace.value) == FASTCD_OVERLAPPING_BUFFERS,
                "%s rejects out == y", product_names[p]);
        CHECK(oracle_mul(expected, storage, storage, dimension)
                && products[p](output, storage, storage, dimension,
                    &workspace.value) == FASTCD_SUCCESS
                && arrays_exact(output, expected, dimension),
                "%s accepts identical read-only inputs", product_names[p]);
        guarded_workspace_free(&workspace);
    }

    for (size_t c = 0; c < sizeof(crosses) / sizeof(crosses[0]); c++) {
        GuardedWorkspace workspace;

        if (!workspace_for_query(&workspace, dimension,
                    fastcd_imag_cross_workspace_size)) {
            CHECK(false, "allocate %s workspace", cross_names[c]);
            continue;
        }
        fill_array(output, dimension, SENTINEL);
        CHECK(crosses[c](storage + 1, storage, separate, dimension,
                    &workspace.value) == FASTCD_OVERLAPPING_BUFFERS
                && arrays_bitwise(storage, snapshot, 9),
                "%s rejects partial output/input overlap", cross_names[c]);
        CHECK(crosses[c](output, storage, storage + 1, dimension,
                    &workspace.value) == FASTCD_OVERLAPPING_BUFFERS
                && array_is(output, dimension, SENTINEL),
                "%s rejects partially overlapping inputs", cross_names[c]);
        CHECK(crosses[c](storage, storage, separate, dimension,
                    &workspace.value) == FASTCD_OVERLAPPING_BUFFERS
                && arrays_bitwise(storage, snapshot, 9),
                "%s rejects out == x", cross_names[c]);
        CHECK(crosses[c](separate, storage, separate, dimension,
                    &workspace.value) == FASTCD_OVERLAPPING_BUFFERS,
                "%s rejects out == y", cross_names[c]);
        CHECK(route_reference(cross_kinds[c], expected, storage, storage, NULL,
                    dimension, t1, t2, t3)
                && crosses[c](output, storage, storage, dimension,
                    &workspace.value) == FASTCD_SUCCESS
                && arrays_exact(output, expected, dimension)
                && array_is(output, dimension, 0.0),
                "%s of identical inputs is zero", cross_names[c]);
        guarded_workspace_free(&workspace);
    }

    for (size_t t = 0; t < sizeof(ternaries) / sizeof(ternaries[0]); t++) {
        GuardedWorkspace workspace;

        if (!workspace_for_query(&workspace, dimension, ternary_queries[t])) {
            CHECK(false, "allocate %s workspace", ternary_names[t]);
            continue;
        }
        fill_array(output, dimension, SENTINEL);
        CHECK(ternaries[t](storage + 1, storage, separate, third, dimension,
                    &workspace.value) == FASTCD_OVERLAPPING_BUFFERS
                && arrays_bitwise(storage, snapshot, 9),
                "%s rejects partial output overlap", ternary_names[t]);
        CHECK(ternaries[t](output, storage, storage + 1, third, dimension,
                    &workspace.value) == FASTCD_OVERLAPPING_BUFFERS
                && array_is(output, dimension, SENTINEL),
                "%s rejects partially overlapping first inputs",
                ternary_names[t]);
        CHECK(ternaries[t](output, separate, storage, storage + 1, dimension,
                    &workspace.value) == FASTCD_OVERLAPPING_BUFFERS
                && array_is(output, dimension, SENTINEL),
                "%s rejects partially overlapping last inputs",
                ternary_names[t]);
        CHECK(ternaries[t](output, storage + 1, separate, storage, dimension,
                    &workspace.value) == FASTCD_OVERLAPPING_BUFFERS
                && array_is(output, dimension, SENTINEL),
                "%s rejects partially overlapping outer inputs",
                ternary_names[t]);
        CHECK(ternaries[t](storage, storage, separate, third, dimension,
                    &workspace.value) == FASTCD_OVERLAPPING_BUFFERS
                && arrays_bitwise(storage, snapshot, 9),
                "%s rejects out == first operand", ternary_names[t]);
        CHECK(ternaries[t](separate, storage, separate, third, dimension,
                    &workspace.value) == FASTCD_OVERLAPPING_BUFFERS,
                "%s rejects out == second operand", ternary_names[t]);
        CHECK(ternaries[t](third, storage, separate, third, dimension,
                    &workspace.value) == FASTCD_OVERLAPPING_BUFFERS,
                "%s rejects out == third operand", ternary_names[t]);
        CHECK(route_reference(ternary_kinds[t], expected, storage, storage,
                    storage, dimension, t1, t2, t3)
                && ternaries[t](output, storage, storage, storage, dimension,
                    &workspace.value) == FASTCD_SUCCESS
                && arrays_exact(output, expected, dimension)
                && array_is(output, dimension, 0.0),
                "%s accepts one array as all three operands and vanishes",
                ternary_names[t]);
        CHECK(route_reference(ternary_kinds[t], expected, storage, separate,
                    storage, dimension, t1, t2, t3)
                && ternaries[t](output, storage, separate, storage, dimension,
                    &workspace.value) == FASTCD_SUCCESS
                && arrays_exact(output, expected, dimension),
                "%s accepts identical outer operands", ternary_names[t]);
        CHECK(route_reference(ternary_kinds[t], expected, storage, storage,
                    third, dimension, t1, t2, t3)
                && ternaries[t](output, storage, storage, third, dimension,
                    &workspace.value) == FASTCD_SUCCESS
                && arrays_exact(output, expected, dimension),
                "%s accepts identical leading operands", ternary_names[t]);
        guarded_workspace_free(&workspace);
    }

    CHECK(fastcd_inner_f64(&value, storage, separate, dimension)
            == FASTCD_SUCCESS && value == 120.0,
            "disjoint scalar output remains valid after overlap checks");
}

/* ---------- Structural operations ---------- */

// Test conjugation, projections, inner products, norms, and reciprocals
static void test_structural_operations(void) {
    const size_t dimension = 8;
    const double x[8] = {3, 4, -2, 1, 0, 2, -1, 3};
    const double y[8] = {-1, 2, 3, 0, -2, 1, 4, -3};
    const double conjugate[8] = {3, -4, 2, -1, 0, -2, 1, -3};
    const double imaginary[8] = {0, 4, -2, 1, 0, 2, -1, 3};
    const double pure_imaginary[8] = {0, 2, 0, 0, 0, 0, 0, 0};
    const double zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    double out[8];
    double alias[8];
    double product[8];
    double identity[8];
    double value = -1.0;

    printf("\n=== structural operations ===\n");

    fill_array(identity, dimension, 0.0);
    identity[0] = 1.0;

    CHECK(fastcd_conjugate_f64(out, x, dimension) == FASTCD_SUCCESS
            && arrays_exact(out, conjugate, dimension), "conjugation");
    memcpy(alias, x, sizeof(alias));
    CHECK(fastcd_conjugate_f64(alias, alias, dimension) == FASTCD_SUCCESS
            && arrays_exact(alias, conjugate, dimension),
            "conjugation supports an exact alias");
    CHECK(fastcd_conjugate_f64(alias, alias, dimension) == FASTCD_SUCCESS
            && arrays_exact(alias, x, dimension),
            "conjugation is an involution");
    CHECK(fastcd_conjugate_f64(out, x, 1) == FASTCD_SUCCESS && out[0] == x[0],
            "conjugation at dimension one is the identity");

    CHECK(fastcd_imag_part_f64(out, x, dimension) == FASTCD_SUCCESS
            && arrays_exact(out, imaginary, dimension),
            "imaginary projection");
    memcpy(alias, x, sizeof(alias));
    CHECK(fastcd_imag_part_f64(alias, alias, dimension) == FASTCD_SUCCESS
            && arrays_exact(alias, imaginary, dimension),
            "imaginary projection supports an exact alias");
    CHECK(fastcd_imag_part_f64(out, x, 1) == FASTCD_SUCCESS && out[0] == 0.0,
            "imaginary projection at dimension one is zero");

    CHECK(fastcd_inner_f64(&value, x, y, dimension) == FASTCD_SUCCESS
            && value == -12.0, "Euclidean inner product");
    value = -1.0;
    CHECK(fastcd_norm_squared_f64(&value, x, dimension) == FASTCD_SUCCESS
            && value == 44.0, "squared norm");
    value = -1.0;
    CHECK(fastcd_norm_f64(&value, x, dimension) == FASTCD_SUCCESS
            && fabs(value - sqrt(44.0)) <= 2.0 * DBL_EPSILON * sqrt(44.0),
            "scaled Euclidean norm");
    value = -1.0;
    CHECK(fastcd_norm_f64(&value, zero, dimension) == FASTCD_SUCCESS
            && value == 0.0, "norm of zero is zero");
    CHECK(fastcd_norm_squared_f64(&value, zero, dimension) == FASTCD_SUCCESS
            && value == 0.0, "squared norm of zero is zero");
    CHECK(fastcd_norm_f64(&value, pure_imaginary, dimension) == FASTCD_SUCCESS
            && value == 2.0, "norm of a single nonzero coordinate is exact");
    CHECK(fastcd_norm_squared_f64(&value, pure_imaginary, dimension)
            == FASTCD_SUCCESS && value == 4.0,
            "squared norm of a single nonzero coordinate is exact");

    CHECK(fastcd_normalize_f64(out, x, dimension) == FASTCD_SUCCESS,
            "normalization succeeds for nonzero input");
    value = -1.0;
    CHECK(fastcd_norm_f64(&value, out, dimension) == FASTCD_SUCCESS
            && fabs(value - 1.0) <= 8.0 * DBL_EPSILON,
            "normalized element has unit norm");
    for (size_t i = 0; i < dimension; i++) {
        CHECK(fabs(out[i] * sqrt(44.0) - x[i]) <= 8.0 * DBL_EPSILON * 4.0,
                "normalized coordinate %zu keeps its direction", i);
    }
    memcpy(alias, x, sizeof(alias));
    CHECK(fastcd_normalize_f64(alias, alias, dimension) == FASTCD_SUCCESS
            && arrays_bitwise(alias, out, dimension),
            "in-place normalization matches disjoint normalization");

    CHECK(fastcd_reciprocal_f64(out, x, dimension) == FASTCD_SUCCESS,
            "reciprocal succeeds for nonzero input");
    for (size_t i = 0; i < dimension; i++) {
        CHECK(fabs(out[i] - conjugate[i] / 44.0) <= 2.0 * DBL_EPSILON,
                "reciprocal coordinate %zu is conjugate over squared norm", i);
    }
    CHECK(mul_auto_allocated(product, x, out, dimension) == FASTCD_SUCCESS
            && within_tolerance(product, identity, dimension, 1.0L),
            "x reciprocal(x) is the multiplicative identity");
    CHECK(mul_auto_allocated(product, out, x, dimension) == FASTCD_SUCCESS
            && within_tolerance(product, identity, dimension, 1.0L),
            "reciprocal(x) x is the multiplicative identity");
    memcpy(alias, x, sizeof(alias));
    CHECK(fastcd_reciprocal_f64(alias, alias, dimension) == FASTCD_SUCCESS
            && arrays_bitwise(alias, out, dimension),
            "reciprocal supports an exact alias");

    CHECK(fastcd_reciprocal_f64(out, pure_imaginary, dimension)
            == FASTCD_SUCCESS && out[0] == 0.0 && out[1] == -0.5
            && array_is(out + 2, dimension - 2, 0.0),
            "reciprocal accepts a nonzero element with zero scalar coordinate");
    CHECK(mul_auto_allocated(product, pure_imaginary, out, dimension)
            == FASTCD_SUCCESS && arrays_exact(product, identity, dimension),
            "pure-imaginary element times its reciprocal is exactly one");
    CHECK(fastcd_reciprocal_f64(out, identity, dimension) == FASTCD_SUCCESS
            && arrays_exact(out, identity, dimension),
            "reciprocal of one is one");

    fill_array(out, dimension, SENTINEL);
    CHECK(fastcd_normalize_f64(out, zero, dimension) == FASTCD_ZERO_NORM
            && array_is(out, dimension, SENTINEL),
            "normalization preserves output for zero norm");
    CHECK(fastcd_reciprocal_f64(out, zero, dimension) == FASTCD_ZERO_NORM
            && array_is(out, dimension, SENTINEL),
            "reciprocal preserves output for zero norm");
    memcpy(alias, zero, sizeof(alias));
    CHECK(fastcd_normalize_f64(alias, alias, dimension) == FASTCD_ZERO_NORM
            && array_is(alias, dimension, 0.0),
            "in-place normalization of zero fails without writing");
    value = SENTINEL;
    CHECK(fastcd_inner_f64(&value, NULL, y, dimension) == FASTCD_NULL_POINTER
            && value == SENTINEL,
            "inner product preserves output for null input");
    CHECK(fastcd_inner_f64(NULL, x, y, dimension) == FASTCD_NULL_POINTER,
            "inner product rejects null output");
    CHECK(fastcd_inner_f64(&value, x, y, 5) == FASTCD_INVALID_DIMENSION
            && value == SENTINEL, "inner product rejects invalid dimension");
    CHECK(fastcd_norm_f64(NULL, x, dimension) == FASTCD_NULL_POINTER,
            "norm rejects null output");
    CHECK(fastcd_norm_f64(&value, NULL, dimension) == FASTCD_NULL_POINTER
            && value == SENTINEL, "norm rejects null input");
    CHECK(fastcd_norm_squared_f64(NULL, x, dimension) == FASTCD_NULL_POINTER,
            "squared norm rejects null output");
    CHECK(fastcd_norm_squared_f64(&value, x, 0) == FASTCD_INVALID_DIMENSION
            && value == SENTINEL, "squared norm rejects dimension zero");
    CHECK(fastcd_conjugate_f64(out, x, 3) == FASTCD_INVALID_DIMENSION
            && array_is(out, dimension, SENTINEL),
            "conjugation preserves output for invalid dimension");
    CHECK(fastcd_conjugate_f64(NULL, x, dimension) == FASTCD_NULL_POINTER,
            "conjugation rejects null output");
    CHECK(fastcd_imag_part_f64(out, NULL, dimension) == FASTCD_NULL_POINTER
            && array_is(out, dimension, SENTINEL),
            "imaginary projection rejects null input");
    CHECK(fastcd_normalize_f64(out, NULL, dimension) == FASTCD_NULL_POINTER
            && array_is(out, dimension, SENTINEL),
            "normalization rejects null input");
    CHECK(fastcd_reciprocal_f64(NULL, x, dimension) == FASTCD_NULL_POINTER,
            "reciprocal rejects null output");
    CHECK(fastcd_reciprocal_f64(out, x, 12) == FASTCD_INVALID_DIMENSION
            && array_is(out, dimension, SENTINEL),
            "reciprocal rejects invalid dimension");
}

// Exercise scaled norm, normalization, reciprocal, and square at IEEE extremes
static void test_numeric_edge_cases(void) {
    const size_t dimension = 4;
    const double true_minimum =
#ifdef DBL_TRUE_MIN
        DBL_TRUE_MIN;
#else
        4.9406564584124654e-324;
#endif
    double huge[4] = {DBL_MAX, DBL_MAX, 0.0, 0.0};
    double tiny[4] = {DBL_MIN, DBL_MIN, 0.0, 0.0};
    double subnormal[4] = {0.0, 0.0, 0.0, 0.0};
    double nan_element[4] = {NAN, 1.0, 0.0, 0.0};
    double infinity_element[4] = {INFINITY, 1.0, 0.0, 0.0};
    double negative_infinity[4] = {1.0, -INFINITY, 0.0, 0.0};
    double out[4];
    double alias[4];
    double value = 0.0;
    const double inverse_root_two = 1.0 / sqrt(2.0);

    printf("\n=== numerical edge cases ===\n");

    CHECK(fastcd_norm_f64(&value, huge, dimension) == FASTCD_SUCCESS
            && isinf(value) && value > 0.0,
            "true huge-vector norm overflows to +infinity");
    CHECK(fastcd_norm_squared_f64(&value, huge, dimension) == FASTCD_SUCCESS
            && isinf(value) && value > 0.0,
            "huge-vector squared norm overflows to +infinity");
    CHECK(fastcd_normalize_f64(out, huge, dimension) == FASTCD_SUCCESS
            && fabs(out[0] - inverse_root_two) <= 4.0 * DBL_EPSILON
            && fabs(out[1] - inverse_root_two) <= 4.0 * DBL_EPSILON
            && out[2] == 0.0 && out[3] == 0.0,
            "scaled normalization preserves a finite huge-vector direction");
    memcpy(alias, huge, sizeof(alias));
    CHECK(fastcd_normalize_f64(alias, alias, dimension) == FASTCD_SUCCESS
            && arrays_bitwise(alias, out, dimension),
            "huge-vector normalization works in place");
    CHECK(fastcd_reciprocal_f64(out, huge, dimension) == FASTCD_SUCCESS
            && out[0] > 0.0 && out[1] < 0.0 && out[0] == -out[1]
            && out[0] <= DBL_MIN,
            "huge-vector reciprocal is a tiny finite or subnormal value");

    {
        double huge_scalar[4] = {DBL_MAX, 0.0, 0.0, 0.0};
        double balanced_huge[4] = {DBL_MAX, DBL_MAX, 0.0, 0.0};
        double huge_mixed[4] = {DBL_MAX, DBL_MIN, 0.0, 0.0};
        double small_mixed[4] = {DBL_MIN, DBL_MAX, 0.0, 0.0};
        double subnormal_mixed[4] = {true_minimum, 0.5, 0.0, 0.0};
        double expected_imaginary = (DBL_MAX * DBL_MIN) * 2.0;
        double huge_imaginary[4] = {0.0, DBL_MAX, 0.0, 0.0};

        CHECK(fastcd_square_f64(out, huge_scalar, dimension) == FASTCD_SUCCESS
                && isinf(out[0]) && out[0] > 0.0 && out[1] == 0.0
                && out[2] == 0.0 && out[3] == 0.0,
                "square avoids infinity-times-zero in a huge real element");
        CHECK(fastcd_square_f64(out, balanced_huge, dimension) == FASTCD_SUCCESS
                && out[0] == 0.0 && isinf(out[1]) && out[1] > 0.0,
                "square avoids infinity-minus-infinity in its scalar coordinate");
        CHECK(fastcd_square_f64(out, huge_mixed, dimension) == FASTCD_SUCCESS
                && isinf(out[0]) && out[0] > 0.0 && isfinite(out[1])
                && out[1] == expected_imaginary,
                "square preserves finite huge-times-small imaginary terms");
        CHECK(fastcd_square_f64(out, small_mixed, dimension) == FASTCD_SUCCESS
                && isinf(out[0]) && out[0] < 0.0 && isfinite(out[1])
                && out[1] == expected_imaginary,
                "square preserves finite small-times-huge imaginary terms");
        CHECK(fastcd_square_f64(out, subnormal_mixed, dimension)
                == FASTCD_SUCCESS && out[1] == true_minimum && out[0] == -0.25,
                "square avoids underflow in a representable doubled product");
        CHECK(fastcd_square_f64(out, huge_imaginary, dimension) == FASTCD_SUCCESS
                && isinf(out[0]) && out[0] < 0.0 && out[1] == 0.0,
                "square of a huge imaginary element is -infinity");
    }

    CHECK(fastcd_reciprocal_f64(out, tiny, dimension) == FASTCD_SUCCESS
            && isfinite(out[0]) && isfinite(out[1]) && out[0] > 0.0
            && out[1] < 0.0 && out[0] == -out[1] && out[2] == 0.0,
            "scaled reciprocal remains finite for DBL_MIN coordinates");
    CHECK(fabs(out[0] * (2.0 * DBL_MIN) - 1.0) <= 4.0 * DBL_EPSILON,
            "tiny reciprocal has the right magnitude");
    memcpy(alias, tiny, sizeof(alias));
    CHECK(fastcd_reciprocal_f64(alias, alias, dimension) == FASTCD_SUCCESS
            && arrays_bitwise(alias, out, dimension),
            "tiny reciprocal works in place");
    CHECK(fastcd_norm_f64(&value, tiny, dimension) == FASTCD_SUCCESS
            && fabs(value - DBL_MIN * sqrt(2.0)) <= 4.0 * DBL_EPSILON * DBL_MIN,
            "tiny norm does not underflow");
    CHECK(fastcd_normalize_f64(out, tiny, dimension) == FASTCD_SUCCESS
            && fabs(out[0] - inverse_root_two) <= 4.0 * DBL_EPSILON
            && fabs(out[1] - inverse_root_two) <= 4.0 * DBL_EPSILON,
            "tiny normalization preserves direction");

    subnormal[0] = true_minimum;
    CHECK(fastcd_norm_f64(&value, subnormal, dimension) == FASTCD_SUCCESS
            && value == true_minimum,
            "norm preserves the smallest positive subnormal");
    CHECK(fastcd_norm_squared_f64(&value, subnormal, dimension)
            == FASTCD_SUCCESS && value == 0.0,
            "subnormal squared norm follows IEEE underflow");
    CHECK(fastcd_normalize_f64(out, subnormal, dimension) == FASTCD_SUCCESS
            && out[0] == 1.0 && array_is(out + 1, dimension - 1, 0.0),
            "normalization does not mistake a subnormal for zero");
    CHECK(fastcd_reciprocal_f64(out, subnormal, dimension) == FASTCD_SUCCESS
            && isinf(out[0]) && out[0] > 0.0,
            "subnormal reciprocal follows IEEE overflow rather than zero-norm error");

    {
        double mixed_subnormal[8] = {
            true_minimum, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
        };
        double reciprocal[8];

        CHECK(fastcd_reciprocal_f64(reciprocal, mixed_subnormal, 8)
                == FASTCD_SUCCESS && reciprocal[0] == true_minimum,
                "reciprocal preserves a representable subnormal scalar coordinate");
        for (size_t i = 1; i < 8; i++) {
            CHECK(fabs(reciprocal[i] + 0.5 / 1.75) <= 2.0 * DBL_EPSILON,
                    "mixed-subnormal reciprocal coordinate %zu", i);
        }
    }

    CHECK(fastcd_norm_f64(&value, nan_element, dimension) == FASTCD_SUCCESS
            && isnan(value), "norm propagates NaN");
    CHECK(fastcd_norm_squared_f64(&value, nan_element, dimension)
            == FASTCD_SUCCESS && isnan(value), "squared norm propagates NaN");
    CHECK(fastcd_normalize_f64(out, nan_element, dimension) == FASTCD_SUCCESS
            && isnan(out[0]) && isnan(out[1]),
            "normalization propagates NaN");
    CHECK(fastcd_reciprocal_f64(out, nan_element, dimension) == FASTCD_SUCCESS
            && isnan(out[0]) && isnan(out[1]),
            "reciprocal propagates NaN");
    value = 0.0;
    CHECK(fastcd_inner_f64(&value, nan_element, nan_element, dimension)
            == FASTCD_SUCCESS && isnan(value), "inner product propagates NaN");
    CHECK(fastcd_square_f64(out, nan_element, dimension) == FASTCD_SUCCESS
            && isnan(out[0]) && isnan(out[1]),
            "square propagates NaN");

    CHECK(fastcd_norm_f64(&value, infinity_element, dimension) == FASTCD_SUCCESS
            && isinf(value) && value > 0.0, "norm propagates infinity");
    CHECK(fastcd_norm_squared_f64(&value, infinity_element, dimension)
            == FASTCD_SUCCESS && isinf(value) && value > 0.0,
            "squared norm propagates infinity");
    CHECK(fastcd_norm_f64(&value, negative_infinity, dimension)
            == FASTCD_SUCCESS && isinf(value) && value > 0.0,
            "norm of a negative infinite coordinate is +infinity");
    CHECK(fastcd_normalize_f64(out, infinity_element, dimension)
            == FASTCD_SUCCESS && isnan(out[0]) && out[1] == 0.0,
            "normalization follows IEEE infinity division");
    CHECK(fastcd_reciprocal_f64(out, infinity_element, dimension)
            == FASTCD_SUCCESS && isnan(out[0]) && out[1] == 0.0,
            "reciprocal follows IEEE infinity division");
    CHECK(fastcd_square_f64(out, infinity_element, dimension) == FASTCD_SUCCESS
            && isinf(out[0]) && out[0] > 0.0 && isinf(out[1]) && out[1] > 0.0,
            "square propagates an infinite scalar coordinate");
    {
        double nan_huge[4] = {NAN, DBL_MAX, 0.0, 0.0};

        CHECK(fastcd_square_f64(out, nan_huge, dimension) == FASTCD_SUCCESS
                && isnan(out[0]) && isnan(out[1]),
                "square with NaN and overflowing terms stays NaN");
    }
}

// Verify deliberate signed-zero behavior in coordinatewise structural calls
static void test_signed_zero(void) {
    const double x[4] = {-0.0, 0.0, -0.0, 1.0};
    const double zero[4] = {0.0, 0.0, 0.0, 0.0};
    const double negative_zero[4] = {-0.0, -0.0, -0.0, -0.0};
    double out[4];

    printf("\n=== signed zero ===\n");

    CHECK(fastcd_conjugate_f64(out, x, 4) == FASTCD_SUCCESS
            && copysign(1.0, out[0]) == -1.0 && copysign(1.0, out[1]) == -1.0
            && copysign(1.0, out[2]) == 1.0 && out[3] == -1.0,
            "conjugation preserves scalar zero and flips imaginary zero signs");
    CHECK(fastcd_neg_f64(out, x, 4) == FASTCD_SUCCESS
            && copysign(1.0, out[0]) == 1.0 && copysign(1.0, out[1]) == -1.0
            && copysign(1.0, out[2]) == 1.0 && out[3] == -1.0,
            "negation flips every signed zero");
    CHECK(fastcd_imag_part_f64(out, x, 4) == FASTCD_SUCCESS
            && out[0] == 0.0 && copysign(1.0, out[0]) == 1.0,
            "imaginary projection writes canonical positive scalar zero");
    CHECK(fastcd_scalar_f64(out, 4, -0.0) == FASTCD_SUCCESS
            && copysign(1.0, out[0]) == -1.0 && copysign(1.0, out[1]) == 1.0
            && copysign(1.0, out[2]) == 1.0 && copysign(1.0, out[3]) == 1.0,
            "scalar constructor stores the given zero and positive zeros");
    CHECK(fastcd_basis_f64(out, 4, 2) == FASTCD_SUCCESS
            && copysign(1.0, out[0]) == 1.0 && copysign(1.0, out[1]) == 1.0
            && copysign(1.0, out[3]) == 1.0 && out[2] == 1.0,
            "basis constructor writes positive zeros");
    CHECK(fastcd_add_f64(out, negative_zero, zero, 4) == FASTCD_SUCCESS
            && array_is(out, 4, 0.0) && copysign(1.0, out[0]) == 1.0,
            "-0 + +0 is +0");
    CHECK(fastcd_add_f64(out, negative_zero, negative_zero, 4) == FASTCD_SUCCESS
            && copysign(1.0, out[0]) == -1.0 && copysign(1.0, out[3]) == -1.0,
            "-0 + -0 is -0");
    CHECK(fastcd_scale_f64(out, zero, -1.0, 4) == FASTCD_SUCCESS
            && copysign(1.0, out[0]) == -1.0 && copysign(1.0, out[3]) == -1.0,
            "scaling +0 by -1 gives -0");
    CHECK(fastcd_square_f64(out, negative_zero, 4) == FASTCD_SUCCESS
            && array_is(out, 4, 0.0),
            "square of a signed-zero element is zero");
    CHECK(fastcd_square_f64(out, zero, 4) == FASTCD_SUCCESS
            && array_is(out, 4, 0.0) && copysign(1.0, out[0]) == 1.0,
            "square of zero is +0 in the scalar coordinate");
    CHECK(fastcd_normalize_f64(out, negative_zero, 4) == FASTCD_ZERO_NORM,
            "normalization treats -0 as zero norm");
    CHECK(fastcd_reciprocal_f64(out, negative_zero, 4) == FASTCD_ZERO_NORM,
            "reciprocal treats -0 as zero norm");
}

/* ---------- Squares ---------- */

// Compare the linear-time square with products at every dimension
static void test_square(void) {
    printf("\n=== squares ===\n");

    for (size_t level = 0; level <= 10; level++) {
        size_t dimension = (size_t)1 << level;
        double* block = allocate_block(4, dimension);
        double* x;
        double* expected;
        double* actual;
        double* alias;
        GuardedWorkspace workspace;

        CHECK(block != NULL, "allocate square fixtures N=%zu", dimension);
        if (block == NULL) continue;
        x = block;
        expected = block + dimension;
        actual = block + 2 * dimension;
        alias = block + 3 * dimension;
        if (!workspace_for_query(&workspace, dimension,
                    fastcd_mul_workspace_size)) {
            CHECK(false, "allocate square workspace N=%zu", dimension);
            free(block);
            continue;
        }

        for (size_t trial = 0; trial < 6; trial++) {
            bool integer = trial < 2;
            long double scale;

            if (integer) fill_small_integers(x, dimension);
            else fill_unit_interval(x, dimension);
            scale = euclidean_norm(x, dimension);
            scale *= scale;
            CHECK(fastcd_mul_f64(expected, x, x, dimension, &workspace.value)
                    == FASTCD_SUCCESS
                    && fastcd_square_f64(actual, x, dimension) == FASTCD_SUCCESS,
                    "square and product succeed N=%zu trial=%zu", dimension,
                    trial);
            if (integer) {
                double* oracle_out = expected;

                CHECK(oracle_mul(oracle_out, x, x, dimension)
                        && arrays_exact(actual, oracle_out, dimension),
                        "integer square is exact N=%zu trial=%zu", dimension,
                        trial);
            } else {
                CHECK(within_tolerance(actual, expected, dimension, scale),
                        "square matches the product N=%zu trial=%zu",
                        dimension, trial);
            }
            memcpy(alias, x, dimension * sizeof(double));
            CHECK(fastcd_square_f64(alias, alias, dimension) == FASTCD_SUCCESS
                    && arrays_bitwise(alias, actual, dimension),
                    "in-place square is identical N=%zu trial=%zu", dimension,
                    trial);
        }

        fill_array(actual, dimension, SENTINEL);
        CHECK(fastcd_square_f64(actual, NULL, dimension) == FASTCD_NULL_POINTER
                && array_is(actual, dimension, SENTINEL),
                "square rejects null input N=%zu", dimension);
        CHECK(fastcd_square_f64(NULL, x, dimension) == FASTCD_NULL_POINTER,
                "square rejects null output N=%zu", dimension);
        guarded_workspace_free(&workspace);
        free(block);
    }
    {
        double out[4] = {SENTINEL, SENTINEL, SENTINEL, SENTINEL};
        double x[4] = {1, 2, 3, 4};
        double expected[4] = {-28, 4, 6, 8};

        CHECK(fastcd_square_f64(out, x, 3) == FASTCD_INVALID_DIMENSION
                && array_is(out, 4, SENTINEL),
                "square rejects invalid dimension without writing");
        CHECK(fastcd_square_f64(out, x, 4) == FASTCD_SUCCESS
                && arrays_exact(out, expected, 4),
                "quaternion square by hand: (1+2i+3j+4k)^2");
    }
}

/* ---------- Basis products ---------- */

// Exhaustively test standard basis products for every algorithm through N=64
static void test_exhaustive_basis_products(void) {
    const size_t dimensions[] = {1, 2, 4, 8, 16, 32, 64};

    printf("\n=== exhaustive basis products ===\n");

    for (size_t d = 0; d < sizeof(dimensions) / sizeof(dimensions[0]); d++) {
        size_t dimension = dimensions[d];
        double* block = allocate_block(7, dimension);
        double* left;
        double* right;
        double* actual;
        double* oracle_out;
        double* expected;
        double* basis_actual;
        double* random_x;
        GuardedWorkspace workspaces[EXPLICIT_ALGORITHM_COUNT];
        GuardedWorkspace auto_workspace;
        bool ready = block != NULL;

        CHECK(block != NULL, "allocate basis fixtures N=%zu", dimension);
        if (block == NULL) continue;
        left = block;
        right = block + dimension;
        actual = block + 2 * dimension;
        oracle_out = block + 3 * dimension;
        expected = block + 4 * dimension;
        basis_actual = block + 5 * dimension;
        random_x = block + 6 * dimension;
        for (size_t a = 0; a < EXPLICIT_ALGORITHM_COUNT; a++) {
            if (!workspace_for_algorithm(&workspaces[a], dimension,
                        explicit_algorithms[a])) {
                ready = false;
            }
        }
        if (!workspace_for_query(&auto_workspace, dimension,
                    fastcd_mul_workspace_size)) {
            ready = false;
        }
        CHECK(ready, "allocate basis workspaces N=%zu", dimension);

        for (size_t i = 0; ready && i < dimension; i++) {
            for (size_t j = 0; j < dimension; j++) {
                FastCDBasisProduct product = {SIZE_MAX, 0};
                int sign = 0;

                fill_array(left, dimension, 0.0);
                fill_array(right, dimension, 0.0);
                left[i] = 1.0;
                right[j] = 1.0;
                fill_array(expected, dimension, 0.0);

                CHECK(reference_cd_basis_sign(dimension, i, j, &sign)
                        == FASTCD_SUCCESS && (sign == 1 || sign == -1),
                        "reference basis sign N=%zu i=%zu j=%zu", dimension, i,
                        j);
                expected[i ^ j] = (double)sign;
                CHECK(oracle_mul(oracle_out, left, right, dimension)
                        && arrays_exact(oracle_out, expected, dimension),
                        "oracle basis product N=%zu i=%zu j=%zu", dimension, i,
                        j);
                CHECK(fastcd_basis_product(&product, dimension, i, j)
                        == FASTCD_SUCCESS && product.index == (i ^ j)
                        && product.sign == sign,
                        "indexed basis product N=%zu i=%zu j=%zu", dimension, i,
                        j);
                for (size_t a = 0; a < EXPLICIT_ALGORITHM_COUNT; a++) {
                    CHECK(fastcd_mul_with_algorithm_f64(actual, left, right,
                                dimension, explicit_algorithms[a],
                                &workspaces[a].value) == FASTCD_SUCCESS
                            && arrays_exact(actual, expected, dimension),
                            "%s basis product N=%zu i=%zu j=%zu",
                            fastcd_algorithm_string(explicit_algorithms[a]),
                            dimension, i, j);
                }
                CHECK(fastcd_mul_f64(actual, left, right, dimension,
                            &auto_workspace.value) == FASTCD_SUCCESS
                        && arrays_exact(actual, expected, dimension),
                        "auto basis product N=%zu i=%zu j=%zu", dimension, i,
                        j);
                CHECK(fastcd_basis_lmul_f64(basis_actual, i, right, dimension)
                        == FASTCD_SUCCESS
                        && arrays_exact(basis_actual, expected, dimension),
                        "basis left action N=%zu i=%zu j=%zu", dimension, i, j);
                CHECK(fastcd_basis_rmul_f64(basis_actual, left, j, dimension)
                        == FASTCD_SUCCESS
                        && arrays_exact(basis_actual, expected, dimension),
                        "basis right action N=%zu i=%zu j=%zu", dimension, i,
                        j);
            }
        }

        fill_small_integers(random_x, dimension);
        for (size_t b = 0; ready && b < dimension; b++) {
            CHECK(fastcd_basis_f64(left, dimension, b) == FASTCD_SUCCESS
                    && oracle_mul(expected, left, random_x, dimension)
                    && fastcd_basis_lmul_f64(basis_actual, b, random_x,
                        dimension) == FASTCD_SUCCESS
                    && arrays_exact(basis_actual, expected, dimension),
                    "basis left action on a dense element N=%zu b=%zu",
                    dimension, b);
            CHECK(oracle_mul(expected, random_x, left, dimension)
                    && fastcd_basis_rmul_f64(basis_actual, random_x, b,
                        dimension) == FASTCD_SUCCESS
                    && arrays_exact(basis_actual, expected, dimension),
                    "basis right action on a dense element N=%zu b=%zu",
                    dimension, b);
        }

        for (size_t a = 0; a < EXPLICIT_ALGORITHM_COUNT; a++) {
            guarded_workspace_free(&workspaces[a]);
        }
        guarded_workspace_free(&auto_workspace);
        free(block);
    }
}

// Test indexed-basis argument validation and unchanged-output behavior
static void test_basis_validation(void) {
    double x[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    double out[8];
    FastCDBasisProduct product = {77, 3};

    printf("\n=== basis validation ===\n");

    CHECK(fastcd_basis_product(NULL, 8, 0, 0) == FASTCD_NULL_POINTER,
            "basis product rejects null output");
    CHECK(fastcd_basis_product(&product, 3, 0, 0) == FASTCD_INVALID_DIMENSION
            && product.index == 77 && product.sign == 3,
            "basis product preserves output for invalid dimension");
    CHECK(fastcd_basis_product(&product, 8, 8, 0) == FASTCD_INVALID_BASIS_INDEX
            && product.index == 77 && product.sign == 3,
            "basis product preserves output for invalid left index");
    CHECK(fastcd_basis_product(&product, 8, 0, 8) == FASTCD_INVALID_BASIS_INDEX
            && product.index == 77 && product.sign == 3,
            "basis product preserves output for invalid right index");
    CHECK(fastcd_basis_product(&product, 1, 0, 0) == FASTCD_SUCCESS
            && product.index == 0 && product.sign == 1,
            "basis product at dimension one is 1 * 1 = 1");

    fill_array(out, 8, SENTINEL);
    CHECK(fastcd_basis_lmul_f64(out, 8, x, 8) == FASTCD_INVALID_BASIS_INDEX
            && array_is(out, 8, SENTINEL),
            "basis left action preserves output for invalid index");
    CHECK(fastcd_basis_rmul_f64(out, x, 8, 8) == FASTCD_INVALID_BASIS_INDEX
            && array_is(out, 8, SENTINEL),
            "basis right action preserves output for invalid index");
    CHECK(fastcd_basis_lmul_f64(out, 0, x, 6) == FASTCD_INVALID_DIMENSION
            && array_is(out, 8, SENTINEL),
            "basis left action rejects invalid dimension");
    CHECK(fastcd_basis_rmul_f64(out, x, 0, 6) == FASTCD_INVALID_DIMENSION
            && array_is(out, 8, SENTINEL),
            "basis right action rejects invalid dimension");
    CHECK(fastcd_basis_lmul_f64(NULL, 1, x, 8) == FASTCD_NULL_POINTER,
            "basis left action rejects null output");
    CHECK(fastcd_basis_lmul_f64(out, 1, NULL, 8) == FASTCD_NULL_POINTER
            && array_is(out, 8, SENTINEL),
            "basis left action rejects null input");
    CHECK(fastcd_basis_rmul_f64(NULL, x, 1, 8) == FASTCD_NULL_POINTER,
            "basis right action rejects null output");
    CHECK(fastcd_basis_rmul_f64(out, NULL, 1, 8) == FASTCD_NULL_POINTER
            && array_is(out, 8, SENTINEL),
            "basis right action rejects null input");
    CHECK(fastcd_basis_lmul_f64(x, 1, x, 8) == FASTCD_OVERLAPPING_BUFFERS,
            "basis left action rejects exact alias");
    CHECK(fastcd_basis_rmul_f64(x, x, 1, 8) == FASTCD_OVERLAPPING_BUFFERS,
            "basis right action rejects exact alias");
    CHECK(fastcd_basis_lmul_f64(out, 0, x, 8) == FASTCD_SUCCESS
            && arrays_exact(out, x, 8), "left action of one is the identity");
    CHECK(fastcd_basis_rmul_f64(out, x, 0, 8) == FASTCD_SUCCESS
            && arrays_exact(out, x, 8), "right action of one is the identity");
}

/* ---------- Randomized products ---------- */

// Compare dense small-integer products exactly with the oracle
static void test_random_integer_products(void) {
    size_t products_checked = 0;

    printf("\n=== randomized exact products ===\n");

    for (size_t level = 0; level <= 10; level++) {
        size_t dimension = (size_t)1 << level;
        size_t trials = dimension <= 64 ? 32 : dimension <= 256 ? 16
            : dimension == 512 ? 8 : 4;
        double* block = allocate_block(3 + MUL_ROUTE_COUNT, dimension);
        double* x;
        double* y;
        double* expected;
        double* results;
        GuardedWorkspace workspaces[MUL_ROUTE_COUNT];
        bool supported[MUL_ROUTE_COUNT];
        FastCDAlgorithm default_algorithm = FASTCD_ALGORITHM_AUTO;
        bool ready = block != NULL;

        CHECK(block != NULL, "allocate random fixtures N=%zu", dimension);
        if (block == NULL) continue;
        x = block;
        y = block + dimension;
        expected = block + 2 * dimension;
        results = block + 3 * dimension;
        CHECK(fastcd_default_algorithm(dimension, &default_algorithm)
                == FASTCD_SUCCESS, "default algorithm N=%zu", dimension);
        for (size_t r = 0; r < MUL_ROUTE_COUNT; r++) {
            supported[r] = fastcd_algorithm_supports_dimension(
                    mul_routes[r].algorithm, dimension);
            CHECK(supported[r]
                    || (mul_routes[r].algorithm
                        == FASTCD_ALGORITHM_CARIOW_CARIOWA
                        && dimension > FASTCD_CARIOW_CARIOWA_MAX_DIMENSION),
                    "%s is available at N=%zu", mul_routes[r].name, dimension);
            workspaces[r].storage = NULL;
            workspaces[r].required = 0;
            workspaces[r].value.data = NULL;
            workspaces[r].value.length = 0;
            if (supported[r] && !workspace_for_algorithm(&workspaces[r],
                        dimension, mul_routes[r].algorithm)) {
                ready = false;
            }
        }
        CHECK(ready, "allocate random workspaces N=%zu", dimension);

        for (size_t trial = 0; ready && trial < trials; trial++) {
            fill_small_integers(x, dimension);
            fill_small_integers(y, dimension);
            CHECK(oracle_mul(expected, x, y, dimension),
                    "oracle product N=%zu trial=%zu", dimension, trial);
            for (size_t r = 0; r < MUL_ROUTE_COUNT; r++) {
                double* actual = results + r * dimension;

                if (!supported[r]) continue;
                fill_array(actual, dimension, SENTINEL);
                CHECK(mul_route_call(&mul_routes[r], actual, x, y, dimension,
                            &workspaces[r].value) == FASTCD_SUCCESS
                        && arrays_exact(actual, expected, dimension),
                        "%s exact product N=%zu trial=%zu", mul_routes[r].name,
                        dimension, trial);
                CHECK(guarded_workspace_intact(&workspaces[r]),
                        "%s preserves workspace guards N=%zu trial=%zu",
                        mul_routes[r].name, dimension, trial);
            }
            CHECK(arrays_bitwise(results, results + dimension, dimension),
                    "public and explicit AUTO agree bitwise N=%zu trial=%zu",
                    dimension, trial);
            for (size_t r = 2; r < MUL_ROUTE_COUNT; r++) {
                if (mul_routes[r].algorithm == default_algorithm) {
                    CHECK(arrays_bitwise(results, results + r * dimension,
                                dimension),
                            "AUTO is bitwise identical to %s N=%zu trial=%zu",
                            mul_routes[r].name, dimension, trial);
                }
            }
            products_checked++;
        }

        for (size_t r = 0; r < MUL_ROUTE_COUNT; r++) {
            guarded_workspace_free(&workspaces[r]);
        }
        free(block);
    }
    CHECK(products_checked >= 200,
            "checked at least 200 randomized exact products (got %zu)",
            products_checked);
}

// Compare random floating products with the oracle under the norm tolerance
static void test_random_floating_products(void) {
    const size_t trials = 8;
    long double worst[MUL_ROUTE_COUNT];

    printf("\n=== randomized floating products ===\n");

    for (size_t r = 0; r < MUL_ROUTE_COUNT; r++) worst[r] = 0.0L;
    for (size_t level = 0; level <= 10; level++) {
        size_t dimension = (size_t)1 << level;
        double* block = allocate_block(4, dimension);
        double* x;
        double* y;
        double* expected;
        double* actual;
        GuardedWorkspace workspaces[MUL_ROUTE_COUNT];
        bool supported[MUL_ROUTE_COUNT];
        bool ready = block != NULL;

        CHECK(block != NULL, "allocate floating fixtures N=%zu", dimension);
        if (block == NULL) continue;
        x = block;
        y = block + dimension;
        expected = block + 2 * dimension;
        actual = block + 3 * dimension;
        for (size_t r = 0; r < MUL_ROUTE_COUNT; r++) {
            supported[r] = fastcd_algorithm_supports_dimension(
                    mul_routes[r].algorithm, dimension);
            workspaces[r].storage = NULL;
            workspaces[r].required = 0;
            workspaces[r].value.data = NULL;
            workspaces[r].value.length = 0;
            if (supported[r] && !workspace_for_algorithm(&workspaces[r],
                        dimension, mul_routes[r].algorithm)) {
                ready = false;
            }
        }
        CHECK(ready, "allocate floating workspaces N=%zu", dimension);

        for (size_t trial = 0; ready && trial < trials; trial++) {
            long double scale;
            long double tolerance;

            fill_unit_interval(x, dimension);
            fill_unit_interval(y, dimension);
            if (trial == trials - 1) fill_array(y, dimension, 0.0);
            scale = euclidean_norm(x, dimension) * euclidean_norm(y, dimension);
            tolerance = scaled_tolerance(dimension, scale);
            CHECK(oracle_mul(expected, x, y, dimension),
                    "oracle floating product N=%zu trial=%zu", dimension,
                    trial);
            for (size_t r = 0; r < MUL_ROUTE_COUNT; r++) {
                long double difference;

                if (!supported[r]) continue;
                CHECK(mul_route_call(&mul_routes[r], actual, x, y, dimension,
                            &workspaces[r].value) == FASTCD_SUCCESS
                        && arrays_finite(actual, dimension),
                        "%s floating product is finite N=%zu trial=%zu",
                        mul_routes[r].name, dimension, trial);
                difference = max_abs_difference(actual, expected, dimension);
                CHECK(scale == 0.0L ? difference == 0.0L
                        : difference <= tolerance,
                        "%s floating tolerance N=%zu trial=%zu "
                        "(error %.3Le, limit %.3Le)", mul_routes[r].name,
                        dimension, trial, difference, tolerance);
                if (scale > 0.0L && tolerance > 0.0L
                        && difference / tolerance > worst[r]) {
                    worst[r] = difference / tolerance;
                }
            }
        }

        for (size_t r = 0; r < MUL_ROUTE_COUNT; r++) {
            guarded_workspace_free(&workspaces[r]);
        }
        free(block);
    }
    for (size_t r = 0; r < MUL_ROUTE_COUNT; r++) {
        printf("  %s: worst error / tolerance = %.4Lf\n", mul_routes[r].name,
                worst[r]);
    }
}

/* ---------- Numerical fixtures ---------- */

// Generate reproducible dense, noninteger coefficients in [-1,1]
static double numerical_random(uint64_t* state) {
    uint64_t mantissa;

    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    mantissa = (*state * UINT64_C(2685821657736338717)) >> 11;
    return 2.0 * (double)mantissa / 9007199254740992.0 - 1.0;
}

/*
 * Sum the signed basis products for each coordinate, independently of every
 * multiplication recurrence.  Compensation reduces accumulation error even
 * on platforms where long double has the same precision as double.  The
 * fixtures keep all products and partial sums in the normal double range.
 */
// Compute a compensated direct basis-product reference in long double
static void numerical_reference(long double* out, const double* x,
        const double* y, const signed char* signs, size_t dimension) {
    for (size_t k = 0; k < dimension; k++) {
        long double sum = 0.0L;
        long double correction = 0.0L;

        for (size_t i = 0; i < dimension; i++) {
            size_t j = i ^ k;
            long double product = (long double)x[i] * (long double)y[j];
            long double next;

            product *= (long double)signs[i * dimension + j];
            next = sum + product;
            if (fabsl(sum) >= fabsl(product)) {
                correction += (sum - next) + product;
            } else {
                correction += (product - next) + sum;
            }
            sum = next;
        }
        out[k] = sum + correction;
    }
}

/*
 * The smallest dense products are approximately 2^-960; the largest are
 * approximately 2^960.  Reciprocal operands separately reach 2^900 and
 * 2^-900.  No fixture relies on overflow, subnormal preservation, or NaN
 * semantics.  Zero and near-zero output coordinates are intentionally common.
 */
// Fill one deterministic finite-range or cancellation fixture
static void numerical_fill(double* x, double* y, size_t dimension,
        size_t fixture, uint64_t* state) {
    for (size_t i = 0; i < dimension; i++) {
        double a = numerical_random(state);
        double b = numerical_random(state);

        switch (fixture) {
            case 1:
                x[i] = a * 0x1p-480;
                y[i] = b * 0x1p-480;
                break;
            case 2:
                x[i] = a * 0x1p+480;
                y[i] = b * 0x1p+480;
                break;
            case 3:
                x[i] = a * 0x1p+900;
                y[i] = b * 0x1p-900;
                break;
            case 4:
                x[i] = a * 0x1p-900;
                y[i] = b * 0x1p+900;
                break;
            case 5:
                x[i] = a;
                y[i] = a;
                break;
            case 6:
                x[i] = a;
                y[i] = a * (1.0 + 0x1p-20) + b * 0x1p-35;
                break;
            case 7:
                x[i] = a * (i % 3 == 0 ? 0x1p400 : 0x1p-400);
                y[i] = b * (i % 5 == 0 ? 0x1p400 : 0x1p-400);
                break;
            case 8:
                x[i] = a * (i == 0 ? 1.0 : 0x1p-40);
                y[i] = b * (i == 0 ? 0x1p-40 : 1.0);
                break;
            default:
                x[i] = a;
                y[i] = b;
                break;
        }
    }
    if ((fixture == 5 || fixture == 6) && dimension > 1) {
        x[0] = 0.0;
        y[0] = 0.0;
    }
}

// Check multiplication accuracy on cancellation and extreme-range fixtures
static void test_numerical_fixtures(void) {
    const char* const fixture_names[] = {
        "dense", "tiny", "large", "large-small", "small-large",
        "imaginary square", "nearly commuting", "mixed scales",
        "scalar-imaginary"
    };
    const size_t fixture_count = sizeof(fixture_names) / sizeof(fixture_names[0]);
    uint64_t state = UINT64_C(0xb711bfbf9573a087);
    size_t products = 0;
    long double maximum_error[MUL_ROUTE_COUNT];

    printf("\n=== multiplication accuracy and finite range ===\n");

    for (size_t r = 0; r < MUL_ROUTE_COUNT; r++) maximum_error[r] = 0.0L;
    for (size_t level = 0; level <= 10; level++) {
        size_t dimension = (size_t)1 << level;
        double* block = allocate_block(3, dimension);
        long double* expected = malloc(dimension * sizeof(long double));
        signed char* signs = malloc(dimension * dimension * sizeof(signed char));
        double* x;
        double* y;
        double* actual;
        GuardedWorkspace workspaces[MUL_ROUTE_COUNT];
        bool supported[MUL_ROUTE_COUNT];
        bool ready = block != NULL && expected != NULL && signs != NULL;

        for (size_t r = 0; r < MUL_ROUTE_COUNT; r++) {
            supported[r] = fastcd_algorithm_supports_dimension(
                    mul_routes[r].algorithm, dimension);
            workspaces[r].storage = NULL;
            workspaces[r].required = 0;
            workspaces[r].value.data = NULL;
            workspaces[r].value.length = 0;
            if (ready && supported[r] && !workspace_for_algorithm(
                        &workspaces[r], dimension, mul_routes[r].algorithm)) {
                ready = false;
            }
        }
        x = block;
        y = block == NULL ? NULL : block + dimension;
        actual = block == NULL ? NULL : block + 2 * dimension;
        for (size_t i = 0; ready && i < dimension; i++) {
            for (size_t j = 0; j < dimension; j++) {
                int sign;

                if (reference_cd_basis_sign(dimension, i, j, &sign)
                        != FASTCD_SUCCESS) {
                    ready = false;
                    break;
                }
                signs[i * dimension + j] = (signed char)sign;
            }
        }
        CHECK(ready, "numerical fixtures ready at dimension %zu", dimension);

        for (size_t fixture = 0; ready && fixture < fixture_count; fixture++) {
            for (size_t trial = 0; trial < 2; trial++) {
                long double scale;
                long double tolerance = 64.0L * (long double)DBL_EPSILON
                    * (long double)(level + 1);

                numerical_fill(x, y, dimension, fixture, &state);
                numerical_reference(expected, x, y, signs, dimension);
                scale = euclidean_norm(x, dimension)
                    * euclidean_norm(y, dimension);
                for (size_t r = 0; r < MUL_ROUTE_COUNT; r++) {
                    FastCDStatus status;
                    bool matched;
                    size_t failed_coordinate = 0;
                    long double error = 0.0L;

                    if (!supported[r]) continue;
                    status = mul_route_call(&mul_routes[r], actual, x, y,
                            dimension, &workspaces[r].value);
                    matched = status == FASTCD_SUCCESS;
                    products++;
                    for (size_t i = 0; matched && i < dimension; i++) {
                        long double difference = fabsl((long double)actual[i]
                                - expected[i]);

                        error = scale == 0.0L ? difference : difference / scale;
                        if (!isfinite(actual[i]) || !isfinite(expected[i])
                                || !isfinite(error)
                                || (scale == 0.0L ? difference != 0.0L
                                    : error > tolerance)) {
                            matched = false;
                            failed_coordinate = i;
                        }
                        if (error > maximum_error[r]) maximum_error[r] = error;
                    }
                    CHECK(matched, "%s %s N=%zu trial=%zu coordinate=%zu "
                            "status=%d norm-scaled error=%.6Le limit=%.6Le",
                            mul_routes[r].name, fixture_names[fixture],
                            dimension, trial, failed_coordinate, (int)status,
                            error, tolerance);
                }
            }
        }
        for (size_t r = 0; r < MUL_ROUTE_COUNT; r++) {
            guarded_workspace_free(&workspaces[r]);
        }
        free(signs);
        free(expected);
        free(block);
    }
    printf("  %zu products; maximum error / (epsilon ||x||2 ||y||2):\n",
            products);
    for (size_t r = 0; r < MUL_ROUTE_COUNT; r++) {
        printf("    %s %.6Lf\n", mul_routes[r].name,
                maximum_error[r] / (long double)DBL_EPSILON);
    }
}

/* ---------- Batched multiplication ---------- */

// Check every completed batch element against scalar calls and the oracle
static bool batch_matches(const MulRoute* route, const double* out,
        size_t out_stride, const double* x, size_t x_stride, const double* y,
        size_t y_stride, size_t count, size_t dimension,
        const FastCDWorkspaceF64* workspace, double* scalar, double* expected) {
    /* An automatic batch of two or more records follows the batch selection,
     * so reproduce it with that concrete algorithm rather than the
     * single-product default. */
    FastCDAlgorithm batch_algorithm = FASTCD_ALGORITHM_AUTO;
    bool use_batch_default = route->algorithm == FASTCD_ALGORITHM_AUTO
        && count >= 2
        && fastcd_default_batch_algorithm(dimension, &batch_algorithm)
            == FASTCD_SUCCESS;

    for (size_t k = 0; k < count; k++) {
        const double* xk = x + k * x_stride;
        const double* yk = y + k * y_stride;
        const double* outk = out + k * out_stride;
        FastCDStatus status = use_batch_default
            ? fastcd_mul_with_algorithm_f64(scalar, xk, yk, dimension,
                    batch_algorithm, workspace)
            : mul_route_call(route, scalar, xk, yk, dimension, workspace);

        if (status != FASTCD_SUCCESS) return false;
        if (!arrays_bitwise(outk, scalar, dimension)) return false;
        if (!oracle_mul(expected, xk, yk, dimension)
                || !arrays_exact(outk, expected, dimension)) {
            return false;
        }
    }
    return true;
}

// Return whether every padding word between batch elements holds one value
static bool padding_is(const double* out, size_t out_stride, size_t count,
        size_t dimension, double value) {
    for (size_t k = 0; k < count; k++) {
        if (!array_is(out + k * out_stride + dimension, out_stride - dimension,
                    value)) {
            return false;
        }
    }
    return true;
}

// Test strided, broadcast, singleton, empty, and invalid batches per algorithm
static void test_batch_multiplication(void) {
    const size_t dimensions[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    const size_t dimension_count = sizeof(dimensions) / sizeof(dimensions[0]);
    const size_t count = 5;
    const size_t huge_dimension = (size_t)1 << (sizeof(size_t) * CHAR_BIT - 1);

    printf("\n=== batched multiplication ===\n");

    for (size_t r = 0; r < MUL_ROUTE_COUNT; r++) {
        const MulRoute* route = &mul_routes[r];
        bool cariow = route->algorithm == FASTCD_ALGORITHM_CARIOW_CARIOWA;

        CHECK(mul_route_batch(route, NULL, 0, NULL, 0, NULL, 0, 0, 8, NULL)
                == FASTCD_SUCCESS, "%s empty batch accepts null storage",
                route->name);
        CHECK(mul_route_batch(route, NULL, SIZE_MAX, NULL, SIZE_MAX, NULL,
                    SIZE_MAX, 0, 64, NULL) == FASTCD_SUCCESS,
                "%s empty batch ignores strides", route->name);
        CHECK(mul_route_batch(route, NULL, 0, NULL, 0, NULL, 0, 0, 3, NULL)
                == FASTCD_INVALID_DIMENSION,
                "%s empty batch still validates dimension", route->name);
        CHECK(mul_route_batch(route, NULL, 0, NULL, 0, NULL, 0, 0, 0, NULL)
                == FASTCD_INVALID_DIMENSION,
                "%s empty batch rejects dimension zero", route->name);
        CHECK(mul_route_batch(route, NULL, 0, NULL, 0, NULL, 0, 0,
                    2 * FASTCD_CARIOW_CARIOWA_MAX_DIMENSION, NULL)
                == (cariow ? FASTCD_UNSUPPORTED_ALGORITHM : FASTCD_SUCCESS),
                "%s empty batch validates the algorithm above the CC limit",
                route->name);
        CHECK(mul_route_batch(route, NULL, 0, NULL, 0, NULL, 0, 0,
                    huge_dimension, NULL)
                == (cariow ? FASTCD_UNSUPPORTED_ALGORITHM : FASTCD_SUCCESS),
                "%s empty batch validates only dimension and algorithm",
                route->name);

        for (size_t d = 0; d < dimension_count; d++) {
            size_t dimension = dimensions[d];
            size_t x_stride = dimension + 2;
            size_t y_stride = dimension + 3;
            size_t out_stride = dimension + 4;
            size_t batch_count = dimension >= 512 ? 3 : count;
            double* x;
            double* y;
            double* out;
            double* contiguous_x;
            double* contiguous_y;
            double* contiguous_out;
            double* scalar;
            double* expected;
            double* block;
            GuardedWorkspace workspace;

            if (!fastcd_algorithm_supports_dimension(route->algorithm,
                        dimension)) {
                double* pair = allocate_block(3, 2 * dimension);

                CHECK(cariow && dimension > FASTCD_CARIOW_CARIOWA_MAX_DIMENSION,
                        "%s is only unavailable above the CC limit N=%zu",
                        route->name, dimension);
                CHECK(pair != NULL, "allocate %s unsupported fixtures N=%zu",
                        route->name, dimension);
                if (pair == NULL) continue;
                fill_small_integers(pair, 4 * dimension);
                fill_array(pair + 4 * dimension, 2 * dimension, SENTINEL);
                CHECK(mul_route_batch(route, pair + 4 * dimension, dimension,
                            pair, dimension, pair + 2 * dimension, dimension, 2,
                            dimension, NULL) == FASTCD_UNSUPPORTED_ALGORITHM
                        && array_is(pair + 4 * dimension, 2 * dimension,
                            SENTINEL),
                        "%s batch is unsupported without writing N=%zu",
                        route->name, dimension);
                free(pair);
                continue;
            }

            block = allocate_block(1, count * (x_stride + y_stride + out_stride)
                    + (3 * count + 2) * dimension);
            CHECK(block != NULL, "allocate %s batch fixtures N=%zu",
                    route->name, dimension);
            if (block == NULL) continue;
            x = block;
            y = x + count * x_stride;
            out = y + count * y_stride;
            contiguous_x = out + count * out_stride;
            contiguous_y = contiguous_x + count * dimension;
            contiguous_out = contiguous_y + count * dimension;
            scalar = contiguous_out + count * dimension;
            expected = scalar + dimension;
            if (!workspace_for_algorithm(&workspace, dimension,
                        route->algorithm)) {
                CHECK(false, "allocate %s batch workspace N=%zu", route->name,
                        dimension);
                free(block);
                continue;
            }

            fill_array(x, count * x_stride, 91.0);
            fill_array(y, count * y_stride, 92.0);
            fill_array(out, count * out_stride, 93.0);
            for (size_t k = 0; k < count; k++) {
                fill_small_integers(x + k * x_stride, dimension);
                fill_small_integers(y + k * y_stride, dimension);
                memcpy(contiguous_x + k * dimension, x + k * x_stride,
                        dimension * sizeof(double));
                memcpy(contiguous_y + k * dimension, y + k * y_stride,
                        dimension * sizeof(double));
            }

            guarded_workspace_arm(&workspace);
            CHECK(mul_route_batch(route, out, out_stride, x, x_stride, y,
                        y_stride, batch_count, dimension, &workspace.value)
                    == FASTCD_SUCCESS
                    && batch_matches(route, out, out_stride, x, x_stride, y,
                        y_stride, batch_count, dimension, &workspace.value,
                        scalar, expected),
                    "%s padded strides N=%zu", route->name, dimension);
            CHECK(padding_is(out, out_stride, batch_count, dimension, 93.0),
                    "%s preserves output padding N=%zu", route->name, dimension);
            CHECK(guarded_workspace_intact(&workspace),
                    "%s preserves workspace guards N=%zu", route->name,
                    dimension);

            fill_array(contiguous_out, count * dimension, 93.0);
            CHECK(mul_route_batch(route, contiguous_out, dimension,
                        contiguous_x, dimension, contiguous_y, dimension,
                        batch_count, dimension, &workspace.value)
                    == FASTCD_SUCCESS
                    && batch_matches(route, contiguous_out, dimension,
                        contiguous_x, dimension, contiguous_y, dimension,
                        batch_count, dimension, &workspace.value, scalar,
                        expected),
                    "%s contiguous strides N=%zu", route->name, dimension);

            fill_array(out, count * out_stride, 93.0);
            CHECK(mul_route_batch(route, out, out_stride, x, 0, y, y_stride,
                        batch_count, dimension, &workspace.value)
                    == FASTCD_SUCCESS
                    && batch_matches(route, out, out_stride, x, 0, y, y_stride,
                        batch_count, dimension, &workspace.value, scalar,
                        expected)
                    && padding_is(out, out_stride, batch_count, dimension, 93.0),
                    "%s broadcasts x N=%zu", route->name, dimension);
            fill_array(out, count * out_stride, 93.0);
            CHECK(mul_route_batch(route, out, out_stride, x, x_stride, y, 0,
                        batch_count, dimension, &workspace.value)
                    == FASTCD_SUCCESS
                    && batch_matches(route, out, out_stride, x, x_stride, y, 0,
                        batch_count, dimension, &workspace.value, scalar,
                        expected)
                    && padding_is(out, out_stride, batch_count, dimension, 93.0),
                    "%s broadcasts y N=%zu", route->name, dimension);
            fill_array(out, count * out_stride, 93.0);
            CHECK(mul_route_batch(route, out, out_stride, x, 0, y, 0,
                        batch_count, dimension, &workspace.value)
                    == FASTCD_SUCCESS
                    && batch_matches(route, out, out_stride, x, 0, y, 0,
                        batch_count, dimension, &workspace.value, scalar,
                        expected),
                    "%s broadcasts both inputs N=%zu", route->name, dimension);

            fill_array(out, count * out_stride, 93.0);
            CHECK(mul_route_batch(route, out, out_stride, x, x_stride, x,
                        x_stride, batch_count, dimension, &workspace.value)
                    == FASTCD_SUCCESS
                    && batch_matches(route, out, out_stride, x, x_stride, x,
                        x_stride, batch_count, dimension, &workspace.value,
                        scalar, expected),
                    "%s accepts identical complete input spans N=%zu",
                    route->name, dimension);

            fill_array(out, count * out_stride, 93.0);
            CHECK(mul_route_batch(route, out, SIZE_MAX, x, SIZE_MAX, y, 0, 1,
                        dimension, &workspace.value) == FASTCD_SUCCESS
                    && batch_matches(route, out, 0, x, 0, y, 0, 1, dimension,
                        &workspace.value, scalar, expected)
                    && array_is(out + dimension, count * out_stride - dimension,
                        93.0),
                    "%s singleton batch ignores strides N=%zu", route->name,
                    dimension);
            fill_array(out, count * out_stride, 93.0);
            CHECK(mul_route_batch(route, out, 0, x, 1, x, SIZE_MAX, 1,
                        dimension, &workspace.value) == FASTCD_SUCCESS
                    && batch_matches(route, out, 0, x, 0, x, 0, 1, dimension,
                        &workspace.value, scalar, expected),
                    "%s singleton batch accepts identical inputs N=%zu",
                    route->name, dimension);

            fill_array(out, count * out_stride, 93.0);
            CHECK(mul_route_batch(route, out, 0, x, x_stride, y, y_stride,
                        batch_count, dimension, &workspace.value)
                    == FASTCD_INVALID_STRIDE
                    && array_is(out, count * out_stride, 93.0),
                    "%s rejects a zero output stride N=%zu", route->name,
                    dimension);
            CHECK(mul_route_batch(route, out, dimension - 1, x, x_stride, y,
                        y_stride, batch_count, dimension, &workspace.value)
                    == FASTCD_INVALID_STRIDE
                    && array_is(out, count * out_stride, 93.0),
                    "%s rejects a short output stride N=%zu", route->name,
                    dimension);
            if (dimension > 1) {
                CHECK(mul_route_batch(route, out, out_stride, x, dimension - 1,
                            y, y_stride, batch_count, dimension,
                            &workspace.value) == FASTCD_INVALID_STRIDE
                        && array_is(out, count * out_stride, 93.0),
                        "%s rejects a short x stride N=%zu", route->name,
                        dimension);
                CHECK(mul_route_batch(route, out, out_stride, x, x_stride, y,
                            dimension - 1, batch_count, dimension,
                            &workspace.value) == FASTCD_INVALID_STRIDE
                        && array_is(out, count * out_stride, 93.0),
                        "%s rejects a short y stride N=%zu", route->name,
                        dimension);
            }

            CHECK(mul_route_batch(route, x, x_stride, x, x_stride, y, y_stride,
                        batch_count, dimension, &workspace.value)
                    == FASTCD_OVERLAPPING_BUFFERS,
                    "%s rejects output aliasing x N=%zu", route->name,
                    dimension);
            CHECK(mul_route_batch(route, y, y_stride, x, x_stride, y, y_stride,
                        batch_count, dimension, &workspace.value)
                    == FASTCD_OVERLAPPING_BUFFERS,
                    "%s rejects output aliasing y N=%zu", route->name,
                    dimension);
            CHECK(mul_route_batch(route, x + 1, x_stride, x, x_stride, y,
                        y_stride, batch_count, dimension, &workspace.value)
                    == FASTCD_OVERLAPPING_BUFFERS,
                    "%s rejects partial output/input overlap N=%zu",
                    route->name, dimension);
            if (dimension > 1) {
                CHECK(mul_route_batch(route, out, out_stride, x, x_stride,
                            x + 1, x_stride, batch_count, dimension,
                            &workspace.value) == FASTCD_OVERLAPPING_BUFFERS
                        && array_is(out, count * out_stride, 93.0),
                        "%s rejects partially overlapping input records N=%zu",
                        route->name, dimension);
            }
            fill_array(out, count * out_stride, 93.0);
            CHECK(mul_route_batch(route, out, out_stride, x, x_stride, x, 0,
                        batch_count, dimension, &workspace.value)
                    == FASTCD_SUCCESS
                    && batch_matches(route, out, out_stride, x, x_stride, x, 0,
                        batch_count, dimension, &workspace.value, scalar,
                        expected)
                    && padding_is(out, out_stride, batch_count, dimension, 93.0),
                    "%s accepts one input broadcasting a record of the other N=%zu",
                    route->name, dimension);
            fill_array(out, count * out_stride, 93.0);
            CHECK(mul_route_batch(route, out, out_stride, x, x_stride,
                        x + x_stride, x_stride, batch_count - 1, dimension,
                        &workspace.value) == FASTCD_SUCCESS
                    && batch_matches(route, out, out_stride, x, x_stride,
                        x + x_stride, x_stride, batch_count - 1, dimension,
                        &workspace.value, scalar, expected)
                    && padding_is(out, out_stride, batch_count - 1, dimension,
                        93.0),
                    "%s accepts adjacent windows of one input array N=%zu",
                    route->name, dimension);
            fill_array(out, count * out_stride, 93.0);

            CHECK(mul_route_batch(route, NULL, out_stride, x, x_stride, y,
                        y_stride, batch_count, dimension, &workspace.value)
                    == FASTCD_NULL_POINTER, "%s rejects a null output N=%zu",
                    route->name, dimension);
            CHECK(mul_route_batch(route, out, out_stride, NULL, x_stride, y,
                        y_stride, batch_count, dimension, &workspace.value)
                    == FASTCD_NULL_POINTER
                    && array_is(out, count * out_stride, 93.0),
                    "%s rejects a null x N=%zu", route->name, dimension);
            CHECK(mul_route_batch(route, out, out_stride, x, x_stride, NULL,
                        y_stride, batch_count, dimension, &workspace.value)
                    == FASTCD_NULL_POINTER
                    && array_is(out, count * out_stride, 93.0),
                    "%s rejects a null y N=%zu", route->name, dimension);
            CHECK(mul_route_batch(route, out, out_stride, x, x_stride, y,
                        y_stride, 2, 3, &workspace.value)
                    == FASTCD_INVALID_DIMENSION
                    && array_is(out, count * out_stride, 93.0),
                    "%s rejects an invalid dimension N=%zu", route->name,
                    dimension);
            CHECK(mul_route_batch(route, out, SIZE_MAX, x, 0, y, 0, 2,
                        dimension, &workspace.value) == FASTCD_SIZE_OVERFLOW
                    && array_is(out, count * out_stride, 93.0),
                    "%s rejects output span overflow N=%zu", route->name,
                    dimension);
            CHECK(mul_route_batch(route, out, out_stride, x, SIZE_MAX, y, 0, 2,
                        dimension, &workspace.value) == FASTCD_SIZE_OVERFLOW
                    && array_is(out, count * out_stride, 93.0),
                    "%s rejects x span overflow N=%zu", route->name, dimension);
            CHECK(mul_route_batch(route, out, out_stride, x, 0, y, SIZE_MAX, 2,
                        dimension, &workspace.value) == FASTCD_SIZE_OVERFLOW
                    && array_is(out, count * out_stride, 93.0),
                    "%s rejects y span overflow N=%zu", route->name, dimension);

            if (workspace.required > 0) {
                FastCDWorkspaceF64 short_workspace = {workspace.value.data,
                    workspace.required - 1};
                FastCDWorkspaceF64 on_out = {out, workspace.required};
                FastCDWorkspaceF64 on_x = {x, workspace.required};
                FastCDWorkspaceF64 on_y = {y, workspace.required};

                CHECK(mul_route_batch(route, out, out_stride, x, x_stride, y,
                            y_stride, batch_count, dimension, NULL)
                        == FASTCD_NULL_POINTER
                        && array_is(out, count * out_stride, 93.0),
                        "%s rejects a missing workspace N=%zu", route->name,
                        dimension);
                CHECK(mul_route_batch(route, out, out_stride, x, x_stride, y,
                            y_stride, batch_count, dimension, &short_workspace)
                        == FASTCD_INSUFFICIENT_WORKSPACE
                        && array_is(out, count * out_stride, 93.0),
                        "%s rejects a short workspace N=%zu", route->name,
                        dimension);
                CHECK(mul_route_batch(route, out, out_stride, x, x_stride, y,
                            y_stride, batch_count, dimension, &on_out)
                        == FASTCD_OVERLAPPING_BUFFERS
                        && array_is(out, count * out_stride, 93.0),
                        "%s rejects a workspace overlapping the output N=%zu",
                        route->name, dimension);
                CHECK(mul_route_batch(route, out, out_stride, x, x_stride, y,
                            y_stride, batch_count, dimension, &on_x)
                        == FASTCD_OVERLAPPING_BUFFERS
                        && array_is(out, count * out_stride, 93.0),
                        "%s rejects a workspace overlapping x N=%zu",
                        route->name, dimension);
                CHECK(mul_route_batch(route, out, out_stride, x, x_stride, y,
                            y_stride, batch_count, dimension, &on_y)
                        == FASTCD_OVERLAPPING_BUFFERS
                        && array_is(out, count * out_stride, 93.0),
                        "%s rejects a workspace overlapping y N=%zu",
                        route->name, dimension);
            } else {
                fill_array(out, count * out_stride, 93.0);
                CHECK(mul_route_batch(route, out, out_stride, x, x_stride, y,
                            y_stride, batch_count, dimension, NULL)
                        == FASTCD_SUCCESS
                        && batch_matches(route, out, out_stride, x, x_stride,
                            y, y_stride, batch_count, dimension, NULL, scalar,
                            expected),
                        "%s accepts a NULL workspace when none is required N=%zu",
                        route->name, dimension);
            }

            guarded_workspace_free(&workspace);
            free(block);
        }
    }
}

// Check floating batches, odd tails, and dimensions beyond the SIMD storage bound
static void test_batch_backends(void) {
    const size_t dimensions[] = {16, 32, 64, 128, 1024, 2048, 4096, 8192};

    printf("\n=== batch backend boundaries ===\n");
    for (size_t d = 0; d < sizeof(dimensions) / sizeof(dimensions[0]); d++) {
        size_t dimension = dimensions[d];
        size_t stride = dimension + 1;
        double* storage = allocate_block(10, stride);
        double* x;
        double* y;
        double* out;
        double* scalar;

        CHECK(storage != NULL, "allocate backend fixtures N=%zu", dimension);
        if (storage == NULL) continue;
        x = storage;
        y = x + 3 * stride;
        out = y + 3 * stride;
        scalar = out + 3 * stride;
        fill_unit_interval(x, 3 * stride);
        fill_unit_interval(y, 3 * stride);
        for (size_t k = 0; k < 3; k++) {
            /* Distinct scales catch mixing the independent SIMD lanes. */
            for (size_t i = 0; i < dimension; i++) {
                x[k * stride + i] *= k == 0 ? 0x1p-20 : 0x1p20;
            }
            if (dimension > 128) {
                fill_array(y + k * stride, dimension, 0.0);
                y[k * stride] = k == 1 ? -2.0 : 0.5;
            }
        }
        for (size_t a = 0; a < EXPLICIT_ALGORITHM_COUNT; a++) {
            FastCDAlgorithm algorithm = explicit_algorithms[a];
            GuardedWorkspace workspace;

            if (!fastcd_algorithm_supports_dimension(algorithm, dimension)) {
                continue;
            }
            if (!workspace_for_algorithm(&workspace, dimension, algorithm)) {
                CHECK(false, "allocate backend workspace N=%zu", dimension);
                continue;
            }
            for (size_t count = 2; count <= 3; count++) {
                fill_array(out, 3 * stride, SENTINEL);
                CHECK(fastcd_mul_batch_with_algorithm_f64(out, stride, x,
                            stride, y, stride, count, dimension, algorithm,
                            &workspace.value) == FASTCD_SUCCESS,
                        "%s backend batch N=%zu count=%zu",
                        fastcd_algorithm_string(algorithm), dimension, count);
                CHECK(padding_is(out, stride, count, dimension, SENTINEL)
                        && array_is(out + count * stride, (3 - count) * stride,
                            SENTINEL) && guarded_workspace_intact(&workspace),
                        "backend preserves padding and guards N=%zu", dimension);
                for (size_t k = 0; k < count; k++) {
                    /* A product by a scalar is exact for direct and
                     * quasilinear, but the Cariow--Cariowa transforms round,
                     * so that algorithm is always checked against its own
                     * single-product result. */
                    if (dimension <= 128
                            || algorithm == FASTCD_ALGORITHM_CARIOW_CARIOWA) {
                        CHECK(fastcd_mul_with_algorithm_f64(scalar,
                                    x + k * stride, y + k * stride, dimension,
                                    algorithm, &workspace.value) == FASTCD_SUCCESS
                                && arrays_bitwise(out + k * stride, scalar,
                                    dimension),
                                "%s scalar/SIMD rounding N=%zu lane=%zu",
                                fastcd_algorithm_string(algorithm), dimension, k);
                    } else {
                        for (size_t i = 0; i < dimension; i++) {
                            scalar[i] = x[k * stride + i] * y[k * stride];
                        }
                        CHECK(arrays_exact(out + k * stride, scalar, dimension),
                                "%s large-dimension scalar factor N=%zu lane=%zu",
                                fastcd_algorithm_string(algorithm), dimension, k);
                    }
                }
            }
            guarded_workspace_free(&workspace);
        }
        free(storage);
    }
}

// Check complete products against the oracle above the SIMD pairing limit
static void test_large_dimension_products(void) {
    const size_t dimensions[] = {2048, 4096, 8192};
    const size_t count = 3;

    printf("\n=== large-dimension products against the oracle ===\n");
    for (size_t d = 0; d < sizeof(dimensions) / sizeof(dimensions[0]); d++) {
        size_t dimension = dimensions[d];
        double* block = allocate_block(3 * count + 2, dimension);
        double* x;
        double* y;
        double* out;
        double* expected;
        double* single;

        CHECK(block != NULL, "allocate large fixtures N=%zu", dimension);
        if (block == NULL) continue;
        x = block;
        y = x + count * dimension;
        out = y + count * dimension;
        expected = out + count * dimension;
        single = expected + dimension;
        fill_small_integers(x, count * dimension);
        fill_small_integers(y, count * dimension);
        for (size_t a = 0; a < EXPLICIT_ALGORITHM_COUNT; a++) {
            FastCDAlgorithm algorithm = explicit_algorithms[a];
            GuardedWorkspace workspace;

            if (!fastcd_algorithm_supports_dimension(algorithm, dimension)) {
                continue;
            }
            if (!workspace_for_algorithm(&workspace, dimension, algorithm)) {
                CHECK(false, "allocate large workspace N=%zu", dimension);
                continue;
            }
            fill_array(out, count * dimension, SENTINEL);
            CHECK(fastcd_mul_batch_with_algorithm_f64(out, dimension, x,
                        dimension, y, dimension, count, dimension, algorithm,
                        &workspace.value) == FASTCD_SUCCESS,
                    "%s large batch N=%zu", fastcd_algorithm_string(algorithm),
                    dimension);
            CHECK(fastcd_mul_with_algorithm_f64(single, x, y, dimension,
                        algorithm, &workspace.value) == FASTCD_SUCCESS
                    && arrays_bitwise(single, out, dimension),
                    "%s single and batch agree bitwise N=%zu",
                    fastcd_algorithm_string(algorithm), dimension);
            CHECK(guarded_workspace_intact(&workspace),
                    "%s preserves workspace guards N=%zu",
                    fastcd_algorithm_string(algorithm), dimension);
            for (size_t k = 0; k < count; k++) {
                CHECK(oracle_mul(expected, x + k * dimension, y + k * dimension,
                            dimension)
                        && arrays_exact(out + k * dimension, expected, dimension),
                        "%s exact large product N=%zu record=%zu",
                        fastcd_algorithm_string(algorithm), dimension, k);
            }
            guarded_workspace_free(&workspace);
        }
        free(block);
    }
}

// Check that NaN poisons every coordinate and infinity leaves none finite
static void test_nonfinite_propagation(void) {
    const size_t dimensions[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
    const double values[] = {NAN, INFINITY, -INFINITY};

    printf("\n=== NaN and infinity propagation ===\n");
    for (size_t d = 0; d < sizeof(dimensions) / sizeof(dimensions[0]); d++) {
        size_t dimension = dimensions[d];
        size_t positions[] = {0, dimension / 2, dimension - 1};
        double* block = allocate_block(3, dimension);
        double* x;
        double* y;
        double* out;

        CHECK(block != NULL, "allocate non-finite fixtures N=%zu", dimension);
        if (block == NULL) continue;
        x = block;
        y = x + dimension;
        out = y + dimension;
        for (size_t r = 0; r < MUL_ROUTE_COUNT; r++) {
            const MulRoute* route = &mul_routes[r];
            GuardedWorkspace workspace;

            if (!fastcd_algorithm_supports_dimension(route->algorithm,
                        dimension)) {
                continue;
            }
            if (!workspace_for_algorithm(&workspace, dimension,
                        route->algorithm)) {
                CHECK(false, "allocate non-finite workspace N=%zu", dimension);
                continue;
            }
            for (size_t p = 0; p < 3; p++) {
                for (size_t side = 0; side < 2; side++) {
                    for (size_t v = 0; v < 3; v++) {
                        double* poisoned = side == 0 ? x : y;
                        bool nan_case = v == 0;
                        bool propagated = true;

                        fill_unit_interval(x, dimension);
                        fill_unit_interval(y, dimension);
                        poisoned[positions[p]] = values[v];
                        fill_array(out, dimension, SENTINEL);
                        if (mul_route_call(route, out, x, y, dimension,
                                    &workspace.value) != FASTCD_SUCCESS) {
                            propagated = false;
                        }
                        for (size_t i = 0; propagated && i < dimension; i++) {
                            propagated = nan_case ? isnan(out[i]) != 0
                                : isfinite(out[i]) == 0;
                        }
                        CHECK(propagated,
                                "%s %s in %s[%zu] reaches every coordinate N=%zu",
                                route->name, nan_case ? "NaN" : "infinity",
                                side == 0 ? "x" : "y", positions[p], dimension);
                    }
                }
            }
            guarded_workspace_free(&workspace);
        }
        free(block);
    }
}

/* ---------- Noncommutative and nonassociative products ---------- */

// Compare cross, commutators, associator, and operator commutators with the oracle
static void test_composite_products(void) {
    const size_t dimensions[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
    const size_t trials = 8;

    printf("\n=== composite products ===\n");

    /* Adjacent vectors must use their own extents, not the scratch extent.
     * The workspace follows the output immediately and has exactly the
     * queried length, whatever the build's kernels require. */
    {
        size_t required = 0;
        double* storage = NULL;
        double expected[64];
        FastCDWorkspaceF64 workspace = {NULL, 0};

        CHECK(fastcd_imag_cross_workspace_size(64, &required)
                == FASTCD_SUCCESS, "adjacent cross fixture workspace query");
        storage = calloc(3 * 64 + required, sizeof(double));
        CHECK(storage != NULL, "allocate adjacent cross fixture");
        if (storage != NULL) {
            workspace.data = storage + 3 * 64;
            workspace.length = required;
            storage[1] = 2.0;
            storage[64 + 2] = 3.0;
            CHECK(reference_cd_mul_f64(expected, storage, storage + 64, 64)
                    == FASTCD_SUCCESS, "adjacent cross fixture oracle");
            CHECK(fastcd_imag_cross_f64(storage + 128, storage, storage + 64,
                        64, &workspace) == FASTCD_SUCCESS
                    && arrays_exact(storage + 128, expected, 64),
                    "cross accepts adjacent disjoint vectors and workspace");
            for (size_t i = 0; i < 64; i++) expected[i] *= 2.0;
            CHECK(fastcd_commutator_f64(storage + 128, storage, storage + 64,
                        64, &workspace) == FASTCD_SUCCESS
                    && arrays_exact(storage + 128, expected, 64),
                    "commutator accepts adjacent disjoint vectors and workspace");
            workspace.data--;
            CHECK(fastcd_imag_cross_f64(storage + 128, storage, storage + 64,
                        64, &workspace) == FASTCD_OVERLAPPING_BUFFERS,
                    "cross still rejects a one-coordinate workspace overlap");
            free(storage);
        }
    }

    for (size_t d = 0; d < sizeof(dimensions) / sizeof(dimensions[0]); d++) {
        size_t dimension = dimensions[d];
        double* block = allocate_block(11, dimension);
        double* x;
        double* y;
        double* z;
        double* imag_x;
        double* imag_y;
        double* actual;
        double* other;
        double* expected;
        double* t1;
        double* t2;
        double* t3;
        GuardedWorkspace cross_workspace = {0};
        GuardedWorkspace associator_workspace = {0};
        GuardedWorkspace lop_workspace = {0};
        GuardedWorkspace rop_workspace = {0};
        bool ready = block != NULL;

        CHECK(block != NULL, "allocate composite fixtures N=%zu", dimension);
        if (block == NULL) continue;
        x = block;
        y = block + dimension;
        z = block + 2 * dimension;
        imag_x = block + 3 * dimension;
        imag_y = block + 4 * dimension;
        actual = block + 5 * dimension;
        other = block + 6 * dimension;
        expected = block + 7 * dimension;
        t1 = block + 8 * dimension;
        t2 = block + 9 * dimension;
        t3 = block + 10 * dimension;
        if (!workspace_for_query(&cross_workspace, dimension,
                    fastcd_imag_cross_workspace_size)
                || !workspace_for_query(&associator_workspace, dimension,
                    fastcd_associator_workspace_size)
                || !workspace_for_query(&lop_workspace, dimension,
                    fastcd_lop_commutator_workspace_size)
                || !workspace_for_query(&rop_workspace, dimension,
                    fastcd_rop_commutator_workspace_size)) {
            ready = false;
        }
        CHECK(ready, "allocate composite workspaces N=%zu", dimension);

        for (size_t trial = 0; ready && trial < trials; trial++) {
            bool integer = trial < trials / 2;
            long double nx;
            long double ny;
            long double nz;
            long double pair_scale;
            long double triple_scale;

            if (integer) {
                fill_small_integers(x, dimension);
                fill_small_integers(y, dimension);
                fill_small_integers(z, dimension);
            } else {
                fill_unit_interval(x, dimension);
                fill_unit_interval(y, dimension);
                fill_unit_interval(z, dimension);
            }
            memcpy(imag_x, x, dimension * sizeof(double));
            memcpy(imag_y, y, dimension * sizeof(double));
            imag_x[0] = 0.0;
            imag_y[0] = 0.0;
            if (integer) {
                x[0] = 17.0;
                y[0] = -23.0;
            }
            nx = euclidean_norm(x, dimension);
            ny = euclidean_norm(y, dimension);
            nz = euclidean_norm(z, dimension);
            pair_scale = nx * ny;
            triple_scale = 2.0L * nx * ny * nz;

            /* imaginary cross product */
            CHECK(route_reference(ROUTE_CROSS, expected, x, y, NULL, dimension,
                        t1, t2, t3), "cross reference N=%zu trial=%zu",
                    dimension, trial);
            fill_array(actual, dimension, SENTINEL);
            CHECK(fastcd_imag_cross_f64(actual, x, y, dimension,
                        &cross_workspace.value) == FASTCD_SUCCESS
                    && (integer ? arrays_exact(actual, expected, dimension)
                        : within_tolerance(actual, expected, dimension,
                            pair_scale)),
                    "imaginary cross product N=%zu trial=%zu", dimension, trial);
            CHECK(actual[0] == 0.0 && copysign(1.0, actual[0]) == 1.0,
                    "cross scalar coordinate is +0.0 N=%zu trial=%zu",
                    dimension, trial);
            CHECK(fastcd_imag_cross_f64(other, imag_x, imag_y, dimension,
                        &cross_workspace.value) == FASTCD_SUCCESS
                    && arrays_bitwise(other, actual, dimension),
                    "cross ignores scalar coordinates N=%zu trial=%zu",
                    dimension, trial);
            CHECK(fastcd_imag_cross_f64(other, y, x, dimension,
                        &cross_workspace.value) == FASTCD_SUCCESS,
                    "cross of swapped inputs computes N=%zu trial=%zu",
                    dimension, trial);
            for (size_t i = 0; i < dimension; i++) other[i] = -other[i];
            CHECK(integer ? arrays_exact(other, actual, dimension)
                    : within_tolerance(other, actual, dimension, pair_scale),
                    "cross is antisymmetric N=%zu trial=%zu", dimension, trial);
            if (dimension <= 2) {
                CHECK(array_is(actual, dimension, 0.0),
                        "cross vanishes at N=%zu", dimension);
            }
            CHECK(guarded_workspace_intact(&cross_workspace),
                    "cross preserves workspace guards N=%zu", dimension);

            /* commutator */
            CHECK(route_reference(ROUTE_COMMUTATOR, expected, x, y, NULL,
                        dimension, t1, t2, t3),
                    "commutator reference N=%zu trial=%zu", dimension, trial);
            fill_array(actual, dimension, SENTINEL);
            CHECK(fastcd_commutator_f64(actual, x, y, dimension,
                        &cross_workspace.value) == FASTCD_SUCCESS
                    && (integer ? arrays_exact(actual, expected, dimension)
                        : within_tolerance(actual, expected, dimension,
                            2.0L * pair_scale)),
                    "commutator N=%zu trial=%zu", dimension, trial);
            CHECK(actual[0] == 0.0 && copysign(1.0, actual[0]) == 1.0,
                    "commutator scalar coordinate is +0.0 N=%zu trial=%zu",
                    dimension, trial);
            CHECK(fastcd_commutator_f64(other, imag_x, imag_y, dimension,
                        &cross_workspace.value) == FASTCD_SUCCESS
                    && arrays_bitwise(other, actual, dimension),
                    "commutator ignores scalar coordinates N=%zu trial=%zu",
                    dimension, trial);
            if (dimension <= 2) {
                CHECK(array_is(actual, dimension, 0.0),
                        "commutator vanishes at N=%zu", dimension);
            }
            CHECK(fastcd_imag_cross_f64(other, x, y, dimension,
                        &cross_workspace.value) == FASTCD_SUCCESS,
                    "cross for commutator comparison N=%zu trial=%zu",
                    dimension, trial);
            for (size_t i = 0; i < dimension; i++) other[i] += other[i];
            CHECK(arrays_bitwise(other, actual, dimension),
                    "commutator is exactly twice the cross N=%zu trial=%zu",
                    dimension, trial);

            /* anticommutator */
            CHECK(route_reference(ROUTE_ANTICOMMUTATOR, expected, x, y, NULL,
                        dimension, t1, t2, t3),
                    "anticommutator reference N=%zu trial=%zu", dimension,
                    trial);
            fill_array(actual, dimension, SENTINEL);
            CHECK(fastcd_anticommutator_f64(actual, x, y, dimension)
                    == FASTCD_SUCCESS
                    && (integer ? arrays_exact(actual, expected, dimension)
                        : within_tolerance(actual, expected, dimension,
                            2.0L * pair_scale)),
                    "anticommutator N=%zu trial=%zu", dimension, trial);
            CHECK(fastcd_anticommutator_f64(other, y, x, dimension)
                    == FASTCD_SUCCESS && arrays_bitwise(other, actual, dimension),
                    "anticommutator is symmetric N=%zu trial=%zu", dimension,
                    trial);

            /* associator */
            CHECK(route_reference(ROUTE_ASSOCIATOR, expected, x, y, z,
                        dimension, t1, t2, t3),
                    "associator reference N=%zu trial=%zu", dimension, trial);
            fill_array(actual, dimension, SENTINEL);
            CHECK(fastcd_associator_f64(actual, x, y, z, dimension,
                        &associator_workspace.value) == FASTCD_SUCCESS
                    && (integer ? arrays_exact(actual, expected, dimension)
                        : within_tolerance(actual, expected, dimension,
                            triple_scale)),
                    "associator N=%zu trial=%zu", dimension, trial);
            if (dimension <= 4 && integer) {
                CHECK(array_is(actual, dimension, 0.0),
                        "associator vanishes in an associative algebra N=%zu",
                        dimension);
            }
            CHECK(guarded_workspace_intact(&associator_workspace),
                    "associator preserves workspace guards N=%zu", dimension);

            /* left operator commutator */
            CHECK(route_reference(ROUTE_LOP, expected, x, y, z, dimension, t1,
                        t2, t3), "left-operator reference N=%zu trial=%zu",
                    dimension, trial);
            fill_array(actual, dimension, SENTINEL);
            CHECK(fastcd_lop_commutator_apply_f64(actual, x, y, z, dimension,
                        &lop_workspace.value) == FASTCD_SUCCESS
                    && (integer ? arrays_exact(actual, expected, dimension)
                        : within_tolerance(actual, expected, dimension,
                            triple_scale)),
                    "left-operator commutator N=%zu trial=%zu", dimension,
                    trial);
            CHECK(guarded_workspace_intact(&lop_workspace),
                    "left operator preserves workspace guards N=%zu",
                    dimension);

            /* right operator commutator */
            CHECK(route_reference(ROUTE_ROP, expected, x, y, z, dimension, t1,
                        t2, t3), "right-operator reference N=%zu trial=%zu",
                    dimension, trial);
            fill_array(actual, dimension, SENTINEL);
            CHECK(fastcd_rop_commutator_apply_f64(actual, x, y, z, dimension,
                        &rop_workspace.value) == FASTCD_SUCCESS
                    && (integer ? arrays_exact(actual, expected, dimension)
                        : within_tolerance(actual, expected, dimension,
                            triple_scale)),
                    "right-operator commutator N=%zu trial=%zu", dimension,
                    trial);
            CHECK(guarded_workspace_intact(&rop_workspace),
                    "right operator preserves workspace guards N=%zu",
                    dimension);
        }

        guarded_workspace_free(&cross_workspace);
        guarded_workspace_free(&associator_workspace);
        guarded_workspace_free(&lop_workspace);
        guarded_workspace_free(&rop_workspace);
        free(block);
    }

    {
        double x[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        double y[8] = {8, 7, 6, 5, 4, 3, 2, 1};
        double out[8];

        fill_array(out, 8, SENTINEL);
        CHECK(fastcd_anticommutator_f64(out, NULL, y, 8) == FASTCD_NULL_POINTER
                && array_is(out, 8, SENTINEL),
                "anticommutator rejects a null first input");
        CHECK(fastcd_anticommutator_f64(out, x, NULL, 8) == FASTCD_NULL_POINTER
                && array_is(out, 8, SENTINEL),
                "anticommutator rejects a null second input");
        CHECK(fastcd_anticommutator_f64(NULL, x, y, 8) == FASTCD_NULL_POINTER,
                "anticommutator rejects a null output");
        CHECK(fastcd_anticommutator_f64(out, x, y, 3) == FASTCD_INVALID_DIMENSION
                && array_is(out, 8, SENTINEL),
                "anticommutator rejects an invalid dimension");
        CHECK(fastcd_anticommutator_f64(out, x, y, 1) == FASTCD_SUCCESS
                && out[0] == 16.0 && array_is(out + 1, 7, SENTINEL),
                "anticommutator at dimension one is 2xy");
    }
}

/* ---------- Size overflow boundaries ---------- */

// Check size arithmetic at every representable power-of-two dimension
static void test_size_overflow_boundaries(void) {
    const struct {
        const char* name;
        WorkspaceQuery query;
        bool cariow;
    } queries[] = {
        {"mul (public)", fastcd_mul_workspace_size, false},
        {"mul (explicit auto)", query_auto_explicit, false},
        {"mul direct", query_direct, false},
        {"mul cariow_cariowa", query_cariow_cariowa, true},
        {"mul quasilinear", query_quasilinear, false},
        {"imag_cross", fastcd_imag_cross_workspace_size, false},
        {"associator", fastcd_associator_workspace_size, false},
        {"lop_commutator", fastcd_lop_commutator_workspace_size, false},
        {"rop_commutator", fastcd_rop_commutator_workspace_size, false}
    };
    const size_t bit_count = sizeof(size_t) * CHAR_BIT;
    size_t huge_dimension = 1;
    size_t huge_dimensions[2];

    printf("\n=== size overflow boundaries ===\n");

    while (huge_dimension <= SIZE_MAX / sizeof(double)) huge_dimension <<= 1;
    huge_dimensions[0] = huge_dimension;
    huge_dimensions[1] = (size_t)1 << (bit_count - 1);

    for (size_t q = 0; q < sizeof(queries) / sizeof(queries[0]); q++) {
        bool failed_before = false;
        bool overflow_seen = false;

        for (size_t level = 0; level < bit_count; level++) {
            size_t dimension = (size_t)1 << level;
            size_t length = 777;
            FastCDStatus status = queries[q].query(dimension, &length);

            if (status == FASTCD_SUCCESS) {
                CHECK(!failed_before && dimension <= SIZE_MAX / sizeof(double)
                        && length <= SIZE_MAX / sizeof(double),
                        "%s query fits addressable storage at level %zu",
                        queries[q].name, level);
            } else {
                bool acceptable = status == FASTCD_SIZE_OVERFLOW
                    || (queries[q].cariow
                        && dimension > FASTCD_CARIOW_CARIOWA_MAX_DIMENSION
                        && status == FASTCD_UNSUPPORTED_ALGORITHM);

                CHECK(acceptable && length == 777,
                        "%s query fails atomically at level %zu (status %d)",
                        queries[q].name, level, (int)status);
                failed_before = true;
                if (status == FASTCD_SIZE_OVERFLOW) overflow_seen = true;
            }
        }
        CHECK(overflow_seen, "%s query has a size-overflow boundary",
                queries[q].name);
    }

    for (size_t h = 0; h < 2; h++) {
        size_t huge = huge_dimensions[h];
        double x[2] = {1.0, 1.0};
        double y[2] = {1.0, 1.0};
        double z[2] = {1.0, 1.0};
        double out[2] = {SENTINEL, SENTINEL};
        double scratch[16];
        double value = SENTINEL;
        FastCDWorkspaceF64 workspace = {scratch, 16};
        FastCDBasisProduct product = {77, 3};
        FastCDAlgorithm algorithm = FASTCD_ALGORITHM_AUTO;
        size_t level = 0;
        FastCDStatus status;
#define OVERFLOWS(call) \
        ((status = (call)) == FASTCD_SIZE_OVERFLOW \
         || status == FASTCD_INVALID_DIMENSION)

        CHECK(OVERFLOWS(fastcd_scalar_f64(out, huge, 1.0)) && out[0] == SENTINEL,
                "scalar constructor rejects an unaddressable dimension (%zu)", h);
        CHECK(OVERFLOWS(fastcd_basis_f64(out, huge, 0)) && out[0] == SENTINEL,
                "basis constructor rejects an unaddressable dimension (%zu)", h);
        CHECK(OVERFLOWS(fastcd_add_f64(out, x, y, huge)) && out[0] == SENTINEL,
                "addition rejects an unaddressable dimension (%zu)", h);
        CHECK(OVERFLOWS(fastcd_sub_f64(out, x, y, huge)) && out[0] == SENTINEL,
                "subtraction rejects an unaddressable dimension (%zu)", h);
        CHECK(OVERFLOWS(fastcd_neg_f64(out, x, huge)) && out[0] == SENTINEL,
                "negation rejects an unaddressable dimension (%zu)", h);
        CHECK(OVERFLOWS(fastcd_scale_f64(out, x, 2.0, huge))
                && out[0] == SENTINEL,
                "scaling rejects an unaddressable dimension (%zu)", h);
        CHECK(OVERFLOWS(fastcd_conjugate_f64(out, x, huge)) && out[0] == SENTINEL,
                "conjugation rejects an unaddressable dimension (%zu)", h);
        CHECK(OVERFLOWS(fastcd_imag_part_f64(out, x, huge)) && out[0] == SENTINEL,
                "imaginary projection rejects an unaddressable dimension (%zu)",
                h);
        CHECK(OVERFLOWS(fastcd_inner_f64(&value, x, y, huge))
                && value == SENTINEL,
                "inner product rejects an unaddressable dimension (%zu)", h);
        CHECK(OVERFLOWS(fastcd_norm_squared_f64(&value, x, huge))
                && value == SENTINEL,
                "squared norm rejects an unaddressable dimension (%zu)", h);
        CHECK(OVERFLOWS(fastcd_norm_f64(&value, x, huge)) && value == SENTINEL,
                "norm rejects an unaddressable dimension (%zu)", h);
        CHECK(OVERFLOWS(fastcd_normalize_f64(out, x, huge)) && out[0] == SENTINEL,
                "normalization rejects an unaddressable dimension (%zu)", h);
        CHECK(OVERFLOWS(fastcd_reciprocal_f64(out, x, huge))
                && out[0] == SENTINEL,
                "reciprocal rejects an unaddressable dimension (%zu)", h);
        CHECK(OVERFLOWS(fastcd_square_f64(out, x, huge)) && out[0] == SENTINEL,
                "square rejects an unaddressable dimension (%zu)", h);
        CHECK(OVERFLOWS(fastcd_mul_f64(out, x, y, huge, &workspace))
                && out[0] == SENTINEL,
                "automatic product rejects an unaddressable dimension (%zu)", h);
        for (size_t a = 0; a < EXPLICIT_ALGORITHM_COUNT; a++) {
            status = fastcd_mul_with_algorithm_f64(out, x, y, huge,
                    explicit_algorithms[a], &workspace);
            CHECK((status == FASTCD_SIZE_OVERFLOW
                        || status == FASTCD_INVALID_DIMENSION
                        || (explicit_algorithms[a]
                            == FASTCD_ALGORITHM_CARIOW_CARIOWA
                            && status == FASTCD_UNSUPPORTED_ALGORITHM))
                    && out[0] == SENTINEL,
                    "%s product rejects an unaddressable dimension (%zu)",
                    fastcd_algorithm_string(explicit_algorithms[a]), h);
        }
        CHECK(OVERFLOWS(fastcd_mul_batch_f64(out, huge, x, huge, y, huge, 1,
                        huge, &workspace)) && out[0] == SENTINEL,
                "singleton batch rejects an unaddressable dimension (%zu)", h);
        CHECK(OVERFLOWS(fastcd_mul_batch_with_algorithm_f64(out, huge, x, huge,
                        y, huge, 2, huge, FASTCD_ALGORITHM_QUASILINEAR,
                        &workspace)) && out[0] == SENTINEL,
                "quasilinear batch rejects an unaddressable dimension (%zu)", h);
        CHECK(OVERFLOWS(fastcd_imag_cross_f64(out, x, y, huge, &workspace))
                && out[0] == SENTINEL,
                "cross rejects an unaddressable dimension (%zu)", h);
        CHECK(OVERFLOWS(fastcd_commutator_f64(out, x, y, huge, &workspace))
                && out[0] == SENTINEL,
                "commutator rejects an unaddressable dimension (%zu)", h);
        CHECK(OVERFLOWS(fastcd_anticommutator_f64(out, x, y, huge))
                && out[0] == SENTINEL,
                "anticommutator rejects an unaddressable dimension (%zu)", h);
        CHECK(OVERFLOWS(fastcd_associator_f64(out, x, y, z, huge, &workspace))
                && out[0] == SENTINEL,
                "associator rejects an unaddressable dimension (%zu)", h);
        CHECK(OVERFLOWS(fastcd_lop_commutator_apply_f64(out, x, y, z, huge,
                        &workspace)) && out[0] == SENTINEL,
                "left operator rejects an unaddressable dimension (%zu)", h);
        CHECK(OVERFLOWS(fastcd_rop_commutator_apply_f64(out, x, y, z, huge,
                        &workspace)) && out[0] == SENTINEL,
                "right operator rejects an unaddressable dimension (%zu)", h);
        CHECK(OVERFLOWS(fastcd_basis_lmul_f64(out, 0, x, huge))
                && out[0] == SENTINEL,
                "basis left action rejects an unaddressable dimension (%zu)", h);
        CHECK(OVERFLOWS(fastcd_basis_rmul_f64(out, x, 0, huge))
                && out[0] == SENTINEL,
                "basis right action rejects an unaddressable dimension (%zu)", h);
#undef OVERFLOWS
        status = fastcd_basis_product(&product, huge, 1, 2);
        CHECK(status != FASTCD_SUCCESS
                || (product.index == 3
                    && (product.sign == 1 || product.sign == -1)),
                "indexed basis product stays consistent at a huge dimension (%zu)",
                h);
        CHECK(fastcd_level(huge, &level) == FASTCD_SUCCESS
                && ((size_t)1 << level) == huge,
                "level recovers a huge power of two (%zu)", h);
        status = fastcd_default_algorithm(huge, &algorithm);
        CHECK(status != FASTCD_SUCCESS || algorithm == FASTCD_ALGORITHM_DIRECT
                || algorithm == FASTCD_ALGORITHM_CARIOW_CARIOWA
                || algorithm == FASTCD_ALGORITHM_QUASILINEAR,
                "default algorithm stays concrete at a huge dimension (%zu)", h);
    }
}

/* ---------- Oracle sanity and hand-computed products ---------- */

// Check the oracle and every algorithm against products computed by hand
static void test_reference_oracle(void) {
    double quaternion_x[4] = {1, 2, 3, 4};
    double quaternion_y[4] = {4, 3, 2, 1};
    const double quaternion_xy[4] = {-12, 6, 24, 12};
    const double quaternion_yx[4] = {-12, 16, 4, 22};
    double complex_x[2] = {1, 2};
    double complex_y[2] = {3, 4};
    const double complex_xy[2] = {-5, 10};
    double scalar_x = -3.0;
    double scalar_y = 2.0;
    double out[4];
    int sign = 19;

    printf("\n=== oracle sanity and hand-computed products ===\n");

    CHECK(reference_cd_basis_sign(4, 1, 2, &sign) == FASTCD_SUCCESS
            && sign == 1, "oracle sign: e1 e2 = e3");
    CHECK(reference_cd_basis_sign(4, 2, 1, &sign) == FASTCD_SUCCESS
            && sign == -1, "oracle sign: e2 e1 = -e3");
    CHECK(reference_cd_basis_sign(4, 2, 3, &sign) == FASTCD_SUCCESS
            && sign == 1, "oracle sign: e2 e3 = e1");
    CHECK(reference_cd_basis_sign(4, 3, 1, &sign) == FASTCD_SUCCESS
            && sign == 1, "oracle sign: e3 e1 = e2");
    CHECK(reference_cd_basis_sign(4, 1, 1, &sign) == FASTCD_SUCCESS
            && sign == -1, "oracle sign: e1 e1 = -1");
    CHECK(reference_cd_basis_sign(4, 0, 3, &sign) == FASTCD_SUCCESS
            && sign == 1, "oracle sign: 1 e3 = e3");
    CHECK(reference_cd_basis_sign(1, 0, 0, &sign) == FASTCD_SUCCESS
            && sign == 1, "oracle sign: real unit squares to one");
    sign = 19;
    CHECK(reference_cd_basis_sign(4, 4, 0, &sign) == FASTCD_INVALID_BASIS_INDEX
            && sign == 19, "oracle sign preserves output for invalid index");
    CHECK(reference_cd_basis_sign(3, 0, 0, &sign) == FASTCD_INVALID_DIMENSION,
            "oracle sign rejects invalid dimension");
    CHECK(reference_cd_basis_sign(4, 0, 0, NULL) == FASTCD_NULL_POINTER,
            "oracle sign rejects null output");
    CHECK(reference_cd_mul_f64(quaternion_x, quaternion_x, quaternion_y, 4)
            == FASTCD_OVERLAPPING_BUFFERS, "oracle rejects out == x");
    CHECK(reference_cd_mul_f64(out, NULL, quaternion_y, 4)
            == FASTCD_NULL_POINTER, "oracle rejects null input");
    CHECK(reference_cd_mul_f64(out, quaternion_x, quaternion_y, 3)
            == FASTCD_INVALID_DIMENSION, "oracle rejects invalid dimension");

    CHECK(reference_cd_mul_f64(out, quaternion_x, quaternion_y, 4)
            == FASTCD_SUCCESS && arrays_exact(out, quaternion_xy, 4),
            "oracle quaternion product (1+2i+3j+4k)(4+3i+2j+k)");
    CHECK(reference_cd_mul_f64(out, quaternion_y, quaternion_x, 4)
            == FASTCD_SUCCESS && arrays_exact(out, quaternion_yx, 4),
            "oracle quaternion product (4+3i+2j+k)(1+2i+3j+4k)");
    CHECK(reference_cd_mul_f64(out, complex_x, complex_y, 2) == FASTCD_SUCCESS
            && arrays_exact(out, complex_xy, 2),
            "oracle complex product (1+2i)(3+4i)");
    CHECK(reference_cd_mul_f64(out, &scalar_x, &scalar_y, 1) == FASTCD_SUCCESS
            && out[0] == -6.0, "oracle real product");
    CHECK(reference_cd_mul_signed_f64(out, quaternion_x, quaternion_y, 4)
            == FASTCD_SUCCESS && arrays_exact(out, quaternion_xy, 4),
            "signed-sum oracle quaternion product");
    CHECK(reference_cd_mul_signed_f64(quaternion_x, quaternion_x, quaternion_y,
                4) == FASTCD_OVERLAPPING_BUFFERS,
            "signed-sum oracle rejects out == x");
    for (size_t level = 0; level <= 8; level++) {
        size_t dimension = (size_t)1 << level;
        double* block = allocate_block(4, dimension);

        CHECK(block != NULL, "allocate oracle cross-check fixtures N=%zu",
                dimension);
        if (block == NULL) continue;
        for (size_t trial = 0; trial < 4; trial++) {
            fill_small_integers(block, 2 * dimension);
            CHECK(reference_cd_mul_f64(block + 2 * dimension, block,
                        block + dimension, dimension) == FASTCD_SUCCESS
                    && reference_cd_mul_signed_f64(block + 3 * dimension, block,
                        block + dimension, dimension) == FASTCD_SUCCESS
                    && arrays_exact(block + 2 * dimension, block + 3 * dimension,
                        dimension),
                    "recursive and signed-sum oracles agree N=%zu trial=%zu",
                    dimension, trial);
        }
        free(block);
    }

    for (size_t r = 0; r < MUL_ROUTE_COUNT; r++) {
        GuardedWorkspace workspace;

        if (!workspace_for_algorithm(&workspace, 4, mul_routes[r].algorithm)) {
            CHECK(false, "allocate %s hand-computed workspace",
                    mul_routes[r].name);
            continue;
        }
        CHECK(mul_route_call(&mul_routes[r], out, quaternion_x, quaternion_y,
                    4, &workspace.value) == FASTCD_SUCCESS
                && arrays_exact(out, quaternion_xy, 4),
                "%s quaternion product by hand", mul_routes[r].name);
        CHECK(mul_route_call(&mul_routes[r], out, quaternion_y, quaternion_x,
                    4, &workspace.value) == FASTCD_SUCCESS
                && arrays_exact(out, quaternion_yx, 4),
                "%s reversed quaternion product by hand", mul_routes[r].name);
        CHECK(mul_route_call(&mul_routes[r], out, complex_x, complex_y, 2,
                    &workspace.value) == FASTCD_SUCCESS
                && arrays_exact(out, complex_xy, 2),
                "%s complex product by hand", mul_routes[r].name);
        CHECK(mul_route_call(&mul_routes[r], out, &scalar_x, &scalar_y, 1,
                    &workspace.value) == FASTCD_SUCCESS && out[0] == -6.0,
                "%s real product by hand", mul_routes[r].name);
        guarded_workspace_free(&workspace);
    }
}

// Run the complete fastCD test suite
int main(void) {
    clock_t start = clock();
    double elapsed;

    printf("fastCD automated test suite (library %s)\n",
            fastcd_version_string());

    test_library_info();
    test_dimension_utilities();
    test_algorithm_selection();
    test_workspace_contracts();
    test_coordinate_operations();
    test_overlap_validation();
    test_structural_operations();
    test_numeric_edge_cases();
    test_signed_zero();
    test_square();
    test_reference_oracle();
    test_exhaustive_basis_products();
    test_basis_validation();
    test_random_integer_products();
    test_random_floating_products();
    test_numerical_fixtures();
    test_batch_multiplication();
    test_batch_backends();
    test_large_dimension_products();
    test_nonfinite_propagation();
    test_composite_products();
    test_size_overflow_boundaries();

    elapsed = (double)(clock() - start) / (double)CLOCKS_PER_SEC;
    printf("\n%zu checks, %zu passed, %zu failed in %.2f s\n", tests_run,
            tests_run - tests_failed, tests_failed, elapsed);
    if (tests_failed > FAILURE_REPORT_LIMIT) {
        fprintf(stderr, "  (%zu failures were not printed)\n",
                tests_failed - FAILURE_REPORT_LIMIT);
    }
    return tests_failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
