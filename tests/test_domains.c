/* Independent coefficient and basis oracle, and exact-workspace boundary
 * tests. The oracle uses schoolbook polynomial arithmetic and the defining CD
 * sign recursion, independent of both production multiplication algorithms. */
#include "fastcd/complex.h"
#include "fastcd/finite.h"
#include <math.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x)                                                                       \
    do {                                                                               \
        if (!(x)) {                                                                    \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x);                    \
            return 0;                                                                  \
        }                                                                              \
    } while (0)

// Obtain basis signs from the defining recursion, without production tables
static int basis_sign(size_t n, size_t i, size_t j) {
    if (n == 1) return 1;
    size_t h = n / 2;
    if (i < h && j < h) return basis_sign(h, i, j);
    if (i < h) return basis_sign(h, j - h, i);
    if (j < h) return (j ? -1 : 1) * basis_sign(h, i - h, j);
    return (j == h ? -1 : 1) * basis_sign(h, j - h, i - h);
}

// Generate repeatable canonical residues without modifying library state
static uint32_t next_word(uint64_t* state, uint32_t p) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (uint32_t)((*state * UINT64_C(2685821657736338717)) % p);
}

// Independently convolve polynomial coefficients and reduce by the actual modulus
static void scalar_reference(
        uint32_t* out, const uint32_t* x, const uint32_t* y, const FastCDField* f) {
    uint64_t t[23] = {0};
    size_t d = f->degree;
    for (size_t i = 0; i < d; i++)
        for (size_t j = 0; j < d; j++)
            t[i + j] = (t[i + j] + (uint64_t)x[i] * y[j]) % f->prime;
    for (size_t k = 2 * d - 1; k-- > d;)
        for (size_t j = 0; j < d; j++)
            t[k - d + j] = (t[k - d + j] + f->prime - t[k] * f->modulus[j] % f->prime) %
                    f->prime;
    for (size_t i = 0; i < d; i++) out[i] = (uint32_t)t[i];
}

// Accumulate signed basis products using only the independent scalar oracle
static void finite_reference(
        uint32_t* out, const uint32_t* x, const uint32_t* y, const FastCDPlanFF* plan) {
    size_t n = plan->dimension, d = plan->field.degree;
    uint32_t t[12], p = plan->field.prime;
    memset(out, 0, plan->words * sizeof(*out));
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            scalar_reference(t, x + i * d, y + j * d, &plan->field);
            int s = basis_sign(n, i, j);
            for (size_t k = 0; k < d; k++) {
                uint64_t old = out[(i ^ j) * d + k];
                out[(i ^ j) * d + k] =
                        (uint32_t)((old + (s > 0 ? t[k] : p - t[k])) % p);
            }
        }
    }
}

// Verify both schedules, exact scratch bounds, batches and parenthesized composites
static int finite_case(uint32_t prime, uint32_t degree, size_t level) {
    FastCDField f;
    FastCDPlanFF p;
    CHECK(fastcd_field_init(&f, prime, degree, NULL) == FASTCD_FF_SUCCESS);
    CHECK(fastcd_ff_plan_init(&p, &f, level) == FASTCD_FF_SUCCESS);
    size_t w = p.words, len;
    CHECK(fastcd_apply_workspace_size_ff(&p, FASTCD_OP_ASSOCIATOR, &len) ==
            FASTCD_FF_SUCCESS);
    uint32_t* data = malloc((7 * w + len + 2) * sizeof(*data));
    CHECK(data != NULL);
    uint32_t *x = data, *y = x + w, *expected = y + w, *out = expected + w;
    uint32_t *other = out + w, *temp = other + w, *third = temp + w,
             *work = third + w + 1;
    uint64_t state = UINT64_C(0x123456789abcdef) + prime + degree + level;
    for (size_t i = 0; i < w; i++) {
        x[i] = next_word(&state, prime);
        y[i] = next_word(&state, prime);
        third[i] = next_word(&state, prime);
    }
    finite_reference(expected, x, y, &p);
    // Check every finite-field algorithm, including the AUTO choice
    static const FastCDAlgorithm algorithms[] = {FASTCD_ALGORITHM_DIRECT,
            FASTCD_ALGORITHM_QUASILINEAR, FASTCD_ALGORITHM_AUTO};
    for (size_t k = 0; k < sizeof algorithms / sizeof *algorithms; k++) {
        FastCDAlgorithm alg = algorithms[k];
        work[-1] = UINT32_C(0xdeadbeef);
        work[p.workspace_words] = UINT32_C(0xbaadf00d);
        CHECK(fastcd_mul_with_algorithm_ff(out, x, y, &p, alg, work,
                      p.workspace_words) == FASTCD_FF_SUCCESS);
        CHECK(memcmp(out, expected, w * sizeof(*out)) == 0);
        CHECK(work[-1] == UINT32_C(0xdeadbeef));
        CHECK(work[p.workspace_words] == UINT32_C(0xbaadf00d));
        memcpy(other, out, w * sizeof(*out));
        CHECK(fastcd_mul_with_algorithm_ff(out, x, y, &p, alg, work,
                      p.workspace_words - 1) == FASTCD_FF_INSUFFICIENT_WORKSPACE);
        CHECK(memcmp(out, other, w * sizeof(*out)) == 0);
        CHECK(fastcd_mul_with_algorithm_ff(x, x, y, &p, alg, work, p.workspace_words) ==
                FASTCD_FF_OVERLAPPING_BUFFERS);
        CHECK(fastcd_mul_with_algorithm_ff(out, x, y, &p, alg, work, SIZE_MAX) ==
                FASTCD_FF_SUCCESS);
    }
    finite_reference(temp, y, y, &p);
    // Repeat the algorithms on a two-record batch with a broadcast operand
    for (size_t k = 0; k < sizeof algorithms / sizeof *algorithms; k++) {
        FastCDAlgorithm alg = algorithms[k];
        work[-1] = UINT32_C(0xdeadbeef);
        work[p.workspace_words] = UINT32_C(0xbaadf00d);
        CHECK(fastcd_mul_batch_ff(out, w, x, w, y, 0, 2, &p, alg, work,
                      p.workspace_words) == FASTCD_FF_SUCCESS);
        CHECK(memcmp(out, expected, w * sizeof(*out)) == 0);
        CHECK(memcmp(other, temp, w * sizeof(*out)) == 0);
        CHECK(work[-1] == UINT32_C(0xdeadbeef));
        CHECK(work[p.workspace_words] == UINT32_C(0xbaadf00d));
    }
    CHECK(fastcd_mul_batch_ff(out, w - 1, x, w, y, 0, 2, &p,
                  FASTCD_ALGORITHM_QUASILINEAR, work, len) == FASTCD_FF_INVALID_STRIDE);
    CHECK(fastcd_mul_batch_ff(out, w, x, w, y, 0, SIZE_MAX, &p,
                  FASTCD_ALGORITHM_QUASILINEAR, work, len) == FASTCD_FF_SIZE_OVERFLOW);
    CHECK(fastcd_apply_ff(temp, x, y, NULL, &p, FASTCD_OP_MULTIPLY, 0,
                  FASTCD_ALGORITHM_DIRECT, work, len) == FASTCD_FF_SUCCESS);
    CHECK(fastcd_mul_with_algorithm_ff(other, temp, third, &p, FASTCD_ALGORITHM_DIRECT,
                  work, len) == FASTCD_FF_SUCCESS);
    CHECK(fastcd_mul_with_algorithm_ff(temp, y, third, &p, FASTCD_ALGORITHM_DIRECT,
                  work, len) == FASTCD_FF_SUCCESS);
    CHECK(fastcd_mul_with_algorithm_ff(out, x, temp, &p, FASTCD_ALGORITHM_DIRECT, work,
                  len) == FASTCD_FF_SUCCESS);
    for (size_t i = 0; i < w; i++)
        expected[i] = (uint32_t)(((uint64_t)other[i] + prime - out[i]) % prime);
    work[-1] = UINT32_C(0xdeadbeef);
    work[len] = UINT32_C(0xbaadf00d);
    CHECK(fastcd_apply_ff(out, x, y, third, &p, FASTCD_OP_ASSOCIATOR, 0,
                  FASTCD_ALGORITHM_QUASILINEAR, work, len) == FASTCD_FF_SUCCESS);
    CHECK(memcmp(out, expected, w * sizeof(*out)) == 0);
    CHECK(work[-1] == UINT32_C(0xdeadbeef) && work[len] == UINT32_C(0xbaadf00d));
    CHECK(fastcd_mul_batch_ff(out, w, x, w, y, w, 0, &p, FASTCD_ALGORITHM_AUTO, work,
                  len) == FASTCD_FF_SUCCESS);
    free(data);
    return 1;
}

// Use dense integer-valued complex inputs with an exact signed-basis reference
static int complex_case(size_t n) {
    size_t len;
    CHECK(fastcd_workspace_size_c128(n, FASTCD_OP_ASSOCIATOR, &len) == FASTCD_SUCCESS);
    FastCDComplexF64* data = malloc(7 * n * sizeof(*data));
    double* work = malloc((len + 2) * sizeof(*work));
    CHECK(data && work);
    FastCDComplexF64 *x = data, *y = x + n, *expected = y + n, *out = expected + n;
    FastCDComplexF64 *temp = out + n, *other = temp + n, *z = other + n;
    uint64_t state = n + 17;
    for (size_t i = 0; i < n; i++) {
        x[i].real = (double)next_word(&state, 31) - 15;
        x[i].imag = (double)next_word(&state, 31) - 15;
        y[i].real = (double)next_word(&state, 31) - 15;
        y[i].imag = (double)next_word(&state, 31) - 15;
        z[i].real = (double)next_word(&state, 31) - 15;
        z[i].imag = (double)next_word(&state, 31) - 15;
        expected[i].real = expected[i].imag = 0;
    }
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++) {
            double sign = (double)basis_sign(n, i, j);
            expected[i ^ j].real +=
                    sign * (x[i].real * y[j].real - x[i].imag * y[j].imag);
            expected[i ^ j].imag +=
                    sign * (x[i].real * y[j].imag + x[i].imag * y[j].real);
        }
    for (int k = 0; k < 2; k++) {
        FastCDAlgorithm alg =
                k ? FASTCD_ALGORITHM_QUASILINEAR : FASTCD_ALGORITHM_DIRECT;
        size_t product_len;
        CHECK(fastcd_workspace_size_c128(n, FASTCD_OP_MULTIPLY, &product_len) ==
                FASTCD_SUCCESS);
        FastCDWorkspaceF64 w = {work + 1, product_len};
        work[0] = -12345;
        work[1 + product_len] = 98765;
        CHECK(fastcd_mul_with_algorithm_c128(out, x, y, n, alg, &w) == FASTCD_SUCCESS);
        CHECK(memcmp(out, expected, n * sizeof(*out)) == 0);
        CHECK(work[0] == -12345 && work[1 + product_len] == 98765);
        w.length--;
        CHECK(fastcd_mul_with_algorithm_c128(out, x, y, n, alg, &w) ==
                FASTCD_INSUFFICIENT_WORKSPACE);
    }
    FastCDWorkspaceF64 w = {work + 1, len};
    CHECK(fastcd_mul_with_algorithm_c128(temp, x, y, n, FASTCD_ALGORITHM_DIRECT, &w) ==
            FASTCD_SUCCESS);
    CHECK(fastcd_mul_with_algorithm_c128(
                  other, temp, z, n, FASTCD_ALGORITHM_DIRECT, &w) == FASTCD_SUCCESS);
    CHECK(fastcd_mul_with_algorithm_c128(temp, y, z, n, FASTCD_ALGORITHM_DIRECT, &w) ==
            FASTCD_SUCCESS);
    CHECK(fastcd_mul_with_algorithm_c128(
                  out, x, temp, n, FASTCD_ALGORITHM_DIRECT, &w) == FASTCD_SUCCESS);
    for (size_t i = 0; i < n; i++) {
        expected[i].real = other[i].real - out[i].real;
        expected[i].imag = other[i].imag - out[i].imag;
    }
    work[0] = -12345;
    work[len + 1] = 98765;
    CHECK(fastcd_apply_c128(out, x, y, z, n, FASTCD_OP_ASSOCIATOR, 0,
                  FASTCD_ALGORITHM_QUASILINEAR, &w) == FASTCD_SUCCESS);
    CHECK(memcmp(out, expected, n * sizeof(*out)) == 0);
    CHECK(work[0] == -12345 && work[len + 1] == 98765);
    free(data);
    free(work);
    return 1;
}

// Check that an explicit algorithm also selects cross/commutator arithmetic
static int complex_cross_algorithm(void) {
    const size_t n = 32;
    FastCDComplexF64 x[32] = {{0}}, y[32] = {{0}};
    FastCDComplexF64 products[2][32], out[32];
    size_t length;
    CHECK(fastcd_workspace_size_c128(n, FASTCD_OP_MULTIPLY, &length) == FASTCD_SUCCESS);
    double* scratch = malloc(length * sizeof(*scratch));
    CHECK(scratch != NULL);
    FastCDWorkspaceF64 workspace = {scratch, length};
    for (size_t i = 1; i < n; i++) {
        double k = (double)i;
        x[i].real = sin(k * 0.3) * (k + 1);
        x[i].imag = cos(k * 0.7) / (k + 1);
        y[i].real = cos(k * 0.5) / (k + 2);
        y[i].imag = sin(k * 1.1) * (k + 2);
    }
    for (int k = 0; k < 2; k++) {
        FastCDAlgorithm algorithm =
                k ? FASTCD_ALGORITHM_QUASILINEAR : FASTCD_ALGORITHM_DIRECT;
        CHECK(fastcd_mul_with_algorithm_c128(
                      products[k], x, y, n, algorithm, &workspace) == FASTCD_SUCCESS);
        for (int commutator = 0; commutator < 2; commutator++) {
            FastCDOperation op =
                    commutator ? FASTCD_OP_COMMUTATOR : FASTCD_OP_IMAG_CROSS;
            CHECK(fastcd_apply_c128(out, x, y, NULL, n, op, 0, algorithm, &workspace) ==
                    FASTCD_SUCCESS);
            CHECK(out[0].real == 0 && out[0].imag == 0);
            for (size_t i = 1; i < n; i++) {
                double scale = commutator ? 2.0 : 1.0;
                CHECK(out[i].real == scale * products[k][i].real);
                CHECK(out[i].imag == scale * products[k][i].imag);
            }
        }
    }
    // Non-dyadic inputs make the schedules round differently, so the test
    // fails if the DIRECT argument is silently ignored
    CHECK(memcmp(products[0] + 1, products[1] + 1, (n - 1) * sizeof(out[0])) != 0);
    free(scratch);
    return 1;
}

/* Canonical residues are a precondition the library does not check. This
 * test violates only that precondition, so sanitizers can confirm that table
 * addresses stay in bounds, without a nondeterministic writer race or a
 * promise about the results. */
// Exercise the scalar table lookups used by eligible 8/16/32-lane backends
static int lookup_bounds(void) {
#if defined(FASTCD_TEST_FINITE_SIMD)
    if (!__builtin_cpu_supports("sse4.1")) return 1;
    const uint32_t degrees[] = {2, 5, 2};
    const size_t levels[] = {4, 4, 5};
    for (size_t i = 0; i < 3; i++) {
        FastCDField field;
        FastCDPlanFF plan;
        CHECK(fastcd_field_init(&field, 5, degrees[i], NULL) == FASTCD_FF_SUCCESS);
        CHECK(fastcd_ff_plan_init(&plan, &field, levels[i]) == FASTCD_FF_SUCCESS);
        uint32_t* data =
                malloc((3 * plan.words + plan.workspace_words) * sizeof(*data));
        CHECK(data != NULL);
        uint32_t* x = data;
        uint32_t* y = x + plan.words;
        uint32_t* out = y + plan.words;
        uint32_t* work = out + plan.words;
        for (size_t j = 0; j < plan.words; j++) {
            x[j] = UINT32_MAX - (uint32_t)j;
            y[j] = (uint32_t)(j % 5);
        }
        CHECK(fastcd_mul_with_algorithm_ff(out, x, y, &plan,
                      FASTCD_ALGORITHM_QUASILINEAR, work,
                      plan.workspace_words) == FASTCD_FF_SUCCESS);
        free(data);
    }
#endif
    return 1;
}

// Exercise rejected setup, isotropic norms, and overflow without huge allocations
static int validation(void) {
    FastCDField f;
    memset(&f, 0x5a, sizeof(f));
    FastCDField saved = f;
    uint32_t reducible[] = {0, 0, 1};
    CHECK(fastcd_field_init(&f, 2, 1, NULL) == FASTCD_FF_CHARACTERISTIC_TWO);
    CHECK(fastcd_field_init(&f, 15, 2, NULL) == FASTCD_FF_INVALID_FIELD);
    CHECK(fastcd_field_init(&f, 5, 2, reducible) == FASTCD_FF_INVALID_FIELD);
    CHECK(memcmp(&saved, &f, sizeof(f)) == 0);
    CHECK(fastcd_field_init(&f, 5, 1, NULL) == FASTCD_FF_SUCCESS);
    FastCDPlanFF p;
    for (size_t level = sizeof(size_t) * CHAR_BIT; level-- > 0;) {
        if (fastcd_ff_plan_init(&p, &f, level) == FASTCD_FF_SUCCESS) {
            size_t limit = SIZE_MAX / sizeof(uint32_t), length = 17;
            FastCDFFStatus expected = p.workspace_words > limit - 2 * p.words
                    ? FASTCD_FF_SIZE_OVERFLOW
                    : FASTCD_FF_SUCCESS;
            CHECK(fastcd_apply_workspace_size_ff(&p, FASTCD_OP_ASSOCIATOR, &length) ==
                    expected);
            if (expected) CHECK(length == 17);
            break;
        }
    }
    CHECK(fastcd_ff_plan_init(&p, &f, 1) == FASTCD_FF_SUCCESS);
    CHECK(fastcd_apply_workspace_size_ff(&p, FASTCD_OP_MULTIPLY, &p.dimension) ==
            FASTCD_FF_OVERLAPPING_BUFFERS);
    CHECK(p.dimension == 2);
    uint32_t x[2] = {1, 2}, out[2] = {17, 19}, q;
    CHECK(fastcd_quadratic_norm_ff(&q, x, &p) == FASTCD_FF_SUCCESS && q == 0);
    CHECK(fastcd_apply_ff(out, x, NULL, NULL, &p, FASTCD_OP_RECIPROCAL, 0,
                  FASTCD_ALGORITHM_AUTO, NULL, 0) == FASTCD_FF_ZERO_NORM);
    CHECK(out[0] == 17 && out[1] == 19);
    FastCDComplexF64 cx[2] = {{1, 0}, {0, 1}}, cq, co[2] = {{17, 19}, {23, 29}};
    CHECK(fastcd_quadratic_norm_c128(&cq, cx, 2) == FASTCD_SUCCESS);
    CHECK(cq.real == 0 && cq.imag == 0);
    CHECK(fastcd_apply_c128(co, cx, NULL, NULL, 2, FASTCD_OP_RECIPROCAL, 0,
                  FASTCD_ALGORITHM_AUTO, NULL) == FASTCD_ZERO_NORM);
    CHECK(co[0].real == 17 && co[1].imag == 29);
    double r[2] = {3, 4}, rq;
    CHECK(fastcd_quadratic_norm_f64(&rq, r, 2) == FASTCD_SUCCESS && rq == 25);
    return 1;
}

// Cover all supported extension degrees and representative CD sizes/characteristics
int main(void) {
    const uint32_t primes[] = {3, 5, 13, 65521, 2147483647};
    for (size_t i = 0; i < sizeof(primes) / sizeof(primes[0]); i++)
        for (uint32_t d = 1; d <= 12; d++)
            if (!finite_case(primes[i], d, 6)) return EXIT_FAILURE;
    for (size_t level = 0; level <= 10; level++) {
        if (!finite_case(3, 1, level) || !finite_case(13, 5, level) ||
                !finite_case(65521, 2, level) || !complex_case((size_t)1 << level))
            return EXIT_FAILURE;
    }
    if (!finite_case(3, 2, 8) || !finite_case(193, 1, 7) || !finite_case(257, 1, 7))
        return EXIT_FAILURE;
    // Cover the dimensions where the packed backends of these fields begin
    for (size_t level = 4; level <= 8; level++)
        if (!finite_case(3, 5, level) || !finite_case(5, 1, level) ||
                !finite_case(13, 1, level))
            return EXIT_FAILURE;
    if (!validation() || !complex_cross_algorithm() || !lookup_bounds())
        return EXIT_FAILURE;
    puts("Complex and finite-field domain checks passed");
    return EXIT_SUCCESS;
}
