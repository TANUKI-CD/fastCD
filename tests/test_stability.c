/* Reduction regressions across generated-leaf and recursive boundaries. The
 * oracle is a signed-XOR convolution from the defining doubling law, and
 * bounded dyadics make both the reference and production results exact. */
#include "fastcd/complex.h"
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FASTCD_REAL double
#define FASTCD_ZERO 0.0
#define FASTCD_NAME(n) stability_##n
#include "balanced_sum.h"
#undef FASTCD_NAME
#undef FASTCD_ZERO
#undef FASTCD_REAL

#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #c); return 0; \
} } while (0)

// Obtain basis signs from the defining recursion, without production tables
static int sign_basis(size_t n, size_t i, size_t j) {
    if (n == 1) return 1;
    size_t h = n / 2;
    if (i < h && j < h) return sign_basis(h, i, j);
    if (i < h) return sign_basis(h, j - h, i);
    if (j < h) return (j ? -1 : 1) * sign_basis(h, i - h, j);
    return (j == h ? -1 : 1) * sign_basis(h, j - h, i - h);
}

// Draw a multiple of 1/8 in [-4, 4) from a linear congruential generator
static double sample(uint64_t* state) {
    *state = *state * UINT64_C(6364136223846793005) + 1;
    return (double)((int)(*state >> 58) - 32) * 0.125;
}

/* Independent recursive description of the blocked summation tree, without the
 * binary counter, occupied slots or helper calls of balanced_sum.h. */
// Sum blocks of FASTCD_SUM_BLOCK terms in sequence and combine halves recursively
static double prior_block_tree(const double* terms, size_t length) {
    if (length == 0) return 0.0;
    if (length <= FASTCD_SUM_BLOCK) {
        double result = terms[0];
        for (size_t i = 1; i < length; i++) result = result + terms[i];
        return result;
    }
    size_t split = FASTCD_SUM_BLOCK;
    while (split <= length / 2) split *= 2;
    if (split == length) split /= 2;
    return prior_block_tree(terms, split) +
            prior_block_tree(terms + split, length - split);
}

/* The whole-block entry points must reproduce the recursive tree exactly in
 * each stream, including incomplete blocks and signed-zero singletons. */
// Check the one-, two- and three-stream block sums against the recursive tree
static int whole_block_cases(const double* first, size_t capacity) {
    double second[2052], third[2052];
    CHECK(capacity == sizeof(second) / sizeof(second[0]));
    for (size_t i = 0; i < capacity; i++) {
        second[i] = -first[(i + 7) % capacity] * 0x1p-13;
        third[i] = first[capacity - 1 - i] * 0x1p11;
    }
    for (size_t length = 0; length < capacity; length++) {
        stability_BalancedSum single;
        stability_sum_init(&single);
        stability_BalancedTree2 two;
        stability_BalancedTree3 three;
        two.blocks = three.blocks = 0;
        size_t index = 0;
        for (; length - index >= FASTCD_SUM_BLOCK; index += FASTCD_SUM_BLOCK) {
            double a = prior_block_tree(first + index, FASTCD_SUM_BLOCK);
            double b = prior_block_tree(second + index, FASTCD_SUM_BLOCK);
            double c = prior_block_tree(third + index, FASTCD_SUM_BLOCK);
            stability_Sum2 pair = {a, b};
            stability_Sum3 triple = {a, b, c};
            stability_sum_block(&single, a);
            stability_sum2_block(&two, pair);
            stability_sum3_block(&three, triple);
        }
        size_t remaining = length - index;
        double expected[3] = {prior_block_tree(first, length),
                prior_block_tree(second, length), prior_block_tree(third, length)};
        stability_Sum3 tail = {prior_block_tree(first + index, remaining),
                prior_block_tree(second + index, remaining),
                prior_block_tree(third + index, remaining)};
        single.block = tail.first;
        single.used = (unsigned)remaining;
        stability_Sum2 pair_tail = {tail.first, tail.second};
        stability_Sum2 pair = stability_sum2_finish(&two, pair_tail, remaining != 0);
        stability_Sum3 triple = stability_sum3_finish(&three, tail, remaining != 0);
        stability_Sum3 arrays = stability_sum3_arrays(first, second, third, length);
        double actual[] = {stability_sum_finish(&single), pair.first, pair.second,
                triple.first, triple.second, triple.third,
                arrays.first, arrays.second, arrays.third};
        const size_t stream[] = {0, 0, 1, 0, 1, 2, 0, 1, 2};
        for (size_t i = 0; i < sizeof(actual) / sizeof(actual[0]); i++)
            CHECK(memcmp(actual + i, expected + stream[i], sizeof(double)) == 0);
        CHECK(single.blocks == length / FASTCD_SUM_BLOCK);
        CHECK(two.blocks == single.blocks && three.blocks == single.blocks);
    }
    return 1;
}

// Check empty and singleton sums, incomplete blocks and every binary carry
// boundary of the accumulator
static int accumulator_cases(void) {
    stability_BalancedSum sum;
    stability_sum_init(&sum);
    CHECK(stability_sum_finish(&sum) == 0);
    stability_sum_push(&sum, -0.0);
    CHECK(signbit(stability_sum_finish(&sum)));
    for (size_t length = 1; length <= 2051; length++) {
        stability_sum_init(&sum);
        double expected = 0;
        for (size_t i = 0; i < length; i++) {
            double value = (double)(i % 17) * (i & 1 ? -0.125 : 0.25);
            expected += value;
            stability_sum_push(&sum, value);
        }
        CHECK(stability_sum_finish(&sum) == expected);
        // Finishing is read-only, so the sum can keep growing
        stability_sum_push(&sum, 0.5);
        CHECK(stability_sum_finish(&sum) == expected + 0.5);
    }
    double terms[2052];
    for (size_t i = 0; i < sizeof(terms) / sizeof(terms[0]); i++) {
        double sign = i & 1 ? -1.0 : 1.0;
        terms[i] = sign * ldexp(1.0 / (double)(i % 19 + 1), (int)(i % 61) - 30);
        if (i % 17 == 0) terms[i] = i & 1 ? -0.0 : 0.0;
    }
    terms[0] = -0.0;
    for (size_t length = 0; length <= 2051; length++) {
        stability_sum_init(&sum);
        for (size_t i = 0; i < length; i++) stability_sum_push(&sum, terms[i]);
        double actual = stability_sum_finish(&sum);
        double expected = prior_block_tree(terms, length);
        CHECK(memcmp(&actual, &expected, sizeof(actual)) == 0);
    }
    CHECK(whole_block_cases(terms, sizeof(terms) / sizeof(terms[0])));
    return 1;
}

/* Small dyadics keep every sum and product exact in families 0 to 7 and 9,
 * and the non-dyadic family 8 is measured against a long-double oracle. */
// Fill n coefficients of x and y with one of the ten operand families
static void fixture(FastCDComplexF64* x, FastCDComplexF64* y, size_t n,
        unsigned family, uint64_t* state) {
    for (size_t i = 0; i < n; i++) {
        x[i].real = sample(state);
        x[i].imag = sample(state);
        y[i].real = sample(state);
        y[i].imag = sample(state);
        if (family == 1) y[i] = x[i];
        if (family == 2) {
            y[i].real = -x[i].real;
            y[i].imag = -x[i].imag;
        }
        if (family == 3 && i != 0 && i != n / 2 && i != n - 1) {
            x[i].real = x[i].imag = y[i].real = y[i].imag = 0;
        }
        if (family == 4 && i != 0) x[i].real = x[i].imag = 0;
        if (family == 5 && i == 0) x[i].real = x[i].imag = 0;
        if (family == 6) {
            x[i].real = i & 1 ? -1.0 : 1.0;
            x[i].imag = 0.5;
            y[i].real = 1.0;
            y[i].imag = i & 2 ? -0.5 : 0.5;
        }
        if (family == 7) {
            x[i].real = (double)(i % 4) * 0.25;
            x[i].imag = (double)(i % 8) * 0.125;
            y[i].real = (double)(i % 3) * -0.5;
            y[i].imag = (double)(i % 2) * 0.25;
        }
        if (family == 8) {
            x[i].real = sin((double)i * 0.71);
            x[i].imag = cos((double)i * 0.13);
            y[i].real = x[i].real + 0x1p-40 * cos((double)i);
            y[i].imag = -x[i].imag + 0x1p-40 * sin((double)i);
        }
    }
    if (family == 9) {
        // Exact cancellation of the real scalar coefficient with 2, 4, ..., 512
        // alternating unit terms, with complex phases that also catch conjugation
        for (size_t i = 0; i < n; i++) {
            x[i].real = 1;
            x[i].imag = 0.5;
            y[i].real = i & 1 ? -1 : 1;
            y[i].imag = i & 1 ? -0.5 : 0.5;
        }
        y[0].real = -1;
        y[0].imag = -0.5;
    }
}

// Compare batched, single and reference products of every family at dimension n
static int run_dimension(size_t n) {
    const size_t count = 3, stride = n + 2;
    FastCDComplexF64* data = calloc(4 * count * stride, sizeof(*data));
    double* real = calloc(4 * count * stride, sizeof(*real));
    CHECK(data && real);
    FastCDComplexF64 *x = data, *y = x + count * stride;
    FastCDComplexF64 *out = y + count * stride, *single = out + count * stride;
    double *rx = real, *ry = rx + count * stride;
    double *ro = ry + count * stride, *rs = ro + count * stride;
    size_t clen, rlen;
    CHECK(fastcd_workspace_size_c128(n, FASTCD_OP_MULTIPLY, &clen) == FASTCD_SUCCESS);
    CHECK(fastcd_mul_algorithm_workspace_size(n, FASTCD_ALGORITHM_QUASILINEAR,
                &rlen) == FASTCD_SUCCESS);
    double* cw = malloc((clen + 2) * sizeof(*cw));
    double* rw = malloc((rlen + 2) * sizeof(*rw));
    CHECK(cw && rw);
    FastCDWorkspaceF64 complex_work = {cw + 1, clen}, real_work = {rw + 1, rlen};
    cw[0] = rw[0] = 12345.5;
    cw[clen + 1] = rw[rlen + 1] = -98765.5;
    uint64_t state = n + 1701;
    for (unsigned family = 0; family < 10; family++) {
        for (size_t p = 0; p < count; p++) {
            fixture(x + p * stride, y + p * stride, n, family, &state);
            for (size_t i = 0; i < n; i++) {
                rx[p * stride + i] = x[p * stride + i].real;
                ry[p * stride + i] = y[p * stride + i].real;
                ro[p * stride + i] = 123;
                out[p * stride + i].real = 123;
            }
            ro[p * stride + n] = out[p * stride + n].real = 34567;
        }
        CHECK(fastcd_mul_batch_with_algorithm_f64(ro, stride, rx, stride, ry,
                    stride, count, n, FASTCD_ALGORITHM_QUASILINEAR,
                    &real_work) == FASTCD_SUCCESS);
        CHECK(fastcd_mul_batch_c128(out, stride, x, stride, y, stride, count, n,
                    FASTCD_ALGORITHM_QUASILINEAR, &complex_work) == FASTCD_SUCCESS);
        for (size_t p = 0; p < count; p++) {
            CHECK(ro[p * stride + n] == 34567 && out[p * stride + n].real == 34567);
            CHECK(fastcd_mul_with_algorithm_f64(rs, rx + p * stride,
                        ry + p * stride, n, FASTCD_ALGORITHM_QUASILINEAR,
                        &real_work) == FASTCD_SUCCESS);
            CHECK(fastcd_mul_with_algorithm_c128(single, x + p * stride,
                        y + p * stride, n, FASTCD_ALGORITHM_QUASILINEAR,
                        &complex_work) == FASTCD_SUCCESS);
            CHECK(memcmp(rs, ro + p * stride, n * sizeof(*rs)) == 0);
            CHECK(memcmp(single, out + p * stride, n * sizeof(*single)) == 0);
            long double scale = 0;
            for (size_t i = 0; i < n; i++) {
                size_t a = p * stride + i;
                scale += fabsl(x[a].real) + fabsl(x[a].imag) +
                        fabsl(y[a].real) + fabsl(y[a].imag);
            }
            for (size_t k = 0; k < n; k++) {
                long double r = 0, im = 0, rr = 0;
                for (size_t i = 0; i < n; i++) {
                    size_t a = p * stride + i, b = p * stride + (i ^ k);
                    long double s = (long double)sign_basis(n, i, i ^ k);
                    r += s * ((long double)x[a].real * y[b].real -
                            (long double)x[a].imag * y[b].imag);
                    im += s * ((long double)x[a].real * y[b].imag +
                            (long double)x[a].imag * y[b].real);
                    rr += s * (long double)rx[a] * ry[b];
                }
                if (family != 8) {
                    CHECK(single[k].real == (double)r);
                    CHECK(single[k].imag == (double)im);
                    CHECK(rs[k] == (double)rr);
                } else {
                    long double tolerance = 32 * DBL_EPSILON * scale * scale;
                    CHECK(fabsl((long double)single[k].real - r) <= tolerance);
                    CHECK(fabsl((long double)single[k].imag - im) <= tolerance);
                    CHECK(fabsl((long double)rs[k] - rr) <= tolerance);
                }
            }
        }
        CHECK(cw[0] == 12345.5 && cw[clen + 1] == -98765.5);
        CHECK(rw[0] == 12345.5 && rw[rlen + 1] == -98765.5);
    }
    // A short workspace or a forbidden output alias must leave the output intact
    if (rlen) {
        real_work.length--;
        double before = ro[0];
        CHECK(fastcd_mul_with_algorithm_f64(ro, rx, ry, n,
                    FASTCD_ALGORITHM_QUASILINEAR, &real_work) ==
                FASTCD_INSUFFICIENT_WORKSPACE);
        CHECK(ro[0] == before);
        real_work.length++;
    }
    CHECK(fastcd_mul_with_algorithm_f64(rx, rx, ry, n,
                FASTCD_ALGORITHM_QUASILINEAR, &real_work) == FASTCD_OVERLAPPING_BUFFERS);
    complex_work.length--;
    CHECK(fastcd_mul_with_algorithm_c128(out, x, y, n,
                FASTCD_ALGORITHM_QUASILINEAR, &complex_work) ==
            FASTCD_INSUFFICIENT_WORKSPACE);
    complex_work.length++;
    CHECK(fastcd_mul_with_algorithm_c128(x, x, y, n,
                FASTCD_ALGORITHM_QUASILINEAR, &complex_work) == FASTCD_OVERLAPPING_BUFFERS);
    free(cw);
    free(rw);
    free(real);
    free(data);
    return 1;
}

// Run the accumulator cases and every dimension around the generated cutoffs
int main(void) {
    // Dimensions on both sides of the generated cutoffs 8 and 16
    const size_t dimensions[] = {2, 4, 8, 16, 32, 64, 128, 256, 512};
    if (sign_basis(4, 1, 2) != 1 || sign_basis(4, 2, 1) != -1)
        return EXIT_FAILURE;
    if (!accumulator_cases()) return EXIT_FAILURE;
    for (size_t d = 0; d < sizeof(dimensions) / sizeof(dimensions[0]); d++)
        if (!run_dimension(dimensions[d])) return EXIT_FAILURE;
    puts("Balanced-reduction real/complex integration checks passed");
    return EXIT_SUCCESS;
}
