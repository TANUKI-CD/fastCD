/* Check the generated direct value schedule over F_13 through dimension 1024.
 * The oracle is a signed-XOR convolution from the defining doubling law, and
 * the test also counts value multiplications and guards the scratch bounds. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef uint32_t Word;

typedef struct {
    size_t* products;
} Context;

// Multiply residues modulo 13 and count every value multiplication
static Word multiply(Word a, Word b, const Context* ctx) {
    ++*ctx->products;
    return a * b % 13;
}

// Instantiate the schedule with plain residue arithmetic modulo 13
#define FASTCD_FF_UNUSED
#define FASTCD_FF_NAME(name) test_##name
#define VADD(a, b) (((a) + (b)) % 13)
#define VSUB(a, b) (((a) + 13 - (b)) % 13)
#define VMUL(a, b) multiply(a, b, ctx)
#define VZERO 0
#include "finite/direct_value.h"

// Obtain basis signs from the defining recursion, without production tables
static int sign(size_t n, size_t i, size_t j) {
    if (n == 1) return 1;
    size_t h = n / 2;
    if (i < h && j < h) return sign(h, i, j);
    if (i < h) return sign(h, j - h, i);
    if (j < h) return (j ? -1 : 1) * sign(h, i - h, j);
    return (j == h ? -1 : 1) * sign(h, j - h, i - h);
}

// Compare exact products, multiplication counts and scratch canaries at each N
int main(void) {
    for (size_t n = 1; n <= 1024; n *= 2) {
        // Leaves through eight coordinates use no scratch; larger records 2N-16
        size_t scratch = n <= 8 ? 0 : 2 * n - 16;
        Word* data = calloc(4 * n + scratch + 2, sizeof(*data));
        if (!data) return EXIT_FAILURE;
        Word *x = data, *y = x + n, *out = y + n, *expected = out + n;
        Word* work = expected + n + 1;
        for (size_t i = 0; i < n; i++) {
            x[i] = (Word)((i * 7 + 3) % 13);
            y[i] = (Word)((i * 11 + 2) % 13);
        }
        // Accumulate the signed-XOR convolution as the reference product
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < n; j++) {
                Word term = x[i] * y[j] % 13;
                expected[i ^ j] =
                        (expected[i ^ j] + (sign(n, i, j) > 0 ? term : 13 - term)) % 13;
            }
        // Guard one word on each side of the scratch region
        size_t products = 0;
        Context ctx = {&products};
        work[-1] = UINT32_C(0xdeadbeef);
        work[scratch] = UINT32_C(0xbaadf00d);
        test_direct_values(out, x, y, n, work, &ctx);
        if (products != n * n || work[-1] != UINT32_C(0xdeadbeef) ||
                work[scratch] != UINT32_C(0xbaadf00d))
            return EXIT_FAILURE;
        for (size_t i = 0; i < n; i++)
            if (out[i] != expected[i]) return EXIT_FAILURE;
        free(data);
    }
    puts("Finite-field direct schedule checks passed");
    return EXIT_SUCCESS;
}
