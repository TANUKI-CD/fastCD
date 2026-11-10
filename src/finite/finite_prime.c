/* Portable register-value arithmetic for prime fields, with no representation
 * conversion. Extension fields use the degree-specialized buffer schedule. */
#include "finite_backend.h"
#include "field_internal.h"
// Value types and operations for the shared register-value schedule
typedef uint32_t Word;
typedef FastCDPlanFF Context;
#define FASTCD_FF_NAME(n) scalar_prime_##n
#define VADD(a, b) fastcd_ff_add(a, b, &ctx->field)
#define VSUB(a, b) fastcd_ff_sub(a, b, &ctx->field)
#define VMUL(a, b) fastcd_ff_mul(a, b, &ctx->field)

// Fuse ab-cd into one bounded reduction, keeping scalars in registers
static inline Word prime_difference(
        Word a, Word b, Word c, Word d, const Context* ctx) {
    // ab + (p-c)d < 2p^2 < 2^63, including c = 0
    return fastcd_ff_reduce(
            (uint64_t)a * b + (uint64_t)(ctx->field.prime - c) * d, &ctx->field);
}

#define VDIFF(a, b, c, d) prime_difference(a, b, c, d, ctx)

// Divide a canonical odd-characteristic residue by two without multiplication
static inline Word prime_half(Word a, const Context* ctx) {
    return (a + (ctx->field.prime & (0u - (a & 1u)))) >> 1;
}

#define VHALF(a) prime_half(a, ctx)
#define VNORMALIZE(a)                                                                  \
    (ctx->inverse_dimension == 1                                                       \
                    ? (a)                                                              \
                    : fastcd_ff_mul(a, ctx->inverse_dimension, &ctx->field))
#define VZERO 0
#include "finite_value.h"

// Execute the portable value schedule with no polynomial or SIMD conversion
void fastcd_ff_finite_prime_scalar(uint32_t* out, const uint32_t* x, const uint32_t* y,
        const FastCDPlanFF* p, uint32_t* work, int loop) {
    if (loop)
        scalar_prime_loop(out, x, y, p->dimension, work, p);
    else
        scalar_prime_cd(out, x, y, p->dimension, work, p);
}
