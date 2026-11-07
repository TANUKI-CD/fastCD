/* One compile-time polynomial-degree specialization of the buffer schedule.
 * This header intentionally has no guard: finite.c includes it for d=1..12.
 * Scalar primitives permit exact aliases, which the nonrecursive-map temporaries
 * use throughout recursion. No scalar call performs a degree dispatch. */
#define FASTCD_FF_WORD uint32_t
#define FASTCD_FF_STEP FASTCD_FF_DEGREE
#define FASTCD_FF_CONTEXT const FastCDPlanFF
#define FASTCD_FF_JOIN_INNER(a, b) a##b
#define FASTCD_FF_JOIN(a, b) FASTCD_FF_JOIN_INNER(a, b)
#define FASTCD_FF_NAME_INNER(d, name) fastcd_ff_d##d##_##name
#define FASTCD_FF_NAME_OUTER(d, name) FASTCD_FF_NAME_INNER(d, name)
#ifndef FASTCD_FF_NAME
#define FASTCD_FF_NAME(name) FASTCD_FF_NAME_OUTER(FASTCD_FF_DEGREE, name)
#endif

// Add one complete polynomial-basis coefficient
static inline void FASTCD_FF_NAME(add)(
        uint32_t* o, const uint32_t* a, const uint32_t* b, const FastCDPlanFF* ctx) {
    for (size_t i = 0; i < FASTCD_FF_DEGREE; i++)
        o[i] = fastcd_ff_add(a[i], b[i], &ctx->field);
}

// Subtract one complete polynomial-basis coefficient
static inline void FASTCD_FF_NAME(sub)(
        uint32_t* o, const uint32_t* a, const uint32_t* b, const FastCDPlanFF* ctx) {
    for (size_t i = 0; i < FASTCD_FF_DEGREE; i++)
        o[i] = fastcd_ff_sub(a[i], b[i], &ctx->field);
}

// Scale by a prime-subfield scalar, specializing identity and modular halving
static inline void FASTCD_FF_NAME(scale)(
        uint32_t* o, const uint32_t* a, uint32_t s, const FastCDPlanFF* ctx) {
    if (s == 1) {
        memmove(o, a, FASTCD_FF_DEGREE * sizeof(uint32_t));
        return;
    }
    if (s == ctx->inverse_two) {
        // Add p only to odd residues before an exact unsigned right shift.
        uint32_t p = ctx->field.prime;
        for (size_t i = 0; i < FASTCD_FF_DEGREE; i++)
            o[i] = (a[i] + (p & (0u - (a[i] & 1u)))) >> 1;
    } else
        for (size_t i = 0; i < FASTCD_FF_DEGREE; i++)
            o[i] = fastcd_ff_mul(a[i], s, &ctx->field);
}

#define FASTCD_FF_ADD(o, a, b, c) FASTCD_FF_NAME(add)(o, a, b, c)
#define FASTCD_FF_SUB(o, a, b, c) FASTCD_FF_NAME(sub)(o, a, b, c)
#define FASTCD_FF_SCALE(o, a, s, c) FASTCD_FF_NAME(scale)(o, a, s, c)
#define FASTCD_FF_MUL(o, a, b, c)                                                      \
    FASTCD_FF_JOIN(fastcd_ff_product, FASTCD_FF_DEGREE)(o, a, b, &(c)->field)
#define FASTCD_FF_COPY(o, a, c)                                                        \
    ((void)(c), memmove(o, a, FASTCD_FF_DEGREE * sizeof(uint32_t)))
#define FASTCD_FF_ZERO(o, c)                                                           \
    ((void)(c), memset(o, 0, FASTCD_FF_DEGREE * sizeof(uint32_t)))
#define FASTCD_FF_INVERSE_TWO(c) ((c)->inverse_two)
#define FASTCD_FF_INVERSE_N(c) ((c)->inverse_dimension)
#include "schedule_impl.h"
#include "direct_instance.h"
#undef FASTCD_FF_INVERSE_N
#undef FASTCD_FF_INVERSE_TWO
#undef FASTCD_FF_ZERO
#undef FASTCD_FF_COPY
#undef FASTCD_FF_MUL
#undef FASTCD_FF_SCALE
#undef FASTCD_FF_SUB
#undef FASTCD_FF_ADD
#undef FASTCD_FF_NAME
#undef FASTCD_FF_NAME_OUTER
#undef FASTCD_FF_NAME_INNER
#undef FASTCD_FF_JOIN
#undef FASTCD_FF_JOIN_INNER
#undef FASTCD_FF_CONTEXT
#undef FASTCD_FF_STEP
#undef FASTCD_FF_WORD
