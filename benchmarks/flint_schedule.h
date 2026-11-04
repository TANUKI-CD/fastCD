#ifndef FASTCD_BENCH_FLINT_SCHEDULE_H
#define FASTCD_BENCH_FLINT_SCHEDULE_H

#include "fastcd/finite.h"
#include <flint/flint.h>
#include <flint/fq_nmod.h>
#include <flint/nmod.h>
#include <flint/nmod_poly.h>

/* Benchmark-only coefficient adapters. The arithmetic recurrence is the
 * maintained production template, rather than an independently optimized
 * or quadratic implementation attributed to FLINT. FLINT supplies the field
 * operations; it does not supply a Cayley--Dickson multiplication routine. */
typedef struct {
    FastCDPlanFF plan;
    size_t workspace_coefficients;
    nmod_t prime;
    fq_nmod_ctx_t extension;
} FlintPlan;

#define FASTCD_FF_STATS_DECL
#define FASTCD_FF_STATS_ARG
#define FASTCD_FF_RECORD_NODE(n) ((void)0)
#define FASTCD_FF_STEP 1
#define FASTCD_FF_CONTEXT const FlintPlan
#define FASTCD_FF_WORD ulong
#define FASTCD_FF_NAME(n) bench_flint_prime_##n
#define FASTCD_FF_ADD(o, a, b, c) ((o)[0] = nmod_add((a)[0], (b)[0], (c)->prime))
#define FASTCD_FF_SUB(o, a, b, c) ((o)[0] = nmod_sub((a)[0], (b)[0], (c)->prime))
#define FASTCD_FF_MUL(o, a, b, c) ((o)[0] = nmod_mul((a)[0], (b)[0], (c)->prime))
#define FASTCD_FF_SCALE(o, a, s, c) ((o)[0] = nmod_mul((a)[0], s, (c)->prime))
#define FASTCD_FF_ZERO(o, c) ((void)(c), (o)[0] = 0)
#define FASTCD_FF_COPY(o, a, c) ((void)(c), (o)[0] = (a)[0])
#define FASTCD_FF_INVERSE_TWO(c) ((c)->plan.inverse_two)
#define FASTCD_FF_INVERSE_N(c) ((c)->plan.inverse_dimension)
#include "schedule_impl.h"
#undef FASTCD_FF_COPY
#undef FASTCD_FF_ZERO
#undef FASTCD_FF_SCALE
#undef FASTCD_FF_MUL
#undef FASTCD_FF_SUB
#undef FASTCD_FF_ADD
#undef FASTCD_FF_NAME
#undef FASTCD_FF_WORD

#define FASTCD_FF_WORD fq_nmod_struct
#define FASTCD_FF_NAME(n) bench_flint_extension_##n
#define FASTCD_FF_ADD(o, a, b, c) fq_nmod_add(o, a, b, (c)->extension)
#define FASTCD_FF_SUB(o, a, b, c) fq_nmod_sub(o, a, b, (c)->extension)
#define FASTCD_FF_MUL(o, a, b, c) fq_nmod_mul(o, a, b, (c)->extension)
#define FASTCD_FF_SCALE(o, a, s, c) fq_nmod_mul_ui(o, a, s, (c)->extension)
#define FASTCD_FF_ZERO(o, c) fq_nmod_zero(o, (c)->extension)
#define FASTCD_FF_COPY(o, a, c) fq_nmod_set(o, a, (c)->extension)
#include "schedule_impl.h"
#undef FASTCD_FF_INVERSE_N
#undef FASTCD_FF_INVERSE_TWO
#undef FASTCD_FF_COPY
#undef FASTCD_FF_ZERO
#undef FASTCD_FF_SCALE
#undef FASTCD_FF_MUL
#undef FASTCD_FF_SUB
#undef FASTCD_FF_ADD
#undef FASTCD_FF_NAME
#undef FASTCD_FF_WORD
#undef FASTCD_FF_CONTEXT
#undef FASTCD_FF_STEP
#undef FASTCD_FF_RECORD_NODE
#undef FASTCD_FF_STATS_ARG
#undef FASTCD_FF_STATS_DECL

// Compare address spans without subtracting pointers to unrelated objects.
static int bench_flint_overlap(const void* a, size_t an, const void* b, size_t bn) {
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return an && bn && (x <= y ? y - x < an : x - y < bn);
}

/* Match the public native entry's complete-record buffer checks. All element
 * objects and their coefficient capacities are initialized before timing.
 * This wrapper validates the contiguous descriptor/word spans; extension
 * coefficient allocations are owned by those disjoint element objects. */
static FastCDFFStatus bench_flint_cd(void* out, const void* x, const void* y,
        const FlintPlan* plan, void* scratch, size_t available) {
    if (!plan || !out || !x || !y) return FASTCD_FF_NULL_POINTER;
    size_t need = plan->workspace_coefficients;
    size_t item = plan->plan.field.degree == 1 ? sizeof(ulong) : sizeof(fq_nmod_struct);
    size_t bytes = plan->plan.dimension * item, work = need * item;
    if (available < need) return FASTCD_FF_INSUFFICIENT_WORKSPACE;
    if (need && !scratch) return FASTCD_FF_NULL_POINTER;
    if (bench_flint_overlap(out, bytes, x, bytes) ||
            bench_flint_overlap(out, bytes, y, bytes) ||
            (x != y && bench_flint_overlap(x, bytes, y, bytes)) ||
            bench_flint_overlap(scratch, work, out, bytes) ||
            bench_flint_overlap(scratch, work, x, bytes) ||
            bench_flint_overlap(scratch, work, y, bytes) ||
            bench_flint_overlap(out, bytes, plan, sizeof(*plan)) ||
            bench_flint_overlap(scratch, work, plan, sizeof(*plan)))
        return FASTCD_FF_OVERLAPPING_BUFFERS;
    if (plan->plan.field.degree == 1)
        bench_flint_prime_cd(out, x, y, plan->plan.dimension, scratch, plan);
    else
        bench_flint_extension_cd(out, x, y, plan->plan.dimension, scratch, plan);
    return FASTCD_FF_SUCCESS;
}

#endif
