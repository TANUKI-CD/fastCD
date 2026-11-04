#include <string.h>
#include "internal.h"
#include "field_internal.h"
#include "finite_backend.h"

/* Instantiate the same buffer-based schedule at every supported polynomial
 * degree. A product chooses its degree once, outside the recursive loops;
 * the compiler then sees constant coefficient sizes and field operations. */
#define FASTCD_FF_STATS_DECL
#define FASTCD_FF_STATS_ARG
#define FASTCD_FF_RECORD_NODE(n) ((void)0)
#define FASTCD_FF_DEGREE 1
#include "finite_instance.h"
#undef FASTCD_FF_DEGREE
#define FASTCD_FF_DEGREE 2
#include "finite_instance.h"
#undef FASTCD_FF_DEGREE
#define FASTCD_FF_DEGREE 3
#include "finite_instance.h"
#undef FASTCD_FF_DEGREE
#define FASTCD_FF_DEGREE 4
#include "finite_instance.h"
#undef FASTCD_FF_DEGREE
#define FASTCD_FF_DEGREE 5
#include "finite_instance.h"
#undef FASTCD_FF_DEGREE
#define FASTCD_FF_DEGREE 6
#include "finite_instance.h"
#undef FASTCD_FF_DEGREE
#define FASTCD_FF_DEGREE 7
#include "finite_instance.h"
#undef FASTCD_FF_DEGREE
#define FASTCD_FF_DEGREE 8
#include "finite_instance.h"
#undef FASTCD_FF_DEGREE
#define FASTCD_FF_DEGREE 9
#include "finite_instance.h"
#undef FASTCD_FF_DEGREE
#define FASTCD_FF_DEGREE 10
#include "finite_instance.h"
#undef FASTCD_FF_DEGREE
#define FASTCD_FF_DEGREE 11
#include "finite_instance.h"
#undef FASTCD_FF_DEGREE
#define FASTCD_FF_DEGREE 12
#include "finite_instance.h"
#undef FASTCD_FF_DEGREE

// Dispatch once per complete product, leaving the recursive loops specialized.
// The private template also defines a loop component; every public CD call
// below passes loop=0 and never enters that companion arithmetic path.
static void dispatch(uint32_t* out, const uint32_t* x, const uint32_t* y,
        const FastCDPlanFF* plan, uint32_t* workspace, int loop) {
    switch (plan->field.degree) {
#define FASTCD_FF_CASE(d)                                                              \
    case d: {                                                                          \
        if (loop) {                                                                    \
            fastcd_ff_d##d##_loop(out, x, y, plan->dimension, workspace, plan);        \
        } else {                                                                       \
            fastcd_ff_d##d##_cd(out, x, y, plan->dimension, workspace, plan);          \
        }                                                                              \
        break;                                                                         \
    }
        FASTCD_FF_CASE(1);
        FASTCD_FF_CASE(2);
        FASTCD_FF_CASE(3);
        FASTCD_FF_CASE(4);
        FASTCD_FF_CASE(5);
        FASTCD_FF_CASE(6);
        FASTCD_FF_CASE(7);
        FASTCD_FF_CASE(8);
        FASTCD_FF_CASE(9);
        FASTCD_FF_CASE(10);
        FASTCD_FF_CASE(11);
        FASTCD_FF_CASE(12);
#undef FASTCD_FF_CASE
    }
}

// Validate a full CD record and its active scratch before entering any backend
FastCDFFStatus fastcd_ff_cd_mul(uint32_t* out, const uint32_t* x, const uint32_t* y,
        const FastCDPlanFF* plan, uint32_t* workspace, size_t available) {
    if (plan == NULL) return FASTCD_FF_NULL_POINTER;
    if (available < plan->workspace_words) return FASTCD_FF_INSUFFICIENT_WORKSPACE;
    FastCDFFStatus status =
            fastcd_ff_validate(out, x, y, plan->words * sizeof(uint32_t), plan,
                    sizeof(*plan), workspace, plan->workspace_words * sizeof(uint32_t),
                    plan->workspace_words * sizeof(uint32_t));
    if (status != FASTCD_FF_SUCCESS) return status;
    fastcd_ff_mul_unchecked(out, x, y, plan, FASTCD_ALGORITHM_QUASILINEAR, workspace);
    return FASTCD_FF_SUCCESS;
}

/* Called after a single-product or complete-batch validation. ISA backends
 * convert canonical coefficients only inside this call; the degree-specific
 * buffer schedule remains the portable fallback when no backend applies. */
// Evaluate one product without repeating public buffer validation
void fastcd_ff_mul_unchecked(uint32_t* out, const uint32_t* x, const uint32_t* y,
        const FastCDPlanFF* plan, FastCDAlgorithm algorithm, uint32_t* workspace) {
    if (algorithm == FASTCD_ALGORITHM_DIRECT)
        fastcd_ff_direct(out, x, y, plan, workspace);
    else if (!fastcd_ff_finite_execute(out, x, y, plan, workspace, 0))
        // Packed backends and the prime value schedule return one when they
        // handled the product. Only extension fallback reaches this dispatch.
        dispatch(out, x, y, plan, workspace, 0);
}

/* Direct multiplication shares only the fixed-degree scalar primitives.
 * Its independent four-product CD recursion uses no tensor evaluation maps,
 * making its arithmetic schedule a separate reference for quasilinear. */
// Dispatch the defining recursion once per complete product
void fastcd_ff_direct(uint32_t* out, const uint32_t* x, const uint32_t* y,
        const FastCDPlanFF* p, uint32_t* work) {
    switch (p->field.degree) {
        // This switch selects a constant-degree function, not an operation
        // table inside the defining recursion or its polynomial products.
#define DIRECT_CASE(d)                                                                 \
    case d:                                                                            \
        fastcd_ff_d##d##_direct(out, x, y, p->dimension, &p->field, work);             \
        break
        DIRECT_CASE(1);
        DIRECT_CASE(2);
        DIRECT_CASE(3);
        DIRECT_CASE(4);
        DIRECT_CASE(5);
        DIRECT_CASE(6);
        DIRECT_CASE(7);
        DIRECT_CASE(8);
        DIRECT_CASE(9);
        DIRECT_CASE(10);
        DIRECT_CASE(11);
        DIRECT_CASE(12);

#undef DIRECT_CASE
    }
}
