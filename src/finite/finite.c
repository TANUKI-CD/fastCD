#include <string.h>
#include "internal.h"
#include "field_internal.h"
#include "finite_backend.h"

/* Instantiate the buffer schedule once per supported polynomial degree. A
 * product selects its degree outside the recursion, so the compiler sees
 * constant coefficient sizes and field operations. */
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

// Run the buffer schedule specialized for the plan's degree
static void dispatch(uint32_t* out, const uint32_t* x, const uint32_t* y,
        const FastCDPlanFF* plan, uint32_t* workspace, int loop) {
    // Public CD calls pass loop = 0; the loop component serves the private template
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

// Evaluate one product after single-product or batch validation
void fastcd_ff_mul_unchecked(uint32_t* out, const uint32_t* x, const uint32_t* y,
        const FastCDPlanFF* plan, FastCDAlgorithm algorithm, uint32_t* workspace) {
    // Resolve AUTO from the dimension
    if (algorithm == FASTCD_ALGORITHM_AUTO)
        algorithm = fastcd_domain_auto_algorithm(plan->dimension);
    if (algorithm == FASTCD_ALGORITHM_DIRECT) {
        /* Packed backends run the defining recursion on their own coefficient
         * encodings. The scalar recursion covers every other case. */
        if (plan->dimension < FASTCD_FF_PACKED_MIN_DIMENSION ||
                !fastcd_ff_finite_execute(
                        out, x, y, plan, workspace, FASTCD_FF_SCHEDULE_DIRECT))
            fastcd_ff_direct(out, x, y, plan, workspace);
    } else if (!fastcd_ff_finite_execute(
                       out, x, y, plan, workspace, FASTCD_FF_SCHEDULE_CD))
        /* Backends convert canonical coefficients inside the call and return
         * one when they handled the product. The prime value schedule always
         * does, so only extension fields without a backend reach the portable
         * buffer schedule. */
        dispatch(out, x, y, plan, workspace, 0);
}

// Run the scalar direct recursion specialized for the plan's degree
void fastcd_ff_direct(uint32_t* out, const uint32_t* x, const uint32_t* y,
        const FastCDPlanFF* p, uint32_t* work) {
    /* Direct products use this recursion below FASTCD_FF_PACKED_MIN_DIMENSION
     * and when no packed backend serves the field. It shares only the
     * fixed-degree field primitives with quasilinear multiplication. */
    switch (p->field.degree) {
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
