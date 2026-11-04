#include <limits.h>
#include "internal.h"
#include "finite_backend.h"

/* ---------- Plans and checked sizes ---------- */

/* Plans copy their field by value and retain no caller pointer. Build the
 * complete result locally, including overflow checks and backend scratch,
 * before publishing it so rejected setup leaves the caller's plan intact. */
// Precompute the dimensions, normalization scalars, and worst-case scratch
FastCDFFStatus fastcd_ff_plan_init(
        FastCDPlanFF* plan, const FastCDField* field, size_t level) {
    if (plan == NULL || field == NULL) return FASTCD_FF_NULL_POINTER;
    if (fastcd_ff_overlap(plan, sizeof(*plan), field, sizeof(*field)))
        return FASTCD_FF_OVERLAPPING_BUFFERS;
    if (field->prime == 2) return FASTCD_FF_CHARACTERISTIC_TWO;
    if (level >= sizeof(size_t) * CHAR_BIT) return FASTCD_FF_SIZE_OVERFLOW;
    size_t n = (size_t)1 << level, d = field->degree;
    if (n > SIZE_MAX / sizeof(uint32_t) / d / 2) return FASTCD_FF_SIZE_OVERFLOW;
    // The buffer schedule reserves 32 coefficients for local maps at each
    // node and four transformed half-size planes above its dimension-4 leaf.
    // Children run sequentially and share the tail of the same scratch.
    size_t cd = 32;
    for (size_t k = n / 2; k >= 4; k /= 2) {
        size_t add = k == 4 ? 32 : 32 + 2 * k;
        if (k > (SIZE_MAX - 32) / 2 || cd > SIZE_MAX - add)
            return FASTCD_FF_SIZE_OVERFLOW;
        // The first 32 slots are an upper bound also used by root temporaries.
        cd += add;
    }
    if (cd > SIZE_MAX / sizeof(uint32_t) / d) return FASTCD_FF_SIZE_OVERFLOW;
    FastCDPlanFF result = {0};
    result.field = *field;
    result.level = level;
    result.dimension = n;
    result.words = n * d;
    result.workspace_words = cd * d;
    // The chosen packed representation may need more scratch than the
    // generic schedule. Reserve both so execution can safely fall back.
    size_t selected_cd;
    fastcd_ff_finite_select(&result, 0, &selected_cd);
    if (selected_cd > SIZE_MAX / sizeof(uint32_t)) return FASTCD_FF_SIZE_OVERFLOW;
    if (selected_cd > result.workspace_words) result.workspace_words = selected_cd;
    // Repeated modular halving computes 1/N without forming N modulo p
    // in a potentially narrower integer type.
    result.inverse_two = (field->prime + 1) / 2;
    uint32_t inverse = 1;
    for (size_t i = 0; i < level; i++)
        inverse = (uint32_t)((uint64_t)inverse * result.inverse_two % field->prime);
    result.inverse_dimension = inverse;
    // The defining recursion needs less than 2*N coefficients of scratch.
    // One immutable plan can therefore serve direct and quasilinear calls.
    if (result.workspace_words < 2 * n * d) result.workspace_words = 2 * n * d;
    *plan = result;
    return FASTCD_FF_SUCCESS;
}

// Return the plan's scratch count in uint32 words, without modifying the plan
FastCDFFStatus fastcd_ff_cd_workspace_size(const FastCDPlanFF* plan, size_t* words) {
    if (plan == NULL || words == NULL) return FASTCD_FF_NULL_POINTER;
    if (fastcd_ff_overlap(words, sizeof(*words), plan, sizeof(*plan)))
        return FASTCD_FF_OVERLAPPING_BUFFERS;
    *words = plan->workspace_words;
    return FASTCD_FF_SUCCESS;
}

// Check active byte spans; equal complete read-only inputs are permitted
FastCDFFStatus fastcd_ff_validate(void* out, const void* x, const void* y, size_t bytes,
        const void* plan, size_t plan_bytes, void* scratch, size_t need,
        size_t available) {
    if (out == NULL || x == NULL || y == NULL) return FASTCD_FF_NULL_POINTER;
    if (available < need) return FASTCD_FF_INSUFFICIENT_WORKSPACE;
    if (need != 0 && scratch == NULL) return FASTCD_FF_NULL_POINTER;
    if (fastcd_ff_overlap(out, bytes, x, bytes) ||
            fastcd_ff_overlap(out, bytes, y, bytes) ||
            (x != y && fastcd_ff_overlap(x, bytes, y, bytes)) ||
            fastcd_ff_overlap(scratch, need, out, bytes) ||
            fastcd_ff_overlap(scratch, need, x, bytes) ||
            fastcd_ff_overlap(scratch, need, y, bytes) ||
            fastcd_ff_overlap(out, bytes, plan, plan_bytes) ||
            fastcd_ff_overlap(scratch, need, plan, plan_bytes))
        return FASTCD_FF_OVERLAPPING_BUFFERS;
    return FASTCD_FF_SUCCESS;
}

// Return a static description, including a safe fallback for unknown statuses
const char* fastcd_ff_status_string(FastCDFFStatus status) {
    static const char* const names[] = {"success", "invalid level", "invalid field",
            "characteristic 2 is unsupported", "insufficient workspace",
            "overlapping buffers", "size overflow", "null pointer",
            "unsupported algorithm", "zero quadratic norm", "invalid basis index",
            "invalid stride"};
    return (unsigned)status < sizeof(names) / sizeof(names[0]) ? names[status]
                                                               : "unknown status";
}
