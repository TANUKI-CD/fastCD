#ifndef FASTCD_FF_INTERNAL_H
#define FASTCD_FF_INTERNAL_H
#include <stdint.h>
#include "fastcd/finite.h"

/* Private record interfaces, with lengths in bytes only at the validation
 * boundary. Arithmetic assumes initialized plans and canonical residues. */

// Report whether two byte ranges overlap, comparing addresses as integers
static inline int fastcd_ff_overlap(
        const void* a, size_t an, const void* b, size_t bn) {
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    if (an == 0 || bn == 0) return 0;
    return x <= y ? y - x < an : x - y < bn;
}

// Validate writable output/scratch against complete inputs and the copied plan
FastCDFFStatus fastcd_ff_validate(void* out, const void* x, const void* y, size_t bytes,
        const void* plan, size_t plan_bytes, void* scratch, size_t need,
        size_t available);
// Evaluate the degree-specialized defining recursion after validation
void fastcd_ff_direct(
        uint32_t*, const uint32_t*, const uint32_t*, const FastCDPlanFF*, uint32_t*);
// Select arithmetic for a validated single record or a record within a batch
void fastcd_ff_mul_unchecked(uint32_t*, const uint32_t*, const uint32_t*,
        const FastCDPlanFF*, FastCDAlgorithm, uint32_t*);
#endif
