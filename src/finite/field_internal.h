#ifndef FASTCD_FF_FIELD_INTERNAL_H
#define FASTCD_FF_FIELD_INTERNAL_H

#include "fastcd/finite.h"

/* Private scalar helpers over canonical residues of an initialized, immutable
 * field. The fixed-degree schedules rely on their bounds. No helper allocates,
 * tests primality or selects a CD algorithm. */
// Attributes for forced inlining and for template functions a unit may not use
#if defined(__GNUC__) || defined(__clang__)
#define FASTCD_FF_ALWAYS_INLINE __attribute__((always_inline))
#define FASTCD_FF_UNUSED __attribute__((unused))
#else
#define FASTCD_FF_ALWAYS_INLINE
#define FASTCD_FF_UNUSED
#endif

// Reduce a 64-bit value with the public inline reduction shared by every degree
static inline uint32_t fastcd_ff_reduce(uint64_t x, const FastCDField* field) {
    return fastcd_ff_detail_reduce64(x, field);
}

// Add canonical residues; a+b <= 2p-2 < 2^32 for p < 2^31
static inline uint32_t fastcd_ff_add(uint32_t a, uint32_t b, const FastCDField* f) {
    uint32_t z = a + b;
    return z >= f->prime ? z - f->prime : z;
}

// Subtract canonical residues in unsigned arithmetic with a masked correction
static inline uint32_t fastcd_ff_sub(uint32_t a, uint32_t b, const FastCDField* f) {
    uint32_t z = a - b;
    return z + (f->prime & (0u - (uint32_t)(a < b)));
}

// Multiply canonical residues; (p-1)^2 fits in uint64 for p < 2^31
static inline uint32_t fastcd_ff_mul(uint32_t a, uint32_t b, const FastCDField* f) {
    return fastcd_ff_detail_reduce64((uint64_t)a * b, f);
}

#include "generated/field_products.h"
#endif
