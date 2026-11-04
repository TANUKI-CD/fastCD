#ifndef FASTCD_FF_FIELD_INTERNAL_H
#define FASTCD_FF_FIELD_INTERNAL_H

#include "fastcd/finite.h"

/* All private scalar helpers require canonical residues and an initialized,
 * immutable field. Their bounds are shared by the fixed-degree schedules;
 * no helper allocates, performs primality checks, or selects a CD algorithm. */
#if defined(__GNUC__) || defined(__clang__)
#define FASTCD_FF_ALWAYS_INLINE __attribute__((always_inline))
#define FASTCD_FF_UNUSED __attribute__((unused))
#else
#define FASTCD_FF_ALWAYS_INLINE
#define FASTCD_FF_UNUSED
#endif

// Share the public inline reduction with every fixed-degree specialization.
static inline uint32_t fastcd_ff_reduce(uint64_t x, const FastCDField* field) {
    return fastcd_ff_detail_reduce64(x, field);
}

// For canonical inputs p < 2^31, a+b <= 2p-2 < 2^32.
static inline uint32_t fastcd_ff_add(uint32_t a, uint32_t b, const FastCDField* f) {
    uint32_t z = a + b;
    return z >= f->prime ? z - f->prime : z;
}

// Unsigned subtraction avoids any signed intermediate or underflow residue.
static inline uint32_t fastcd_ff_sub(uint32_t a, uint32_t b, const FastCDField* f) {
    uint32_t z = a - b;
    return z + (f->prime & (0u - (uint32_t)(a < b)));
}

// Widen before multiplication: (p-1)^2 fits in uint64 for p < 2^31
static inline uint32_t fastcd_ff_mul(uint32_t a, uint32_t b, const FastCDField* f) {
    return fastcd_ff_detail_reduce64((uint64_t)a * b, f);
}

#include "generated/field_products.h"
#endif
