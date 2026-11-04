#ifndef FASTCD_FF_FINITE_MAPS_H
#define FASTCD_FF_FINITE_MAPS_H
#include <string.h>
#include "finite_backend.h"

/* Universal convolution maps precede reduction by the user's field modulus.
 * Rows have 64 bytes, allowing two complete packed loads at every rank;
 * zero padding avoids masked reads when the active tensor rank is smaller. */
typedef struct {
    uint8_t rank, total, reserved[62];
    uint8_t input[12][64];
    uint8_t output[25][64];
} CdlSmallMap;

FASTCD_FF_PRIVATE extern const CdlSmallMap fastcd_ff_small_maps[3][12];
FASTCD_FF_PRIVATE extern const size_t fastcd_ff_linear_offset[12];
FASTCD_FF_PRIVATE extern const uint64_t fastcd_ff_linear_low[2300];
FASTCD_FF_PRIVATE extern const uint16_t fastcd_ff_linear_high[2300];

// Compute t^k modulo the actual field polynomial to project universal CRT maps
static inline __attribute__((unused)) void fastcd_ff_small_remainders(
        uint32_t r[25][12], const FastCDField* f, size_t total, uint32_t prime) {
    memset(r, 0, 25 * 12 * sizeof(uint32_t));
    size_t d = f->degree;
    for (size_t k = 0; k < d; k++) r[k][k] = 1;
    for (size_t k = d; k < total; k++) {
        uint32_t carry = r[k - 1][d - 1];
        for (size_t j = 0; j < d; j++)
            r[k][j] = ((j ? r[k - 1][j - 1] : 0) + carry * f->negative[j]) % prime;
    }
}
#endif
