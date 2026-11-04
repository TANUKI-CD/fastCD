/* Exercise the public inline reducer with the portable 32-bit-limb path,
 * even on compilers that also have a 128-bit integer extension. */
#include "fastcd/finite.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    const uint32_t primes[] = {3, 5, 13, 257, 65521, 2147483647};
    for (size_t i = 0; i < sizeof(primes) / sizeof(primes[0]); i++) {
        FastCDField field;
        if (fastcd_field_init(&field, primes[i], 1, NULL)) return EXIT_FAILURE;
        uint64_t state = UINT64_C(0xdeadbeef12345678);
        for (size_t k = 0; k < 100000; k++) {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            uint64_t value = state * UINT64_C(2685821657736338717);
            if (k < 4) value = UINT64_MAX - k;
            if (fastcd_ff_field_reduce64(value, &field) != value % primes[i]) {
                fprintf(stderr,
                        "portable reduction failed: %" PRIu64 " mod %" PRIu32 "\n",
                        value, primes[i]);
                return EXIT_FAILURE;
            }
        }
    }
    return EXIT_SUCCESS;
}
