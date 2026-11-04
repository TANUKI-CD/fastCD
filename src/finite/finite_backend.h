#ifndef FASTCD_FF_FINITE_BACKEND_H
#define FASTCD_FF_FINITE_BACKEND_H
#include "fastcd/finite.h"
#if defined(__GNUC__) || defined(__clang__)
#define FASTCD_FF_ALWAYS_INLINE __attribute__((always_inline))
#define FASTCD_FF_UNUSED __attribute__((unused))
#else
#define FASTCD_FF_ALWAYS_INLINE
#define FASTCD_FF_UNUSED
#endif

/* This interface is private to the library. Public buffers always contain
 * canonical polynomial-basis uint32_t coefficients and need only 4B alignment. */
typedef enum {
    FASTCD_FF_FINITE_GENERIC,
    FASTCD_FF_FINITE_BYTE_PRIME,
    FASTCD_FF_FINITE_BYTE_QUADRATIC,
    FASTCD_FF_FINITE_TERNARY16,
    FASTCD_FF_FINITE_TERNARY32,
    FASTCD_FF_FINITE_TERNARY64,
    FASTCD_FF_FINITE_MEDIUM_BAR,
    FASTCD_FF_FINITE_MEDIUM_MONT,
    FASTCD_FF_FINITE_MEDIUM_BAR_PRIME,
    FASTCD_FF_FINITE_SHORT_PRIME,
    FASTCD_FF_FINITE_MEDIUM_MONT_PRIME,
    FASTCD_FF_FINITE_MONT4,
    FASTCD_FF_FINITE_MONT8,
    FASTCD_FF_FINITE_MONT16,
    FASTCD_FF_FINITE_EXT4,
    FASTCD_FF_FINITE_EXT8,
    FASTCD_FF_FINITE_EXT16,
    FASTCD_FF_FINITE_SMALL8,
    FASTCD_FF_FINITE_SMALL16,
    FASTCD_FF_FINITE_SMALL32,
    FASTCD_FF_FINITE_SMALL_PRIME32,
    FASTCD_FF_FINITE_SMALL_PLAIN32,
    FASTCD_FF_FINITE_SHORT8,
    FASTCD_FF_FINITE_SHORT16,
    FASTCD_FF_FINITE_SHORT32
} FastCDFFBackend;

#if defined(__GNUC__) || defined(__clang__)
#define FASTCD_FF_PRIVATE __attribute__((visibility("hidden")))
#else
#define FASTCD_FF_PRIVATE
#endif

// Choose a backend using only field parameters, dimension and CPU features;
// report scratch in uint32 words. Public CD calls always pass loop=0.
FASTCD_FF_PRIVATE FastCDFFBackend fastcd_ff_finite_select(
        const FastCDPlanFF*, int, size_t*);
// Execute selected arithmetic after validation; zero requests the buffer fallback
FASTCD_FF_PRIVATE int fastcd_ff_finite_execute(uint32_t*, const uint32_t*,
        const uint32_t*, const FastCDPlanFF*, uint32_t*, int);
// Run the register-value schedule with ordinary canonical prime residues
FASTCD_FF_PRIVATE void fastcd_ff_finite_prime_scalar(uint32_t*, const uint32_t*,
        const uint32_t*, const FastCDPlanFF*, uint32_t*, int);

// Packed extension entries share a rank-channel interface. Matching input
// and output matrices describe evaluation and projection; they are per-call
// setup values, never public coefficient representations or retained pointers.
typedef void (*FastCDFFProduct)(uint32_t*, const uint32_t*, const uint32_t*,
        const FastCDPlanFF*, uint32_t*, int, size_t, const uint32_t[48][12],
        const uint32_t[48][12]);
#define FASTCD_FF_FINITE_DECLARE(name)                                                 \
    FASTCD_FF_PRIVATE void name(uint32_t*, const uint32_t*, const uint32_t*,           \
            const FastCDPlanFF*, uint32_t*, int, size_t, const uint32_t[48][12],       \
            const uint32_t[48][12])
FASTCD_FF_FINITE_DECLARE(fastcd_ff_mont4_prime_tensor);
FASTCD_FF_FINITE_DECLARE(fastcd_ff_mont8_prime_tensor);
FASTCD_FF_FINITE_DECLARE(fastcd_ff_mont16_prime_tensor);
FASTCD_FF_FINITE_DECLARE(fastcd_ff_extension4_tensor);
FASTCD_FF_FINITE_DECLARE(fastcd_ff_extension8_tensor);
FASTCD_FF_FINITE_DECLARE(fastcd_ff_extension16_tensor);
FASTCD_FF_FINITE_DECLARE(fastcd_ff_small8_tensor);
FASTCD_FF_FINITE_DECLARE(fastcd_ff_small16_tensor);
FASTCD_FF_FINITE_DECLARE(fastcd_ff_small32_tensor);
FASTCD_FF_FINITE_DECLARE(fastcd_ff_ternary16_tensor);
FASTCD_FF_FINITE_DECLARE(fastcd_ff_ternary32_tensor);
FASTCD_FF_FINITE_DECLARE(fastcd_ff_ternary64_tensor);
FASTCD_FF_FINITE_DECLARE(fastcd_ff_medium_bar_tensor);
FASTCD_FF_FINITE_DECLARE(fastcd_ff_medium_mont_tensor);
FASTCD_FF_FINITE_DECLARE(fastcd_ff_small32_plain_tensor);
FASTCD_FF_FINITE_DECLARE(fastcd_ff_short8_tensor);
FASTCD_FF_FINITE_DECLARE(fastcd_ff_short16_tensor);
FASTCD_FF_FINITE_DECLARE(fastcd_ff_short32_tensor);
FASTCD_FF_PRIVATE void fastcd_ff_small32_prime(uint32_t*, const uint32_t*,
        const uint32_t*, const FastCDPlanFF*, uint32_t*, int);
FASTCD_FF_PRIVATE void fastcd_ff_short32_prime(uint32_t*, const uint32_t*,
        const uint32_t*, const FastCDPlanFF*, uint32_t*, int);
FASTCD_FF_PRIVATE void fastcd_ff_medium_mont_prime(uint32_t*, const uint32_t*,
        const uint32_t*, const FastCDPlanFF*, uint32_t*, int);
FASTCD_FF_PRIVATE void fastcd_ff_medium_bar_prime(uint32_t*, const uint32_t*,
        const uint32_t*, const FastCDPlanFF*, uint32_t*, int);
FASTCD_FF_PRIVATE void fastcd_ff_byte64_prime(uint32_t*, const uint32_t*,
        const uint32_t*, const FastCDPlanFF*, uint32_t*, int);
FASTCD_FF_PRIVATE void fastcd_ff_byte64_quadratic(uint32_t*, const uint32_t*,
        const uint32_t*, const FastCDPlanFF*, uint32_t*, int);
// Reduce universal small-prime CRT maps into the user's actual field basis
FASTCD_FF_PRIVATE size_t fastcd_ff_prepare_small(
        uint32_t[48][12], uint32_t[48][12], const FastCDField*);
// Prepare linear-point Lagrange reconstruction modulo the user's polynomial
FASTCD_FF_PRIVATE void fastcd_ff_prepare_linear(uint32_t[48][12], const FastCDField*);
/* Encoded, Montgomery point arrays; used only inside their matching TU. */
#define FASTCD_FF_FINITE_POINTS(L)                                                     \
    FASTCD_FF_PRIVATE void fastcd_ff_points##L##_cd_encoded(uint32_t*,                 \
            const uint32_t*, const uint32_t*, const FastCDPlanFF*, uint32_t*, size_t); \
    FASTCD_FF_PRIVATE void fastcd_ff_points##L##_finish_encoded(                       \
            uint32_t*, const FastCDPlanFF*, uint32_t*, size_t)
FASTCD_FF_FINITE_POINTS(4);
FASTCD_FF_FINITE_POINTS(8);
FASTCD_FF_FINITE_POINTS(16);
#undef FASTCD_FF_FINITE_POINTS
#undef FASTCD_FF_FINITE_DECLARE
#endif
