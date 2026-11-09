#ifndef FASTCD_FF_FINITE_BACKEND_H
#define FASTCD_FF_FINITE_BACKEND_H
// Private interface between finite-field dispatch and the packed backends
#include "fastcd/finite.h"
// Attributes for forced inlining and for template functions a unit may not use
#if defined(__GNUC__) || defined(__clang__)
#define FASTCD_FF_ALWAYS_INLINE __attribute__((always_inline))
#define FASTCD_FF_UNUSED __attribute__((unused))
#else
#define FASTCD_FF_ALWAYS_INLINE
#define FASTCD_FF_UNUSED
#endif
/* Optimized builds inline the small vector helpers of the packed backends on
 * their own. Unoptimized builds would call them, passing vector structs by
 * value on the stack, which crashed a GCC 15 AddressSanitizer build, so there
 * the helpers are forced inline. */
#if (defined(__GNUC__) || defined(__clang__)) && !defined(__OPTIMIZE__)
#define FASTCD_FF_HELPER_INLINE __attribute__((always_inline))
#else
#define FASTCD_FF_HELPER_INLINE
#endif

/* Packed backends. Public buffers hold canonical polynomial-basis uint32_t
 * coefficients and need only 4-byte alignment. */
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

// Arithmetic schedule a backend evaluates: the CD root, the private loop
// component of the shared templates, or the defining direct recursion
typedef enum {
    FASTCD_FF_SCHEDULE_CD,
    FASTCD_FF_SCHEDULE_LOOP,
    FASTCD_FF_SCHEDULE_DIRECT
} FastCDFFSchedule;

/* Smallest dimension any packed backend serves. Below it, direct products skip
 * backend selection and run the scalar recursion. */
#define FASTCD_FF_PACKED_MIN_DIMENSION 16

// Hide private backend symbols from the shared library
#if defined(__GNUC__) || defined(__clang__)
#define FASTCD_FF_PRIVATE __attribute__((visibility("hidden")))
#else
#define FASTCD_FF_PRIVATE
#endif

// Choose a backend from field parameters, dimension and CPU features, and report
// its scratch in uint32 words
FASTCD_FF_PRIVATE FastCDFFBackend fastcd_ff_finite_select(
        const FastCDPlanFF*, int, size_t*);
// Execute selected arithmetic after validation; zero requests the buffer fallback
FASTCD_FF_PRIVATE int fastcd_ff_finite_execute(uint32_t*, const uint32_t*,
        const uint32_t*, const FastCDPlanFF*, uint32_t*, FastCDFFSchedule);

// Choose the AUTO algorithm over C and finite fields: direct through
// dimension 4, quasilinear above it
static inline FastCDAlgorithm fastcd_domain_auto_algorithm(size_t dimension) {
    return dimension <= 4 ? FASTCD_ALGORITHM_DIRECT : FASTCD_ALGORITHM_QUASILINEAR;
}

// Run the register-value schedule with ordinary canonical prime residues
FASTCD_FF_PRIVATE void fastcd_ff_finite_prime_scalar(uint32_t*, const uint32_t*,
        const uint32_t*, const FastCDPlanFF*, uint32_t*, int);

/* Packed extension entries share one rank-channel interface. The matching input
 * and output matrices are per-call evaluation and projection setup, never public
 * coefficient representations or pointers kept after the call. */
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
// Prime-field and byte-encoded packed entries
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
// Reduce the universal small-prime CRT maps into the field's polynomial basis
FASTCD_FF_PRIVATE size_t fastcd_ff_prepare_small(
        uint32_t[48][12], uint32_t[48][12], const FastCDField*);
// Prepare linear-point Lagrange reconstruction modulo the field polynomial
FASTCD_FF_PRIVATE void fastcd_ff_prepare_linear(uint32_t[48][12], const FastCDField*);
// Encoded Montgomery point-array entries, used only inside their matching unit
#define FASTCD_FF_FINITE_POINTS(L)                                                     \
    FASTCD_FF_PRIVATE void fastcd_ff_points##L##_cd_encoded(uint32_t*,                 \
            const uint32_t*, const uint32_t*, const FastCDPlanFF*, uint32_t*, size_t,  \
            int);                                                                      \
    FASTCD_FF_PRIVATE void fastcd_ff_points##L##_finish_encoded(                       \
            uint32_t*, const FastCDPlanFF*, uint32_t*, size_t)
FASTCD_FF_FINITE_POINTS(4);
FASTCD_FF_FINITE_POINTS(8);
FASTCD_FF_FINITE_POINTS(16);
#undef FASTCD_FF_FINITE_POINTS
#undef FASTCD_FF_FINITE_DECLARE
#endif
