#ifndef FASTCD_FF_H
#define FASTCD_FF_H

#include "arithmetic.h"
#include "fastcd.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Standard Cayley--Dickson algebras over odd-characteristic finite fields. At
 * level n >= 0, N = 2^n and an element has N coefficients. An extension
 * coefficient occupies d consecutive canonical uint32_t residues, lowest
 * polynomial power first. Lengths and strides count words, never bytes.
 *
 * Plans and field contexts are caller-owned values, immutable after
 * initialization, and separate workspaces allow concurrent products. No
 * arithmetic routine allocates memory or keeps caller buffer pointers. Unless
 * a declaration permits an alias, writable storage must be disjoint from every
 * active input, context, plan and workspace. Read-only operands may be the
 * same complete array; checked operations reject other input overlaps, and
 * the unchecked field operations do not support them. Every status error
 * leaves the outputs unchanged. Operands must have the documented lengths.
 * Initialized, unmodified plans and contexts, aligned buffers and canonical
 * residues are preconditions that arithmetic does not check again.
 */

/* ---------- Definitions of enums and structs ---------- */

typedef enum {
    FASTCD_FF_SUCCESS = 0,
    FASTCD_FF_INVALID_LEVEL,
    FASTCD_FF_INVALID_FIELD,
    FASTCD_FF_CHARACTERISTIC_TWO,
    FASTCD_FF_INSUFFICIENT_WORKSPACE,
    FASTCD_FF_OVERLAPPING_BUFFERS,
    FASTCD_FF_SIZE_OVERFLOW,
    FASTCD_FF_NULL_POINTER,
    FASTCD_FF_UNSUPPORTED_ALGORITHM,
    FASTCD_FF_ZERO_NORM,
    FASTCD_FF_INVALID_BASIS_INDEX,
    FASTCD_FF_INVALID_STRIDE
} FastCDFFStatus;

#define FASTCD_FF_MAX_DEGREE 12
#define FASTCD_FF_HAVE_INLINE_FIELD_API 1

typedef struct FastCDField {
    uint32_t prime;
    uint32_t degree;
    uint32_t modulus[FASTCD_FF_MAX_DEGREE + 1];
    uint32_t negative[FASTCD_FF_MAX_DEGREE];
    uint64_t reciprocal;
} FastCDField;

typedef struct FastCDPlanFF {
    FastCDField field;
    size_t level;
    size_t dimension;
    size_t words; // CD element: dimension * degree
    size_t workspace_words;
    uint32_t inverse_two;
    uint32_t inverse_dimension;
} FastCDPlanFF;

/* ---------- Field construction and scalar arithmetic ---------- */

/* modulus has degree+1 entries and may be NULL for degree 1, which uses the
 * canonical modulus t. Setup needs no external arithmetic library. */
// Validate a prime 2..INT32_MAX and a canonical monic irreducible modulus of
// degree 1..12
FASTCD_API FastCDFFStatus fastcd_ff_field_init(
        FastCDField* field, uint32_t prime, uint32_t degree, const uint32_t* modulus);

/* Never use this with unvalidated mathematical parameters. */
// Run the same shape and range checks for a caller that has already
// established primality and irreducibility
FASTCD_API FastCDFFStatus fastcd_ff_field_init_trusted(
        FastCDField* field, uint32_t prime, uint32_t degree, const uint32_t* modulus);

/* out may equal x or y exactly. No other overlap is supported, and nothing is
 * checked at run time. */
// Add or subtract scalars of exactly degree canonical words
FASTCD_API void fastcd_ff_field_add(
        uint32_t* out, const uint32_t* x, const uint32_t* y, const FastCDField* field);
FASTCD_API void fastcd_ff_field_sub(
        uint32_t* out, const uint32_t* x, const uint32_t* y, const FastCDField* field);

/*
 * Small prime operations and the degree-one multiplication path are inline in
 * ordinary clients. The extension fallback stays out of line so that its
 * larger stack frame cannot burden a prime product. The library also exports
 * out-of-line symbols for C clients; define FASTCD_FF_FIELD_EXTERN_ONLY before
 * including this header to use those symbols explicitly.
 * FASTCD_FF_FIELD_INLINES_C is reserved for the library's symbol emitter.
 */

/* Applications call fastcd_ff_field_mul. This entry needs degree 2..12 and the
 * scalar aliasing contract above. */
// Multiply extension-field scalars for the inline fastcd_ff_field_mul
FASTCD_API void fastcd_ff_field_mul_extension(
        uint32_t* out, const uint32_t* x, const uint32_t* y, const FastCDField* field);

#if defined(FASTCD_FF_FIELD_EXTERN_ONLY)
FASTCD_API uint32_t fastcd_ff_field_reduce64(uint64_t x, const FastCDField* field);
FASTCD_API uint32_t fastcd_ff_prime_mul(
        uint32_t a, uint32_t b, const FastCDField* field);
FASTCD_API void fastcd_ff_field_mul(
        uint32_t* out, const uint32_t* x, const uint32_t* y, const FastCDField* field);
#else
#if defined(FASTCD_FF_FIELD_INLINES_C)
#define FASTCD_FF_FIELD_INLINE FASTCD_API
#else
#define FASTCD_FF_FIELD_INLINE static inline
#endif

/* Keeping this local avoids per-coefficient PLT calls in shared-library
 * builds. */
// Reduce a 64-bit value modulo the prime, shared by header clients and the
// exported symbols
static inline uint32_t fastcd_ff_detail_reduce64(uint64_t x, const FastCDField* field) {
#if defined(__SIZEOF_INT128__)
    uint64_t q = (uint64_t)((__uint128_t)x * field->reciprocal >> 64);
#else
    const uint64_t mask = UINT64_C(0xffffffff);
    uint64_t a = x >> 32, b = x & mask;
    uint64_t c = field->reciprocal >> 32, d = field->reciprocal & mask;
    uint64_t t = b * d, u = a * d + (t >> 32), v = b * c + (u & mask);
    uint64_t q = a * c + (u >> 32) + (v >> 32);
#endif
    uint64_t r = x - q * field->prime;
    return (uint32_t)(r >= field->prime ? r - field->prime : r);
}

/* The reciprocal quotient underestimates floor(x/p) by at most one, so one
 * correction suffices. */
// Return canonical x modulo field->prime for every uint64_t x
FASTCD_FF_FIELD_INLINE uint32_t fastcd_ff_field_reduce64(
        uint64_t x, const FastCDField* field) {
    return fastcd_ff_detail_reduce64(x, field);
}

/* An extension context may supply its base prime, but this is not a product
 * of extension elements. Context and residue validity are caller
 * preconditions. */
// Multiply canonical prime residues and return the result by value
FASTCD_FF_FIELD_INLINE uint32_t fastcd_ff_prime_mul(
        uint32_t a, uint32_t b, const FastCDField* field) {
    return fastcd_ff_detail_reduce64((uint64_t)a * b, field);
}

/* The degree-one path allocates nothing, validates nothing again and has no
 * hidden scalar dispatch. Other degrees use the shared out-of-line fallback. */
// Multiply complete field elements, with exact in-place aliases supported
FASTCD_FF_FIELD_INLINE void fastcd_ff_field_mul(
        uint32_t* out, const uint32_t* x, const uint32_t* y, const FastCDField* field) {
    if (field->degree == 1) {
        out[0] = fastcd_ff_detail_reduce64((uint64_t)x[0] * y[0], field);
        return;
    }
    fastcd_ff_field_mul_extension(out, x, y, field);
}

#undef FASTCD_FF_FIELD_INLINE
#endif

/* ---------- Finite-field plans and products ---------- */

/* Characteristic 2 is rejected, and level zero is the coefficient field
 * itself. */
// Precompute dimensions, workspace lengths and normalization scalars
FASTCD_API FastCDFFStatus fastcd_ff_plan_init(
        FastCDPlanFF* plan, const FastCDField* field, size_t level);

// Query caller-provided scratch in uint32_t words, covering both algorithms
FASTCD_API FastCDFFStatus fastcd_ff_cd_workspace_size(
        const FastCDPlanFF* plan, size_t* words);

// Multiply N*d-word records with the quasilinear schedule in the standard
// basis convention, whatever AUTO would select
FASTCD_API FastCDFFStatus fastcd_ff_cd_mul(uint32_t* out, const uint32_t* x,
        const uint32_t* y, const FastCDPlanFF* plan, uint32_t* workspace,
        size_t workspace_words);

// Return a static status description
FASTCD_API const char* fastcd_ff_status_string(FastCDFFStatus status);

/* ---------- General coefficient and CD operations ---------- */

/* No setup operation allocates memory. */
// Validate an odd prime and explicit modulus, or choose a deterministic
// monic irreducible modulus when modulus is NULL
FASTCD_API FastCDFFStatus fastcd_field_init(
        FastCDField* field, uint32_t prime, uint32_t degree, const uint32_t* modulus);

/* Inversion rejects zero with FASTCD_FF_ZERO_NORM before writing the
 * output. */
// Invert or negate one degree-word scalar, with exact out == x aliases safe
FASTCD_API FastCDFFStatus fastcd_field_inv(
        uint32_t* out, const uint32_t* x, const FastCDField* field);
FASTCD_API void fastcd_field_neg(
        uint32_t* out, const uint32_t* x, const FastCDField* field);

// Multiply one N*d-word record using AUTO, DIRECT, or QUASILINEAR
FASTCD_API FastCDFFStatus fastcd_mul_with_algorithm_ff(uint32_t* out, const uint32_t* x,
        const uint32_t* y, const FastCDPlanFF* plan, FastCDAlgorithm algorithm,
        uint32_t* workspace, size_t workspace_words);

// Multiply using the finite default, independently of the real AUTO tables
FASTCD_API FastCDFFStatus fastcd_mul_ff(uint32_t* out, const uint32_t* x,
        const uint32_t* y, const FastCDPlanFF* plan, uint32_t* workspace,
        size_t workspace_words);

// Query scratch for a record operation, in uint32 words, including composites
FASTCD_API FastCDFFStatus fastcd_apply_workspace_size_ff(
        const FastCDPlanFF* plan, FastCDOperation operation, size_t* workspace_words);

/* Coordinatewise operations, reciprocal and square allow an exact out == x
 * alias, and addition and subtraction also allow out == y. Every other output
 * must be disjoint from the inputs. */
// Apply one record operation, where reductions write one degree-word coefficient
FASTCD_API FastCDFFStatus fastcd_apply_ff(uint32_t* out, const uint32_t* x,
        const uint32_t* y, const uint32_t* z, const FastCDPlanFF* plan,
        FastCDOperation operation, size_t basis_index, FastCDAlgorithm algorithm,
        uint32_t* workspace, size_t workspace_words);

// Return Q(x)=sum_i x_i^2 in the same field, not the relative norm to F_p
FASTCD_API FastCDFFStatus fastcd_quadratic_norm_ff(
        uint32_t* value, const uint32_t* x, const FastCDPlanFF* plan);

/* A zero input stride broadcasts one complete record. Every span is validated
 * before any output is written. */
// Multiply count records, with strides in uint32 words
FASTCD_API FastCDFFStatus fastcd_mul_batch_ff(uint32_t* out, size_t out_stride,
        const uint32_t* x, size_t x_stride, const uint32_t* y, size_t y_stride,
        size_t count, const FastCDPlanFF* plan, FastCDAlgorithm algorithm,
        uint32_t* workspace, size_t workspace_words);

#ifdef __cplusplus
}
#endif
#endif
