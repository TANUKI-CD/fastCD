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
 * Standard Cayley--Dickson algebras over odd-characteristic finite fields.
 * At level n >= 0, N = 2^n and an element has N coefficients. An extension
 * coefficient occupies d consecutive canonical uint32_t residues, least
 * polynomial power first. Lengths and strides count words, never bytes.
 *
 * Plans and field contexts are caller-owned values. After initialization,
 * treat them as immutable; separate workspaces allow concurrent products.
 * No arithmetic routine allocates memory or retains caller buffer pointers.
 * Unless explicitly permitted, writable storage must be disjoint from every
 * active input, context, plan, and workspace. Read-only operands may be the
 * same complete array; other input overlaps are unsupported. Every status
 * error leaves outputs unchanged. Operands must have the documented lengths.
 * Initialized, unmodified plans/contexts and correctly aligned buffers are
 * preconditions; arithmetic does not repeatedly validate field data/residues.
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

// Validate prime 2..INT32_MAX and a canonical monic irreducible modulus of
// degree 1..12. modulus has degree+1 entries; it may be NULL for degree 1
// (the canonical modulus t is used). Setup needs no external arithmetic library.
FASTCD_API FastCDFFStatus fastcd_ff_field_init(
        FastCDField* field, uint32_t prime, uint32_t degree, const uint32_t* modulus);

// Same shape/range checks, but caller has separately established primality
// and irreducibility. Never use with unvalidated mathematical parameters.
FASTCD_API FastCDFFStatus fastcd_ff_field_init_trusted(
        FastCDField* field, uint32_t prime, uint32_t degree, const uint32_t* modulus);

// Scalar operations take exactly degree canonical words. out may equal x
// or y exactly. No other overlap is supported; no runtime checks are made.
FASTCD_API void fastcd_ff_field_add(
        uint32_t* out, const uint32_t* x, const uint32_t* y, const FastCDField* field);
FASTCD_API void fastcd_ff_field_sub(
        uint32_t* out, const uint32_t* x, const uint32_t* y, const FastCDField* field);

/*
 * Tiny prime operations and the degree-one multiplication path are inline in
 * ordinary clients. The extension fallback
 * stays out of line so its larger stack frame cannot burden a prime product.
 * The library also exports out-of-line symbols for C clients.
 * Define FASTCD_FF_FIELD_EXTERN_ONLY before including this header to use those
 * symbols explicitly. FASTCD_FF_FIELD_INLINES_C is reserved for the library emitter.
 */

// Implementation entry for the inline front end; applications should call
// fastcd_ff_field_mul. Requires degree 2..12 and the scalar aliasing contract above.
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

// Share one small body with header clients and the ABI emitter. Keeping this
// helper local also avoids per-coefficient PLT calls in shared-library builds.
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

// Return canonical x modulo field->prime for every uint64_t x. The reciprocal
// quotient underestimates floor(x/p) by at most one, so one correction suffices.
FASTCD_FF_FIELD_INLINE uint32_t fastcd_ff_field_reduce64(
        uint64_t x, const FastCDField* field) {
    return fastcd_ff_detail_reduce64(x, field);
}

// Multiply canonical prime residues, returning the result by value. An
// extension context may also supply its base prime; this is not a product of
// extension elements. Context and residue validity remain caller preconditions.
FASTCD_FF_FIELD_INLINE uint32_t fastcd_ff_prime_mul(
        uint32_t a, uint32_t b, const FastCDField* field) {
    return fastcd_ff_detail_reduce64((uint64_t)a * b, field);
}

// Multiply complete field elements. Exact in-place aliases are supported.
// No allocation, repeated validation, or hidden scalar dispatch is performed
// on the degree-one path. Other degrees use the shared external fallback.
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

// Precompute dimensions, workspace lengths and normalization scalars. Reject
// characteristic 2; level zero is the coefficient field itself.
FASTCD_API FastCDFFStatus fastcd_ff_plan_init(
        FastCDPlanFF* plan, const FastCDField* field, size_t level);

// Query caller-provided scratch in uint32_t words, including direct scratch.
FASTCD_API FastCDFFStatus fastcd_ff_cd_workspace_size(
        const FastCDPlanFF* plan, size_t* words);

// Exact quasilinear CD product on N*d words, with the same basis convention
// as fastCD. This exposes the signed-component kernel for independent tests.
FASTCD_API FastCDFFStatus fastcd_ff_cd_mul(uint32_t* out, const uint32_t* x,
        const uint32_t* y, const FastCDPlanFF* plan, uint32_t* workspace,
        size_t workspace_words);

// Return a static status description
FASTCD_API const char* fastcd_ff_status_string(FastCDFFStatus status);

/* ---------- General coefficient and CD operations ---------- */

// Validate an odd prime and explicit modulus, or choose a deterministic
// monic irreducible modulus when NULL. No setup operation allocates memory.
FASTCD_API FastCDFFStatus fastcd_field_init(
        FastCDField* field, uint32_t prime, uint32_t degree, const uint32_t* modulus);

// Invert or negate one degree-word scalar. Exact out==x aliases are safe;
// inversion rejects zero before writing output, with FASTCD_FF_ZERO_NORM.
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

// Apply one record operation; reductions write one degree-word coefficient.
// Exact out==x aliases are allowed for coordinatewise operations, reciprocal,
// and square; addition/subtraction also accept out==y. Other outputs disjoint.
FASTCD_API FastCDFFStatus fastcd_apply_ff(uint32_t* out, const uint32_t* x,
        const uint32_t* y, const uint32_t* z, const FastCDPlanFF* plan,
        FastCDOperation operation, size_t basis_index, FastCDAlgorithm algorithm,
        uint32_t* workspace, size_t workspace_words);

// Return Q(x)=sum_i x_i^2 in the same field, not the relative norm to F_p
FASTCD_API FastCDFFStatus fastcd_quadratic_norm_ff(
        uint32_t* value, const uint32_t* x, const FastCDPlanFF* plan);

// Multiply count records. Strides count uint32 words; zero input stride
// broadcasts one complete record. Validate all spans before writing any output.
FASTCD_API FastCDFFStatus fastcd_mul_batch_ff(uint32_t* out, size_t out_stride,
        const uint32_t* x, size_t x_stride, const uint32_t* y, size_t y_stride,
        size_t count, const FastCDPlanFF* plan, FastCDAlgorithm algorithm,
        uint32_t* workspace, size_t workspace_words);

#ifdef __cplusplus
}
#endif
#endif
