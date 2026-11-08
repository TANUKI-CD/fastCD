/*
 * Checked coefficient and Cayley--Dickson operations over F_(p^d).
 *
 * A field scalar occupies d canonical residues, constant coefficient first, and
 * a CD record occupies N*d words. Addition acts on residues, products and
 * inverses reduce in the context's polynomial basis, and every returned
 * coefficient is canonical.
 *
 * Multiplication schedules and CPU dispatch live in other units. This layer
 * selects algorithms, validates byte spans and scratch, and implements the
 * structural formulas without heap allocation. Contexts are immutable values,
 * and no operation keeps caller storage.
 *
 * CD conjugation fixes F_(p^d), so the bilinear pairing and the quadratic norm
 * are sum_i x_i*y_i and sum_i x_i^2 in the coefficient field itself, not the
 * relative field norm to F_p.
 */
#include "field_internal.h"
#include "internal.h"
#include "finite_backend.h"
#include <string.h>

// Validate an explicit presentation or choose a deterministic irreducible modulus
FastCDFFStatus fastcd_field_init(
        FastCDField* out, uint32_t p, uint32_t d, const uint32_t* modulus) {
    if (!out) return FASTCD_FF_NULL_POINTER;
    if (p == 2) return FASTCD_FF_CHARACTERISTIC_TWO;
    if (d < 1 || d > FASTCD_FF_MAX_DEGREE) return FASTCD_FF_INVALID_FIELD;
    if (modulus || d == 1) return fastcd_ff_field_init(out, p, d, modulus);
    FastCDField base;
    FastCDFFStatus status = fastcd_ff_field_init(&base, p, 1, NULL);
    if (status) return status;
    uint32_t coefficients[FASTCD_FF_MAX_DEGREE + 1] = {0};
    coefficients[d] = 1;
    /*
     * A fixed seed spreads the first candidates without global random state,
     * and after 4096 attempts an exhaustive base-p enumeration reaches every
     * candidate. Only setup pays this cost.
     */
    uint64_t state = UINT64_C(0x9e3779b97f4a7c15) ^ p ^ ((uint64_t)d << 32);
    for (size_t attempt = 0;; attempt++) {
        if (attempt < 4096) {
            for (size_t i = 0; i < d; i++) {
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                coefficients[i] =
                        (uint32_t)((state * UINT64_C(2685821657736338717)) % p);
            }
        } else {
            if (attempt == 4096) memset(coefficients, 0, d * sizeof(uint32_t));
            for (size_t i = 0; i < d; i++) {
                if (++coefficients[i] < p) break;
                coefficients[i] = 0;
            }
        }
        // A nontrivial irreducible polynomial has a nonzero constant term
        if (!coefficients[0]) continue;
        FastCDField candidate;
        status = fastcd_ff_field_init(&candidate, p, d, coefficients);
        if (status == FASTCD_FF_SUCCESS) {
            // Publish only a fully validated context, so a failure never leaves
            // a partially initialized field in the caller's storage
            *out = candidate;
            return status;
        }
        if (status != FASTCD_FF_INVALID_FIELD) return status;
    }
}

// Negate every canonical polynomial coordinate; exact in-place use is safe
void fastcd_field_neg(uint32_t* out, const uint32_t* x, const FastCDField* f) {
    for (size_t j = 0; j < f->degree; j++) out[j] = x[j] ? f->prime - x[j] : 0;
}

// Invert a nonzero scalar by x^(p^d-2), without narrowing the field order
FastCDFFStatus fastcd_field_inv(
        uint32_t* out, const uint32_t* x, const FastCDField* f) {
    if (!out || !x || !f) return FASTCD_FF_NULL_POINTER;
    uint32_t nonzero = 0;
    for (size_t j = 0; j < f->degree; j++) nonzero |= x[j];
    if (!nonzero) return FASTCD_FF_ZERO_NORM;
    // p^d-2 in little-endian 32-bit limbs, since the order may exceed uint64_t
    uint32_t exponent[13] = {1};
    size_t limbs = 1;
    for (size_t j = 0; j < f->degree; j++) {
        uint64_t carry = 0;
        for (size_t k = 0; k < limbs; k++) {
            uint64_t t = (uint64_t)exponent[k] * f->prime + carry;
            exponent[k] = (uint32_t)t;
            carry = t >> 32;
        }
        if (carry) exponent[limbs++] = (uint32_t)carry;
    }
    // Subtract two across limbs before scanning the little-endian exponent
    uint64_t borrow = 2;
    for (size_t k = 0; k < limbs && borrow; k++) {
        uint64_t old = exponent[k];
        exponent[k] = (uint32_t)(old - borrow);
        borrow = old < borrow;
    }
    // Products stay in stack temporaries, so out may equal x, and a zero input
    // was rejected before the caller's output is touched
    uint32_t result[12] = {1}, base[12] = {0};
    memcpy(base, x, f->degree * sizeof(uint32_t));
    for (size_t k = 0; k < limbs; k++) {
        uint32_t bits = exponent[k];
        for (unsigned bit = 0; bit < 32; bit++) {
            if (bits & 1) fastcd_ff_field_mul(result, result, base, f);
            bits >>= 1;
            fastcd_ff_field_mul(base, base, base, f);
        }
    }
    memcpy(out, result, f->degree * sizeof(uint32_t));
    return FASTCD_FF_SUCCESS;
}

/* ---------- Product selection and scratch ---------- */

// Validate and evaluate a direct or quasilinear product on one complete record
FastCDFFStatus fastcd_mul_with_algorithm_ff(uint32_t* out, const uint32_t* x,
        const uint32_t* y, const FastCDPlanFF* p, FastCDAlgorithm algorithm,
        uint32_t* work, size_t available) {
    if (!p) return FASTCD_FF_NULL_POINTER;
    // Resolve AUTO first, since the two algorithms validate on different paths
    if (algorithm == FASTCD_ALGORITHM_AUTO)
        algorithm = fastcd_ff_auto_algorithm(p->dimension);
    if (algorithm == FASTCD_ALGORITHM_QUASILINEAR)
        return fastcd_ff_cd_mul(out, x, y, p, work, available);
    if (algorithm != FASTCD_ALGORITHM_DIRECT) return FASTCD_FF_UNSUPPORTED_ALGORITHM;
    if (available < p->workspace_words) return FASTCD_FF_INSUFFICIENT_WORKSPACE;
    FastCDFFStatus s = fastcd_ff_validate(out, x, y, p->words * sizeof(uint32_t), p,
            sizeof(*p), work, p->workspace_words * sizeof(uint32_t),
            p->workspace_words * sizeof(uint32_t));
    if (s) return s;
    fastcd_ff_mul_unchecked(out, x, y, p, FASTCD_ALGORITHM_DIRECT, work);
    return FASTCD_FF_SUCCESS;
}

// Multiply with the finite-field AUTO choice; the real AUTO tables do not apply
FastCDFFStatus fastcd_mul_ff(uint32_t* out, const uint32_t* x, const uint32_t* y,
        const FastCDPlanFF* p, uint32_t* work, size_t length) {
    return fastcd_mul_with_algorithm_ff(
            out, x, y, p, FASTCD_ALGORITHM_AUTO, work, length);
}

// Query scratch in uint32 words; composite formulas hold two extra CD records
FastCDFFStatus fastcd_apply_workspace_size_ff(
        const FastCDPlanFF* p, FastCDOperation op, size_t* length) {
    if (!p || !length) return FASTCD_FF_NULL_POINTER;
    if (fastcd_ff_overlap(length, sizeof(*length), p, sizeof(*p)))
        return FASTCD_FF_OVERLAPPING_BUFFERS;
    if (op < FASTCD_OP_MULTIPLY || op > FASTCD_OP_RIGHT_OPERATOR_COMMUTATOR)
        return FASTCD_FF_UNSUPPORTED_ALGORITHM;
    size_t extra =
            (op == FASTCD_OP_ASSOCIATOR || op == FASTCD_OP_LEFT_OPERATOR_COMMUTATOR ||
                    op == FASTCD_OP_RIGHT_OPERATOR_COMMUTATOR)
            ? 2 * p->words
            : 0;
    size_t need = (op == FASTCD_OP_MULTIPLY || op == FASTCD_OP_IMAG_CROSS ||
                          op == FASTCD_OP_COMMUTATOR || extra)
            ? p->workspace_words
            : 0;
    size_t limit = SIZE_MAX / sizeof(uint32_t);
    if (extra > limit || need > limit - extra) return FASTCD_FF_SIZE_OVERFLOW;
    *length = extra + need;
    return FASTCD_FF_SUCCESS;
}

/* ---------- Bilinear and structural operations ---------- */

// Accumulate field products, reducing after each term; out is separate storage
static void dot(uint32_t* out, const uint32_t* x, const uint32_t* y, size_t n,
        const FastCDField* f) {
    uint32_t t[12];
    size_t d = f->degree;
    memset(out, 0, d * sizeof(uint32_t));
    for (size_t i = 0; i < n; i++) {
        fastcd_ff_field_mul(t, x + i * d, y + i * d, f);
        fastcd_ff_field_add(out, out, t, f);
    }
}

// Validate one operation before any output write, then evaluate its exact formula
FastCDFFStatus fastcd_apply_ff(uint32_t* out, const uint32_t* x, const uint32_t* y,
        const uint32_t* z, const FastCDPlanFF* p, FastCDOperation op, size_t index,
        FastCDAlgorithm algorithm, uint32_t* work, size_t available) {
    size_t need;
    FastCDFFStatus status = fastcd_apply_workspace_size_ff(p, op, &need);
    if (status) return status;
    if (algorithm != FASTCD_ALGORITHM_AUTO && algorithm != FASTCD_ALGORITHM_DIRECT &&
            algorithm != FASTCD_ALGORITHM_QUASILINEAR)
        return FASTCD_FF_UNSUPPORTED_ALGORITHM;
    if (!out || !x) return FASTCD_FF_NULL_POINTER;
    size_t d = p->field.degree, n = p->dimension, words = p->words,
           bytes = words * sizeof(uint32_t);
    int binary = op == FASTCD_OP_MULTIPLY || op == FASTCD_OP_ADD ||
            op == FASTCD_OP_SUBTRACT || op == FASTCD_OP_SCALE ||
            op == FASTCD_OP_DIVIDE || op == FASTCD_OP_INNER ||
            op == FASTCD_OP_IMAG_CROSS || op == FASTCD_OP_COMMUTATOR ||
            op == FASTCD_OP_ANTICOMMUTATOR || op == FASTCD_OP_ASSOCIATOR ||
            op == FASTCD_OP_LEFT_OPERATOR_COMMUTATOR ||
            op == FASTCD_OP_RIGHT_OPERATOR_COMMUTATOR;
    int ternary = op == FASTCD_OP_ASSOCIATOR ||
            op == FASTCD_OP_LEFT_OPERATOR_COMMUTATOR ||
            op == FASTCD_OP_RIGHT_OPERATOR_COMMUTATOR;
    if ((binary && !y) || (ternary && !z) || (need && !work))
        return FASTCD_FF_NULL_POINTER;
    if (available < need) return FASTCD_FF_INSUFFICIENT_WORKSPACE;
    if ((op == FASTCD_OP_BASIS_LEFT || op == FASTCD_OP_BASIS_RIGHT) && index >= n)
        return FASTCD_FF_INVALID_BASIS_INDEX;
    // Reductions write one field coefficient and scaling reads one on the right,
    // while every other active operand spans a complete N*d-word record
    size_t ob = (op == FASTCD_OP_INNER || op == FASTCD_OP_QUADRATIC_NORM)
            ? d * sizeof(uint32_t)
            : bytes;
    size_t yb = (op == FASTCD_OP_SCALE || op == FASTCD_OP_DIVIDE) ? d * sizeof(uint32_t)
                                                                  : bytes;
    /*
     * Allow exact aliases only for formulas that save their shared scalars
     * before storing. Partial aliases and writable plan or scratch aliases fail.
     */
    int alias = op == FASTCD_OP_ADD || op == FASTCD_OP_SUBTRACT ||
            op == FASTCD_OP_NEGATE || op == FASTCD_OP_SCALE || op == FASTCD_OP_DIVIDE ||
            op == FASTCD_OP_CONJUGATE || op == FASTCD_OP_IMAG_PART ||
            op == FASTCD_OP_RECIPROCAL || op == FASTCD_OP_SQUARE;
    if ((fastcd_ff_overlap(out, ob, x, bytes) && !(alias && out == x)) ||
            (binary && fastcd_ff_overlap(out, ob, y, yb) &&
                    !((op == FASTCD_OP_ADD || op == FASTCD_OP_SUBTRACT) && out == y)) ||
            (ternary && fastcd_ff_overlap(out, ob, z, bytes)) ||
            (binary && op != FASTCD_OP_SCALE && op != FASTCD_OP_DIVIDE && x != y &&
                    fastcd_ff_overlap(x, bytes, y, yb)) ||
            fastcd_ff_overlap(out, ob, p, sizeof(*p)) ||
            fastcd_ff_overlap(work, need * 4, p, sizeof(*p)) ||
            fastcd_ff_overlap(work, need * 4, out, ob) ||
            fastcd_ff_overlap(work, need * 4, x, bytes) ||
            (binary && fastcd_ff_overlap(work, need * 4, y, yb)) ||
            (ternary && fastcd_ff_overlap(work, need * 4, z, bytes)))
        return FASTCD_FF_OVERLAPPING_BUFFERS;
    const FastCDField* f = &p->field;
    uint32_t q[12], t[12], u[12], v[12], alpha[12], beta[12];
    if (op == FASTCD_OP_MULTIPLY)
        return fastcd_mul_with_algorithm_ff(out, x, y, p, algorithm, work, available);
    if (op == FASTCD_OP_ADD || op == FASTCD_OP_SUBTRACT) {
        for (size_t i = 0; i < words; i++)
            out[i] = op == FASTCD_OP_ADD ? fastcd_ff_add(x[i], y[i], f)
                                         : fastcd_ff_sub(x[i], y[i], f);
    } else if (op == FASTCD_OP_NEGATE || op == FASTCD_OP_CONJUGATE ||
            op == FASTCD_OP_IMAG_PART) {
        // Apply the CD involution to whole coefficients, not polynomial powers
        for (size_t i = 0; i < n; i++) {
            if (op == FASTCD_OP_IMAG_PART && i == 0)
                memset(out, 0, d * 4);
            else if (op == FASTCD_OP_NEGATE || (op == FASTCD_OP_CONJUGATE && i))
                fastcd_field_neg(out + i * d, x + i * d, f);
            else
                memmove(out + i * d, x + i * d, d * 4);
        }
    } else if (op == FASTCD_OP_SCALE || op == FASTCD_OP_DIVIDE) {
        // Copy or invert the scalar before modifying any coordinate of x
        memcpy(t, y, d * 4);
        if (op == FASTCD_OP_DIVIDE) {
            status = fastcd_field_inv(t, y, f);
            if (status) return status;
        }
        for (size_t i = 0; i < n; i++)
            fastcd_ff_field_mul(out + i * d, x + i * d, t, f);
    } else if (op == FASTCD_OP_INNER || op == FASTCD_OP_QUADRATIC_NORM) {
        // Stage the complete reduction, whose result lies in F_(p^d), not F_p
        dot(q, x, op == FASTCD_OP_INNER ? y : x, n, f);
        memcpy(out, q, d * 4);
    } else if (op == FASTCD_OP_RECIPROCAL) {
        /*
         * Nonzero isotropic CD elements have Q(x) = 0 and are rejected.
         * Otherwise conjugate(x)/Q(x) is a two-sided canonical reciprocal.
         */
        dot(q, x, x, n, f);
        status = fastcd_field_inv(t, q, f);
        if (status) return status;
        for (size_t i = 0; i < n; i++) {
            fastcd_ff_field_mul(out + i * d, x + i * d, t, f);
            if (i) fastcd_field_neg(out + i * d, out + i * d, f);
        }
    } else if (op == FASTCD_OP_SQUARE || op == FASTCD_OP_ANTICOMMUTATOR) {
        // For x = alpha+X and y = beta+Y the symmetric part needs only <X,Y> and
        // alpha*Y+beta*X, saved first so that squares can run in place
        if (op == FASTCD_OP_SQUARE) y = x;
        memcpy(alpha, x, d * 4);
        memcpy(beta, y, d * 4);
        dot(q, x + d, y + d, n - 1, f);
        fastcd_ff_field_mul(t, alpha, beta, f);
        fastcd_ff_field_sub(t, t, q, f);
        if (op == FASTCD_OP_ANTICOMMUTATOR) fastcd_ff_field_add(t, t, t, f);
        for (size_t i = 1; i < n; i++) {
            fastcd_ff_field_mul(u, alpha, y + i * d, f);
            fastcd_ff_field_mul(v, beta, x + i * d, f);
            fastcd_ff_field_add(out + i * d, u, v, f);
            if (op == FASTCD_OP_ANTICOMMUTATOR)
                fastcd_ff_field_add(out + i * d, out + i * d, out + i * d, f);
        }
        memcpy(out, t, d * 4);
    } else if (op == FASTCD_OP_IMAG_CROSS || op == FASTCD_OP_COMMUTATOR) {
        /*
         * Subtract the scalar and imaginary terms from xy to leave X cross Y.
         * The commutator doubles that alternating product in odd characteristic.
         */
        status = fastcd_mul_with_algorithm_ff(out, x, y, p, algorithm, work, available);
        if (status) return status;
        memset(out, 0, d * 4);
        for (size_t i = 1; i < n; i++) {
            fastcd_ff_field_mul(t, x, y + i * d, f);
            fastcd_ff_field_mul(u, y, x + i * d, f);
            fastcd_ff_field_sub(out + i * d, out + i * d, t, f);
            fastcd_ff_field_sub(out + i * d, out + i * d, u, f);
            if (op == FASTCD_OP_COMMUTATOR)
                fastcd_ff_field_add(out + i * d, out + i * d, out + i * d, f);
        }
    } else if (op == FASTCD_OP_BASIS_LEFT || op == FASTCD_OP_BASIS_RIGHT) {
        // Reuse the real basis convention's signed XOR permutation on CD indices
        for (size_t i = 0; i < n; i++) {
            FastCDBasisProduct b;
            fastcd_basis_product(&b, n, op == FASTCD_OP_BASIS_LEFT ? index : i,
                    op == FASTCD_OP_BASIS_LEFT ? i : index);
            if (b.sign < 0)
                fastcd_field_neg(out + b.index * d, x + i * d, f);
            else
                memcpy(out + b.index * d, x + i * d, d * 4);
        }
    } else {
        /*
         * a is reusable intermediate storage, and b holds the first fully
         * parenthesized term. All four products share the remaining scratch.
         */
        uint32_t* a = work;
        uint32_t* b = a + words;
        uint32_t* scratch = b + words;
#define MUL(o, l, r)                                                                   \
    do {                                                                               \
        status = fastcd_mul_with_algorithm_ff(                                         \
                o, l, r, p, algorithm, scratch, available - 2 * words);                \
        if (status) return status;                                                     \
    } while (0)
        if (op == FASTCD_OP_ASSOCIATOR) {
            MUL(a, x, y);
            MUL(b, a, z);
            MUL(a, y, z);
            MUL(out, x, a);
        } else if (op == FASTCD_OP_LEFT_OPERATOR_COMMUTATOR) {
            MUL(a, y, z);
            MUL(b, x, a);
            MUL(a, x, z);
            MUL(out, y, a);
        } else {
            MUL(a, z, y);
            MUL(b, a, x);
            MUL(a, z, x);
            MUL(out, a, y);
        }
#undef MUL
        for (size_t i = 0; i < words; i++) out[i] = fastcd_ff_sub(b[i], out[i], f);
    }
    return FASTCD_FF_SUCCESS;
}

/* ---------- Strided batches ---------- */

// Validate the batch once and keep dispatch and validation out of its product loop
FastCDFFStatus fastcd_mul_batch_ff(uint32_t* out, size_t os, const uint32_t* x,
        size_t xs, const uint32_t* y, size_t ys, size_t count, const FastCDPlanFF* p,
        FastCDAlgorithm algorithm, uint32_t* work, size_t length) {
    if (!p) return FASTCD_FF_NULL_POINTER;
    if (algorithm != FASTCD_ALGORITHM_AUTO && algorithm != FASTCD_ALGORITHM_DIRECT &&
            algorithm != FASTCD_ALGORITHM_QUASILINEAR)
        return FASTCD_FF_UNSUPPORTED_ALGORITHM;
    if (!count) return FASTCD_FF_SUCCESS;
    if (count == 1)
        return fastcd_mul_with_algorithm_ff(out, x, y, p, algorithm, work, length);
    size_t w = p->words;
    if (os < w || (xs && xs < w) || (ys && ys < w)) return FASTCD_FF_INVALID_STRIDE;
    if (!out || !x || !y || !work) return FASTCD_FF_NULL_POINTER;
    if (length < p->workspace_words) return FASTCD_FF_INSUFFICIENT_WORKSPACE;
    /*
     * Strides count uint32 words, and a zero input stride means broadcast.
     * Check the final record's extent before multiplying any span by four.
     */
    size_t limit = SIZE_MAX / sizeof(uint32_t);
    if (w > limit || (count - 1) > (limit - w) / os ||
            (xs && (count - 1) > (limit - w) / xs) ||
            (ys && (count - 1) > (limit - w) / ys))
        return FASTCD_FF_SIZE_OVERFLOW;
    size_t ob = ((count - 1) * os + w) * 4, xb = ((count - 1) * xs + w) * 4,
           yb = ((count - 1) * ys + w) * 4;
    size_t wb = p->workspace_words * 4;
    if (fastcd_ff_overlap(out, ob, x, xb) || fastcd_ff_overlap(out, ob, y, yb) ||
            fastcd_ff_overlap(out, ob, p, sizeof(*p)) ||
            fastcd_ff_overlap(work, wb, p, sizeof(*p)) ||
            fastcd_ff_overlap(work, wb, out, ob) ||
            fastcd_ff_overlap(work, wb, x, xb) || fastcd_ff_overlap(work, wb, y, yb))
        return FASTCD_FF_OVERLAPPING_BUFFERS;
    // Reject every forbidden input alias before starting the unchecked loop
    for (size_t i = 0; i < count; i++)
        if (x + i * xs != y + i * ys &&
                fastcd_ff_overlap(x + i * xs, w * 4, y + i * ys, w * 4))
            return FASTCD_FF_OVERLAPPING_BUFFERS;
    for (size_t i = 0; i < count; i++) {
        fastcd_ff_mul_unchecked(
                out + i * os, x + i * xs, y + i * ys, p, algorithm, work);
    }
    return FASTCD_FF_SUCCESS;
}

// Return one field coefficient Q(x)=sum_i x_i^2 without allocating scratch
FastCDFFStatus fastcd_quadratic_norm_ff(
        uint32_t* out, const uint32_t* x, const FastCDPlanFF* plan) {
    return fastcd_apply_ff(out, x, NULL, NULL, plan, FASTCD_OP_QUADRATIC_NORM, 0,
            FASTCD_ALGORITHM_AUTO, NULL, 0);
}
