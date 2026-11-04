/*
 * Coefficient evaluation and reconstruction around the CD schedule.
 *
 * As in the original real quasilinear kernel, preprocessing and recursive
 * arithmetic are separate stages. Here a polynomial CRT tensor expresses
 * each extension-field product as independent prime-field products; packed
 * backends run the same bilinear CD schedule on those product channels.
 * Linear, quadratic and, when needed, cubic factors cover the convolution.
 *
 * The maps depend on the actual monic defining polynomial. They are built
 * within each complete call, never cached globally or specialized to an
 * operand. Public inputs and outputs remain canonical uint32 coefficients;
 * Montgomery, byte, short and shuffled encodings exist only in scratch.
 */
#include <string.h>
#include "finite_backend.h"
#include "field_internal.h"

#if defined(FASTCD_FF_FINITE_SIMD)
#define RANK_LIMIT 48
#define POLY_LIMIT 25

typedef struct {
    size_t rank;
    // Each row evaluates one product channel; its matching output row
    // projects that channel's result back to d polynomial coordinates.
    uint32_t input[RANK_LIMIT][12];
    uint32_t output[RANK_LIMIT][12];
} Tensor;

// Exponentiate a prime residue for setup-only inverses and elimination pivots
static uint32_t powmod(uint32_t a, uint32_t e, const FastCDField* f) {
    uint32_t r = 1;
    while (e) {
        if (e & 1) r = fastcd_ff_mul(r, a, f);
        a = fastcd_ff_mul(a, a, f);
        e >>= 1;
    }
    return r;
}

/* In-place division by a monic polynomial, returning its remainder. */
static void poly_remainder(uint32_t* a, size_t length, const uint32_t* g, size_t degree,
        const FastCDField* f) {
    for (size_t k = length; k-- > degree;) {
        uint32_t v = a[k];
        for (size_t j = 0; j < degree; j++)
            a[k - degree + j] =
                    fastcd_ff_sub(a[k - degree + j], fastcd_ff_mul(v, g[j], f), f);
        a[k] = 0;
    }
}

// For degree two or three, absence of a prime-field root proves irreducibility
static int has_root(const uint32_t* g, size_t h, const FastCDField* f) {
    for (uint32_t a = 0; a < f->prime; a++) {
        uint32_t v = g[h];
        for (size_t j = h; j-- > 0;) v = fastcd_ff_add(fastcd_ff_mul(v, a, f), g[j], f);
        if (v == 0) return 1;
    }
    return 0;
}

/* Invert q modulo monic g of degree <=3 by solving its multiplication map.
 * The factors are irreducible and pairwise distinct, so pivots exist. */
static void inverse(uint32_t* out, const uint32_t* q, const uint32_t* g, size_t h,
        const FastCDField* f) {
    uint32_t m[3][4] = {{0}};
    for (size_t col = 0; col < h; col++) {
        uint32_t t[6] = {0};
        for (size_t j = 0; j < h; j++) t[j + col] = q[j];
        poly_remainder(t, 2 * h - 1, g, h, f);
        for (size_t row = 0; row < h; row++) m[row][col] = t[row];
    }
    m[0][h] = 1;
    for (size_t j = 0; j < h; j++) {
        size_t pivot = j;
        while (pivot < h && m[pivot][j] == 0) pivot++;
        /* A failed pivot is an implementation error, never an input case. */
        if (pivot == h) __builtin_trap();
        for (size_t k = 0; k <= h; k++) {
            uint32_t v = m[j][k];
            m[j][k] = m[pivot][k];
            m[pivot][k] = v;
        }
        uint32_t s = powmod(m[j][j], f->prime - 2, f);
        for (size_t k = j; k <= h; k++) m[j][k] = fastcd_ff_mul(m[j][k], s, f);
        for (size_t row = 0; row < h; row++)
            if (row != j) {
                s = m[row][j];
                for (size_t k = j; k <= h; k++)
                    m[row][k] =
                            fastcd_ff_sub(m[row][k], fastcd_ff_mul(s, m[j][k], f), f);
            }
    }
    for (size_t j = 0; j < h; j++) out[j] = m[j][h];
}

/* A product of degree-(d-1) polynomials has degree at most 2d-2. Distinct
 * irreducible factors with total degree >= 2d-1 therefore determine the
 * whole convolution, before reduction by the user's field polynomial. */
// Construct matching evaluation and reconstruction maps for a field presentation
static void prepare_tensor(Tensor* t, const FastCDField* f) {
    uint32_t factors[23][4] = {{0}};
    size_t degrees[23], nf = 0, total = 0, target = 2 * f->degree - 1;
    // Use distinct linear factors first. Higher-degree enumeration is needed
    // only for small primes, when there are too few distinct linear points.
    for (uint32_t a = 0; a < f->prime && total < target; a++) {
        degrees[nf] = 1;
        factors[nf][0] = a ? f->prime - a : 0;
        factors[nf][1] = 1;
        nf++;
        total++;
    }
    for (size_t h = 2; h <= 3 && total < target; h++) {
        uint32_t candidates = 1;
        for (size_t j = 0; j < h; j++) candidates *= f->prime;
        for (uint32_t code = 1; code < candidates && total < target; code++) {
            uint32_t g[4] = {0}, value = code;
            for (size_t j = 0; j < h; j++) {
                g[j] = value % f->prime;
                value /= f->prime;
            }
            g[h] = 1;
            if (has_root(g, h, f)) continue;
            degrees[nf] = h;
            memcpy(factors[nf], g, sizeof(g));
            nf++;
            total += h;
        }
    }
    if (total < target || total >= POLY_LIMIT) __builtin_trap();
    // Multiply the selected factors to obtain the monic CRT nodal polynomial.
    uint32_t nodal[POLY_LIMIT] = {1};
    size_t len = 1;
    for (size_t i = 0; i < nf; i++) {
        uint32_t next[POLY_LIMIT] = {0};
        size_t h = degrees[i];
        for (size_t j = 0; j < len; j++)
            for (size_t k = 0; k <= h; k++)
                next[j + k] = fastcd_ff_add(
                        next[j + k], fastcd_ff_mul(nodal[j], factors[i][k], f), f);
        len += h;
        memcpy(nodal, next, sizeof(nodal));
    }
    memset(t, 0, sizeof(*t));
    for (size_t i = 0; i < nf; i++) {
        size_t h = degrees[i];
        const uint32_t* g = factors[i];
        // q = nodal/g. Then q * inverse(q mod g) is one modulo g and
        // zero modulo all other factors: the CRT reconstruction idempotent.
        uint32_t temp[POLY_LIMIT], q[POLY_LIMIT] = {0};
        memcpy(temp, nodal, sizeof(temp));
        for (size_t k = total + 1; k-- > h;) {
            uint32_t v = temp[k];
            q[k - h] = v;
            for (size_t j = 0; j <= h; j++)
                temp[k - h + j] =
                        fastcd_ff_sub(temp[k - h + j], fastcd_ff_mul(v, g[j], f), f);
        }
        uint32_t small[POLY_LIMIT];
        memcpy(small, q, sizeof(small));
        poly_remainder(small, total - h + 1, g, h, f);
        uint32_t inv[3] = {0};
        inverse(inv, small, g, h, f);
        uint32_t basis[POLY_LIMIT] = {0};
        for (size_t j = 0; j <= total - h; j++)
            for (size_t k = 0; k < h; k++)
                basis[j + k] =
                        fastcd_ff_add(basis[j + k], fastcd_ff_mul(q[j], inv[k], f), f);
        // Reduce t^j times that idempotent into the user's field basis.
        // These weights combine CRT reconstruction and final field reduction.
        uint32_t w[3][12] = {{0}};
        for (size_t j = 0; j < h; j++) {
            memcpy(temp, basis, sizeof(temp));
            poly_remainder(temp, total, f->modulus, f->degree, f);
            memcpy(w[j], temp, f->degree * sizeof(uint32_t));
            uint32_t carry = basis[total - 1];
            for (size_t k = total; k-- > 1;) basis[k] = basis[k - 1];
            basis[0] = 0;
            for (size_t k = 0; k < total; k++)
                basis[k] =
                        fastcd_ff_sub(basis[k], fastcd_ff_mul(carry, nodal[k], f), f);
        }
        // Record the coordinate map from the user's polynomial basis to
        // residues modulo this factor; it is independent of operand values.
        uint32_t rem[3][12] = {{0}};
        for (size_t j = 0; j < f->degree; j++) {
            memset(temp, 0, sizeof(temp));
            temp[j] = 1;
            poly_remainder(temp, j + 1, g, h, f);
            for (size_t k = 0; k < h; k++) rem[k][j] = temp[k];
        }
        // Quadratic and cubic convolution use three and six symmetric
        // products. The tables restore their monomial coefficients from
        // diagonal products and products of sums, before reduction modulo g.
        size_t products = h == 1 ? 1 : h == 2 ? 3 : 6;
        static const int8_t cubic[6][5] = {{1, -1, -1, 0, 0}, {0, -1, 1, -1, 0},
                {0, 0, -1, -1, 1}, {0, 1, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 0, 0, 1, 0}};
        static const int8_t quadratic[3][3] = {{1, -1, 0}, {0, -1, 1}, {0, 1, 0}};
        for (size_t r = 0; r < products; r++) {
            size_t first = r, second = r;
            if (h == 2 && r == 2) {
                first = 0;
                second = 1;
            }
            if (h == 3 && r >= 3) {
                first = r == 5 ? 1 : 0;
                second = r == 3 ? 1 : 2;
            }
            // Evaluation rows contain either one residual coordinate or the
            // sum of a pair; the matching projection undoes that product basis.
            for (size_t j = 0; j < f->degree; j++)
                t->input[t->rank][j] = first == second
                        ? rem[first][j]
                        : fastcd_ff_add(rem[first][j], rem[second][j], f);
            memset(temp, 0, sizeof(temp));
            for (size_t k = 0; k < 2 * h - 1; k++) {
                int8_t v = h == 1 ? 1 : h == 2 ? quadratic[r][k] : cubic[r][k];
                temp[k] = v < 0 ? f->prime - 1 : (uint32_t)v;
            }
            poly_remainder(temp, 2 * h - 1, g, h, f);
            for (size_t j = 0; j < f->degree; j++)
                for (size_t k = 0; k < h; k++)
                    t->output[t->rank][j] = fastcd_ff_add(t->output[t->rank][j],
                            fastcd_ff_mul(temp[k], w[k][j], f), f);
            t->rank++;
        }
    }
    // With d<=12 the selected factors fit these fixed setup arrays. Failure
    // here indicates an internal schedule inconsistency, not a field input.
    if (t->rank > RANK_LIMIT) __builtin_trap();
}

#endif

#if defined(FASTCD_FF_FINITE_SIMD)
/* The small-characteristic ranks follow from the available irreducible CRT
 * factors; no defining polynomial or operand is singled out by dispatch. */
static size_t tensor_rank(const FastCDField* f) {
    size_t target = 2 * f->degree - 1;
    size_t degree = f->prime < target ? f->prime : target, rank = degree;
    if (degree == target) return rank;
    size_t quadratics = (size_t)f->prime * (f->prime - 1) / 2;
    while (degree < target && quadratics) {
        degree += 2;
        rank += 3;
        quadratics--;
    }
    if (degree < target) rank += 6 * ((target - degree + 2) / 3);
    return rank;
}

// Minimize rounded product channels; break ties in favor of wider point groups
static size_t point_width(size_t rank, size_t lanes) {
    size_t best = lanes / 4, cost = SIZE_MAX;
    for (size_t k = best; k <= lanes; k *= 2) {
        size_t rounded = (rank + k - 1) / k * k;
        if (rounded <= cost) {
            best = k;
            cost = rounded;
        }
    }
    return best;
}

// Return a word count or an overflow sentinel, checking its byte extent too
static size_t checked_words(size_t n, size_t factor, size_t extra) {
    if (n > (SIZE_MAX / sizeof(uint32_t) - extra) / factor) return SIZE_MAX;
    return n * factor + extra;
}
#endif

/* ---------- Representation and instruction dispatch ---------- */

/* Selection depends only on field parameters, dimension and CPU features.
 * The loop flag belongs to private shared templates; public fastCD calls
 * always select the CD component (loop=0). Scratch counts include encoding
 * and reconstruction, allowing the plan to reserve one safe shared buffer. */
// Select a range-safe backend and report its scratch requirement in uint32 words
FastCDFFBackend fastcd_ff_finite_select(
        const FastCDPlanFF* p, int loop, size_t* words) {
    *words = 0;
#if defined(FASTCD_FF_FINITE_SIMD)
    size_t n = p->dimension, d = p->field.degree, lanes = 0;
    uint32_t prime = p->field.prime;
    int wide = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq");
    int bytes = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw");
    int avx = __builtin_cpu_supports("avx2"), sse = __builtin_cpu_supports("sse4.1");
    if (prime == 3 && d <= 2 && bytes && n >= 256) {
        *words = checked_words(n / 4, d == 1 ? (loop ? 25 : 23) : (loop ? 33 : 29), 64);
        /* n is a multiple of 256; the encoded arrays use byte lanes. */
        return d == 1 ? FASTCD_FF_FINITE_BYTE_PRIME : FASTCD_FF_FINITE_BYTE_QUADRATIC;
    }
    if (d == 1) {
        // Prime fields skip tensor construction and use the narrowest safe
        // arithmetic before falling back to full-width Montgomery lanes.
        if (prime < 8 && bytes && n >= 128) {
            *words = checked_words(n, loop ? 14 : 10, 64);
            return FASTCD_FF_FINITE_SMALL_PRIME32;
        }
        if (prime < 32768 && bytes && n >= 128) {
            *words = checked_words(n, loop ? 14 : 10, 64);
            if (prime < 182) return FASTCD_FF_FINITE_MEDIUM_BAR_PRIME;
            return prime < 256 ? FASTCD_FF_FINITE_SHORT_PRIME
                               : FASTCD_FF_FINITE_MEDIUM_MONT_PRIME;
        }
        if (wide && n >= 64)
            lanes = 16;
        else if (avx && n >= 32)
            lanes = 8;
        else if (sse && n >= 16)
            lanes = 4;
        if (lanes) {
            size_t levels = lanes == 16 ? 4 : lanes == 8 ? 3 : 2;
            *words = checked_words(n, (loop ? 13 : 5) + 3 * levels, 64);
            return lanes == 16   ? FASTCD_FF_FINITE_MONT16
                    : lanes == 8 ? FASTCD_FF_FINITE_MONT8
                                 : FASTCD_FF_FINITE_MONT4;
        }
    } else if (prime < 8 && n >= 16) {
        // Too few linear evaluation points: pack channels of the small-prime
        // CRT tensor, with ternary encodings when p=3 and the ISA permits them.
        size_t rank = tensor_rank(&p->field);
        if (prime == 3 && bytes && n >= 32) {
            if (rank <= 16 && n >= 128) {
                *words = checked_words(n, loop ? 22 : 20, 64);
                return FASTCD_FF_FINITE_TERNARY16;
            }
            if (rank <= 32 && n >= 64) {
                *words = checked_words(n, loop ? 38 : 34, 64);
                return FASTCD_FF_FINITE_TERNARY32;
            }
            *words = checked_words(n, loop ? 64 : 56, 64);
            return FASTCD_FF_FINITE_TERNARY64;
        }
        if (bytes && n >= 32) {
            if (rank > 16 && rank <= 32) {
                *words = checked_words(n, loop ? 144 : 80, 0);
                return FASTCD_FF_FINITE_SMALL_PLAIN32;
            }
            size_t points = point_width(rank, 32), levels = 0;
            for (size_t k = points; k < 32; k *= 2) levels++;
            *words = checked_words(n, ((loop ? 7 : 5) + 3 * levels) * points / 2, 64);
            return FASTCD_FF_FINITE_SMALL32;
        }
        lanes = avx && rank > 8 ? 16 : sse ? 8 : 0;
        if (lanes) {
            *words = checked_words(n, (loop ? 9 : 5) * lanes / 2, 0);
            return lanes == 16 ? FASTCD_FF_FINITE_SMALL16 : FASTCD_FF_FINITE_SMALL8;
        }
    } else if (prime >= 2 * d - 1 && n >= 16) {
        // Enough distinct linear points give the minimal 2*d-1 convolution
        // channels. Grouping controls lane utilization and scratch layout.
        if (bytes && prime < 32768 && n >= 32) {
            size_t points = point_width(2 * d - 1, 32), levels = 0;
            for (size_t k = points; k < 32; k *= 2) levels++;
            size_t groups = (2 * d - 2 + points) / points;
            *words = checked_words(
                    n, ((loop ? 4 : 2) * groups + 3 * levels + 3) * points / 2, 64);
            return prime < 182 ? FASTCD_FF_FINITE_MEDIUM_BAR
                               : FASTCD_FF_FINITE_MEDIUM_MONT;
        }
        lanes = wide ? 16 : avx ? 8 : sse ? 4 : 0;
        if (lanes) {
            size_t points = point_width(2 * d - 1, lanes), levels = 0;
            for (size_t k = points; k < lanes; k *= 2) levels++;
            size_t groups = (2 * d - 2 + points) / points;
            *words = checked_words(
                    n, ((loop ? 4 : 2) * groups + 3 * levels + 3) * points, 64);
            return lanes == 16   ? FASTCD_FF_FINITE_EXT16
                    : lanes == 8 ? FASTCD_FF_FINITE_EXT8
                                 : FASTCD_FF_FINITE_EXT4;
        }
    } else if (prime < 256 && n >= 64) {
        // For remaining small fields, shuffle the general CRT channels in
        // short lanes; larger fields retain the generic degree specialization.
        lanes = bytes ? 32 : avx ? 16 : sse ? 8 : 0;
        if (lanes) {
            *words = checked_words(n, (loop ? 13 : 5) * lanes / 2, 0);
            return lanes == 32    ? FASTCD_FF_FINITE_SHORT32
                    : lanes == 16 ? FASTCD_FF_FINITE_SHORT16
                                  : FASTCD_FF_FINITE_SHORT8;
        }
    }
#else
    (void)p;
    (void)loop;
#endif
    return FASTCD_FF_FINITE_GENERIC;
}

#if defined(FASTCD_FF_FINITE_SIMD)
/* Keep the matrix stack frame out of the prime-field entry path, just as
 * the original real batch kernels isolate SIMD packing stack frames. */
// Prepare field-specific maps once and pass them to the chosen packed kernel
static __attribute__((noinline)) void execute_extension(FastCDFFProduct product,
        uint32_t* out, const uint32_t* x, const uint32_t* y, const FastCDPlanFF* p,
        uint32_t* work, int loop) {
    Tensor tensor;
    if (p->field.prime < 8 && __builtin_cpu_supports("avx512f") &&
            __builtin_cpu_supports("avx512bw"))
        tensor.rank = fastcd_ff_prepare_small(tensor.input, tensor.output, &p->field);
    else if (p->field.prime >= 2 * p->field.degree - 1 &&
            __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq")) {
        memset(&tensor, 0, sizeof tensor);
        tensor.rank = 2 * p->field.degree - 1;
        fastcd_ff_prepare_linear(tensor.output, &p->field);
    } else
        prepare_tensor(&tensor, &p->field);
    product(out, x, y, p, work, loop, tensor.rank, (const uint32_t(*)[12])tensor.input,
            (const uint32_t(*)[12])tensor.output);
}
#endif

// Execute a validated backend, returning zero only for the generic buffer fallback
int fastcd_ff_finite_execute(uint32_t* out, const uint32_t* x, const uint32_t* y,
        const FastCDPlanFF* p, uint32_t* work, int loop) {
    size_t need;
    FastCDFFBackend backend = fastcd_ff_finite_select(p, loop, &need);
    /* Retain the portable schedule as a workspace-safe fallback. */
    size_t available = p->workspace_words;
    if (need > available) backend = FASTCD_FF_FINITE_GENERIC;
#if defined(FASTCD_FF_FINITE_SIMD)
    switch (backend) {
        case FASTCD_FF_FINITE_BYTE_PRIME:
            fastcd_ff_byte64_prime(out, x, y, p, work, loop);
            return 1;
        case FASTCD_FF_FINITE_BYTE_QUADRATIC:
            fastcd_ff_byte64_quadratic(out, x, y, p, work, loop);
            return 1;
#define PRIME_CASE(kind, name)                                                         \
    case kind:                                                                         \
        name(out, x, y, p, work, loop, 0, NULL, NULL);                                 \
        return 1
            PRIME_CASE(FASTCD_FF_FINITE_MONT4, fastcd_ff_mont4_prime_tensor);
            PRIME_CASE(FASTCD_FF_FINITE_MONT8, fastcd_ff_mont8_prime_tensor);
            PRIME_CASE(FASTCD_FF_FINITE_MONT16, fastcd_ff_mont16_prime_tensor);
            PRIME_CASE(FASTCD_FF_FINITE_TERNARY16, fastcd_ff_ternary16_tensor);
            PRIME_CASE(FASTCD_FF_FINITE_TERNARY32, fastcd_ff_ternary32_tensor);
            PRIME_CASE(FASTCD_FF_FINITE_TERNARY64, fastcd_ff_ternary64_tensor);
#undef PRIME_CASE
        case FASTCD_FF_FINITE_MEDIUM_BAR_PRIME:
            fastcd_ff_medium_bar_prime(out, x, y, p, work, loop);
            return 1;
        case FASTCD_FF_FINITE_SHORT_PRIME:
            fastcd_ff_short32_prime(out, x, y, p, work, loop);
            return 1;
        case FASTCD_FF_FINITE_MEDIUM_MONT_PRIME:
            fastcd_ff_medium_mont_prime(out, x, y, p, work, loop);
            return 1;
        case FASTCD_FF_FINITE_SMALL_PRIME32:
            fastcd_ff_small32_prime(out, x, y, p, work, loop);
            return 1;
#define EXT_CASE(kind, name)                                                           \
    case kind:                                                                         \
        execute_extension(name, out, x, y, p, work, loop);                             \
        return 1
            EXT_CASE(FASTCD_FF_FINITE_MEDIUM_BAR, fastcd_ff_medium_bar_tensor);
            EXT_CASE(FASTCD_FF_FINITE_MEDIUM_MONT, fastcd_ff_medium_mont_tensor);
            EXT_CASE(FASTCD_FF_FINITE_EXT4, fastcd_ff_extension4_tensor);
            EXT_CASE(FASTCD_FF_FINITE_EXT8, fastcd_ff_extension8_tensor);
            EXT_CASE(FASTCD_FF_FINITE_EXT16, fastcd_ff_extension16_tensor);
            EXT_CASE(FASTCD_FF_FINITE_SMALL8, fastcd_ff_small8_tensor);
            EXT_CASE(FASTCD_FF_FINITE_SMALL16, fastcd_ff_small16_tensor);
            EXT_CASE(FASTCD_FF_FINITE_SMALL32, fastcd_ff_small32_tensor);
            EXT_CASE(FASTCD_FF_FINITE_SMALL_PLAIN32, fastcd_ff_small32_plain_tensor);
            EXT_CASE(FASTCD_FF_FINITE_SHORT8, fastcd_ff_short8_tensor);
            EXT_CASE(FASTCD_FF_FINITE_SHORT16, fastcd_ff_short16_tensor);
            EXT_CASE(FASTCD_FF_FINITE_SHORT32, fastcd_ff_short32_tensor);
#undef EXT_CASE
        default:
            break;
    }
#else
    (void)backend;
#endif
    if (p->field.degree == 1) {
        fastcd_ff_finite_prime_scalar(out, x, y, p, work, loop);
        return 1;
    }
    return 0;
}
