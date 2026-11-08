#include <limits.h>
#include <string.h>

#define FASTCD_FF_FIELD_INLINES_C
#include "field_internal.h"
#include "internal.h"

/*
 * Field setup, kept separate from the optimized product kernels. Primality
 * tests, polynomial products, Frobenius powers and gcds use bounded stack
 * arrays and canonical residues, and no setup work runs inside a product.
 *
 * Clients receive a fully checked field by value: its reciprocal serves prime
 * reduction, and its negated modulus coefficients serve the fixed-degree
 * generated convolution, with no mutable tables or allocation.
 */

/* ---------- FLINT-independent setup ---------- */

// Raise a residue to a power for primality tests and polynomial setup
static uint32_t power(uint32_t a, uint64_t exponent, const FastCDField* f) {
    uint32_t value = 1;
    while (exponent != 0) {
        if ((exponent & 1) != 0) value = fastcd_ff_mul(value, a, f);
        a = fastcd_ff_mul(a, a, f);
        exponent >>= 1;
    }
    return value;
}

// Test primality with Miller--Rabin witnesses exact for all positive int32 values
static int is_prime(const FastCDField* f) {
    static const uint32_t bases[] = {2, 3, 5, 7, 11};
    uint32_t p = f->prime;
    uint32_t odd = p - 1;
    unsigned shifts = 0;
    if (p == 2) return 1;
    if ((p & 1) == 0) return 0;
    while ((odd & 1) == 0) {
        odd >>= 1;
        shifts++;
    }
    for (size_t j = 0; j < sizeof(bases) / sizeof(bases[0]); j++) {
        // Write p-1 = odd*2^shifts and look for the required +/-1 witness
        if (p == bases[j]) return 1;
        uint32_t z = power(bases[j], odd, f);
        if (z == 1 || z == p - 1) continue;
        unsigned k;
        for (k = 1; k < shifts; k++) {
            z = fastcd_ff_mul(z, z, f);
            if (z == p - 1) break;
        }
        if (k == shifts) return 0;
    }
    return 1;
}

// Multiply setup polynomials modulo the supplied monic f
static void poly_product(
        uint32_t* out, const uint32_t* a, const uint32_t* b, const FastCDField* f) {
    uint32_t t[2 * FASTCD_FF_MAX_DEGREE - 1] = {0};
    size_t d = f->degree;
    for (size_t i = 0; i < d; i++) {
        for (size_t j = 0; j < d; j++)
            t[i + j] = fastcd_ff_add(t[i + j], fastcd_ff_mul(a[i], b[j], f), f);
    }
    /*
     * Eliminate high powers in descending order with the monic modulus. The
     * temporary convolution makes exact operand aliases safe.
     */
    for (size_t k = 2 * d - 1; k-- > d;) {
        for (size_t j = 0; j < d; j++)
            t[k - d + j] = fastcd_ff_sub(
                    t[k - d + j], fastcd_ff_mul(t[k], f->modulus[j], f), f);
    }
    memcpy(out, t, d * sizeof(uint32_t));
}

// Raise a residue polynomial to the prime-th power with fixed stack storage
static void frobenius(uint32_t* out, const uint32_t* a, const FastCDField* f) {
    uint32_t value[FASTCD_FF_MAX_DEGREE] = {1};
    uint32_t base[FASTCD_FF_MAX_DEGREE];
    memcpy(base, a, f->degree * sizeof(uint32_t));
    uint32_t e = f->prime;
    while (e != 0) {
        if ((e & 1) != 0) poly_product(value, value, base, f);
        poly_product(base, base, base, f);
        e >>= 1;
    }
    memcpy(out, value, f->degree * sizeof(uint32_t));
}

// Return whether gcd(f, a) is constant, using ordinary polynomial division
static int coprime(const uint32_t* a, const FastCDField* f) {
    uint32_t u[FASTCD_FF_MAX_DEGREE + 1], v[FASTCD_FF_MAX_DEGREE + 1] = {0};
    memcpy(u, f->modulus, sizeof(u));
    memcpy(v, a, f->degree * sizeof(uint32_t));
    int du = (int)f->degree, dv = (int)f->degree - 1;
    while (dv >= 0 && v[dv] == 0) dv--;
    while (dv >= 0) {
        uint32_t inverse = power(v[dv], f->prime - 2, f);
        for (int k = du; k >= dv; k--) {
            uint32_t c = fastcd_ff_mul(u[k], inverse, f);
            for (int j = 0; j <= dv; j++)
                u[k - dv + j] =
                        fastcd_ff_sub(u[k - dv + j], fastcd_ff_mul(c, v[j], f), f);
        }
        // Euclid step: replace (u,v) by (v,u mod v), then trim its degree
        uint32_t swap[FASTCD_FF_MAX_DEGREE + 1];
        memcpy(swap, u, sizeof(u));
        memcpy(u, v, sizeof(u));
        memcpy(v, swap, sizeof(v));
        du = dv;
        while (dv >= 0 && v[dv] == 0) dv--;
    }
    return du == 0;
}

// Test irreducibility by Rabin: x^(p^d) = x and a gcd check per prime divisor of d
static int is_irreducible(const FastCDField* f) {
    if (f->degree == 1) return 1;
    uint32_t x[FASTCD_FF_MAX_DEGREE] = {0, 1};
    uint32_t h[FASTCD_FF_MAX_DEGREE] = {0, 1};
    /*
     * h holds x^(p^k) modulo f after each iteration. Only k = d/r for prime
     * divisors r needs a gcd test, and the final check h = x completes the
     * criterion.
     */
    for (uint32_t k = 1; k <= f->degree; k++) {
        frobenius(h, h, f);
        if (k < f->degree && f->degree % k == 0) {
            uint32_t quotient = f->degree / k;
            // With d <= 12 the prime divisors are among 2, 3, 5, 7 and 11
            if (quotient == 2 || quotient == 3 || quotient == 5 || quotient == 7 ||
                    quotient == 11) {
                uint32_t difference[FASTCD_FF_MAX_DEGREE];
                for (size_t j = 0; j < f->degree; j++)
                    difference[j] = fastcd_ff_sub(h[j], x[j], f);
                if (!coprime(difference, f)) return 0;
            }
        }
    }
    return memcmp(x, h, f->degree * sizeof(uint32_t)) == 0;
}

// Build the field locally so a rejected setup leaves the caller's storage untouched
static FastCDFFStatus field_init(FastCDField* field, uint32_t prime, uint32_t degree,
        const uint32_t* modulus, int trusted) {
    if (field == NULL) return FASTCD_FF_NULL_POINTER;
    if (prime < 2 || prime > INT32_MAX || degree < 1 || degree > FASTCD_FF_MAX_DEGREE)
        return FASTCD_FF_INVALID_FIELD;
    if (degree > 1 && modulus == NULL) return FASTCD_FF_NULL_POINTER;
    if (modulus != NULL &&
            fastcd_ff_overlap(field, sizeof(*field), modulus,
                    ((size_t)degree + 1) * sizeof(uint32_t)))
        return FASTCD_FF_OVERLAPPING_BUFFERS;
    FastCDField result = {0};
    result.prime = prime;
    result.degree = degree;
    result.reciprocal = UINT64_MAX / prime;
    if (modulus == NULL)
        result.modulus[1] = 1;
    else {
        if (modulus[degree] != 1) return FASTCD_FF_INVALID_FIELD;
        for (size_t i = 0; i <= degree; i++) {
            if (modulus[i] >= prime) return FASTCD_FF_INVALID_FIELD;
            result.modulus[i] = modulus[i];
        }
    }
    // Cache -modulus[i] canonically so the product kernels add each high-degree
    // correction without branching on signed coefficients
    for (size_t i = 0; i < degree; i++)
        result.negative[i] = result.modulus[i] == 0 ? 0 : prime - result.modulus[i];
    if (!trusted && (!is_prime(&result) || !is_irreducible(&result)))
        return FASTCD_FF_INVALID_FIELD;
    *field = result;
    return FASTCD_FF_SUCCESS;
}

// Establish primality and irreducibility before returning a usable field
FastCDFFStatus fastcd_ff_field_init(
        FastCDField* f, uint32_t p, uint32_t d, const uint32_t* m) {
    return field_init(f, p, d, m, 0);
}

// Skip the primality and irreducibility checks a trusted caller has already made
FastCDFFStatus fastcd_ff_field_init_trusted(
        FastCDField* f, uint32_t p, uint32_t d, const uint32_t* m) {
    return field_init(f, p, d, m, 1);
}

/* ---------- Public scalar entry points ---------- */

// Add complete field scalars coordinatewise; exact in-place aliases are safe
void fastcd_ff_field_add(
        uint32_t* out, const uint32_t* x, const uint32_t* y, const FastCDField* f) {
    for (size_t i = 0; i < f->degree; i++) out[i] = fastcd_ff_add(x[i], y[i], f);
}

// Subtract complete field scalars while keeping every coordinate canonical
void fastcd_ff_field_sub(
        uint32_t* out, const uint32_t* x, const uint32_t* y, const FastCDField* f) {
    for (size_t i = 0; i < f->degree; i++) out[i] = fastcd_ff_sub(x[i], y[i], f);
}

// Multiply extension-field scalars out of line, away from the prime-field inline path
void fastcd_ff_field_mul_extension(
        uint32_t* out, const uint32_t* x, const uint32_t* y, const FastCDField* f) {
    switch (f->degree) {
#define FASTCD_FF_PRODUCT_CASE(d)                                                      \
    case d:                                                                            \
        fastcd_ff_product##d(out, x, y, f);                                            \
        break
        FASTCD_FF_PRODUCT_CASE(2);
        FASTCD_FF_PRODUCT_CASE(3);
        FASTCD_FF_PRODUCT_CASE(4);
        FASTCD_FF_PRODUCT_CASE(5);
        FASTCD_FF_PRODUCT_CASE(6);
        FASTCD_FF_PRODUCT_CASE(7);
        FASTCD_FF_PRODUCT_CASE(8);
        FASTCD_FF_PRODUCT_CASE(9);
        FASTCD_FF_PRODUCT_CASE(10);
        FASTCD_FF_PRODUCT_CASE(11);
        FASTCD_FF_PRODUCT_CASE(12);
#undef FASTCD_FF_PRODUCT_CASE
    }
}
