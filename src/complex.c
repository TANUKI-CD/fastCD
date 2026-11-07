/*
 * Standard Cayley--Dickson arithmetic with complex coefficients.
 *
 * The CD involution fixes each complex coefficient; it negates only the
 * non-scalar CD coordinates. Consequently the quadratic form and pairing
 * are bilinear, not Hermitian: Q(x) = sum_i x_i^2 can vanish for x != 0.
 *
 * Direct multiplication uses the defining four-product recursion. The
 * quasilinear path instantiates the original real path's complex alternating
 * recursion, including its 243-operation complex octonion leaf, and then
 * restores the scalar/imaginary terms of the complete product.
 *
 * Public coefficients are interleaved {real, imag} pairs. The alternating
 * kernel uses separate planes, so packing, outputs, and recursive scratch
 * all live in caller-provided workspace. No original real kernel is changed.
 *
 * The auxiliary alternating recursion has two complex children. For the
 * original real-input path they can be conjugate partners, allowing one
 * child at the root; arbitrary complex coefficients do not have that
 * symmetry. Here both children run through the existing complex template.
 * Its auxiliary iota and CD basis units are separate from coefficient-level
 * complex conjugation, which this implementation never uses as the CD star.
 */
#include "fastcd/complex.h"
#include <math.h>
#include <stdint.h>
#include <string.h>
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#ifndef FASTCD_QL_COMPLEX_LEAF
#define FASTCD_QL_COMPLEX_LEAF 8
#endif
#ifndef FASTCD_QL_FULL_LEAF
#define FASTCD_QL_FULL_LEAF 16
#endif
#define FASTCD_REAL double
#define FASTCD_ZERO 0.0
#define FASTCD_NAME(n) c128_##n
#include "generated/quasilinear_kernels.h"
#include "quasilinear_impl.h"
#undef FASTCD_NAME
#undef FASTCD_ZERO
#undef FASTCD_REAL
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

// Keep coefficient arithmetic in two explicit doubles, matching public
// storage without depending on a compiler-specific C complex-number ABI.
typedef FastCDComplexF64 C;

/* ---------- Complex coefficient arithmetic ---------- */

// Add two coefficient-field scalars
static C add(C a, C b) {
    C r = {a.real + b.real, a.imag + b.imag};
    return r;
}

// Subtract two coefficient-field scalars
static C sub(C a, C b) {
    C r = {a.real - b.real, a.imag - b.imag};
    return r;
}

// Negate a scalar without applying complex conjugation
static C neg(C a) {
    C r = {-a.real, -a.imag};
    return r;
}

// Multiply complex coefficients, independently of the CD level
static C mul(C a, C b) {
    C r = {a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real};
    return r;
}

// Test the exact zero used by division and reciprocal preflight
static int zero(C a) {
    return a.real == 0.0 && a.imag == 0.0;
}

/* Scale by the larger component before forming the squared denominator.
 * The caller rejects zero first; a.real^2 + a.imag^2 could overflow even
 * when the reciprocal itself is representable. */
static C inverse(C a) {
    C r;
    if (fabs(a.real) >= fabs(a.imag)) {
        double t = a.imag / a.real, den = 1 + t * t;
        r.real = (1 / a.real) / den;
        r.imag = -(t / a.real) / den;
    } else {
        double t = a.real / a.imag, den = 1 + t * t;
        r.real = (t / a.imag) / den;
        r.imag = -(1 / a.imag) / den;
    }
    return r;
}

// Compare byte intervals without subtracting unrelated C pointers
static int overlap(const void* a, size_t an, const void* b, size_t bn) {
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return an && bn && (x <= y ? y - x < an : x - y < bn);
}

/* Reduce two coefficient components with one block/carry schedule. Keep this
 * frame outside small dots and leaf-only products. No conjugation is applied. */
static FASTCD_NOINLINE C dot_blocks(const C* x, const C* y, size_t begin, size_t n) {
    c128_BalancedTree2 tree;
    c128_Sum2 total = {0, 0};
    tree.blocks = 0;
    for (size_t base = begin; base < n; base += FASTCD_SUM_BLOCK) {
        size_t count = n - base < FASTCD_SUM_BLOCK ? n - base : FASTCD_SUM_BLOCK;
        C first = mul(x[base], y[base]);
        total.first = first.real;
        total.second = first.imag;
        for (size_t i = base + 1; i < base + count; i++) {
            C term = mul(x[i], y[i]);
            total.first = total.first + term.real;
            total.second = total.second + term.imag;
        }
        if (count == FASTCD_SUM_BLOCK) c128_sum2_block(&tree, total);
    }
    total = c128_sum2_finish(&tree, total, (n - begin) % FASTCD_SUM_BLOCK != 0);
    C result = {total.first, total.second};
    return result;
}

// A short left fold is exactly the one-block schedule, without tree storage.
static C dot_range(const C* x, const C* y, size_t begin, size_t n) {
    if (n - begin > FASTCD_SUM_BLOCK) return dot_blocks(x, y, begin, n);
    C result = {0, 0};
    if (begin == n) return result;
    result = mul(x[begin], y[begin]);
    for (size_t i = begin + 1; i < n; i++) result = add(result, mul(x[i], y[i]));
    return result;
}

/* ---------- Defining and quasilinear products ---------- */

/* Evaluate (a,b)(c,d) = (ac - conjugate(d)b, da + b conjugate(c)).
 * This is the same ordered recursion as the original real direct kernel;
 * every conjugation here fixes the complex coefficient field. */
static void direct(C* out, const C* x, const C* y, size_t n, C* work) {
    if (n <= 4) {
        // Through quaternions, the cyclic orientation gives every basis sign.
        // Initialize from x[0] y[k] to avoid a redundant add to zero.
        // XOR identifies the unique partner j for each output k. Products
        // with j=0 are positive; equal nonzero units square to -1; the
        // remaining sign follows e1*e2=e3 and its cyclic permutations.
        for (size_t k = 0; k < n; k++) {
            C sum = mul(x[0], y[k]);
            for (size_t i = 1; i < n; i++) {
                size_t j = i ^ k;
                C term = mul(x[i], y[j]);
                int positive = j == 0 || (i != j && i % 3 + 1 == j);
                sum = positive ? add(sum, term) : sub(sum, term);
            }
            out[k] = sum;
        }
        return;
    }
    size_t h = n / 2;
    // Two half-size temporaries precede the shared child workspace.
    // The four products run sequentially, so none needs separate child scratch.
    C* temp = work;
    C* product = temp + h;
    C* child = product + h;
    // Lower half: ac - conjugate(d)b. Keep the multiplication order intact.
    direct(out, x, y, h, child);
    temp[0] = y[h];
    for (size_t i = 1; i < h; i++) temp[i] = neg(y[h + i]);
    direct(product, temp, x + h, h, child);
    for (size_t i = 0; i < h; i++) out[i] = sub(out[i], product[i]);
    // Upper half: da + b conjugate(c), reusing the same two temporaries.
    direct(out + h, y + h, x, h, child);
    temp[0] = y[0];
    for (size_t i = 1; i < h; i++) temp[i] = neg(y[i]);
    direct(product, x + h, temp, h, child);
    for (size_t i = 0; i < h; i++) out[h + i] = add(out[h + i], product[i]);
}

// Evaluate a validated product; AUTO uses the complex alternating recursion
static void product(
        C* out, const C* x, const C* y, size_t n, FastCDAlgorithm alg, double* work) {
    if (alg == FASTCD_ALGORITHM_DIRECT) {
        direct(out, x, y, n, (C*)work);
        return;
    }
    // Four packed input planes, two result planes, then the original
    // complex_cross scratch. Coordinate zero of the cross input is ignored.
    double* xr = work;
    double* xi = xr + n;
    double* yr = xi + n;
    double* yi = yr + n;
    double* rr = yi + n;
    double* ri = rr + n;
    for (size_t i = 0; i < n; i++) {
        xr[i] = x[i].real;
        xi[i] = x[i].imag;
        yr[i] = y[i].real;
        yi[i] = y[i].imag;
    }
    c128_complex_cross(rr, ri, xr, xi, yr, yi, n, ri + n);
    // rr[0],ri[0] are the zero scalar of an alternating product. Assign the
    // complete product's scalar below instead of treating it as another
    // imaginary coordinate. Both output planes are consumed before returning.
    // For x = alpha + X and y = beta + Y, restore alpha Y + beta X
    // and scalar alpha beta - <X,Y> around the alternating product X cross Y.
    C q = mul(x[0], y[0]);
    for (size_t i = 1; i < n; i++) {
        C cross = {rr[i], ri[i]};
        out[i] = add(cross, add(mul(x[0], y[i]), mul(y[0], x[i])));
    }
    if (n > 1) {
        C pairing = dot_range(x, y, 1, n);
        q = sub(q, pairing);
    }
    out[0] = q;
}

/* ---------- Checked public operations ---------- */

// Query scratch in doubles, with all byte-size arithmetic checked first
FastCDStatus fastcd_workspace_size_c128(size_t n, FastCDOperation op, size_t* length) {
    if (!length) return FASTCD_NULL_POINTER;
    if (!fastcd_valid_dimension(n)) return FASTCD_INVALID_DIMENSION;
    if (op < FASTCD_OP_MULTIPLY || op > FASTCD_OP_RIGHT_OPERATOR_COMMUTATOR)
        return FASTCD_UNSUPPORTED_ALGORITHM;
    if (n > (SIZE_MAX / sizeof(double) - 64) / 14) return FASTCD_SIZE_OVERFLOW;
    // A product reserves six N-sized planes plus at most four N-sized
    // recursion planes. Composites retain two additional complex CD records.
    // The extra 64 doubles cover small-leaf temporaries conservatively.
    // Lengths count doubles even though each public coefficient occupies two.
    int composite = op == FASTCD_OP_ASSOCIATOR ||
            op == FASTCD_OP_LEFT_OPERATOR_COMMUTATOR ||
            op == FASTCD_OP_RIGHT_OPERATOR_COMMUTATOR;
    *length = (op == FASTCD_OP_MULTIPLY || op == FASTCD_OP_IMAG_CROSS ||
                      op == FASTCD_OP_COMMUTATOR || composite)
            ? (composite ? 14 : 10) * n + 64
            : 0;
    return FASTCD_SUCCESS;
}

// Accumulate the coefficient-bilinear pairing; no scalar is conjugated
static C dot(const C* x, const C* y, size_t n) {
    return dot_range(x, y, 0, n);
}

// Validate the entire call before writing output, then apply one record operation
FastCDStatus fastcd_apply_c128(C* out, const C* x, const C* y, const C* z, size_t n,
        FastCDOperation op, size_t index, FastCDAlgorithm alg,
        const FastCDWorkspaceF64* w) {
    size_t need;
    FastCDStatus s = fastcd_workspace_size_c128(n, op, &need);
    if (s) return s;
    if (alg != FASTCD_ALGORITHM_AUTO && alg != FASTCD_ALGORITHM_DIRECT &&
            alg != FASTCD_ALGORITHM_QUASILINEAR)
        return FASTCD_UNSUPPORTED_ALGORITHM;
    if (!out || !x) return FASTCD_NULL_POINTER;
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
    if ((binary && !y) || (ternary && !z) || (need && (!w || !w->data)))
        return FASTCD_NULL_POINTER;
    if (need && w->length < need) return FASTCD_INSUFFICIENT_WORKSPACE;
    if ((op == FASTCD_OP_BASIS_LEFT || op == FASTCD_OP_BASIS_RIGHT) && index >= n)
        return FASTCD_INVALID_BASIS_INDEX;
    // Reductions write one scalar; scale/divide read one scalar on the right.
    // Other inputs and outputs occupy a complete N-coefficient CD record.
    size_t bytes = n * sizeof(C),
           ob = (op == FASTCD_OP_INNER || op == FASTCD_OP_QUADRATIC_NORM) ? sizeof(C)
                                                                          : bytes;
    size_t yb = (op == FASTCD_OP_SCALE || op == FASTCD_OP_DIVIDE) ? sizeof(C) : bytes;
    // Coordinatewise formulas, reciprocal, and square permit exact aliases;
    // shared scalars are saved before stores. Products require disjoint output.
    int alias = op == FASTCD_OP_ADD || op == FASTCD_OP_SUBTRACT ||
            op == FASTCD_OP_NEGATE || op == FASTCD_OP_SCALE || op == FASTCD_OP_DIVIDE ||
            op == FASTCD_OP_CONJUGATE || op == FASTCD_OP_IMAG_PART ||
            op == FASTCD_OP_RECIPROCAL || op == FASTCD_OP_SQUARE;
    double* work = need ? w->data : NULL;
    // Only the active scratch prefix participates in overlap checks, matching
    // the original real API when callers supply a larger workspace allocation.
    if ((overlap(out, ob, x, bytes) && !(alias && out == x)) ||
            (binary && overlap(out, ob, y, yb) &&
                    !((op == FASTCD_OP_ADD || op == FASTCD_OP_SUBTRACT) && out == y)) ||
            (ternary && overlap(out, ob, z, bytes)) ||
            (binary && op != FASTCD_OP_SCALE && op != FASTCD_OP_DIVIDE && x != y &&
                    overlap(x, bytes, y, yb)) ||
            overlap(work, need * 8, out, ob) || overlap(work, need * 8, x, bytes) ||
            (binary && overlap(work, need * 8, y, yb)) ||
            (ternary && overlap(work, need * 8, z, bytes)))
        return FASTCD_OVERLAPPING_BUFFERS;
    C q, t, alpha, beta;
    if (op == FASTCD_OP_MULTIPLY)
        product(out, x, y, n, alg, work);
    else if (op == FASTCD_OP_ADD || op == FASTCD_OP_SUBTRACT) {
        for (size_t i = 0; i < n; i++)
            out[i] = op == FASTCD_OP_ADD ? add(x[i], y[i]) : sub(x[i], y[i]);
    } else if (op == FASTCD_OP_NEGATE || op == FASTCD_OP_CONJUGATE ||
            op == FASTCD_OP_IMAG_PART) {
        for (size_t i = 0; i < n; i++) {
            if (op == FASTCD_OP_IMAG_PART && i == 0) {
                C a = {0, 0};
                out[i] = a;
            } else
                out[i] = op == FASTCD_OP_NEGATE || (op == FASTCD_OP_CONJUGATE && i)
                        ? neg(x[i])
                        : x[i];
        }
    } else if (op == FASTCD_OP_SCALE || op == FASTCD_OP_DIVIDE) {
        // Copy the scalar and reject a zero divisor before the first store.
        t = y[0];
        if (op == FASTCD_OP_DIVIDE) {
            if (zero(t)) return FASTCD_ZERO_NORM;
            t = inverse(t);
        }
        for (size_t i = 0; i < n; i++) out[i] = mul(x[i], t);
    } else if (op == FASTCD_OP_INNER || op == FASTCD_OP_QUADRATIC_NORM)
        out[0] = dot(x, op == FASTCD_OP_INNER ? y : x, n);
    else if (op == FASTCD_OP_RECIPROCAL) {
        // Q(x), rather than x != 0, controls existence of this reciprocal.
        // Computing Q and its inverse first also makes out == x safe.
        q = dot(x, x, n);
        if (zero(q)) return FASTCD_ZERO_NORM;
        t = inverse(q);
        for (size_t i = 0; i < n; i++) out[i] = mul(i ? neg(x[i]) : x[i], t);
    } else if (op == FASTCD_OP_SQUARE || op == FASTCD_OP_ANTICOMMUTATOR) {
        // Use scalar/imaginary decomposition to avoid a full product.
        // Preserve alpha, beta and the pairing before any in-place stores.
        if (op == FASTCD_OP_SQUARE) y = x;
        alpha = x[0];
        beta = y[0];
        q = sub(mul(alpha, beta), dot(x + 1, y + 1, n - 1));
        if (op == FASTCD_OP_ANTICOMMUTATOR) q = add(q, q);
        for (size_t i = 1; i < n; i++) {
            t = add(mul(alpha, y[i]), mul(beta, x[i]));
            out[i] = op == FASTCD_OP_ANTICOMMUTATOR ? add(t, t) : t;
        }
        out[0] = q;
    } else if (op == FASTCD_OP_IMAG_CROSS || op == FASTCD_OP_COMMUTATOR) {
        // Remove the scalar/imaginary reconstruction terms; [x,y] = 2 X cross Y.
        product(out, x, y, n, alg, work);
        C a = {0, 0};
        out[0] = a;
        for (size_t i = 1; i < n; i++) {
            t = sub(out[i], add(mul(x[0], y[i]), mul(y[0], x[i])));
            out[i] = op == FASTCD_OP_COMMUTATOR ? add(t, t) : t;
        }
    } else if (op == FASTCD_OP_BASIS_LEFT || op == FASTCD_OP_BASIS_RIGHT) {
        // A basis product is a signed permutation of complete coefficients.
        for (size_t i = 0; i < n; i++) {
            FastCDBasisProduct b;
            fastcd_basis_product(&b, n, op == FASTCD_OP_BASIS_LEFT ? index : i,
                    op == FASTCD_OP_BASIS_LEFT ? i : index);
            out[b.index] = b.sign < 0 ? neg(x[i]) : x[i];
        }
    } else {
        // Retain the first parenthesized result in b, build the second in out,
        // and reuse a plus the same product scratch for every intermediate.
        C* a = (C*)work;
        C* b = a + n;
        double* scratch = (double*)(b + n);
        // For associators preserve (xy)z separately from x(yz). Operator
        // commutators likewise retain their exact left/right parentheses;
        // higher CD algebras do not permit reassociation of these terms.
        if (op == FASTCD_OP_ASSOCIATOR) {
            product(a, x, y, n, alg, scratch);
            product(b, a, z, n, alg, scratch);
            product(a, y, z, n, alg, scratch);
            product(out, x, a, n, alg, scratch);
        } else if (op == FASTCD_OP_LEFT_OPERATOR_COMMUTATOR) {
            product(a, y, z, n, alg, scratch);
            product(b, x, a, n, alg, scratch);
            product(a, x, z, n, alg, scratch);
            product(out, y, a, n, alg, scratch);
        } else {
            product(a, z, y, n, alg, scratch);
            product(b, a, x, n, alg, scratch);
            product(a, z, x, n, alg, scratch);
            product(out, a, y, n, alg, scratch);
        }
        for (size_t i = 0; i < n; i++) out[i] = sub(b[i], out[i]);
    }
    return FASTCD_SUCCESS;
}

// Multiply one complex CD record using the requested supported algorithm
FastCDStatus fastcd_mul_with_algorithm_c128(C* out, const C* x, const C* y, size_t n,
        FastCDAlgorithm a, const FastCDWorkspaceF64* w) {
    return fastcd_apply_c128(out, x, y, NULL, n, FASTCD_OP_MULTIPLY, 0, a, w);
}

// Multiply one complex CD record using the default algorithm
FastCDStatus fastcd_mul_c128(
        C* out, const C* x, const C* y, size_t n, const FastCDWorkspaceF64* w) {
    return fastcd_mul_with_algorithm_c128(out, x, y, n, FASTCD_ALGORITHM_AUTO, w);
}

// Return Q(x) = sum_i x_i^2 as one complex coefficient
FastCDStatus fastcd_quadratic_norm_c128(C* out, const C* x, size_t n) {
    return fastcd_apply_c128(out, x, NULL, NULL, n, FASTCD_OP_QUADRATIC_NORM, 0,
            FASTCD_ALGORITHM_AUTO, NULL);
}

// Validate a strided batch once, then reuse its scratch for each product
FastCDStatus fastcd_mul_batch_c128(C* out, size_t os, const C* x, size_t xs, const C* y,
        size_t ys, size_t count, size_t n, FastCDAlgorithm alg,
        const FastCDWorkspaceF64* w) {
    size_t need;
    FastCDStatus s = fastcd_workspace_size_c128(n, FASTCD_OP_MULTIPLY, &need);
    if (s) return s;
    if (alg != FASTCD_ALGORITHM_AUTO && alg != FASTCD_ALGORITHM_DIRECT &&
            alg != FASTCD_ALGORITHM_QUASILINEAR)
        return FASTCD_UNSUPPORTED_ALGORITHM;
    if (!count) return FASTCD_SUCCESS;
    if (count == 1) return fastcd_mul_with_algorithm_c128(out, x, y, n, alg, w);
    if (os < n || (xs && xs < n) || (ys && ys < n)) return FASTCD_INVALID_STRIDE;
    if (!out || !x || !y || !w || !w->data) return FASTCD_NULL_POINTER;
    if (w->length < need) return FASTCD_INSUFFICIENT_WORKSPACE;
    // Strides count complex coefficients. Zero input strides broadcast;
    // compute byte spans only after proving the final record fits size_t.
    size_t limit = SIZE_MAX / sizeof(C);
    if (n > limit || count - 1 > (limit - n) / os ||
            (xs && count - 1 > (limit - n) / xs) ||
            (ys && count - 1 > (limit - n) / ys))
        return FASTCD_SIZE_OVERFLOW;
    size_t ob = ((count - 1) * os + n) * sizeof(C),
           xb = ((count - 1) * xs + n) * sizeof(C),
           yb = ((count - 1) * ys + n) * sizeof(C);
    if (overlap(out, ob, x, xb) || overlap(out, ob, y, yb) ||
            overlap(w->data, need * 8, out, ob) || overlap(w->data, need * 8, x, xb) ||
            overlap(w->data, need * 8, y, yb))
        return FASTCD_OVERLAPPING_BUFFERS;
    // Detect all rejected input overlaps before any result is written.
    for (size_t i = 0; i < count; i++)
        if (x + i * xs != y + i * ys &&
                overlap(x + i * xs, n * sizeof(C), y + i * ys, n * sizeof(C)))
            return FASTCD_OVERLAPPING_BUFFERS;
    for (size_t i = 0; i < count; i++)
        product(out + i * os, x + i * xs, y + i * ys, n, alg, w->data);
    return FASTCD_SUCCESS;
}
