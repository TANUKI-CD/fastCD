/* Complexification of the standard CD tower. Reuses the existing optimized
 * complex alternating recursion, including its 243-operation octonion leaf. */
#include "fastcd/complex.h"
#include <math.h>
#include <stdint.h>
#include <string.h>
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#define FASTCD_QL_COMPLEX_LEAF 8
#define FASTCD_QL_FULL_LEAF 16
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

typedef FastCDComplexF64 C;
static C add(C a, C b) {
    C r = {a.real + b.real, a.imag + b.imag};
    return r;
}
static C sub(C a, C b) {
    C r = {a.real - b.real, a.imag - b.imag};
    return r;
}
static C neg(C a) {
    C r = {-a.real, -a.imag};
    return r;
}
static C mul(C a, C b) {
    C r = {a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real};
    return r;
}
static int zero(C a) { return a.real == 0.0 && a.imag == 0.0; }
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
static int overlap(const void *a, size_t an, const void *b, size_t bn) {
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return an && bn && (x <= y ? y - x < an : x - y < bn);
}
static void direct(C *out, const C *x, const C *y, size_t n, C *work) {
    if (n <= 4) {
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
    C *temp = work;
    C *product = temp + h;
    C *child = product + h;
    direct(out, x, y, h, child);
    temp[0] = y[h];
    for (size_t i = 1; i < h; i++)
        temp[i] = neg(y[h + i]);
    direct(product, temp, x + h, h, child);
    for (size_t i = 0; i < h; i++)
        out[i] = sub(out[i], product[i]);
    direct(out + h, y + h, x, h, child);
    temp[0] = y[0];
    for (size_t i = 1; i < h; i++)
        temp[i] = neg(y[i]);
    direct(product, x + h, temp, h, child);
    for (size_t i = 0; i < h; i++)
        out[h + i] = add(out[h + i], product[i]);
}
static void product(C *out, const C *x, const C *y, size_t n, FastCDAlgorithm alg, double *work) {
    if (alg == FASTCD_ALGORITHM_DIRECT) {
        direct(out, x, y, n, (C *)work);
        return;
    }
    double *xr = work;
    double *xi = xr + n;
    double *yr = xi + n;
    double *yi = yr + n;
    double *rr = yi + n;
    double *ri = rr + n;
    for (size_t i = 0; i < n; i++) {
        xr[i] = x[i].real;
        xi[i] = x[i].imag;
        yr[i] = y[i].real;
        yi[i] = y[i].imag;
    }
    c128_complex_cross(rr, ri, xr, xi, yr, yi, n, ri + n);
    C q = mul(x[0], y[0]);
    for (size_t i = 1; i < n; i++) {
        C cross = {rr[i], ri[i]};
        out[i] = add(cross, add(mul(x[0], y[i]), mul(y[0], x[i])));
        q = sub(q, mul(x[i], y[i]));
    }
    out[0] = q;
}
FastCDStatus fastcd_workspace_size_c128(size_t n, FastCDOperation op, size_t *length) {
    if (!length)
        return FASTCD_NULL_POINTER;
    if (!fastcd_valid_dimension(n))
        return FASTCD_INVALID_DIMENSION;
    if (op < FASTCD_OP_MULTIPLY || op > FASTCD_OP_RIGHT_OPERATOR_COMMUTATOR)
        return FASTCD_UNSUPPORTED_ALGORITHM;
    if (n > (SIZE_MAX / sizeof(double) - 64) / 14)
        return FASTCD_SIZE_OVERFLOW;
    int composite = op == FASTCD_OP_ASSOCIATOR || op == FASTCD_OP_LEFT_OPERATOR_COMMUTATOR ||
                    op == FASTCD_OP_RIGHT_OPERATOR_COMMUTATOR;
    *length = (op == FASTCD_OP_MULTIPLY || op == FASTCD_OP_IMAG_CROSS ||
               op == FASTCD_OP_COMMUTATOR || composite)
                  ? (composite ? 14 : 10) * n + 64
                  : 0;
    return FASTCD_SUCCESS;
}
static C dot(const C *x, const C *y, size_t n) {
    C result = {0, 0};
    for (size_t i = 0; i < n; i++)
        result = add(result, mul(x[i], y[i]));
    return result;
}
FastCDStatus fastcd_apply_c128(C *out, const C *x, const C *y, const C *z, size_t n,
                               FastCDOperation op, size_t index, FastCDAlgorithm alg,
                               const FastCDWorkspaceF64 *w) {
    size_t need;
    FastCDStatus s = fastcd_workspace_size_c128(n, op, &need);
    if (s)
        return s;
    if (alg != FASTCD_ALGORITHM_AUTO && alg != FASTCD_ALGORITHM_DIRECT &&
        alg != FASTCD_ALGORITHM_QUASILINEAR)
        return FASTCD_UNSUPPORTED_ALGORITHM;
    if (!out || !x)
        return FASTCD_NULL_POINTER;
    int binary = op == FASTCD_OP_MULTIPLY || op == FASTCD_OP_ADD || op == FASTCD_OP_SUBTRACT ||
                 op == FASTCD_OP_SCALE || op == FASTCD_OP_DIVIDE || op == FASTCD_OP_INNER ||
                 op == FASTCD_OP_IMAG_CROSS || op == FASTCD_OP_COMMUTATOR ||
                 op == FASTCD_OP_ANTICOMMUTATOR || op == FASTCD_OP_ASSOCIATOR ||
                 op == FASTCD_OP_LEFT_OPERATOR_COMMUTATOR ||
                 op == FASTCD_OP_RIGHT_OPERATOR_COMMUTATOR;
    int ternary = op == FASTCD_OP_ASSOCIATOR || op == FASTCD_OP_LEFT_OPERATOR_COMMUTATOR ||
                  op == FASTCD_OP_RIGHT_OPERATOR_COMMUTATOR;
    if ((binary && !y) || (ternary && !z) || (need && (!w || !w->data)))
        return FASTCD_NULL_POINTER;
    if (need && w->length < need)
        return FASTCD_INSUFFICIENT_WORKSPACE;
    if ((op == FASTCD_OP_BASIS_LEFT || op == FASTCD_OP_BASIS_RIGHT) && index >= n)
        return FASTCD_INVALID_BASIS_INDEX;
    size_t bytes = n * sizeof(C),
           ob = (op == FASTCD_OP_INNER || op == FASTCD_OP_QUADRATIC_NORM) ? sizeof(C) : bytes;
    size_t yb = (op == FASTCD_OP_SCALE || op == FASTCD_OP_DIVIDE) ? sizeof(C) : bytes;
    int alias = op == FASTCD_OP_ADD || op == FASTCD_OP_SUBTRACT || op == FASTCD_OP_NEGATE ||
                op == FASTCD_OP_SCALE || op == FASTCD_OP_DIVIDE || op == FASTCD_OP_CONJUGATE ||
                op == FASTCD_OP_IMAG_PART || op == FASTCD_OP_RECIPROCAL || op == FASTCD_OP_SQUARE;
    double *work = need ? w->data : NULL;
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
    } else if (op == FASTCD_OP_NEGATE || op == FASTCD_OP_CONJUGATE || op == FASTCD_OP_IMAG_PART) {
        for (size_t i = 0; i < n; i++) {
            if (op == FASTCD_OP_IMAG_PART && i == 0) {
                C a = {0, 0};
                out[i] = a;
            } else
                out[i] =
                    op == FASTCD_OP_NEGATE || (op == FASTCD_OP_CONJUGATE && i) ? neg(x[i]) : x[i];
        }
    } else if (op == FASTCD_OP_SCALE || op == FASTCD_OP_DIVIDE) {
        t = y[0];
        if (op == FASTCD_OP_DIVIDE) {
            if (zero(t))
                return FASTCD_ZERO_NORM;
            t = inverse(t);
        }
        for (size_t i = 0; i < n; i++)
            out[i] = mul(x[i], t);
    } else if (op == FASTCD_OP_INNER || op == FASTCD_OP_QUADRATIC_NORM)
        out[0] = dot(x, op == FASTCD_OP_INNER ? y : x, n);
    else if (op == FASTCD_OP_RECIPROCAL) {
        q = dot(x, x, n);
        if (zero(q))
            return FASTCD_ZERO_NORM;
        t = inverse(q);
        for (size_t i = 0; i < n; i++)
            out[i] = mul(i ? neg(x[i]) : x[i], t);
    } else if (op == FASTCD_OP_SQUARE || op == FASTCD_OP_ANTICOMMUTATOR) {
        if (op == FASTCD_OP_SQUARE)
            y = x;
        alpha = x[0];
        beta = y[0];
        q = sub(mul(alpha, beta), dot(x + 1, y + 1, n - 1));
        if (op == FASTCD_OP_ANTICOMMUTATOR)
            q = add(q, q);
        for (size_t i = 1; i < n; i++) {
            t = add(mul(alpha, y[i]), mul(beta, x[i]));
            out[i] = op == FASTCD_OP_ANTICOMMUTATOR ? add(t, t) : t;
        }
        out[0] = q;
    } else if (op == FASTCD_OP_IMAG_CROSS || op == FASTCD_OP_COMMUTATOR) {
        product(out, x, y, n, FASTCD_ALGORITHM_QUASILINEAR, work);
        C a = {0, 0};
        out[0] = a;
        for (size_t i = 1; i < n; i++) {
            t = sub(out[i], add(mul(x[0], y[i]), mul(y[0], x[i])));
            out[i] = op == FASTCD_OP_COMMUTATOR ? add(t, t) : t;
        }
    } else if (op == FASTCD_OP_BASIS_LEFT || op == FASTCD_OP_BASIS_RIGHT) {
        for (size_t i = 0; i < n; i++) {
            FastCDBasisProduct b;
            fastcd_basis_product(&b, n, op == FASTCD_OP_BASIS_LEFT ? index : i,
                                 op == FASTCD_OP_BASIS_LEFT ? i : index);
            out[b.index] = b.sign < 0 ? neg(x[i]) : x[i];
        }
    } else {
        C *a = (C *)work;
        C *b = a + n;
        double *scratch = (double *)(b + n);
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
        for (size_t i = 0; i < n; i++)
            out[i] = sub(b[i], out[i]);
    }
    return FASTCD_SUCCESS;
}
FastCDStatus fastcd_mul_with_algorithm_c128(C *out, const C *x, const C *y, size_t n,
                                            FastCDAlgorithm a, const FastCDWorkspaceF64 *w) {
    return fastcd_apply_c128(out, x, y, NULL, n, FASTCD_OP_MULTIPLY, 0, a, w);
}
FastCDStatus fastcd_mul_c128(C *out, const C *x, const C *y, size_t n,
                             const FastCDWorkspaceF64 *w) {
    return fastcd_mul_with_algorithm_c128(out, x, y, n, FASTCD_ALGORITHM_AUTO, w);
}
FastCDStatus fastcd_quadratic_norm_c128(C *out, const C *x, size_t n) {
    return fastcd_apply_c128(out, x, NULL, NULL, n, FASTCD_OP_QUADRATIC_NORM, 0,
                             FASTCD_ALGORITHM_AUTO, NULL);
}
FastCDStatus fastcd_mul_batch_c128(C *out, size_t os, const C *x, size_t xs, const C *y, size_t ys,
                                   size_t count, size_t n, FastCDAlgorithm alg,
                                   const FastCDWorkspaceF64 *w) {
    size_t need;
    FastCDStatus s = fastcd_workspace_size_c128(n, FASTCD_OP_MULTIPLY, &need);
    if (s)
        return s;
    if (alg != FASTCD_ALGORITHM_AUTO && alg != FASTCD_ALGORITHM_DIRECT &&
        alg != FASTCD_ALGORITHM_QUASILINEAR)
        return FASTCD_UNSUPPORTED_ALGORITHM;
    if (!count)
        return FASTCD_SUCCESS;
    if (count == 1)
        return fastcd_mul_with_algorithm_c128(out, x, y, n, alg, w);
    if (os < n || (xs && xs < n) || (ys && ys < n))
        return FASTCD_INVALID_STRIDE;
    if (!out || !x || !y || !w || !w->data)
        return FASTCD_NULL_POINTER;
    if (w->length < need)
        return FASTCD_INSUFFICIENT_WORKSPACE;
    size_t limit = SIZE_MAX / sizeof(C);
    if (n > limit || count - 1 > (limit - n) / os || (xs && count - 1 > (limit - n) / xs) ||
        (ys && count - 1 > (limit - n) / ys))
        return FASTCD_SIZE_OVERFLOW;
    size_t ob = ((count - 1) * os + n) * sizeof(C), xb = ((count - 1) * xs + n) * sizeof(C),
           yb = ((count - 1) * ys + n) * sizeof(C);
    if (overlap(out, ob, x, xb) || overlap(out, ob, y, yb) || overlap(w->data, need * 8, out, ob) ||
        overlap(w->data, need * 8, x, xb) || overlap(w->data, need * 8, y, yb))
        return FASTCD_OVERLAPPING_BUFFERS;
    for (size_t i = 0; i < count; i++)
        if (x + i * xs != y + i * ys &&
            overlap(x + i * xs, n * sizeof(C), y + i * ys, n * sizeof(C)))
            return FASTCD_OVERLAPPING_BUFFERS;
    for (size_t i = 0; i < count; i++)
        product(out + i * os, x + i * xs, y + i * ys, n, alg, w->data);
    return FASTCD_SUCCESS;
}
