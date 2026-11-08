/*
 * Shared exact quasilinear schedule, included once per fixed polynomial degree
 * with coefficient operations supplied by the including unit. Macros
 * instantiate one recursion, as in src/quasilinear_impl.h, with no run-time
 * function tables inside its arithmetic.
 *
 * FASTCD_FF_WORD is the storage word and FASTCD_FF_STEP the number of words per
 * coefficient. The supplied operations allow exact aliases and keep no
 * operands. Every temporary is a complete field coefficient, and the caller has
 * validated records and workspace, so the recursion performs no public checks.
 * The header has no include guard.
 */

// Recover a pair coefficient (p-q, w-p-q) by the three-product identity
static void FASTCD_FF_NAME(recover)(FASTCD_FF_WORD* real, FASTCD_FF_WORD* imag,
        const FASTCD_FF_WORD* p, const FASTCD_FF_WORD* q, const FASTCD_FF_WORD* w,
        FASTCD_FF_CONTEXT* ctx) {
    FASTCD_FF_SUB(real, p, q, ctx);
    FASTCD_FF_SUB(imag, w, p, ctx);
    FASTCD_FF_SUB(imag, imag, q, ctx);
}

// Evaluate one coordinate of the nonrecursive bilinear map B and its pairing term
static void FASTCD_FF_NAME(nonrecursive)(FASTCD_FF_WORD* first, FASTCD_FF_WORD* scalar,
        FASTCD_FF_WORD* second, const FASTCD_FF_WORD* u, const FASTCD_FF_WORD* v,
        const FASTCD_FF_WORD* s, const FASTCD_FF_WORD* t, const FASTCD_FF_WORD* gamma,
        const FASTCD_FF_WORD* delta, FASTCD_FF_WORD* tmp, FASTCD_FF_CONTEXT* ctx) {
    FASTCD_FF_MUL(first, gamma, t, ctx);
    FASTCD_FF_MUL(tmp, delta, s, ctx);
    FASTCD_FF_SUB(first, first, tmp, ctx);
    FASTCD_FF_MUL(scalar, s, v, ctx);
    FASTCD_FF_MUL(tmp, t, u, ctx);
    FASTCD_FF_SUB(scalar, scalar, tmp, ctx);
    FASTCD_FF_MUL(second, delta, u, ctx);
    FASTCD_FF_MUL(tmp, gamma, v, ctx);
    FASTCD_FF_SUB(second, second, tmp, ctx);
}

// Evaluate the alternating product over K[iota]/(iota^2+1) with two sequential children
static void FASTCD_FF_NAME(pair_cross)(FASTCD_FF_WORD* out_real,
        FASTCD_FF_WORD* out_imag, const FASTCD_FF_WORD* xr, const FASTCD_FF_WORD* xi,
        const FASTCD_FF_WORD* yr, const FASTCD_FF_WORD* yi, size_t n,
        FASTCD_FF_WORD* workspace, FASTCD_FF_CONTEXT* ctx FASTCD_FF_STATS_DECL) {
    /*
     * Planes hold the real and iota coefficients, and CD conjugation fixes
     * iota. Index zero is ignored on input and returned as zero on both output
     * planes. The pair algebra is an arithmetic device even when iota^2+1
     * splits over the field, and odd characteristic supplies the halvings, so no
     * square root of -1 or extra defining polynomial is needed.
     */
    FASTCD_FF_RECORD_NODE(n);
    const size_t h = n / 2, step = FASTCD_FF_STEP;
    if (n <= 2) {
        for (size_t i = 0; i < n; i++) {
            FASTCD_FF_ZERO(out_real + i * step, ctx);
            FASTCD_FF_ZERO(out_imag + i * step, ctx);
        }
        return;
    }
    FASTCD_FF_WORD* nonrecursive = workspace;
    // Above n = 4 the children fill the output halves and share the transformed
    // operands and child scratch, while n = 4 uses the nonrecursive map alone
    if (n > 4) {
        // Leave 32 coefficients for nonrecursive maps, then four transformed
        // half-size planes and one child scratch region used in sequence
        FASTCD_FF_WORD* axr = workspace + 32 * step;
        FASTCD_FF_WORD* axi = axr + h * step;
        FASTCD_FF_WORD* byr = axi + h * step;
        FASTCD_FF_WORD* byi = byr + h * step;
        FASTCD_FF_WORD* child = byi + h * step;
        FASTCD_FF_ZERO(axr, ctx);
        FASTCD_FF_ZERO(axi, ctx);
        FASTCD_FF_ZERO(byr, ctx);
        FASTCD_FF_ZERO(byi, ctx);
        // Form u + iota*s and v + iota*t, keeping the scalar input coordinates zero
        // so both children stay in their imaginary CD subspaces
        for (size_t i = 1; i < h; i++) {
            size_t k = i * step, l = (h + i) * step;
            FASTCD_FF_SUB(axr + k, xr + k, xi + l, ctx);
            FASTCD_FF_ADD(axi + k, xi + k, xr + l, ctx);
            FASTCD_FF_SUB(byr + k, yr + k, yi + l, ctx);
            FASTCD_FF_ADD(byi + k, yi + k, yr + l, ctx);
        }
        // Z_plus fills the lower halves, after which the input planes are free
        FASTCD_FF_NAME(pair_cross)
        (out_real, out_imag, axr, axi, byr, byi, h, child, ctx FASTCD_FF_STATS_ARG);
        // Reuse those planes for u - iota*s and v - iota*t
        for (size_t i = 1; i < h; i++) {
            size_t k = i * step, l = (h + i) * step;
            FASTCD_FF_ADD(axr + k, xr + k, xi + l, ctx);
            FASTCD_FF_SUB(axi + k, xi + k, xr + l, ctx);
            FASTCD_FF_ADD(byr + k, yr + k, yi + l, ctx);
            FASTCD_FF_SUB(byi + k, yi + k, yr + l, ctx);
        }
        // Z_minus fills the upper halves, with the same child scratch
        FASTCD_FF_NAME(pair_cross)
        (out_real + h * step, out_imag + h * step, axr, axi, byr, byi, h, child,
                ctx FASTCD_FF_STATS_ARG);
    } else {
        for (size_t i = 0; i < n; i++) {
            FASTCD_FF_ZERO(out_real + i * step, ctx);
            FASTCD_FF_ZERO(out_imag + i * step, ctx);
        }
    }
#define T(j) (nonrecursive + (j) * step)
    /*
     * Nonrecursive slots 0..2 accumulate the scalar pairings, 3..4 hold the
     * gamma and delta sums, 5..18 hold the p, q, w maps and their shared input
     * sums, and 19..26 hold the recovered map coordinates and the recombined
     * recursive outputs. Slots below 32 never overlap a live transformed input
     * plane, and the p, q, w maps share their three scalar pairing accumulators.
     */
    FASTCD_FF_ZERO(T(0), ctx);
    FASTCD_FF_ZERO(T(1), ctx);
    FASTCD_FF_ZERO(T(2), ctx);
    FASTCD_FF_ADD(T(3), xr + h * step, xi + h * step, ctx);
    FASTCD_FF_ADD(T(4), yr + h * step, yi + h * step, ctx);
    for (size_t i = 1; i < h; i++) {
        size_t k = i * step, l = (h + i) * step;
        // Three real-coefficient evaluations of B recover the pair value, including
        // the scalar pairing, without forming iota in the field
        FASTCD_FF_NAME(nonrecursive)
        (T(5), T(6), T(7), xr + k, yr + k, xr + l, yr + l, xr + h * step, yr + h * step,
                T(8), ctx);
        FASTCD_FF_NAME(nonrecursive)
        (T(9), T(10), T(11), xi + k, yi + k, xi + l, yi + l, xi + h * step,
                yi + h * step, T(8), ctx);
        FASTCD_FF_ADD(T(12), xr + k, xi + k, ctx);
        FASTCD_FF_ADD(T(13), yr + k, yi + k, ctx);
        FASTCD_FF_ADD(T(14), xr + l, xi + l, ctx);
        FASTCD_FF_ADD(T(15), yr + l, yi + l, ctx);
        FASTCD_FF_NAME(nonrecursive)
        (T(16), T(17), T(18), T(12), T(13), T(14), T(15), T(3), T(4), T(8), ctx);
        FASTCD_FF_ADD(T(0), T(0), T(6), ctx);
        FASTCD_FF_ADD(T(1), T(1), T(10), ctx);
        FASTCD_FF_ADD(T(2), T(2), T(17), ctx);
        FASTCD_FF_NAME(recover)(T(19), T(20), T(5), T(9), T(16), ctx);
        FASTCD_FF_NAME(recover)(T(21), T(22), T(7), T(11), T(18), ctx);
        // The first half gets (Z+ + Z-)/2 and the second iota*(Z+ - Z-)/2
        FASTCD_FF_ADD(T(23), out_real + k, out_real + l, ctx);
        FASTCD_FF_ADD(T(24), out_imag + k, out_imag + l, ctx);
        FASTCD_FF_SUB(T(25), out_real + k, out_real + l, ctx);
        FASTCD_FF_SUB(T(26), out_imag + l, out_imag + k, ctx);
        FASTCD_FF_SCALE(T(23), T(23), FASTCD_FF_INVERSE_TWO(ctx), ctx);
        FASTCD_FF_SCALE(T(24), T(24), FASTCD_FF_INVERSE_TWO(ctx), ctx);
        FASTCD_FF_SCALE(T(25), T(25), FASTCD_FF_INVERSE_TWO(ctx), ctx);
        FASTCD_FF_SCALE(T(26), T(26), FASTCD_FF_INVERSE_TWO(ctx), ctx);
        FASTCD_FF_ADD(out_real + k, T(19), T(23), ctx);
        FASTCD_FF_ADD(out_imag + k, T(20), T(24), ctx);
        FASTCD_FF_ADD(out_real + l, T(21), T(26), ctx);
        FASTCD_FF_ADD(out_imag + l, T(22), T(25), ctx);
    }
    // Restore the zero scalar coordinates and the central lambda coordinate
    FASTCD_FF_ZERO(out_real, ctx);
    FASTCD_FF_ZERO(out_imag, ctx);
    FASTCD_FF_NAME(recover)
    (out_real + h * step, out_imag + h * step, T(0), T(1), T(2), ctx);
#undef T
}

// Multiply records: one-call root over K, then scalar--imaginary reconstruction
static void FASTCD_FF_NAME(cd)(FASTCD_FF_WORD* out, const FASTCD_FF_WORD* x,
        const FASTCD_FF_WORD* y, size_t n, FASTCD_FF_WORD* workspace,
        FASTCD_FF_CONTEXT* ctx FASTCD_FF_STATS_DECL) {
    size_t h = n / 2, step = FASTCD_FF_STEP;
    FASTCD_FF_WORD* tmp = workspace;
    if (n >= 8) {
        // At the coefficient-field root one pair child supplies both halves
        FASTCD_FF_NAME(pair_cross)
        (out, out + h * step, x, x + h * step, y, y + h * step, h, workspace,
                ctx FASTCD_FF_STATS_ARG);
    } else {
        for (size_t i = 0; i < n; i++) FASTCD_FF_ZERO(out + i * step, ctx);
    }
    if (n >= 4) {
        // Add the root nonrecursive map, whose second half reverses the child's sign
        FASTCD_FF_ZERO(tmp, ctx);
        for (size_t i = 1; i < h; i++) {
            size_t k = i * step, l = (h + i) * step;
            FASTCD_FF_NAME(nonrecursive)
            (tmp + step, tmp + 2 * step, tmp + 3 * step, x + k, y + k, x + l, y + l,
                    x + h * step, y + h * step, tmp + 4 * step, ctx);
            FASTCD_FF_ADD(tmp, tmp, tmp + 2 * step, ctx);
            FASTCD_FF_ADD(out + k, out + k, tmp + step, ctx);
            FASTCD_FF_SUB(out + l, tmp + 3 * step, out + l, ctx);
        }
        FASTCD_FF_COPY(out + h * step, tmp, ctx);
    }
    // Reconstruct xy from X cross Y, alpha*Y+beta*X and alpha*beta-<X,Y>,
    // keeping every field coordinate canonical
    FASTCD_FF_MUL(tmp, x, y, ctx);
    for (size_t i = 1; i < n; i++) {
        size_t k = i * step;
        FASTCD_FF_MUL(tmp + step, x + k, y + k, ctx);
        FASTCD_FF_SUB(tmp, tmp, tmp + step, ctx);
        FASTCD_FF_MUL(tmp + 2 * step, x, y + k, ctx);
        FASTCD_FF_MUL(tmp + 3 * step, y, x + k, ctx);
        FASTCD_FF_ADD(tmp + 2 * step, tmp + 2 * step, tmp + 3 * step, ctx);
        FASTCD_FF_ADD(out + k, out + k, tmp + 2 * step, ctx);
    }
    FASTCD_FF_COPY(out, tmp, ctx);
}

// Apply an unnormalized Walsh--Hadamard transform in place with canonical butterflies
static void FASTCD_FF_NAME(hadamard)(
        FASTCD_FF_WORD* a, size_t n, FASTCD_FF_WORD* tmp, FASTCD_FF_CONTEXT* ctx) {
    size_t step = FASTCD_FF_STEP;
    for (size_t width = 1; width < n; width *= 2) {
        for (size_t base = 0; base < n; base += 2 * width) {
            for (size_t j = 0; j < width; j++) {
                FASTCD_FF_WORD* u = a + (base + j) * step;
                FASTCD_FF_WORD* v = u + width * step;
                FASTCD_FF_COPY(tmp, u, ctx);
                FASTCD_FF_ADD(u, tmp, v, ctx);
                FASTCD_FF_SUB(v, tmp, v, ctx);
            }
        }
    }
}

// Evaluate the loop component of the private template with one CD scratch region
static void FASTCD_FF_NAME(loop)(FASTCD_FF_WORD* out, const FASTCD_FF_WORD* x,
        const FASTCD_FF_WORD* y, size_t n, FASTCD_FF_WORD* workspace,
        FASTCD_FF_CONTEXT* ctx FASTCD_FF_STATS_DECL) {
    /*
     * The public CD API never calls this loop. It stays beside its Hadamard map
     * so the template instances share one arithmetic definition.
     */
    size_t step = FASTCD_FF_STEP, size = n * step;
    FASTCD_FF_WORD* c = workspace;
    FASTCD_FF_WORD* u = c + size;
    FASTCD_FF_WORD* cp = u + size;
    FASTCD_FF_WORD* up = cp + size;
    FASTCD_FF_WORD* g = up + size;
    FASTCD_FF_WORD* scratch = g + size;
    for (size_t i = 0; i < n; i++) {
        size_t k = i * step;
        FASTCD_FF_ADD(c + k, x + k, x + size + k, ctx);
        FASTCD_FF_SUB(u + k, x + k, x + size + k, ctx);
        FASTCD_FF_ADD(cp + k, y + k, y + size + k, ctx);
        FASTCD_FF_SUB(up + k, y + k, y + size + k, ctx);
    }
    FASTCD_FF_NAME(hadamard)(c, n, scratch, ctx);
    FASTCD_FF_NAME(hadamard)(cp, n, scratch, ctx);
    for (size_t i = 0; i < n; i++)
        FASTCD_FF_MUL(c + i * step, c + i * step, cp + i * step, ctx);
    FASTCD_FF_NAME(hadamard)(c, n, scratch, ctx);
    for (size_t i = 0; i < n; i++)
        FASTCD_FF_SCALE(c + i * step, c + i * step, FASTCD_FF_INVERSE_N(ctx), ctx);
    FASTCD_FF_NAME(cd)(g, u, up, n, scratch, ctx FASTCD_FF_STATS_ARG);
    for (size_t i = 0; i < n; i++) {
        size_t k = i * step;
        FASTCD_FF_ADD(out + k, c + k, g + k, ctx);
        FASTCD_FF_SUB(out + size + k, c + k, g + k, ctx);
        FASTCD_FF_SCALE(out + k, out + k, FASTCD_FF_INVERSE_TWO(ctx), ctx);
        FASTCD_FF_SCALE(
                out + size + k, out + size + k, FASTCD_FF_INVERSE_TWO(ctx), ctx);
    }
}
