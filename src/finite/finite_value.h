/* Register-value specialization of the exact CD bilinear schedule.
 * Word and Context are supplied by the arithmetic backend. Value operations
 * VADD, VSUB, VMUL, VDIFF, VHALF and VNORMALIZE use the local ctx pointer.
 * No field representation, modulus or machine instruction is assumed here.
 *
 * As in src/quasilinear_impl.h, the local map and recursive children are
 * separate stages. Values live in scalar registers or independent SIMD
 * lanes; only transformed operand planes occupy recursive scratch.
 * Index zero of each alternating input plane is ignored. The auxiliary
 * pair unit iota satisfies iota^2=-1 and is fixed by the CD involution.
 * This template intentionally has no include guard. */

typedef struct {
    Word first, scalar, second;
} FASTCD_FF_NAME(ValueLocal);

// Evaluate B=(gamma*t-delta*s, s*v-t*u, delta*u-gamma*v) in value arithmetic
static inline FASTCD_FF_ALWAYS_INLINE FASTCD_FF_NAME(ValueLocal)
        FASTCD_FF_NAME(vlocal)(Word u, Word v, Word s, Word t, Word gamma, Word delta,
                const Context* restrict ctx) {
    (void)ctx;
    FASTCD_FF_NAME(ValueLocal)
    r = {VDIFF(gamma, t, delta, s), VDIFF(s, v, t, u), VDIFF(delta, u, gamma, v)};
    return r;
}

/* Two children share scratch and write to the lower/upper output halves.
 * Reconstruction uses only addition, subtraction and modular halving, so
 * the recurrence works whether or not -1 is a square in the field. */
// Evaluate the auxiliary-pair alternating product on n coefficient values
static FASTCD_FF_UNUSED void FASTCD_FF_NAME(pair_cross)(Word* restrict real,
        Word* restrict imag, const Word* xr, const Word* xi, const Word* yr,
        const Word* yi, size_t n, Word* restrict workspace,
        const Context* restrict ctx) {
    size_t h = n / 2;
    // Alternation vanishes at dimensions one and two; dimension four uses
    // only the local map because both recursive children would vanish.
    if (n <= 2) {
        for (size_t i = 0; i < n; i++) real[i] = imag[i] = VZERO;
        return;
    }
    if (n > 4) {
        // Four half-size input planes precede one reusable child workspace.
        Word* ar = workspace;
        Word* ai = ar + h;
        Word* br = ai + h;
        Word* bi = br + h;
        Word* child = bi + h;
        ar[0] = ai[0] = br[0] = bi[0] = VZERO;
        // Z_plus = Alt(u+iota*s, v+iota*t), with zero scalar input entries.
        for (size_t i = 1; i < h; i++) {
            ar[i] = VSUB(xr[i], xi[h + i]);
            ai[i] = VADD(xi[i], xr[h + i]);
            br[i] = VSUB(yr[i], yi[h + i]);
            bi[i] = VADD(yi[i], yr[h + i]);
        }
        FASTCD_FF_NAME(pair_cross)(real, imag, ar, ai, br, bi, h, child, ctx);
        // Reuse the input planes and child scratch for Z_minus.
        for (size_t i = 1; i < h; i++) {
            ar[i] = VADD(xr[i], xi[h + i]);
            ai[i] = VSUB(xi[i], xr[h + i]);
            br[i] = VADD(yr[i], yi[h + i]);
            bi[i] = VSUB(yi[i], yr[h + i]);
        }
        FASTCD_FF_NAME(pair_cross)(real + h, imag + h, ar, ai, br, bi, h, child, ctx);
    } else
        for (size_t i = 0; i < n; i++) real[i] = imag[i] = VZERO;
    // Extract gamma, delta and their component sums for the three B maps.
    // For p=B(real), q=B(imag), w=B(real+imag), the pair result is
    // (p-q) + iota*((w-p)-q). This includes the scalar pairing accumulator,
    // rather than reconstructing scalar products in a separate pass.
    Word gp = xr[h], gq = xi[h], dp = yr[h], dq = yi[h];
    Word gw = VADD(gp, gq), dw = VADD(dp, dq);
    Word ap = VZERO, aq = VZERO, aw = VZERO;
    for (size_t i = 1; i < h; i++) {
        Word ur = xr[i], ui = xi[i], vr = yr[i], vi = yi[i];
        Word sr = xr[h + i], si = xi[h + i], tr = yr[h + i], ti = yi[h + i];
        // Recover B over the pair algebra from B(real), B(imag), and B(sum).
        // Accumulate scalar pairings before recovering the central coordinate.
        FASTCD_FF_NAME(ValueLocal)
        p = FASTCD_FF_NAME(vlocal)(ur, vr, sr, tr, gp, dp, ctx);
        FASTCD_FF_NAME(ValueLocal)
        q = FASTCD_FF_NAME(vlocal)(ui, vi, si, ti, gq, dq, ctx);
        FASTCD_FF_NAME(ValueLocal)
        w = FASTCD_FF_NAME(vlocal)(
                VADD(ur, ui), VADD(vr, vi), VADD(sr, si), VADD(tr, ti), gw, dw, ctx);
        ap = VADD(ap, p.scalar);
        aq = VADD(aq, q.scalar);
        aw = VADD(aw, w.scalar);
        Word fr = VSUB(p.first, q.first), fi = VSUB(VSUB(w.first, p.first), q.first);
        Word gr = VSUB(p.second, q.second),
             gi = VSUB(VSUB(w.second, p.second), q.second);
        // Lower half adds (Z_plus+Z_minus)/2; upper adds
        // iota*(Z_plus-Z_minus)/2, with iota*(a+iota*b)=-b+iota*a.
        // VHALF acts in the odd-characteristic coefficient field, including
        // Montgomery or packed encodings supplied by the owning backend.
        Word pr = real[i], pi = imag[i], mr = real[h + i], mi = imag[h + i];
        real[i] = VADD(fr, VHALF(VADD(pr, mr)));
        imag[i] = VADD(fi, VHALF(VADD(pi, mi)));
        real[h + i] = VADD(gr, VHALF(VSUB(mi, pi)));
        imag[h + i] = VADD(gi, VHALF(VSUB(pr, mr)));
    }
    real[0] = imag[0] = VZERO;
    real[h] = VSUB(ap, aq);
    imag[h] = VSUB(VSUB(aw, ap), aq);
}

// Evaluate the coefficient-field root, then restore scalar/imaginary terms
static FASTCD_FF_UNUSED void FASTCD_FF_NAME(cd)(Word* restrict out, const Word* x,
        const Word* y, size_t n, Word* restrict workspace,
        const Context* restrict ctx) {
    size_t h = n / 2;
    // One pair child gives the root's two output halves. For smaller
    // dimensions those children are zero and only root reconstruction remains.
    if (n >= 8)
        FASTCD_FF_NAME(pair_cross)(out, out + h, x, x + h, y, y + h, h, workspace, ctx);
    else
        for (size_t i = 0; i < n; i++) out[i] = VZERO;
    if (n >= 4) {
        Word g = x[h], d = y[h], sum = VZERO;
        for (size_t i = 1; i < h; i++) {
            FASTCD_FF_NAME(ValueLocal)
            v = FASTCD_FF_NAME(vlocal)(x[i], y[i], x[h + i], y[h + i], g, d, ctx);
            sum = VADD(sum, v.scalar);
            out[i] = VADD(out[i], v.first);
            out[h + i] = VSUB(v.second, out[h + i]);
        }
        out[h] = sum;
    }
    // Complete xy=(alpha*beta-<X,Y>) + alpha*Y+beta*X + X cross Y.
    // All input coefficients remain live until reconstruction ends; callers
    // provide disjoint output, as for the original real multiplication kernel.
    Word dot = VMUL(x[0], y[0]);
    for (size_t i = 1; i < n; i++) {
        dot = VSUB(dot, VMUL(x[i], y[i]));
        out[i] = VADD(out[i], VADD(VMUL(x[0], y[i]), VMUL(y[0], x[i])));
    }
    out[0] = dot;
}

// Apply the private loop component's unnormalized Walsh--Hadamard transform
static FASTCD_FF_UNUSED void FASTCD_FF_NAME(hadamard)(
        Word* a, size_t n, const Context* restrict ctx) {
    (void)ctx;
    size_t start = 1;
    // Fuse the first three butterfly levels to keep each eight-value tile
    // in registers before continuing with larger widths.
    if (n >= 8) {
        for (size_t base = 0; base < n; base += 8) {
            Word a0 = a[base], a1 = a[base + 1], a2 = a[base + 2], a3 = a[base + 3];
            Word a4 = a[base + 4], a5 = a[base + 5], a6 = a[base + 6], a7 = a[base + 7];
            Word b0 = VADD(a0, a1), b1 = VSUB(a0, a1), b2 = VADD(a2, a3),
                 b3 = VSUB(a2, a3);
            Word b4 = VADD(a4, a5), b5 = VSUB(a4, a5), b6 = VADD(a6, a7),
                 b7 = VSUB(a6, a7);
            Word c0 = VADD(b0, b2), c1 = VADD(b1, b3), c2 = VSUB(b0, b2),
                 c3 = VSUB(b1, b3);
            Word c4 = VADD(b4, b6), c5 = VADD(b5, b7), c6 = VSUB(b4, b6),
                 c7 = VSUB(b5, b7);
            a[base] = VADD(c0, c4);
            a[base + 1] = VADD(c1, c5);
            a[base + 2] = VADD(c2, c6);
            a[base + 3] = VADD(c3, c7);
            a[base + 4] = VSUB(c0, c4);
            a[base + 5] = VSUB(c1, c5);
            a[base + 6] = VSUB(c2, c6);
            a[base + 7] = VSUB(c3, c7);
        }
        start = 8;
    }
    for (size_t width = start; width < n; width *= 2)
        for (size_t base = 0; base < n; base += 2 * width)
            for (size_t j = 0; j < width; j++) {
                Word u = a[base + j], v = a[base + width + j];
                a[base + j] = VADD(u, v);
                a[base + width + j] = VSUB(u, v);
            }
}

/* Private companion retained by the shared template; public fastCD products
 * call cd above, not this loop component. Its buffers and normalizations
 * belong to a different wrapper and do not change the CD record layout. */
// Combine Hadamard and signed CD components for the private loop schedule
static FASTCD_FF_UNUSED void FASTCD_FF_NAME(loop)(Word* restrict out, const Word* x,
        const Word* y, size_t n, Word* restrict workspace,
        const Context* restrict ctx) {
    Word* c = workspace;
    Word* u = c + n;
    Word* cp = u + n;
    Word* up = cp + n;
    Word* g = up + n;
    Word* scratch = g + n;
    for (size_t i = 0; i < n; i++) {
        c[i] = VADD(x[i], x[n + i]);
        u[i] = VSUB(x[i], x[n + i]);
        cp[i] = VADD(y[i], y[n + i]);
        up[i] = VSUB(y[i], y[n + i]);
    }
    FASTCD_FF_NAME(hadamard)(c, n, ctx);
    FASTCD_FF_NAME(hadamard)(cp, n, ctx);
    for (size_t i = 0; i < n; i++) c[i] = VMUL(c[i], cp[i]);
    FASTCD_FF_NAME(hadamard)(c, n, ctx);
    for (size_t i = 0; i < n; i++) c[i] = VNORMALIZE(c[i]);
    FASTCD_FF_NAME(cd)(g, u, up, n, scratch, ctx);
    for (size_t i = 0; i < n; i++) {
        out[i] = VHALF(VADD(c[i], g[i]));
        out[n + i] = VHALF(VSUB(c[i], g[i]));
    }
}
