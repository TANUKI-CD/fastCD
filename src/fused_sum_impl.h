/*
 * Fused sums of products, shared by every kernel. The including translation
 * unit defines the eight-lane vector type FS_V, its operations and FS_NAME,
 * and this file instantiates the sum for that vector type. Lane l of a
 * vector holds element 8g + l of a group g of eight elements.
 *
 * Operations required: FS_ZERO, FS_LOAD(p), FS_STORE(p, v), FS_SET1(s),
 * FS_ADD, FS_SUB, FS_MUL, FS_FMA(a, b, c) = a b + c, FS_FNMA(a, b, c) =
 * c - a b, both with one rounding wherever the target can fuse them, and
 * FS_LANES(v, d), which writes the eight lanes to d. A unit that defines FS_TRANSPOSE also provides
 * FS_NAME(transpose).
 *
 * Schedule. An operand block holds the 2N coordinate rows of at most
 * FASTCD_FUSED_CHUNK elements. With D the distance between the real and the
 * imaginary plane of an operand (half its rows), the four planes of the
 * complex operands of every node of the quasilinear recursion lie at a base,
 * base + D, base + 2D and base + 3D. Each loop runs "coordinate outer,
 * element inner" with its partial sums in registers and adds them to the
 * accumulators in memory when the coordinate is finished. A node overwrites
 * the rows of (u, s) and (v, t) with the operands of its two children,
 * (u + iota s, v + iota t) and (u - iota s, v - iota t). The output additions
 * of the recursion are applied to the accumulators in FS_NAME(finish).
 */

#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "fused_sum.h"

#if defined(__GNUC__) || defined(__clang__)
#  define FS_NOINLINE __attribute__((noinline))
#else
#  define FS_NOINLINE
#endif

// Add a vector to an accumulator in memory
#define FS_ACCUMULATE(p, v) FS_STORE(p, FS_ADD(FS_LOAD(p), v))

#include "generated/fused_sum_kernels.h"

/* ---------- Complex kernel ---------- */

/*
 * The arithmetic of one node coordinate on loaded operands: u, s (x) and v, t
 * (y) with real and imaginary parts, and gamma, delta with their sums g_s,
 * d_s. The children go to the rows at wr and wq of a block whose planes are
 * WD apart, and the three real nonrecursive maps (real parts, imaginary
 * parts, sums) to the partial sums f, g, l.
 */
#define FS_NODE_CORE(wr, wq, WD) \
    FS_STORE(wr, FS_SUB(ur, si)); FS_STORE(wr + WD, FS_ADD(ui, sr)); \
    FS_STORE(wr + 2 * WD, FS_SUB(vr, ti)); FS_STORE(wr + 3 * WD, FS_ADD(vi, tr)); \
    FS_STORE(wq, FS_ADD(ur, si)); FS_STORE(wq + WD, FS_SUB(ui, sr)); \
    FS_STORE(wq + 2 * WD, FS_ADD(vr, ti)); FS_STORE(wq + 3 * WD, FS_SUB(vi, tr)); \
    fD = FS_FMA(g_r, tr, fD); fD = FS_FNMA(d_r, sr, fD); \
    gD = FS_FMA(d_r, ur, gD); gD = FS_FNMA(g_r, vr, gD); \
    lD = FS_FMA(sr, vr, lD); lD = FS_FNMA(tr, ur, lD); \
    fO = FS_FMA(g_i, ti, fO); fO = FS_FNMA(d_i, si, fO); \
    gO = FS_FMA(d_i, ui, gO); gO = FS_FNMA(g_i, vi, gO); \
    lO = FS_FMA(si, vi, lO); lO = FS_FNMA(ti, ui, lO); \
    { \
        const FS_V us = FS_ADD(ur, ui), ss = FS_ADD(sr, si); \
        const FS_V vs = FS_ADD(vr, vi), ts = FS_ADD(tr, ti); \
        fW = FS_FMA(g_s, ts, fW); fW = FS_FNMA(d_s, ss, fW); \
        gW = FS_FMA(d_s, us, gW); gW = FS_FNMA(g_s, vs, gW); \
        lW = FS_FMA(ss, vs, lW); lW = FS_FNMA(ts, us, lW); \
    }

// Add the six partial sums of node coordinate i to its accumulators
#define FS_NODE_FLUSH(acc, i) \
    { \
        double* const a_ = (acc) + 8 * (3 + 6 * (i)); \
        FS_ACCUMULATE(a_, fD); FS_ACCUMULATE(a_ + 8, gD); \
        FS_ACCUMULATE(a_ + 16, fO); FS_ACCUMULATE(a_ + 24, gO); \
        FS_ACCUMULATE(a_ + 32, fW); FS_ACCUMULATE(a_ + 40, gW); \
    }

// Accumulate the complex nonrecursive map of dimension 4, which has no children
static FS_NOINLINE void FS_NAME(leaf4)(double* restrict acc, const double* xr,
        const double* xi, const double* yr, const double* yi, size_t pitch,
        size_t length) {
    FS_V fD = FS_ZERO, gD = FS_ZERO, lD = FS_ZERO, fO = FS_ZERO, gO = FS_ZERO;
    FS_V lO = FS_ZERO, fW = FS_ZERO, gW = FS_ZERO, lW = FS_ZERO;

    // Rows 1, 2, 3 of a plane hold u, gamma, s
    for (size_t j = 0; j < length; j += 8) {
        const FS_V ur = FS_LOAD(xr + pitch + j), ui = FS_LOAD(xi + pitch + j);
        const FS_V g_r = FS_LOAD(xr + 2 * pitch + j);
        const FS_V g_i = FS_LOAD(xi + 2 * pitch + j);
        const FS_V sr = FS_LOAD(xr + 3 * pitch + j);
        const FS_V si = FS_LOAD(xi + 3 * pitch + j);
        const FS_V vr = FS_LOAD(yr + pitch + j), vi = FS_LOAD(yi + pitch + j);
        const FS_V d_r = FS_LOAD(yr + 2 * pitch + j);
        const FS_V d_i = FS_LOAD(yi + 2 * pitch + j);
        const FS_V tr = FS_LOAD(yr + 3 * pitch + j);
        const FS_V ti = FS_LOAD(yi + 3 * pitch + j);
        const FS_V us = FS_ADD(ur, ui), g_s = FS_ADD(g_r, g_i), ss = FS_ADD(sr, si);
        const FS_V vs = FS_ADD(vr, vi), d_s = FS_ADD(d_r, d_i), ts = FS_ADD(tr, ti);
        fD = FS_FMA(g_r, tr, fD); fD = FS_FNMA(d_r, sr, fD);
        gD = FS_FMA(d_r, ur, gD); gD = FS_FNMA(g_r, vr, gD);
        lD = FS_FMA(sr, vr, lD); lD = FS_FNMA(tr, ur, lD);
        fO = FS_FMA(g_i, ti, fO); fO = FS_FNMA(d_i, si, fO);
        gO = FS_FMA(d_i, ui, gO); gO = FS_FNMA(g_i, vi, gO);
        lO = FS_FMA(si, vi, lO); lO = FS_FNMA(ti, ui, lO);
        fW = FS_FMA(g_s, ts, fW); fW = FS_FNMA(d_s, ss, fW);
        gW = FS_FMA(d_s, us, gW); gW = FS_FNMA(g_s, vs, gW);
        lW = FS_FMA(ss, vs, lW); lW = FS_FNMA(ts, us, lW);
    }
    FS_ACCUMULATE(acc, lD); FS_ACCUMULATE(acc + 8, lO); FS_ACCUMULATE(acc + 16, lW);
    FS_ACCUMULATE(acc + 24, fD); FS_ACCUMULATE(acc + 32, gD);
    FS_ACCUMULATE(acc + 40, fO); FS_ACCUMULATE(acc + 48, gO);
    FS_ACCUMULATE(acc + 56, fW); FS_ACCUMULATE(acc + 64, gW);
}

static void FS_NAME(subtree)(size_t K, double* b, size_t D, size_t pitch,
        size_t length, double* restrict acc);

/* One node of dimension K >= 16 on an operand block, in place. acc holds the
 * accumulators of the node and, after them, those of its two subtrees. */
static FS_NOINLINE void FS_NAME(node)(size_t K, double* b, size_t D,
        size_t pitch, size_t length, double* restrict acc) {
    const size_t half = K / 2, E = half * pitch;
    FS_V lD = FS_ZERO, lO = FS_ZERO, lW = FS_ZERO;

    for (size_t i = 1; i < half; i++) {
        double* r = b + i * pitch;
        double* q = r + E;
        const double* w = b + E;
        const double* const end = r + length;
        FS_V fD = FS_ZERO, gD = FS_ZERO, fO = FS_ZERO, gO = FS_ZERO;
        FS_V fW = FS_ZERO, gW = FS_ZERO;
        for (; r < end; r += 8, q += 8, w += 8) {
            const FS_V ur = FS_LOAD(r), ui = FS_LOAD(r + D);
            const FS_V vr = FS_LOAD(r + 2 * D), vi = FS_LOAD(r + 3 * D);
            const FS_V sr = FS_LOAD(q), si = FS_LOAD(q + D);
            const FS_V tr = FS_LOAD(q + 2 * D), ti = FS_LOAD(q + 3 * D);
            const FS_V g_r = FS_LOAD(w), g_i = FS_LOAD(w + D);
            const FS_V d_r = FS_LOAD(w + 2 * D), d_i = FS_LOAD(w + 3 * D);
            const FS_V g_s = FS_ADD(g_r, g_i), d_s = FS_ADD(d_r, d_i);
            FS_NODE_CORE(r, q, D)
        }
        FS_NODE_FLUSH(acc, i)
    }
    FS_ACCUMULATE(acc, lD); FS_ACCUMULATE(acc + 8, lO); FS_ACCUMULATE(acc + 16, lW);
    acc += 8 * (3 + 6 * half);
    FS_NAME(subtree)(half, b, D, pitch, length, acc);
    FS_NAME(subtree)(half, b + E, D, pitch, length,
            acc + 8 * fastcd_sum_node_vectors(half));
}

// Run the complex kernel of dimension K on the block rows that start at b
static void FS_NAME(subtree)(size_t K, double* b, size_t D, size_t pitch,
        size_t length, double* restrict acc) {
    if (K <= 2) return;
    if (K == 4) {
        FS_NAME(leaf4)(acc, b, b + D, b + 2 * D, b + 3 * D, pitch, length);
    } else if (K == 8) {
        FS_NAME(leaf8)(acc, b, b + D, b + 2 * D, b + 3 * D, pitch, length);
    } else {
        FS_NAME(node)(K, b, D, pitch, length, acc);
    }
}

/* ---------- Real root ---------- */

/*
 * One root coordinate: rows u, s of x and v, t of y, with the scalars
 * al = x[0], ga = x[N/2], be = y[0], de = y[N/2]. a and b receive the output
 * coordinates i and N/2 + i without their recursive part, dd the pairing
 * <X, Y> and ll the scalar of the nonrecursive map.
 */
#define FS_ROOT_STEP(u, s, v, t, a, b, dd, ll) \
    a = FS_FMA(al, v, a); a = FS_FMA(be, u, a); \
    a = FS_FMA(ga, t, a); a = FS_FNMA(de, s, a); \
    b = FS_FMA(al, t, b); b = FS_FMA(be, s, b); \
    b = FS_FMA(de, u, b); b = FS_FNMA(ga, v, b); \
    dd = FS_FMA(u, v, dd); dd = FS_FMA(s, t, dd); \
    ll = FS_FMA(s, v, ll); ll = FS_FNMA(t, u, ll);

// The same with every dependency chain two products long
#define FS_ROOT_STEP2(u, s, v, t, a, a_, b, b_, dd, ll) \
    a = FS_FMA(al, v, a); a = FS_FMA(be, u, a); \
    a_ = FS_FMA(ga, t, a_); a_ = FS_FNMA(de, s, a_); \
    b = FS_FMA(al, t, b); b = FS_FMA(be, s, b); \
    b_ = FS_FMA(de, u, b_); b_ = FS_FNMA(ga, v, b_); \
    dd = FS_FMA(u, v, dd); dd = FS_FMA(s, t, dd); \
    ll = FS_FMA(s, v, ll); ll = FS_FNMA(t, u, ll);

#define FS_ROOT_SCALARS \
    const FS_V al = FS_LOAD(zx), ga = FS_LOAD(zx + D); \
    const FS_V be = FS_LOAD(zy), de = FS_LOAD(zy + D);

/* The four products of the scalar coordinates and, when h > 1, the root
 * coordinate c. R holds the N root accumulators and h = N / 2. */
static FS_NOINLINE void FS_NAME(root_first)(double* restrict R, const double* x,
        const double* y, size_t D, size_t pitch, size_t h, size_t c,
        size_t length) {
    const double* zx = x;
    const double* zy = y;
    const double* const end = x + length;
    FS_V s0 = FS_ZERO, s1 = FS_ZERO, sh0 = FS_ZERO, sh1 = FS_ZERO;

    if (h == 1) {
        for (; zx < end; zx += 8, zy += 8) {
            FS_ROOT_SCALARS
            s0 = FS_FMA(al, be, s0); s1 = FS_FNMA(ga, de, s1);
            sh0 = FS_FMA(al, de, sh0); sh1 = FS_FMA(be, ga, sh1);
        }
        FS_ACCUMULATE(R, FS_ADD(s0, s1));
        FS_ACCUMULATE(R + 8, FS_ADD(sh0, sh1));
        return;
    }
    {
        const double* rx = x + c * pitch;
        const double* ry = y + c * pitch;
        FS_V a1 = FS_ZERO, a2 = FS_ZERO, b1 = FS_ZERO, b2 = FS_ZERO;
        FS_V dd = FS_ZERO, ll = FS_ZERO;
        for (; zx < end; zx += 8, zy += 8, rx += 8, ry += 8) {
            FS_ROOT_SCALARS
            const FS_V u = FS_LOAD(rx), s = FS_LOAD(rx + D);
            const FS_V v = FS_LOAD(ry), t = FS_LOAD(ry + D);
            s0 = FS_FMA(al, be, s0); s1 = FS_FNMA(ga, de, s1);
            sh0 = FS_FMA(al, de, sh0); sh1 = FS_FMA(be, ga, sh1);
            FS_ROOT_STEP2(u, s, v, t, a1, a2, b1, b2, dd, ll)
        }
        FS_ACCUMULATE(R + 8 * c, FS_ADD(a1, a2));
        FS_ACCUMULATE(R + 8 * (h + c), FS_ADD(b1, b2));
        FS_ACCUMULATE(R, FS_SUB(FS_ADD(s0, s1), dd));
        FS_ACCUMULATE(R + 8 * h, FS_ADD(FS_ADD(sh0, sh1), ll));
    }
}

// The root coordinates c and c + 1
static FS_NOINLINE void FS_NAME(root_pair)(double* restrict R, const double* x,
        const double* y, size_t D, size_t pitch, size_t h, size_t c,
        size_t length) {
    const double* zx = x;
    const double* zy = y;
    const double* rx = x + c * pitch;
    const double* ry = y + c * pitch;
    const double* const end = x + length;
    FS_V a1 = FS_ZERO, b1 = FS_ZERO, a2 = FS_ZERO, b2 = FS_ZERO;
    FS_V a3 = FS_ZERO, b3 = FS_ZERO, a4 = FS_ZERO, b4 = FS_ZERO;
    FS_V dd = FS_ZERO, ll = FS_ZERO, d2 = FS_ZERO, l2 = FS_ZERO;

    for (; zx < end; zx += 8, zy += 8, rx += 8, ry += 8) {
        FS_ROOT_SCALARS
        const FS_V u = FS_LOAD(rx), s = FS_LOAD(rx + D);
        const FS_V v = FS_LOAD(ry), t = FS_LOAD(ry + D);
        const FS_V u2 = FS_LOAD(rx + pitch), s2 = FS_LOAD(rx + pitch + D);
        const FS_V v2 = FS_LOAD(ry + pitch), t2 = FS_LOAD(ry + pitch + D);
        FS_ROOT_STEP2(u, s, v, t, a1, a3, b1, b3, dd, ll)
        FS_ROOT_STEP2(u2, s2, v2, t2, a2, a4, b2, b4, d2, l2)
    }
    FS_ACCUMULATE(R + 8 * c, FS_ADD(a1, a3));
    FS_ACCUMULATE(R + 8 * (h + c), FS_ADD(b1, b3));
    FS_ACCUMULATE(R + 8 * (c + 1), FS_ADD(a2, a4));
    FS_ACCUMULATE(R + 8 * (h + c + 1), FS_ADD(b2, b4));
    FS_STORE(R, FS_SUB(FS_LOAD(R), FS_ADD(dd, d2)));
    FS_ACCUMULATE(R + 8 * h, FS_ADD(ll, l2));
}

/*
 * The root and the top complex node in one pass, for N >= 32. Coordinate i of
 * the node reads the rows of the root coordinates i and h/2 + i, so one loop
 * accumulates both root coordinates and the node's nonrecursive map and
 * writes the node's children to dst, a block with rows dpitch apart. dst may
 * hold the operands themselves.
 */
static FS_NOINLINE void FS_NAME(top)(double* restrict R, double* restrict acc,
        const double* x, const double* y, size_t D, size_t pitch, size_t h,
        size_t length, double* dst, size_t dpitch) {
    const size_t half = h / 2, E = half * pitch, WD = h * dpitch;
    FS_V lD = FS_ZERO, lO = FS_ZERO, lW = FS_ZERO, dd = FS_ZERO, ll = FS_ZERO;

    for (size_t i = 1; i < half; i++) {
        const double* rx = x + i * pitch;
        const double* ry = y + i * pitch;
        const double* zx = x;
        const double* zy = y;
        double* wr = dst + i * dpitch;
        double* wq = wr + half * dpitch;
        const double* const end = rx + length;
        FS_V fD = FS_ZERO, gD = FS_ZERO, fO = FS_ZERO, gO = FS_ZERO;
        FS_V fW = FS_ZERO, gW = FS_ZERO;
        FS_V a1 = FS_ZERO, b1 = FS_ZERO, a2 = FS_ZERO, b2 = FS_ZERO;
        for (; rx < end; rx += 8, ry += 8, zx += 8, zy += 8, wr += 8, wq += 8) {
            const FS_V ur = FS_LOAD(rx), ui = FS_LOAD(rx + D);
            const FS_V sr = FS_LOAD(rx + E), si = FS_LOAD(rx + E + D);
            const FS_V vr = FS_LOAD(ry), vi = FS_LOAD(ry + D);
            const FS_V tr = FS_LOAD(ry + E), ti = FS_LOAD(ry + E + D);
            {
                FS_ROOT_SCALARS
                FS_ROOT_STEP(ur, ui, vr, vi, a1, b1, dd, ll)
                FS_ROOT_STEP(sr, si, tr, ti, a2, b2, dd, ll)
            }
            {
                // gamma and delta of the node are row h/2 of each plane
                const FS_V g_r = FS_LOAD(zx + E), g_i = FS_LOAD(zx + E + D);
                const FS_V d_r = FS_LOAD(zy + E), d_i = FS_LOAD(zy + E + D);
                const FS_V g_s = FS_ADD(g_r, g_i), d_s = FS_ADD(d_r, d_i);
                FS_NODE_CORE(wr, wq, WD)
            }
        }
        FS_ACCUMULATE(R + 8 * i, a1);
        FS_ACCUMULATE(R + 8 * (h + i), b1);
        FS_ACCUMULATE(R + 8 * (half + i), a2);
        FS_ACCUMULATE(R + 8 * (h + half + i), b2);
        FS_NODE_FLUSH(acc, i)
    }
    FS_STORE(R, FS_SUB(FS_LOAD(R), dd));
    FS_ACCUMULATE(R + 8 * h, ll);
    FS_ACCUMULATE(acc, lD); FS_ACCUMULATE(acc + 8, lO); FS_ACCUMULATE(acc + 16, lW);
}

/* Accumulate `length` elements of a quasilinear sum. x and y are 2 * (N / 2)
 * rows each, pitch apart. For N >= 32 the block receives the children of the
 * top node and is then consumed; x and y may be the two halves of the block. */
static void FS_NAME(quasilinear_chunk)(double* restrict acc, const double* x,
        const double* y, size_t pitch, size_t length, size_t N, double* block) {
    const size_t h = N / 2, D = h * pitch;
    double* const nodes = acc + 8 * N;

    if (N >= 32) {
        const size_t half = h / 2, WD = h * FASTCD_FUSED_PITCH;
        double* const children = nodes + 8 * (3 + 6 * half);
        FS_NAME(root_first)(acc, x, y, D, pitch, h, half, length);
        FS_NAME(top)(acc, nodes, x, y, D, pitch, h, length, block,
                FASTCD_FUSED_PITCH);
        FS_NAME(subtree)(half, block, WD, FASTCD_FUSED_PITCH, length, children);
        FS_NAME(subtree)(half, block + half * FASTCD_FUSED_PITCH, WD,
                FASTCD_FUSED_PITCH, length,
                children + 8 * fastcd_sum_node_vectors(half));
        return;
    }
    FS_NAME(root_first)(acc, x, y, D, pitch, h, 1, length);
    for (size_t i = 2; i < h; i += 2) {
        FS_NAME(root_pair)(acc, x, y, D, pitch, h, i, length);
    }
    // Z = C(u + iota s, v + iota t): the operand halves are its planes
    if (h == 4) FS_NAME(leaf4)(nodes, x, x + D, y, y + D, pitch, length);
    if (h == 8) FS_NAME(leaf8)(nodes, x, x + D, y, y + D, pitch, length);
}

/* ---------- Direct sums ---------- */

// Add or subtract a partial sum of x_i y_j according to the basis sign
#define FS_DIRECT_FLUSH(i, j, s) \
    { \
        double* const a_ = acc + 8 * ((i) ^ (j)); \
        if (fastcd_sum_sign(N, i, j) > 0) FS_ACCUMULATE(a_, s); \
        else FS_STORE(a_, FS_SUB(FS_LOAD(a_), s)); \
    }

/* Accumulate the N^2 products of a direct sum of any dimension. Each pass
 * over the elements multiplies one row of x with four rows of y. */
static void FS_NAME(direct_any)(double* restrict acc, const double* x,
        const double* y, size_t pitch, size_t length, size_t N) {
    for (size_t i = 0; i < N; i++) {
        const double* xi = x + i * pitch;
        size_t j = 0;
        for (; j + 4 <= N; j += 4) {
            const double* yj = y + j * pitch;
            FS_V s0 = FS_ZERO, s1 = FS_ZERO, s2 = FS_ZERO, s3 = FS_ZERO;
            for (size_t p = 0; p < length; p += 8) {
                const FS_V v = FS_LOAD(xi + p);
                s0 = FS_FMA(v, FS_LOAD(yj + p), s0);
                s1 = FS_FMA(v, FS_LOAD(yj + pitch + p), s1);
                s2 = FS_FMA(v, FS_LOAD(yj + 2 * pitch + p), s2);
                s3 = FS_FMA(v, FS_LOAD(yj + 3 * pitch + p), s3);
            }
            FS_DIRECT_FLUSH(i, j, s0)
            FS_DIRECT_FLUSH(i, j + 1, s1)
            FS_DIRECT_FLUSH(i, j + 2, s2)
            FS_DIRECT_FLUSH(i, j + 3, s3)
        }
        for (; j < N; j++) {
            const double* yj = y + j * pitch;
            FS_V s = FS_ZERO;
            for (size_t p = 0; p < length; p += 8) {
                s = FS_FMA(FS_LOAD(xi + p), FS_LOAD(yj + p), s);
            }
            FS_DIRECT_FLUSH(i, j, s)
        }
    }
}

static void FS_NAME(direct_chunk)(double* restrict acc, const double* x,
        const double* y, size_t pitch, size_t length, size_t N) {
    switch (N) {
        case 2: FS_NAME(direct_2)(acc, x, y, pitch, length); return;
        case 4: FS_NAME(direct_4)(acc, x, y, pitch, length); return;
        case 8: FS_NAME(direct_8)(acc, x, y, pitch, length); return;
        case 16: FS_NAME(direct_16)(acc, x, y, pitch, length); return;
        default: FS_NAME(direct_any)(acc, x, y, pitch, length, N); return;
    }
}

/* ---------- Output additions ---------- */

// Add the eight lanes of an accumulator in a fixed order
static inline double FS_NAME(reduce)(FS_V v) {
    double d[8];

    FS_LANES(v, d);
    return ((d[0] + d[1]) + (d[2] + d[3])) + ((d[4] + d[5]) + (d[6] + d[7]));
}

// One complex output of the three real nonrecursive-map sums
#define FS_RECOVER(re, im, d, o, w) \
    { \
        const FS_V d_ = FS_LOAD(d), o_ = FS_LOAD(o); \
        FS_STORE(re, FS_SUB(d_, o_)); \
        FS_STORE(im, FS_SUB(FS_SUB(FS_LOAD(w), d_), o_)); \
    }

/* Apply the output additions of the complex kernel of dimension K to its
 * accumulators. re and im receive K vectors each; scratch holds 32 K doubles. */
static void FS_NAME(finish_node)(size_t K, const double* acc, double* re,
        double* im, double* scratch) {
    const FS_V zero = FS_ZERO;

    FS_STORE(re, zero); FS_STORE(im, zero);
    if (K <= 2) {
        if (K == 2) { FS_STORE(re + 8, zero); FS_STORE(im + 8, zero); }
        return;
    }
    if (K == 4) {
        FS_RECOVER(re + 16, im + 16, acc, acc + 8, acc + 16)
        FS_RECOVER(re + 8, im + 8, acc + 24, acc + 40, acc + 56)
        FS_RECOVER(re + 24, im + 24, acc + 32, acc + 48, acc + 64)
        return;
    }
    if (K == 8) {
        for (size_t k = 1; k < 8; k++) {
            FS_RECOVER(re + 8 * k, im + 8 * k, acc + 8 * (k - 1),
                    acc + 8 * (k + 6), acc + 8 * (k + 13))
        }
        return;
    }
    {
        const size_t half = K / 2;
        const FS_V one_half = FS_SET1(0.5);
        double* const zpr = scratch;
        double* const zpi = zpr + 8 * half;
        double* const zmr = zpi + 8 * half;
        double* const zmi = zmr + 8 * half;
        double* const below = zmi + 8 * half;
        const double* const child = acc + 8 * (3 + 6 * half);
        FS_NAME(finish_node)(half, child, zpr, zpi, below);
        FS_NAME(finish_node)(half, child + 8 * fastcd_sum_node_vectors(half),
                zmr, zmi, below);
        FS_RECOVER(re + 8 * half, im + 8 * half, acc, acc + 8, acc + 16)
        for (size_t i = 1; i < half; i++) {
            const double* a = acc + 8 * (3 + 6 * i);
            const FS_V pr = FS_LOAD(zpr + 8 * i), pi = FS_LOAD(zpi + 8 * i);
            const FS_V mr = FS_LOAD(zmr + 8 * i), mi = FS_LOAD(zmi + 8 * i);
            const FS_V fd = FS_LOAD(a), gd = FS_LOAD(a + 8);
            const FS_V fo = FS_LOAD(a + 16), go = FS_LOAD(a + 24);
            const FS_V fr = FS_SUB(fd, fo);
            const FS_V fi = FS_SUB(FS_SUB(FS_LOAD(a + 32), fd), fo);
            const FS_V gr = FS_SUB(gd, go);
            const FS_V gi = FS_SUB(FS_SUB(FS_LOAD(a + 40), gd), go);
            // ((Z_+ + Z_-)/2 + f,  g + iota (Z_+ - Z_-)/2)
            FS_STORE(re + 8 * i, FS_ADD(FS_MUL(FS_ADD(pr, mr), one_half), fr));
            FS_STORE(im + 8 * i, FS_ADD(FS_MUL(FS_ADD(pi, mi), one_half), fi));
            FS_STORE(re + 8 * (half + i),
                    FS_SUB(gr, FS_MUL(FS_SUB(pi, mi), one_half)));
            FS_STORE(im + 8 * (half + i),
                    FS_ADD(gi, FS_MUL(FS_SUB(pr, mr), one_half)));
        }
    }
}

// Write the sum from the accumulators of a quasilinear schedule
static void FS_NAME(quasilinear_finish)(double* restrict out, const double* acc,
        size_t N, double* scratch) {
    const size_t h = N / 2;
    double* const zr = scratch;
    double* const zi = zr + 8 * h;

    FS_NAME(finish_node)(h, acc + 8 * N, zr, zi, zi + 8 * h);
    out[0] = FS_NAME(reduce)(FS_LOAD(acc));
    out[h] = FS_NAME(reduce)(FS_LOAD(acc + 8 * h));
    for (size_t i = 1; i < h; i++) {
        // X cross Y = (Re Z + f, lambda + g - Im Z)
        out[i] = FS_NAME(reduce)(FS_ADD(FS_LOAD(acc + 8 * i), FS_LOAD(zr + 8 * i)));
        out[h + i] = FS_NAME(reduce)(
                FS_SUB(FS_LOAD(acc + 8 * (h + i)), FS_LOAD(zi + 8 * i)));
    }
}

/* ---------- Operand block ---------- */

// Apply an operand map to one vector of coordinate c
static inline FS_V FS_NAME(map_vector)(FS_V v, const FastCDSumOperandMap* map,
        size_t c) {
    if (map->pre != NULL) v = FS_MUL(v, FS_SET1(map->pre[c]));
    if (map->shift != NULL) v = FS_SUB(v, FS_SET1(map->shift[c]));
    if (map->offset != NULL) v = FS_SUB(v, FS_SET1(map->offset[c]));
    if (map->scale != NULL) v = FS_MUL(v, FS_SET1(map->scale[c]));
    return v;
}

// Apply an operand map to one coordinate, with the operations of map_vector
static inline double FS_NAME(map_scalar)(double v, const FastCDSumOperandMap* map,
        size_t c) {
    if (map->pre != NULL) v = v * map->pre[c];
    if (map->shift != NULL) v = v - map->shift[c];
    if (map->offset != NULL) v = v - map->offset[c];
    if (map->scale != NULL) v = v * map->scale[c];
    return v;
}

/*
 * Write `count` <= FASTCD_FUSED_CHUNK records of dimension N as rows of a
 * block, completing the last group of eight with zero elements, and apply
 * the operand map to every coordinate.
 */
static void FS_NAME(pack)(double* restrict rows, const double* a, size_t stride,
        size_t count, size_t N, const FastCDSumOperandMap* map) {
    const size_t padded = (count + 7) & ~(size_t)7;
    size_t done = 0;

#if defined(FS_TRANSPOSE)
    if (N >= 4) {
        for (; done + 8 <= count; done += 8) {
            const double* r = a + done * stride;
            for (size_t c = 0; c < N; c += 4) {
                FS_V o[4];
                FS_NAME(transpose)(o, r, stride, c);
                for (size_t k = 0; k < 4; k++) {
                    FS_STORE(rows + (c + k) * FASTCD_FUSED_PITCH + done,
                            FS_NAME(map_vector)(o[k], map, c + k));
                }
            }
        }
    }
#endif
    for (size_t c = 0; c < N; c++) {
        double* const row = rows + c * FASTCD_FUSED_PITCH;
        for (size_t p = done; p < count; p++) {
            row[p] = FS_NAME(map_scalar)(a[p * stride + c], map, c);
        }
        for (size_t p = count; p < padded; p++) row[p] = 0.0;
    }
}

/* ---------- Driver ---------- */

// Accumulate one chunk of either schedule
static void FS_NAME(chunk)(const FastCDSumJob* job, double* restrict acc,
        const double* x, const double* y, size_t pitch, size_t length,
        double* block) {
    if (job->quasilinear) {
        FS_NAME(quasilinear_chunk)(acc, x, y, pitch, length, job->dimension, block);
    } else {
        FS_NAME(direct_chunk)(acc, x, y, pitch, length, job->dimension);
    }
}

void FS_PUBLIC(double* restrict out, const FastCDSumJob* job,
        double* restrict work) {
    const size_t N = job->dimension;
    const size_t vectors = job->quasilinear ? fastcd_sum_vectors(N) : N;
    double* const acc = work;
    double* const block = work + 8 * fastcd_sum_vectors(N);
    double* const scratch = block + 2 * N * FASTCD_FUSED_PITCH;
    double* const bx = block;
    double* const by = block + N * FASTCD_FUSED_PITCH;

    memset(acc, 0, 8 * vectors * sizeof(double));
    /* Both layouts cut the elements into the same chunks, so that they add
     * the same partial sums in the same order. The position advances by the
     * elements consumed, which cannot pass the count. */
    for (size_t p = 0; p < job->count;) {
        const size_t length = job->count - p < FASTCD_FUSED_CHUNK
            ? job->count - p : FASTCD_FUSED_CHUNK;
        const size_t padded = (length + 7) & ~(size_t)7;
        if (!job->planes) {
            FS_NAME(pack)(bx, job->x + p * job->x_stride, job->x_stride, length,
                    N, &job->x_map);
            FS_NAME(pack)(by, job->y + p * job->y_stride, job->y_stride, length,
                    N, &job->y_map);
            FS_NAME(chunk)(job, acc, bx, by, FASTCD_FUSED_PITCH, padded, block);
        } else if (padded == length) {
            FS_NAME(chunk)(job, acc, job->x + p, job->y + p, job->pitch, length,
                    block);
        } else {
            // Complete the last group of eight with zero elements
            for (size_t i = 0; i < N; i++) {
                double* const rx = bx + i * FASTCD_FUSED_PITCH;
                double* const ry = by + i * FASTCD_FUSED_PITCH;
                for (size_t k = 0; k < length; k++) {
                    rx[k] = job->x[i * job->pitch + p + k];
                    ry[k] = job->y[i * job->pitch + p + k];
                }
                for (size_t k = length; k < padded; k++) {
                    rx[k] = 0.0;
                    ry[k] = 0.0;
                }
            }
            FS_NAME(chunk)(job, acc, bx, by, FASTCD_FUSED_PITCH, padded, block);
        }
        p += length;
    }
    if (job->quasilinear && N >= 2) {
        FS_NAME(quasilinear_finish)(out, acc, N, scratch);
    } else {
        for (size_t k = 0; k < N; k++) {
            out[k] = FS_NAME(reduce)(FS_LOAD(acc + 8 * k));
        }
    }
}
