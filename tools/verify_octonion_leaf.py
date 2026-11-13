#!/usr/bin/env python3
"""Verify the octonion leaf of the quasilinear Cayley--Dickson schedule.

Everything runs in exact rational arithmetic (fractions.Fraction); a complex
coefficient a + iota*b is the pair (a, b), as in fastCD's real/imaginary planes.

  1. The identity of Theorem "Octonion leaf",
         C_3(X,Y) = -iota L_1^T(L_1X x L_1Y) + 1/2 sum_{j=2..4} L_j^T(L_jX x L_jY),
     on all 49 pairs of basis vectors.  Both sides are bilinear with structure
     constants in Z[1/2][iota], so this proves it over every field containing
     1/2 and a square root iota of -1.
  2. The three-stage real evaluation of that identity (the schedule emitted by
     tools/generate_kernels.py: octonion_cross) on all 196 pairs of real basis
     vectors of the planes, with every operation counted: 72 input-dependent
     multiplications, 156 additions/subtractions, 15 halvings.
  3. The same check on the generated C kernel ql_complex_cross_8 itself, read
     from src/generated/quasilinear_kernels.h (found automatically inside a
     fastCD checkout, or pass its path as the only argument).
  4. The closed forms for the kernel and full-product counts against the
     recurrences they solve, for levels up to 40.

Usage:  python3 verify_octonion_leaf.py [path/to/quasilinear_kernels.h]
"""
import sys
from fractions import Fraction
from pathlib import Path


# ---- Reference: the defining Cayley--Dickson product with every mu_k = -1 ----
def cd_mul(x, y):
    """(a,b)(c,d) = (ac - conj(d) b, da + b conj(c)), the paper's eq. (2.1)."""
    if len(x) == 1:
        return [x[0] * y[0]]
    h = len(x) // 2
    def conj(v): return [v[0]] + [-c for c in v[1:]]
    a, b, c, d = x[:h], x[h:], y[:h], y[h:]
    first = [p - q for p, q in zip(cd_mul(a, c), cd_mul(conj(d), b), strict=True)]
    second = [p + q for p, q in zip(cd_mul(d, a), cd_mul(b, conj(c)), strict=True)]
    return first + second

def alt(X, Y):
    """X x_3 Y = (XY - YX)/2 for rational imaginary octonions (coordinates e1..e7)."""
    xy, yx = cd_mul([0] + X, [0] + Y), cd_mul([0] + Y, [0] + X)
    return [Fraction(p - q, 2) for p, q in zip(xy[1:], yx[1:], strict=True)]

def reference(xr, xi, yr, yi):
    """Complex-bilinear extension of x_3, by planes: (rr - ii) + iota (ri + ir)."""
    rr, ii, ri, ir = alt(xr, yr), alt(xi, yi), alt(xr, yi), alt(xi, yr)
    return ([p - q for p, q in zip(rr, ii, strict=True)],
            [p + q for p, q in zip(ri, ir, strict=True)])

# ---- Counted numbers: every real operation of an evaluation is recorded ----
COUNT = {"M": 0, "A": 0, "E": 0}

class Num:
    """A rational with a tag: 'x' or 'y' (linear in one operand) or 'xy' (bilinear).

    The assertions make the evaluation a bilinear algorithm by construction:
    sums stay within one tag and every product is an x-form times a y-form.
    Sign changes are free, as in the paper's counting convention.
    """
    def __init__(self, value, tag):
        self.v, self.tag = Fraction(value), tag
    def _sum(self, other, sign):
        assert self.tag == other.tag
        COUNT["A"] += 1
        return Num(self.v + sign * other.v, self.tag)
    def __add__(self, other): return self._sum(other, 1)
    def __sub__(self, other): return self._sum(other, -1)
    def __neg__(self): return Num(-self.v, self.tag)
    def __mul__(self, other):
        assert {self.tag, other.tag} == {"x", "y"}
        COUNT["M"] += 1
        return Num(self.v * other.v, "xy")
    def half(self):
        COUNT["E"] += 1
        return Num(self.v / 2, self.tag)

# ---- Complex helpers on pairs; they work for Fraction and for Num ----
def cadd(a, b): return (a[0] + b[0], a[1] + b[1])
def csub(a, b): return (a[0] - b[0], a[1] - b[1])
def times_i(a): return (-a[1], a[0])                    # free: swap and one sign
def chalf(a): return (a[0].half(), a[1].half())
def cmul(a, b): return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])
def cross(a, b):                                        # 6 products, 3 subtractions
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]

def leaf_maps(X):
    """L_1X..L_4X for X = (x1,x2,x3,h,y1,y2,y3): 9 complex additions."""
    x1, x2, x3, h, y1, y2, y3 = X
    p = (h[1], -h[0])                                   # p = -iota h, free
    z = cadd(x1, p)
    bp, bm = cadd(x2, times_i(y2)), csub(x2, times_i(y2))   # b_+ and b_-
    cp, cm = cadd(x3, times_i(y3)), csub(x3, times_i(y3))   # c_+ and c_-
    return [(x1, y1, p), (cadd(bp, cm), csub(bm, cp), p),
            (cadd(z, times_i(y1)), bp, cp), (csub(z, times_i(y1)), bm, cm)]

# ---- Check 1: the identity as stated, with ordinary (unconjugated) transposes ----
def check_identity():
    zero, one = (Fraction(0), Fraction(0)), (Fraction(1), Fraction(0))
    basis = [[one if k == a else zero for k in range(7)] for a in range(7)]
    columns = [leaf_maps(e) for e in basis]             # columns[k][j][i] = (L_j)_{i,k}
    def ccross(u, v):
        return [csub(cmul(u[i], v[j]), cmul(u[j], v[i]))
                for i, j in ((1, 2), (2, 0), (0, 1))]
    def unit(a): return [Fraction(int(k == a)) for k in range(7)]
    for a in range(7):
        for b in range(7):
            rhs = [zero] * 7
            for j in range(4):
                c = ccross(columns[a][j], columns[b][j])    # L_j e_a x L_j e_b
                for k in range(7):                          # coordinate k of L_j^T c
                    t = zero
                    for i in range(3):
                        t = cadd(t, cmul(columns[k][j][i], c[i]))
                    # weight -iota for j = 1 and 1/2 for j = 2, 3, 4
                    t = times_i((-t[0], -t[1])) if j == 0 else (t[0] / 2, t[1] / 2)
                    rhs[k] = cadd(rhs[k], t)
            assert [r[0] for r in rhs] == alt(unit(a), unit(b)), (a, b)
            assert all(r[1] == 0 for r in rhs), (a, b)
    return 49

# ---- Check 2: the real straight-line evaluation, stage by stage ----
def leaf(xr, xi, yr, yi, stages=None):
    """Mirror of octonion_cross in tools/generate_kernels.py; returns two planes."""
    def mark(): return stages.append(dict(COUNT)) if stages is not None else None
    # Stage 1, input linear maps: 9 complex additions per operand
    LX = leaf_maps(list(zip(xr, xi, strict=True)))
    LY = leaf_maps(list(zip(yr, yi, strict=True)))
    mark()
    # Stage 2, four complex cross products, each by three real cross products
    # Delta = re x re, Omega = im x im, W = (re+im) x (re+im); the plane sum of
    # p is shared by L_1 and L_2 (they have the same third coordinate)
    def plane_sums(L):
        p_sum = L[0][2][0] + L[0][2][1]
        return [[p_sum if j < 2 and i == 2 else c[0] + c[1] for i, c in enumerate(t)]
                for j, t in enumerate(L)]
    SX, SY = plane_sums(LX), plane_sums(LY)
    results, thirds = [], []
    for j in range(4):
        D = cross([c[0] for c in LX[j]], [c[0] for c in LY[j]])
        O = cross([c[1] for c in LX[j]], [c[1] for c in LY[j]])  # noqa: E741
        W = cross(SX[j], SY[j])
        # Upsilon_1, Upsilon_2 enter only the e4 output, so they stay unreconstructed
        keep = 2 if j < 2 else 3
        if j < 2:
            thirds.append((D[2], O[2], W[2]))
        results.append([(D[i] - O[i], (W[i] - D[i]) - O[i]) for i in range(keep)])
    mark()
    # Stage 3, output reconstruction (the sparse transposes, collected)
    (a1, b1), (a2, b2), (a3, b3, c3), (a4, b4, c4) = results
    t, u = cadd(a3, a4), csub(a3, a4)                   # Theta_+, Theta_-
    d1, d2 = cadd(a2, b3), cadd(b2, b4)                 # Pi_1, Pi_2
    e1, e2 = csub(c3, b2), cadd(a2, c4)                 # Sigma_1, Sigma_2
    half_t = chalf(t)
    (D1, O1, W1), (D2, O2, W2) = thirds
    phi, chi = D1 + O2.half(), O1 - D2.half()
    h = (((chi - phi) + W2.half()) + half_t[1], ((phi + chi) - W1) - half_t[0])
    out = [csub(half_t, times_i(a1)), chalf(cadd(d1, d2)), chalf(cadd(e1, e2)), h,
           times_i(csub(chalf(u), b1)), times_i(chalf(csub(d1, d2))),
           times_i(chalf(csub(e1, e2)))]
    mark()
    return [c[0] for c in out], [c[1] for c in out]

# ---- Check 3: execute the generated C kernel text itself ----
def c_kernel(text):
    start = text.index("FASTCD_NAME(ql_complex_cross_8)(")
    body = text[text.index("{", start) + 1:text.index("\n}", start)]
    lines = [s.strip().rstrip(";") for s in body.splitlines() if s.strip()]
    def run(xr, xi, yr, yi):
        env = {f"{n}[{i + 1}]": v for n, plane in (("x_real", xr), ("x_imag", xi),
               ("y_real", yr), ("y_imag", yi)) for i, v in enumerate(plane)}
        def val(s): return -env[s[1:]] if s[0] == "-" else env[s]
        for line in lines:
            lhs, rhs = line.replace("const FASTCD_REAL ", "").split(" = ")
            tok = rhs.split()
            if rhs == "FASTCD_ZERO":
                continue                               # scalar coordinate of the output
            if len(tok) == 1:
                env[lhs] = val(tok[0])                  # copy or sign change: free
            elif tok[1] in "+-":
                env[lhs] = (val(tok[0]) + val(tok[2]) if tok[1] == "+"
                            else val(tok[0]) - val(tok[2]))
            elif tok[2] == "0.5":
                env[lhs] = val(tok[0]).half()
            else:
                env[lhs] = val(tok[0]) * val(tok[2])
        return ([env[f"out_real[{i}]"] for i in range(1, 8)],
                [env[f"out_imag[{i}]"] for i in range(1, 8)])
    return run

def check_program(program, with_stages=False):
    """Run a leaf on all 14 x 14 real basis pairs; compare values and counts."""
    for a in range(14):
        for b in range(14):
            planes = [[Fraction(int(k == i)) for k in range(7)]
                      for i in (a, a - 7, b, b - 7)]    # xr, xi, yr, yi
            tagged = [[Num(v, tag) for v in plane]
                      for plane, tag in zip(planes, "xxyy", strict=True)]
            COUNT.update(M=0, A=0, E=0)
            stages = [] if with_stages else None
            got = program(*tagged, stages) if with_stages else program(*tagged)
            values = tuple([n.v for n in plane] for plane in got)
            assert values == reference(*planes), (a, b)
            assert (COUNT["M"], COUNT["A"], COUNT["E"]) == (72, 156, 15), COUNT
            if with_stages:                             # the stage table in the paper
                s1, s2, s3 = [(s["M"], s["A"], s["E"]) for s in stages]
                assert s1 == (0, 36, 0)
                assert tuple(q - p for p, q in zip(s1, s2, strict=True)) == (72, 88, 0)
                assert tuple(q - p for p, q in zip(s2, s3, strict=True)) == (0, 32, 15)
    return 196

# ---- Check 4: closed forms against the recurrences ----
def check_closed_forms(top=40):
    M, P, E = 72, 156, 15                               # the leaf, level k = 3
    for k in range(3, top + 1):
        if k > 3:                                       # one two-call node at level k
            m = 2 ** (k - 1) - 1
            M, P, E = 2 * M + 18 * m, 2 * P + 38 * m + 2, 2 * E + 4 * m
        assert (M, P, E) == ((36 * k - 81) * 2 ** (k - 2) + 18,
                             (76 * k - 168) * 2 ** (k - 2) + 36,
                             (16 * k - 37) * 2 ** (k - 3) + 4)
        n, N = k + 1, 2 ** (k + 1)                 # full product: root + reconstruction
        full = (M + 6 * (N // 2 - 1) + 3 * N - 2, P + (3 * N - 7) + (3 * N - 3), E)
        assert full == ((36 * n - 69) * N // 8 + 10, (76 * n - 196) * N // 8 + 26,
                        (16 * n - 53) * N // 16 + 4)
    return top

if __name__ == "__main__":
    print(f"1. identity (A.2): {check_identity()} basis pairs, exact")
    print(f"2. straight-line evaluation: {check_program(leaf, True)} plane basis "
          "pairs, counts (M, A, E) = (72, 156, 15), "
          "stages (0,36,0) + (72,88,0) + (0,32,15)")
    root = Path(__file__).resolve().parents[1]
    default = root / "src/generated/quasilinear_kernels.h"
    header = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    if header.is_file():
        n = check_program(c_kernel(header.read_text()))
        print(f"3. C kernel ql_complex_cross_8 ({header}): {n} plane basis pairs, "
              "counts (72, 156, 15)")
    else:
        print("3. C kernel: skipped "
              "(pass the path of src/generated/quasilinear_kernels.h)")
    print("4. closed forms for kernel and full counts: "
          f"levels 3..{check_closed_forms()}")
    print("all checks passed")
