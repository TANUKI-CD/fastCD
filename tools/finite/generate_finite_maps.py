#!/usr/bin/env python3
"""Field-independent CRT and integer Lagrange constants, for every degree.

All small-characteristic tensors are verified on every pair of polynomial
basis vectors before emission. The caller's defining polynomial is never an
input to these tables; reduction modulo that polynomial remains in each call.
"""

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "src/finite/generated"


def mul(a, b, p=None):
    """Convolve integer polynomials, optionally reducing each coefficient modulo p."""

    c = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            c[i + j] += x * y
    return [x % p for x in c] if p else c


def remainder(a, g, p):
    """Divide by a monic polynomial and return its fixed-width remainder."""

    a = a[:]
    for k in range(len(a) - 1, len(g) - 2, -1):
        v = a[k]
        for j in range(len(g)):
            a[k - len(g) + 1 + j] = (a[k - len(g) + 1 + j] - v * g[j]) % p
    return (a + [0] * len(g))[: len(g) - 1]


def quotient(a, g, p):
    """Return an exact monic quotient, asserting that its remainder is zero."""

    a = a[:]
    q = [0] * (len(a) - len(g) + 1)
    for k in range(len(a) - 1, len(g) - 2, -1):
        v = a[k]
        q[k - len(g) + 1] = v
        for j in range(len(g)):
            a[k - len(g) + 1 + j] = (a[k - len(g) + 1 + j] - v * g[j]) % p
    assert not any(a)
    return q


def inverse(q, g, p):
    """Invert q modulo g by solving its small multiplication matrix over F_p."""

    h = len(g) - 1
    columns = [remainder([0] * j + q, g, p) for j in range(h)]
    a = [[columns[col][row] for col in range(h)] + [int(row == 0)] for row in range(h)]
    for j in range(h):
        pivot = next(i for i in range(j, h) if a[i][j])
        a[j], a[pivot] = a[pivot], a[j]
        scale = pow(a[j][j], p - 2, p)
        a[j] = [x * scale % p for x in a[j]]
        for i in range(h):
            if i != j:
                scale = a[i][j]
                a[i] = [(x - scale * y) % p for x, y in zip(a[i], a[j], strict=False)]
    return [a[i][-1] for i in range(h)]


def tensor(p, d):
    """Build CRT product channels and prove their bilinear convolution identity."""

    factors = []
    total = 0
    target = 2 * d - 1
    for a in range(min(p, target)):
        factors.append([(-a) % p, 1])
        total += 1
    for h in (2, 3):
        if total >= target:
            break
        for code in range(1, p**h):
            v = code
            g = []
            for _j in range(h):
                g.append(v % p)
                v //= p
            g.append(1)
            if any(
                sum(c * pow(a, j, p) for j, c in enumerate(g)) % p == 0
                for a in range(p)
            ):
                continue
            factors.append(g)
            total += h
            if total >= target:
                break
    assert target <= total <= 24
    nodal = [1]
    for g in factors:
        nodal = mul(nodal, g, p)
    inputs = []
    outputs = []
    quadratic = [[1, -1, 0], [0, -1, 1], [0, 1, 0]]
    cubic = [
        [1, -1, -1, 0, 0],
        [0, -1, 1, -1, 0],
        [0, 0, -1, -1, 1],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
    ]
    for g in factors:
        h = len(g) - 1
        q = quotient(nodal, g, p)
        basis = mul(q, inverse(remainder(q, g, p), g, p), p)
        basis = (basis + [0] * total)[:total]
        w = []
        for _j in range(h):
            w.append(basis)
            basis = remainder([0] + basis, nodal, p)
        rem = [remainder([0] * j + [1], g, p) for j in range(d)]
        products = 1 if h == 1 else 3 if h == 2 else 6
        for r in range(products):
            first = second = r
            if h == 2 and r == 2:
                first, second = 0, 1
            if h == 3 and r >= 3:
                first, second = ((0, 1), (0, 2), (1, 2))[r - 3]
            inputs.append(
                [
                    (rem[j][first] + (rem[j][second] if first != second else 0)) % p
                    for j in range(d)
                ]
            )
            v = [1] if h == 1 else quadratic[r] if h == 2 else cubic[r]
            v = remainder([x % p for x in v], g, p)
            outputs.append(
                [sum(v[k] * w[k][j] for k in range(h)) % p for j in range(total)]
            )
    rank = len(inputs)
    assert rank <= 48
    # Both sides are bilinear. These basis identities therefore verify all
    # input polynomials over F_p, before reduction by any defining polynomial.
    for i in range(d):
        for j in range(d):
            for k in range(total):
                assert sum(
                    inputs[r][i] * inputs[r][j] * outputs[r][k] for r in range(rank)
                ) % p == int(k == i + j), (p, d, i, j, k)
    return rank, total, inputs, outputs


def write(path, content):
    """Replace an emitted artifact only when its contents have changed."""

    if not path.exists() or path.read_text() != content:
        path.write_text(content)


def generate():
    """Emit universal CRT and Lagrange tables, independent of the user's modulus."""

    small = []
    identities = 0
    for p in (3, 5, 7):
        contexts = []
        for d in range(1, 13):
            rank, total, a, b = tensor(p, d)
            identities += d * d

            def rows(vectors):
                return (
                    "{"
                    + ",\n".join("{" + ",".join(map(str, row)) + "}" for row in vectors)
                    + "}"
                )

            inputs = [[a[r][j] for r in range(rank)] for j in range(d)]
            outputs = [[b[r][k] for r in range(rank)] for k in range(total)]
            contexts.append(
                "{"
                + f"{rank},{total},{{0}},"
                + rows(inputs)
                + ","
                + rows(outputs)
                + "}"
            )
        small.append("{" + ",\n".join(contexts) + "}")
    low = []
    high = []
    offset = []
    maximum = 0
    for d in range(1, 13):
        rank = 2 * d - 1
        offset.append(len(low))
        polys = []
        for i in range(rank):
            poly = [1]
            for j in range(rank):
                if j != i:
                    poly = mul(poly, [-j, 1])
            assert all(v == 0 or (v < 0) == bool(k & 1) for k, v in enumerate(poly))
            denominator = (
                (-1 if i & 1 else 1) * math.factorial(i) * math.factorial(rank - 1 - i)
            )
            for point in range(rank):
                value = 0
                for coefficient in reversed(poly):
                    value = value * point + coefficient
                assert value == (denominator if point == i else 0)
            polys.append(poly)
        for k in range(rank):
            for i in range(rank):
                v = abs(polys[i][k])
                maximum = max(maximum, v)
                assert v < 1 << 79
                low.append(v & ((1 << 64) - 1))
                high.append(v >> 64)
    assert max(high) < 65536
    source = (
        """/*
 * Generated by tools/finite/generate_finite_maps.py. Do not edit by hand.
 * CRT maps reconstruct polynomial convolution before field reduction;
 * the caller's actual defining polynomial is handled by per-call setup.
 * Linear-point Lagrange numerator magnitudes are split into low/high limbs;
 * their alternating signs are recovered by the reconstruction code.
 */
#include "finite_maps.h"
const CdlSmallMap fastcd_ff_small_maps[3][12] __attribute__((aligned(64)))={
"""
        + ",\n".join(small)
        + "\n};\n"
    )
    source += (
        "const size_t fastcd_ff_linear_offset[12]={"
        + ",".join(map(str, offset))
        + "};\n"
    )
    source += (
        f"const uint64_t fastcd_ff_linear_low[{len(low)}]={{\n"
        + ",\n".join(
            ",".join("UINT64_C(" + str(v) + ")" for v in low[i : i + 8])
            for i in range(0, len(low), 8)
        )
        + "};\n"
    )
    source += (
        f"const uint16_t fastcd_ff_linear_high[{len(high)}]={{\n"
        + ",\n".join(
            ",".join(map(str, high[i : i + 16])) for i in range(0, len(high), 16)
        )
        + "};\n"
    )
    write(OUT / "finite_maps.c", source)
    return {
        "small_tensors": 36,
        "basis_pair_identities": identities,
        "integer_numerators": len(low),
        "maximum_numerator_bits": maximum.bit_length(),
    }


if __name__ == "__main__":
    print(generate())
