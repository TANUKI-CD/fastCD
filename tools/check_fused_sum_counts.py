#!/usr/bin/env python3
"""Count the arithmetic of the fused sums in the actual kernel source.

The probe instantiates ``src/fused_sum_impl.h`` with a vector type whose
operations count themselves and runs sums of 8, 16, 64 and 72 elements. The
difference between 8 and 16 is what one more group of eight elements costs
inside a block: the products and input additions per element, in vector
operations. The difference between 64 and 72 starts a second block of 64,
and exceeds the first difference by the additions that merge the partial
sums of a block into the accumulated sums. The output additions, spent once
per sum, cancel in both differences. The counts are compared with the
recurrences stated in docs/mathematics.md.

Requires a C compiler only for this development check.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

PROBE = r'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fused_sum.h"

typedef struct Vector { double lane[8]; } Vector;
static unsigned long long products, additions;

static Vector vzero(void) { Vector r; memset(&r, 0, sizeof r); return r; }
static Vector load(const double* p) {
    Vector r;
    memcpy(r.lane, p, sizeof r.lane);
    return r;
}
static void store(double* p, Vector v) { memcpy(p, v.lane, sizeof v.lane); }
static Vector set1(double s) {
    Vector r;
    for (int k = 0; k < 8; k++) r.lane[k] = s;
    return r;
}
static Vector add(Vector a, Vector b) {
    additions++;
    for (int k = 0; k < 8; k++) a.lane[k] += b.lane[k];
    return a;
}
static Vector sub(Vector a, Vector b) {
    additions++;
    for (int k = 0; k < 8; k++) a.lane[k] -= b.lane[k];
    return a;
}
static Vector mul(Vector a, Vector b) {
    for (int k = 0; k < 8; k++) a.lane[k] *= b.lane[k];
    return a;
}
static Vector fma_(Vector a, Vector b, Vector c) {
    products++;
    for (int k = 0; k < 8; k++) c.lane[k] += a.lane[k] * b.lane[k];
    return c;
}
static Vector fnma_(Vector a, Vector b, Vector c) {
    products++;
    for (int k = 0; k < 8; k++) c.lane[k] -= a.lane[k] * b.lane[k];
    return c;
}

#define FS_V Vector
#define FS_NAME(name) probe_##name
#define FS_PUBLIC probe_sum
#define FS_ZERO vzero()
#define FS_LOAD(p) load(p)
#define FS_STORE(p, v) store(p, v)
#define FS_SET1(s) set1(s)
#define FS_ADD(a, b) add(a, b)
#define FS_SUB(a, b) sub(a, b)
#define FS_MUL(a, b) mul(a, b)
#define FS_FMA(a, b, c) fma_(a, b, c)
#define FS_FNMA(a, b, c) fnma_(a, b, c)
#define FS_LANES(v, d) store(d, v)

void probe_sum(double* restrict out, const FastCDSumJob* job, double* restrict work);

#include "fused_sum_impl.h"

int main(void) {
    for (size_t n = 1; n <= 1024; n *= 2) {
        for (int quasilinear = 0; quasilinear < 2; quasilinear++) {
            static const size_t counts[4] = {8, 16, 64, 72};
            unsigned long long seen[4][2];
            double* rows = calloc(2 * n * 80, sizeof(double));
            double* out = calloc(n, sizeof(double));
            double* work = calloc(fastcd_sum_kernel_doubles(n), sizeof(double));
            if (n == 1 && quasilinear) continue;
            for (size_t c = 0; c < 4; c++) {
                FastCDSumJob job;
                memset(&job, 0, sizeof job);
                job.x = rows;
                job.y = rows + n * 80;
                job.pitch = 80;
                job.count = counts[c];
                job.dimension = n;
                job.planes = true;
                job.quasilinear = quasilinear != 0;
                products = additions = 0;
                probe_sum(out, &job, work);
                seen[c][0] = products;
                seen[c][1] = additions;
            }
            printf("%zu %d %llu %llu %llu %llu\n", n, quasilinear,
                    seen[1][0] - seen[0][0], seen[1][1] - seen[0][1],
                    seen[3][0] - seen[2][0],
                    (seen[3][1] - seen[2][1]) - (seen[1][1] - seen[0][1]));
            free(rows); free(out); free(work);
        }
    }
    return 0;
}
'''


def complex_kernel(K: int) -> tuple[int, int]:
    """Products and input additions of the complex kernel of dimension K."""

    if K <= 2:
        return 0, 0
    if K == 4:
        # The nonrecursive map alone: 18 products, 6 sums
        return 18, 6
    if K == 8:
        # Three real seven-dimensional cross products and 14 sums
        return 126, 14
    m = K // 2 - 1
    p, a = complex_kernel(K // 2)
    return 18 * m + 2 * p, 14 * m + 2 * a


def expected(n: int, quasilinear: bool) -> tuple[int, int]:
    """Products and additions per element of a fused sum of dimension n."""

    if not quasilinear:
        return n * n, 0
    p, a = complex_kernel(n // 2)
    # Splitting and real nonrecursive map: 12 products per coordinate pair
    return 6 * n - 8 + p, a


def block_subtree(K: int) -> int:
    """Additions that merge one block's partial sums of a complex kernel."""

    if K <= 2:
        return 0
    if K == 4:
        return 9
    if K == 8:
        # Two partial sums per output of each of the three cross products
        return 42
    return 6 * (K // 2 - 1) + 3 + 2 * block_subtree(K // 2)


def expected_block(n: int, quasilinear: bool) -> int:
    """Additions per block of 64 elements beyond those of its groups."""

    if not quasilinear:
        # Generated kernels keep two partial sums per output from dimension 4;
        # the loop for other dimensions merges every product separately
        return {1: 1, 2: 2, 4: 8, 8: 16, 16: 32}.get(n, n * n)
    h = n // 2
    if h == 1:
        return 4
    if n >= 32:
        # Root and top node in one pass, then the two subtrees
        return 10 + 10 * (h // 2 - 1) + 5 + 2 * block_subtree(h // 2)
    return 10 + 12 * ((h - 2) // 2) + block_subtree(h)


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    compiler = shlex.split(os.environ.get("CC", "cc"))
    with tempfile.TemporaryDirectory(prefix="fastcd-counts-") as tmp:
        source = Path(tmp) / "probe.c"
        binary = Path(tmp) / "probe"
        source.write_text(PROBE)
        subprocess.run(compiler + ["-std=c11", "-O1", "-I", str(root / "src"),
                                   str(source), "-o", str(binary)], check=True)
        result = subprocess.run([str(binary)], check=True, text=True,
                                capture_output=True)
    failures = 0
    print(f"{'N':>5} {'algorithm':>12} {'products':>9} {'additions':>10} "
          f"{'per block':>10}")
    for line in result.stdout.splitlines():
        n, ql, products, additions, again, block = (int(v) for v in line.split())
        want = (*expected(n, bool(ql)), expected_block(n, bool(ql)))
        name = "quasilinear" if ql else "direct"
        got = (products, additions, block)
        status = "" if got == want and again == products else f"  expected {want}"
        failures += bool(status)
        print(f"{n:>5} {name:>12} {products:>9} {additions:>10} {block:>10}{status}")
    if failures:
        print(f"{failures} counts differ", file=sys.stderr)
        return 1
    print("fused sum counts match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
