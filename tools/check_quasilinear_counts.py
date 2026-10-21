#!/usr/bin/env python3
"""Count arithmetic in the actual scalar QL template and generated leaves.

Requires a C++ compiler only for this development check. The library remains
C11. Operator instrumentation distinguishes input-dependent products from
fixed scalings; known-zero operations and sign changes are free, as in the
paper. These are arithmetic counts, not machine-instruction counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

PROBE = r'''
#include <cstdio>
#include <vector>

struct Number {
    double value;
    bool fixed;
    Number(double v = 0.0, bool f = true) : value(v), fixed(f) {}
};
static unsigned long long products, additions, scalings;
static Number operator-(Number a) { return Number(-a.value, a.fixed); }
static Number operator+(Number a, Number b) {
    if (a.fixed && b.fixed) return Number(a.value + b.value);
    if (a.fixed && a.value == 0.0) return b;
    if (b.fixed && b.value == 0.0) return a;
    ++additions;
    return Number(a.value + b.value, false);
}
static Number operator-(Number a, Number b) {
    if (a.fixed && b.fixed) return Number(a.value - b.value);
    if (b.fixed && b.value == 0.0) return a;
    if (a.fixed && a.value == 0.0) return -b;
    ++additions;
    return Number(a.value - b.value, false);
}
static Number operator*(Number a, Number b) {
    if (a.fixed && b.fixed) return Number(a.value * b.value);
    if (a.fixed || b.fixed) {
        double constant = a.fixed ? a.value : b.value;
        if (constant == 0.0) return Number(0.0);
        if (constant != 1.0 && constant != -1.0) ++scalings;
    } else {
        ++products;
    }
    return Number(a.value * b.value, false);
}

#define restrict __restrict
#define FASTCD_REAL Number
#define FASTCD_ZERO Number(0.0)
#define FASTCD_NAME(name) measured_##name
#define FASTCD_QL_FULL_LEAF 16
#define FASTCD_QL_COMPLEX_LEAF 8
#include "generated/quasilinear_kernels.h"
#include "quasilinear_impl.h"

static void report(const char* kind, size_t n) {
    std::printf("{\"kind\":\"%s\",\"dimension\":%zu,"
        "\"M\":%llu,\"A\":%llu,\"S\":%llu,\"T\":%llu}\n",
        kind, n, products, additions, scalings, products + additions + scalings);
}
int main() {
    for (size_t n = 1; n <= 256; n *= 2) {
        std::vector<Number> x(n), y(n), xi(n), yi(n), out(n), oi(n), scratch(8*n);
        for (size_t i = 0; i < n; ++i) {
            x[i] = Number(0.25, false); y[i] = Number(0.5, false);
            xi[i] = Number(0.125, false); yi[i] = Number(-0.25, false);
        }
        products = additions = scalings = 0;
        measured_ql_mul_recursive(out.data(), x.data(), y.data(), n, scratch.data());
        report("full", n);
        products = additions = scalings = 0;
        measured_complex_cross(out.data(), oi.data(), x.data(), xi.data(),
            y.data(), yi.data(), n, scratch.data());
        report("complex", n);
    }
}
'''


def expected(kind: str, n: int) -> tuple[int, int, int]:
    level = n.bit_length() - 1
    if kind == "full":
        if n <= 8:
            return {1: (1, 0, 0), 2: (4, 2, 0), 4: (16, 12, 0),
                    8: (58, 62, 0)}[n]
        return ((36 * level - 69) * n // 8 + 10,
                (76 * level - 195) * n // 8 + 26,
                (8 * level - 27) * n // 8 + 4)
    if n <= 4:
        return (18, 24, 0) if n == 4 else (0, 0, 0)
    return ((36 * level - 81) * n // 4 + 18,
            (76 * level - 167) * n // 4 + 36,
            (8 * level - 19) * n // 4 + 4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    with tempfile.TemporaryDirectory(prefix="fastcd-counts-") as tmp:
        source, binary = Path(tmp) / "counts.cpp", Path(tmp) / "counts"
        source.write_text(PROBE)
        subprocess.run(compiler + ["-std=c++11", "-O0", "-I", str(root / "src"),
                                   str(source), "-o", str(binary)], check=True)
        result = subprocess.run([str(binary)], check=True, text=True,
                                capture_output=True)
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    for row in rows:
        actual = tuple(row[key] for key in ("M", "A", "S"))
        assert actual == expected(row["kind"], row["dimension"]), row
    files = ("src/quasilinear_impl.h", "src/generated/quasilinear_kernels.h",
             "tools/generate_kernels.py", "tools/check_quasilinear_counts.py")
    document = {
        "status": "ok", "method": "operator-instrumented scalar C++ instantiation",
        "convention": "paper arithmetic; fixed-zero operations and signs excluded",
        "compiler": compiler, "counts": rows,
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                          for name in files},
    }
    if args.output:
        args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
