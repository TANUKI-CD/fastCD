#!/usr/bin/env python3
"""Instrument actual scalar Cariow-Cariowa leaves and table kernels through 1024.

Optional C++ development check only; the library remains allocation-free C11.
Counts follow docs/mathematics.md: constant-zero operations and signs are free.
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

PROBE = r'''#include <cstdio>
#include <cstdint>
#include <vector>
struct Number {
    double value;
    bool fixed;
    Number(double v = 0.0, bool f = true) : value(v), fixed(f) {}
};
static unsigned long long products, additions, scalings, executed_additions;
static Number operator-(Number a) { return Number(-a.value, a.fixed); }
static Number operator+(Number a, Number b) {
    ++executed_additions;
    if (a.fixed && b.fixed) return Number(a.value + b.value);
    if (a.fixed && a.value == 0.0) return b;
    if (b.fixed && b.value == 0.0) return a;
    ++additions;
    return Number(a.value + b.value, false);
}
static Number operator-(Number a, Number b) {
    ++executed_additions;
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


static size_t cc_expand_word(uint64_t bits, uint8_t* positions) {
    size_t count = 0;
    for (unsigned b = 0; b < 64; b++)
        if ((bits >> b) & 1) positions[count++] = (uint8_t)b;
    return count;
}
#define restrict __restrict
#define FASTCD_REAL Number
#define FASTCD_ZERO Number(0.0)
#define FASTCD_NAME(name) measured_##name
#define FASTCD_NOINLINE
#include "generated/cariow_cariowa_masks.h"
#include "generated/cariow_cariowa_kernels.h"
#include "cariow_cariowa_impl.h"
static const uint64_t* masks(size_t n) {
    switch (n) {
        case 16: return &fastcd_cc_mask_16[0][0];
        case 32: return &fastcd_cc_mask_32[0][0];
        case 64: return &fastcd_cc_mask_64[0][0];
        case 128: return &fastcd_cc_mask_128[0][0];
        case 256: return &fastcd_cc_mask_256[0][0];
        case 512: return &fastcd_cc_mask_512[0][0];
        default: return &fastcd_cc_mask_1024[0][0];
    }
}
int main() {
    for (size_t n = 1; n <= 1024; n *= 2) {
        std::vector<Number> x(n), y(n), out(n), scratch(2*n);
        for (size_t i = 0; i < n; i++) {
            x[i] = Number(0.25, false); y[i] = Number(0.5, false);
        }
        products = additions = scalings = executed_additions = 0;
        if (n > FASTCD_CC_LEAF) {
            measured_cc_table_mul(out.data(), x.data(), y.data(), n,
                                  masks(n), scratch.data());
        } else switch (n) {
            case 1: measured_cc_mul_1(out.data(), x.data(), y.data()); break;
            case 2: measured_cc_mul_2(out.data(), x.data(), y.data()); break;
            case 4: measured_cc_mul_4(out.data(), x.data(), y.data()); break;
            case 8: measured_cc_mul_8(out.data(), x.data(), y.data()); break;
            case 16: measured_cc_mul_16(out.data(), x.data(), y.data()); break;
            case 32: measured_cc_mul_32(out.data(), x.data(), y.data()); break;
            case 64: measured_cc_mul_64(out.data(), x.data(), y.data()); break;
        }
        size_t groups = 0;
        if (n > FASTCD_CC_LEAF) {
            size_t words = n > 64 ? n / 64 : 1;
            const uint64_t* table = masks(n);
            for (size_t k = 0; k < n; k++) {
                for (size_t g = 0; g < words; g += FASTCD_CC_WORD_BLOCK) {
                    uint64_t present = 0;
                    for (size_t w = g; w < words && w - g < FASTCD_CC_WORD_BLOCK; w++)
                        present |= table[k * words + w];
                    if (present) groups++;
                }
            }
        }
        std::printf("{\"dimension\":%zu,\"M\":%llu,\"A\":%llu,\"E\":%llu,"
                    "\"A_executed\":%llu,\"zero_additions\":%llu,"
                    "\"nonempty_word_groups\":%zu}\n",
                    n, products, additions, scalings, executed_additions,
                    executed_additions - additions, groups);
    }
}
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--leaf", type=int, choices=(8, 16, 32, 64), default=64)
    parser.add_argument("--chains", type=int, choices=(4, 8), default=4)
    parser.add_argument("--words", type=int, choices=(1, 2, 4), default=4)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    with tempfile.TemporaryDirectory(prefix="fastcd-cc-counts-") as tmp:
        source, binary = Path(tmp) / "counts.cpp", Path(tmp) / "counts"
        source.write_text(PROBE)
        subprocess.run(compiler + ["-std=c++17", "-O0", "-I", str(root / "src"),
                                  f"-DFASTCD_CC_LEAF={args.leaf}",
                                  f"-DFASTCD_CC_TABLE_CHAINS={args.chains}",
                                  f"-DFASTCD_CC_WORD_BLOCK={args.words}",
                                  str(source), "-o", str(binary)], check=True)
        result = subprocess.run([str(binary)], check=True, text=True,
                                capture_output=True)
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    for row in rows:
        n = row["dimension"]
        expected = ((2, 1, 1) if n == 1 else
                    (n * (n - 1) // 2 + 2,
                     3 * n * (n.bit_length() - 1) + (n * (n - 3) + 4) // 2,
                     3 if n == 2 else 2 * n))
        assert tuple(row[k] for k in ("M", "A", "E")) == expected, row
        if n > args.leaf:
            assert row["A_executed"] == row["A"] + args.chains * row[
                "nonempty_word_groups"], row
            row["A_executed_before"] = row["A"] + args.chains * n
        else:
            row["A_executed_before"] = row["A_executed"]
    files = ["tools/check_cariow_counts.py", "src/cariow_cariowa.c",
             "src/cariow_cariowa_impl.h",
             "src/generated/cariow_cariowa_kernels.h",
             "src/generated/cariow_cariowa_masks.h", "tools/generate_kernels.py"]
    document = {"status": "ok", "method": "operator-instrumented actual scalar kernels",
                "leaf": args.leaf, "chains": args.chains,
                "word_block": args.words, "counts": rows,
                "source_sha256": {name: hashlib.sha256(
                    (root / name).read_bytes()).hexdigest() for name in files},
                "scope": "uniform CC schedule; optional outer doubling is unchanged",
                "A_convention": "published model excludes known-zero operations",
                "A_executed_convention": ("source-level executed binary additions; "
                                          "includes exact zero operations, "
                                          "not a native-instruction count"),
                "small_dimension_note": ("N1 unchanged generated schedule has "
                                         "M2/A1/E1; N2 E3")}
    if args.output:
        args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()
