![fastCD](https://git.tanuki-cd.com/algebraity/fastCD/raw/branch/main/assets/fastCD.png)

# fastCD - v1.2.0

fastCD is a lean C kernel for Cayley-Dickson multiplication over real and complex numbers and finite fields, with a NumPy-compatible Python extension, written for researchers and engineers in need of faster Cayley-Dickson multiplication. The key advantage of fastCD is that it implements a **new O(N log N) algorithm** which, to the author's knowledge, has never been presented before. See the [accompanying paper](https://arxiv.org/abs/2609.11588) for more information on the mathematical foundations and rigorous derivation of correctness and time complexity. fastCD may not be completely suitable for every application, but it is licensed under the [MIT License](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/LICENSE) so that it may be improved upon and adapted to the user's needs completely, with no questions asked.

In addition to this new quasilinear algorithm, both direct Cayley-Dickson multiplication and an algorithm due to [Cariow and Cariowa](https://doi.org/10.15199/48.2015.02.09) are also included in the library for real coefficients; complex and finite-field coefficients use the direct and quasilinear algorithms. The default multiplication algorithm for each dimension has been selected based on explicit single-core benchmarking for both single and batched multiplication. Quasilinear has the best asymptotic complexity of these three algorithms, but it is not the fastest at every lower dimension. The direct and Cariow-Cariowa algorithms are also used for some low dimensions over the real numbers; see [Benchmark](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/benchmark.md) for more information on this.

The canonical source repository is
[git.tanuki-cd.com/algebraity/fastCD](https://git.tanuki-cd.com/algebraity/fastCD).

## Convention

```text
A_0 = F,   A_n = A_(n-1) x A_(n-1),   (a,b)(c,d) = (ac - conjugate(d) b, d a + b conjugate(c))
```

The coefficient field `F` is the real numbers by default, the complex numbers, or an
odd-characteristic finite field `F_(p^d)`. An element is stored as a contiguous array of
`N = 2^n` coefficients in the recursive standard basis (`double` over R, `complex128` over C,
and `d` `uint32` residues per coefficient over `F_(p^d)`), where coordinate zero is the scalar
and the two halves of the array are the pair `(a, b)`. [Conventions and results](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/mathematics.md)
should be read for full information on conventions and a brief explanation of the mathematical
foundations of fastCD.

## Using the C library

The C library can be used by including the `fastcd.h` header file and using fastCD functions
within a normal C program; `<fastcd/complex.h>` and `<fastcd/finite.h>` declare the complex and
finite-field arithmetic. Here is an example of supported usage over the real numbers.

```c
#include <stdlib.h>
#include <fastcd/fastcd.h>

int main(void) {
    const size_t N = 16;
    double x[16] = {1.0, 2.0};
    double y[16] = {3.0, 4.0};
    double product[16];
    size_t length;

    if (fastcd_mul_workspace_size(N, &length) != FASTCD_SUCCESS) return 1;
    double* scratch = length == 0 ? NULL : malloc(length * sizeof(*scratch));
    if (length != 0 && scratch == NULL) return 1;

    FastCDWorkspaceF64 workspace = {scratch, length};
    FastCDStatus status = fastcd_mul_f64(product, x, y, N,
            length == 0 ? NULL : &workspace);
    free(scratch);
    return status == FASTCD_SUCCESS ? 0 : 1;
}
```

For more information on how to get started using fastCD in C and to familiarize yourself with the API, take a look at [C quick start](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/quickstart-c.md) and [C API reference](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/c-api.md).

## Using Python

To use `pyfastcd`, make sure you have CPython 3.10 or newer and NumPy 1.26 or newer installed. Building from source additionally requires a C11 compiler, Python development headers,
and CMake 3.18 or newer.

To install the pip package, run this command:

```sh
python -m pip install .
```

After doing so, you may use the `Algebra`, `FiniteField`, `Element`, and `ElementArray` classes directly in Python after importing `pyfastcd`. Here's an example of what you can do over the real numbers; `Algebra(dimension=8, char=13, deg=5)` or `Algebra(dimension=8, char=0, deg=2)` selects octonions over `F_(13^5)` or over the complex numbers.

```python
import numpy as np
import pyfastcd as fcd

O = fcd.Algebra(dimension=8)
a = O([1, 2, 3, 4, 5, 6, 7, 8])
b = O.basis(2)

product = a * b
square = a ** 2
commutator = a.commutator(b)

x = O.asarray(np.ones((1000, 8)))
y = x * b                                  # batch product, leading axes broadcast
z = O.multiply(x, b, algorithm="quasilinear")
```

The `Element` class is immutable and represents a unique Cayley-Dickson element, while `ElementArray` stores a mutable array of Elements and processes batches of shape `(..., N)`
(or `(..., N, d)` over an extension field) in compiled code. See the [Python quick start](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/quickstart-python.md) and
the [Python API reference](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/python-api.md) for more information on how each class works.

## Building the C library

A C11 compiler and CMake 3.18 or newer are required to build. Unix-like systems link
the system math library. From the repository root, run:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
cmake -E chdir build ctest --output-on-failure
cmake --install build --prefix /desired/prefix
```

The default configuration builds the static library, the test suite, and the
benchmark runner together. `-DBUILD_SHARED_LIBS=ON` builds a shared library, and
`-DBUILD_TESTING=OFF -DFASTCD_BUILD_BENCHMARK=OFF` builds the library alone.
The installation provides the `<fastcd/*.h>` headers, `libfastcd`, the license, and
a CMake package: use `find_package(fastcd CONFIG REQUIRED)` and link
`fastcd::fastcd`.

## Benchmark

All timings below were measured on 2026-11-04 on a Dell PowerEdge T440 with an Intel Xeon Gold 6148, one logical CPU per run, GCC 16.2.1 and Linux 7.2.11, from the v1.2.0 source. The archived build records list the SHA-256 of every measured source file and the build identifier, and they match the v1.2.0 tree; git describes the run as `c500d40-dirty` because the 1.2.0 version string and documentation were committed after the measurement. The host is shared, and no governor, turbo, or SMT settings were changed. Only the C kernel is timed, with storage already allocated: Python overhead, allocation, and verification are outside the timed region. Every product of every operand pair was checked against an independent implementation before timing. The raw observations, environment, commands, and build records are archived in [benchmark-results/t440-20261104/](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/benchmark-results/t440-20261104/), and the [benchmark documentation](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/benchmark.md) describes the protocol and how to rerun it.

### Real coefficients

Mean nanoseconds per product with two-sided 95% Student-t intervals over 24 timing blocks of 10,000 operand pairs, in batch mode (one call for all 10,000 products, two products per SIMD instruction where the kernels support it) and single mode (one validated call per product). All 600,000 verification checks passed with zero mismatches. `fastest mean` names the lowest mean, or a tie when the paired per-block ratio of the two lowest means does not separate them, and `default` is the algorithm that automatic selection uses. The automatic choices agree with this run; at dimension 4, single products are a tie between direct and quasilinear, and quasilinear is kept.

#### Batch products (ns per product)

| N | direct | cariow_cariowa | quasilinear | fastest mean | default |
| ---: | --- | --- | --- | --- | --- |
| 2 | 1.33 [1.31, 1.35] | 1.58 [1.54, 1.62] | 1.03 [1.02, 1.04] | quasilinear | quasilinear |
| 4 | 3.89 [3.83, 3.95] | 4.68 [4.59, 4.76] | 3.80 [3.73, 3.87] | quasilinear | quasilinear |
| 8 | 12.58 [12.37, 12.78] | 14.45 [14.25, 14.64] | 18.03 [17.72, 18.35] | direct | direct |
| 16 | 56.72 [55.77, 57.67] | 51.01 [50.22, 51.79] | 56.06 [55.37, 56.74] | cariow_cariowa | cariow_cariowa |
| 32 | 262.45 [257.07, 267.82] | 237.29 [234.06, 240.52] | 178.47 [174.59, 182.34] | quasilinear | quasilinear |
| 64 | 1145.54 [1119.18, 1171.90] | 1013.19 [994.89, 1031.49] | 453.45 [445.57, 461.33] | quasilinear | quasilinear |
| 128 | 4680.78 [4577.72, 4783.84] | 4978.12 [4879.90, 5076.33] | 1238.11 [1219.50, 1256.71] | quasilinear | quasilinear |
| 256 | 18671.60 [18492.21, 18850.98] | 17894.54 [17496.60, 18292.47] | 3005.67 [2962.63, 3048.72] | quasilinear | quasilinear |
| 512 | 71373.60 [70472.12, 72275.07] | 65361.92 [64656.19, 66067.66] | 6758.91 [6622.16, 6895.66] | quasilinear | quasilinear |
| 1024 | 282743.07 [280338.13, 285148.01] | 422651.68 [411684.98, 433618.38] | 16113.68 [15991.54, 16235.82] | quasilinear | quasilinear |

#### Single products (ns per product)

| N | direct | cariow_cariowa | quasilinear | fastest mean | default |
| ---: | --- | --- | --- | --- | --- |
| 2 | 16.17 [15.93, 16.40] | 17.39 [16.88, 17.90] | 15.57 [15.29, 15.84] | quasilinear | quasilinear |
| 4 | 18.26 [18.04, 18.49] | 20.32 [20.00, 20.64] | 18.52 [18.17, 18.88] | tie (direct, quasilinear) | quasilinear |
| 8 | 28.75 [28.22, 29.29] | 41.40 [40.66, 42.13] | 37.63 [37.05, 38.20] | direct | direct |
| 16 | 82.05 [80.88, 83.23] | 114.32 [112.88, 115.75] | 113.73 [111.20, 116.27] | direct | direct |
| 32 | 300.37 [295.88, 304.86] | 464.58 [455.96, 473.20] | 215.83 [213.02, 218.64] | quasilinear | quasilinear |
| 64 | 1144.89 [1128.71, 1161.07] | 1918.87 [1879.66, 1958.08] | 500.59 [492.62, 508.57] | quasilinear | quasilinear |
| 128 | 4392.20 [4322.03, 4462.36] | 8313.62 [7963.52, 8663.73] | 1244.21 [1224.54, 1263.89] | quasilinear | quasilinear |
| 256 | 17607.86 [17418.09, 17797.63] | 29052.88 [28576.28, 29529.49] | 3227.60 [3173.17, 3282.03] | quasilinear | quasilinear |
| 512 | 69288.27 [68796.10, 69780.45] | 107763.51 [106749.81, 108777.20] | 7225.82 [7108.12, 7343.52] | quasilinear | quasilinear |
| 1024 | 285131.22 [283178.93, 287083.52] | 420241.56 [417302.94, 423180.17] | 16319.92 [16153.98, 16485.86] | quasilinear | quasilinear |

### Finite fields and complex coefficients

The same protocol compares the direct and quasilinear algorithms over the nine default finite fields and over C, with 1,000 operand pairs per cell and 24 timing blocks. Every pair is verified before timing, against independent schoolbook polynomial arithmetic over the finite fields and against a long-double oracle over C; all 100,000 pairs passed. Each cell below gives the quasilinear mean in nanoseconds per single product and, in parentheses, the paired direct/quasilinear ratio, so a ratio above 1 favors quasilinear. Direct is faster at dimension 2 everywhere and at dimension 4 for most fields; from dimension 8 upward quasilinear is faster in every cell, and automatic selection uses it over every finite field and over C. Batch mode is within a few percent of single mode from dimension 16 upward. Both modes, every dimension and the confidence intervals are in `domains.md` in the archive.

#### Single products over finite fields and C: quasilinear ns per product (direct/quasilinear)

| Field | N = 16 | N = 64 | N = 256 | N = 1024 |
| --- | ---: | ---: | ---: | ---: |
| F_3 | 303.8 (2.7x) | 829.4 (17.0x) | 917.3 (242x) | 2301.1 (1541x) |
| F_5 | 303.5 (2.6x) | 815.3 (16.2x) | 1206.8 (173x) | 3987.6 (842x) |
| F_13 | 307.7 (2.4x) | 866.7 (14.3x) | 1493.8 (131x) | 5276.1 (597x) |
| F_(3^5) | 1013.3 (15.2x) | 1735.0 (142x) | 5536.5 (716x) | 19127.6 (3292x) |
| F_(5^2) | 576.9 (5.5x) | 1875.6 (27.3x) | 7214.3 (113x) | 36872.8 (356x) |
| F_(13^5) | 3581.7 (4.7x) | 4700.3 (57.0x) | 18798.5 (228x) | 98388.9 (698x) |
| F_65521 | 307.9 (2.3x) | 784.5 (15.1x) | 2737.7 (69.3x) | 13467.8 (226x) |
| F_(65521^2) | 1259.1 (2.5x) | 2859.2 (17.8x) | 11929.9 (68.0x) | 58161.9 (219x) |
| F_2147483647 | 304.5 (2.3x) | 794.0 (14.6x) | 2697.7 (68.4x) | 13236.6 (225x) |
| C | 310.2 (1.6x) | 1628.3 (4.8x) | 8310.5 (15.0x) | 44971.2 (44.9x) |

### Comparison with FLINT

FLINT (Fast Library for Number Theory) is an independent exact-arithmetic C library. The comparison times fastCD's finite-field arithmetic against FLINT 3.6.0 on two workloads with identical operands and defining polynomials, 1,000 operand pairs per cell and 24 paired blocks, every pair verified against the same independent implementation; all 132,000 pairs passed. The scalar workload is one field multiplication (`fastcd_ff_field_mul` against `nmod_mul` for prime fields or `fq_nmod_mul` for extensions). The CD workload is one complete product at each dimension, where both sides evaluate the same quasilinear schedule and only the coefficient arithmetic differs: fastCD's packed kernels against FLINT's field operations. Ratios are paired geometric means of FLINT time over fastCD time, so a ratio above 1 favors fastCD. Every dimension and the intervals for the CD cells are in `flint.md` in the archive.

#### Scalar field multiplication: ns per product

| Field | fastCD | FLINT | FLINT/fastCD [95% CI] |
| --- | ---: | ---: | --- |
| F_3 | 1.96 | 7.00 | 3.56 [3.54, 3.58] |
| F_5 | 1.96 | 6.98 | 3.56 [3.52, 3.60] |
| F_13 | 1.96 | 7.00 | 3.57 [3.54, 3.60] |
| F_(3^5) | 54.51 | 156.22 | 2.87 [2.85, 2.88] |
| F_(5^2) | 10.24 | 64.12 | 6.27 [6.14, 6.39] |
| F_(13^5) | 60.24 | 249.13 | 4.14 [4.12, 4.16] |
| F_65521 | 1.99 | 6.94 | 3.49 [3.47, 3.52] |
| F_(65521^2) | 10.03 | 70.52 | 7.05 [6.87, 7.23] |
| F_2147483647 | 1.96 | 6.91 | 3.52 [3.49, 3.56] |
| F_(3^12) | 261.58 | 618.20 | 2.36 [2.35, 2.38] |
| F_(13^12) | 331.16 | 647.50 | 1.96 [1.95, 1.97] |
| F_(2147483647^12) | 348.56 | 1279.71 | 3.67 [3.64, 3.71] |

#### Complete CD multiplication: fastCD ns / FLINT ns (FLINT/fastCD)

| Field | N = 16 | N = 64 | N = 256 | N = 1024 |
| --- | ---: | ---: | ---: | ---: |
| F_3 | 307 / 2379 (7.7x) | 835 / 17761 (21.4x) | 922 / 105030 (114x) | 2346 / 554734 (237x) |
| F_5 | 305 / 2527 (8.3x) | 824 / 18829 (23.0x) | 1273 / 111903 (88.1x) | 4038 / 591983 (147x) |
| F_13 | 306 / 2604 (8.5x) | 909 / 19523 (21.6x) | 1465 / 115965 (79.2x) | 5213 / 616155 (118x) |
| F_(3^5) | 1061 / 38626 (36.4x) | 1717 / 283232 (165x) | 5830 / 1667425 (286x) | 19324 / 8732460 (452x) |
| F_(5^2) | 586 / 18240 (31.2x) | 1907 / 138270 (72.5x) | 7594 / 828234 (109x) | 37045 / 4318879 (117x) |
| F_(13^5) | 3662 / 52486 (14.4x) | 4808 / 383334 (79.8x) | 20401 / 2225527 (110x) | 96623 / 11921825 (123x) |
| F_65521 | 300 / 2625 (8.7x) | 789 / 19699 (25.0x) | 2776 / 116827 (42.2x) | 12968 / 619622 (47.8x) |
| F_(65521^2) | 1223 / 17933 (14.7x) | 2889 / 138469 (48.1x) | 11924 / 785990 (66.3x) | 57588 / 4146121 (72.2x) |
| F_2147483647 | 304 / 2629 (8.7x) | 833 / 19726 (23.8x) | 2825 / 115862 (41.3x) | 12634 / 618896 (49.0x) |
| F_(3^12) | 4778 / 122805 (25.7x) | 4903 / 877191 (179x) | 17409 / 5111730 (294x) | 75023 / 26939417 (360x) |
| F_(13^12) | 59962 / 125842 (2.1x) | 42495 / 906465 (21.4x) | 96463 / 5290183 (54.9x) | 359604 / 27450258 (76.4x) |
| F_(2147483647^12) | 15464 / 242336 (15.7x) | 28443 / 1677399 (59.1x) | 102107 / 9797402 (96.2x) | 457545 / 51368227 (112x) |

## Documentation

- [C quick start](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/quickstart-c.md) and [C API reference](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/c-api.md)
- [Python quick start](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/quickstart-python.md) and
  [Python API reference](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/python-api.md)
- [Conventions and results](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/mathematics.md)
- [Benchmark](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/benchmark.md)

## License and Attribution

All code is released under the [MIT License](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/LICENSE) and is presented with no warranty or guarantees. You may use it in any way for any purpose and are not restricted by the original author of the code. To cite the software, use the metadata in [CITATION.cff](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/CITATION.cff). The changelog is in [CHANGELOG.md](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/CHANGELOG.md).

The quasilinear algorithm is due to Harrison Lemley (me!), [*Quasilinear
multiplication in the Cayley–Dickson tower*](https://arxiv.org/abs/2609.11588) . The
Cariow–Cariowa algorithm follows Aleksandr Cariow and Galina Cariowa,
[*An unified approach for developing rationalized algorithms for hypercomplex
number multiplication*](https://doi.org/10.15199/48.2015.02.09), *Przegląd
Elektrotechniczny* 91(2), 36–39 (2015).
