![fastCD](https://git.tanuki-cd.com/algebraity/fastCD/raw/branch/main/assets/fastCD.png)

# fastCD - v1.3.1

fastCD is a lean C kernel for Cayley–Dickson multiplication over real and complex numbers and finite fields, with a NumPy-compatible Python extension, written for researchers and engineers in need of faster Cayley–Dickson multiplication. The key advantage of fastCD is that it implements a **new O(N log N) algorithm** which, to the author's knowledge, has never been presented before. See the [accompanying paper](https://arxiv.org/abs/2609.11588) for more information on the mathematical foundations and rigorous derivation of correctness and time complexity. fastCD may not be completely suitable for every application, but it is licensed under the [MIT License](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/LICENSE) so that it may be improved upon and adapted to the user's needs completely, with no questions asked.

Besides the quasilinear algorithm, fastCD includes direct Cayley–Dickson multiplication and the algorithm of [Cariow and Cariowa](https://doi.org/10.15199/48.2015.02.09). All three are available for real coefficients, while complex and finite-field coefficients use direct and quasilinear. Quasilinear has the best asymptotic complexity of the three, but it is not the fastest at every low dimension, so the default algorithm for each dimension was chosen from single-core benchmarks of single and batched products. Over the real numbers, direct is the default for single products at dimension 4, and direct or Cariow–Cariowa is the default at dimensions 8 and 16; see [Benchmark](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/benchmark.md).

In an exact-reference accuracy study, the median floating-point error of the quasilinear product on pooled Gaussian and uniform inputs is within a factor of two of direct multiplication at every tested dimension; see [Numerical stability](#numerical-stability).

The canonical source repository is
[git.tanuki-cd.com/algebraity/fastCD](https://git.tanuki-cd.com/algebraity/fastCD).

## Convention

```text
A_0 := F,   A_n := A_(n-1) x A_(n-1),   (a,b)(c,d) := (ac - conjugate(d) b, d a + b conjugate(c))
```

The coefficient field `F` is the real numbers (the default), the complex numbers, or an
odd-characteristic finite field `F_(p^d)`. An element is a contiguous array of `N := 2^n`
coefficients in the recursive standard basis: coordinate zero is the scalar part, and the two
halves of the array are the pair `(a, b)`. Each coefficient is one `double` over R, two
`double`s with the real part first over C (`complex128` in NumPy), and `d` `uint32` residues
over `F_(p^d)`. [Conventions and results](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/mathematics.md)
gives the full conventions and explains the algorithms: the basis signs, the quadratic form,
how the quasilinear recursion works, and the exact operation counts.

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

fastCD can be used in Python via the package `pyfastcd`. It requires CPython 3.10 or newer
and NumPy 1.26 or newer. To install it from PyPI with `pip`, run:

```sh
pip install pyfastcd
```

The package can also be built from source from this repository. This also requires a C11
compiler, Python development headers, and CMake 3.18 or newer. To build and install it
from the repository root, run:

```sh
python -m pip install .
```

After importing `pyfastcd`, you can use the `Algebra`, `FiniteField`, `Element`, and `ElementArray` classes. Here's an example of what you can do over the real numbers; `Algebra(dimension=8, char=13, deg=5)` or `Algebra(dimension=8, char=0, deg=2)` selects octonions over `F_(13^5)` or over the complex numbers.

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

An `Element` is one immutable Cayley–Dickson element. An `ElementArray` holds a batch of elements in a NumPy coefficient array of shape `(..., N)`
(or `(..., N, d)` over an extension field) and evaluates batch operations in compiled code. See the [Python quick start](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/quickstart-python.md) and
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

All timings below were measured on 2026-11-09 on a Dell PowerEdge T440 with an Intel Xeon Gold 6148, using one logical CPU per run, GCC 16.2.1 and Linux 7.2.11. The host is shared, and no governor, turbo, or SMT settings were changed.

Only the C kernel is timed, with storage already allocated: Python overhead, allocation, and verification are outside the timed region. Every product of every operand pair was also checked against an independent implementation.

The measured source is commit `33addc9`, built from a clean checkout; the archived build records list the SHA-256 of every measured source file. The raw observations, environment, commands, and build records are archived in [benchmark-results/t440-20261109/](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/benchmark-results/t440-20261109/), and the [benchmark documentation](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/benchmark.md) describes the protocol and how to rerun it.

### Real coefficients

Mean nanoseconds per product with two-sided 95% Student-t intervals over 24 timing blocks of 10,000 operand pairs, in batch mode (one call for all 10,000 products, two products per SIMD instruction where the kernels support it) and single mode (one validated call per product). All 600,000 verification checks passed with zero mismatches. `fastest mean` names the lowest mean, or a tie when the paired per-block ratio of the two lowest means does not separate them, and `default` is the algorithm that automatic selection uses. The automatic choices agree with every cell of this run; single products at dimension 2 are a tie between direct and quasilinear, and the default there is quasilinear.

#### Batch products (ns per product)

| N | direct | cariow_cariowa | quasilinear | fastest mean | default |
| ---: | --- | --- | --- | --- | --- |
| 2 | 1.33 [1.31, 1.35] | 1.59 [1.57, 1.62] | 1.03 [1.02, 1.04] | quasilinear | quasilinear |
| 4 | 3.94 [3.89, 3.98] | 4.68 [4.62, 4.73] | 3.81 [3.75, 3.86] | quasilinear | quasilinear |
| 8 | 12.74 [12.64, 12.85] | 14.57 [14.36, 14.79] | 18.28 [18.00, 18.56] | direct | direct |
| 16 | 57.40 [56.52, 58.27] | 52.74 [51.68, 53.80] | 55.10 [54.16, 56.04] | cariow_cariowa | cariow_cariowa |
| 32 | 268.23 [264.09, 272.37] | 237.26 [233.80, 240.73] | 190.71 [188.03, 193.39] | quasilinear | quasilinear |
| 64 | 1171.27 [1141.31, 1201.22] | 1013.51 [990.61, 1036.42] | 478.38 [468.63, 488.13] | quasilinear | quasilinear |
| 128 | 4659.72 [4568.53, 4750.90] | 5194.87 [5030.70, 5359.04] | 1282.23 [1255.74, 1308.72] | quasilinear | quasilinear |
| 256 | 18411.96 [18131.73, 18692.19] | 18509.70 [18092.51, 18926.89] | 3072.16 [3024.09, 3120.22] | quasilinear | quasilinear |
| 512 | 70453.07 [69518.40, 71387.74] | 70924.53 [69837.36, 72011.69] | 6940.28 [6843.35, 7037.21] | quasilinear | quasilinear |
| 1024 | 276258.74 [274098.13, 278419.34] | 426435.28 [423564.06, 429306.50] | 16196.47 [15993.55, 16399.39] | quasilinear | quasilinear |

#### Single products (ns per product)

| N | direct | cariow_cariowa | quasilinear | fastest mean | default |
| ---: | --- | --- | --- | --- | --- |
| 2 | 15.95 [15.70, 16.19] | 16.52 [16.31, 16.73] | 16.15 [15.81, 16.50] | tie (direct, quasilinear) | quasilinear |
| 4 | 17.70 [17.47, 17.93] | 19.67 [19.28, 20.05] | 18.36 [18.07, 18.66] | direct | direct |
| 8 | 28.36 [26.88, 29.83] | 40.41 [39.93, 40.88] | 34.56 [33.87, 35.25] | direct | direct |
| 16 | 86.52 [83.22, 89.81] | 113.79 [111.14, 116.43] | 106.36 [104.37, 108.35] | direct | direct |
| 32 | 304.51 [301.10, 307.91] | 467.38 [457.54, 477.22] | 230.14 [226.99, 233.29] | quasilinear | quasilinear |
| 64 | 1174.60 [1136.91, 1212.29] | 1921.76 [1874.63, 1968.89] | 527.82 [519.25, 536.40] | quasilinear | quasilinear |
| 128 | 4577.54 [4510.54, 4644.55] | 8450.41 [8127.34, 8773.48] | 1341.54 [1319.44, 1363.65] | quasilinear | quasilinear |
| 256 | 17751.24 [17557.41, 17945.07] | 29374.01 [28802.52, 29945.50] | 3284.33 [3228.72, 3339.93] | quasilinear | quasilinear |
| 512 | 69656.86 [69120.36, 70193.36] | 113587.77 [112582.42, 114593.12] | 7423.90 [7345.57, 7502.23] | quasilinear | quasilinear |
| 1024 | 276236.76 [274697.43, 277776.09] | 435957.96 [423208.54, 448707.38] | 16073.92 [15903.01, 16244.83] | quasilinear | quasilinear |

### Finite fields and complex coefficients

The same protocol compares direct and quasilinear multiplication over C and over the nine finite fields in the table below, which are the field benchmark's default set: the prime fields F_3, F_5, F_13, F_65521 and F_2147483647 and the extension fields F_(3^5), F_(5^2), F_(13^5) and F_(65521^2). Each cell uses 1,000 operand pairs and 24 timing blocks. Every pair is verified before timing, against independent schoolbook polynomial arithmetic over the finite fields and against a long-double reference over C; all 100,000 pairs passed. Each cell below gives the quasilinear mean in nanoseconds per single product and, in parentheses, the paired direct/quasilinear ratio, so a ratio above 1 favors quasilinear. Direct is faster at dimension 2 everywhere and at dimension 4 over C and five of the nine fields. From dimension 8 upward quasilinear is faster in every cell except two ties: F_(3^5) at dimension 64 and single products over F_(65521^2) at dimension 16. Automatic selection uses direct for `N <= 4` and quasilinear otherwise, over C and every finite field. From dimension 16 upward, batch and single means differ by at most 6%, except over C at dimension 16, where batching saves 10%. Each cell runs in one process, so its interval covers variation between timing blocks but not between processes: in repeated runs on this host, individual cells from dimension 64 upward differed by up to 17%, and direct/quasilinear ratios by up to 8%. Both modes, every dimension and the confidence intervals are in [domains.md](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/benchmark-results/t440-20261109/domains.md) in the archive.

#### Single products over finite fields and C: quasilinear ns per product (direct/quasilinear)

| Field | N = 16 | N = 64 | N = 256 | N = 1024 |
| --- | ---: | ---: | ---: | ---: |
| F_3 | 300.2 (1.6x) | 817.8 (2.2x) | 1003.7 (2.6x) | 2233.6 (17.0x) |
| F_5 | 303.8 (1.5x) | 852.6 (2.2x) | 1238.8 (6.5x) | 3986.8 (32.9x) |
| F_13 | 298.8 (1.5x) | 785.0 (2.3x) | 1437.1 (7.7x) | 4890.7 (34.6x) |
| F_(3^5) | 1013.1 (1.1x) | 1725.5 (1.0x) | 5464.3 (2.2x) | 18978.3 (8.2x) |
| F_(5^2) | 572.2 (1.3x) | 2048.9 (4.0x) | 7947.7 (15.2x) | 36455.1 (50.4x) |
| F_(13^5) | 3528.9 (1.2x) | 4558.8 (6.4x) | 19148.8 (22.9x) | 102629.4 (67.7x) |
| F_65521 | 302.8 (1.5x) | 862.1 (2.2x) | 3026.8 (10.0x) | 13626.3 (35.9x) |
| F_(65521^2) | 1235.3 (1.0x) | 3072.6 (3.9x) | 11679.9 (14.3x) | 58890.0 (45.3x) |
| F_2147483647 | 305.3 (1.5x) | 819.6 (2.3x) | 2959.4 (9.8x) | 14129.8 (35.0x) |
| C | 322.8 (1.7x) | 1684.8 (4.9x) | 9026.5 (15.0x) | 50744.4 (42.9x) |

### Comparison with FLINT

FLINT (Fast Library for Number Theory) is an independent exact-arithmetic C library. The comparison times fastCD's finite-field arithmetic against FLINT 3.6.0 on two workloads with identical operands and defining polynomials, 1,000 operand pairs per cell and 24 paired blocks, every pair verified against the same independent implementation; all 132,000 pairs passed. The scalar workload is one field multiplication (`fastcd_ff_field_mul` against `nmod_mul` for prime fields or `fq_nmod_mul` for extensions). The CD workload is one complete product at each dimension, where both sides evaluate the same quasilinear schedule and only the coefficient arithmetic differs: fastCD's packed kernels against FLINT's field operations. Ratios are paired geometric means of FLINT time over fastCD time, so a ratio above 1 favors fastCD. As in the finite-field comparison, each cell runs in one process, so its interval does not include variation between processes or runs. Every dimension and the intervals for the CD cells are in [flint.md](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/benchmark-results/t440-20261109/flint.md) in the archive.

#### Scalar field multiplication: ns per product

| Field | fastCD | FLINT | FLINT/fastCD [95% CI] |
| --- | ---: | ---: | --- |
| F_3 | 1.94 | 6.89 | 3.55 [3.50, 3.60] |
| F_5 | 1.90 | 6.89 | 3.62 [3.58, 3.66] |
| F_13 | 1.99 | 7.02 | 3.53 [3.49, 3.58] |
| F_(3^5) | 54.10 | 154.64 | 2.86 [2.83, 2.89] |
| F_(5^2) | 10.10 | 61.77 | 6.13 [5.98, 6.29] |
| F_(13^5) | 60.83 | 255.37 | 4.20 [4.13, 4.26] |
| F_65521 | 1.93 | 6.88 | 3.57 [3.52, 3.62] |
| F_(65521^2) | 10.11 | 70.08 | 6.96 [6.67, 7.27] |
| F_2147483647 | 1.92 | 6.92 | 3.60 [3.57, 3.64] |
| F_(3^12) | 265.51 | 616.81 | 2.32 [2.30, 2.35] |
| F_(13^12) | 335.90 | 652.98 | 1.94 [1.93, 1.96] |
| F_(2147483647^12) | 353.09 | 1308.24 | 3.70 [3.66, 3.75] |

#### Complete CD multiplication: fastCD ns / FLINT ns (FLINT/fastCD)

| Field | N = 16 | N = 64 | N = 256 | N = 1024 |
| --- | ---: | ---: | ---: | ---: |
| F_3 | 303 / 2362 (7.8x) | 840 / 17594 (21.1x) | 913 / 104938 (115x) | 2336 / 546771 (235x) |
| F_5 | 295 / 2478 (8.4x) | 821 / 18331 (22.5x) | 1247 / 109128 (87.7x) | 3903 / 581704 (149x) |
| F_13 | 298 / 2582 (8.7x) | 809 / 19285 (23.9x) | 1475 / 112901 (76.8x) | 5189 / 599455 (116x) |
| F_(3^5) | 1013 / 37660 (37.2x) | 1681 / 280934 (167x) | 5597 / 1619491 (290x) | 18846 / 8576140 (456x) |
| F_(5^2) | 575 / 17881 (31.2x) | 1921 / 137145 (71.6x) | 7013 / 796825 (114x) | 35224 / 4224652 (120x) |
| F_(13^5) | 3762 / 52571 (14.1x) | 4746 / 376379 (79.7x) | 20215 / 2197665 (110x) | 104739 / 11692104 (112x) |
| F_65521 | 310 / 2608 (8.4x) | 838 / 19354 (23.3x) | 2705 / 114722 (42.6x) | 13378 / 610645 (46.1x) |
| F_(65521^2) | 1152 / 17639 (15.4x) | 2825 / 130734 (46.5x) | 11436 / 763240 (67.1x) | 56266 / 4067320 (72.6x) |
| F_2147483647 | 303 / 2632 (8.7x) | 880 / 19412 (22.2x) | 2636 / 115993 (44.1x) | 13907 / 611017 (44.2x) |
| F_(3^12) | 4918 / 122958 (25.1x) | 4969 / 882224 (178x) | 18232 / 5160745 (283x) | 76514 / 27158287 (356x) |
| F_(13^12) | 60505 / 127745 (2.1x) | 44342 / 913148 (20.6x) | 101741 / 5351779 (52.7x) | 369688 / 27891116 (75.5x) |
| F_(2147483647^12) | 14845 / 237839 (16.1x) | 31680 / 1713785 (54.4x) | 112363 / 10125557 (90.5x) | 479970 / 52261339 (109x) |

## Numerical stability

Every floating-point sum inside the quasilinear product whose length grows with the dimension is accumulated in blocks of eight with a balanced tree between blocks, and above dimension 256 the Cariow–Cariowa correction sums are combined by a balanced tree. Accuracy was measured against an exact reference on 13,584 operand pairs from 21 real and 19 complex input families. The error of a computed product `fl(xy)` is reported as `eta = ||fl(xy) - xy||_2 / (u ||x||_2 ||y||_2)`, with `u = 2^-53` the binary64 unit roundoff. On pooled Gaussian and uniform inputs, the median `eta` of the quasilinear product is within a factor of two of direct multiplication at every tested dimension:

| N | 16 | 64 | 256 | 512 | 1024 | 2048 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| quasilinear over R | 1.46 | 2.05 | 2.41 | 2.58 | 2.73 | 2.91 |
| direct over R | 1.26 | 1.50 | 1.62 | 1.71 | 1.74 | 1.80 |
| quasilinear over C | 1.71 | 2.22 | 2.50 | 2.68 | | |
| direct over C | 1.11 | 1.28 | 1.44 | 1.52 | | |

The largest error observed anywhere in the corpus is 3.7 over R and 4.1 over C (direct: 2.9 and 2.2; sequential accumulation on the same corpus: 78 and 18). These figures describe the measured corpus, not a worst case. From dimension 64 up, balanced accumulation costs at most 6% more time on default real products, and 5–13% on complex products and on builds without the explicit SIMD kernels. The method, the input families and the archived summaries and charts are described in [Numerical stability](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/stability.md). The raw operands and observations are kept on the author's machine and are not included in Git or the source distributions; reruns on another NumPy version can draw different random inputs.

## Documentation

- [C quick start](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/quickstart-c.md) and [C API reference](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/c-api.md)
- [Python quick start](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/quickstart-python.md) and
  [Python API reference](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/python-api.md)
- [Conventions and results](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/mathematics.md)
- [Benchmark](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/benchmark.md)
- [Numerical stability](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/stability.md)

## License and Attribution

All code is released under the [MIT License](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/LICENSE) and is presented with no warranty or guarantees. You may use it in any way for any purpose and are not restricted by the original author of the code. To cite the software, use the metadata in [CITATION.cff](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/CITATION.cff). The changelog is in [CHANGELOG.md](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/CHANGELOG.md).

The quasilinear algorithm is due to Harrison Lemley (me!), [*Quasilinear
multiplication in the Cayley–Dickson tower*](https://arxiv.org/abs/2609.11588). The
Cariow–Cariowa algorithm follows Aleksandr Cariow and Galina Cariowa,
[*An unified approach for developing rationalized algorithms for hypercomplex
number multiplication*](https://doi.org/10.15199/48.2015.02.09), *Przegląd
Elektrotechniczny* 91(2), 36–39 (2015).
