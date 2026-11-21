![fastCD](https://git.tanuki-cd.com/algebraity/fastCD/raw/branch/main/assets/fastCD.png)

# fastCD - v1.5.0

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

A sum of many products does not need the products themselves. `fastcd_mul_sum_f64` returns `x_0 y_0 + ... + x_(count-1) y_(count-1)` as a *fused sum*, which accumulates the multiplications of the algorithm over all pairs and applies its additions once, and `fastcd_covariance_f64` uses the same sum for the covariance of two samples of elements. Both are available for real coefficients.

```c
#include <stdlib.h>
#include <fastcd/fastcd.h>

int main(void) {
    enum { N = 16, COUNT = 1000 };
    static double xs[COUNT * N], ys[COUNT * N];    /* COUNT elements each */
    double sum[N], cov[N];
    size_t length;

    for (size_t i = 0; i < COUNT * N; i++) {
        xs[i] = (double)(i % 7);
        ys[i] = (double)(i % 5);
    }
    if (fastcd_covariance_workspace_size(N, &length) != FASTCD_SUCCESS) return 1;
    double* scratch = malloc(length * sizeof(*scratch));
    if (scratch == NULL) return 1;
    FastCDWorkspaceF64 workspace = {scratch, length};

    /* sum = xs[0] ys[0] + ... + xs[COUNT - 1] ys[COUNT - 1] */
    FastCDStatus status = fastcd_mul_sum_f64(sum, xs, N, ys, N, COUNT, N, NULL,
            &workspace);
    /* cov = sum_p (x_p - mean(x)) conjugate(y_p - mean(y)) / (COUNT - 1) */
    if (status == FASTCD_SUCCESS) {
        status = fastcd_covariance_f64(cov, xs, N, ys, N, COUNT, N, 1, &workspace);
    }
    free(scratch);
    return status == FASTCD_SUCCESS ? 0 : 1;
}
```

Note that covariance and fused sums of products are currently **experimental** features as of v1.5.0!!

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

For real coefficients, `multiply_sum` and `covariance` return the fused sum of products and the covariance of two samples as one `Element`, without storing the products:

```python
rng = np.random.default_rng(0)
s = O.asarray(rng.normal(size=(1000, 8)))
t = O.asarray(rng.normal(size=(1000, 8)))

total = O.multiply_sum(s, t)               # s[0] * t[0] + ... + s[999] * t[999]
cov = O.covariance(s, t)                   # sum((s - mean) * conjugate(t - mean)) / 999
```

At dimension 2, with each sample stored as `[real, imag]`, `covariance` is the ordinary complex covariance, `numpy.cov(z, w, ddof=ddof)[0, 1]`.

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

All timings below were measured on a Dell PowerEdge T440 with an Intel Xeon Gold 6148, using one logical CPU per run, GCC 16.2.1 and Linux 7.2.11. The host is shared, and no governor, turbo, or SMT settings were changed.

Only the C kernel is timed, with storage already allocated: Python overhead, allocation, and verification are outside the timed region. Every result that is timed was also checked against an independent implementation, as each subsection describes.

All results were measured on 2026-11-20 and 2026-11-21 from commit `93c9bd8`, built from a clean checkout, and are archived in [benchmark-results/t440-20261120/](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/benchmark-results/t440-20261120/). The archive holds the raw observations, environment, commands, and build records, which list the SHA-256 of every measured source file. The [benchmark documentation](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/benchmark.md) describes the protocols and how to rerun them.

### Real coefficients

Mean nanoseconds per product with two-sided 95% Student-t intervals over 24 timing blocks of 10,000 operand pairs, in batch mode (one call for all 10,000 products, two products per SIMD instruction where the kernels support it) and single mode (one validated call per product). All 600,000 verification checks passed with zero mismatches. `fastest mean` names the lowest mean, or a tie when the paired per-block ratio of the two lowest means does not separate them, and `default` is the algorithm that automatic selection uses. The automatic choices agree with every cell of this run.

#### Batch products (ns per product)

| N | direct | cariow_cariowa | quasilinear | fastest mean | default |
| ---: | --- | --- | --- | --- | --- |
| 2 | 1.32 [1.30, 1.34] | 1.52 [1.49, 1.55] | 1.02 [1.00, 1.04] | quasilinear | quasilinear |
| 4 | 3.84 [3.80, 3.88] | 4.65 [4.60, 4.69] | 3.68 [3.62, 3.73] | quasilinear | quasilinear |
| 8 | 12.18 [11.99, 12.37] | 14.05 [13.85, 14.25] | 17.59 [17.34, 17.85] | direct | direct |
| 16 | 54.79 [53.87, 55.71] | 49.92 [48.69, 51.14] | 52.41 [51.06, 53.76] | cariow_cariowa | cariow_cariowa |
| 32 | 256.56 [252.77, 260.35] | 227.38 [221.50, 233.25] | 184.50 [182.24, 186.76] | quasilinear | quasilinear |
| 64 | 1133.91 [1109.34, 1158.48] | 973.51 [951.64, 995.38] | 465.99 [460.22, 471.75] | quasilinear | quasilinear |
| 128 | 4530.78 [4453.12, 4608.44] | 5006.53 [4926.07, 5086.99] | 1242.81 [1221.67, 1263.95] | quasilinear | quasilinear |
| 256 | 17665.04 [17305.84, 18024.25] | 17695.42 [17442.11, 17948.73] | 3046.46 [2999.09, 3093.83] | quasilinear | quasilinear |
| 512 | 69956.80 [69154.63, 70758.97] | 68578.92 [67922.14, 69235.70] | 6920.47 [6825.76, 7015.18] | quasilinear | quasilinear |
| 1024 | 275135.85 [273397.86, 276873.84] | 433524.98 [419774.76, 447275.19] | 16209.21 [16034.88, 16383.55] | quasilinear | quasilinear |

#### Single products (ns per product)

| N | direct | cariow_cariowa | quasilinear | fastest mean | default |
| ---: | --- | --- | --- | --- | --- |
| 2 | 15.76 [15.52, 16.01] | 16.76 [16.56, 16.95] | 15.30 [15.12, 15.48] | quasilinear | quasilinear |
| 4 | 17.75 [17.48, 18.02] | 20.22 [19.97, 20.48] | 18.08 [17.91, 18.26] | direct | direct |
| 8 | 28.22 [27.86, 28.58] | 38.16 [37.71, 38.60] | 36.25 [35.84, 36.67] | direct | direct |
| 16 | 76.28 [74.68, 77.88] | 110.28 [108.18, 112.37] | 108.25 [106.07, 110.43] | direct | direct |
| 32 | 298.82 [295.63, 302.01] | 432.14 [423.85, 440.43] | 221.13 [217.79, 224.47] | quasilinear | quasilinear |
| 64 | 1163.54 [1146.73, 1180.35] | 1835.74 [1797.03, 1874.46] | 518.34 [511.99, 524.69] | quasilinear | quasilinear |
| 128 | 4495.22 [4430.30, 4560.14] | 7910.79 [7706.85, 8114.74] | 1333.57 [1308.25, 1358.88] | quasilinear | quasilinear |
| 256 | 17445.55 [17268.47, 17622.63] | 29214.67 [28321.53, 30107.80] | 3244.06 [3196.11, 3292.01] | quasilinear | quasilinear |
| 512 | 69310.72 [68887.00, 69734.44] | 108032.81 [107477.81, 108587.82] | 7427.95 [7327.56, 7528.34] | quasilinear | quasilinear |
| 1024 | 277013.17 [274746.54, 279279.79] | 419988.45 [417575.06, 422401.83] | 16258.80 [16093.16, 16424.44] | quasilinear | quasilinear |

### Fused sums of products

The sums benchmark compares three ways of computing the sum of the products of a sample of real-coefficient elements. *Products* is what a program without the fused sums would call: one batch of products with automatic selection, followed by a sum. *Fused* is `fastcd_mul_sum_f64` on the same records. *Matrix product* computes the matrix of real cross moments `sum_p x_p[i] y_p[j]` with one `dgemm` and adds its entries with the basis signs. fastCD contains no matrix product, so the benchmark loads Intel MKL 2026.2.1 for it at run time, with one thread.

The table gives mean nanoseconds per element for samples of 1,024 elements, from 8,192 elements per cell and 60 timing blocks, with the AVX-512 kernel of the fused sums. Ratios are paired geometric means with 95% Student-t intervals, and a ratio above 1 favors the fused sum. The first sample of every cell was verified against a long double evaluation of the cross moments, and on every sample the three computations had to agree within a norm-scaled bound; all checks passed.

| N | products | fused | matrix product | products/fused [95% CI] | matrix product/fused [95% CI] |
| ---: | ---: | ---: | ---: | --- | --- |
| 2 | 4.14 | 2.33 | 1.52 | 1.77 [1.75, 1.80] | 0.65 [0.64, 0.66] |
| 4 | 7.02 | 2.53 | 2.13 | 2.79 [2.72, 2.86] | 0.84 [0.82, 0.86] |
| 8 | 16.33 | 5.42 | 3.44 | 3.01 [2.96, 3.07] | 0.63 [0.62, 0.65] |
| 16 | 59.29 | 17.59 | 14.20 | 3.37 [3.32, 3.43] | 0.81 [0.79, 0.82] |
| 32 | 201.84 | 43.63 | 48.29 | 4.64 [4.56, 4.72] | 1.11 [1.08, 1.13] |
| 64 | 515.56 | 108.01 | 155.17 | 4.78 [4.69, 4.88] | 1.44 [1.39, 1.48] |
| 128 | 1300.28 | 265.89 | 734.73 | 4.90 [4.81, 4.99] | 2.75 [2.65, 2.85] |
| 256 | 3032.47 | 704.42 | 2147.15 | 4.33 [4.23, 4.42] | 3.05 [2.98, 3.13] |
| 512 | 7109.87 | 1774.34 | 7900.96 | 4.02 [3.95, 4.09] | 4.45 [4.34, 4.56] |
| 1024 | 16834.66 | 4991.62 | 28766.20 | 3.38 [3.31, 3.45] | 5.78 [5.63, 5.92] |

With this kernel the fused sum is faster than a batch of products followed by a sum at every dimension, and faster than the matrix product from dimension 32 upward; below that the matrix product is faster for samples of this size. Samples of 64 and of 4,096 elements, and elements stored as coordinate planes, which `fastcd_mul_sum_planes_f64` reads without rearranging them, are in [sums.md](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/benchmark-results/t440-20261120/sums.md) in the archive.

The speed depends on the kernel that the processor selects. With the AVX2 kernel the fused sum is 1.67 to 2.52 times as fast as the products at dimensions 8 to 64. Builds without either kernel use a portable one, whose speed depends on dimension and layout. It saves product storage (see [Numerical stability](#numerical-stability) for accuracy). Automatic selection uses the direct sum through dimension 8 and the quasilinear sum above: at dimension 16 the direct sum takes 4% to 16% less time with AVX-512, between 3% less and 10% more with AVX2, and 1.8 times as much with the portable kernel, so the quasilinear sum is the choice that serves every kernel.

### Finite fields and complex coefficients

The same protocol compares direct and quasilinear multiplication over C and over the nine finite fields in the table below, which are the field benchmark's default set: the prime fields F_3, F_5, F_13, F_65521 and F_2147483647 and the extension fields F_(3^5), F_(5^2), F_(13^5) and F_(65521^2). Each cell uses 1,000 operand pairs and 24 timing blocks. Every pair is verified before timing, against independent schoolbook polynomial arithmetic over the finite fields and against a long-double reference over C; all 100,000 pairs passed. Each cell below gives the quasilinear mean in nanoseconds per single product and, in parentheses, the paired direct/quasilinear ratio, so a ratio above 1 favors quasilinear. Direct is faster at dimension 2 in every cell except single products over F_(65521^2), a tie, and at dimension 4 over C, five of the nine fields and single products over F_(13^5). From dimension 8 upward quasilinear is faster in every cell except two ties: F_(3^5) at dimension 64 and batch products over F_(65521^2) at dimension 16. Automatic selection uses direct for `N <= 4` and quasilinear otherwise, over C and every finite field. From dimension 16 upward, batch and single means differ by at most 8%, the most over C at dimension 16, where batching saves 8%. Each cell runs in one process, so its interval covers variation between timing blocks but not between processes: in repeated runs on this host, individual cells from dimension 64 upward differed by up to 17%, and direct/quasilinear ratios by up to 8%. Both modes, every dimension and the confidence intervals are in [domains.md](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/benchmark-results/t440-20261120/domains.md) in the archive.

#### Single products over finite fields and C: quasilinear ns per product (direct/quasilinear)

| Field | N = 16 | N = 64 | N = 256 | N = 1024 |
| --- | ---: | ---: | ---: | ---: |
| F_3 | 303.1 (1.5x) | 729.5 (2.4x) | 950.0 (2.8x) | 2202.4 (17.5x) |
| F_5 | 302.1 (1.5x) | 837.0 (2.2x) | 1222.4 (6.6x) | 3955.0 (33.1x) |
| F_13 | 303.8 (1.5x) | 844.7 (2.2x) | 1479.5 (7.3x) | 5001.4 (35.0x) |
| F_(3^5) | 1053.9 (1.1x) | 1715.1 (1.0x) | 5265.2 (2.2x) | 18651.3 (8.6x) |
| F_(5^2) | 570.0 (1.3x) | 1885.8 (4.2x) | 7405.6 (15.6x) | 35498.8 (50.3x) |
| F_(13^5) | 3443.6 (1.2x) | 4611.2 (6.2x) | 19779.3 (21.8x) | 96222.8 (70.8x) |
| F_65521 | 299.1 (1.5x) | 804.6 (2.3x) | 2942.7 (10.1x) | 12742.1 (36.6x) |
| F_(65521^2) | 1231.4 (1.1x) | 2828.1 (3.9x) | 11629.4 (14.3x) | 56283.8 (45.9x) |
| F_2147483647 | 298.4 (1.5x) | 849.2 (2.2x) | 2698.7 (10.4x) | 12466.6 (37.7x) |
| C | 312.9 (1.7x) | 1704.3 (4.9x) | 8701.9 (15.3x) | 48752.4 (43.8x) |

### Comparison with FLINT

FLINT (Fast Library for Number Theory) is an independent exact-arithmetic C library. The comparison times fastCD's finite-field arithmetic against FLINT 3.6.0 on two workloads with identical operands and defining polynomials, 1,000 operand pairs per cell and 24 paired blocks, every pair verified against the same independent implementation; all 132,000 pairs passed. The scalar workload is one field multiplication (`fastcd_ff_field_mul` against `nmod_mul` for prime fields or `fq_nmod_mul` for extensions). The CD workload is one complete product at each dimension, where both sides evaluate the same quasilinear schedule and only the coefficient arithmetic differs: fastCD's packed kernels against FLINT's field operations. Ratios are paired geometric means of FLINT time over fastCD time, so a ratio above 1 favors fastCD. As in the finite-field comparison, each cell runs in one process, so its interval does not include variation between processes or runs. Every dimension and the intervals for the CD cells are in [flint.md](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/benchmark-results/t440-20261120/flint.md) in the archive.

#### Scalar field multiplication: ns per product

| Field | fastCD | FLINT | FLINT/fastCD [95% CI] |
| --- | ---: | ---: | --- |
| F_3 | 1.97 | 6.88 | 3.50 [3.44, 3.56] |
| F_5 | 1.95 | 6.88 | 3.54 [3.49, 3.58] |
| F_13 | 1.92 | 6.79 | 3.53 [3.49, 3.58] |
| F_(3^5) | 52.90 | 151.49 | 2.86 [2.83, 2.89] |
| F_(5^2) | 9.79 | 60.89 | 6.22 [6.12, 6.32] |
| F_(13^5) | 60.02 | 252.10 | 4.20 [4.15, 4.25] |
| F_65521 | 1.91 | 6.79 | 3.55 [3.50, 3.60] |
| F_(65521^2) | 9.94 | 69.75 | 7.02 [6.90, 7.14] |
| F_2147483647 | 1.99 | 6.95 | 3.49 [3.46, 3.52] |
| F_(3^12) | 256.75 | 605.46 | 2.36 [2.33, 2.38] |
| F_(13^12) | 333.59 | 655.09 | 1.96 [1.95, 1.98] |
| F_(2147483647^12) | 345.47 | 1274.63 | 3.69 [3.63, 3.75] |

#### Complete CD multiplication: fastCD ns / FLINT ns (FLINT/fastCD)

| Field | N = 16 | N = 64 | N = 256 | N = 1024 |
| --- | ---: | ---: | ---: | ---: |
| F_3 | 303 / 2304 (7.6x) | 786 / 17361 (22.2x) | 887 / 103072 (116x) | 2246 / 545555 (243x) |
| F_5 | 299 / 2471 (8.3x) | 796 / 18553 (23.4x) | 1182 / 108894 (92.2x) | 3849 / 576029 (150x) |
| F_13 | 296 / 2567 (8.7x) | 751 / 19130 (25.6x) | 1483 / 113451 (76.7x) | 5004 / 600521 (120x) |
| F_(3^5) | 1029 / 37610 (36.6x) | 1688 / 276146 (164x) | 5628 / 1614466 (288x) | 18632 / 8654929 (465x) |
| F_(5^2) | 562 / 17677 (31.5x) | 1841 / 133960 (72.9x) | 7191 / 797087 (111x) | 37879 / 4215793 (112x) |
| F_(13^5) | 3524 / 51723 (14.8x) | 4448 / 371973 (83.7x) | 20086 / 2195058 (110x) | 93398 / 11414018 (122x) |
| F_65521 | 303 / 2576 (8.5x) | 787 / 19267 (24.6x) | 2547 / 113299 (44.6x) | 12292 / 601869 (49.2x) |
| F_(65521^2) | 1266 / 18949 (15.0x) | 2831 / 129650 (46.1x) | 11412 / 760777 (67.1x) | 55952 / 4096556 (73.4x) |
| F_2147483647 | 300 / 2620 (8.7x) | 791 / 19541 (24.7x) | 2736 / 115812 (42.5x) | 12658 / 613825 (48.6x) |
| F_(3^12) | 4733 / 122398 (25.9x) | 4794 / 876796 (183x) | 17578 / 5104046 (291x) | 72578 / 26821623 (370x) |
| F_(13^12) | 60584 / 127766 (2.1x) | 43241 / 921304 (21.3x) | 96070 / 5211185 (54.3x) | 354900 / 27519067 (77.6x) |
| F_(2147483647^12) | 14639 / 237063 (16.3x) | 29288 / 1694677 (58.0x) | 100696 / 9885669 (98.5x) | 460726 / 52064812 (113x) |

## Numerical stability

Every floating-point sum inside the quasilinear product whose length grows with the dimension is accumulated in blocks of eight with a balanced tree between blocks, and above dimension 256 the Cariow–Cariowa correction sums are combined by a balanced tree. Accuracy was measured against an exact reference on 13,584 operand pairs from 21 real and 19 complex input families. The error of a computed product `fl(xy)` is reported as `eta = ||fl(xy) - xy||_2 / (u ||x||_2 ||y||_2)`, with `u = 2^-53` the binary64 unit roundoff. On pooled Gaussian and uniform inputs, the median `eta` of the quasilinear product is within a factor of two of direct multiplication at every tested dimension:

| N | 16 | 64 | 256 | 512 | 1024 | 2048 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| quasilinear over R | 1.46 | 2.05 | 2.41 | 2.58 | 2.73 | 2.91 |
| direct over R | 1.26 | 1.50 | 1.62 | 1.71 | 1.74 | 1.80 |
| quasilinear over C | 1.71 | 2.22 | 2.50 | 2.68 | | |
| direct over C | 1.11 | 1.28 | 1.44 | 1.52 | | |

The largest error observed anywhere in the corpus is 3.7 over R and 4.1 over C (direct: 2.9 and 2.2; sequential accumulation on the same corpus: 78 and 18). These figures describe the measured corpus, not a worst case. From dimension 64 up, balanced accumulation costs at most 4% more time on default real products, and 7–11% on complex products and on builds without the explicit SIMD kernels. The method, the input families and the archived summaries and charts are described in [Numerical stability](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/stability.md). The raw operands and observations are kept on the author's machine and are not included in Git or the source distributions; reruns on another NumPy version can draw different random inputs.

Fused sums were measured separately, on 689 sums and 84 covariances of real-coefficient samples, against a reference that uses no fastCD arithmetic. For a computed sum `S_hat` of the products of `P` pairs, the error is reported as `eta = ||S_hat - S||_2 / (u sum_p ||x_p||_2 ||y_p||_2)`. On Gaussian and uniform samples of 1,024 elements, the median `eta` of the fused sum is a third to a half of that of the same products added in sequence, and at most 1.5 times that of the products added pairwise:

| N | 16 | 64 | 256 | 1024 |
| --- | ---: | ---: | ---: | ---: |
| fused sum | 0.09 | 0.11 | 0.13 | 0.15 |
| products added in sequence | 0.28 | 0.29 | 0.31 | 0.30 |
| products added pairwise | 0.08 | 0.08 | 0.09 | 0.10 |
| cross moments by matrix product | 0.29 | 0.19 | 0.24 | 0.35 |

Over all nine input families, with samples of 64, 1,024 and 4,096 elements, the largest `eta` of the fused sum is 3.4, against 1.9 for products added pairwise, 25 for products added in sequence and 20 for the cross moments. For the covariance, `eta` is scaled by the centered samples and is at most 0.21, including on samples whose mean is `10^9` times their spread, where centering with a rounded mean beforehand gives several thousand: `fastcd_covariance_f64` holds each mean as the first record plus the mean of the differences from it. These figures also describe the measured corpus, not a worst case, and they were measured with the AVX-512 kernel. The method and the full tables are in [Numerical stability](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/stability.md#fused-sums), and the results in [accuracy.md](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/benchmark-results/t440-20261120/accuracy.md) in the archive.

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
