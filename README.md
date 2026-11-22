![fastCD](https://git.tanuki-cd.com/algebraity/fastCD/raw/branch/main/assets/fastCD.png)

# fastCD - v1.5.1

fastCD is a lean C kernel for Cayley-Dickson multiplication over real and complex numbers and finite fields, with a NumPy-compatible Python extension, written for researchers and engineers in need of faster Cayley-Dickson multiplication. The key advantage of fastCD is that it implements a **new O(N log N) algorithm** which, to the author's knowledge, has never been presented before. See the [accompanying paper](https://arxiv.org/abs/2609.11588) for more information on the mathematical foundations and rigorous derivation of correctness and time complexity. fastCD may not be completely suitable for every application, but it is licensed under the [MIT License](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/LICENSE) so that it may be improved upon and adapted to the user's needs completely, with no questions asked.

Besides the quasilinear algorithm, fastCD includes direct Cayley-Dickson multiplication and the algorithm of [Cariow and Cariowa](https://doi.org/10.15199/48.2015.02.09). All three are available for real coefficients, while complex and finite-field coefficients use direct and quasilinear. Quasilinear has the best asymptotic complexity of the three, but it is not the fastest at every low dimension, so the default algorithm for each dimension was chosen from single-core benchmarks of single and batched products. Over the real numbers, direct is the default for single products at dimension 4, and direct or Cariow-Cariowa is the default at dimensions 8 and 16; see [Benchmark](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/benchmark.md).

In an exact-reference accuracy study, the median floating-point error of the quasilinear product on pooled Gaussian and uniform inputs is within a factor of two of direct multiplication at every tested dimension; see [Numerical stability](#numerical-stability).

The canonical source repository is
[git.tanuki-cd.com/algebraity/fastCD](https://git.tanuki-cd.com/algebraity/fastCD).

## Convention

The `n`th Cayley-Dickson algebra is denoted by `A_n` and it is defined recursively by

```text
A_0 := F,   A_n := A_(n-1) x A_(n-1),   (a,b)(c,d) := (ac - conjugate(d) b, d a + b conjugate(c))
```

where the coefficient field `F` is the real numbers (the default), the complex numbers, or an
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

Note that covariance and fused sums of products are currently experimental features as of v1.5.x and may not be fully optimized or ready for production use. The code implementing them may change over time, but the API is expected to remain the same for these functions.

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

An `Element` is one immutable Cayley-Dickson element. An `ElementArray` holds a batch of elements in a NumPy coefficient array of shape `(..., N)`
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

fastCD chooses its multiplication algorithm by dimension from single-core benchmarks of the shipped kernels, measured on an Intel Xeon Gold 6148 in a Dell PowerEdge T440 from commit `93c9bd8`, with every timed product verified against an independent implementation. The [benchmark documentation](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/benchmark.md) describes the method, gives the full results, and explains how to rerun them.

### Real coefficients

Mean nanoseconds per product in batch mode, with one call for all products, and in single mode, with one call per product, where CC is Cariow-Cariowa and QL is quasilinear multiplication:

| N | Batch: direct | Batch: CC | Batch: QL | Single: direct | Single: CC | Single: QL |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1.32 | 1.52 | 1.02 | 15.76 | 16.76 | 15.30 |
| 4 | 3.84 | 4.65 | 3.68 | 17.75 | 20.22 | 18.08 |
| 8 | 12.18 | 14.05 | 17.59 | 28.22 | 38.16 | 36.25 |
| 16 | 54.79 | 49.92 | 52.41 | 76.28 | 110.28 | 108.25 |
| 32 | 256.56 | 227.38 | 184.50 | 298.82 | 432.14 | 221.13 |
| 64 | 1133.91 | 973.51 | 465.99 | 1163.54 | 1835.74 | 518.34 |
| 128 | 4530.78 | 5006.53 | 1242.81 | 4495.22 | 7910.79 | 1333.57 |
| 256 | 17665.04 | 17695.42 | 3046.46 | 17445.55 | 29214.67 | 3244.06 |
| 512 | 69956.80 | 68578.92 | 6920.47 | 69310.72 | 108032.81 | 7427.95 |
| 1024 | 275135.85 | 433524.98 | 16209.21 | 277013.17 | 419988.45 | 16258.80 |

Quasilinear multiplication is the fastest from dimension 32 on in both modes and about 17 times faster than direct multiplication at dimension 1024, and the automatic choice of algorithm is the fastest one at every dimension. For sums of many products, the experimental fused sum was 1.6 to 5.1 times faster than a batch of products followed by a sum with the AVX-512 kernel (see [Fused sums](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/benchmark.md#fused-sums)).

### Complex coefficients and finite fields

Quasilinear nanoseconds per single product, with the ratio of the direct time to the quasilinear time in parentheses:

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

Direct multiplication is faster at the smallest dimensions, and quasilinear multiplication from dimension 8 on, apart from two ties, with an advantage at dimension 1024 of 8.6 to 71 times, depending on the field. fastCD implements its own finite-field arithmetic instead of depending on FLINT (Fast Library for Number Theory), an independent exact-arithmetic C library, because in the same workloads its arithmetic was 2 to 7 times faster for a single field multiplication and 2 to 465 times faster within complete products from dimension 16 on (see [Comparison with FLINT](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/benchmark.md#comparison-with-flint)).

## Numerical stability

Every floating-point sum in the quasilinear product whose length grows with the dimension is added in blocks of eight, with the block totals combined by a balanced tree, which keeps its rounding error close to that of direct multiplication. Measured against exact products on 13,584 operand pairs, with the error `eta = ||fl(xy) - xy||_2 / (u ||x||_2 ||y||_2)` and `u = 2^-53`, the median error of the quasilinear product on Gaussian and uniform inputs is within a factor of two of that of direct multiplication at every tested dimension:

| N | 16 | 64 | 256 | 512 | 1024 | 2048 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| quasilinear over R | 1.46 | 2.05 | 2.41 | 2.58 | 2.73 | 2.91 |
| direct over R | 1.26 | 1.50 | 1.62 | 1.71 | 1.74 | 1.80 |
| quasilinear over C | 1.71 | 2.22 | 2.50 | 2.68 | | |
| direct over C | 1.11 | 1.28 | 1.44 | 1.52 | | |

The largest error in the whole corpus is 3.7 over R and 4.1 over C, against 2.9 and 2.2 for direct multiplication; these figures describe the inputs tested, not a worst case, and [Numerical stability](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/stability.md) gives the method, the full results, and the cost of the balanced sums. The error of a fused sum of products was below that of the same products computed as a batch and added in sequence.

## Documentation

- [C quick start](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/quickstart-c.md) and [C API reference](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/c-api.md)
- [Python quick start](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/quickstart-python.md) and
  [Python API reference](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/python-api.md)
- [Conventions and results](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/mathematics.md)
- [Benchmarks](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/benchmark.md)
- [Numerical stability](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/stability.md)

## License and Attribution

All code is released under the [MIT License](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/LICENSE) and is presented with no warranty or guarantees. You may use it in any way for any purpose and are not restricted by the original author of the code. To cite the software, use the metadata in [CITATION.cff](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/CITATION.cff). The changelog is in [CHANGELOG.md](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/CHANGELOG.md).

The quasilinear algorithm is due to Harrison Lemley (me!), [*Quasilinear
multiplication in the Cayley-Dickson tower*](https://arxiv.org/abs/2609.11588). The
Cariow-Cariowa algorithm follows Aleksandr Cariow and Galina Cariowa,
[*An unified approach for developing rationalized algorithms for hypercomplex
number multiplication*](https://doi.org/10.15199/48.2015.02.09), *Przegląd
Elektrotechniczny* 91(2), 36-39 (2015).
