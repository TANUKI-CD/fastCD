![fastCD](https://git.tanuki-cd.com/algebraity/fastCD/raw/branch/main/assets/fastCD.png)

# fastCD - v1.1.0

fastCD is a lean C kernel for Cayley-Dickson multiplication with a NumPy-compatible Python extension written for researchers and engineers in need of faster Cayley-Dickson multiplication. The key advantage of fastCD is that it implements a **new O(N log N) algorithm** which, to the author's knowledge, has never been presented before. See the [accompanying paper](https://arxiv.org/abs/2609.11588) for more information on the mathematical foundations and rigorous derivation of correctness and time complexity. fastCD may not be completely suitable for every application, but it is licensed under the [MIT License](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/LICENSE) so that it may be improved upon and adapted to the user's needs completely, with no questions asked.

In addition to this new quasilinear algorithm, both direct Cayley-Dickson multiplication and an algorithm due to [Cariow and Cariowa](https://doi.org/10.15199/48.2015.02.09) are also included in the library. The default multiplication algorithm for each dimension has been selected based on explicit single-core benchmarking for both single and batched multiplication. Quasilinear has the best asymptotic complexity of these three algorithms, but it is not the fastest at every lower dimension. The direct and Cariow-Cariowa algorithms are also used for some low dimensions; see [Benchmark](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/benchmark.md) for more information on this.

The canonical source repository is
[git.tanuki-cd.com/algebraity/fastCD](https://git.tanuki-cd.com/algebraity/fastCD).

## Convention

```text
A_0 = R,   A_n = A_(n-1) x A_(n-1),   (a,b)(c,d) = (ac - conjugate(d) b, d a + b conjugate(c))
```

An element is stored as a contiguous array of `N = 2^n` `double` coefficients in the
recursive standard basis, where coordinate zero is the scalar and the two halves of
the array are the pair `(a, b)`. [Conventions and results](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/mathematics.md)
should be read for full information on conventions and a brief explanation of the mathematical
foundations of fastCD.

## Using the C library

The C library can be used by importing the `fastcd.h` header file and using fastCD functions
within a normal C program. Here is an example of supported usage.

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

After doing so, you may use the `Algebra`, `Element`, and `ElementArray` classes directly in Python after importing `pyfastcd`. Here's an example of what you can do.

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
in compiled code. See the [Python quick start](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/quickstart-python.md) and
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
The installation provides `<fastcd/fastcd.h>`, `libfastcd`, the license, and
a CMake package: use `find_package(fastcd CONFIG REQUIRED)` and link
`fastcd::fastcd`.

## Benchmark

The full comparison for the v2 paper was rerun on 2026-10-23 on the Dell PowerEdge T440, using one logical CPU of an Intel Xeon Gold 6148 and the revised **243-operation complex octonion leaf**. The measured source is revision `7da5a5f` plus the archived [source patch](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/benchmark-results/paper-v2-octonion243-20261023/source.patch); exact source and build hashes are recorded. [Raw results](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/benchmark-results/paper-v2-octonion243-20261023/results.json), the [complete generated report](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/benchmark-results/paper-v2-octonion243-20261023/results.md), and [environment and build details](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/benchmark-results/paper-v2-octonion243-20261023/environment.json) are archived together.

The tables give mean nanoseconds per product with two-sided 95% Student-t intervals over 24 timing blocks. They reproduce the original protocol: dimensions 2 through 1024, all three algorithms in single and batch modes, 10,000 operand pairs per dimension, two warmup passes, a 20 ms timing target, and CPU 0. Every input corpus matches the original SHA-256 hash. Python object overhead, allocation, and verification are outside the timed region, so the benchmark focuses on the multiplication kernel alone. See the [benchmark documentation](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/benchmark.md) for the timing boundaries and implementation details.

The `fastest mean` column names the lowest verified mean, or a tie when the paired per-block ratio of the two lowest means does not separate them. The `default` column shows AUTO for the measured build. Dimension-4 singles remain a close tie: quasilinear averaged 18.12 ns versus direct's 18.18 ns, with a paired quasilinear/direct 95% interval of 0.979–1.015.

The [earlier full v2 comparison](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/benchmark-results/paper-v2-20261023/results.md), using the 244-operation leaf at `f9a61b2`, remains archived alongside the [original paper data](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/benchmark-results/paper-results.json) and [report](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/benchmark-results/paper-results.md), measured at `14bf305`, and the [October 19 comparison](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/benchmark-results/t440-20261019/README.md). [Comparisons with the previous leaf](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/benchmark-results/paper-v2-octonion243-20261023/comparison-previous-leaf.json) are descriptive ratios of means from separate runs. This run and the previous full v2 run use GCC 16.2.1 and Linux 7.2.11, while the original used GCC 16.1.1 and Linux 7.1.6. The T440 is a shared host, with no governor, turbo, or SMT changes made for this run, so timing differences cannot be attributed solely to the leaf change.

All 600,000 algorithm/mode verification checks passed with zero mismatches and finite outputs. The [validation record](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/benchmark-results/paper-v2-octonion243-20261023/validation.json) also records the build, test, arithmetic-count, and data-integrity checks.

### Batch products (ns per product)

| N | direct | cariow_cariowa | quasilinear | fastest mean | default |
| ---: | --- | --- | --- | --- | --- |
| 2 | 1.31 [1.29, 1.32] | 1.57 [1.54, 1.60] | 1.01 [1.00, 1.02] | quasilinear | quasilinear |
| 4 | 3.84 [3.78, 3.89] | 4.58 [4.50, 4.65] | 3.77 [3.72, 3.81] | tie (quasilinear, direct) | quasilinear |
| 8 | 12.41 [12.20, 12.61] | 14.19 [13.82, 14.56] | 17.86 [17.57, 18.15] | direct | direct |
| 16 | 55.34 [54.18, 56.49] | 50.46 [49.49, 51.43] | 55.15 [54.24, 56.06] | cariow_cariowa | cariow_cariowa |
| 32 | 257.28 [252.91, 261.66] | 230.56 [227.19, 233.93] | 174.49 [172.83, 176.15] | quasilinear | quasilinear |
| 64 | 1152.26 [1133.60, 1170.93] | 1012.92 [993.30, 1032.54] | 446.49 [440.03, 452.95] | quasilinear | quasilinear |
| 128 | 4748.29 [4688.47, 4808.10] | 5136.17 [4962.06, 5310.28] | 1249.72 [1219.47, 1279.97] | quasilinear | quasilinear |
| 256 | 18377.31 [18145.54, 18609.07] | 18353.65 [17953.07, 18754.24] | 3059.44 [3007.34, 3111.55] | quasilinear | quasilinear |
| 512 | 70080.95 [69127.78, 71034.12] | 66708.04 [65482.11, 67933.97] | 6791.69 [6697.58, 6885.80] | quasilinear | quasilinear |
| 1024 | 272762.00 [270235.59, 275288.41] | 406124.61 [403922.66, 408326.56] | 15426.99 [15279.98, 15573.99] | quasilinear | quasilinear |

### Single products (ns per product)

| N | direct | cariow_cariowa | quasilinear | fastest mean | default |
| ---: | --- | --- | --- | --- | --- |
| 2 | 15.93 [15.71, 16.16] | 16.73 [16.52, 16.95] | 15.14 [14.85, 15.42] | quasilinear | quasilinear |
| 4 | 18.18 [17.88, 18.47] | 20.02 [19.67, 20.37] | 18.12 [17.85, 18.39] | tie (quasilinear, direct) | quasilinear |
| 8 | 27.72 [27.19, 28.26] | 40.65 [40.10, 41.20] | 36.63 [36.01, 37.24] | direct | direct |
| 16 | 81.65 [79.95, 83.34] | 115.26 [113.79, 116.74] | 113.28 [111.07, 115.48] | direct | direct |
| 32 | 299.20 [294.51, 303.89] | 454.81 [449.85, 459.78] | 210.52 [207.60, 213.43] | quasilinear | quasilinear |
| 64 | 1162.63 [1141.36, 1183.91] | 1873.65 [1833.21, 1914.10] | 508.11 [501.06, 515.17] | quasilinear | quasilinear |
| 128 | 4592.40 [4521.08, 4663.72] | 7850.82 [7698.55, 8003.09] | 1240.04 [1224.28, 1255.79] | quasilinear | quasilinear |
| 256 | 17641.62 [17449.49, 17833.76] | 29390.25 [28810.93, 29969.57] | 3259.17 [3214.80, 3303.53] | quasilinear | quasilinear |
| 512 | 69431.66 [68744.19, 70119.14] | 108745.08 [107367.31, 110122.86] | 7281.07 [7135.00, 7427.14] | quasilinear | quasilinear |
| 1024 | 270678.46 [269014.65, 272342.27] | 406446.84 [404362.16, 408531.51] | 15713.75 [15492.33, 15935.16] | quasilinear | quasilinear |

## Documentation

- [C quick start](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/quickstart-c.md) and [C API reference](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/c-api.md)
- [Python quick start](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/quickstart-python.md) and
  [Python API reference](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/python-api.md)
- [Conventions and results](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/mathematics.md)
- [Benchmark](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/docs/benchmark.md)

## License and Attribution

All code is released under the [MIT License](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/LICENSE) and is presented with no warranty or guarantees. You may use it in any way for any purpose and are not restricted by the original author of the code. To cite the software, use the metadata in [CITATION.cff](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/CITATION.cff). The changelog is in [CHANGELOG.md](https://git.tanuki-cd.com/algebraity/fastCD/src/branch/main/CHANGELOG.md).

The quasilinear algorithm is due to Harrison Lemley (me!), [*Quasilinear
multiplication in the real Cayley–Dickson tower*](https://arxiv.org/abs/2609.11588) . The
Cariow–Cariowa algorithm follows Aleksandr Cariow and Galina Cariowa,
[*An unified approach for developing rationalized algorithms for hypercomplex
number multiplication*](https://doi.org/10.15199/48.2015.02.09), *Przegląd
Elektrotechniczny* 91(2), 36–39 (2015).
