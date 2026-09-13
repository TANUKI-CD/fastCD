![fastCD](assets/fastCD.png)

# fastCD - v1.0.0

fastCD is a lean C kernel for Cayley-Dickson multiplication with a NumPy-compatible Python extension written for researchers and engineers in need of faster Cayley-Dickson multiplication. The key advantage of fastCD is that it implements a **new O(N log N) algorithm** which, to the author's knowledge, has never been presented before. See the [accompanying paper](https://arxiv.org/abs/2609.11588) for more information on the mathematical foundations and rigorous derivation of correctness and time complexity. fastCD may not be completely suitable for every application, but it is licensed under the [MIT License](LICENSE) so that it may be improved upon and adapted to the user's needs completely, with no questions asked.

In addition to this new quasilinear algorithm, both direct Cayley-Dickson multiplication and an algorithm due to [Cariow and Cariowa](https://doi.org/10.15199/48.2015.02.09) are also included in the library. The default multiplication algorithm for each dimension has been selected based on explicit single-core benchmarking for both single and batched multiplication, and while quasilinear is provably asymptotically the fastest, it is not the fastest for all lower dimensions. The direct and Cariow-Cariowa algorithms are also used for some low dimensions; see [Benchmark](#benchmark) for more information on this.

The canonical source repository is
[git.tanuki-cd.com/algebraity/fastCD](https://git.tanuki-cd.com/algebraity/fastCD).

## Convention

```text
A_0 = R,   A_n = A_(n-1) x A_(n-1),   (a,b)(c,d) = (ac - conjugate(d) b, d a + b conjugate(c))
```

An element is stored as a contiguous array of `N = 2^n` `double` coefficients in the
recursive standard basis, where coordinate zero is the scalar and the two halves of
the array are the pair `(a, b)`. [Conventions and results](docs/mathematics.md)
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

For more information on how to get started using fastCD in C and to familiarize yourself with the API, take a look at [C quick start](docs/quickstart-c.md) and [C API reference](docs/c-api.md).

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
in compiled code. See the [Python quick start](docs/quickstart-python.md) and
the [Python API reference](docs/python-api.md) for more information on how each class works.

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

In order to demonstrate the efficiency of the new quasilinear algorithm in comparison with the direct and Cariow-Cariowa algorithms, and to justify the choice of default algorithm for batched and single multiplications, a benchmark has been designed. The benchmark generates 10,000 pairs of Cayley-Dickson elements and performs multiplications with them in blocks, timing each algorithm while also verifying the accuracy of each one via an independent direct implementation.

The results of the canonical benchmark from which algorithm defaults were chosen for the 1.0.0 release candidate are located at `benchmark-results/paper-results.json`, with the generated report beside it in `benchmark-results/paper-results.md`. More information on the results is also available in the [Benchmark](docs/benchmark.md) documentation.

Here are the exact results of the canonical benchmark, run on one core of an Intel Xeon Gold 6148 at commit `14bf305` with a clean tree. The `fastest mean` column names the lowest verified mean, or a tie when the paired per-block ratio of the two lowest means does not separate them, and the `default` column shows the algorithm the library selects for that mode and dimension. Cariow-Cariowa is timed at every dimension; the [Benchmark](docs/benchmark.md) documentation explains why its times above `N = 64` reflect its irregular sign corrections rather than its operation count.

### Batch products (ns per product)

| N | direct | cariow_cariowa | quasilinear | fastest mean | default |
| ---: | --- | --- | --- | --- | --- |
| 2 | 1.34 [1.33, 1.35] | 1.53 [1.50, 1.56] | 1.02 [1.01, 1.03] | quasilinear | quasilinear |
| 4 | 3.87 [3.78, 3.97] | 4.58 [4.49, 4.66] | 3.80 [3.71, 3.89] | tie (quasilinear, direct) | quasilinear |
| 8 | 11.89 [11.68, 12.09] | 13.53 [13.25, 13.82] | 16.22 [15.96, 16.49] | direct | direct |
| 16 | 54.68 [53.91, 55.45] | 49.38 [48.60, 50.16] | 56.95 [56.04, 57.85] | cariow_cariowa | cariow_cariowa |
| 32 | 257.63 [252.67, 262.58] | 221.01 [217.27, 224.76] | 193.27 [191.00, 195.54] | quasilinear | quasilinear |
| 64 | 1124.05 [1101.38, 1146.72] | 947.05 [935.01, 959.09] | 492.65 [483.39, 501.92] | quasilinear | quasilinear |
| 128 | 4468.17 [4362.50, 4573.85] | 4908.83 [4771.93, 5045.73] | 1322.56 [1304.49, 1340.63] | quasilinear | quasilinear |
| 256 | 17757.92 [17535.51, 17980.33] | 17663.82 [17465.57, 17862.08] | 3149.78 [3101.79, 3197.78] | quasilinear | quasilinear |
| 512 | 69138.03 [68183.42, 70092.64] | 65786.36 [64987.60, 66585.12] | 7485.61 [7376.06, 7595.15] | quasilinear | quasilinear |
| 1024 | 273693.44 [270572.17, 276814.71] | 408225.87 [406200.31, 410251.42] | 17483.64 [17250.55, 17716.73] | quasilinear | quasilinear |

### Single products (ns per product)

| N | direct | cariow_cariowa | quasilinear | fastest mean | default |
| ---: | --- | --- | --- | --- | --- |
| 2 | 16.90 [15.83, 17.96] | 17.92 [16.97, 18.87] | 15.69 [15.45, 15.92] | quasilinear | quasilinear |
| 4 | 19.33 [18.06, 20.60] | 22.15 [20.77, 23.52] | 19.89 [18.76, 21.02] | tie (direct, quasilinear) | direct |
| 8 | 27.78 [27.35, 28.21] | 39.42 [38.68, 40.16] | 36.45 [35.45, 37.46] | direct | direct |
| 16 | 78.06 [76.52, 79.60] | 103.90 [102.22, 105.58] | 119.59 [117.61, 121.57] | direct | direct |
| 32 | 286.89 [282.76, 291.03] | 439.21 [430.02, 448.41] | 225.86 [221.96, 229.76] | quasilinear | quasilinear |
| 64 | 1121.00 [1105.23, 1136.76] | 1851.90 [1819.35, 1884.44] | 542.68 [534.57, 550.79] | quasilinear | quasilinear |
| 128 | 4390.47 [4334.93, 4446.01] | 7833.73 [7680.01, 7987.44] | 1334.23 [1316.20, 1352.27] | quasilinear | quasilinear |
| 256 | 17314.96 [17119.93, 17509.99] | 28830.79 [28140.17, 29521.40] | 3381.47 [3323.64, 3439.30] | quasilinear | quasilinear |
| 512 | 69029.59 [68325.12, 69734.07] | 107873.78 [106494.36, 109253.20] | 7913.86 [7851.24, 7976.47] | quasilinear | quasilinear |
| 1024 | 274511.31 [272437.03, 276585.59] | 411259.09 [406462.40, 416055.78] | 17525.01 [17348.44, 17701.58] | quasilinear | quasilinear |


## Documentation

- [C quick start](docs/quickstart-c.md) and [C API reference](docs/c-api.md)
- [Python quick start](docs/quickstart-python.md) and
  [Python API reference](docs/python-api.md)
- [Conventions and results](docs/mathematics.md)
- [Benchmark](docs/benchmark.md)

## License and Attribution

All code is released under the [MIT License](LICENSE) and is presented with no warranty or guarantees. You may use it in any way for any purpose and are not restricted by the original author of the code. To cite the software, use the metadata in [CITATION.cff](CITATION.cff); the changelog is in [CHANGELOG.md](CHANGELOG.md).

The quasilinear algorithm is due to Harrison Lemley (me!), [*Quasilinear
multiplication in the real Cayley–Dickson tower*](https://arxiv.org/abs/2609.11588) . The
Cariow–Cariowa algorithm follows Aleksandr Cariow and Galina Cariowa,
[*An unified approach for developing rationalized algorithms for hypercomplex
number multiplication*](https://doi.org/10.15199/48.2015.02.09), *Przegląd
Elektrotechniczny* 91(2), 36–39 (2015).
