![fastCD](assets/fastCD.png)

# fastCD

fastCD is a lean C kernel for Cayley-Dickson multiplication with a NumPy-compatible Python extension written for researchers and engineers in need of faster Cayley-Dickson multiplication. The key advantage of fastCD is that it implements a **new O(N log N) algorithm** which, to the author's knowledge, has never been presented before. See the accompanying paper referenced at the end of the README for more information on the mathematical foundations and rigorous derivation of correctness and time complexity. fastCD may not be completely suitable for every application, but it is licensed under the [MIT License](LICENSE) so that it may be improved upon and adapted to the user's needs completely, with no questions asked.

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

The results of the canonical benchmark from which algorithm defaults were chosen for the 1.0.0 release candidate will be located at `benchmark-results/results.json`, with the generated report beside it in `benchmark-results/results.md`. More information on the results is also available in the [Benchmark](docs/benchmark.md) documentation.

Here are the exact results of the canonical benchmark. The `default` column shows the algorithm the current library selects for that mode and dimension; the single-product default at `N = 4` was moved from quasilinear to direct on the strength of this run, so it differs from the generated report, which recorded the selection in effect at the time.

### Batch products (ns per product)

| N | direct | cariow_cariowa | quasilinear | fastest mean | default |
| ---: | --- | --- | --- | --- | --- |
| 2 | 1.33 [1.31, 1.34] | 1.53 [1.50, 1.57] | 1.05 [1.04, 1.06] | quasilinear | quasilinear |
| 4 | 3.70 [3.64, 3.76] | 4.42 [4.37, 4.47] | 3.67 [3.60, 3.75] | quasilinear | quasilinear |
| 8 | 11.89 [11.67, 12.12] | 13.52 [13.30, 13.74] | 16.53 [16.21, 16.84] | direct | direct |
| 16 | 54.49 [53.76, 55.22] | 49.72 [48.90, 50.55] | 57.54 [56.30, 58.77] | cariow_cariowa | cariow_cariowa |
| 32 | 257.56 [254.42, 260.69] | 225.64 [220.86, 230.43] | 194.15 [191.72, 196.58] | quasilinear | quasilinear |
| 64 | 1142.56 [1118.52, 1166.60] | 965.57 [953.03, 978.11] | 492.83 [483.75, 501.90] | quasilinear | quasilinear |
| 128 | 4315.41 [4240.20, 4390.62] | unavailable | 1317.35 [1290.37, 1344.34] | quasilinear | quasilinear |
| 256 | 17909.41 [17622.23, 18196.60] | unavailable | 3302.53 [3244.81, 3360.25] | quasilinear | quasilinear |
| 512 | 68998.46 [68171.09, 69825.82] | unavailable | 7354.67 [7247.14, 7462.21] | quasilinear | quasilinear |
| 1024 | 274110.65 [271699.14, 276522.17] | unavailable | 20632.11 [20386.63, 20877.58] | quasilinear | quasilinear |

### Single products (ns per product)

| N | direct | cariow_cariowa | quasilinear | fastest mean | default |
| ---: | --- | --- | --- | --- | --- |
| 2 | 15.11 [14.96, 15.27] | 15.12 [14.88, 15.36] | 14.99 [14.82, 15.17] | quasilinear | quasilinear |
| 4 | 17.15 [16.94, 17.35] | 18.95 [18.65, 19.24] | 17.81 [17.49, 18.13] | direct | direct |
| 8 | 27.82 [27.47, 28.17] | 38.92 [38.23, 39.61] | 37.35 [36.63, 38.08] | direct | direct |
| 16 | 72.38 [69.40, 75.37] | 106.51 [104.40, 108.62] | 119.45 [117.85, 121.05] | direct | direct |
| 32 | 301.81 [293.82, 309.79] | 435.81 [427.12, 444.50] | 305.90 [302.68, 309.12] | direct | quasilinear |
| 64 | 1109.27 [1094.34, 1124.21] | 1880.48 [1843.79, 1917.16] | 720.21 [711.52, 728.90] | quasilinear | quasilinear |
| 128 | 4485.89 [4401.46, 4570.33] | unavailable | 1703.93 [1668.53, 1739.33] | quasilinear | quasilinear |
| 256 | 17637.42 [17394.69, 17880.15] | unavailable | 4257.83 [4188.85, 4326.81] | quasilinear | quasilinear |
| 512 | 68183.41 [67628.16, 68738.67] | unavailable | 9417.32 [9302.18, 9532.45] | quasilinear | quasilinear |
| 1024 | 275157.63 [272944.99, 277370.26] | unavailable | 20549.62 [20347.77, 20751.47] | quasilinear | quasilinear |


## Documentation

- [C quick start](docs/quickstart-c.md) and [C API reference](docs/c-api.md)
- [Python quick start](docs/quickstart-python.md) and
  [Python API reference](docs/python-api.md)
- [Conventions and results](docs/mathematics.md)
- [Benchmark](docs/benchmark.md)

## License and Attribution

All code is released under the [MIT License](LICENSE) and is presented with no warranty or guarantees. You may use it in any way for any purpose and are not restricted by the original author of the code. To cite the software, use the metadata in [CITATION.cff](CITATION.cff); the changelog is in [CHANGELOG.md](CHANGELOG.md).

The quasilinear algorithm is due to Harrison Lemley (me!), *Quasilinear
multiplication in the real Cayley–Dickson tower* (forthcoming). The
Cariow–Cariowa algorithm follows Aleksandr Cariow and Galina Cariowa,
[*An unified approach for developing rationalized algorithms for hypercomplex
number multiplication*](https://doi.org/10.15199/48.2015.02.09), *Przegląd
Elektrotechniczny* 91(2), 36–39 (2015).
