![fastCD](assets/fastCD.png)

# fastCD - v1.1.0

fastCD is a lean C kernel for Cayley-Dickson multiplication with a NumPy-compatible Python extension written for researchers and engineers in need of faster Cayley-Dickson multiplication. The key advantage of fastCD is that it implements a **new O(N log N) algorithm** which, to the author's knowledge, has never been presented before. See the [accompanying paper](https://arxiv.org/abs/2609.11588) for more information on the mathematical foundations and rigorous derivation of correctness and time complexity. fastCD may not be completely suitable for every application, but it is licensed under the [MIT License](LICENSE) so that it may be improved upon and adapted to the user's needs completely, with no questions asked.

In addition to this new quasilinear algorithm, both direct Cayley-Dickson multiplication and an algorithm due to [Cariow and Cariowa](https://doi.org/10.15199/48.2015.02.09) are also included in the library. The default multiplication algorithm for each dimension has been selected based on explicit single-core benchmarking for both single and batched multiplication. Quasilinear has the best asymptotic complexity of these three algorithms, but it is not the fastest at every lower dimension. The direct and Cariow-Cariowa algorithms are also used for some low dimensions; see [Benchmark](#benchmark) for more information on this.

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

The full comparison for the v2 paper was rerun on 2026-10-23 on the Dell PowerEdge T440, using one logical CPU of an Intel Xeon Gold 6148 and source revision `f9a61b2`. The current quasilinear implementation includes the fused local bilinear map and complex octonion base case. [Raw results](benchmark-results/paper-v2-20261023/results.json), the [complete generated report](benchmark-results/paper-v2-20261023/results.md), and [environment and build details](benchmark-results/paper-v2-20261023/environment.json) are archived together.

The tables give mean nanoseconds per product with two-sided 95% Student-t intervals over 24 timing blocks. They reproduce the original protocol: dimensions 2 through 1024, all three algorithms in single and batch modes, 10,000 operand pairs per dimension, two warmup passes, a 20 ms timing target, and CPU 0. Every input corpus matches the original SHA-256 hash. Python object overhead, allocation, and verification are outside the timed region. See the [benchmark documentation](docs/benchmark.md) for the timing boundaries and implementation details.

The `fastest mean` column names the lowest verified mean, or a tie when the paired per-block ratio of the two lowest means does not separate them. The `default` column shows the library's current automatic selection for that mode and dimension. After this run, dimension-4 singles switched to quasilinear on the strength of its lower measured mean (18.55 ns versus 19.12 ns for direct). This is a tuning choice: the paired quasilinear/direct 95% interval, 0.952–1.004, still includes equal performance. The archived reports retain the defaults used when measured.

The [original paper data](benchmark-results/paper-results.json) and [report](benchmark-results/paper-results.md), measured at `14bf305`, remain archived, as does the [October 19 comparison through dimension 256](benchmark-results/t440-20261019/README.md). The new run uses GCC 16.2.1 and Linux 7.2.11; the original used GCC 16.1.1 and Linux 7.1.6. Historical timing differences therefore include compiler and host effects as well as the code changes. The T440 is a shared host, with no governor, turbo, SMT, or service changes made for this run.

All 600,000 algorithm/mode verification checks passed with zero mismatches and finite outputs; the [validation record](benchmark-results/paper-v2-20261023/validation.json) also records the build, test, arithmetic-count, and data-integrity checks.

### Batch products (ns per product)

| N | direct | cariow_cariowa | quasilinear | fastest mean | default |
| ---: | --- | --- | --- | --- | --- |
| 2 | 1.32 [1.31, 1.34] | 1.53 [1.49, 1.57] | 1.18 [1.16, 1.20] | quasilinear | quasilinear |
| 4 | 3.72 [3.57, 3.87] | 4.41 [4.35, 4.48] | 3.64 [3.49, 3.79] | tie (quasilinear, direct) | quasilinear |
| 8 | 12.44 [12.26, 12.62] | 14.35 [14.12, 14.57] | 17.79 [17.56, 18.01] | direct | direct |
| 16 | 55.70 [54.75, 56.66] | 50.26 [49.49, 51.04] | 54.72 [53.89, 55.55] | cariow_cariowa | cariow_cariowa |
| 32 | 260.03 [255.89, 264.16] | 228.57 [225.45, 231.69] | 182.12 [179.61, 184.64] | quasilinear | quasilinear |
| 64 | 1157.56 [1135.01, 1180.10] | 1030.75 [1017.72, 1043.79] | 460.16 [454.79, 465.53] | quasilinear | quasilinear |
| 128 | 4617.52 [4536.22, 4698.83] | 4915.14 [4791.39, 5038.89] | 1234.76 [1209.71, 1259.82] | quasilinear | quasilinear |
| 256 | 18115.11 [17890.99, 18339.23] | 17592.57 [17353.49, 17831.64] | 2993.77 [2949.73, 3037.82] | quasilinear | quasilinear |
| 512 | 69768.05 [68774.94, 70761.16] | 64511.31 [63946.62, 65076.00] | 6791.81 [6688.70, 6894.93] | quasilinear | quasilinear |
| 1024 | 272717.79 [270536.99, 274898.58] | 413297.56 [402950.61, 423644.51] | 15688.11 [15549.41, 15826.82] | quasilinear | quasilinear |

### Single products (ns per product)

| N | direct | cariow_cariowa | quasilinear | fastest mean | default |
| ---: | --- | --- | --- | --- | --- |
| 2 | 16.42 [16.21, 16.63] | 17.00 [16.78, 17.23] | 15.37 [15.20, 15.53] | quasilinear | quasilinear |
| 4 | 19.12 [17.56, 20.68] | 21.80 [19.50, 24.10] | 18.55 [17.77, 19.32] | tie (quasilinear, direct) | quasilinear |
| 8 | 28.07 [27.66, 28.47] | 41.11 [40.56, 41.65] | 37.20 [36.69, 37.71] | direct | direct |
| 16 | 82.44 [80.56, 84.32] | 114.64 [113.13, 116.15] | 111.54 [109.99, 113.08] | direct | direct |
| 32 | 300.32 [296.59, 304.05] | 456.03 [450.17, 461.88] | 217.40 [212.48, 222.32] | quasilinear | quasilinear |
| 64 | 1148.56 [1133.29, 1163.83] | 1944.67 [1921.60, 1967.75] | 511.60 [504.11, 519.09] | quasilinear | quasilinear |
| 128 | 4516.08 [4453.89, 4578.27] | 7732.47 [7608.13, 7856.80] | 1227.89 [1206.02, 1249.75] | quasilinear | quasilinear |
| 256 | 17367.15 [17238.01, 17496.29] | 28262.75 [27874.95, 28650.56] | 3176.66 [3131.20, 3222.12] | quasilinear | quasilinear |
| 512 | 68849.27 [68178.55, 69519.99] | 106804.97 [105999.77, 107610.16] | 7307.34 [7191.12, 7423.57] | quasilinear | quasilinear |
| 1024 | 274053.28 [271828.85, 276277.71] | 408040.98 [406100.25, 409981.72] | 15924.07 [15659.90, 16188.25] | quasilinear | quasilinear |

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
