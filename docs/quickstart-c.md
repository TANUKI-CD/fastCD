# C quick start

fastCD's C library multiplies Cayley-Dickson elements, and computes the other
operations of the algebras, over the real and complex numbers and over finite
fields of odd characteristic. It represents an element of the algebra of
dimension `N = 2^n` as an array of `N` coefficients whose coordinate 0 holds the
scalar part, and it never allocates memory itself: every input, output, and
temporary buffer belongs to the calling program.

This guide introduces the basics of how to use the library one idea at a time,
from building it and linking a program against it through a first product, the
handling of errors, batches, the choice of algorithm, and the other operations
to complex and finite-field coefficients and sums of products. Each section
builds on the sections that come before it, and each example is a complete
program, which compiles as described under [Building and
linking](#building-and-linking) and prints the output shown after it. Users of
fastCD are recommended to start here and read this guide in its entirety,
compiling and running the examples for themselves before using the library in
production. The [C API reference](c-api.md) gives the complete rules for every
function used here and should be read when more detail is needed than this quick
start guide can provide.

## Building and linking

The library needs a C11 compiler and CMake 3.18 or newer, and on Unix-like
systems it links the system math library. From the repository root, the
following commands configure a release build, compile it, run the test suite,
and install the library, here under `$HOME/.local`:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
cmake -E chdir build ctest --output-on-failure
cmake --install build --prefix "$HOME/.local"
```

The default configuration builds a static library together with the test suite
and the benchmark runner, while `-DBUILD_SHARED_LIBS=ON` builds a shared library
instead and `-DBUILD_TESTING=OFF -DFASTCD_BUILD_BENCHMARK=OFF` builds the
library alone. The installation provides the headers under `<fastcd/...>`, the
library `libfastcd`, and a CMake package, so a CMake project links fastCD with

```cmake
find_package(fastcd CONFIG REQUIRED)
target_link_libraries(example PRIVATE fastcd::fastcd)
```

adding `-DCMAKE_PREFIX_PATH="$HOME/.local"` at configure time when the library
is not installed in a standard location. Without CMake, each example of this
guide compiles with

```sh
cc -std=c11 example.c -o example \
    -I"$HOME/.local/include" -L"$HOME/.local/lib" -lfastcd -lm
```

where some systems install the library under `lib64` instead of `lib`. A program
that links the shared library without CMake must also define `FASTCD_SHARED`, as
[Linking](c-api.md#linking) explains.

## A first product

The first program multiplies two octonions, elements of the algebra of dimension
8:

```c
#include <stdio.h>
#include <stdlib.h>
#include <fastcd/fastcd.h>

int main(void) {
    const size_t N = 8;                         /* octonions */
    const double x[8] = {1.0, 2.0};             /* 1 + 2 e_1 */
    const double y[8] = {3.0, 0.0, 4.0};        /* 3 + 4 e_2 */
    double product[8];
    size_t length;

    /* Ask how many doubles of workspace a product of this dimension needs */
    if (fastcd_mul_workspace_size(N, &length) != FASTCD_SUCCESS) return 1;
    double* scratch = length == 0 ? NULL : malloc(length * sizeof(*scratch));
    if (length != 0 && scratch == NULL) return 1;
    FastCDWorkspaceF64 workspace = {scratch, length};

    FastCDStatus status = fastcd_mul_f64(product, x, y, N,
            length == 0 ? NULL : &workspace);
    free(scratch);
    if (status != FASTCD_SUCCESS) return 1;

    for (size_t i = 0; i < N; i++) printf("%g ", product[i]);
    putchar('\n');
    return 0;
}
```

The program prints `3 6 4 8 0 0 0 0`, the coefficients of `(1 + 2 e_1)(3 + 4
e_2) = 3 + 6 e_1 + 4 e_2 + 8 e_3`, and it shows the three conventions that every
function of the library follows.

**Records.** An element of dimension `N`, a power of two, is stored as a
*record* of `N` consecutive `double` coefficients in the standard basis
`e_0 = 1, e_1, ..., e_(N-1)`, with the scalar part first, and the output of a
product must be an array separate from its inputs, although the two inputs may
be the same array.

**Workspaces.** The library never allocates memory on the heap, so a function
that needs temporary storage takes a *workspace* from the caller, described by a
`FastCDWorkspaceF64` and sized by a query function such as
`fastcd_mul_workspace_size`. A program queries the length once for a dimension
and can reuse the same workspace for every product of that dimension, and when
the length is zero, as it is here, the workspace may be `NULL`.

**Statuses.** Every arithmetic function returns a `FastCDStatus`, which is
`FASTCD_SUCCESS` when the call succeeded, and a function writes its output only
on success.

## Handling errors

A status other than `FASTCD_SUCCESS` says why a call failed, and
`fastcd_status_string` turns it into a fixed English message for a log:

```c
#include <stdio.h>
#include <fastcd/fastcd.h>

int main(void) {
    const double x[8] = {0.0};                  /* the zero octonion */
    double out[8];

    /* 6 is not a power of two, so this call fails and writes nothing */
    FastCDStatus status = fastcd_conjugate_f64(out, x, 6);
    printf("conjugate: %s\n", fastcd_status_string(status));

    /* Zero has no reciprocal */
    status = fastcd_reciprocal_f64(out, x, 8);
    printf("reciprocal: %s\n", fastcd_status_string(status));
    return status == FASTCD_ZERO_NORM ? 0 : 1;
}
```

The program prints

```text
conjugate: Invalid Cayley--Dickson dimension
reciprocal: Element has zero Euclidean norm
```

The status values describe problems with the call itself, such as an invalid
dimension or basis index, a null pointer, too little workspace, overlapping
buffers, or an unsupported algorithm, and a program can branch on them as
ordinary enumeration values. Numerical problems, by contrast, are not errors: a
NaN, an infinity, or an overflow appears in the result, and the call still
returns `FASTCD_SUCCESS`.

## Batches

To multiply many pairs of elements in one call, `fastcd_mul_batch_f64` takes
arrays of records together with their *strides*, the distance in doubles between
the starts of successive records:

```c
#include <stdio.h>
#include <stdlib.h>
#include <fastcd/fastcd.h>

int main(void) {
    enum { N = 8, COUNT = 1000 };
    double* xs = malloc(COUNT * N * sizeof(double));   /* COUNT records */
    double* out = malloc(COUNT * N * sizeof(double));
    double y[N];
    size_t length;

    if (xs == NULL || out == NULL) return 1;
    if (fastcd_mul_workspace_size(N, &length) != FASTCD_SUCCESS) return 1;
    double* scratch = length == 0 ? NULL : malloc(length * sizeof(*scratch));
    if (length != 0 && scratch == NULL) return 1;
    FastCDWorkspaceF64 workspace = {scratch, length};

    for (size_t i = 0; i < COUNT * N; i++) xs[i] = (double)(i % N);
    fastcd_basis_f64(y, N, 1);                         /* y = e_1 */

    /* out[k] = xs[k] e_1 for every k; y has stride 0, so it is reused */
    FastCDStatus status = fastcd_mul_batch_f64(out, N, xs, N, y, 0, COUNT, N,
            length == 0 ? NULL : &workspace);
    if (status == FASTCD_SUCCESS) {
        for (size_t i = 0; i < N; i++) printf("%g ", out[i]);   /* record 0 */
        putchar('\n');
    }
    free(scratch);
    free(out);
    free(xs);
    return status != FASTCD_SUCCESS;
}
```

The program prints `-1 0 3 -2 5 -4 -7 6`, the product of the first record,
`e_1 + 2 e_2 + ... + 7 e_7`, with `e_1`. A stride of zero repeats one record
throughout the batch, as it does here for `y`, and the batch uses one workspace,
of the length that `fastcd_mul_workspace_size` reports, for all of its products.
The output must not overlap either input.

## Choosing the algorithm

fastCD multiplies real coefficients with three algorithms, direct,
Cariow-Cariowa, and quasilinear, and `fastcd_mul_f64` and `fastcd_mul_batch_f64`
use `FASTCD_ALGORITHM_AUTO`, which chooses one by dimension, separately for
single products and for batches. A program can ask which algorithm `AUTO`
selects and can select one itself:

```c
#include <stdio.h>
#include <stdlib.h>
#include <fastcd/fastcd.h>

int main(void) {
    const size_t N = 16;
    const double x[16] = {1.0, 2.0};
    const double y[16] = {3.0, 0.0, 4.0};
    double product[16];
    FastCDAlgorithm single, batch;

    fastcd_default_algorithm(N, &single);
    fastcd_default_batch_algorithm(N, &batch);
    printf("auto selects %s (single) and %s (batch)\n",
            fastcd_algorithm_string(single), fastcd_algorithm_string(batch));

    /* An explicitly chosen algorithm needs the workspace reported for it */
    const FastCDAlgorithm chosen = FASTCD_ALGORITHM_QUASILINEAR;
    size_t length;
    FastCDStatus status =
            fastcd_mul_algorithm_workspace_size(N, chosen, &length);
    if (status != FASTCD_SUCCESS) return 1;
    double* scratch = length == 0 ? NULL : malloc(length * sizeof(*scratch));
    if (length != 0 && scratch == NULL) return 1;
    FastCDWorkspaceF64 workspace = {scratch, length};

    status = fastcd_mul_with_algorithm_f64(product, x, y, N, chosen,
            length == 0 ? NULL : &workspace);
    free(scratch);
    if (status != FASTCD_SUCCESS) return 1;
    printf("%g %g %g %g\n", product[0], product[1], product[2], product[3]);
    return 0;
}
```

The program prints

```text
auto selects direct (single) and cariow_cariowa (batch)
3 6 4 8
```

on a build with the two-lane SIMD batch kernels, which GCC and Clang provide on
SSE2 and AArch64 targets; on other builds batches follow the choice for single
products. The choices come from benchmarks of the three algorithms, which [How
the results set the defaults](benchmark.md#how-the-results-set-the-defaults)
reports. An explicitly selected algorithm needs the workspace that
`fastcd_mul_algorithm_workspace_size` reports for it, and
`fastcd_algorithm_supports_dimension` tells whether an algorithm is available at
a dimension: direct and quasilinear multiplication support every dimension, and
Cariow-Cariowa supports dimensions through 1024.

## Other operations

The other operations of the algebra follow the same pattern as the product, and
an operation that multiplies takes a workspace sized by its own query. This
program allocates those workspaces with a small helper:

```c
#include <stdio.h>
#include <stdlib.h>
#include <fastcd/fastcd.h>

enum { N = 8 };

/* Allocate the workspace reported by one of the workspace queries. */
static int allocate(FastCDWorkspaceF64* w,
        FastCDStatus (*query)(size_t dimension, size_t* length)) {
    w->data = NULL;
    if (query(N, &w->length) != FASTCD_SUCCESS) return 0;
    if (w->length == 0) return 1;
    w->data = malloc(w->length * sizeof(double));
    return w->data != NULL;
}

int main(void) {
    const double x[N] = {1.0, 2.0, 0.0, 3.0};         /* 1 + 2 e_1 + 3 e_3 */
    const double y[N] = {0.0, 1.0, 1.0};              /* e_1 + e_2 */
    const double z[N] = {2.0, 0.0, 0.0, 0.0, 1.0};    /* 2 + e_4 */
    double c[N], u[N], r[N], s[N], w[N], value;
    FastCDWorkspaceF64 cross, assoc, lop;

    if (!allocate(&cross, fastcd_imag_cross_workspace_size)
            || !allocate(&assoc, fastcd_associator_workspace_size)
            || !allocate(&lop, fastcd_lop_commutator_workspace_size))
        return 1;

    fastcd_conjugate_f64(c, x, N);                /* c may equal x */
    fastcd_norm_f64(&value, x, N);
    fastcd_normalize_f64(u, x, N);                /* fails if x == 0 */
    fastcd_reciprocal_f64(r, x, N);               /* conjugate(x) / <x, x> */
    fastcd_square_f64(s, x, N);                   /* s may equal x */
    fastcd_imag_cross_f64(w, x, y, N, &cross);    /* (XY - YX)/2 */
    fastcd_commutator_f64(w, x, y, N, &cross);    /* xy - yx */
    fastcd_anticommutator_f64(w, x, y, N);        /* xy + yx, linear time */
    fastcd_lop_commutator_apply_f64(w, x, y, z, N, &lop); /* x(yz) - y(xz) */
    fastcd_associator_f64(w, x, y, z, N, &assoc); /* (xy)z - x(yz) */

    printf("|x| = %g\n", value);
    printf("(xy)z - x(yz) =");
    for (size_t i = 0; i < N; i++) printf(" %g", w[i]);
    putchar('\n');
    free(cross.data);
    free(assoc.data);
    free(lop.data);
    return 0;
}
```

The program prints

```text
|x| = 3.74166
(xy)z - x(yz) = 0 0 0 0 0 -6 6 4
```

and the nonzero associator shows that octonion multiplication is not
associative. The alternating product `(XY - YX)/2`, whose `X` and `Y` are the
imaginary parts of `x` and `y` (see [Derived
operations](mathematics.md#derived-operations)), shares
`fastcd_imag_cross_workspace_size` with the commutator, while the associator and
the two operator commutators each have their own query, which reserves room for
two intermediate elements; `fastcd_rop_commutator_workspace_size` sizes
`fastcd_rop_commutator_apply_f64`, which computes `(zy)x - (zx)y`.

Products of basis elements need no workspace at all, because the product of two
basis elements is another basis element up to sign, and multiplying an element
by a basis element only permutes its coordinates and changes some of their
signs:

```c
#include <stdio.h>
#include <fastcd/fastcd.h>

int main(void) {
    enum { N = 8 };
    const double x[N] = {1, 2, 3, 4, 5, 6, 7, 8};
    double left[N], right[N];
    FastCDBasisProduct p;

    fastcd_basis_product(&p, N, 1, 2);       /* e_1 e_2 = sign e_index */
    fastcd_basis_lmul_f64(left, 3, x, N);    /* e_3 x */
    fastcd_basis_rmul_f64(right, x, 3, N);   /* x e_3 */
    printf("e_1 e_2 = %+d e_%zu\n", p.sign, p.index);
    printf("e_3 x =");
    for (size_t i = 0; i < N; i++) printf(" %g", left[i]);
    putchar('\n');
    return 0;
}
```

The program prints

```text
e_1 e_2 = +1 e_3
e_3 x = -4 -3 2 1 -8 7 -6 5
```

## Complex coefficients

Complex arithmetic is declared in `<fastcd/complex.h>`. A complex coefficient is
a `FastCDComplexF64`, two doubles with the real part first, so an element of
dimension `N` is an array of `N` of them, while workspace lengths still count
doubles:

```c
#include <stdio.h>
#include <stdlib.h>
#include <fastcd/complex.h>

int main(void) {
    enum { N = 4 };                             /* quaternions over C */
    /* x = (1 + i) + 2i e_1 and y = e_2 */
    const FastCDComplexF64 x[N] = {{1.0, 1.0}, {0.0, 2.0}};
    const FastCDComplexF64 y[N] = {{0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}};
    FastCDComplexF64 product[N];
    size_t length;

    FastCDStatus status =
            fastcd_workspace_size_c128(N, FASTCD_OP_MULTIPLY, &length);
    if (status != FASTCD_SUCCESS) return 1;
    double* scratch = length == 0 ? NULL : malloc(length * sizeof(*scratch));
    if (length != 0 && scratch == NULL) return 1;
    FastCDWorkspaceF64 workspace = {scratch, length};

    status = fastcd_mul_c128(product, x, y, N,
            length == 0 ? NULL : &workspace);
    free(scratch);
    if (status != FASTCD_SUCCESS) return 1;
    for (size_t i = 0; i < N; i++)
        printf("(%g%+gi) ", product[i].real, product[i].imag);
    putchar('\n');
    return 0;
}
```

The program prints `(0+0i) (0+0i) (1+1i) (0+2i)`, that is, `((1 + i) + 2i e_1)
e_2 = (1 + i) e_2 + 2i e_3`. The complex number `i` is a coefficient, not one of
the basis elements `e_k`, and Cayley-Dickson conjugation, which negates the
coordinates other than the scalar one, leaves the complex coefficients
themselves unchanged.

## Finite fields

Finite-field arithmetic is declared in `<fastcd/finite.h>`, and it needs two
objects that the real and complex functions do not. A `FastCDField` describes
the field F_(p^d), including the irreducible polynomial whose residue classes
represent its elements, and a `FastCDPlanFF` fixes the Cayley-Dickson dimension
over that field, so that the size of a record and of the workspace is computed
once. Each coefficient occupies `d` consecutive `uint32_t` residues, constant
term first, so an element of dimension `N` is `N * d` words, coefficient `i`
starts at word `i * d`, and lengths, strides, and workspaces count words:

```c
#include <stdio.h>
#include <stdlib.h>
#include <fastcd/finite.h>

int main(void) {
    FastCDField field;
    FastCDPlanFF plan;
    size_t words;

    /* F_(13^5) with a deterministic irreducible polynomial */
    if (fastcd_field_init(&field, 13, 5, NULL) != FASTCD_FF_SUCCESS) return 1;
    /* level 2: dimension 4, so an element is 4 * 5 = 20 words */
    if (fastcd_ff_plan_init(&plan, &field, 2) != FASTCD_FF_SUCCESS) return 1;
    if (fastcd_ff_cd_workspace_size(&plan, &words) != FASTCD_FF_SUCCESS)
        return 1;

    uint32_t x[20] = {1};       /* the scalar 1 */
    uint32_t y[20] = {0};
    uint32_t out[20], q[5];
    y[5] = 1;                   /* coefficient 1 starts at word 5: y = e_1 */

    uint32_t* work = malloc(words * sizeof(*work));
    if (work == NULL) return 1;
    FastCDFFStatus status = fastcd_mul_ff(out, x, y, &plan, work, words);
    free(work);
    if (status != FASTCD_FF_SUCCESS) return 1;
    if (fastcd_quadratic_norm_ff(q, out, &plan) != FASTCD_FF_SUCCESS) return 1;
    printf("Q(xy) = %u\n", (unsigned)q[0]);     /* Q(e_1) = 1 */
    return 0;
}
```

The program prints `Q(xy) = 1`: the product of 1 and `e_1` is `e_1`, whose
quadratic form `Q(x) = sum x_k^2` is 1. Finite-field functions return their own
status type, `FastCDFFStatus`, and [Arithmetic across coefficient
fields](c-api.md#arithmetic-across-coefficient-fields) describes the complete
complex and finite-field interface.

## Sums of products and covariance

Sums of products and covariances are available for real coefficients only, and
they are experimental features of fastCD: they may not be fully optimized or
ready for production use, and the code implementing them may change over time,
but their API is expected to remain the same. `fastcd_mul_sum_f64` adds up the
products of `count` pairs of records without storing them, because it
accumulates the multiplications of the algorithm over all the pairs and applies
the rest of the algorithm once (see [Fused sums of
products](mathematics.md#fused-sums-of-products)), and `fastcd_covariance_f64`
uses the same sum for the covariance of two samples:

```c
#include <stdio.h>
#include <stdlib.h>
#include <fastcd/fastcd.h>

int main(void) {
    enum { N = 16, COUNT = 1000 };
    double* xs = malloc(COUNT * N * sizeof(double));
    double* ys = malloc(COUNT * N * sizeof(double));
    double sum[N], cov[N];
    size_t length;

    if (xs == NULL || ys == NULL) return 1;
    /* The covariance needs the longer workspace of the two */
    if (fastcd_covariance_workspace_size(N, &length) != FASTCD_SUCCESS)
        return 1;
    double* scratch = malloc(length * sizeof(*scratch));
    if (scratch == NULL) return 1;
    FastCDWorkspaceF64 workspace = {scratch, length};

    for (size_t i = 0; i < COUNT * N; i++) {
        xs[i] = (double)(i % 7);
        ys[i] = (double)(i % 5);
    }

    /* sum = xs[0] ys[0] + ... + xs[COUNT - 1] ys[COUNT - 1] */
    FastCDStatus status = fastcd_mul_sum_f64(sum, xs, N, ys, N, COUNT, N, NULL,
            &workspace);
    if (status != FASTCD_SUCCESS) return 1;

    /* cov = sum_p (x_p - mean(x)) conjugate(y_p - mean(y)) / (COUNT - 1) */
    status = fastcd_covariance_f64(cov, xs, N, ys, N, COUNT, N, 1, &workspace);
    if (status == FASTCD_SUCCESS) printf("%g %g\n", sum[0], cov[0]);
    free(scratch);
    free(ys);
    free(xs);
    return status != FASTCD_SUCCESS;
}
```

The program prints `-84016 0.01001`, the scalar coordinates of the sum and of
the covariance. The eighth argument of `fastcd_mul_sum_f64`, here `NULL`, is an
optional `FastCDSumMapF64`, which shifts and scales the coordinates of the
records as they are read, and when each coordinate of a sample is already stored
as one contiguous row, `fastcd_mul_sum_planes_f64` reads that layout directly
(see [Fused sums](c-api.md#fused-sums)). Whether a fused sum is faster than a
batch of products followed by a sum depends on the dimension, the number of
pairs, and the processor, and the [benchmark](benchmark.md#fused-sums) reports
measurements.

## Where to go next

- The [C API reference](c-api.md) documents every function, type, and rule.
- [Conventions and results](mathematics.md) describes the algebra, the
  multiplication algorithms, and their operation counts.
- The [benchmark](benchmark.md) and the [stability study](stability.md) report
  the speed and the accuracy of the algorithms.
