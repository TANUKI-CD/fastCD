# C quick start

To use fastCD in a C program, include `<fastcd/fastcd.h>` and link
`fastcd::fastcd` through CMake, or use `-lfastcd -lm` when linking by hand.
The arithmetic functions and queries with output parameters return a
`FastCDStatus`, and they write their results only when the call returns
`FASTCD_SUCCESS`. String and Boolean queries return their values directly.
The library does not allocate on the heap. An element of dimension `N` is
an array of `N` `double` coefficients, scalar first, and all lengths and
strides are measured in `double` elements.

Each example below is a complete program.

## One product

```c
#include <stdio.h>
#include <stdlib.h>
#include <fastcd/fastcd.h>

int main(void) {
    const size_t N = 8;                       /* octonions */
    const double x[8] = {1.0, 2.0};           /* 1 + 2 e_1 */
    const double y[8] = {3.0, 0.0, 4.0};      /* 3 + 4 e_2 */
    double product[8];
    size_t length;
    FastCDStatus status = fastcd_mul_workspace_size(N, &length);

    if (status != FASTCD_SUCCESS) {
        fprintf(stderr, "%s\n", fastcd_status_string(status));
        return 1;
    }
    double* scratch = length == 0 ? NULL : malloc(length * sizeof(*scratch));
    if (length != 0 && scratch == NULL) return 1;
    FastCDWorkspaceF64 workspace = {scratch, length};

    status = fastcd_mul_f64(product, x, y, N, length == 0 ? NULL : &workspace);
    free(scratch);
    if (status != FASTCD_SUCCESS) {
        fprintf(stderr, "%s\n", fastcd_status_string(status));
        return 1;
    }
    for (size_t i = 0; i < N; i++) printf("%g ", product[i]);
    putchar('\n');
    return 0;
}
```

The program prints `3 6 4 8 0 0 0 0`, that is,
`(1 + 2 e_1)(3 + 4 e_2) = 3 + 6 e_1 + 4 e_2 + 8 e_3`. Query the workspace
once for a dimension and reuse the allocation for every multiplication at
that dimension; the length from `fastcd_mul_workspace_size` also covers
`fastcd_mul_batch_f64`. The output must be a separate array, although the
two inputs may be the same array.

## Choosing the algorithm

`fastcd_mul_f64` uses `FASTCD_ALGORITHM_AUTO`, which picks an algorithm by
dimension from the single-core benchmark reported in the README: direct at
dimensions 8 and 16 and quasilinear elsewhere. Batches of two or more
products use Cariow–Cariowa at dimension 16 instead, because they evaluate
two products per SIMD instruction. You can check these choices or select
an algorithm yourself. An explicitly selected algorithm needs the workspace
reported for that algorithm:

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

    const FastCDAlgorithm chosen = FASTCD_ALGORITHM_CARIOW_CARIOWA;
    size_t length;
    if (fastcd_mul_algorithm_workspace_size(N, chosen, &length) != FASTCD_SUCCESS)
        return 1;  /* FASTCD_UNSUPPORTED_ALGORITHM above dimension 1024 */
    double* scratch = length == 0 ? NULL : malloc(length * sizeof(*scratch));
    if (length != 0 && scratch == NULL) return 1;
    FastCDWorkspaceF64 workspace = {scratch, length};

    FastCDStatus status = fastcd_mul_with_algorithm_f64(product, x, y, N, chosen,
            length == 0 ? NULL : &workspace);
    free(scratch);
    if (status != FASTCD_SUCCESS) return 1;
    printf("%g %g %g %g\n", product[0], product[1], product[2], product[3]);
    return 0;
}
```

The program prints `auto selects direct (single) and cariow_cariowa (batch)`
and then `3 6 4 8`. The Cariow–Cariowa algorithm is available through
dimension 1024, though quasilinear is faster above dimension 64 in the
recorded benchmarks; `fastcd_algorithm_supports_dimension` reports whether
an algorithm is available at a dimension. The direct and quasilinear
algorithms support every dimension.

## Batches

To multiply `count` pairs of elements in one call, use
`fastcd_mul_batch_f64`. Each stride is the distance, in doubles, between
the starts of successive elements. An input stride of zero repeats the same
element throughout the batch:

```c
#include <stdio.h>
#include <stdlib.h>
#include <fastcd/fastcd.h>

int main(void) {
    enum { N = 8, COUNT = 1000 };
    double* xs = malloc(COUNT * N * sizeof(double));   /* COUNT elements */
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
    printf("%s\n", fastcd_status_string(status));
    free(scratch);
    free(out);
    free(xs);
    return status != FASTCD_SUCCESS;
}
```

The batch reuses one workspace for all of its products, and the output must
not overlap either input.

## Operators and structure

The other operations follow the same pattern. Operations that multiply
need a workspace from their own query, which this program allocates with a
small helper:

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
    if (w->length != 0 && (w->data = malloc(w->length * sizeof(double))) == NULL)
        return 0;
    return 1;
}

int main(void) {
    const double x[N] = {1.0, 2.0, 0.0, 3.0};
    const double y[N] = {0.0, 1.0, 1.0};
    const double z[N] = {2.0, 0.0, 0.0, 0.0, 1.0};
    double c[N], u[N], r[N], s[N], w[N], value;
    FastCDWorkspaceF64 cross, assoc, lop;

    if (!allocate(&cross, fastcd_imag_cross_workspace_size)
            || !allocate(&assoc, fastcd_associator_workspace_size)
            || !allocate(&lop, fastcd_lop_commutator_workspace_size))
        return 1;

    fastcd_conjugate_f64(c, x, N);                /* c may equal x */
    fastcd_norm_f64(&value, x, N);
    fastcd_normalize_f64(u, x, N);                /* FASTCD_ZERO_NORM if x == 0 */
    fastcd_reciprocal_f64(r, x, N);               /* conjugate(x) / <x, x> */
    fastcd_square_f64(s, x, N);                   /* linear time; s may equal x */
    fastcd_imag_cross_f64(w, x, y, N, &cross);    /* alternating product (XY - YX)/2 */
    fastcd_commutator_f64(w, x, y, N, &cross);    /* xy - yx */
    fastcd_anticommutator_f64(w, x, y, N);        /* xy + yx, linear time */
    fastcd_associator_f64(w, x, y, z, N, &assoc); /* (xy)z - x(yz) */
    fastcd_lop_commutator_apply_f64(w, x, y, z, N, &lop); /* x(yz) - y(xz) */

    printf("|x| = %g\n", value);
    free(cross.data);
    free(assoc.data);
    free(lop.data);
    return 0;
}
```

In the alternating product, `X` and `Y` are the imaginary parts of `x` and
`y`; see [Derived operations](mathematics.md#derived-operations). The
alternating product and the commutator share
`fastcd_imag_cross_workspace_size`. The associator and the two operator
commutators each have their own query, which reserves room for two
intermediate elements and the multiplication workspace;
`fastcd_rop_commutator_workspace_size` sizes `fastcd_rop_commutator_apply_f64`.

You can also compute products of basis elements, or multiply an element by
a basis element, which permutes its coordinates and changes some signs:

```c
#include <stdio.h>
#include <fastcd/fastcd.h>

int main(void) {
    enum { N = 8 };
    const double x[N] = {1, 2, 3, 4, 5, 6, 7, 8};
    double left[N], right[N];
    FastCDBasisProduct p;

    fastcd_basis_product(&p, N, 1, 2);       /* e_1 e_2 = p.sign * e_(p.index) */
    fastcd_basis_lmul_f64(left, 3, x, N);    /* e_3 x */
    fastcd_basis_rmul_f64(right, x, 3, N);   /* x e_3 */
    printf("e_1 e_2 = %+d e_%zu\n", p.sign, p.index);
    return 0;
}
```

The program prints `e_1 e_2 = +1 e_3`.

## Errors

The returned status tells you why a call failed. It may indicate an invalid
dimension or level, a basis index outside the valid range, a null pointer,
insufficient workspace, overlapping buffers, an invalid stride, size
overflow, a zero norm, or an unsupported algorithm. Use the enum values to
handle these cases in your code and `fastcd_status_string` to obtain a fixed
English message for a log. NaN, infinity, and numerical overflow can appear
in a result without producing an error status.

## Complex coefficients

Complex arithmetic uses `<fastcd/complex.h>`. A complex coefficient is a
`FastCDComplexF64`, two doubles with the real part first, so an element of
dimension `N` is an array of `N` of them. Workspace lengths still count
doubles:

```c
#include <stdio.h>
#include <stdlib.h>
#include <fastcd/complex.h>

int main(void) {
    enum { N = 4 };                                          /* quaternions over C */
    const FastCDComplexF64 x[N] = {{1.0, 1.0}, {0.0, 2.0}};  /* (1+i) + 2i e_1 */
    const FastCDComplexF64 y[N] = {{0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}}; /* e_2 */
    FastCDComplexF64 product[N];
    size_t length;

    if (fastcd_workspace_size_c128(N, FASTCD_OP_MULTIPLY, &length) != FASTCD_SUCCESS)
        return 1;
    double* scratch = length == 0 ? NULL : malloc(length * sizeof(*scratch));
    if (length != 0 && scratch == NULL) return 1;
    FastCDWorkspaceF64 workspace = {scratch, length};

    FastCDStatus status = fastcd_mul_c128(product, x, y, N,
            length == 0 ? NULL : &workspace);
    free(scratch);
    if (status != FASTCD_SUCCESS) return 1;
    for (size_t i = 0; i < N; i++)
        printf("(%g%+gi) ", product[i].real, product[i].imag);
    putchar('\n');
    return 0;
}
```

The program prints `(0+0i) (0+0i) (1+1i) (0+2i)`, that is,
`((1 + i) + 2i e_1) e_2 = (1 + i) e_2 + 2i e_3`. Cayley–Dickson conjugation
does not conjugate the complex coefficients.

## Odd-characteristic finite fields

Finite-field arithmetic uses `<fastcd/finite.h>`. A `FastCDField` describes
F_(p^d), a `FastCDPlanFF` fixes the Cayley–Dickson dimension over that
field, and the caller supplies the workspace. Each coefficient occupies `d`
consecutive `uint32_t` residues, constant term first, so an element of
dimension `N` is `N * d` words and coefficient `i` starts at word `i * d`.
Lengths, strides and workspaces count words:

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
    if (fastcd_ff_cd_workspace_size(&plan, &words) != FASTCD_FF_SUCCESS) return 1;

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

The program prints `Q(xy) = 1`. See
[Arithmetic across coefficient fields](c-api.md#arithmetic-across-coefficient-fields)
for the complete complex and finite-field API.
