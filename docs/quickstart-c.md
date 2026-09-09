# C quick start

To use fastCD in a C program, include `<fastcd/fastcd.h>` and link
`fastcd::fastcd` through CMake, or use `-lfastcd -lm` when linking by hand.
The arithmetic functions and queries with output parameters return a
`FastCDStatus`, and they write their results only when the call returns
`FASTCD_SUCCESS`. String and Boolean queries return their values directly.
The library does not allocate on the heap, and all lengths and strides are
measured in `double` elements.

## One product

```c
#include <stdio.h>
#include <stdlib.h>
#include <fastcd/fastcd.h>

int main(void) {
    const size_t N = 8;                       /* octonions */
    const double x[8] = {1.0, 2.0};
    const double y[8] = {3.0, 0.0, 4.0};
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

You can query the workspace once for a dimension and reuse the allocation
for each multiplication. The output must be a separate array, although the
two inputs may be the same array.

## Choosing the algorithm

The default algorithm used by `fastcd_mul_f64` has been selected for each
dimension based on the benchmark. Batches of two or more products may use a
different algorithm because they can evaluate two products per SIMD
instruction. You can check these choices or select an algorithm yourself:

```c
FastCDAlgorithm chosen;
FastCDAlgorithm chosen_for_batches;
fastcd_default_algorithm(N, &chosen);
fastcd_default_batch_algorithm(N, &chosen_for_batches);
printf("auto selects %s (single) and %s (batch)\n",
        fastcd_algorithm_string(chosen),
        fastcd_algorithm_string(chosen_for_batches));

size_t length;
if (fastcd_algorithm_supports_dimension(FASTCD_ALGORITHM_CARIOW_CARIOWA, N)) {
    fastcd_mul_algorithm_workspace_size(N, FASTCD_ALGORITHM_CARIOW_CARIOWA, &length);
    /* allocate length doubles, then: */
    fastcd_mul_with_algorithm_f64(product, x, y, N,
            FASTCD_ALGORITHM_CARIOW_CARIOWA, &workspace);
}
```

The Cariow–Cariowa algorithm is available through dimension 1024, though it
is not recommended above 64, where quasilinear is already faster. Asking for
it at a higher dimension returns `FASTCD_UNSUPPORTED_ALGORITHM`, while the
direct and quasilinear algorithms support every dimension.

## Batches

To multiply `count` pairs of contiguous elements in one call, use
`fastcd_mul_batch_f64`. Each stride gives the distance between the starts of
successive elements, and an input stride of zero repeats the same element
throughout the batch.

```c
enum { N = 8, COUNT = 1000 };
double* xs = malloc(COUNT * N * sizeof(double));   /* COUNT elements */
double* ys = malloc(COUNT * N * sizeof(double));
double* out = malloc(COUNT * N * sizeof(double));
/* fill xs and ys */
status = fastcd_mul_batch_f64(out, N, xs, N, ys, N, COUNT, N, workspace_or_null);
```

The batch reuses one workspace for all of its products, and the output must
not overlap either input.

## Operators and structure

```c
fastcd_conjugate_f64(c, x, N);              /* c may equal x */
fastcd_norm_f64(&value, x, N);
fastcd_normalize_f64(u, x, N);              /* FASTCD_ZERO_NORM for x == 0 */
fastcd_reciprocal_f64(r, x, N);             /* conjugate(x) / <x,x> */
fastcd_square_f64(s, x, N);                 /* linear time, s may equal x */
fastcd_imag_cross_f64(w, x, y, N, ws);      /* (uv - vu)/2 on imaginary parts */
fastcd_commutator_f64(w, x, y, N, ws);      /* xy - yx */
fastcd_anticommutator_f64(w, x, y, N);      /* xy + yx, linear time */
fastcd_associator_f64(w, x, y, z, N, ws2);  /* (xy)z - x(yz) */
fastcd_lop_commutator_apply_f64(w, a, b, x, N, ws2);   /* a(bx) - b(ax) */
```

Use `fastcd_imag_cross_workspace_size` to size `ws`. For `ws2`, use the query
that matches the operation: `fastcd_associator_workspace_size`,
`fastcd_lop_commutator_workspace_size`, or `fastcd_rop_commutator_workspace_size`.
These queries reserve space for two temporary elements and the multiplication
scratch.

You can also compute products of basis elements or multiply an element by a
basis element:

```c
FastCDBasisProduct p;
fastcd_basis_product(&p, N, 1, 2);          /* e_1 e_2 = p.sign * e_(p.index) */
fastcd_basis_lmul_f64(w, 3, x, N);          /* e_3 x, a signed permutation */
fastcd_basis_rmul_f64(w, x, 3, N);          /* x e_3 */
```

## Errors

The returned status tells you why a call failed. It may indicate an invalid
dimension or level, a basis index outside the valid range, a null pointer,
insufficient workspace, overlapping buffers, an invalid stride, size
overflow, a zero norm, or an unsupported algorithm. Use the enum values to
handle these cases in your code and `fastcd_status_string` to obtain a fixed
English message for a log. NaN, infinity, and numerical overflow can appear
in a result without producing an error status.
