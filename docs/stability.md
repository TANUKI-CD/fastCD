# Numerical stability

Over the real and complex numbers, fastCD computes in IEEE binary64 arithmetic,
where every addition and multiplication rounds its result. Two algorithms that
compute the same exact product therefore return slightly different results, and
an algorithm that saves operations by reorganizing the arithmetic can lose
accuracy in the process. This page asks whether that happens to the quasilinear
product, which fastCD selects automatically at every large dimension,
investigating whether its error is larger than that of direct multiplication,
which evaluates the defining formula. Finite-field arithmetic is exact, so the
question does not arise there and no measurements of error are needed in that
setting.

The conclusion is that, measured against exact products, the error of the
quasilinear product is not much larger than that of direct. On inputs with
random coordinates, its median error is at most 1.6 times that of direct
multiplication over R and 1.8 times over C at every dimension tested, and it
grows slowly with the dimension. The reason is the order in which fastCD adds
long sums, and the sections below explain why that order matters and how fastCD
chooses it, describe how accuracy was measured, report the results for products
and for fused sums of products, give the cost of the summation order, and end
with what the results mean for code that uses fastCD.

## Why the order of a sum matters

A floating-point sum rounds after every addition, and each rounding can change
the partial sum by up to `u = 2^-53` relative to its value. A term added early
in a sum therefore passes through every later addition and is exposed to every
later rounding, and the *rounding depth* of a sum, the largest number of
additions that any one term passes through on its way to the total, measures
that exposure. Added in sequence, a sum of `m` terms has rounding depth `m - 1`,
and its error can grow in proportion to its length.

Direct multiplication adds only short chains, evaluating the doubling formula
`(a, b)(c, d) = (ac - conjugate(d) b, da + b conjugate(c))` recursively. For
real coefficients, generated kernels at dimension 16 and below add the terms of
each output coordinate in sequence, at most 16 of them, and each level of the
recursion above adds one more term to every coordinate. At dimension `N >= 16`
the rounding depth of direct multiplication is therefore at most
`15 + log2(N / 16)`, which is 22 at dimension 2048.

The quasilinear product, in contrast, contains sums whose length grows with the
dimension. Writing `alpha` and `beta` for the scalar parts of `x` and `y`, `X`
and `Y` for their imaginary parts, `<X, Y>` for the sum of the products of the
coordinates of `X` and `Y`, and `X cross Y` for their alternating product, it
splits a product as `xy = (alpha beta - <X, Y>) + alpha Y + beta X + X cross Y`
(see [How quasilinear multiplication
works](mathematics.md#how-quasilinear-multiplication-works)). The pairing
`<X, Y>`, which gives the scalar coordinate of the product, is a sum of `N - 1`
terms. The alternating product is computed recursively, and at every level of
the recursion the *nonrecursive map*, the part of the level that needs no
recursive call, produces one coordinate as a sum of up to `N/2 - 1` terms. Added
in sequence, a term of the pairing at dimension 2048 would pass through 2,046
additions, against at most 22 in direct multiplication, so these long sums are
where the quasilinear product could lose accuracy over the direct algorithm.

## Balanced accumulation

fastCD adds the long sums of the quasilinear product by *balanced accumulation*,
in which the terms are added in sequence in blocks of eight and the block totals
are combined pairwise, like the digits of a binary counter: whenever two totals
cover the same number of blocks, they are added. A sum of `m` terms still takes
`m - 1` additions, but its rounding depth is at most
`8 + ceil(log2(ceil(m / 8)))`. For the pairing at dimension 2048, that is 16
instead of 2,046, comparable to the depth of direct multiplication.

Balanced accumulation covers every sum of growing length in the quasilinear
product, over R and over C, and over C it also covers the pairing of complex
coefficients that the quadratic norm, the square, the anticommutator, and the
reciprocal use. Their real counterparts, by contrast, add their sums in
sequence: `fastcd_inner_f64`, the square, and the anticommutator directly, and
the norm functions and the reciprocal with scaling to avoid overflow. The rest
of fastCD's arithmetic needs no balancing, because direct multiplication, the
fixed formula at the bottom of the quasilinear recursion called the octonion
leaf, and finite-field arithmetic contain no sums of growing length.

The algorithm of Cariow and Cariowa, available for real coefficients, has long
sums of its own. It computes each output coordinate as an entry of a
Walsh-Hadamard convolution minus twice a *correction sum*, which adds the
products selected by a sign pattern from the `N` coordinate indices. From
dimension 128 on, the indices of a correction sum are taken in blocks of 256,
and within a block the selected terms go to four interleaved partial sums that
are combined pairwise at the end of the block. Above dimension 256, where a
correction sum spans more than one block, the block totals are combined by
balanced accumulation.

Every form of a product adds its terms in the same order: the scalar and the
two-lane SIMD kernels, single products, and batches. Their results are therefore
identical bit for bit, and the accuracy study below confirmed this on every
operand pair it measured, for the quasilinear product and for Cariow-Cariowa.

## How accuracy was measured

### Error measures

The study measures errors with the Euclidean norm `||.||_2`, which over C is the
Euclidean norm of the complex coefficients rather than the quadratic form of
[The quadratic form](mathematics.md#the-quadratic-form), since that form can
vanish on nonzero elements. For operands `x` and `y` with exact product `z` and
computed product `z_hat`, it records four quantities:

```text
E_abs = ||z_hat - z||_2
E_rel = E_abs / ||z||_2                    (z nonzero)
eta   = E_abs / (u ||x||_2 ||y||_2)
chi   = ||x||_2 ||y||_2 / ||z||_2          (z nonzero)
```

The main measure is `eta`, the error in units of `u ||x||_2 ||y||_2`, the scale
set by the sizes of the operands, so that an `eta` of 1 is an error of one unit
roundoff times the product of the operand norms. Unlike the relative error, it
is defined for every operand pair, which matters because nonzero operands can
have a zero product in real dimension 16 and above and in complex dimension 2
and above.

The other two measures are related by `E_rel = u eta chi`, where `chi` is large
when the exact product is much smaller than its operands, as it is near a zero
divisor, so that there even a small `eta` means a large relative error.

### Exact reference

`benchmarks/stability.py` computes every reference product exactly, without
fastCD arithmetic, by converting each binary64 coefficient to an integer times a
power of two and evaluating the product as a signed sum of coefficient products
in arbitrary-precision integers. The basis signs used in this reference come
from the doubling formula in integer arithmetic and are checked on all 341 basis
pairs through dimension 16 against a separate rational implementation of the
formula. Norms and errors are evaluated with MPFR at 256 bits, and selected
products are repeated at 512 bits as a check. The script needs NumPy and gmpy2,
which the library itself does not use.

### Three computations

Every operand pair is multiplied in three ways:

- **B** is the quasilinear product with balanced accumulation, as fastCD
  computes it.
- **S** is the quasilinear product with every long sum added in sequence.
- **D** is direct multiplication.

S and B perform the same operations and differ only in the order of their
additions, so comparing them isolates the effect of the summation order, while
comparing B with D answers the question of this page.

S comes from a build of commit `982b9da`, which adds these sums in sequence, and
B and D come from a build of commit `93c9bd8`, the source of fastCD 1.5.0. The
two builds contain the same direct multiplication and return identical direct
products in every case. Both were compiled with GCC 16.2.1 on Linux, with
`-O3 -DNDEBUG -fno-fast-math -ffp-contract=off`, so that every operation rounds
as written and no multiplication is fused with an addition, and the study ran on
an Intel Xeon Gold 6148.

### Inputs

A *case* is one operand pair at one dimension, multiplied by all three
computations. The study has 13,584 cases, 6,976 real and 6,608 complex, at real
dimensions 2 through 2048 and complex dimensions 2 through 512, and they are
drawn from 21 real and 19 complex *families*, each of which exercises one kind
of input:

- **Dense:** unit-norm Gaussian and uniform coordinates, the typical case, and
  exactly representable Rademacher coordinates.
- **Sparse:** one, four, or `N/8` nonzero coordinates.
- **Correlated:** `y = x`, and `y = +/-x + 2^-40 noise`.
- **Cancellation:** a scalar coordinate constructed to cancel, exact dyadic
  cancellation, and long sums whose end terms `+/-2^53` cancel each other.
- **Magnitude:** coordinates whose magnitudes alternate over the exponents `-24`
  to `24`.
- **Structure:** scalar-only and imaginary-only operands, and operands built
  from repeated or recursively signed blocks.
- **Zero divisors (real, dimension 16 and above):** the exact zero divisor
  `(e1 + e10)(e4 - e15) = 0`, and perturbations of it by `2^-20`, `2^-40`, and
  `2^-52`.
- **Isotropic (complex):** operands with `Q(x) = 0`, exact and perturbed by
  `2^-40`.

The dense families have 512 cases per dimension through dimension 32, 192 at 64
and 128, 64 at 256 and 512, and 16 above. Their seeds derive from
`numpy.random.SeedSequence(20261105, ...)`, and every operand pair is stored
with the raw results. Some constructed families produce the same operands for
several seeds, so the 13,584 cases contain 12,572 distinct operand pairs.

## Accuracy of products

The tables below are generated by `benchmarks/stability_report.py` from the
archived results of the *full profile* of the study, its complete run, as
opposed to the short smoke profile that only tests the tooling. They come in a
real set and a complex set, each with three views: the three computations on
dense inputs at every dimension, the growth of the error of B with the
dimension, and every family at dimension 512.

In every table, a cell of the form "S / B / D" gives one statistic for the three
computations, the column "pairs" counts cases, "p99" is the 99th percentile, and
the dense rows pool the Gaussian and uniform families. The *scalar error* is the
absolute error of coordinate 0, the coordinate that the pairing `<X, Y>`
computes, so it shows the effect of the summation order most directly. "B better
/ tied" gives the fraction of cases in which B has a smaller error than S and
the fraction in which the two errors are equal. The growth tables divide the
median `eta` by two reference rates of growth, `sqrt(N) log2 N` and `N`, and
give the median and largest relative errors.

<!-- BEGIN STABILITY ACCURACY -->

13,584 paired cases from the full profile. S is quasilinear multiplication with sequential accumulation, B is quasilinear multiplication with balanced accumulation, and D is direct multiplication; all three use the same stored operands. Dense rows pool the Gaussian and uniform families. The scalar error is the absolute error of coordinate 0, and "B better / tied" compares the stored `E_abs` of B with that of S. Empirical p99 values and maxima describe this corpus only.

**Real dense inputs**

| N | pairs | median eta S / B / D | p99 eta S / B / D | max eta S / B / D | median scalar error S / B / D | B better / tied |
|---:|---:|:---|:---|:---|:---|:---|
| 2 | 512 | 0.444 / 0.444 / 0.444 | 1.22 / 1.22 / 1.22 | 1.51 / 1.51 / 1.51 | 1.98e-17 / 1.98e-17 / 1.98e-17 | 0% / 100% |
| 4 | 512 | 0.642 / 0.642 / 0.708 | 1.59 / 1.59 / 1.6 | 1.81 / 1.81 / 1.93 | 2.78e-17 / 2.78e-17 / 2.78e-17 | 0% / 100% |
| 8 | 512 | 0.985 / 0.985 / 0.963 | 2.03 / 2.03 / 1.87 | 2.65 / 2.65 / 2.21 | 3.14e-17 / 3.14e-17 / 3.97e-17 | 0% / 100% |
| 16 | 512 | 1.48 / 1.46 / 1.26 | 2.95 / 2.92 / 2.1 | 3.68 / 3.69 / 2.38 | 4.11e-17 / 3.65e-17 / 4.51e-17 | 27% / 53% |
| 32 | 512 | 1.8 / 1.79 / 1.4 | 3.13 / 3.13 / 1.99 | 3.72 / 3.7 / 2.48 | 4.71e-17 / 3.8e-17 / 4.46e-17 | 54% / 16% |
| 64 | 192 | 2.07 / 2.05 / 1.5 | 2.79 / 2.79 / 1.84 | 3 / 2.9 / 1.88 | 5.41e-17 / 2.77e-17 / 4.16e-17 | 68% / 1% |
| 128 | 192 | 2.26 / 2.24 / 1.57 | 2.96 / 2.87 / 1.79 | 3.06 / 2.93 / 1.95 | 6.55e-17 / 2.95e-17 / 2.66e-17 | 71% / 1% |
| 256 | 64 | 2.47 / 2.41 / 1.62 | 2.83 / 2.78 / 1.77 | 2.9 / 2.8 / 1.8 | 5.57e-17 / 2.7e-17 / 5.27e-17 | 89% / 0% |
| 512 | 64 | 2.66 / 2.58 / 1.71 | 2.98 / 2.88 / 1.81 | 3.02 / 2.9 / 1.82 | 7.5e-17 / 1.66e-17 / 4.48e-17 | 95% / 0% |
| 1024 | 16 | 2.81 / 2.73 / 1.74 | 3.03 / 2.93 / 1.81 | 3.04 / 2.94 / 1.81 | 9.98e-17 / 2.49e-17 / 2.41e-17 | 100% / 0% |
| 2048 | 16 | 3.03 / 2.91 / 1.8 | 3.15 / 3.02 / 1.83 | 3.15 / 3.03 / 1.84 | 4.33e-16 / 4.09e-16 / 4.14e-16 | 100% / 0% |

**Real growth with dimension**, B on dense inputs

| N | median eta | eta / (sqrt(N) log2 N) | eta / N | median relative error | max relative error |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.444 | 0.314 | 0.222 | 4.93e-17 | 1.67e-16 |
| 4 | 0.642 | 0.16 | 0.16 | 7.13e-17 | 2.01e-16 |
| 8 | 0.985 | 0.116 | 0.123 | 1.09e-16 | 2.94e-16 |
| 16 | 1.46 | 0.0916 | 0.0916 | 1.64e-16 | 4.57e-16 |
| 32 | 1.79 | 0.0632 | 0.0559 | 2e-16 | 4.02e-16 |
| 64 | 2.05 | 0.0426 | 0.032 | 2.27e-16 | 3.48e-16 |
| 128 | 2.24 | 0.0283 | 0.0175 | 2.51e-16 | 3.15e-16 |
| 256 | 2.41 | 0.0188 | 0.00942 | 2.66e-16 | 3.07e-16 |
| 512 | 2.58 | 0.0127 | 0.00505 | 2.89e-16 | 3.16e-16 |
| 1024 | 2.73 | 0.00854 | 0.00267 | 3.03e-16 | 3.23e-16 |
| 2048 | 2.91 | 0.00585 | 0.00142 | 3.26e-16 | 3.39e-16 |

**Real families at N=512**

| family | pairs | median eta S / B / D | max relative error B | B better / tied |
|:---|---:|:---|---:|:---|
| gaussian | 32 | 2.75 / 2.66 / 1.69 | 3.16e-16 | 97% / 0% |
| uniform | 32 | 2.58 / 2.52 / 1.71 | 3.12e-16 | 94% / 0% |
| rademacher | 8 | 0 / 0 / 0 | 0 | 0% / 100% |
| sparse_basis | 8 | 0.375 / 0.375 / 0.375 | 6.69e-17 | 0% / 100% |
| sparse_four | 8 | 0.976 / 0.976 / 0.355 | 1.4e-16 | 0% / 100% |
| sparse_eighth | 8 | 2.25 / 2.26 / 0.9 | 2.56e-16 | 75% / 0% |
| correlated_plus | 8 | 3.46 / 1.92 / 1.59 | 2.33e-16 | 88% / 12% |
| correlated_minus | 8 | 6.36 / 1.93 / 1.61 | 2.48e-16 | 100% / 0% |
| repeated | 8 | 3.85 / 0.623 / 0.67 | 1.59e-16 | 88% / 0% |
| scalar_cancellation | 8 | 2.03 / 1.99 / 1.63 | 2.3e-16 | 88% / 0% |
| dyadic_cancellation | 8 | 0 / 0 / 0 | 0 | 0% / 100% |
| long_sum_cancellation | 8 | 16 / 1.08 / 2.87 | 1.2e-16 | 100% / 0% |
| alternating_magnitude | 8 | 1.7 / 1.68 / 0.422 | 2.09e-16 | 88% / 0% |
| scalar_only | 8 | 0.316 / 0.316 / 0.316 | 9.58e-17 | 0% / 100% |
| imaginary_only | 8 | 2.67 / 2.61 / 1.66 | 3.12e-16 | 88% / 0% |
| repeated_blocks | 8 | 6.67 / 1.91 / 1.67 | 2.68e-16 | 88% / 12% |
| recursive_blocks | 8 | 1.61 / 1.59 / 1.45 | 2.89e-16 | 88% / 12% |
| zero_divisor | 8 | 0 / 0 / 0 | — | 0% / 100% |
| near_zero_divisor_20 | 8 | 2.39 / 2.39 / 0.786 | 3.14e-10 | 100% / 0% |
| near_zero_divisor_40 | 8 | 2.11 / 2.11 / 0.25 | 0.000267 | 0% / 100% |
| near_zero_divisor_52 | 8 | 1.27 / 1.27 / 2.04e-16 | 0.648 | 0% / 100% |

**Complex dense inputs**

| N | pairs | median eta S / B / D | p99 eta S / B / D | max eta S / B / D | median scalar error S / B / D | B better / tied |
|---:|---:|:---|:---|:---|:---|:---|
| 2 | 512 | 0.679 / 0.679 / 0.679 | 1.48 / 1.48 / 1.48 | 1.75 / 1.75 / 1.75 | 4.84e-17 / 4.84e-17 / 4.84e-17 | 0% / 100% |
| 4 | 512 | 0.919 / 0.919 / 0.837 | 1.96 / 1.96 / 1.67 | 3.51 / 3.51 / 1.93 | 5.93e-17 / 5.56e-17 / 5.93e-17 | 32% / 38% |
| 8 | 512 | 1.44 / 1.43 / 0.981 | 2.85 / 2.85 / 1.54 | 3.22 / 3.2 / 1.8 | 7.36e-17 / 6.86e-17 / 6.2e-17 | 37% / 21% |
| 16 | 512 | 1.71 / 1.71 / 1.11 | 2.75 / 2.75 / 1.53 | 3.58 / 3.55 / 1.66 | 8.71e-17 / 6.4e-17 / 5.79e-17 | 49% / 9% |
| 32 | 512 | 1.97 / 1.94 / 1.21 | 2.76 / 2.76 / 1.49 | 3.89 / 3.91 / 1.62 | 9.89e-17 / 7.17e-17 / 4.78e-17 | 63% / 1% |
| 64 | 192 | 2.25 / 2.22 / 1.28 | 2.87 / 2.86 / 1.53 | 3.07 / 2.89 / 1.54 | 9.93e-17 / 3.97e-17 / 4.36e-17 | 73% / 0% |
| 128 | 192 | 2.41 / 2.35 / 1.37 | 2.84 / 2.82 / 1.51 | 3.05 / 3.05 / 1.54 | 9.74e-17 / 7.77e-17 / 4.98e-17 | 86% / 0% |
| 256 | 64 | 2.56 / 2.5 / 1.44 | 3.23 / 2.81 / 1.54 | 3.3 / 2.85 / 1.55 | 1.86e-16 / 1.86e-16 / 2.8e-17 | 89% / 0% |
| 512 | 64 | 2.76 / 2.68 / 1.52 | 2.97 / 2.89 / 1.59 | 2.97 / 2.9 / 1.59 | 1.21e-15 / 1.22e-16 / 1.38e-16 | 98% / 0% |

**Complex growth with dimension**, B on dense inputs

| N | median eta | eta / (sqrt(N) log2 N) | eta / N | median relative error | max relative error |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.679 | 0.48 | 0.34 | 7.73e-17 | 2.51e-16 |
| 4 | 0.919 | 0.23 | 0.23 | 1.06e-16 | 3.72e-16 |
| 8 | 1.43 | 0.169 | 0.179 | 1.62e-16 | 4.26e-16 |
| 16 | 1.71 | 0.107 | 0.107 | 1.93e-16 | 3.83e-16 |
| 32 | 1.94 | 0.0687 | 0.0607 | 2.18e-16 | 4.09e-16 |
| 64 | 2.22 | 0.0462 | 0.0347 | 2.43e-16 | 3.28e-16 |
| 128 | 2.35 | 0.0297 | 0.0184 | 2.61e-16 | 3.27e-16 |
| 256 | 2.5 | 0.0195 | 0.00976 | 2.83e-16 | 3.2e-16 |
| 512 | 2.68 | 0.0132 | 0.00524 | 2.98e-16 | 3.29e-16 |

**Complex families at N=512**

| family | pairs | median eta S / B / D | max relative error B | B better / tied |
|:---|---:|:---|---:|:---|
| gaussian | 32 | 2.82 / 2.74 / 1.51 | 3.29e-16 | 97% / 0% |
| uniform | 32 | 2.71 / 2.64 / 1.52 | 3.03e-16 | 100% / 0% |
| rademacher | 8 | 0 / 0 / 0 | 0 | 0% / 100% |
| sparse_basis | 8 | 0.636 / 0.636 / 0.34 | 1.54e-16 | 0% / 100% |
| sparse_four | 8 | 1.76 / 1.76 / 0.609 | 2.58e-16 | 0% / 100% |
| sparse_eighth | 8 | 2.45 / 2.43 / 1.04 | 2.83e-16 | 88% / 0% |
| correlated_plus | 8 | 1.91 / 1.91 / 1.38 | 3.37e-15 | 88% / 0% |
| correlated_minus | 8 | 1.98 / 1.89 / 1.37 | 3e-15 | 88% / 0% |
| repeated | 8 | 0.163 / 0.101 / 0.508 | 2.44e-16 | 75% / 0% |
| scalar_cancellation | 8 | 2.04 / 2.01 / 1.46 | 2.36e-16 | 100% / 0% |
| dyadic_cancellation | 8 | 0 / 0 / 0 | 0 | 0% / 100% |
| long_sum_cancellation | 8 | 16 / 1.16 / 1.39 | 1.29e-16 | 100% / 0% |
| alternating_magnitude | 8 | 1.84 / 1.76 / 0.634 | 2.42e-16 | 88% / 0% |
| scalar_only | 8 | 0.32 / 0.32 / 0.32 | 1.01e-16 | 0% / 100% |
| imaginary_only | 8 | 2.86 / 2.77 / 1.52 | 3.15e-16 | 100% / 0% |
| repeated_blocks | 8 | 7.76 / 1.89 / 1.26 | 2.97e-16 | 100% / 0% |
| recursive_blocks | 8 | 1.83 / 1.77 / 1.33 | 3.23e-16 | 100% / 0% |
| isotropic | 8 | 1.15 / 1.15 / 0.455 | 1.36e-16 | 0% / 100% |
| near_isotropic | 8 | 2.21 / 2 / 1.39 | 2.4e-16 | 100% / 0% |

The archived CSV summary also gives, for every family, the count, the median, mean, p95, p99 and maximum of eta, the relative and scalar errors, the paired improvement fractions and descriptive Wilson 95% intervals. Groups with fewer than 100 samples have an empirical p99 close to their maximum and large sampling uncertainty.

<!-- END STABILITY ACCURACY -->

### Reading the tables

**Dense inputs.** On dense inputs the three computations are about equally
accurate: the median `eta` of B is between 0.91 and 1.61 times that of D over R
and between 1.00 and 1.77 times over C. From dimension 64 on it grows by less
than 0.2 for each doubling of the dimension, reaching 2.91 at real dimension
2048, and the largest relative error of B on these inputs is `4.6e-16` over R
and `4.3e-16` over C, about four unit roundoffs. The sequential sums of S are
barely less accurate here, because the rounding errors of a long sum of random
terms tend to cancel one another.

**Small dimensions.** Over R, S and B are tied in every case through dimension
8, since no sum there has more than eight terms, so balanced accumulation adds
every sum in sequence and the two computations perform identical arithmetic. At
dimension 16 the pairing has 15 terms, and the two computations first differ, in
the scalar coordinate only. Over C, by contrast, they already differ at
dimension 4, because B forms the pairing `<X, Y>` before subtracting it from
`alpha beta`, while S subtracts its terms from `alpha beta` one at a time.

**Where the order matters.** The order of a sum matters most when its terms are
structured rather than random. On long sums whose end terms cancel, at dimension
512, the median `eta` falls from 16.0 with S to 1.08 with B over R, where D
gives 2.87, and from 16.0 to 1.16 over C, where D gives 1.39. On operands built
from repeated blocks it falls from 6.7 to 1.9 over R and from 7.8 to 1.9 over C.

**Largest errors.** The largest `eta` of B anywhere in the corpus is 3.7 over R
and 4.1 over C, against 2.9 and 2.2 for D and 78 and 18 for S, and both of the
largest errors of S come from operands built from repeated blocks, at real
dimension 2048 and complex dimension 512.

**Not better on every input.** No order of summation is the most accurate on
every input: pooling the dense cases from dimension 32 on, B is more accurate
than S in 66% of the real and 73% of the complex cases, but less accurate in 27%
of each. The largest increases in `eta` are from 0.60 to 1.83, for a real
operand pair at dimension 16 with `y = -x + 2^-40 noise`, and from 0.30 to 1.45,
for a near-isotropic complex pair at dimension 4, both at dimensions where the
two computations differ only in the scalar coordinate.

**Near zero divisors.** When the exact product is tiny compared with its
operands, a small `eta` can still mean a large relative error. For the zero
divisor perturbed by `2^-52` at dimension 512, `chi` is about `4.5e15`, and
although S and B both have an `eta` of 1.27, an ordinary error for operands of
these sizes, their relative error is 0.64. Direct multiplication, by contrast,
is nearly exact on these sparse operands, with a relative error of `9.9e-17`.

### Cariow-Cariowa

The study also multiplied every real case through dimension 1024 with
Cariow-Cariowa, in both builds, and balanced correction sums change its results
only at dimensions 512 and 1024, where a correction sum spans more than one
block. Of the 6,808 cases, 265 are more accurate with balanced correction sums,
23 are less accurate, and the rest are equally accurate. Every dense case at
dimensions 512 and 1024 is more accurate: at 1024 the median `eta` falls from
5.59 to 3.80, and the median relative error from `6.2e-16` to `4.3e-16`. The
largest `eta` among all Cariow-Cariowa cases falls from 183 to 80, and the
largest increase is from 35.3 to 41.4, for an operand pair built from repeated
blocks at dimension 512. The archived `cc-accuracy-summary.json` holds the
details.

## Accuracy of fused sums

fastCD also adds up many products in a way whose additions differ from those of
the products themselves. A *fused sum* computes
`x_0 y_0 + ... + x_(P-1) y_(P-1)` for `P` pairs of real-coefficient elements
without forming the products: it accumulates each multiplication of the
algorithm over all the pairs and applies the algorithm's output additions once,
to the accumulated values (see [Fused sums of
products](mathematics.md#fused-sums-of-products)). Its additions therefore run
in a different order from those of a batch of products followed by a sum: each
of eight lanes adds its terms in sequence within a block of 64 elements, the
block totals are added in sequence, and the eight lanes are combined pairwise at
the end.

A separate study, `benchmarks/stability_sums.py`, measured the accuracy of this
order on 689 sums and 84 covariances at dimensions 1 through 1024. For a
computed sum `T_hat` with exact value `T`, it reports

```text
eta = ||T_hat - T||_2 / (u sum_p ||x_p||_2 ||y_p||_2),     u = 2^-53,
```

which measures the error against the operand scales of the individual products,
as the `eta` of a single product does. The reference uses no fastCD arithmetic:
every term `x_p[i] y_p[j]` is split into its rounded product and the exact
rounding error, and `math.fsum` adds the terms of an output coordinate with a
single rounding, while a second pass with the negated result gives the
remainder, so each coordinate is known to about 106 bits. Exact integer
arithmetic checks this reference on 54 cases.

The fused sum, with the automatic algorithm, is compared with three other ways
to compute the same sum:

- **Products, in sequence:** a batch of products added in sequence.
- **Products, pairwise:** the same products added with NumPy's pairwise `sum`.
- **Cross moments:** the binary64 matrix of cross moments `sum_p x_p[i] y_p[j]`,
  computed with NumPy 2.5.5 `matmul` and contracted with the basis signs.

Each cell of the following table is the median and the maximum of `eta` over
Gaussian and uniform samples of 1,024 elements:

| N | Cases | Fused | Products, in sequence | Products, pairwise | Cross moments |
| ---: | ---: | --- | --- | --- | --- |
| 1 | 16 | 0.05 / 0.25 | 0.16 / 0.75 | 0.05 / 0.15 | 0.06 / 0.20 |
| 2 | 16 | 0.06 / 0.18 | 0.15 / 0.52 | 0.06 / 0.14 | 0.35 / 0.78 |
| 4 | 16 | 0.06 / 0.10 | 0.27 / 0.58 | 0.05 / 0.08 | 0.29 / 0.68 |
| 8 | 16 | 0.07 / 0.09 | 0.22 / 0.35 | 0.06 / 0.08 | 0.33 / 0.46 |
| 16 | 16 | 0.09 / 0.13 | 0.28 / 0.42 | 0.08 / 0.13 | 0.29 / 0.39 |
| 32 | 16 | 0.09 / 0.13 | 0.30 / 0.43 | 0.08 / 0.09 | 0.19 / 0.22 |
| 64 | 16 | 0.11 / 0.14 | 0.29 / 0.36 | 0.08 / 0.10 | 0.19 / 0.22 |
| 128 | 8 | 0.12 / 0.14 | 0.30 / 0.32 | 0.09 / 0.10 | 0.20 / 0.22 |
| 256 | 8 | 0.13 / 0.15 | 0.31 / 0.35 | 0.09 / 0.10 | 0.24 / 0.25 |
| 512 | 4 | 0.14 / 0.14 | 0.30 / 0.34 | 0.10 / 0.10 | 0.28 / 0.29 |
| 1024 | 2 | 0.15 / 0.15 | 0.30 / 0.31 | 0.10 / 0.10 | 0.35 / 0.35 |

On these samples the median `eta` of the fused sum grows only from 0.05 to 0.15
over the whole range of dimensions, a third to a half of the error of the
products added in sequence and up to 1.5 times that of the products added
pairwise. The advantage over the sequential sum follows from the order of
additions, since the longest chain of additions in a fused sum is the chain of
block totals, one addition for every 64 elements, far shorter than a sequential
sum of 1,024 products.

The differences are larger on the other families, which give these maxima of
`eta`, pooled over all dimensions with samples of 1,024 elements:

| Family | Cases | Fused | Products, in sequence | Products, pairwise | Cross moments |
| --- | ---: | ---: | ---: | ---: | ---: |
| `y = x` | 64 | 2.60 | 14.97 | 1.40 | 14.44 |
| `y = conjugate(x)` | 64 | 2.07 | 18.63 | 1.89 | 7.58 |
| second half cancels the first | 64 | 0.05 | 0.64 | 0.00 | 0.42 |
| coordinates `10^6` plus Gaussian | 67 | 2.06 | 21.95 | 1.07 | 14.89 |
| element magnitudes `2^-24` to `2^24` | 64 | 0.87 | 2.50 | 0.75 | 2.03 |
| one nonzero coordinate per element | 56 | 0.12 | 0.73 | 0.09 | 0.27 |

On these structured samples, where the errors of the products do not cancel at
random, the largest error of the fused sum is 2.60, against up to 21.95 for the
products added in sequence and 14.89 for the cross moments. The one family not
shown, sums of `+/-1` coordinates, was exact in all 64 cases and every evaluation.

The choice of algorithm inside a fused sum also matters. The fused sum with the
direct algorithm, which an explicit request can select at any dimension, is less
accurate at high dimension, because each of its output coordinates accumulates
`N` products for every element: on the dense samples its median `eta` is 0.07 at
dimension 8, the largest at which the automatic algorithm uses it, and grows to
0.32 at dimension 64 and 1.19 at 1024, against 0.11 and 0.15 for the quasilinear
sum.

The covariance is built on the fused sum: `fastcd_covariance_f64` computes
`(1 / (P - ddof)) sum_p (x_p - mean(x)) conjugate(y_p - mean(y))` as a fused sum
of centered records. Its error is measured against the scale
`sum_p ||x_p - mean(x)||_2 ||y_p - mean(y)||_2 / (P - 1)`, and it is compared
with products and with cross moments of samples centered beforehand with NumPy,
on 28 cases per family at dimensions 1 through 64. Each cell is the median and
the maximum of `eta`:

| Sample | `fastcd_covariance_f64` | Centered products | Centered cross moments |
| --- | --- | --- | --- |
| Gaussian | 0.09 / 0.14 | 0.30 / 0.64 | 0.22 / 0.87 |
| `10^6` plus Gaussian | 0.08 / 0.11 | 0.29 / 0.47 | 0.21 / 0.46 |
| `10^9` plus Gaussian | 0.11 / 0.21 | 4,903 / 8,166 | 4,903 / 8,166 |

The covariance keeps its accuracy when the mean is `10^9` times the spread,
where centering with a rounded mean gives errors of several thousand, because a
rounded mean of about `10^9` can be off by `6e-8` and that error enters every
centered value. fastCD avoids this by never forming the mean: it holds it as the
first record `r` plus the mean `d` of the differences `x_p - r`, and centers
each record as `(x_p - r) - d`. Since `x_p` and `r` are close, `x_p - r` is
exact, and `d` is small, so its rounding error is small compared with the spread
(see [Covariance](mathematics.md#covariance)).

Although the study ran only the AVX-512 kernel, its results also hold for the
AVX2 kernel and for the portable kernel on targets with a fused multiply-add
instruction, which return the same bits, as `tests/test_fused_sum.c` checks; on
other targets the portable kernel rounds each product and addition separately,
and its results differ in the last bits. Elements stored as records and as
coordinate planes gave identical bits in all 689 sums, so the results hold for
both layouts. They also extend to other sample sizes: with samples of 64 and of
4,096 elements at dimensions through 64, the largest `eta` of the fused sum was
3.44 and 2.04.

## What balanced accumulation costs

Balanced accumulation keeps the number of operations the same, but storing and
combining block totals takes time. To measure that cost,
`benchmarks/stability_speed.py` timed both builds on one logical CPU in twelve
paired blocks; in each block both builds time the same products, and which build
goes first alternates from block to block. The two builds contain the same
direct multiplication, so the ratio of their direct timings, the *direct
control*, shows how much two builds of identical code differ in time on this
host, and a cost within its range cannot be told apart from that difference.

From dimension 64 on, balanced accumulation makes real products 1% to 4% slower
with the default SIMD kernels, close to the direct control's range of 0.96 to
1.01, and a consistent 9% to 11% slower in builds without the explicit SIMD
kernels. Complex products are 7% to 12% slower, about as much as their direct
control of 1.07 to 1.09, so the study cannot separate that cost from the
difference between builds. Balanced correction sums make Cariow-Cariowa at most
6% slower at dimensions 512 and 1024. The full tables are in the archived
`speed-tables.md` and `cc-speed-tables.md`.

## What this means for your code

- **Use the default algorithms.** On dense inputs the median error of the
  quasilinear product is at most 1.6 times that of direct multiplication over R
  and 1.8 times over C, and its largest error in the whole corpus is 3.7 over R
  and 4.1 over C, in units of `u ||x||_2 ||y||_2`.
- **Prefer the default to Cariow-Cariowa at high dimensions.** Even with
  balanced correction sums, its median `eta` on dense inputs at dimension 1024
  is 3.80, against 2.73 for the quasilinear product.
- **Judge nearly vanishing products by `eta`.** When a product can be much
  smaller than its operands, as near a zero divisor, its error stays small
  compared with `||x||_2 ||y||_2`, but its relative error can be large with any
  algorithm.
- **Sum many products with a fused sum.** On every family measured that rounds
  at all, its median error is below that of a batch of products added in
  sequence.
- **Expect identical bits only within one build and algorithm.** Single
  products, batches, and the scalar and SIMD kernels return identical results,
  but different algorithms, releases, and compilers can round differently.

## Limitations

- The study is empirical. Its medians and maxima describe this corpus, not all
  inputs, and they do not show that the quasilinear product is as accurate as
  direct multiplication on every input.
- Nothing was measured above real dimension 2048 or complex dimension 512, or
  with overflow, underflow, subnormal numbers, infinities, NaNs, or nondefault
  rounding modes.
- Associators, operator commutators, and reciprocals multiply rounded
  intermediate results and were not measured, so the results for products do not
  describe them.
- The fused sums were measured on real samples of 64 to 4,096 elements with the
  AVX-512 kernel, with at most 16 cases per dimension and family, and the
  covariance only through dimension 64.
- Accuracy and timings were measured only with GCC on Linux.

## Checking and reproducing the results

The study of products is archived in
[`benchmark-results/stability-20261120/`](../benchmark-results/stability-20261120/README.md),
whose README lists every file. Its published files are part of the repository
and the source distributions: the summaries, the tables, the charts, the build
records, and `key-figures.json`, which holds the figures quoted on this page.
The raw per-case results, the stored operands, the timing observations, and the
logs are under `raw/` and are not distributed, although `SHA256SUMS` lists both
kinds. The fused-sum study is archived separately, with its results and the
command that produced them, in
[`benchmark-results/t440-20261120/`](../benchmark-results/t440-20261120/README.md).

To check the study, run from the repository root:

```sh
python3 tools/check_stability_archives.py
```

The checker uses only the Python standard library. It compares every published
file with `SHA256SUMS`, checks that the tables on this page are the tables of
the study, and, when the git history is available, checks that the measured
commit contains exactly the source files that the study recorded. With `--full`
it also requires the raw files, checks their hashes, and regenerates the tables
from the raw results, which needs NumPy.

To repeat the study, run `benchmarks/stability_study.py` with a Python that has
NumPy, gmpy2, and Matplotlib:

```sh
python3 benchmarks/stability_study.py --baseline-commit 982b9da --commit 93c9bd8 \
    --output benchmark-results/stability-YYYYMMDD --cpu CPU --accuracy-cpu CPU
```

The script builds both commits from clean worktrees, pins the timings and the
accuracy study to the given logical CPUs, runs every step, and writes the
README, the tables, and `SHA256SUMS`; with `--document docs/stability.md` it
also rewrites the tables on this page. Running the same command again resumes an
interrupted study and skips the finished steps.

A rerun on another version of NumPy can draw slightly different random operands,
because NumPy does not guarantee that the distribution methods of its
`Generator` return identical bits across versions. The dense rows of the tables
can then differ slightly, while the structured families and the exact reference
are unaffected, and the raw results record a digest of all measured operands
against which a rerun can be checked where those files are available.

The test suite guards the properties that the study relies on.
`tests/test_stability.c` (CTest `fastcd.stability`) checks balanced accumulation
against an independent recursive description for every sum length from 0 to
2051, and checks real and complex products at dimensions 2 through 512 against
an exact oracle on dyadic inputs. `tests/test_fused_sum.c` (CTest
`fastcd.fused_sum`) checks that every fused-sum kernel and layout returns the
exact sum on dyadic inputs and identical bits on rounded inputs, and checks the
covariance exactly on samples translated by `2^53` and on samples near `2^1023`,
`2^511`, and `2^-600`, where the differences, the sums of squares, or the
squares leave the range of doubles. CI runs both tests with GCC and Clang on
Linux, Clang on macOS, and MSVC on Windows.
