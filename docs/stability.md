# Numerical stability

## Summary

In the quasilinear product, every floating-point sum whose length grows with
the dimension adds its terms in blocks of eight and combines the block totals
with a balanced tree. Above dimension 256, the Cariow–Cariowa correction sums
combine their block totals the same way. The number of additions that any one
term passes through on its way to the total then grows logarithmically with
the length of the sum instead of linearly.

A study measured the effect on 13,584 cases against an exact reference. It
compared three methods on the same stored operands: quasilinear
multiplication with sequential accumulation (S), quasilinear multiplication
with balanced accumulation (B), and direct multiplication (D). Errors are in
units of `u ||x|| ||y||`, where `u = 2^-53` is the unit roundoff of binary64
and the norms are Euclidean (see [Error measures](#error-measures)). The
results describe the measured corpus; they are not bounds.

- On dense inputs (Gaussian and uniform coordinates), the median error of B
  is within a factor of two of that of D at every tested dimension.
- The largest error of B anywhere in the corpus is 3.7 over R and 4.1 over
  C. The largest errors of D are 2.9 and 2.2, and those of S are 78 and 18.
- From dimension 64 up, balanced accumulation makes real products at most
  6% slower with the default SIMD kernels and 9–13% slower without them.
  Complex products are 5–11% slower, about as much as the timing noise
  between the two builds.

## Balanced accumulation

### Which sums are balanced

The quasilinear product, described in
[mathematics.md](mathematics.md#how-quasilinear-multiplication-works),
contains four kinds of sum whose length grows with the dimension:

| Sum | Terms | Code |
|:---|---:|:---|
| The scalar output `<s, v> - <t, u>` of the nonrecursive map, accumulated separately in each of the three real evaluations `Delta`, `Omega` and `W` in one call to `C_k` | `K/2 - 1` | `complex_cross_node` in `src/quasilinear_impl.h` and its SIMD counterpart |
| The scalar output `lambda = <s, v> - <t, u>` of the real nonrecursive map at the one-call root | `N/2 - 1` | `ql_cross_recursive` |
| The pairing `<X, Y>` in `xy = (alpha beta - <X, Y>) + alpha Y + beta X + X cross Y` | `N - 1` | `imaginary_dot` |
| Both parts of the complex-coefficient pairing `sum_i x[i] y[i]` | `N - 1` | `dot_range` in `src/complex.c` |

Here `K = 2^k` is the dimension of the call to `C_k`, and each term of the
first two sums is one difference `s_i v_i - t_i u_i`. Over C, quadratic
norms, squares, anticommutators and reciprocals use the complex-coefficient
pairing, so they are balanced as well. Over R, `fastcd_inner_f64`, the norm
functions, `fastcd_square_f64` and `fastcd_anticommutator_f64` use
sequential sums. The [octonion leaf](mathematics.md#the-octonion-leaf) and
the direct kernels contain no floating-point sums of growing length, and
finite-field arithmetic is exact, so none of them is affected.

### The summation tree

`src/balanced_sum.h` adds the terms of a sum left to right in blocks of
eight. Each finished block total is combined with the earlier totals like a
binary counter: whenever two totals cover the same number of blocks, they
are added. When the terms run out, the remaining totals are added from the
smallest up, together with the last, partial block. A sum of `m` terms uses
`m - 1` additions, as a sequential sum does. Its rounding depth, the largest
number of additions that any one term passes through on its way to the
total, is at most `8 + ceil(log2(ceil(m / 8)))`, against `m - 1` for a
sequential sum. An empty sum is zero and a single term is returned as is,
so signed zeros are preserved.

The generated leaves in `src/generated/` unroll the same tree. The scalar,
two-lane SIMD, single-product and batched paths add every term in the same
order, so their results agree bit for bit. The compile-time option
`FASTCD_SUM_BLOCK`, used to time other block sizes (see [Cost](#cost)), sets
the block size to 1, 4, 8 or 16; the generated leaves always use 8.

### Cariow–Cariowa correction sums

Cariow–Cariowa computes each output coordinate as an entry of a
Walsh–Hadamard convolution minus twice a correction sum of at most
`N/2 - 1` selected products. From dimension 128 on, where the table-driven kernel
runs, a correction sum is divided into blocks of 256 indices. Within a
block, the terms go to four interleaved partial sums, which are combined
pairwise at the end of the block. Through dimension 256 a correction sum
fits in one block. Above 256, a balanced tree combines the block totals, so
the rounding depth of a correction sum is a constant plus `log2(N / 256)`
instead of growing in proportion to `N`. In the sequential build every
correction sum is a single block, whose four partial sums run over all of
its terms.

### Effect on results

Balanced accumulation changes only the order of additions, so the operation
counts and workspace sizes in [mathematics.md](mathematics.md) apply to both
kinds of accumulation. Rounding can differ: quasilinear results from
dimension 16 up, complex-coefficient results, and Cariow–Cariowa results at
dimensions 512 and 1024 can round differently from sequential accumulation.
Direct and finite-field results are identical under both.

## How accuracy was measured

### Exact reference

`benchmarks/stability.py` computes every reference product exactly, without
using fastCD arithmetic. It derives the basis signs from the doubling
formula in integer arithmetic and checks them on all 341 basis pairs through
dimension 16 against a separate rational implementation of the doubling
formula. It converts each binary64 coefficient to an integer times a power of
two, sums the signed-XOR convolution in arbitrary-precision integers, and
evaluates norms and errors with MPFR at 256 bits, or 512 bits for selected
products as a check. Direct multiplication is measured on the same inputs
for comparison; it is not the reference. The script needs NumPy and gmpy2,
which the library does not use.

### Error measures

For a computed product `z_hat` with exact value `z`, and `u = 2^-53`,

```text
E_abs = ||z_hat - z||_2
E_rel = E_abs / ||z||_2                    (z nonzero)
eta   = E_abs / (u ||x||_2 ||y||_2)
chi   = ||x||_2 ||y||_2 / ||z||_2          (z nonzero)
```

so that `E_rel = u eta chi`. Over C, `||.||_2` is the Euclidean norm of the
complex coefficients, not the quadratic form `Q` of
[mathematics.md](mathematics.md#the-quadratic-form), which can vanish on
nonzero elements. Nonzero operands can have a zero product in real dimension
16 and above and in complex dimension 2 and above. `E_rel` and `chi` are
undefined for such products, so `eta` is the main measure in the tables.

### Corpus

A case is one operand pair at one dimension, multiplied by S, B and D. The
corpus has 13,584 cases, 6,976 real and 6,608 complex, drawn from 21 real
and 19 complex families:

- **Dense:** unit-norm Gaussian and uniform coordinates, and exactly
  representable Rademacher coordinates.
- **Sparse:** supports of one, four and `N/8` coordinates.
- **Correlated:** `y = x`, and `y = ±x + 2^-40 noise`.
- **Cancellation:** a scalar coordinate constructed to cancel, exact dyadic
  cancellation, and long sums with end terms `±2^53`.
- **Magnitude:** alternating magnitudes over exponents `[-24, 24]`.
- **Structure:** scalar-only and imaginary-only operands, and repeated and
  recursively signed blocks.
- **Zero divisors (real, dimension 16 and above):** the exact zero divisor
  `(e1 + e10)(e4 - e15) = 0`, and perturbations of it by `2^-20`, `2^-40`
  and `2^-52`.
- **Isotropic (complex):** operands with `Q(x) = 0`, exact and perturbed by
  `2^-40`.

Real dimensions run from 2 to 2048 and complex dimensions from 2 to 512,
with 512 dense cases per dimension through 32 and fewer above. Seeds derive
from `numpy.random.SeedSequence(20261105, ...)`, and every operand pair is
stored with the raw results of the study. Some constructed families produce
the same operands for different seeds, so the 13,584 cases contain 12,572
distinct operand pairs, and the case counts of those families overstate
their variety.

### Builds

The study compared two builds: the sequential build at commit `982b9da` and
the balanced build at commit `558e609`. Both were compiled with GCC 16.2.1
on Linux, with `-O3 -DNDEBUG`, `-fno-fast-math -ffp-contract=off`, warnings
as errors and the default leaf cutoffs, and run on an Intel Xeon Gold 6148.
The quasilinear, Cariow–Cariowa and direct kernels of fastCD 1.3.1 are those
of the balanced build.

## Accuracy

### Quasilinear multiplication

The tables below are generated from the archived results by
`benchmarks/stability_report.py`. The full profile is the complete study, as
opposed to the short smoke profile used to test the tooling.

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

On dense inputs the medians barely move; balanced accumulation matters most
where long sums cancel or repeat. In the long-sum cancellation family at
dimension 512, the median `eta` falls from 16.0 with sequential accumulation
to 1.08 with balanced accumulation over R, where direct multiplication gives
2.87, and from 16.0 to 1.16 over C, where it gives 1.39. For repeated blocks
over R it falls from 6.7 to 1.9.

No reordering of a sum is more accurate on every input. Pooling the dense
cases from dimension 32 up, B is more accurate than S in 66% of real and
73% of complex cases and less accurate in 27% of each. The largest
increases in `eta` are from 0.60 to 1.83, for a real dimension-16 input
with `y = -x + 2^-40 noise`, and from 0.30 to 1.45, for a complex
dimension-4 near-isotropic input. In both, only the error of the scalar
coordinate grows; the other coordinates are equally accurate with either
kind of accumulation.

The tables count a case as tied when B and S have the same stored error. Of
the 7,788 tied cases, 7,573 have exactly equal errors. The other 215 are
near-zero-divisor cases whose errors differ by less than the precision of
the stored values; by their exact squared errors, 152 favor balanced and 63
favor sequential accumulation.

Near a zero divisor the relative error can be large for every method, since
`E_rel = u eta chi` and `chi` is large there. At perturbation `2^-52` and
dimension 512, S and B both have a relative error of about 0.64, with `eta`
about 1.27 and `chi` about `4.5e15`, while D is nearly exact on these sparse
inputs.

### Cariow–Cariowa

Balanced correction sums change Cariow–Cariowa results only at dimensions
512 and 1024. Of the 6,808 real cases that Cariow–Cariowa supports, 265 are
more accurate with balanced correction sums, 23 are less accurate and 6,520
are tied, where ties in the stored errors are decided by exact squared
errors. Every dense case at 512 and 1024 is more accurate. At 1024 the
median relative error falls from `6.2e-16` to `4.3e-16` and the median
`eta` from 5.59 to 3.80, and the largest `eta` among all Cariow–Cariowa
cases falls from 183 to 80. The largest increase is for a repeated-block
input at 512, from 35.3 to 41.4. The archived `cc-accuracy-summary.json`
holds the details.

## Cost

`benchmarks/stability_speed.py` timed both builds with the native benchmark
runners on separate timing corpora of uniform random operands, on one
logical CPU, in twelve paired blocks.
In each block both builds time the same products, and which build goes
first alternates from block to block. Each observation has two warmup passes
and a 20 ms calibration target, and every timed product is verified.

In the table, generated by `benchmarks/stability_speed_report.py`, the
`simd` rows use the default build and the `scalar` rows a build without the
explicit SIMD kernels. The balanced/sequential ratio and the direct control
are medians of the twelve per-block ratios. The direct control divides the
direct time in the balanced build by the direct time in the sequential
build. The direct code is the same in both, so the control shows how much
two builds of the same code differ in time on this host.

<!-- BEGIN STABILITY SPEED PERFORMANCE -->

Paired timings on one host: 12 blocks pinned to CPU 18. Times are medians in ns per product, and brackets are paired bootstrap 95% intervals. The direct control is the ratio of direct timings between the two builds, whose direct arithmetic is identical, so it shows the noise between builds.

| domain | kernels | mode | N | sequential ns | balanced ns | balanced / sequential [95% CI] | direct ns | direct / balanced | direct control |
|:---|:---|:---|---:|---:|---:|:---|---:|---:|---:|
| real | simd | single | 16 | 117.9 | 113.7 | 0.9646 [0.9207, 1.01] | 88.96 | 0.7733 | 0.9753 |
| real | simd | batch | 16 | 54.75 | 51.96 | 0.9477 [0.9137, 0.982] | 55.87 | 1.071 | 0.982 |
| real | scalar | single | 16 | 112.2 | 105.4 | 0.938 [0.8782, 1.015] | 87.94 | 0.8258 | 1.016 |
| real | scalar | batch | 16 | 91.17 | 91.75 | 1.017 [1.002, 1.058] | 55.71 | 0.601 | 0.9719 |
| real | simd | single | 64 | 502 | 522.3 | 1.059 [1.001, 1.107] | 1154 | 2.217 | 1.005 |
| real | simd | batch | 64 | 424.6 | 446.4 | 1.035 [0.9973, 1.088] | 1124 | 2.442 | 1.06 |
| real | scalar | single | 64 | 651.2 | 709.4 | 1.098 [1.073, 1.118] | 1148 | 1.611 | 1.023 |
| real | scalar | batch | 64 | 617.8 | 677.4 | 1.09 [1.061, 1.129] | 1061 | 1.586 | 0.961 |
| real | simd | single | 256 | 2698 | 2682 | 1.01 [0.9818, 1.034] | 1.763e+04 | 6.433 | 0.982 |
| real | simd | batch | 256 | 2589 | 2673 | 1.031 [1.019, 1.054] | 1.697e+04 | 6.461 | 1.016 |
| real | scalar | single | 256 | 3401 | 3722 | 1.098 [1.062, 1.125] | 1.83e+04 | 4.929 | 0.9908 |
| real | scalar | batch | 256 | 3343 | 3722 | 1.107 [1.07, 1.152] | 1.837e+04 | 4.878 | 1.005 |
| real | simd | single | 1024 | 1.41e+04 | 1.434e+04 | 1.015 [0.9923, 1.062] | 2.771e+05 | 19.28 | 0.9774 |
| real | simd | batch | 1024 | 1.425e+04 | 1.468e+04 | 1.049 [0.9838, 1.081] | 2.795e+05 | 18.89 | 0.9905 |
| real | scalar | single | 1024 | 1.75e+04 | 1.945e+04 | 1.124 [1.09, 1.169] | 2.773e+05 | 14.31 | 0.9967 |
| real | scalar | batch | 1024 | 1.713e+04 | 1.933e+04 | 1.128 [1.091, 1.147] | 2.898e+05 | 15 | 1.011 |
| complex | simd | single | 16 | 300 | 306.3 | 1.021 [0.9805, 1.035] | 531.1 | 1.719 | 1.058 |
| complex | simd | batch | 16 | 271 | 274.6 | 1.008 [0.9867, 1.059] | 510 | 1.854 | 1.061 |
| complex | scalar | single | 16 | 302.3 | 314.5 | 1.036 [1.016, 1.056] | 544.1 | 1.736 | 1.084 |
| complex | scalar | batch | 16 | 265.8 | 283.8 | 1.04 [1.022, 1.069] | 510.9 | 1.814 | 1.063 |
| complex | simd | single | 64 | 1588 | 1676 | 1.059 [1.038, 1.1] | 8396 | 4.971 | 1.081 |
| complex | simd | batch | 64 | 1533 | 1592 | 1.053 [1.034, 1.063] | 8381 | 5.208 | 1.078 |
| complex | scalar | single | 64 | 1516 | 1639 | 1.068 [1.049, 1.082] | 8375 | 5.16 | 1.089 |
| complex | scalar | batch | 64 | 1477 | 1600 | 1.085 [1.035, 1.106] | 8287 | 5.114 | 1.086 |
| complex | simd | single | 256 | 7793 | 8690 | 1.115 [1.098, 1.122] | 1.329e+05 | 15.39 | 1.08 |
| complex | simd | batch | 256 | 7835 | 8519 | 1.111 [1.071, 1.122] | 1.334e+05 | 15.79 | 1.107 |
| complex | scalar | single | 256 | 8013 | 8623 | 1.058 [1.051, 1.094] | 1.332e+05 | 15.51 | 1.08 |
| complex | scalar | batch | 256 | 7810 | 8492 | 1.099 [1.085, 1.12] | 1.306e+05 | 15.47 | 1.059 |

<!-- END STABILITY SPEED PERFORMANCE -->

For real products the direct control lies between 0.96 and 1.06, so
differences of a few percent are noise between builds. From dimension 64
up, balanced accumulation costs at most 6% on the default build, within
that range, and a consistent 9–13% on builds without the SIMD kernels,
outside it. For complex products the cost, 5–11% from dimension 64 up, is
about the size of the complex control, 1.06 to 1.11, so this study cannot
separate it from the difference between builds. Even with balanced
accumulation, quasilinear multiplication is about 19 times faster than
direct multiplication at dimension 1024.

Cariow–Cariowa was timed the same way, with balanced correction sums against
the single-block correction sums of the sequential build; the archived
`cc-speed-tables.md` holds the table. Its direct control lies between 0.95
and 1.05. Only two ratios exceed that range: batches at dimension 512 on the
default build, which are 11% slower with balanced correction sums, and
single products at 512 without the SIMD kernels, which are 8% slower. The
other ratios at 512 and 1024 lie between 1.00 and 1.05.

Blocks of 4, 8 and 16 were timed with the same runners
(`benchmarks/stability_block_size.py`; archived in `block-size-tables.md`).
Blocks of 4 were up to 9% slower than blocks of 8 and never measurably
faster. Blocks of 16 were up to 7% faster on some real products but up to
3% slower on others and 1–7% slower on complex products. The default is 8.

## Tests and checks

- `tests/test_stability.c` (CTest `fastcd.stability`) checks the summation
  tree against an independent recursive description for every length from 0
  to 2051, including signed zeros and the helpers that accumulate two or
  three sums together. It then checks real and complex single products and
  odd-length strided batches at dimensions 2 through 512 against a
  signed-XOR oracle on exact dyadic inputs, with workspace canaries, minimum
  and short workspaces, and forbidden aliasing. CI runs it with GCC and
  Clang on Linux, Clang on macOS and MSVC on Windows.
- `tools/check_quasilinear_counts.py` confirms the operation counts in
  [mathematics.md](mathematics.md#operation-counts) through dimension 1024.
- The study compared the scalar, SIMD, single-product and batched forms of
  quasilinear multiplication bit for bit in all 13,584 cases (40,752
  comparisons), and the same four forms of Cariow–Cariowa in all 6,808 real
  cases through dimension 1024 (20,424 comparisons), with no mismatches. It
  also checked that the two builds give bit-for-bit identical direct
  products in every case.

## Limitations

- The study is empirical. Its medians and maxima describe this corpus, not
  all inputs, and they do not show that quasilinear multiplication is as
  accurate as direct multiplication on every input. Nothing was measured
  above real dimension 2048 or complex dimension 512, or with overflow,
  underflow, subnormals, infinities, NaNs or non-default rounding modes.
- Composite operations (associators, operator commutators, reciprocals)
  multiply already-rounded intermediates and were not measured; the product
  results above do not describe them.
- Accuracy and timings were measured only with GCC on Linux. Other compilers
  run the regression test in CI but were not part of the study.
- Bitwise agreement across versions or across compilers is not promised.

## Archives and reproduction

The study is in
[`benchmark-results/stability-20261107/`](../benchmark-results/stability-20261107/README.md),
whose README lists every file. Its published files are in the repository and
the source distributions: summaries, tables, charts, build records, and
`key-figures.json`, which holds the key figures quoted on this page. Raw
per-case results, stored operands, timing observations and logs are under
`raw/`; they stay on the machine that ran the study and are not distributed.
`SHA256SUMS` lists both.

To check the study, run from the repository root:

```sh
python3 tools/check_stability_archives.py
```

The checker uses only the Python standard library. It compares every
published file with `SHA256SUMS`, checks that the tables on this page are the
tables of the study, and, when the git history is available, checks that the
measured commit contains exactly the source files the study recorded. With
`--full` it also requires the raw files, checks their hashes and regenerates
the tables from the raw results; this mode needs NumPy.

To repeat the study, run `benchmarks/stability_study.py` with a Python that
has NumPy, gmpy2 and Matplotlib:

```sh
python3 benchmarks/stability_study.py --baseline-commit 982b9da --commit 558e609 \
    --output benchmark-results/stability-YYYYMMDD --cpu CPU --accuracy-cpu CPU
```

Commit `982b9da` has sequential accumulation and `558e609` is the measured
commit. The script builds both from clean worktrees, pins the timings and the
accuracy study to the given logical CPUs, runs every step, and writes the
README, the tables and `SHA256SUMS`; `--document docs/stability.md` also
rewrites the tables on this page. Running the same command again resumes an
interrupted study and skips the finished steps.

Operands come from the recorded seeds through NumPy's `Generator`, whose
distribution methods are not guaranteed to be bit-identical across NumPy
versions. On another NumPy version the random families, and with them the
dense rows of the tables, can differ slightly. The structured families and
the exact reference are not affected. The raw results record a digest of all
measured operands, against which a rerun can be checked where those files
are available.
