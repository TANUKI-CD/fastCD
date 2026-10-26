# Conventions and results

This page describes the conventions used to represent Cayley-Dickson
elements in fastCD and the mathematical results behind the implementation.
The [paper (v2)](https://arxiv.org/abs/2609.11588v2) derives the quasilinear
recursion over a general coefficient ring and gives the arithmetic schedule
implemented here, including the fused local map, its addition-count lower
bound, and the 243-operation complex octonion leaf. This page describes that
schedule as implemented for the standard real tower, using the notation of
the paper.

## Algebra and coordinates

fastCD uses the standard real Cayley-Dickson tower, defined by

```text
A_0 = R,     A_n = A_(n-1) x A_(n-1)   (ordered pairs)
conjugate((a, b)) = (conjugate(a), -b)
(a, b)(c, d) = (ac - conjugate(d) b,  d a + b conjugate(c)).
```

An element of `A_n` is stored as a contiguous array of `N = 2^n` `double`
coefficients in the recursive standard basis `e_0, ..., e_(N-1)`, where
`e_0 = 1`. When `N = 2M`, the first `M` coordinates hold `a` and the last
`M` hold `b`. The real part is therefore `Re(x) = x[0]`, and conjugation
gives `conjugate(x) = (x[0], -x[1], ..., -x[N-1])`. The imaginary part is
obtained by setting coordinate zero to zero.

| Dimension | Algebra | Laws that still hold |
| ---: | --- | --- |
| 1 | Reals | Commutative, associative |
| 2 | Complex numbers | Commutative, associative |
| 4 | Quaternions | Associative |
| 8 | Octonions | Alternative |
| 16 and above | Sedenions and higher | Flexible and power-associative; zero divisors exist |

The literature uses several different doubling formulas and orientations.
If your coefficient arrays use another convention, convert them with a
signed permutation of their coordinates before using them with fastCD.

## Basis products

The product of two basis elements is another basis element with a sign:

```text
e_i e_j = sigma_N(i, j) e_(i XOR j),     sigma_N(i, j) in {+1, -1}.
```

The index is the bitwise XOR of the two input indices, while the sign is
determined by the doubling formula. For `i, j < M` and `N = 2M`, this gives

```text
sigma_N(i,     j)     =  sigma_M(i, j)
sigma_N(i,     M + j) =  sigma_M(j, i)
sigma_N(M + i, j)     =  epsilon(j) sigma_M(i, j)
sigma_N(M + i, M + j) = -epsilon(j) sigma_M(j, i),
```

with `sigma_1(0, 0) = 1`, `epsilon(0) = 1`, and `epsilon(j) = -1` for
`j > 0`. The function `fastcd_basis_product` evaluates this recurrence in
`O(log N)` integer steps. The following products show the orientation used
by fastCD:

```text
e_i e_0 = e_0 e_i = e_i,   e_i^2 = -1 (i > 0)
quaternions:  e_1 e_2 = e_3,   e_2 e_3 = e_1,   e_3 e_1 = e_2
octonions:    e_1 e_2 = e_3,   e_1 e_4 = e_5,   e_2 e_4 = e_6,   e_3 e_4 = e_7
```

For distinct nonzero indices, `e_j e_i = -e_i e_j`. Equal indices and
products with `e_0` follow the identities above.

## Derived operations

Writing `x = alpha + u` and `y = beta + v`, where `u` and `v` are imaginary,
the library evaluates the following operations:

```text
u x v          = Im(uv) = (uv - vu) / 2                 (imaginary cross product)
[x, y]         = xy - yx = 2 (u x v)
{x, y}         = xy + yx = 2 (alpha beta - <u, v>) + 2 (alpha v + beta u)
x^2            = (alpha^2 - <u, u>) + 2 alpha u
A(x, y, z)     = (xy) z - x (yz)
[L_a, L_b](x)  = a (b x) - b (a x)
[R_a, R_b](x)  = (x b) a - (x a) b
x^(-1)         = conjugate(x) / <x, x>,   x x^(-1) = x^(-1) x = 1.
```

Here `<x, y> = sum_i x[i] y[i]` is the Euclidean inner product, and
`Q(x) = <x, x>` is the squared norm. The reciprocal identity assumes
`x != 0`. The parentheses in these formulas matter. From dimension 16
onward, the algebra is neither associative nor
alternative, so `x^(-1)(xy)` need not equal `y`, and multiplication by
`ab - ba` need not give either operator commutator.

## Multiplication algorithms

All three algorithms compute the same product, but use different numbers of
arithmetic operations and can round differently. Write `M` for input-dependent
real multiplications, `A` for additions and subtractions, and `E` for
multiplication by fixed scalars other than `0, +1, -1`. The total is
`T = M + A + E`, including preprocessing of both operands. Known-zero terms
are omitted, sums are initialized from their first term, and coordinate
extraction, copies, permutations, and sign changes are free in this
arithmetic model. Here `N = 2^n`.

| Algorithm | `M(N)` | `A(N)` | `E(N)` |
| --- | --- | --- | --- |
| Direct | `N^2` | `N (N - 1)` | `0` |
| Cariow–Cariowa (`4 <= N <= 1024`) | `N (N - 1) / 2 + 2` | `3 N n + (N (N - 3) + 4) / 2` | `2N` |
| Quasilinear (`N >= 16`) | `(36n - 69) N / 8 + 10` | `(76n - 196) N / 8 + 26` | `(16n - 53) N / 16 + 4` |

The quasilinear counts `(M, A, E)` at `N = 1, 2, 4, 8` are respectively
`(1, 0, 0)`, `(4, 2, 0)`, `(16, 12, 0)`, and `(58, 62, 0)`. At `N = 2` the
Cariow–Cariowa schedule uses `E = 3` rather than `2N`, because one
correction coordinate is zero.
For `N >= 16` the quasilinear total is `(240n - 583) N / 16 + 40`. At
dimension 16, this gives 417 operations, compared with 496 for direct
multiplication and 452 for uniform Cariow–Cariowa. Quasilinear uses fewer
input-dependent multiplications than direct multiplication from dimension 8
onward and than Cariow–Cariowa from dimension 32 onward. Its total equals
direct multiplication at dimensions 2, 4, and 8, is smaller from dimension
16 onward, and is smaller than Cariow–Cariowa at every dimension through
1024; the advantage over Cariow–Cariowa from dimension 16 onward also holds
when fixed-scalar costs are omitted, since `M + A = 420 > 417` for
Cariow–Cariowa at dimension 16. Direct and Cariow–Cariowa remain quadratic;
quasilinear uses `Theta(N log N)` arithmetic. These are schedule counts, not
compiler-instruction counts or timing predictions. They use the default leaf
cutoffs and the uniform Cariow–Cariowa schedule (`FASTCD_CC_DOUBLING=0`).
`python3 tools/check_quasilinear_counts.py` checks them through dimension 1024
by instrumenting the actual scalar template and generated kernels, reporting
`E` under the JSON key `S`; this optional development check requires a C++
compiler, while the library itself is C11.

The resulting quasilinear schedule counts are:

| N | M | A | E | T |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 0 | 0 | 1 |
| 2 | 4 | 2 | 0 | 6 |
| 4 | 16 | 12 | 0 | 28 |
| 8 | 58 | 62 | 0 | 120 |
| 16 | 160 | 242 | 15 | 417 |
| 32 | 454 | 762 | 58 | 1274 |
| 64 | 1186 | 2106 | 176 | 3468 |
| 128 | 2938 | 5402 | 476 | 8816 |
| 256 | 7018 | 13210 | 1204 | 21432 |
| 512 | 16330 | 31258 | 2916 | 50504 |
| 1024 | 37258 | 72218 | 6852 | 116328 |

For `x = alpha + X`, `y = beta + Y` with imaginary `X, Y`, quasilinear
multiplication reconstructs

```text
xy = (alpha beta - <X,Y>) + alpha Y + beta X + X cross Y.
```

Above its base cases, the complex alternating kernel splits
`X = (u, gamma + s)` and `Y = (v, delta + t)`. It evaluates two half-size children on
`u +/- iota s` and `v +/- iota t`, over one fixed coefficient field with
`iota^2 = -1`. At the real-input root those children are complex conjugates,
so one child suffices. This auxiliary `iota` is a coefficient unit, and is
distinct from the Cayley–Dickson basis units and involution.

The local bilinear map is evaluated as a whole. For the standard tower, the
paper's `mu_n (delta s - gamma t)` is `gamma t - delta s`, and complex inputs
`X = a_0 + iota a_1` and `Y = b_0 + iota b_1` use three real evaluations:

```text
B((u,gamma,s), (v,delta,t)) = (gamma t - delta s,
                              <s,v> - <t,u>, delta u - gamma v)
Delta = B(a_0, b_0)
Omega = B(a_1, b_1)
W     = B(a_0 + a_1, b_0 + b_1)
B_complex(X,Y) = (Delta - Omega) + iota ((W - Delta) - Omega).
```

If `(F, lambda, G) = B_complex(X,Y)` and the child outputs are `Z_plus`
and `Z_minus`, the output is

```text
C_k(X,Y) = ((Z_plus + Z_minus)/2 + F,
            lambda + G + iota (Z_plus - Z_minus)/2).
```

Here `lambda` occupies the scalar coordinate of the second half. At a real
root, evaluate `Z = C_(n-1)(u + iota s, v + iota t)` once and return
`(Re(Z) + F, lambda + G - Im(Z))`, with the local map evaluated over the reals.

The three scalar reductions are accumulated separately before recovering the
complex pairing. At a complex node of dimension `K = 2^k`, with `m = K/2 - 1`,
this local map costs `18m` real products and `22m + 2` additions. Write
`M_k`, `P_k`, and `H_k` for the products, additions, and halvings of one
complex-kernel evaluation of dimension `K`, including its recursive calls.
Including input transforms and child recombination, these satisfy
`M_k = 2M_(k-1) + 18m`, `P_k = 2P_(k-1) + 38m + 2`, and
`H_k = 2H_(k-1) + 4m` above the octonion base.

The `22m + 2` addition count is optimal for this prescribed
`18m`-product decomposition with `m >= 1`:
three real evaluations of `B`, using its six coordinate products per index.
The incoming coordinates and elementary products are fixed, and the products
are treated as independent inputs to the linear output program. Even when
fixed real scalar multiplications are free, the two linear input programs
and the linear output program together need at least `22m + 2` binary
additions and subtractions.

To see the lower bound, write `q = 2m + 1`. The input programs must form
`q` real-plus-imaginary coordinate sums for each operand. These are `2q`
distinct nontrivial linear forms, so they require at least `2q` additions.
The output program maps `18m` independent products to `2q` real output
coordinates. Its transpose requires, for each real/imaginary output pair
`(r, s)`, the two forms `r - s` and `r + s`, up to signs, together with
copies of `s`. Across the `q` pairs these give `2q` distinct nontrivial
directions, each requiring an addition even with free fixed scalings.
This output map has no zero rows or columns, so the transposition principle
of Kaminski, Kirkpatrick, and Bshouty (1988) gives an addition-count upper
bound of `L + 2q - 18m` for its transpose, where `L` is its original addition
count and fixed scalings remain free. Consequently `L >= 18m`.
Adding the input bound gives `2q + 18m = 22m + 2`, attained by the
implemented schedule. This argument concerns only the fixed local-map
decomposition; it does not establish global optimality of the octonion
leaf or full Cayley-Dickson multiplication.

The dimension-8 complex base computes the alternating product on seven
complex imaginary-octonion coordinates. For
`X = (x1,x2,x3,h,y1,y2,y3)`, form

```text
p = -iota h, z = x1 + p
b = x2 + iota y2, r = x2 - iota y2
c = x3 + iota y3, s = x3 - iota y3
L1 X = (x1, y1, p)              L2 X = (b+s, r-c, p)
L3 X = (z+iota y1, b, c)        L4 X = (z-iota y1, r, s).
```

With the ordinary three-dimensional cross product and ordinary transposes
(without complex conjugation), the result is
`-iota L1^T(L1 X cross L1 Y) + (1/2) sum_(j=2..4) Lj^T(Lj X cross Lj Y)`.
For each pair of transformed operands `u = u_0 + iota u_1` and
`v = v_0 + iota v_1`, form three real crosses:

```text
Delta_j = u_0 cross v_0
Omega_j = u_1 cross v_1
W_j     = (u_0 + u_1) cross (v_0 + v_1)
L_j X cross L_j Y = (Delta_j - Omega_j) + iota ((W_j - Delta_j) - Omega_j).
```

Write the complex cross outputs as `(Xi_j, Psi_j, Gamma_j)` and form

```text
Theta = Xi_3 + Xi_4,  U = Xi_3 - Xi_4
d1 = Xi_2 + Psi_3,    d2 = Psi_2 + Psi_4
e1 = Gamma_3 - Psi_2, e2 = Xi_2 + Gamma_4.
```

The six output coordinates other than `h_hat` are

```text
x1_hat = Theta/2 - iota Xi_1    y1_hat = iota (U/2 - Psi_1)
x2_hat = (d1 + d2)/2            y2_hat = iota (d1 - d2)/2
x3_hat = (e1 + e2)/2            y3_hat = iota (e1 - e2)/2.
```

Only ten complex cross-output coordinates are reconstructed separately:
`Gamma_1` and `Gamma_2` are needed only for
`h_hat = -iota (Gamma_2 + Theta)/2 - Gamma_1`.
Instead of forming them, retain the third coordinates `p_j, eta_j, omega_j`
of `Delta_j, Omega_j, W_j` for `j = 1, 2`. Reuse `theta_R = Re_C(Theta/2)`
and `theta_I = Im_C(Theta/2)` from `x1_hat`, and compute

```text
e = p1 + eta2/2
f = eta1 - p2/2
Re_C h_hat = ((f - e) + omega2/2) + theta_I
Im_C h_hat = ((e + f) - omega1) - theta_R.
```

Both input maps cost 36 real additions. The twelve real cross products cost
72 products and 36 additions; their real-plus-imaginary inputs cost another
22 additions, sharing the sum for `p` between `L1` and `L2` in each operand.
Reconstructing ten complex coordinates costs 30 more additions. The final
sparse reconstruction uses 32 additions and 15 halvings:

| Phase | M | A | E |
| --- | ---: | ---: | ---: |
| Input linear maps | 0 | 36 | 0 |
| Four complex cross products | 72 | 88 | 0 |
| Output reconstruction | 0 | 32 | 15 |
| Total | 72 | 156 | 15 |

The complex leaf therefore uses **243 operations**, one fewer than the
previous `(72, 158, 14)` schedule: two additions are saved at the cost of
one extra halving. It first enters a full real product at dimension 16.
The smaller complex dimension-4 base still uses the fused local map with
counts `(18, 24, 0)`.

With `(M_3, P_3, H_3) = (72, 156, 15)`, the complex-kernel recurrences give,
for `K = 2^k` and `k >= 3`,

```text
M_k = (36k - 81) K/4 + 18
P_k = (76k - 168) K/4 + 36
H_k = (16k - 37) K/8 + 4.
```

The real-input root and full scalar-imaginary reconstruction add `6N - 8`
products and `6N - 10` additions to one complex kernel of dimension `N/2`,
with no halvings, so `M = M_(n-1) + 6N - 8`, `A = P_(n-1) + 6N - 10`, and
`E = H_(n-1)`, yielding the full-product counts above. There are `N/16`
complex octonion leaves in a full real product for `N >= 16`, so the new
leaf saves `N/16` total operations compared with the previous schedule.

Cariow–Cariowa is available through dimension 1024. Through dimension 64 it
uses generated straight-line kernels; above that its irregular sign corrections
use table-driven gathered loads. Fewer arithmetic operations do not necessarily
make that access pattern faster (see [benchmark.md](benchmark.md)).

Workspace requirements are `O(N)` doubles, at most the order of one operand, and
independent of the batch size: a batch reuses one workspace for all of its
products, and the paired batch kernels use fixed internal storage instead.
With the default leaf cutoffs, the direct recursion and the scalar
quasilinear recursion keep one level of half-dimensional temporaries
per recursion level above the straight-line
leaves, `2N - 32` doubles in total. The single-product quasilinear SIMD
path holds the transformed input planes and the outputs of both children of
the top complex node in pair layout, `3N` doubles, plus one alignment word
and the paired child's scratch of `2N - 64`, for `5N - 63` doubles.
Cariow–Cariowa needs no workspace through `N = 64`, where its kernels are
generated straight-line code with the sign pattern built in; from `N = 128`
the same operations run as loops over generated sign tables, with the
correction terms of each output summed in four interleaved partial sums
instead of one sequential chain, and keep the two transformed operands in
`2N` doubles. The associator and the operator
commutators add `2N` doubles for two intermediate elements. These sizes
describe the current implementation; use the workspace queries to obtain
them in your code.

The scalar and paired SIMD kernels use the same generated leaves and
recursive arithmetic. Each SIMD lane holds an independent product, so
batching does not change the number of operations per product. A single
quasilinear product above dimension 16 places the two children of the
top complex node in the two lanes, and continues through the paired kernel.
It preserves the operations and their order per coordinate, so finite results
agree bitwise with the scalar and batched forms; NaN payloads may differ.
The [C API reference](c-api.md)
describes the internal packing storage and the cases where batches use scalar
kernels.

The default algorithms used by `fastcd_mul_f64` and ordinary Python
multiplication have been selected for each dimension based on the benchmark.
Single products and batches of two or more products have separate choices
because the paired SIMD kernels change the ranking below dimension 32.
When fastCD is built without SIMD pairing, batches use the single-product
choices.

The current tables select direct for single products at `N = 8, 16` and
quasilinear elsewhere. For batches, they select direct at `N = 8`,
Cariow–Cariowa at `N = 16`, and quasilinear elsewhere. Both therefore use
quasilinear from `N = 32` upward. You can inspect the choices with
`fastcd_default_algorithm` and `fastcd_default_batch_algorithm`. To check
them on your CPU, the benchmark described in the README times all three
algorithms in both modes on the same input corpus.

## Floating-point statements

Coefficients are IEEE binary64 `double` values. Multiplication evaluates a
sequence of floating-point operations, so the complete product need not be
correctly rounded, and the three algorithms can give different rounded
results. The benchmark compares every coordinate with an independent
float64 implementation of the defining recursion, using

```text
|computed - oracle| <= 128 * DBL_EPSILON * N * ||x||_2 * ||y||_2
```

as its acceptance tolerance. This measures the disagreement with a
floating-point oracle; it does not measure the error against an exact
product or establish a universal error bound.

Intermediate quantities such as transformed sums, dot products, and
half-sums can overflow or lose precision even when the final result would
not, so operands should be scaled away from the exponent limits. Norms,
normalization, reciprocals, and squares use scaled evaluation to avoid
needless intermediate overflow. See the [C API reference](c-api.md) for how
each function handles NaN, infinities, and signed zeros.
