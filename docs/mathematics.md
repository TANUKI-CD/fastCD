# Conventions and results

This page describes the conventions used to represent Cayley-Dickson
elements in fastCD and the mathematical results behind the implementation.
The accompanying paper gives the derivations and proofs.

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

Reversing the factors changes the sign in every nontrivial case.

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
`Q(x) = <x, x>` is the squared norm. The parentheses in these formulas
matter. From dimension 16 onward, the algebra is neither associative nor
alternative, so `x^(-1)(xy)` need not equal `y`, and multiplication by
`ab - ba` need not give either operator commutator.

## Multiplication algorithms

All three algorithms compute the same product, but use different numbers of
arithmetic operations and can round differently. Write `M` for input-dependent
real multiplications, `A` for additions and subtractions, and `S` for
multiplication by fixed scalars other than `0, +1, -1`. The total is
`T = M + A + S`, including preprocessing of both operands. Known-zero terms,
copies, permutations, and sign changes are free in this arithmetic model.
Here `N = 2^n`.

| Algorithm | `M(N)` | `A(N)` | `S(N)` |
| --- | --- | --- | --- |
| Direct | `N^2` | `N (N - 1)` | `0` |
| Cariow–Cariowa (`4 <= N <= 1024`) | `N (N - 1) / 2 + 2` | `3 N n + (N (N - 3) + 4) / 2` | `2N` |
| Quasilinear (`N >= 16`) | `(36n - 69) N / 8 + 10` | `(76n - 195) N / 8 + 26` | `(8n - 27) N / 8 + 4` |

The quasilinear counts `(M, A, S)` at `N = 1, 2, 4, 8` are respectively
`(1, 0, 0)`, `(4, 2, 0)`, `(16, 12, 0)`, and `(58, 62, 0)`.
For `N >= 16` its total is `(120n - 291) N / 8 + 40`. At dimension 16,
this gives 418 operations, compared with 496 for direct multiplication and
452 for uniform Cariow–Cariowa. Quasilinear has fewer total operations than
both from dimension 16 onward, and ties direct at dimension 8. Direct and
Cariow–Cariowa remain quadratic; quasilinear uses `Theta(N log N)` arithmetic.
These are schedule counts, not compiler-instruction counts or timing predictions.
`python3 tools/check_quasilinear_counts.py` checks them through dimension 256
by instrumenting the actual scalar template and generated kernels; this optional
development check requires a C++ compiler, while the library itself is C11.

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

The local bilinear map is evaluated as a whole:

```text
B((u,gamma,s), (v,delta,t)) = (gamma t - delta s,
                              <s,v> - <t,u>, delta u - gamma v)
P = B(a_real, b_real)
Q = B(a_imag, b_imag)
W = B(a_real + a_imag, b_real + b_imag)
B_complex(a,b) = (P - Q) + iota ((W - P) - Q).
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
this local map costs `18m` real products and `22m + 2` additions. Including
input transforms and child recombination, the complex-kernel counts satisfy
`M_k = 2M_(k-1) + 18m`, `A_k = 2A_(k-1) + 38m + 2`, and
`S_k = 2S_(k-1) + 4m` above the octonion base.

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
Each complex three-dimensional cross uses three real crosses. The sparse
linear maps and transpose reconstruction give `(M, A, S) = (72, 158, 14)`,
or 244 operations. This complex leaf first enters a full real product at
dimension 16. The smaller complex dimension-4 base uses the fused local map,
with counts `(18, 24, 0)`.

Cariow–Cariowa is available through dimension 1024. Through dimension 64 it
uses generated straight-line kernels; above that its irregular sign corrections
use table-driven gathered loads. Fewer arithmetic operations do not necessarily
make that access pattern faster (see [benchmark.md](benchmark.md)).

Every workspace is `Theta(N)` doubles, the same order as one operand, and
independent of the batch size: a batch reuses one workspace for all of its
products, and the paired batch kernels use fixed internal storage instead.
The direct recursion and the scalar quasilinear recursion keep one level of
half-dimensional temporaries per recursion level above the straight-line
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
It preserves the operations and their order per coordinate, so it agrees
bitwise with the scalar and batched forms. The [C API reference](c-api.md)
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

Coefficients are IEEE binary64 `double` values, and each arithmetic
operation is correctly rounded. This does not make the complete product
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
