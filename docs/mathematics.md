# Conventions and results

This page fixes the conventions fastCD uses for Cayley–Dickson elements and
summarizes the mathematics behind its multiplication algorithms. The
[paper](https://arxiv.org/abs/2609.11588) derives the quasilinear recursion
over a general coefficient ring. The
[technical supplement](quasilinear-supplement-fastCD.pdf) specializes it to
the standard real tower and proves the results quoted below for the
nonrecursive map, the complex octonion base case and the exact operation
counts. Operation counts on this page are for real coefficients; the complex
and finite-field kernels use different coefficient arithmetic and are not
counted.

## The algebra

fastCD works in the standard Cayley–Dickson tower, in which every doubling
parameter is `-1`, over a coefficient field F: the real numbers R, the
complex numbers C, or a finite field F_(p^d) with p an odd prime. Starting
from `A_0 = F`, each level consists of ordered pairs from the level below:

```text
A_n = A_(n-1) x A_(n-1)
conjugate((a, b)) = (conjugate(a), -b)
(a, b)(c, d) = (ac - conjugate(d) b,  d a + b conjugate(c))
```

Conjugation is the identity on `A_0 = F`, so it fixes every coefficient:
over C it does not conjugate complex coefficients, and over F_(p^d) it does
not apply the Frobenius map.

An element of `A_n` has `N = 2^n` coordinates in F with respect to the
standard basis `e_0 = 1, e_1, ..., e_(N-1)`. The basis follows the pairs:
when `N = 2M`, the first `M` coordinates are those of `a` and the last `M`
are those of `b`. The scalar part of `x` is therefore `Re(x) = x[0]`, its
imaginary part `Im(x)` is `x` with coordinate 0 set to zero, and

```text
conjugate(x) = (x[0], -x[1], ..., -x[N-1]).
```

The names in the following table are those of the real algebras. Over C and
finite fields the same identities hold, but statements about division and
positive-definiteness can fail (see [The quadratic form](#the-quadratic-form)).

| Dimension | Real algebra | Identities |
| ---: | --- | --- |
| 1 | Real numbers | Commutative, associative |
| 2 | Complex numbers | Commutative, associative |
| 4 | Quaternions | Associative |
| 8 | Octonions | Alternative |
| 16 and above | Sedenions and higher | Flexible and power-associative; zero divisors exist |

The literature uses several doubling formulas and orientations. Coefficient
arrays written in another convention can be converted to fastCD's with a
signed permutation of their coordinates.

### Coefficient storage

Real coefficients are binary64 `double` values, and a complex coefficient is
a pair of doubles, real part first. An element of F_(p^d) is a polynomial
`a_0 + a_1 t + ... + a_(d-1) t^(d-1)` with coefficients in `0, ..., p-1`,
reduced modulo the monic irreducible polynomial `FiniteField.modulus`. The C
kernels store each coefficient as a canonical `uint32` residue.

In Python, `char` and `deg` select F: `(0, 1)` is R (`float64`), `(0, 2)` is
C (`complex128`), and `(p, d)` is F_(p^d). One element has shape `(N,)` over
R, C and prime fields, and `(N, d)` over extension fields. The trailing axis
holds the polynomial coefficients and is not part of the Cayley–Dickson
dimension.

## Basis products

The product of two basis elements is a signed basis element whose index is
the bitwise XOR of the two indices:

```text
e_i e_j = sigma_N(i, j) e_(i XOR j),     sigma_N(i, j) in {+1, -1}.
```

The signs follow from the doubling formula. For `N = 2M` and `i, j < M`,

```text
sigma_N(i,     j)     =  sigma_M(i, j)
sigma_N(i,     M + j) =  sigma_M(j, i)
sigma_N(M + i, j)     =  epsilon(j) sigma_M(i, j)
sigma_N(M + i, M + j) = -epsilon(j) sigma_M(j, i),
```

where `sigma_1(0, 0) = 1`, `epsilon(0) = 1`, and `epsilon(j) = -1` for
`j > 0`. The function `fastcd_basis_product` evaluates this recurrence in
`O(log N)` integer steps. It follows that `e_0` is the identity, that
`e_i^2 = -1` for `i > 0`, and that `e_j e_i = -e_i e_j` for distinct nonzero
`i` and `j`. The orientation is fixed by

```text
quaternions:  e_1 e_2 = e_3,   e_2 e_3 = e_1,   e_3 e_1 = e_2
octonions:    e_1 e_2 = e_3,   e_1 e_4 = e_5,   e_2 e_4 = e_6,   e_3 e_4 = e_7
```

## The quadratic form

Write `<x, y> = sum_i x[i] y[i]` for the coordinate pairing and
`Q(x) = <x, x>` for the quadratic form. Both are defined by the same sums
over every field; in particular, over C they involve no complex conjugation.
Over R, `Q(x)` is the squared Euclidean length.

Q is the norm form of the tower. By induction on the level, the doubling
formula gives

```text
x conjugate(x) = conjugate(x) x = Q(x) 1,
x^2 - 2 x[0] x + Q(x) 1 = 0.
```

Since the characteristic of F is never 2, the pairing is recovered from the
form by polarization, `<x, y> = (Q(x + y) - Q(x) - Q(y)) / 2`, and it is
nondegenerate. The composition law `Q(xy) = Q(x) Q(y)` holds through
dimension 8 and fails in general from dimension 16 on.

Over C and finite fields, Q can vanish on a nonzero element: for example,
`Q((1, i)) = 0` over C and `Q((1, 2)) = 0` over F_5. Such elements have no
reciprocal (see [Derived operations](#derived-operations)).

`quadratic_norm()` returns `Q(x)` over every field as a coefficient of F: a
field `Element` over finite fields (an `ElementArray` for a batch), and a
complex scalar or array over C. For a coefficient-field scalar `c` it
returns `c*c`. `norm_squared()` returns the same form; over R it uses scaled
accumulation to avoid intermediate overflow. `norm()`, `abs(x)` and
`normalized()` return the Euclidean norm `sqrt(Q(x))` and the unit vector
`x / sqrt(Q(x))`, and exist only for real coefficients: a finite field has
no ordering or positive square root, and fastCD provides no Hermitian norm
over C.

## Derived operations

Write `x = alpha + X` and `y = beta + Y`, where `alpha` and `beta` are the
scalar parts and `X` and `Y` are imaginary. The alternating product of two
imaginary elements is

```text
X cross Y = (XY - YX) / 2.
```

It is bilinear, alternating and imaginary, and it is the imaginary part of
`XY`: `XY = -<X, Y> + X cross Y`. At dimension 4 it is the ordinary
three-dimensional cross product. The library exposes it as `imag_cross`.
Every product therefore splits as

```text
xy = (alpha beta - <X, Y>) + alpha Y + beta X + X cross Y,          (1)
```

which is the starting point of the quasilinear algorithm. The library
evaluates the following derived operations:

```text
[x, y]         = xy - yx = 2 (X cross Y)
{x, y}         = xy + yx = 2 (alpha beta - <X, Y>) + 2 (alpha Y + beta X)
x^2            = (alpha^2 - <X, X>) + 2 alpha X
A(x, y, z)     = (xy) z - x (yz)
[L_a, L_b](x)  = a (b x) - b (a x)
[R_a, R_b](x)  = (x b) a - (x a) b
x^(-1)         = conjugate(x) / Q(x),     x x^(-1) = x^(-1) x = 1
```

The parentheses matter. Through dimension 4 the algebras are associative,
so `[L_a, L_b](x) = (ab - ba) x` and `[R_a, R_b](x) = -x (ab - ba)`; from
dimension 8 on, neither identity holds in general. The octonions are
alternative, so `x^(-1) (xy) = y` holds through dimension 8. From dimension
16 on it can fail, and the reciprocal identity gives no cancellation law for
products.

The reciprocal requires `Q(x) != 0`. Over R this means `x != 0`; over C and
finite fields it also excludes the nonzero elements with `Q(x) = 0`. The
reciprocal, scalar division and negative powers check the denominator and
raise `ZeroDivisionError` when it is zero, including for those elements.

## Multiplication algorithms

fastCD has three multiplication algorithms:

- **Direct** evaluates the doubling formula recursively.
- **Quasilinear** is the algorithm of the paper, described below.
- **Cariow–Cariowa** is the uniform method of
  [Cariow and Cariowa](https://doi.org/10.15199/48.2015.02.09), which works
  at every dimension. It computes the unsigned XOR convolution of the two
  coordinate vectors with Walsh–Hadamard transforms, then corrects the sign
  of each output coordinate with a sparse sum.

All three are available for real coefficients; complex and finite-field
coefficients use direct and quasilinear. The algorithms compute the same
product, but they use different numbers of operations, and in floating point
they can round differently.

### Operation counts

`M` counts multiplications of two input-dependent values, `A` counts
additions and subtractions, and `E` counts multiplications by fixed scalars
other than `0` and `±1`. The total `T = M + A + E` includes the preprocessing
of both operands. Known-zero terms are skipped, each sum starts from its
first term, and reading coordinates, copying, permuting and changing signs
are free. Throughout, `N = 2^n`.

| Algorithm | `M(N)` | `A(N)` | `E(N)` |
| --- | --- | --- | --- |
| Direct | `N^2` | `N (N - 1)` | `0` |
| Cariow–Cariowa (`4 <= N <= 1024`) | `N (N - 1) / 2 + 2` | `3 N n + (N (N - 3) + 4) / 2` | `2N` |
| Quasilinear (`N >= 16`) | `(36n - 69) N / 8 + 10` | `(76n - 196) N / 8 + 26` | `(16n - 53) N / 16 + 4` |

At `N = 2`, Cariow–Cariowa uses `(M, A, E) = (3, 7, 3)`: only the scalar
coordinate needs a sign correction, so `E` counts the two scalings by `1/N`
and a single doubling. For `N >= 16` the quasilinear total is
`T = (240n - 583) N / 16 + 40`. The exact quasilinear counts, including the
small base cases, are:

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

Comparing the three algorithms:

- Quasilinear uses fewer input-dependent multiplications than direct from
  `N = 8` on, and fewer than Cariow–Cariowa from `N = 32` on. At `N = 16`
  Cariow–Cariowa uses 122 against 160.
- The quasilinear total equals the direct total at `N = 2, 4, 8` and is
  smaller from `N = 16` on: 417 against 496 at `N = 16`.
- The quasilinear total is smaller than the Cariow–Cariowa total at every
  dimension from 2 through 1024: 417 against 452 at `N = 16`. This holds
  even when Cariow–Cariowa's fixed-scalar multiplications are not counted,
  since its `M + A` is 420 at `N = 16` and stays above the quasilinear total
  at every larger dimension.
- Direct and Cariow–Cariowa use `Theta(N^2)` operations; quasilinear uses
  `Theta(N log N)`.

These counts describe the arithmetic schedule. They are not machine
instruction counts and do not predict running time. They assume the default
leaf cutoffs and the default Cariow–Cariowa build (`FASTCD_CC_DOUBLING=0`).
`tools/check_quasilinear_counts.py` and `tools/check_cariow_counts.py`
instrument the generated kernels and check the counts through dimension
1024.

### How quasilinear multiplication works

The splitting (1) reduces multiplication to the alternating product
`X cross Y`, since its other terms cost `O(N)` operations. The algorithm
computes the alternating product recursively, with two half-size recursive
calls per level.

**Splitting the imaginary part.** The first half of an imaginary element is
imaginary, while the second half may have a nonzero scalar coordinate. An
imaginary element at level `n` therefore has the form `X = (u, gamma + s)`,
where `u` and `s` are imaginary at level `n - 1` and `gamma` is a scalar.
Write `Y = (v, delta + t)` in the same way. Each of `u, v, s, t` has
`m = N/2 - 1` coordinates. Expanding the doubling formula gives

```text
X cross Y = (D + f,  lambda + g - H),
D = u cross v - s cross t,     H = u cross t + s cross v,
```

where `(f, lambda, g)` is the value of the nonrecursive bilinear map

```text
B_n((u, gamma, s), (v, delta, t)) = (gamma t - delta s,  <s, v> - <t, u>,  delta u - gamma v),
```

and `lambda` occupies the scalar coordinate of the second half. `B_n`
collects every term that needs no recursive call, and its three outputs have
`m`, `1` and `m` coordinates. For a general doubling parameter `mu_n` the
first output is `mu_n (delta s - gamma t)`, as in the paper; here
`mu_n = -1`.

**Two recursive calls.** `D` and `H` need four alternating products at level
`n - 1`. Let `iota` be a coefficient with `iota^2 = -1`. It is a scalar, so
it commutes with everything and is fixed by conjugation; it is not one of
the basis units `e_i`. Write `C_k` for the alternating product at level `k`
with coefficients of the form `a + iota b`. Bilinearity gives
`C_(n-1)(u ± iota s, v ± iota t) = D ± iota H`, so two calls suffice:

```text
Z_+ = C_(n-1)(u + iota s, v + iota t)
Z_- = C_(n-1)(u - iota s, v - iota t)
C_n(X, Y) = ((Z_+ + Z_-)/2 + f,  lambda + g + iota (Z_+ - Z_-)/2).
```

Each level makes two calls of half the size and does `O(N)` other work, so
the recursion uses `O(N log N)` operations. At level 2 the half-size
alternating products vanish and `C_2(X, Y) = (f, lambda + g)`; at level 3 the
kernel uses the octonion leaf described below.

**The one-call root.** For real-coefficient inputs, `u, v, s, t` are real,
so `D` and `H` are real and `Z_+ = D + iota H` alone determines both. One
call therefore suffices at the top level:

```text
Z = C_(n-1)(u + iota s, v + iota t)
X cross Y = (Re_C(Z) + f,  lambda + g - Im_C(Z)),
```

where `Re_C` and `Im_C` take the parts `a` and `b` of each coefficient
`a + iota b`, and `B_n` is evaluated over the reals. Below the root the
inputs have complex coefficients, and both calls are needed.

**Evaluating the nonrecursive map.** With complex coefficients, `B_k` is
evaluated as a whole by the three-product formula for complex
multiplication. For operands `a_0 + iota a_1` and `b_0 + iota b_1` with real
`a_0, a_1, b_0, b_1`,

```text
Delta = B_k(a_0, b_0)
Omega = B_k(a_1, b_1)
W     = B_k(a_0 + a_1, b_0 + b_1)
B_k(a_0 + iota a_1, b_0 + iota b_1) = (Delta - Omega) + iota ((W - Delta) - Omega).
```

Applying the formula to the whole map, rather than to each coefficient
product inside it, forms each input sum once and reconstructs each output
coordinate once. In particular, the scalar output `<s, v> - <t, u>` is
accumulated separately in each of `Delta`, `Omega` and `W`, and the complex
value is recovered from the three sums. At level `k`, with `K = 2^k` and
`m = K/2 - 1`, this costs `18m` real products and `22m + 2` additions.

**The addition count is optimal for this evaluation.** Fix the `18m`
products above, six for each coordinate index in each of `Delta`, `Omega`
and `W`, and treat their values as independent. Then no evaluation of `B_k`
from these products uses fewer than `22m + 2` additions and subtractions,
even when multiplications by fixed scalars are free. Let `q = 2m + 1` be the
number of coordinates of each operand and of the output. Each operand must
supply its `q` real-plus-imaginary sums as factors of `W`. These are
distinct linear forms, so the two operands need at least `2q` additions. The
output map sends the `18m` products to the `2q` real and imaginary output
coordinates. Its transpose needs exactly `2q` additions, so by the
transposition principle of Kaminski, Kirkpatrick and Bshouty (1988) the map
itself needs `18m`. The total is `2q + 18m = 22m + 2`, which the evaluation
above attains. The bound concerns only this decomposition of the
nonrecursive map; it says nothing about the optimality of the octonion leaf
or of Cayley–Dickson multiplication as a whole. The
[supplement](quasilinear-supplement-fastCD.pdf) gives the full proof.

**Cost of one level.** Outside its recursive calls, one call to `C_k` with
`k >= 4` evaluates the nonrecursive map, spends `8m` additions forming
`u ± iota s` and `v ± iota t`, `4m` forming `Z_+ ± Z_-` and `4m` adding `f`
and `g`, and halves `Z_+ ± Z_-` with `4m` real halvings. Multiplication by
`iota` swaps the two parts of a coefficient and changes one sign, so it is
free, and the halvings are the only fixed-scalar operations in the kernel.
Write `M_k`, `P_k` and `H_k` for the products, additions and halvings of one
call to `C_k`, including its recursive calls; `P` and `H` stand in for `A`
and `E`, which are reserved for the full product. Then, for `k >= 4`,

```text
M_k = 2 M_(k-1) + 18m
P_k = 2 P_(k-1) + 38m + 2
H_k = 2 H_(k-1) + 4m.
```

### The octonion leaf

At level 3 the kernel evaluates the alternating product of complexified
imaginary octonions, which have seven coordinates, by a direct formula.
Write

```text
X = (x1, x2, x3, h, y1, y2, y3),
```

so that `x1, x2, x3` are the coordinates of `e_1, e_2, e_3`, `h` that of
`e_4`, and `y1, y2, y3` those of `e_5, e_6, e_7`. Define the linear forms

```text
p   = -iota h,           z   = x1 + p,
b_+ = x2 + iota y2,      b_- = x2 - iota y2,
c_+ = x3 + iota y3,      c_- = x3 - iota y3,
```

and four linear maps from seven coordinates to three:

```text
L_1 X = (x1, y1, p)                  L_2 X = (b_+ + c_-, b_- - c_+, p)
L_3 X = (z + iota y1, b_+, c_+)      L_4 X = (z - iota y1, b_-, c_-).
```

With `cross` the ordinary three-dimensional cross product and `^T` the
ordinary matrix transpose, without complex conjugation,

```text
C_3(X, Y) = -iota L_1^T (L_1 X cross L_1 Y) + (1/2) sum_(j=2..4) L_j^T (L_j X cross L_j Y).
```

Each complex cross product is evaluated by three real cross products, using
the same three-product formula as the nonrecursive map:

```text
Delta_j = Re_C(L_j X) cross Re_C(L_j Y)
Omega_j = Im_C(L_j X) cross Im_C(L_j Y)
W_j     = (Re_C + Im_C)(L_j X) cross (Re_C + Im_C)(L_j Y)
L_j X cross L_j Y = (Delta_j - Omega_j) + iota ((W_j - Delta_j) - Omega_j).
```

Write the complex results as `L_j X cross L_j Y = (Xi_j, Psi_j, Upsilon_j)`.
The transposes are sparse. With

```text
Theta_+ = Xi_3 + Xi_4,          Theta_- = Xi_3 - Xi_4,
Pi_1    = Xi_2 + Psi_3,         Pi_2    = Psi_2 + Psi_4,
Sigma_1 = Upsilon_3 - Psi_2,    Sigma_2 = Xi_2 + Upsilon_4,
```

the output coordinates are

```text
x1_hat = Theta_+/2 - iota Xi_1          y1_hat = iota (Theta_-/2 - Psi_1)
x2_hat = (Pi_1 + Pi_2)/2                y2_hat = iota (Pi_1 - Pi_2)/2
x3_hat = (Sigma_1 + Sigma_2)/2          y3_hat = iota (Sigma_1 - Sigma_2)/2
h_hat  = -iota (Upsilon_2 + Theta_+)/2 - Upsilon_1.
```

`Upsilon_1` and `Upsilon_2` appear only in `h_hat`, so they are never
reconstructed. Instead, let `p_j`, `eta_j` and `omega_j` be the third
coordinates of `Delta_j`, `Omega_j` and `W_j` for `j = 1, 2`, reuse
`theta_R = Re_C(Theta_+/2)` and `theta_I = Im_C(Theta_+/2)` from `x1_hat`,
and compute

```text
phi = p_1 + eta_2/2
chi = eta_1 - p_2/2
Re_C(h_hat) = ((chi - phi) + omega_2/2) + theta_I
Im_C(h_hat) = ((phi + chi) - omega_1) - theta_R.
```

The leaf costs:

- **Input maps:** for each operand, 9 complex additions (1 for `z`, 4 for
  `b_±` and `c_±`, 2 for the first two coordinates of `L_2 X`, 2 for
  `z ± iota y1`), so 36 real additions for both operands.
- **Cross products:** 12 real cross products with 6 multiplications and
  3 subtractions each, for 72 multiplications and 36 additions. Their summed
  inputs `(Re_C + Im_C)(L_j X)` and `(Re_C + Im_C)(L_j Y)` cost 22 additions:
  6 for each of the four complex cross products, less 2 because `L_1` and
  `L_2` share the coordinate `p`, whose sum is formed once per operand.
  Reconstructing the 10 complex coordinates other than `Upsilon_1` and
  `Upsilon_2` costs 30 more.
- **Output:** 12 complex additions for the six intermediate quantities and
  the six outputs other than `h_hat`, 2 real additions for `phi` and `chi`,
  and 3 for each part of `h_hat`, for 32 real additions. The halvings are 4
  for `Theta_±`, 8 for the outputs built from `Pi` and `Sigma`, and 3 for
  `eta_2`, `p_2` and `omega_2`.

| Stage | M | A | E |
| --- | ---: | ---: | ---: |
| Input linear maps | 0 | 36 | 0 |
| Four complex cross products | 72 | 88 | 0 |
| Output reconstruction | 0 | 32 | 15 |
| Total | 72 | 156 | 15 |

The leaf therefore uses **243 operations**. Because of the one-call root, a
real product first reaches it at dimension 16, and a real product of
dimension `N >= 16` evaluates `N/16` leaves. At dimension 8 the root calls
`C_2`, which is the nonrecursive map alone and costs `(18, 24, 0)`.

### Closed forms

With the leaf counts `(M_3, P_3, H_3) = (72, 156, 15)`, the recurrences
give, for `K = 2^k` and `k >= 3`,

```text
M_k = (36k - 81) K/4 + 18
P_k = (76k - 168) K/4 + 36
H_k = (16k - 37) K/8 + 4.
```

A real product of dimension `N >= 16` adds the following to one call of
`C_(n-1)`, with `m = N/2 - 1`:

- the real nonrecursive map at the root: `6m` products and `4m - 1`
  additions;
- combining its vector outputs with `Re_C(Z)` and `Im_C(Z)`: `2m` additions;
- the splitting (1): `3N - 2` products and `3N - 3` additions.

That is `6N - 8` products, `6N - 10` additions and no halvings, so

```text
M = M_(n-1) + 6N - 8,     A = P_(n-1) + 6N - 10,     E = H_(n-1),
```

which gives the quasilinear formulas in [Operation counts](#operation-counts).

### Cariow–Cariowa implementation

Cariow–Cariowa is implemented through dimension 1024. Through dimension 64 it
uses generated straight-line kernels with the sign pattern built in. From
dimension 128 on, the same operations run as loops over generated
sign-correction tables, which need gathered loads. Fewer arithmetic
operations do not make this access pattern faster; see
[benchmark.md](benchmark.md).

## Real-coefficient implementation

### Workspace

Workspace sizes are `O(N)` doubles and do not depend on the batch size: a
batch reuses one workspace for all its products, and the paired SIMD batch
kernels pack operands in a fixed internal buffer. With the default leaf
cutoffs:

- **Direct and scalar quasilinear:** none through `N = 16`. Above that,
  half-dimensional temporaries at each recursion level above the
  straight-line leaves, `2N - 32` doubles in all.
- **Single quasilinear product with SIMD pairing:** none through `N = 16`.
  Above that, `5N - 63` doubles: `3N` for the transformed inputs and the
  outputs of both recursive calls of `C_(n-1)`, stored in pair layout;
  `2N - 64` for the paired recursion below them; and one word for alignment.
- **Cariow–Cariowa:** none through `N = 64`. From `N = 128`, `2N` doubles
  for the two transformed operands.
- **Associator and operator commutators:** `2N` more, for two intermediate
  elements.

Programs should obtain these sizes from the
[workspace queries](c-api.md#workspace-queries) rather than from these
formulas.

### SIMD kernels

The scalar and two-lane SIMD kernels use the same generated leaves and the
same recursion. In a batch, each lane holds a separate product, so batching
does not change the operations per product. A single quasilinear product
above dimension 16 instead places the two recursive calls of `C_(n-1)` in
the two lanes and continues with the paired kernel. Either way, each
coordinate is computed by the same operations in the same order, so finite
results agree bitwise between scalar, single-product SIMD and batched
evaluation; NaN payloads may differ. The [C API reference](c-api.md)
describes the packing storage and the cases where batches use the scalar
kernels.

### Automatic algorithm selection

For real coefficients, `fastcd_mul_f64` and ordinary Python multiplication
choose the algorithm by dimension, following the single-core benchmark
reported in the README. Single products and batches have separate tables,
because the paired SIMD kernels change the ranking below dimension 32:

| N | Single products | Batches |
| ---: | --- | --- |
| 2, 4 | quasilinear | quasilinear |
| 8 | direct | direct |
| 16 | direct | Cariow–Cariowa |
| 32 and above | quasilinear | quasilinear |

Builds without SIMD pairing use the single-product choices for batches.
`fastcd_default_algorithm` and `fastcd_default_batch_algorithm` report the
choices. To check them on your CPU, the [benchmark](benchmark.md) times all
three algorithms in both modes on one input corpus.

## Complex coefficients

Direct multiplication evaluates the doubling formula with complex
arithmetic. Quasilinear multiplication identifies `iota` with the
coefficient `i`, applies the complex kernel `C_n` directly to the imaginary
parts `X` and `Y`, and adds the remaining terms of (1) in complex
arithmetic. The one-call root needs real halves `u, v, s, t`, so over C both
recursive calls are made at every level, including the top.

The benchmark can also time a variant built from three real products. For
`x = a + i b` and `y = c + i d` with `a, b, c, d` in the real algebra, it
computes `ac`, `bd` and `(a + b)(c + d)` and returns
`(ac - bd) + i ((a + b)(c + d) - ac - bd)`. Its measured time includes
packing and reconstruction.

A complex product needs `10N + 64` doubles of workspace, and the associator
and operator commutators need `4N` more for two complex intermediates. A
batch reuses one workspace for all its products.

## Finite fields

The finite-field kernels were developed for fastCD. They use the same basis
and doubling formula as the real kernels, with exact arithmetic on the
residues described in [Coefficient storage](#coefficient-storage).

The quasilinear kernel runs the recursion above with coefficients in
`F[iota]/(iota^2 + 1)`. The paper's construction asks only for a commutative
coefficient ring, finite-dimensional over F, that contains a square root of
`-1` and has an F-linear projection onto F. It need not be a field, so this
works for every odd p, whether or not `-1` is already a square in F. In odd
characteristic every power of 2 is invertible, so the halvings are exact.

The kernels are specialized by extension degree. On x86, SSE4.1, AVX2 and
AVX-512 versions are compiled separately and chosen at run time by CPU
features, characteristic, degree and dimension; a portable kernel covers
other targets. The C library allocates no heap memory and keeps no global
state.

Direct multiplication evaluates the doubling formula recursively. Below
dimension 16, and wherever no packed kernel serves the CPU and field, it uses
scalar field arithmetic with quaternion-sized leaves, specialized by degree.
From dimension 16 on it uses the packed field arithmetic of the quasilinear
kernels, with leaves of eight coordinates. Quasilinear multiplication uses
`O(N log N)` operations in F and `O(N)` caller workspace; direct uses
`O(N^2)` operations. The real-coefficient counts on this page are not instruction
counts for these kernels: polynomial multiplication, reduction and
representation conversions add costs that depend on the characteristic and
degree, so the speedup over direct varies between fields.

Finite-field workspace depends on the kernel and degree, and a batch reuses
one workspace. Field contexts are owned by the caller and immutable after
setup, and calls with separate workspaces can run concurrently. Query
workspace lengths rather than hard-coding them.

## Floating point

Real coefficients, and the real and imaginary parts of complex coefficients,
are IEEE binary64 doubles. A product is computed by a sequence of rounded
operations, so it is not correctly rounded as a whole, and different
algorithms can return different results for the same operands.

### Accumulation order

For real coefficients, the algorithms add their terms in these orders:

- **Direct:** the generated leaves add the terms of each output coordinate
  in sequence. Above the leaf cutoff, each level of the doubling recursion
  adds or subtracts half-dimensional products coordinatewise.
- **Quasilinear:** every sum whose length grows with `N`, namely the scalar
  output of each nonrecursive map and the pairing `<X, Y>` in (1), adds its
  terms in sequence within blocks of eight and combines the block totals
  with a balanced tree. The generated leaves have fixed schedules.
- **Cariow–Cariowa:** the Walsh–Hadamard transforms run in butterfly stages,
  and through dimension 64 the straight-line kernels fix the order of every
  sum. From dimension 128 on, the correction sum of each output is split
  into blocks of 256 indices. Within a block, the terms go to four
  interleaved partial sums that are combined pairwise; above dimension 256,
  a balanced tree combines the block totals.

The complex-coefficient pairing is accumulated in the same way as the
quasilinear sums. None of these orders changes the operation counts. In the
quasilinear product and in large Cariow–Cariowa corrections, the balanced
trees make the rounding depth of long sums grow like `log N` rather than
`N`; [stability.md](stability.md) measures the resulting accuracy against an
exact reference.

### Benchmark tolerance

The real-coefficient benchmark compares every coordinate with an independent
binary64 evaluation of the doubling formula and accepts

```text
|computed - reference| <= 128 * DBL_EPSILON * N * ||x||_2 * ||y||_2.
```

This bounds the disagreement between two floating-point evaluations. It is
not an error against the exact product and not a general error bound.

### Overflow and special values

Intermediate values such as transformed sums, pairings and half-sums can
overflow or lose precision even when the final result would not, so operands
should be scaled away from the exponent limits. Over R, norms,
normalization, reciprocals and squares use scaled evaluation to avoid
unnecessary intermediate overflow. The [C API reference](c-api.md) describes
how each function handles NaN, infinities and signed zeros.

### Complex and finite-field checks

Finite-field arithmetic is exact. The
[field benchmark](benchmark.md#complex-and-finite-fields) checks every
finite-field product against an independent schoolbook polynomial
implementation, and every complex product against a long-double evaluation
of the doubling formula with a norm-scaled tolerance. An optional protocol
compares the finite-field arithmetic with FLINT (Fast Library for Number
Theory), an independent exact-arithmetic C library; see the
[benchmark guide](benchmark.md#comparing-finite-field-arithmetic-with-flint).
