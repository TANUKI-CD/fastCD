# Conventions and results

This page fixes the conventions in which fastCD represents Cayley-Dickson
elements, defines what each of its operations computes, and explains how its
multiplication algorithms work and what they cost. The
[paper](https://arxiv.org/abs/2609.11588) derives the quasilinear algorithm over
general rings of coefficients and proves the results quoted here, while this
page states them for the algebras that fastCD implements; the speed of the
algorithms is reported in the [benchmark](benchmark.md) and their accuracy in
the [stability study](stability.md).

The page has four parts:

1. **The algebra.** The coefficient fields, the doubling formula, the standard
   coordinates, the products of basis elements, and the quadratic form.
2. **The operations.** What fastCD computes besides the product: the alternating
   product, commutators, associators, operator commutators, and the reciprocal.
3. **The multiplication algorithms.** Direct, Cariow-Cariowa, and quasilinear
   multiplication, how the quasilinear algorithm works, what each algorithm
   costs, and how the algorithms run over complex and finite-field coefficients.
4. **Sums of products.** How a sum of many products is computed without forming
   the products, and the covariance built on that sum.

## The algebra

### Coefficient fields and the doubling formula

fastCD works in the standard Cayley-Dickson tower over a coefficient field `F`,
which is the real numbers R, the complex numbers C, or a finite field F_(p^d)
with `p` an odd prime. Starting from `A_0 = F`, each level consists of ordered
pairs of elements of the level below, with conjugation and multiplication

```text
A_n = A_(n-1) x A_(n-1)
conjugate((a, b)) = (conjugate(a), -b)
(a, b)(c, d) = (ac - conjugate(d) b,  d a + b conjugate(c))
```

Conjugation is the identity on `A_0 = F`, so it fixes every coefficient: over C
it does not conjugate complex coefficients, and over F_(p^d) it does not apply
the Frobenius map. The paper allows a general doubling parameter `mu_n` at each
level, in the product `(ac + mu_n conjugate(d) b, d a + b conjugate(c))`, and
coefficients in any commutative ring in which 2 is invertible; the standard
tower is the case in which every parameter is `-1`, and it is the one that
fastCD implements.

### Standard coordinates

An element of `A_n` has `N = 2^n` coordinates in `F`, its *standard
coordinates*, with respect to the basis `e_0 = 1, e_1, ..., e_(N-1)`. The basis
follows the pairs: when `N = 2M`, the first `M` coordinates of `(a, b)` are
those of `a` and the last `M` are those of `b`. Coordinate 0 is therefore the
scalar part `Re(x) = x[0]`, the imaginary part `Im(x)` is `x` with coordinate 0
set to zero, and conjugation negates the imaginary part:

```text
conjugate(x) = (x[0], -x[1], ..., -x[N-1]).
```

The literature uses several doubling formulas and orientations, and coefficient
arrays written in another convention convert to fastCD's by a signed permutation
of their coordinates. The [C API reference](c-api.md) and the [Python API
reference](python-api.md) describe how the coordinates are stored.

### Basis products

The product of two basis elements is a signed basis element whose index is the
bitwise XOR of the two indices,

```text
e_i e_j = sigma_N(i, j) e_(i XOR j),     sigma_N(i, j) in {+1, -1},
```

and the signs follow from the doubling formula: for `N = 2M` and `i, j < M`,

```text
sigma_N(i,     j)     =  sigma_M(i, j)
sigma_N(i,     M + j) =  sigma_M(j, i)
sigma_N(M + i, j)     =  epsilon(j) sigma_M(i, j)
sigma_N(M + i, M + j) = -epsilon(j) sigma_M(j, i),
```

where `sigma_1(0, 0) = 1`, `epsilon(0) = 1`, and `epsilon(j) = -1` for `j > 0`.
The function `fastcd_basis_product` evaluates this recurrence in `O(log N)`
integer steps. It follows that `e_0` is the identity, that `e_i^2 = -1` for
`i > 0`, and that `e_j e_i = -e_i e_j` for distinct nonzero `i` and `j`, and the
orientation is fixed by

```text
quaternions:  e_1 e_2 = e_3,   e_2 e_3 = e_1,   e_3 e_1 = e_2
octonions:    e_1 e_2 = e_3,   e_1 e_4 = e_5,   e_2 e_4 = e_6,   e_3 e_4 = e_7
```

The identities that the product satisfies weaken as the dimension grows, as the
names of the real algebras record. Over C and finite fields the same identities
hold, while statements about division can fail (see [The quadratic
form](#the-quadratic-form)):

| Dimension | Real algebra | Identities |
| ---: | --- | --- |
| 1 | Real numbers | Commutative, associative |
| 2 | Complex numbers | Commutative, associative |
| 4 | Quaternions | Associative |
| 8 | Octonions | Alternative |
| 16 and above | Sedenions and higher | Flexible and power-associative; zero divisors exist |

### The quadratic form

Write `<x, y> = sum_i x[i] y[i]` for the coordinate pairing and `Q(x) = <x, x>`
for the quadratic form. Both are defined by the same sums over every field, so
over C they involve no complex conjugation, and over R, `Q(x)` is the squared
Euclidean length.

`Q` is the norm form of the tower: by induction on the level, the doubling
formula gives

```text
x conjugate(x) = conjugate(x) x = Q(x) 1,
x^2 - 2 x[0] x + Q(x) 1 = 0.
```

Since the characteristic of `F` is never 2, the pairing is recovered from the
form by polarization, `<x, y> = (Q(x + y) - Q(x) - Q(y)) / 2`, and it is
nondegenerate. The composition law `Q(xy) = Q(x) Q(y)` holds through dimension 8
and fails in general from dimension 16 on.

Over C and finite fields, `Q` can vanish on a nonzero element, for example
`Q((1, i)) = 0` over C and `Q((1, 2)) = 0` over F_5, and such an element has no
reciprocal (see [Derived operations](#derived-operations)).

## Derived operations

Products of imaginary elements carry most of the structure of the algebra, so
fastCD writes `x = alpha + X` and `y = beta + Y`, where `alpha` and `beta` are
the scalar parts and `X` and `Y` are imaginary, and defines the *alternating
product* of two imaginary elements as

```text
X cross Y = (XY - YX) / 2.
```

It is bilinear, alternating, and imaginary, at dimension 4 it is the ordinary
three-dimensional cross product, and the library exposes it as `imag_cross`.
Since `XY + YX = -2 <X, Y>`, the product of two imaginary elements is
`XY = -<X, Y> + X cross Y`, and every product therefore splits into terms that
need no further multiplication except the alternating product:

```text
xy = (alpha beta - <X, Y>) + alpha Y + beta X + X cross Y.          (1)
```

The derived operations of fastCD are defined as follows. By (1), the commutator
needs only the alternating product, and the anticommutator and the square need
no product at all and take linear time:

```text
[x, y]         = xy - yx = 2 (X cross Y)
{x, y}         = xy + yx = 2 (alpha beta - <X, Y>) + 2 (alpha Y + beta X)
x^2            = (alpha^2 - <X, X>) + 2 alpha X
A(x, y, z)     = (xy) z - x (yz)
[L_a, L_b](x)  = a (b x) - b (a x)
[R_a, R_b](x)  = (x b) a - (x a) b
x^(-1)         = conjugate(x) / Q(x),     x x^(-1) = x^(-1) x = 1
```

The parentheses in these definitions matter. Through dimension 4 the algebras
are associative, so `[L_a, L_b](x) = (ab - ba) x` and
`[R_a, R_b](x) = -x (ab - ba)`, but from dimension 8 on neither identity holds
in general. The octonions are alternative, so `x^(-1) (xy) = y` still holds
through dimension 8, but from dimension 16 on it can fail, and the reciprocal
gives no cancellation law for products.

The reciprocal requires `Q(x) != 0`. Over R this means `x != 0`, while over C
and finite fields it also excludes the nonzero elements with `Q(x) = 0`, and the
reciprocal, scalar division, and negative powers report division by zero for
those elements as they do for zero.

## Multiplication algorithms

fastCD has three multiplication algorithms, which compute the same product with
different numbers of operations:

- **Direct** multiplication evaluates the doubling formula recursively, with
  `N^2` multiplications.
- **Cariow-Cariowa** is the uniform method of [Cariow and
  Cariowa](https://doi.org/10.15199/48.2015.02.09). It computes the unsigned XOR
  convolution of the two coordinate vectors with Walsh-Hadamard transforms and
  then corrects the sign of each output coordinate with a sparse sum, using
  `N (N - 1) / 2 + 2` multiplications; fastCD implements it for real
  coefficients through dimension 1024.
- **Quasilinear** multiplication is the algorithm of the paper, with
  `O(N log N)` operations.

All three multiply real coefficients, while complex and finite-field
coefficients use direct and quasilinear multiplication. In floating point the
algorithms round differently, so their results can differ in the last bits.

### How quasilinear multiplication works

By (1), multiplication reduces to the alternating product `X cross Y`, since its
other terms cost `O(N)` operations, and the quasilinear algorithm computes the
alternating product recursively, with two half-size recursive calls at each
level. The recursive structure behind it is classical: up to normalization, it
is the coordinate form of the trace-zero commutator recursion of Brown and
Hopkins, as the paper explains, and what the paper contributes is the
quasilinear arithmetic schedule built from it.

**Splitting the imaginary part.** The first half of an imaginary element is
imaginary, while the second half may have a nonzero scalar coordinate, so an
imaginary element at level `n` has the form `X = (u, gamma + s)`, where `u` and
`s` are imaginary at level `n - 1` and `gamma` is a scalar. Writing
`Y = (v, delta + t)` in the same way, with each of `u`, `v`, `s`, and `t` having
`m = N/2 - 1` coordinates, the doubling formula gives

```text
X cross Y = (D + f,  lambda + g - H),
D = u cross v - s cross t,     H = u cross t + s cross v,
```

where `(f, lambda, g)` is the value of the *nonrecursive map*

```text
B_n((u, gamma, s), (v, delta, t))
    = (gamma t - delta s,  <s, v> - <t, u>,  delta u - gamma v).
```

`B_n` collects every term that needs no recursive call, its three outputs have
`m`, `1`, and `m` coordinates, and `lambda` occupies the scalar coordinate of
the second half. For a general doubling parameter `mu_n` the first output is
`mu_n (delta s - gamma t)`, as in the paper.

**Two recursive calls.** `D` and `H` need four alternating products at level
`n - 1`, and a coefficient `iota` with `iota^2 = -1` reduces them to two. This
`iota` is a scalar, not one of the basis elements `e_i`, so it commutes with
everything and is fixed by conjugation. Write `C_k` for the alternating product
at level `k` with coefficients of the form `a + iota b`, the *kernel* of the
algorithm. Bilinearity gives `C_(n-1)(u +/- iota s, v +/- iota t) = D +/- iota H`, so
two calls suffice:

```text
Z_+ = C_(n-1)(u + iota s, v + iota t)
Z_- = C_(n-1)(u - iota s, v - iota t)
C_n(X, Y) = ((Z_+ + Z_-)/2 + f,  lambda + g + iota (Z_+ - Z_-)/2).
```

Each level makes two calls of half the size and does `O(N)` other work, so the
recursion uses `O(N log N)` operations.

**The one-call root.** For real inputs, `u`, `v`, `s`, and `t` are real, so `D`
and `H` are real and `Z_+ = D + iota H` alone determines both. One call
therefore suffices at the top level:

```text
Z = C_(n-1)(u + iota s, v + iota t)
X cross Y = (Re_C(Z) + f,  lambda + g - Im_C(Z)),
```

where `Re_C` and `Im_C` take the parts `a` and `b` of each coefficient
`a + iota b`, and `B_n` is evaluated over the reals. Below the root the inputs
have coefficients of the form `a + iota b`, and both calls are needed.

**Evaluating the nonrecursive map.** With coefficients of the form `a + iota b`,
`B_k` is evaluated as a whole by the three-product formula for complex
multiplication: for operands `a_0 + iota a_1` and `b_0 + iota b_1` with real
`a_0`, `a_1`, `b_0`, and `b_1`,

```text
Delta = B_k(a_0, b_0)
Omega = B_k(a_1, b_1)
W     = B_k(a_0 + a_1, b_0 + b_1)
B_k(a_0 + iota a_1, b_0 + iota b_1)
    = (Delta - Omega) + iota (W - Delta - Omega).
```

Applying the formula to the whole map, rather than to each product of
coefficients inside it, forms each input sum once and reconstructs each output
coordinate once, so that at level `k`, with `m = 2^(k-1) - 1`, the map costs
`18m` real products and `22m + 2` additions.

**Base cases.** The recursion ends in one of three ways. At level 1 the
alternating product is zero. At level 2 the half-size alternating products
vanish, so `C_2(X, Y) = (f, lambda + g)` is the nonrecursive map alone. At level
3 the kernel uses the octonion leaf described next, which a real product first
reaches at dimension 16, because of the one-call root.

### The octonion leaf

At level 3 the kernel evaluates the alternating product of two imaginary
octonions with coefficients `a + iota b` by a direct formula. Write such an
element, which has seven coordinates, as

```text
X = (x1, x2, x3, h, y1, y2, y3),
```

so that `x1`, `x2`, and `x3` are the coordinates of `e_1`, `e_2`, and `e_3`, `h`
is that of `e_4`, and `y1`, `y2`, and `y3` are those of `e_5`, `e_6`, and `e_7`.
The formula uses the linear forms

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

With `cross` the ordinary three-dimensional cross product, extended bilinearly
to coefficients `a + iota b`, and `^T` the ordinary matrix transpose, without
conjugation, the paper proves

```text
C_3(X, Y) = -iota L_1^T (L_1 X cross L_1 Y)
            + (1/2) sum_(j=2..4) L_j^T (L_j X cross L_j Y).
```

fastCD evaluates this formula in three stages. The first applies the maps `L_j`
to both operands, with nine additions of coefficients per operand, since forming
`p` is free. The second evaluates the four cross products `L_j X cross L_j Y`,
each by three real cross products according to the three-product formula above;
the sum of the two parts of `p` is formed once per operand and shared by `L_1`
and `L_2`, and the third coordinates of the first two cross products, which
enter only the output coordinate of `e_4`, are combined there without being
reconstructed separately. The third stage collects the entries of the sparse
transposes into the seven output coordinates. The stages cost:

| Stage | M | A | E |
| --- | ---: | ---: | ---: |
| Input linear maps | 0 | 36 | 0 |
| Four cross products | 72 | 88 | 0 |
| Output reconstruction | 0 | 32 | 15 |
| Total | 72 | 156 | 15 |

The leaf therefore uses 243 operations, in the counting conventions of the next
section. Its implementation is the straight-line kernel `ql_complex_cross_8`,
which `tools/generate_kernels.py` generates.

### Operation counts

The counts on this page follow the conventions of the paper. `M` counts
multiplications of two input-dependent values, `A` counts additions and
subtractions, and `E` counts multiplications by fixed scalars other than `0` and
`+/-1`, and the total `T = M + A + E` includes the preprocessing of both operands.
Reading coordinates, copying, permuting, and changing signs are free, and so is
multiplication by `iota`, which swaps the two parts of a coefficient and changes
one sign, so that halvings are the only fixed-scalar multiplications in the
kernel. Known-zero terms are skipped, and each sum starts from its first term.
The counts are for real coefficients; [Other coefficient
fields](#other-coefficient-fields) says what carries over.

**Cost of one level.** Write `M_k`, `P_k`, and `E_k` for the products,
additions, and halvings of one call to `C_k`, including its recursive calls.
Outside its recursive calls, a call with `k >= 4` and `m = 2^(k-1) - 1` costs:

| Work | M | A | E |
| --- | ---: | ---: | ---: |
| Nonrecursive map | `18m` | `22m + 2` | 0 |
| Forming `u +/- iota s` and `v +/- iota t` | 0 | `8m` | 0 |
| Forming `Z_+ +/- Z_-` | 0 | `4m` | 0 |
| Adding `f` and `g` | 0 | `4m` | 0 |
| Halving `Z_+ +/- Z_-` | 0 | 0 | `4m` |
| Total | `18m` | `38m + 2` | `4m` |

The counts of a call therefore satisfy

```text
M_k = 2 M_(k-1) + 18m
P_k = 2 P_(k-1) + 38m + 2
E_k = 2 E_(k-1) + 4m
```

and with the leaf counts `(M_3, P_3, E_3) = (72, 156, 15)` these recurrences
give, for `K = 2^k` and `k >= 3`,

```text
M_k = (36k - 81) K/4 + 18
P_k = (76k - 168) K/4 + 36
E_k = (16k - 37) K/8 + 4
```

**The full product.** A real product of dimension `N >= 16` adds three pieces of
work to its one call of `C_(n-1)`, with `m = N/2 - 1`: the real nonrecursive map
at the root, with `6m` products and `4m - 1` additions; the combination of its
vector outputs with `Re_C(Z)` and `Im_C(Z)`, with `2m` additions; and the
splitting (1), with `3N - 2` products and `3N - 3` additions. That is `6N - 8`
products, `6N - 10` additions, and no halvings, so

```text
M = M_(n-1) + 6N - 8,     A = P_(n-1) + 6N - 10,     E = E_(n-1),
```

which gives the quasilinear row of the following table, alongside the counts of
the other two algorithms (`N = 2^n` throughout):

| Algorithm | `M(N)` | `A(N)` | `E(N)` |
| --- | --- | --- | --- |
| Direct | `N^2` | `N (N - 1)` | `0` |
| Cariow-Cariowa (`4 <= N <= 1024`) | `N (N - 1) / 2 + 2` | `3 N n + (N (N - 3) + 4) / 2` | `2N` |
| Quasilinear (`N >= 16`) | `(36n - 69) N / 8 + 10` | `(76n - 196) N / 8 + 26` | `(16n - 53) N / 16 + 4` |

At `N = 2`, Cariow-Cariowa uses `(M, A, E) = (3, 7, 3)`: only the scalar
coordinate needs a sign correction, so `E` counts the two scalings by `1/N` and
a single doubling. For `N >= 16` the quasilinear total is
`T = (240n - 583) N / 16 + 40`, and the exact quasilinear counts, including the
small dimensions, where the recursion does not reach the leaf, are:

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

The three algorithms compare as follows:

- Quasilinear multiplication uses fewer input-dependent multiplications than
  direct multiplication from `N = 8` on, and fewer than Cariow-Cariowa from
  `N = 32` on; at `N = 16` Cariow-Cariowa uses 122 against 160.
- The quasilinear total equals the direct total at `N = 2`, `4`, and `8` and is
  smaller from `N = 16` on: 417 against 496 at `N = 16`.
- The quasilinear total is smaller than the Cariow-Cariowa total at every
  dimension from 2 through 1024: 417 against 452 at `N = 16`. This holds even
  when the fixed-scalar multiplications of Cariow-Cariowa are not counted, since
  its `M + A` is 420 at `N = 16` and stays above the quasilinear total at every
  larger dimension.
- Direct multiplication and Cariow-Cariowa use `Theta(N^2)` operations, and
  quasilinear multiplication uses `Theta(N log N)`.

Because the quasilinear algorithm is bilinear, multiplying linear forms in one
operand with linear forms in the other, its number of multiplications is also a
bound on the tensor rank of multiplication in `A_n`. The paper proves the bound
`(36n - 69) 2^(n-3) + 10` from `n = 4` on, which is `M` above, over R and over
every finite field of odd characteristic.

**What the counts describe.** These counts describe the arithmetic schedules as
fastCD ships them, in the default build configuration. They are not counts of
machine instructions and do not predict running time, which also depends on how
each schedule maps onto the hardware, as the [benchmark](benchmark.md) shows.
The tools `tools/check_quasilinear_counts.py` and `tools/check_cariow_counts.py`
instrument the generated kernels and check the counts through dimension 1024.

## Other coefficient fields

### Complex coefficients

Direct multiplication evaluates the doubling formula with complex arithmetic.
Quasilinear multiplication identifies `iota` with the complex coefficient `i`,
which, like `iota`, is a scalar that commutes with everything and is fixed by
conjugation. It therefore applies the kernel `C_n` directly to the imaginary
parts `X` and `Y`, with the octonion leaf at level 3, and adds the remaining
terms of (1) in complex arithmetic. The one-call root needs real halves `u`,
`v`, `s`, and `t`, so both recursive calls are made at every level, including
the top. For `n >= 3` the alternating product then uses `M_n` real
multiplications, `(36n - 81) 2^(n-2) + 18`, and (1) adds `3N - 2` products of
complex coefficients, against the `N^2` products of complex coefficients of
direct multiplication.

### Finite fields

Over F_(p^d) the quasilinear algorithm runs the same recursion with coefficients
in the ring `F[iota]/(iota^2 + 1)`, whose elements are `a + iota b` with `a` and
`b` in `F`. The paper's construction asks only for a commutative coefficient
ring, finitely generated over `F`, that contains a square root of `-1` and has
an `F`-linear map onto `F` that fixes 1, so the ring need not be a field. It is
the field F_(p^(2d)) when `-1` is not a square in `F`, that is, when
`p^d = 3 (mod 4)`, and the same recursion works when `p^d = 1 (mod 4)`, where
the ring is not a field. Every power of 2 is invertible in odd characteristic,
so the halvings are exact, like all finite-field arithmetic.

The finite-field kernels end the recursion at level 2, with the nonrecursive map
alone, rather than at the octonion leaf, and each of their coefficient
operations is a polynomial multiplication, reduction, or addition whose cost
depends on the characteristic and the degree. The real counts above therefore do
not describe them, although the quasilinear algorithm still uses `O(N log N)`
operations in `F` and direct multiplication `O(N^2)`.

## Fused sums of products

A fused sum computes `S = x_0 y_0 + ... + x_(P-1) y_(P-1)` for `P` pairs of
real-coefficient elements without forming the products, and `fastcd_mul_sum_f64`
and `fastcd_covariance_f64` are built on it.

### The sum schedule

Every multiplication algorithm above is bilinear: it applies linear maps `In_1`
and `In_2` to the operands, multiplies the results coordinate by coordinate, and
applies a linear map `Out` to the products,

```text
x y = Out((In_1 x) o (In_2 y)),
```

where `o` is the coordinatewise product. Summing over the elements gives

```text
S = Out(sum_p (In_1 x_p) o (In_2 y_p)),
```

so each multiplication can be accumulated over the elements, and `Out` applied
once, to the accumulated sums. For each element only the additions inside `In_1`
and `In_2` and one multiply-add per multiplication remain, while the additions
of `Out`, about half of the quasilinear total, are spent once per sum and do not
depend on `P`.

For direct multiplication, `In_1` and `In_2` only select coordinates, the
accumulated sums are the entries `sum_p x_p[i] y_p[j]` of the matrix of real
cross moments, and `Out` adds them with the basis signs. For quasilinear
multiplication, the accumulated sums are those of the nonrecursive maps, of
`B_n` at the root and of the three real evaluations `Delta`, `Omega`, and `W` at
every node below, and `Out` consists of the complex recovery, the half-sums
`(Z_+ +/- Z_-)/2`, and the additions of `f`, `lambda`, and `g`, applied from the
leaves to the root.

### The leaf of the sum

The octonion leaf would leave 36 sums to accumulate at each leaf, one for each
coordinate of its twelve real cross products, which is more than the kernels can
keep in registers. The fused sums therefore evaluate `C_3` instead as the
seven-dimensional cross product of the two complex imaginary octonions, by the
three-product formula: three real seven-dimensional cross products, of the real
parts, of the imaginary parts, and of their sums. Each has 42 multiplications
and 7 accumulated sums, and forming the operands of the third costs 14
additions, so the leaf has 126 multiplications and 14 input additions.

### Counts per element

For one element, the root, with the splitting (1) and the real map `B_n`, uses
`6N - 8` multiplications and no additions. A node `C_k` with `k >= 4` and
`m = 2^(k-1) - 1` uses `18m` multiplications, `8m` additions for the operands of
its two children, and `6m` additions for the sums of real and imaginary parts:
four for each coordinate index and two for `gamma` and `delta`, which this
schedule forms again for every coordinate. The leaf `C_3` uses 126
multiplications and 14 additions, and at `N = 8`, where the recursion ends in
`C_2`, the complex nonrecursive map uses 18 multiplications and 6 additions. The
output additions of `Out` are spent once per sum and are not counted here:

| N | Direct multiplications | Quasilinear multiplications | Quasilinear additions | Quasilinear total |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 4 | 4 | 0 | 4 |
| 4 | 16 | 16 | 0 | 16 |
| 8 | 64 | 58 | 6 | 64 |
| 16 | 256 | 214 | 14 | 228 |
| 32 | 1024 | 562 | 126 | 688 |
| 64 | 4096 | 1402 | 462 | 1864 |
| 128 | 16384 | 3370 | 1358 | 4728 |
| 256 | 65536 | 7882 | 3598 | 11480 |
| 512 | 262144 | 18058 | 8974 | 27032 |
| 1024 | 1048576 | 40714 | 21518 | 62232 |

The per-element counts of the two algorithms agree through `N = 8`, and the
quasilinear sum has fewer operations from `N = 16` on.
`tools/check_fused_sum_counts.py` instruments the kernel source and checks these
counts, together with the additions that the kernels spend merging partial sums,
through dimension 1024.

### Covariance

For two samples of `P` elements, `fastcd_covariance_f64` returns

```text
cov(x, y) = (1 / (P - ddof)) sum_p (x_p - mean(x)) conjugate(y_p - mean(y)).
```

Its scalar coordinate is `sum_i cov(x[i], y[i])`, the sum of the real
covariances of the coordinates. In exact arithmetic `cov(x, x)` has the total
variance as its scalar coordinate and a vanishing imaginary part, while in
floating point its imaginary coordinates are rounding residues of the size of
the errors in [Accuracy of fused sums](stability.md#accuracy-of-fused-sums). At
dimension 2, with elements `(real, imag)`, it is the covariance
`E[(z - mean(z)) conjugate(w - mean(w))]` of complex samples.

The centering and the conjugate form an affine map of the elements, which is
applied as they are read, and the sum is a fused sum. Two details reduce the
rounding error:

- **The mean is kept in two parts.** With `r` the first element and `d` the mean
  of the differences `x_p - r`, the mean is `r + d`, but that sum is never
  formed, because it need not be representable: for coordinates near `2^53` with
  a spread of a few units, rounding it would lose the part that matters. Each
  element is centered as `(x_p - r) - d` instead, where `x_p - r` is exact when
  `x_p` and `r` are close and `d` is small.
- **Extreme magnitudes are scaled.** If the largest magnitude among the varying
  coordinates of a sample lies outside `[2^-257, 2^256)`, the sample is scaled
  toward unit magnitude by a power of two, and the result is scaled back at the
  end. Small coordinates can underflow when the range of magnitudes is wide.

## Floating point

Real coefficients, and the real and imaginary parts of complex coefficients, are
IEEE binary64 values, so a product is computed by a sequence of rounded
operations and is not correctly rounded as a whole, and different algorithms can
return different results for the same operands. The [stability
study](stability.md) measures the accuracy of the algorithms against exact
products and explains the order in which they add their long sums, and
finite-field arithmetic is exact.

Intermediate values such as transformed sums, pairings, and half-sums can
overflow or lose precision even when the final result would not, so operands
should be scaled away from the exponent limits. Over R, norms, normalization,
reciprocals, and squares use scaled evaluation to avoid unnecessary intermediate
overflow, and the [C API reference](c-api.md) describes how each function
handles NaN, infinities, and signed zeros.
