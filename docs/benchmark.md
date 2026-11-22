# Benchmark

fastCD multiplies with three algorithms, and the library chooses among them
automatically by dimension. Direct multiplication and the algorithm of Cariow
and Cariowa use a number of operations that grows with the square of the
dimension, while the quasilinear algorithm uses `O(N log N)` operations (see
[Multiplication algorithms](mathematics.md#multiplication-algorithms)), but a
smaller operation count does not by itself guarantee that an algorithm will be
faster at a certain dimension for a given implementation. On an actual CPU the
running time also depends on how the structure of an algorithm maps onto the
hardware: how regularly it reads memory, how much of its work can run in SIMD
registers, how much bookkeeping its recursion needs, and how large the fixed
cost of a call is compared with the arithmetic. An algorithm can therefore be
faster or slower at a given dimension than its operation count suggests, and the
only way to know which algorithm is fastest there is to time the implementations
themselves.

This project's benchmark does exactly this in order to justify the choices of
`AUTO`, the automatic selection, with reproducible data. It does not try to
prove that one algorithm is faster than another in general, for example that
quasilinear multiplication is faster than Cariow-Cariowa at some dimension for
all implementations on every machine; instead, it measures fastCD's
implementations as the library ships them, on one host, and asks which of them
the library should call at each dimension.

The benchmark also justifies a second choice, which concerns the arithmetic
underneath the products rather than the algorithms. Cayley-Dickson products over
finite fields need fast arithmetic in prime fields and their extensions, and
instead of relying for it on an external library such as FLINT (Fast Library for
Number Theory), a widely used C library for number theory and exact arithmetic,
fastCD uses a finite-field kernel developed for the project, because that kernel
was measured against FLINT and found to be significantly faster. The comparison
with FLINT on this page provides that evidence. Like the rest of the benchmark,
it does not try to show that fastCD's finite-field arithmetic is better than
FLINT's in general; it only explains why fastCD implements this arithmetic
itself instead of depending on an existing, mature library, with FLINT as the
example.

The measurements show that over the real numbers the quasilinear product is the
fastest from dimension 32 on, both in single products and in batches, and about
17 times faster than direct multiplication at dimension 1024, while below
dimension 32 the fastest algorithm depends on the dimension and on whether the
products are batched. Over the complex numbers and finite fields, the
quasilinear product is the fastest from dimension 8 on, and direct
multiplication is the fastest at dimension 2, with the two being tied at
dimension 4. Cariow-Cariowa, although it needs fewer operations than direct
multiplication, is the fastest only for batches at dimension 16. fastCD's
finite-field arithmetic, finally, is faster than FLINT's in every field and at
every dimension measured, by factors of 2 to 7 for a single field multiplication
and of 2 to 465 for complete products from dimension 16 on. The sections below
describe what is measured and how the timings are made, report the results for
real coefficients, for complex coefficients and finite fields, for fastCD's
finite-field arithmetic against FLINT, and for fused sums of products, show how
these results set the automatic choices, and end with the limitations of the
measurements and with instructions for running the benchmark.

## What is measured

Not all of the three algorithms are available for every coefficient field. All
three are supported with real coefficients, with Cariow-Cariowa available over
the real numbers through dimension 1024, while complex and finite-field
coefficients have direct and quasilinear multiplication only. `AUTO` accordingly
makes four kinds of choice: one for real single products, one for real batches,
one for complex and finite-field products, and one for fused sums (see [Fused
sums of products](mathematics.md#fused-sums-of-products)).

Only the native calls of the C library are timed, with every input, output, and
workspace allocated beforehand, so that allocation, Python overhead, and the
verification of results stay outside the timers. The timings therefore show what
a program calling the C library pays for the arithmetic itself.

Products are timed in two *modes*, which can rank the algorithms differently.
*Batch mode* makes one call for all the products of a pass over the inputs, and
with the default SIMD kernels on supported GCC and Clang targets it evaluates
two products at a time, one in each lane of a SIMD register: for quasilinear
multiplication and Cariow-Cariowa at dimensions 16 through 512, and for direct
multiplication at 64 through 512. *Single mode* makes one call per product, and
since every call validates its arguments, which costs about 15 ns on the
archived host, that cost dominates the time of a product below dimension 8. From
dimension 32, a single quasilinear product also uses both SIMD lanes, by
evaluating the two recursive calls of its top level side by side (see [How
quasilinear multiplication
works](mathematics.md#how-quasilinear-multiplication-works)), while single
direct and Cariow-Cariowa products run scalar code that the compiler may
vectorize.

The driver `benchmarks/benchmark.py` runs four experiments, called *protocols*,
each of which answers one question:

- **Real coefficients** (`real`): which of the three algorithms is the fastest
  in each mode at each dimension from 2 through 1024?
- **Complex coefficients and finite fields** (`domains`): where does quasilinear
  multiplication overtake direct multiplication over the complex numbers and
  over finite fields of odd characteristic?
- **Comparison with FLINT** (`flint`): is fastCD's finite-field arithmetic
  enough faster than FLINT's to justify implementing it in fastCD rather than
  depending on FLINT?
- **Fused sums** (`sums`): when is a fused sum, which computes
  `x_0 y_0 + ... + x_(P-1) y_(P-1)` without forming the products (see [Fused
  sums of products](mathematics.md#fused-sums-of-products)), faster than a batch
  of products followed by a sum, and which algorithm should it use?

## How timings are made

### Corpora and verification

Every protocol times a fixed *corpus*, a deterministic set of operand pairs for
each dimension and, where the protocol has them, for each field. The driver
generates the corpus with NumPy from a recorded seed, stores it for reuse, and
binds it to a manifest of its parameters and its hash, so that a run never times
inputs other than the ones it reports. Every product is also checked against an
independent implementation outside the timers, and a failed check is reported as
a failure, never as a timing; the result sections below name the reference that
each protocol uses.

### Observations, blocks, and tasks

An *observation* times complete passes over the corpus: after two untimed warmup
passes, the runner calibrates the number of passes so that the observation lasts
at least 20 ms, and divides the elapsed time by the number of products. A *task*
is one thing being timed, such as one algorithm in one mode, or one library in
the comparison with FLINT, and a *cell* is one dimension, together with one
field or one sample size where the protocol has them.

Each cell is timed in a run of *blocks*, 24 by default, and each block takes one
observation of every task. The order of the tasks rotates from block to block,
so that each task holds each position equally often, and every ratio between two
tasks is formed within a block, so that a slow change in the state of the host,
such as a drift in clock frequency, affects both sides of the ratio.

### Code and stack placement

A timing can also depend on where the runner's code and stack happen to lie in
memory, which has nothing to do with the arithmetic being measured, and the
runners are built so that placement does not decide the results. The real
protocol starts a new runner process for every observation, which gives each
observation its own placement in memory. The other protocols run all the blocks
of a cell in one process and instead give each block its own stack offset,
spread evenly in 16-byte steps over a 4 KB page, because some finite-field
kernels run markedly slower at a few stack positions, and one position shared by
a whole cell would shift all of its observations together.

Code placement matters in a second way on the archived processor. On
Skylake-family processors from Intel, a jump instruction that crosses or ends on
a 32-byte boundary runs markedly slower (Intel's JCC erratum), so an unrelated
edit to a runner could move a timing loop onto such a boundary and change its
timings. Where the compiler supports it, CMake therefore builds the four runners
with `-mbranches-within-32B-boundaries`, which pads their code so that no branch
lies across such a boundary, while the library itself is built as it ships; the
build record lists the option as `runner_options`.

### Statistics

A time is the mean over the blocks, in nanoseconds per product, with a two-sided
95% Student-t interval whose degrees of freedom are capped at 30. A ratio is the
geometric mean of the per-block ratios, with its own 95% interval, and each
table below states which side a ratio above 1 favors. The *fastest mean* of a
real cell names the algorithm with the lowest mean, or a tie when the interval
of the paired ratio between the two lowest means contains 1.

What an interval covers depends on the protocol, since it describes only the
variation between the blocks of one run. In the real protocol every observation
has its own process, so the interval covers differences between processes as
well. In the other protocols a cell keeps one process, and with it one placement
in physical memory, which affects cache behavior once the data of a cell no
longer fit in the per-core L2 cache: in repeated runs on the archived host,
individual finite-field cells from dimension 64 upward differed by up to 17%,
and direct/quasilinear ratios by up to 8%, more than their intervals suggest.

## The archived run

All results on this page come from one run, archived in
[`benchmark-results/t440-20261120/`](../benchmark-results/t440-20261120/README.md)
and measured on 2026-11-20 and 2026-11-21 from a clean checkout of commit
`93c9bd8`, the source of fastCD 1.5.0. The host is a Dell PowerEdge T440 with an
Intel Xeon Gold 6148, a Skylake-family server processor, running Linux 7.2.11.
It is shared with other work, and no frequency governor, turbo, or SMT settings
were changed, but each protocol ran alone, pinned to one logical CPU. The
library was built by CMake in its Release configuration
(`-O3 -DNDEBUG -fno-fast-math -ffp-contract=off`) with GCC 16.2.1, and the
domains and FLINT protocols used 1,000 operand pairs per cell instead of the
default 10,000, which keeps their run times to hours rather than days.

Every check of the run passed: 600,000 real products, 100,000 complex and
finite-field pairs, 132,000 pairs in the comparison with FLINT, and every
agreement check of the fused sums. The README of the archive records the
commands, the builds, and the load on the host during the run, and every JSON
file in it records the hash of every measured source file.

## Real coefficients

The real protocol times the three algorithms in both modes on 10,000 operand
pairs at each dimension, with independent uniform coordinates scaled to give
norms between 0.5 and 2, and in the first block it checks every product against
an independent binary64 implementation of the defining recursion. A product
passes when every coordinate lies within 128 * DBL_EPSILON * N * ||x||_2 *
||y||_2` of the reference, a bound on the disagreement between two
floating-point evaluations rather than on the error against the exact product.
The tables give the mean time per product with its 95% interval, the fastest
mean as defined in [Statistics](#statistics), and the algorithm that `AUTO`
selects for the mode.

**Batch products** (ns per product)

| N | direct | cariow_cariowa | quasilinear | fastest mean | default |
| ---: | --- | --- | --- | --- | --- |
| 2 | 1.32 [1.30, 1.34] | 1.52 [1.49, 1.55] | 1.02 [1.00, 1.04] | quasilinear | quasilinear |
| 4 | 3.84 [3.80, 3.88] | 4.65 [4.60, 4.69] | 3.68 [3.62, 3.73] | quasilinear | quasilinear |
| 8 | 12.18 [11.99, 12.37] | 14.05 [13.85, 14.25] | 17.59 [17.34, 17.85] | direct | direct |
| 16 | 54.79 [53.87, 55.71] | 49.92 [48.69, 51.14] | 52.41 [51.06, 53.76] | cariow_cariowa | cariow_cariowa |
| 32 | 256.56 [252.77, 260.35] | 227.38 [221.50, 233.25] | 184.50 [182.24, 186.76] | quasilinear | quasilinear |
| 64 | 1133.91 [1109.34, 1158.48] | 973.51 [951.64, 995.38] | 465.99 [460.22, 471.75] | quasilinear | quasilinear |
| 128 | 4530.78 [4453.12, 4608.44] | 5006.53 [4926.07, 5086.99] | 1242.81 [1221.67, 1263.95] | quasilinear | quasilinear |
| 256 | 17665.04 [17305.84, 18024.25] | 17695.42 [17442.11, 17948.73] | 3046.46 [2999.09, 3093.83] | quasilinear | quasilinear |
| 512 | 69956.80 [69154.63, 70758.97] | 68578.92 [67922.14, 69235.70] | 6920.47 [6825.76, 7015.18] | quasilinear | quasilinear |
| 1024 | 275135.85 [273397.86, 276873.84] | 433524.98 [419774.76, 447275.19] | 16209.21 [16034.88, 16383.55] | quasilinear | quasilinear |

**Single products** (ns per product)

| N | direct | cariow_cariowa | quasilinear | fastest mean | default |
| ---: | --- | --- | --- | --- | --- |
| 2 | 15.76 [15.52, 16.01] | 16.76 [16.56, 16.95] | 15.30 [15.12, 15.48] | quasilinear | quasilinear |
| 4 | 17.75 [17.48, 18.02] | 20.22 [19.97, 20.48] | 18.08 [17.91, 18.26] | direct | direct |
| 8 | 28.22 [27.86, 28.58] | 38.16 [37.71, 38.60] | 36.25 [35.84, 36.67] | direct | direct |
| 16 | 76.28 [74.68, 77.88] | 110.28 [108.18, 112.37] | 108.25 [106.07, 110.43] | direct | direct |
| 32 | 298.82 [295.63, 302.01] | 432.14 [423.85, 440.43] | 221.13 [217.79, 224.47] | quasilinear | quasilinear |
| 64 | 1163.54 [1146.73, 1180.35] | 1835.74 [1797.03, 1874.46] | 518.34 [511.99, 524.69] | quasilinear | quasilinear |
| 128 | 4495.22 [4430.30, 4560.14] | 7910.79 [7706.85, 8114.74] | 1333.57 [1308.25, 1358.88] | quasilinear | quasilinear |
| 256 | 17445.55 [17268.47, 17622.63] | 29214.67 [28321.53, 30107.80] | 3244.06 [3196.11, 3292.01] | quasilinear | quasilinear |
| 512 | 69310.72 [68887.00, 69734.44] | 108032.81 [107477.81, 108587.82] | 7427.95 [7327.56, 7528.34] | quasilinear | quasilinear |
| 1024 | 277013.17 [274746.54, 279279.79] | 419988.45 [417575.06, 422401.83] | 16258.80 [16093.16, 16424.44] | quasilinear | quasilinear |

**Small dimensions.** Below dimension 32 the fixed costs of a call decide the
ranking, and since these costs differ between the modes, so do the rankings. In
single mode every product pays for the validation of its call, which at
dimension 2 makes a single product take about 15 times as long as one product of
a batch, and among single products direct multiplication is the fastest at
dimensions 4, 8, and 16. In batch mode, where the cost of the call is shared by
all the products, the quasilinear product is the fastest at dimensions 2 and 4,
direct multiplication at 8, and Cariow-Cariowa at 16.

**From dimension 32.** From dimension 32 on, the quasilinear product is the
fastest in both modes, and its lead over direct multiplication grows with the
dimension, from about 1.4 times at dimension 32 to between 5.4 and 5.8 times at
256 and 17 times at 1024.

**Operation counts are not running time.** Cariow-Cariowa shows most clearly
that operation counts alone do not decide running time. At dimensions 128
through 1024 it needs only 0.51 to 0.58 times the
[operations](mathematics.md#operation-counts) of direct multiplication, yet its
single products are 1.5 to 1.8 times slower than direct, and its batches are 11%
slower at dimension 128, within 2% at 256 and 512, and 1.6 times slower at 1024.
The reason lies in its structure on the hardware: through dimension 64 it runs
generated straight-line kernels, and there its batches are faster than those of
direct multiplication from dimension 16 on, but above 64 it runs a table-driven
kernel whose sign corrections need index lookups and gathered loads that are not
vectorized within one product.

## Complex coefficients and finite fields

Over the complex numbers and finite fields the choice is only between direct and
quasilinear multiplication, but where the quasilinear product overtakes direct
multiplication cannot be read off from the real results, because an operation on
complex or finite-field coefficients costs differently from an operation on real
ones, and finite-field operations cost differently in every field. The domains
protocol therefore compares the two algorithms over the complex numbers and over
nine finite fields: the prime fields F_3, F_5, F_13, F_65521, and F_2147483647,
and the extension fields F_(3^5), F_(5^2), F_(13^5), and F_(65521^2). A
finite-field coefficient is stored as `deg` residues in the polynomial basis of
a deterministic irreducible polynomial, which the run records, and every pair is
verified before timing, finite-field products against independent schoolbook
polynomial arithmetic and complex products against long-double arithmetic with a
norm-scaled tolerance.

The first table gives, for single products, the paired ratio of the direct time
to the quasilinear time, so that a ratio above 1 favors quasilinear
multiplication, and an asterisk marks a ratio whose 95% interval contains 1:

| Field | N = 2 | N = 4 | N = 8 | N = 16 | N = 32 | N = 64 | N = 128 | N = 256 | N = 512 | N = 1024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F_3 | 0.72 | 0.94 | 1.55 | 1.53 | 2.08 | 2.38 | 2.23 | 2.76 | 7.33 | 17.50 |
| F_5 | 0.72 | 0.94 | 1.50 | 1.52 | 2.08 | 2.21 | 2.16 | 6.59 | 16.98 | 33.08 |
| F_13 | 0.72 | 0.95 | 1.43 | 1.53 | 2.02 | 2.19 | 2.73 | 7.26 | 17.72 | 35.05 |
| F_65521 | 0.74 | 0.96 | 1.38 | 1.54 | 2.10 | 2.27 | 5.38 | 10.15 | 18.85 | 36.58 |
| F_2147483647 | 0.73 | 0.95 | 1.39 | 1.50 | 2.02 | 2.20 | 5.32 | 10.41 | 18.83 | 37.68 |
| F_(3^5) | 0.95 | 0.99* | 1.12 | 1.08 | 1.06 | 0.98* | 1.10 | 2.23 | 4.43 | 8.58 |
| F_(5^2) | 0.91 | 1.01 | 1.18 | 1.27 | 1.86 | 4.17 | 8.31 | 15.61 | 29.75 | 50.35 |
| F_(13^5) | 0.97 | 0.99 | 1.10 | 1.20 | 3.01 | 6.18 | 12.08 | 21.79 | 39.37 | 70.77 |
| F_(65521^2) | 0.99* | 1.06 | 1.18 | 1.09 | 1.96 | 3.94 | 7.45 | 14.30 | 25.85 | 45.93 |
| C | 0.84 | 0.98 | 1.16 | 1.72 | 2.91 | 4.94 | 8.72 | 15.25 | 26.01 | 43.84 |

The second table gives the time of a single quasilinear product, in nanoseconds:

| Field | N = 16 | N = 64 | N = 256 | N = 1024 |
| --- | ---: | ---: | ---: | ---: |
| F_3 | 303.1 | 729.5 | 950.0 | 2202.4 |
| F_5 | 302.1 | 837.0 | 1222.4 | 3955.0 |
| F_13 | 303.8 | 844.7 | 1479.5 | 5001.4 |
| F_65521 | 299.1 | 804.6 | 2942.7 | 12742.1 |
| F_2147483647 | 298.4 | 849.2 | 2698.7 | 12466.6 |
| F_(3^5) | 1053.9 | 1715.1 | 5265.2 | 18651.3 |
| F_(5^2) | 570.0 | 1885.8 | 7405.6 | 35498.8 |
| F_(13^5) | 3443.6 | 4611.2 | 19779.3 | 96222.8 |
| F_(65521^2) | 1231.4 | 2828.1 | 11629.4 | 56283.8 |
| C | 312.9 | 1704.3 | 8701.9 | 48752.4 |

**Crossover.** Direct multiplication is faster at dimension 2 in every cell
except single products over F_(65521^2), where the two tie. At dimension 4 the
picture is mixed: direct multiplication is faster over C and the five prime
fields, the two tie over F_(3^5), direct is about 1% faster for single products
over F_(13^5) and ties for batches, and quasilinear multiplication is 1% to 6%
faster over F_(5^2) and F_(65521^2). From dimension 8 on, quasilinear
multiplication is faster in every cell except two ties, F_(3^5) at dimension 64
and batches over F_(65521^2) at dimension 16.

**Why the gains differ between fields.** At dimension 1024 the quasilinear
product is between 8.6 times (F_(3^5)) and 70.8 times (F_(13^5)) faster than
direct multiplication. The spread comes from the coefficient arithmetic, whose
polynomial multiplications, reductions, and changes of representation cost
different amounts at each characteristic and degree, so that the operation
counts of [mathematics.md](mathematics.md#operation-counts), which count field
operations, do not translate into the same speedup in every field.

**Batches and single products.** From dimension 16 on, the batch and single
times of each algorithm differ by at most 8%, the most over C at dimension 16,
where batching saves 8%, so the tables of single products also describe batches.
The archive's `domains.md` gives both modes at every dimension, with the
confidence intervals.

## Comparison with FLINT

This comparison provides the evidence for the second choice described in the
introduction: that fastCD implements its own finite-field arithmetic instead of
depending on FLINT, the C library for number theory and exact arithmetic that it
would otherwise use for prime fields and their extensions. The comparison of
fastCD's algorithms in the previous section cannot provide this evidence,
because all of them run on that same arithmetic, so this protocol times fastCD's
arithmetic against FLINT's directly. As in the rest of the benchmark, the
question is narrow: whether the arithmetic developed for fastCD is fast enough
to justify itself, not whether it is faster than FLINT's in general.

FLINT has no Cayley-Dickson multiplication, so the comparison measures the
arithmetic underneath it, in two workloads. The *scalar workload* times one
field multiplication, `fastcd_ff_field_mul` against FLINT's `nmod_mul` for prime
fields or `fq_nmod_mul` for extension fields, and so isolates the coefficient
arithmetic. The *CD workload* times one complete product: fastCD's quasilinear
product against the same quasilinear schedule (`benchmarks/flint_schedule.h`)
evaluated with FLINT's field operations, so that both sides perform the same
algorithm with the same basis convention and the same defining polynomial, and
the ratio compares their coefficient arithmetic, not two algorithms.

Both implementations receive identical canonical operands, and the extension
fields use FLINT's polynomial-basis representation `fq_nmod`, not Zech
logarithms. Conversion to FLINT objects, the construction of contexts, corpus
input and output, allocation, warmup, and verification are outside the timers,
while the work inside FLINT's arithmetic routines is inside them, and FLINT is
restricted to one thread. Each observation replays the full corpus with a pass
count common to both implementations, calibrated so that even the faster one
reaches the 20 ms target, and the two alternate their positions across the 24
paired blocks. An independent schoolbook implementation checks every pair before
timing and every final timed output afterwards. Besides the nine fields of the
domains protocol, the comparison includes three fields of degree 12, F_(3^12),
F_(13^12), and F_(2147483647^12), and FLINT is version 3.6.0. The ratios are
paired geometric means of the FLINT time over the fastCD time, so a ratio above
1 favors fastCD.

**Scalar field multiplication** (ns per product)

| Field | fastCD | FLINT | FLINT/fastCD [95% CI] |
| --- | ---: | ---: | --- |
| F_3 | 1.97 | 6.88 | 3.50 [3.44, 3.56] |
| F_5 | 1.95 | 6.88 | 3.54 [3.49, 3.58] |
| F_13 | 1.92 | 6.79 | 3.53 [3.49, 3.58] |
| F_(3^5) | 52.90 | 151.49 | 2.86 [2.83, 2.89] |
| F_(5^2) | 9.79 | 60.89 | 6.22 [6.12, 6.32] |
| F_(13^5) | 60.02 | 252.10 | 4.20 [4.15, 4.25] |
| F_65521 | 1.91 | 6.79 | 3.55 [3.50, 3.60] |
| F_(65521^2) | 9.94 | 69.75 | 7.02 [6.90, 7.14] |
| F_2147483647 | 1.99 | 6.95 | 3.49 [3.46, 3.52] |
| F_(3^12) | 256.75 | 605.46 | 2.36 [2.33, 2.38] |
| F_(13^12) | 333.59 | 655.09 | 1.96 [1.95, 1.98] |
| F_(2147483647^12) | 345.47 | 1274.63 | 3.69 [3.63, 3.75] |

**Complete CD multiplication:** fastCD ns / FLINT ns (FLINT/fastCD)

| Field | N = 16 | N = 64 | N = 256 | N = 1024 |
| --- | ---: | ---: | ---: | ---: |
| F_3 | 303 / 2304 (7.6x) | 786 / 17361 (22.2x) | 887 / 103072 (116x) | 2246 / 545555 (243x) |
| F_5 | 299 / 2471 (8.3x) | 796 / 18553 (23.4x) | 1182 / 108894 (92.2x) | 3849 / 576029 (150x) |
| F_13 | 296 / 2567 (8.7x) | 751 / 19130 (25.6x) | 1483 / 113451 (76.7x) | 5004 / 600521 (120x) |
| F_(3^5) | 1029 / 37610 (36.6x) | 1688 / 276146 (164x) | 5628 / 1614466 (288x) | 18632 / 8654929 (465x) |
| F_(5^2) | 562 / 17677 (31.5x) | 1841 / 133960 (72.9x) | 7191 / 797087 (111x) | 37879 / 4215793 (112x) |
| F_(13^5) | 3524 / 51723 (14.8x) | 4448 / 371973 (83.7x) | 20086 / 2195058 (110x) | 93398 / 11414018 (122x) |
| F_65521 | 303 / 2576 (8.5x) | 787 / 19267 (24.6x) | 2547 / 113299 (44.6x) | 12292 / 601869 (49.2x) |
| F_(65521^2) | 1266 / 18949 (15.0x) | 2831 / 129650 (46.1x) | 11412 / 760777 (67.1x) | 55952 / 4096556 (73.4x) |
| F_2147483647 | 300 / 2620 (8.7x) | 791 / 19541 (24.7x) | 2736 / 115812 (42.5x) | 12658 / 613825 (48.6x) |
| F_(3^12) | 4733 / 122398 (25.9x) | 4794 / 876796 (183x) | 17578 / 5104046 (291x) | 72578 / 26821623 (370x) |
| F_(13^12) | 60584 / 127766 (2.1x) | 43241 / 921304 (21.3x) | 96070 / 5211185 (54.3x) | 354900 / 27519067 (77.6x) |
| F_(2147483647^12) | 14639 / 237063 (16.3x) | 29288 / 1694677 (58.0x) | 100696 / 9885669 (98.5x) | 460726 / 52064812 (113x) |

Both workloads favor fastCD's arithmetic in every field and at every dimension
measured, which is the evidence for implementing that arithmetic in fastCD: its
scalar multiplication is between 1.96 and 7.02 times as fast as FLINT's in these
fields, and its complete products are between 2.1 and 465 times as fast at the
dimensions shown, with the ratio growing with the dimension in every field. The
complete products gain more than the scalar multiplications because they also
include fastCD's specialization by degree, its SIMD packing of coefficients, and
the field-specific maps that split extension-field products into prime-field
products, as well as differences in data layout and fastCD's validation of
buffers, so the CD ratios do not isolate the contribution of scalar
multiplication. The archive's `flint.md` gives every dimension and the
confidence intervals of the CD cells.

## Fused sums

Many computations need the sum of the products of many pairs of elements rather
than the products themselves, a covariance of samples of elements being one
example. Computed as a batch of products followed by a sum, such a sum stores
every product and spends the output additions of the multiplication algorithm
once for every product, whereas a fused sum accumulates the multiplications of
the algorithm over all the pairs and spends its output additions only once (see
[Fused sums of products](mathematics.md#fused-sums-of-products)). The sums
protocol measures how much time this saves, which algorithm a fused sum should
use at each dimension, and how a fused sum compares with a matrix product of the
cross moments of the samples, another way to obtain the same sum with a
general-purpose linear-algebra library.

The sums protocol cuts a corpus of standard normal real-coefficient elements
into *samples* of a given size and times four ways of computing the sum of the
products of each sample:

- **Products:** `fastcd_mul_batch_f64` on the sample, followed by a sum over the
  products, which is what a program without the fused sums would call.
- **Records:** `fastcd_mul_sum_f64`, the fused sum of the elements as stored,
  one record after another.
- **Planes:** `fastcd_mul_sum_planes_f64`, the same fused sum of elements stored
  as coordinate planes, in which each coordinate of the sample is one contiguous
  row.
- **Matrix product:** the matrix of real cross moments
  `G[i][j] = sum_p x_p[i] y_p[j]`, computed by one `dgemm`, and contracted with
  the basis signs. fastCD contains no matrix product, so the runner loads
  `cblas_dgemm` at run time from a shared BLAS library, in the archived run
  Intel MKL 2026.2.1 with one thread.

Times are per element. A cell has 8,192 elements in samples of 64, 1,024, or
4,096 elements and is timed in 60 blocks. The first sample of every cell is
verified against a long-double contraction of the cross moments, every sample
must give the same sum in every task within a norm-scaled bound, and the planes
must return the same bits as the records wherever both run the same algorithm.
The following table gives the means with the AVX-512 kernel for samples of 1,024
elements, and each ratio divides the time of the first task by that of the
second, so a ratio above 1 favors the second:

| N | Products | Records | Planes | `dgemm` | Products/records | `dgemm`/records | `dgemm`/planes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 4.14 | 2.33 | 0.64 | 1.52 | 1.77 | 0.65 | 2.36 |
| 4 | 7.02 | 2.53 | 1.35 | 2.13 | 2.79 | 0.84 | 1.58 |
| 8 | 16.33 | 5.42 | 3.19 | 3.44 | 3.01 | 0.63 | 1.08 |
| 16 | 59.29 | 17.59 | 16.02 | 14.20 | 3.37 | 0.81 | 0.89 |
| 32 | 201.84 | 43.63 | 40.03 | 48.29 | 4.64 | 1.11 | 1.21 |
| 64 | 515.56 | 108.01 | 104.81 | 155.17 | 4.78 | 1.44 | 1.48 |
| 128 | 1300.28 | 265.89 | 286.14 | 734.73 | 4.90 | 2.75 | 2.55 |
| 256 | 3032.47 | 704.42 | 1049.73 | 2147.15 | 4.33 | 3.05 | 2.04 |
| 512 | 7109.87 | 1774.34 | 2638.46 | 7900.96 | 4.02 | 4.45 | 2.99 |
| 1024 | 16834.66 | 4991.62 | 6348.93 | 28766.20 | 3.38 | 5.78 | 4.53 |

**Against a batch of products.** With the AVX-512 kernel the fused sum of
records is between 1.6 and 5.1 times as fast as a batch of products followed by
a sum, at every dimension and with all three sample sizes.

**Against the matrix product.** With samples of 1,024 or 4,096 elements, the
fused sum of records is slower than the matrix product at dimensions 2 through
16, while with samples of 64 elements it is faster at dimensions 2 and 4 and
within 6% at 8 and 16. From dimension 32 on it is faster with every sample size:
1.1 to 1.3 times at dimension 32, 1.4 to 1.7 times at 64, and 5.7 to 8.4 times
at 1024.

**Layouts.** Through dimension 64 the planes are faster than the records, which
the kernels must first rewrite into planes, but from dimension 128 on, with
samples of 1,024 or 4,096 elements, they are slower, because with 8,192 elements
per cell the rows of a sample lie on separate pages, while the records are read
as one stream. With a single sample of 1,024 elements, which keeps the operands
in cache through dimension 32, the two layouts are within 5% of each other from
dimension 128 through 512, and the planes take 17% longer at 1024
(`sums-one-sample.md` in the archive).

**Kernels.** The library chooses the kernel of the fused sums at run time, by
the features of the processor, so the speed of a fused sum depends on the
processor. With the AVX2 kernel the fused sum of records is 1.67 to 2.52 times
as fast as a batch of products at dimensions 8 through 64, against 2.73 to 4.86
times with the AVX-512 kernel.

## How the results set the defaults

For real products, `AUTO` takes its choices from the tables
`auto_single_by_level` and `auto_batch_by_level` in `src/fastcd.c`, which the C
and Python interfaces share, and each choice is the fastest mean of its cell in
the archived run:

| N | Single products | Batches |
| ---: | --- | --- |
| 2 | quasilinear | quasilinear |
| 4 | direct | quasilinear |
| 8 | direct | direct |
| 16 | direct | Cariow-Cariowa |
| 32 and above | quasilinear | quasilinear |

The batch table describes builds with the two-lane SIMD kernels, in which the
rankings of batches were measured; builds without them compute batches one
product at a time with the single-product kernels and therefore use the
single-product table for batches as well.

Over the complex numbers and finite fields, `AUTO` uses direct multiplication
through dimension 4 and quasilinear multiplication above, in both modes and for
every field. This single rule matches the fastest algorithm in every cell except
at dimension 4 over F_(5^2) and F_(65521^2), where quasilinear multiplication is
1% to 6% faster, a margin within the 8% by which direct/quasilinear ratios
varied between repeated runs.

Fused sums use the direct algorithm through dimension 8 and the quasilinear
algorithm above, with every kernel. The choice at dimension 16 rests on all
three kernels, since the ratio of direct to quasilinear time there depends on
the kernel (a ratio above 1 favors quasilinear):

| Kernel | Direct/quasilinear, N = 8 | Direct/quasilinear, N = 16 |
| --- | --- | --- |
| AVX-512 | 0.72 to 0.86 | 0.84 to 0.96 |
| AVX2 | 0.83 to 0.91 | 0.97 to 1.10 |
| Portable | 0.89 to 0.91 | 1.76 to 1.78 |

The ranges are over the three sample sizes. At dimension 8 the direct sum is
faster with every kernel, while at dimension 16 it is faster only with the
AVX-512 kernel, by 4% to 16%, the two are level with AVX2, and the direct sum
takes 1.8 times as long with the portable kernel, so the quasilinear sum is the
choice that serves every kernel, at the cost of that margin for AVX-512. Above
dimension 16 the direct sum is a plain loop over the `N^2` products, and at
dimension 32 it takes 11 times as long as the quasilinear sum with AVX-512.

These tables reflect one run on one processor, so they should be reviewed
against the results of any new run, especially on a different kind of processor.

## Limitations

- All results come from one shared Skylake-family host with an unpinned clock
  frequency, built with one compiler. Other processors and compilers can rank
  the algorithms differently where the margins are small.
- The library is measured as it ships, so the placement of its own code is part
  of each timing. Where two algorithms are within a few percent of each other,
  as for single products at dimensions 2 and 4, the ranking can change with an
  unrelated edit to the library.
- The domains and FLINT protocols used 1,000 pairs per cell, and their intervals
  cover only the variation within one process.
- The finite fields form a representative selection, not a survey of all primes
  or all irreducible polynomials, and the comparison with FLINT covers only its
  polynomial-basis representation of extension fields.
- The matrix product in the sums protocol comes from one BLAS library running on
  one thread, and only the AVX-512 kernel was compared with it.

## Running the benchmark

### The real protocol

To build fastCD and run the real protocol on a GNU/Linux system:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
cmake --build build --parallel
python3 benchmarks/benchmark.py --runner build/fastcd_benchmark \
    --output benchmark-results/results-current.json
```

The driver requires NumPy. It writes a JSON file with the raw observations, the
corpus hashes, the hashes of the source files and executables, the build
options, and the state of the CPU (the frequency governor, turbo or boost, and
SMT, where the kernel exposes them), and a Markdown report beside it. It never
overwrites an existing report, so each run needs a new output path, and after
each dimension it saves the results so far as `<name>.partial.json`, which it
removes when the run completes. Rebuild before running, since CMake writes the
build record to `fastcd-build-Release.txt` beside the runner and the driver
refuses to run if the sources have changed since; for a multi-configuration
build, `--build-metadata` selects the record.

On Linux the driver and its runner processes stay on the first allowed CPU
unless `--cpu` selects another, and a full run takes about forty minutes on a
server core, most of it spent on the two quadratic algorithms and the
verification at dimension 1024. `--modes batch` or `--modes single` restricts
the run to one mode, and `--max-cc-dimension 64` stops timing Cariow-Cariowa
above dimension 64, in which case the report marks those cells "not timed", as
opposed to "unavailable" where the library does not support an algorithm. The
three algorithms in two modes make six tasks, or four when one algorithm is not
timed, so a block count that is a multiple of 12, such as the default 24, keeps
every task in every position equally often in both cases. If a check fails, the
driver still writes both reports, marks the run and each failing algorithm with
the status `verification_failure`, together with the mismatch counts and the
largest errors, and exits with a nonzero status.

`benchmarks/run_dedicated.sh` performs a complete run on a dedicated machine: a
fresh Release build, the C test suite, and the full benchmark into a timestamped
file under `benchmark-results/`, optionally followed by the same steps for a
build with `FASTCD_CC_DOUBLING=1` (`--cc-doubling`).

| Option | Default | Meaning |
| --- | --- | --- |
| `--dimensions` | `2,...,1024` | comma-separated powers of two to time |
| `--trials` | `10000` | pairs per dimension |
| `--blocks` | `24` | timing blocks; each yields one observation per task |
| `--warmup` | `2` | untimed corpus passes before each observation |
| `--min-time-ms` | `20` | calibration target per observation |
| `--seed` | `20260906` | master corpus seed |
| `--norm-min`, `--norm-max` | `0.5`, `2.0` | operand norm range |
| `--corpora` | `benchmark-corpora` | directory of `corpus-nN.f64` files |
| `--max-direct-dimension` | `1024` | largest dimension at which direct is timed |
| `--max-cc-dimension` | `1024` | largest dimension at which Cariow-Cariowa is timed |
| `--verify-trials` | all | pairs verified in the first block |
| `--cpu` | first allowed | logical CPU to pin (Linux) |
| `--subprocess-timeout` | `3600` | seconds allowed for one runner invocation |
| `--build-metadata` | beside runner | build record for a multi-configuration build |

### The domains protocol

```sh
python3 benchmarks/benchmark.py --protocol domains \
    --runner build/fastcd_domains_benchmark \
    --output benchmark-results/domains-current.json
```

`--fields char:deg,...` chooses the fields, where `0:2` denotes C, the
characteristic is an odd prime at most 2147483647, and the degree is 1 through
12; for example, `--fields 13:5,0:2 --dimensions 8,32,128` selects two
coefficient fields and three dimensions. The corpora are separate from those of
the real protocol, and a manifest that does not match an existing corpus file is
rejected. The two algorithms in two modes make four tasks, and the block count
must be a multiple of the task count.

The default of 10,000 pairs per cell makes this protocol long, because direct
multiplication and the verification both perform `O(N^2)` coefficient products
per pair, and the verification computes each extension-field product with about
`deg^2` residue products, so the largest cells dominate the run time. The
archived run used `--trials 1000` and took 54 minutes, 15 of them in its largest
cell, F_(13^5) at dimension 1024, and the default would take roughly ten times
as long. Each cell runs in its own runner process with a one-hour limit, which a
10,000-pair cell at dimension 1024 over an extension field can exceed, so such
runs need a larger `--subprocess-timeout`, in seconds; a smaller `--trials`
gives a quick pilot, whose results should be read with that value in mind.

`benchmarks/domains.py` accepts the same options when run directly. For complex
coefficients, `--fields 0:2 --baseline` also times a quasilinear implementation
that splits each complex product into three real products, with packing and
reconstruction inside its timer; this adds a third algorithm, so the block count
must then be a multiple of six.

### The FLINT protocol

Install FLINT's development headers and library through your package manager or
the [FLINT installation instructions](https://flintlib.org/doc/install.html),
then build and run the comparison runner:

```sh
cmake -S . -B build-flint -DCMAKE_BUILD_TYPE=Release \
    -DFASTCD_BUILD_FLINT_BENCHMARK=ON
cmake --build build-flint --parallel
ctest --test-dir build-flint --output-on-failure
python3 benchmarks/benchmark.py --protocol flint \
    --runner build-flint/fastcd_flint_benchmark \
    --output benchmark-results/flint-current.json
```

For a nonstandard installation, add `-DCMAKE_PREFIX_PATH=/path/to/flint` or
expose its `flint.pc` through `PKG_CONFIG_PATH`. Only the comparison runner
links FLINT: the library, the installed headers, and the Python package do not
depend on it, and a missing FLINT is reported only when
`FASTCD_BUILD_FLINT_BENCHMARK=ON`. CTest checks every supported extension degree
against the independent oracle, together with the workspace and overlap errors
of the FLINT adapters. The driver checks that the compile-time and run-time
FLINT versions match, and CMake records the hashes of the linked FLINT library
and headers, but neither can recover how an externally installed FLINT was
compiled, so FLINT's own build options should be recorded alongside published
results.

`--workloads scalar` or `--workloads cd` selects one workload, and `--fields`
accepts any supported characteristic and degree, for example
`3:12,13:5,2147483647:12`; scalar multiplication is measured once per field, and
CD multiplication at the selected `--dimensions`. The comparison makes one call
per product, because its adapters have no batch entry, and the run-time notes of
the domains protocol apply here as well: the cells of degree 12 at dimension
1024, where FLINT's extension arithmetic is slowest, took more than an hour each
with 1,000 pairs in the archived run, so they need a larger
`--subprocess-timeout`.

### The sums protocol

```sh
python3 benchmarks/benchmark.py --protocol sums \
    --runner build/fastcd_sums_benchmark \
    --blas /path/to/libmkl_rt.so --blas-name "Intel MKL" \
    --output benchmark-results/sums-current.json
```

`--blas` names the shared BLAS library from which the matrix product is loaded,
and without it that task is left out; `--blas-symbol` and `--blas-ilp64` select
another entry point or 64-bit integer arguments, and the report records the path
and hash of the library. `--counts` takes several sample sizes, each of which
must divide `--trials`, and since a cell has three to five tasks, the block
count must be a multiple of the number of tasks at every dimension, which the
default of 60 is for every combination. The direct sum has `N^2` multiplications
per element and is timed through `--max-direct-dimension`, 64 by default and at
least 8. With the default 8,192 elements the operands of a cell do not fit the
L2 cache from dimension 16 on, as in a pass over a large array, while
`--trials 1024` with one sample of 1,024 elements keeps them in cache through
dimension 32; the two settings can rank the tasks differently at low dimensions,
where loading the operands takes as long as the arithmetic.

`benchmarks/sums.py` accepts the same options when run directly, and the
protocol is not part of `--protocol all`. Since the library chooses the kernel
of the fused sums at run time, one build measures one kernel on a given
processor; to measure another, configure a second build with
`-DFASTCD_TUNING_DEFINITIONS=FASTCD_SUM_MAX_KERNEL=2`, which keeps the AVX-512
kernel from being selected, or with `-DFASTCD_ENABLE_SUM_SIMD=OFF`, which leaves
only the portable kernel.

### All protocols

`--protocol all` runs the real, domains, and FLINT protocols one after another
from one build:

```sh
python3 benchmarks/benchmark.py --protocol all \
    --runner build-flint/fastcd_benchmark \
    --output benchmark-results/all-current.json
```

`--runner` names `fastcd_benchmark`, and the other runners are found beside it
or named with `--domain-runner` and `--flint-runner`, while
`--domain-build-metadata` and `--flint-build-metadata` select their build
records in a multi-configuration build. Before timing starts, the driver checks
every runner, build record, and required dependency. With
`--output results.json` it writes `results.real.json`, `results.domains.json`,
and `results.flint.json`, each with its own Markdown report, and `results.json`
and `results.md` index them; existing files are never overwritten, and results
from different protocols are never combined into one statistic. The corpus and
timing options apply to every protocol, `--modes` applies to the real and
domains protocols, and `--fields` selects the domains fields and, through its
finite subset, the FLINT fields, so at least one finite field is required. Each
protocol's corpora go into their own subdirectory of `--corpora`.

### Build options that change what is measured

Two build options, both passed through `-DFASTCD_TUNING_DEFINITIONS=...` at
configure time, change the kernels being measured. `FASTCD_DISABLE_SIMD=1`
removes the explicit SIMD kernels but not the compiler's own vectorization, so a
scalar-only comparison with GCC also needs `-fno-tree-vectorize` in
`CMAKE_C_FLAGS`. `FASTCD_CC_DOUBLING=1` evaluates Cariow-Cariowa products at
dimensions 16 through 64 as the four half-dimensional products of one
Cayley-Dickson doubling step, two per SIMD lane; this is a different arithmetic
schedule from the fixed-dimension procedure of Cariow and Cariowa, so it is off
by default and its results should be reported separately.
