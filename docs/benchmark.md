# Benchmark

fastCD multiplies real coefficients with three algorithms (direct,
Cariow–Cariowa and quasilinear) and complex and finite-field coefficients
with two (direct and quasilinear). `AUTO` chooses among them by dimension.
The benchmark measures the algorithms as the library ships them, so that
these choices rest on reproducible data. A third experiment compares
fastCD's finite-field arithmetic with FLINT (Fast Library for Number
Theory), an independent exact-arithmetic C library. The driver
`benchmarks/benchmark.py` runs each experiment as a *protocol*:

| `--protocol` | Measurements | Native runner |
| --- | --- | --- |
| `real` (default) | The three algorithms over R, single and batch | `fastcd_benchmark` |
| `domains` | Direct vs quasilinear over C and odd finite fields | `fastcd_domains_benchmark` |
| `flint` | Scalar field multiplication and complete CD products, fastCD vs FLINT | `fastcd_flint_benchmark` |
| `sums` | Fused sums of products against a batch of products and a matrix product | `fastcd_sums_benchmark` |
| `all` | The real, domains and FLINT protocols, run one after another with separate reports | Those three runners |

This page first describes how the protocols measure and report, which they
largely share, then each protocol and how to run it, and finally the
archived results that the README reports. The accuracy and cost of balanced
accumulation, measured with the same runners, are covered by the
[stability study](stability.md).

## How measurements are made

### Corpora and verification

Every protocol times a fixed *corpus*: a deterministic set of operand pairs
for each dimension, and for each field where there is one. The driver
generates it with NumPy from a recorded seed, stores it for reuse, and binds
it to a manifest with its parameters and hash, so a run never times inputs
different from the ones it reports. Each product is also checked against an
independent implementation, outside the timers; the protocol sections below
name the reference each one uses. A failed check is reported as a failure,
never as a timing.

### Observations, modes and blocks

An *observation* times complete passes over the corpus. After two untimed
warmup passes, the runner calibrates the number of passes so that the
observation lasts at least 20 ms, and divides the elapsed time by the number
of products. Storage is allocated beforehand and only native calls are
timed, so allocation, verification and Python overhead are excluded.

Products are timed in two *modes*. Batch mode makes one validated call for
the whole corpus in each pass, and single mode makes one validated call per
product. Single mode therefore includes the per-call validation, about
15 ns on the archived Xeon runs, which dominates below dimension 8 and is
why the two modes can rank the algorithms differently.

A run is divided into 24 *blocks*. Each block takes one observation of every
*task*, that is, of every algorithm and mode, or of every implementation in
the FLINT comparison, and the order of the tasks rotates from block to block
so that each task holds each position equally often. Ratios are formed
within a block, so a slow change in the state of the host affects both
sides of a ratio.

### Keeping the harness out of the results

A timing can depend on where the runner's own code and stack happen to lie
in memory, and the runners are built to keep that from deciding results.
The real protocol starts a new runner process for every observation, so
each observation gets its own memory placement. The domains and FLINT
protocols run all blocks of a cell in one process and give each block its
own stack offset, spread evenly in 16-byte steps over a 4 KB page: some
finite-field kernels run markedly slower at a few stack positions, and one
position shared by a whole cell would shift all its observations together.

Where the compiler supports it, CMake also builds the three runners, but not
the library, with branch padding (`-mbranches-within-32B-boundaries`), so
that no branch in a timing loop straddles a 32-byte boundary. On
Skylake-family processors such a branch runs markedly slower (Intel's JCC
erratum), and an unrelated edit to a runner could otherwise move a loop onto
a boundary and change its timings. The build record lists the option as
`runner_options`.

### Reading the reports

Each run writes a JSON file with the raw observations, the corpus hashes,
the hashes of the source files and executables, the build options and the
state of the CPU, and a Markdown report beside it. A time is the mean over
the blocks, in nanoseconds per product, with a two-sided 95% Student-t
interval whose degrees of freedom are capped at 30. A ratio is the
geometric mean of the per-block ratios with its 95% interval; each protocol
section says which ratio it reports and which direction favors which side.
The real report's "fastest mean" column names the lowest mean, or a tie when
the interval of the paired ratio between the two lowest means contains one.
The fraction of an interval below one is not a probability that one
algorithm is faster.

An interval describes the variation between the blocks of one run. In the
real protocol every observation has its own process, so the interval also
covers differences between processes. In the domains and FLINT protocols a
cell keeps one process, and with it one placement in physical memory, which
affects cache behavior once a cell no longer fits in the per-core L2 cache.
In repeated runs on the archive's host, individual finite-field cells from
dimension 64 upward differed by up to 17%, and direct/quasilinear ratios by
up to 8%, so small differences between cells should be read with this in
mind. The recorded CPU state (frequency governor, turbo and boost settings
and SMT, where the kernel exposes them) shows whether two runs had
comparable machine states.

## Real coefficients

### What is measured

The real protocol times direct, Cariow–Cariowa and quasilinear
multiplication with the kernels the library ships. With the default SIMD
configuration on supported GCC and Clang targets, batch mode evaluates two
products per SIMD instruction, for quasilinear and Cariow–Cariowa at
dimensions 16 through 512 and for direct at 64 through 512, and uses scalar
kernels elsewhere. From dimension 32, a single quasilinear product evaluates
the two recursive calls of `C_(n-1)` in the two lanes (see
[How quasilinear multiplication works](mathematics.md#how-quasilinear-multiplication-works)).
Single direct and Cariow–Cariowa products run scalar code that the compiler
may vectorize.

Cariow–Cariowa shows that operation counts do not decide running time.
Through dimension 64 it runs generated straight-line kernels; above 64 it
runs a table-driven kernel whose sign corrections need index lookups and
gathered loads that are not vectorized within one product. At dimensions
128 through 1024 it needs only 0.51 to 0.58 times the operations of direct
multiplication, yet in the archived run its single products are 1.6 to 1.8
times slower than direct. Its batches are 11% slower than direct at 128,
within 1% at 256 and 512, and 1.5 times slower at 1024. These cells measure
how the irregular corrections map onto the hardware, and the algorithm is
not recommended above 64. It is timed at every dimension the library
supports, through 1024; `--max-cc-dimension 64` stops timing it there, and
the report then marks the cells "not timed", as opposed to "unavailable"
where the library does not support an algorithm. Above dimension 256 the
kernel combines the block totals of each correction sum with a balanced
tree (see [Accumulation order](mathematics.md#accumulation-order)).

Two build options change what is measured, and both are passed through
`-DFASTCD_TUNING_DEFINITIONS=...` at configure time.
`FASTCD_DISABLE_SIMD=1` removes the explicit SIMD kernels but not the
compiler's own vectorization; add `-fno-tree-vectorize` to `CMAKE_C_FLAGS`
for a scalar-only comparison with GCC. `FASTCD_CC_DOUBLING=1` evaluates
Cariow–Cariowa products at dimensions 16 through 64 as the four
half-dimensional products of one Cayley–Dickson doubling step, two per SIMD
lane. That is a different arithmetic schedule from the paper's
fixed-dimension procedure, so it is off by default and reported separately.
Measured with the real protocol on the archive's host, it makes single
products 1.25 to 1.9 times faster at those dimensions, while batches are 56%
slower at 16, 9% slower at 32 and 3% faster at 64.

### Protocol

The corpus has 10,000 pairs at each dimension `N = 2, 4, ..., 1024`,
generated by NumPy's PCG64 from `SeedSequence([20260906, N])`. Their
coordinates are independent uniform samples, normalized and scaled to give
norms in `[0.5, 2)`, and the corpora are kept in `benchmark-corpora/`.

The three algorithms in two modes make six tasks, shuffled once and then
rotated from block to block. Where one algorithm is not timed, four tasks
remain, so a block count that is a multiple of 12, such as the default 24,
keeps every task in every position equally often in both cases. In the
first block every product of every task is checked against an independent
implementation of the defining recursion; `--verify-trials` limits that to
a prefix of the corpus. If a check fails, the driver still writes both
reports, marks the run and each failing algorithm with the status
`verification_failure` together with the mismatch counts and largest errors,
and exits with a nonzero status.

### How the results set the defaults

`AUTO` takes its choices for real coefficients from the tables
`auto_single_by_level` and `auto_batch_by_level` at the start of the
algorithm-selection section of `src/fastcd.c`, which the C and Python
interfaces share. Single products use direct at dimensions 4, 8 and 16 and
quasilinear otherwise. Batches use direct at 8, Cariow–Cariowa at 16 and
quasilinear otherwise on builds with the two-lane SIMD kernels, and follow
the single-product table on other builds. These choices agree with every
cell of the archived run; dimension-2 single products are a tie between
direct and quasilinear, and quasilinear is chosen. Review the tables against
the results of any new run.

### Running it

To build fastCD and run the real protocol on a GNU/Linux system:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
cmake --build build --parallel
python3 benchmarks/benchmark.py --runner build/fastcd_benchmark \
    --output benchmark-results/results-current.json
```

The driver requires NumPy. It writes the JSON file and the Markdown report
beside it, and it never overwrites an existing report, so choose a new
output path for each run. After each dimension it saves the results so far
as `<name>.partial.json`, which it removes when the run completes. Rebuild
before running: CMake writes the build record to `fastcd-build-Release.txt`
beside the runner, and the driver refuses to run if the sources have changed
since; for a multi-configuration build, select the record with
`--build-metadata`.

On Linux the driver and its runner processes stay on the first allowed CPU
unless `--cpu` selects another. A full run takes about forty minutes on a
server core, most of it spent on the two quadratic algorithms and the
verification at `N = 1024`. `--modes batch` or `--modes single` restricts
the run to one mode. `benchmarks/run_dedicated.sh` performs a complete run
on a dedicated machine: a fresh Release build, the C test suite, and the
full benchmark into a timestamped file under `benchmark-results/`,
optionally followed by the same steps for the `FASTCD_CC_DOUBLING=1` build
(`--cc-doubling`).

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
| `--max-cc-dimension` | `1024` | largest dimension at which Cariow–Cariowa is timed |
| `--verify-trials` | all | pairs verified in the first block |
| `--cpu` | first allowed | logical CPU to pin (Linux) |
| `--subprocess-timeout` | `3600` | seconds allowed for one runner invocation |
| `--build-metadata` | beside runner | build record for a multi-configuration build |

## Complex and finite fields

The domains protocol compares direct and quasilinear multiplication over C
and over odd-characteristic finite fields, the two algorithms available for
these coefficients. Its results set their `AUTO` rule: direct for `N <= 4`
and quasilinear otherwise, for single products and batches alike. The
default fields are F_3, F_5, F_13, F_(3^5), F_(5^2), F_(13^5), F_65521,
F_(65521^2), F_2147483647 and C, at powers of two from 2 through 1024.
`--fields char:deg,...` chooses others: `0:2` denotes C, the characteristic
is an odd prime at most 2147483647, and the extension degree is 1 through
12. For example, `--fields 13:5,0:2 --dimensions 8,32,128` selects two
coefficient fields and three CD dimensions.

Each finite coefficient occupies `deg` canonical `uint32` residues in the
polynomial basis, and a record has N such coefficients; setup chooses and
validates a deterministic monic irreducible defining polynomial and records
it in the JSON. Complex records hold N `complex128` coefficients. These
corpora are separate from the real protocol's, and a manifest that does not
match an existing corpus file is rejected. Every pair is verified before
timing, finite products against independent schoolbook polynomial
arithmetic with signed CD basis accumulation, and complex products against
long-double accumulation with a norm-scaled tolerance.

The timings measure the shipped implementations rather than coefficient
operation counts. The quasilinear timings, and the direct timings from
dimension 16 on where a packed kernel serves the field, include the degree
specialization, the packed SIMD arithmetic and the field-specific maps that
each call builds to split extension-field products into prime-field
products. The report gives the paired direct/quasilinear ratio, so a ratio
above 1 favors quasilinear.

By default the protocol uses 10,000 pairs per field and dimension, with the
shared settings of 24 blocks, two warmup passes and a 20 ms target, and each
algorithm and mode calibrates its passes independently. The two algorithms
in two modes make four tasks, two with one mode, and the block count must be
a multiple of the task count. Direct multiplication and the verification
both perform O(N^2) coefficient products per pair, and the verification
computes each extension-field product with about `deg`^2 residue products,
so the largest cells dominate the run time. The archived run used
`--trials 1000` and took 54 minutes, 15 of them in its largest cell,
F_(13^5) at dimension 1024; the default 10,000 pairs would take roughly ten
times as long. Each cell runs in its own runner process with a one-hour
limit, which a 10,000-pair cell at dimension 1024 over an extension field
can exceed, so raise `--subprocess-timeout` (in seconds) for such runs. A
smaller `--trials` gives a quick pilot; keep that value in mind when
reading its results.

```sh
python3 benchmarks/benchmark.py --protocol domains \
    --runner build/fastcd_domains_benchmark \
    --output benchmark-results/domains-current.json
```

`benchmarks/domains.py` accepts the same options when run directly. For
complex coefficients, `--fields 0:2 --baseline` also times a quasilinear
implementation that splits each complex product into three real products,
with packing and reconstruction inside its timer; the baseline adds a third
algorithm, so the block count must then be a multiple of six.

## Comparison with FLINT

[FLINT](https://flintlib.org/) has no Cayley–Dickson multiplication, so the
FLINT protocol compares the arithmetic underneath it, in two separate
workloads:

* **Scalar field multiplication:** `fastcd_ff_field_mul` against FLINT's
  `nmod_mul` for prime fields or `fq_nmod_mul` for extensions. This
  isolates the coefficient arithmetic.
* **Complete CD multiplication:** fastCD's quasilinear product against the
  same quasilinear schedule (`benchmarks/flint_schedule.h`) evaluated with
  FLINT's field operations. Both multiply N field coefficients with the same
  basis convention and the same defining polynomial, so the ratio compares
  coefficient arithmetic, not algorithms.

Scalar multiplication is measured once per field, and CD multiplication at
the selected `--dimensions`. Both use the nine finite fields of the domains
protocol by default; C is not part of this protocol. `--workloads scalar` or
`--workloads cd` selects one workload, and `--fields` accepts any supported
characteristic and degree, for example `3:12,13:5,2147483647:12`. The default
grid is a representative experiment, not a survey of all primes or all
irreducible polynomials.

### Timing and verification

Both implementations receive identical canonical operand coefficients and
the same defining polynomial; the extension comparison uses FLINT's
`fq_nmod` polynomial-basis representation, not Zech logarithms. Conversion
to FLINT objects, context construction, corpus I/O, allocation, warmup and
verification are outside the timers, and the work inside FLINT's arithmetic
routines is inside them. FLINT is restricted to one thread. Each observation
replays the full corpus with a pass count common to both implementations,
calibrated so that even the faster one reaches the target, and the two
alternate their first and second positions across the 24 paired blocks.
Wall-clock times are primary, and thread CPU times are kept where the
platform provides them. An independent schoolbook polynomial implementation
checks every pair before timing and every final timed output afterwards;
mismatches, missing observations or changed build files prevent a
successful report, and partial JSON files keep the completed cells of an
interrupted run.

The report gives each implementation's mean time per product and the paired
**FLINT time / fastCD time**, so a ratio above 1 favors fastCD. Scalar
results isolate the coefficient multiply. Complete CD results also include
fastCD's degree specialization, SIMD packing and field-specific maps, the
differences in data layout and fastCD's buffer validation, so they do not
isolate the scalar multiply's contribution. Neither workload establishes a
universal speedup or covers every representation FLINT offers.

### Building and running

Install FLINT's development headers and library through your package
manager or the [FLINT installation instructions](https://flintlib.org/doc/install.html),
then build the comparison runner:

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
links FLINT; the library, the installed headers and the Python package do
not depend on it, and a missing FLINT is reported only when
`FASTCD_BUILD_FLINT_BENCHMARK=ON`. CTest checks every supported extension
degree against the independent oracle, together with the workspace and
overlap errors of the FLINT adapters. The driver checks that the compile-time
and runtime FLINT versions match, and CMake records the hashes of the linked
FLINT library and headers, but it cannot recover how an externally installed
FLINT was compiled, so record FLINT's own build options alongside published
results. The run-time notes of the domains protocol apply here as well, and
the degree-12 cells at dimension 1024, where FLINT's extension arithmetic is
slowest, are the most expensive: in the archived run each of them took more
than an hour with 1,000 pairs, so they need a larger `--subprocess-timeout`.

## Fused sums

The sums protocol measures the [fused sums](c-api.md#fused-sums) of
real-coefficient elements. Its corpus is a run of standard normal elements
cut into *samples* of `count` elements, and every task computes the sum of
the products of each sample:

| Task | Computation |
| --- | --- |
| `products` | `fastcd_mul_batch_f64` on the sample, then a sum over the products |
| `records` | `fastcd_mul_sum_f64` on the records |
| `planes_direct` | `fastcd_mul_sum_planes_with_algorithm_f64` with the direct algorithm |
| `planes_quasilinear` | the same with the quasilinear algorithm |
| `dgemm` | one `dgemm` for the matrix of real cross moments, then its contraction with the basis signs |

The first task is what a program without the fused sums would call. The two
plane tasks read each sample from coordinate planes, written before timing
with rows `count + 8` doubles apart, and they decide the algorithm that
`AUTO` uses for fused sums. The direct sum has `N^2` multiplications per
element and is timed through `--max-direct-dimension`, 64 by default and at
least 8, through which it is the automatic algorithm.

The `dgemm` task compares with the other way to obtain such a sum: the
matrix `G[i][j] = sum_p x_p[i] y_p[j]` is one matrix product of the records,
and the sum of products is `sum over i xor j = k of sigma(i, j) G[i][j]`.
fastCD contains no matrix product, so the runner loads one at run time from
the shared BLAS library named by `--blas`, calls its `cblas_dgemm` with one
thread, and applies the contraction itself, inside the timer. Without
`--blas` the task is left out. `--blas-symbol` and `--blas-ilp64` select
another entry point or 64-bit integer arguments, and the report records the
library's path and hash.

Times are per element. By default a cell has 8,192 elements in samples of
1,024, with 60 blocks, two warmup passes and a 20 ms target; `--counts`
takes several sample sizes, each of which must divide `--trials`. A cell has
three to five tasks, and the block count must be a multiple of the number of
tasks at every dimension, which 60 is for every combination. The first
sample of each cell is verified against a long double contraction of the
cross moments, and on every sample each task must agree with the fused sum
of records within the same norm-scaled bound; where the plane task runs the
algorithm that `AUTO` selects, it must return the same bits as the records.
The report gives the paired ratios products/records, products/planes,
dgemm/records, dgemm/planes and direct/quasilinear, each the time of the
first divided by that of the second.

With 8,192 elements the operands of a cell do not fit the L2 cache from
dimension 16 on, as in a pass over a large array. `--trials 1024` with one
sample of 1,024 keeps them in cache through dimension 32, and the two
settings can rank the tasks differently at low dimensions, where loading the
operands takes as long as the arithmetic.

```sh
python3 benchmarks/benchmark.py --protocol sums \
    --runner build/fastcd_sums_benchmark \
    --blas /path/to/libmkl_rt.so --blas-name "Intel MKL" \
    --output benchmark-results/sums-current.json
```

`benchmarks/sums.py` accepts the same options when run directly. The
protocol is not part of `--protocol all`.

The library chooses the kernel of the fused sums at run time, so one build
measures one kernel on a given processor. To measure another, configure a
second build with `-DFASTCD_TUNING_DEFINITIONS=FASTCD_SUM_MAX_KERNEL=2`,
which keeps the AVX-512 kernel from being selected, or with
`-DFASTCD_ENABLE_SUM_SIMD=OFF`, which leaves only the portable kernel.

## Running all protocols

`--protocol all` runs the real, domains and FLINT protocols one after another
from one build:

```sh
python3 benchmarks/benchmark.py --protocol all \
    --runner build-flint/fastcd_benchmark \
    --output benchmark-results/all-current.json
```

`--runner` names `fastcd_benchmark`; the other runners are found beside it,
or named with `--domain-runner` and `--flint-runner`, and
`--domain-build-metadata` and `--flint-build-metadata` select their build
records in a multi-configuration build. Before timing starts, the driver
checks every runner, build record and required dependency. With
`--output results.json` it writes `results.real.json`,
`results.domains.json` and `results.flint.json`, each with its own Markdown
report, and `results.json` and `results.md` index them; existing files are
never overwritten, and results from different protocols are never combined
into one statistic.

The corpus and timing options apply to every protocol. `--modes` applies to
the real and domains protocols; the FLINT comparison makes one call per
product, because its adapters have no batch entry, and the domains and FLINT
protocols always verify every pair. `--fields` selects the domains fields,
and its finite subset the FLINT fields, so at least one finite field is
required. Each protocol's corpora go into its own subdirectory of
`--corpora`.

## Archived results

The README tables come from the run archived in
[`benchmark-results/t440-20261109/`](../benchmark-results/t440-20261109/README.md),
measured on 2026-11-09 from commit `33addc9`, with each protocol pinned to
one logical CPU of an Intel Xeon Gold 6148 in a Dell PowerEdge T440.
`real.json` and `real.md` hold the real protocol with its default settings,
`domains.*` the domains protocol, and `flint.*` the comparison with FLINT
3.6.0, which adds F_(3^12), F_(13^12) and F_(2147483647^12) to the nine
default fields. The domains and FLINT protocols were measured with 1,000
pairs per cell instead of the default 10,000. Every check passed: 600,000
real products, 100,000 complex and finite-field pairs and 132,000
FLINT-comparison pairs. The archive's README records the host, the commands
and the load on the host during the run.

The fused sums were measured on the same host on 2026-11-20 and are
archived in
[`benchmark-results/sums-t440-20261120/`](../benchmark-results/sums-t440-20261120/README.md),
with Intel MKL 2026.2.1 as the BLAS library. Mean nanoseconds per element
with the AVX-512 kernel, for samples of 1,024 elements from a corpus of
8,192 elements per cell:

| N | Products | Records | Planes | `dgemm` | Products/records | `dgemm`/records | `dgemm`/planes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 3.96 | 2.22 | 0.59 | 1.43 | 1.78 | 0.65 | 2.43 |
| 4 | 6.70 | 2.45 | 1.26 | 1.97 | 2.74 | 0.81 | 1.56 |
| 8 | 16.00 | 5.37 | 3.16 | 3.36 | 2.98 | 0.63 | 1.06 |
| 16 | 56.47 | 17.16 | 15.40 | 13.74 | 3.29 | 0.80 | 0.89 |
| 32 | 203.72 | 44.41 | 40.94 | 48.41 | 4.60 | 1.09 | 1.18 |
| 64 | 498.82 | 103.91 | 100.34 | 144.93 | 4.81 | 1.39 | 1.44 |
| 128 | 1258.21 | 249.60 | 285.04 | 616.63 | 5.05 | 2.46 | 2.16 |
| 256 | 3059.31 | 735.90 | 1083.93 | 2208.61 | 4.17 | 3.00 | 2.03 |
| 512 | 7036.62 | 1808.07 | 2557.97 | 7954.76 | 3.91 | 4.40 | 3.10 |
| 1024 | 16419.67 | 4870.83 | 6051.27 | 28279.26 | 3.38 | 5.81 | 4.67 |

`Planes` is the plane task of the automatic algorithm. With this kernel a
fused sum of records was 1.5 to 5.1 times as fast as a batch of products
followed by a sum, over every dimension and the three sample sizes 64,
1,024 and 4,096. Against the matrix product, the records were slower at
dimensions 8 and 16 with samples of 1,024 or more (0.61 to 0.81) and level
with samples of 64; at dimensions 2 and 4 they were faster with samples of
64 (1.25 and 1.35) and slower with larger ones. From dimension 32 the
records were faster at every sample size: 1.1 to 1.3 at dimension 32, 1.4
to 1.7 at 64 and 5.6 to 8.2 at 1024.

From dimension 128 the planes were slower than the records in this
setting: with 8,192 elements the rows of a sample lie on separate pages,
while records are read as one stream. With a single sample of 1,024
elements, which stays in cache through dimension 32, the two layouts were
within 9% of each other from dimension 128 (`sums-one-sample.md`).

The same protocol was run without a BLAS library on two further builds, one
limited to the AVX2 kernel and one with the portable kernel, which on this
processor rounds products and sums separately. Ranges are over the three
sample sizes:

| Kernel | Products/records, N = 8 to 64 | Direct/quasilinear, N = 8 | Direct/quasilinear, N = 16 |
| --- | --- | --- | --- |
| AVX-512 | 2.67 to 4.83 | 0.73 to 0.87 | 0.83 to 0.96 |
| AVX2 | 1.64 to 2.55 | 0.84 to 0.92 | 0.95 to 1.10 |
| Portable | 0.74 to 1.02 | 0.90 to 0.92 | 1.78 to 1.83 |

The direct sum is faster than the quasilinear sum at dimension 8 with every
kernel. At dimension 16 it is faster only with the AVX-512 kernel, by 4% to
17%; the two are level with AVX2, and the direct sum takes 1.8 times as
long with the portable kernel. `AUTO` therefore uses the direct sum through
dimension 8 and the quasilinear sum from dimension 16 with every kernel,
which costs the AVX-512 kernel that margin at dimension 16. Above dimension
16 the direct sum is a plain loop over the `N^2` products and took 11 times
as long as the quasilinear sum at dimension 32 with AVX-512.

With the portable kernel the fused sum of records was faster than a batch
of products followed by a sum at dimension 2 (1.34 to 1.41). At
dimensions 4 to 32 it took between 0.99 and 1.36 times as long, from 64 to
256 it was within 4%, and at 512 and 1024 it took 1.11 to 1.19 times as
long. At dimension 4 the direct sum of planes reached 2.03 times the batch
speed. See [stability.md](stability.md#fused-sums) for accuracy.

The archive was measured from a working tree with uncommitted changes on
commit `793c36f`; its JSON files record the hash of every measured source
file.

The run reported in version 2 of the paper is archived in
[`benchmark-results/paper-v2-octonion243-20261023/`](../benchmark-results/paper-v2-octonion243-20261023/results.md),
and other directories record earlier builds; each report names the source it
measured. The host is shared, so timing differences between archived runs do
not isolate source changes.
