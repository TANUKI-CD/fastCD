# Benchmark

The default benchmark compares the three real-coefficient multiplication
algorithms on the same random inputs so that the default algorithm for each
dimension can be selected and checked in one reproducible run. It measures
batched multiplication, with one validated batch call per pass, and single
multiplication, with one validated call per product. The product of every
input pair is checked once per algorithm and mode in the first block
against an independent implementation of the defining recursion.

Two further protocols cover complex and finite-field coefficients and an
optional comparison with an external arithmetic library; see
[Selecting a protocol](#selecting-a-protocol).

The [stability study](stability.md) uses the same native timing drivers to
compare the balanced and sequential accumulation schedules on one host and
measures accuracy against an exact reference; its data are in
`benchmark-results/stability-20261107/`.

## Archived results

The README tables come from the run archived in
[`benchmark-results/t440-20261104/`](../benchmark-results/t440-20261104/README.md),
measured on 2026-11-04 from the v1.2.0 source on one logical CPU of an
Intel Xeon Gold 6148 in a Dell PowerEdge T440. `real.json` and `real.md`
hold the real-coefficient run with the default protocol below. `domains.*`
hold the complex and finite-field run, and `flint.*` the comparison with
FLINT (Fast Library for Number Theory), an independent exact-arithmetic C
library; both protocols are described later on this page and were measured
with 1,000 operand pairs per cell rather than the 10,000-pair default. Each
file records the commit, source and executable hashes, build options,
corpus hashes, CPU state and raw observations. All 600,000 real-coefficient
verification checks passed with zero mismatches. The run reported in
version 2 of the paper is archived in
[`benchmark-results/paper-v2-octonion243-20261023/`](../benchmark-results/paper-v2-octonion243-20261023/results.md),
and other directories record earlier builds; each report names the source
it measured. The host is shared, so timing differences between archived
runs do not isolate source changes.

The tables give the mean time per product with a 95% confidence interval,
paired ratios against the quasilinear algorithm, and a paired single/batch
ratio for each algorithm. The intervals use Student's t distribution with
degrees of freedom capped at 30. The "fastest mean" column names the lowest
verified mean, or reports a tie when the 95% interval of the paired
per-block ratio between the two lowest means contains one.

The automatic choices follow this run: single products use direct at
dimensions 8 and 16 and quasilinear otherwise; batches use direct at 8,
Cariow–Cariowa at 16 and quasilinear otherwise. Dimension-4 single products
are a statistical tie between quasilinear and direct, and quasilinear is
chosen. The fraction of a confidence interval below one is not a probability
that one algorithm is faster.

## What is measured

Both modes time the kernels as the library ships them. With the default SIMD
configuration on supported GCC/Clang targets, batch mode runs two products
per SIMD instruction for quasilinear and Cariow–Cariowa at dimensions 16
through 512 and for direct at 64 through 512, with scalar kernels elsewhere.
From dimension 32, a single quasilinear product evaluates the two recursive
calls of `C_(n-1)` in the two lanes (see
[How quasilinear multiplication works](mathematics.md#how-quasilinear-multiplication-works)).
Single direct and Cariow–Cariowa
products run scalar code that the compiler may vectorize. Single mode
includes per-call validation, which cost roughly 15 ns on the archived Xeon
runs and dominated below dimension 8.

Cariow–Cariowa is timed at every dimension the library supports, through
1024. Through 64 it runs generated straight-line kernels; above 64 it runs a
table-driven kernel whose sign corrections need index lookups and gathered
loads that are not vectorized within one product. At dimensions 128 through
1024 it needs only 0.52 to 0.58 times the operations of direct
multiplication, yet in the T440 run of the README its single products are
1.5 to 1.9 times slower than direct. Its batches are within 9% of direct at
dimensions 128 through 512, slightly faster at 256 and 512, and 1.5 times
slower at 1024. Those cells measure how the irregular corrections map onto
the hardware, not the operation count, and the algorithm is not
recommended above 64. Pass `--max-cc-dimension 64` to
stop timing it there; the report then marks the cells "not timed", as
opposed to "unavailable" when the library does not support an algorithm at
a dimension.

Above dimension 256 the Cariow–Cariowa kernel combines the block totals of
each correction sum with a balanced tree (see
[Accumulation order](mathematics.md#accumulation-order)); the
[stability study](stability.md) measures the accuracy and cost of this
order.

Two build options change what is measured. `FASTCD_DISABLE_SIMD=1` removes
the explicit SIMD kernels but not the compiler's own vectorization; add
`-fno-tree-vectorize` to `CMAKE_C_FLAGS` for a scalar-only comparison with
GCC. `FASTCD_CC_DOUBLING=1` evaluates Cariow–Cariowa products at dimensions
16 through 64 as the four half-dimensional products of one Cayley–Dickson
doubling step, two per SIMD lane. That is a different arithmetic schedule
from the paper's fixed-dimension procedure, so it is off by default and
reported separately; on the development machine it made single products
1.3 to 2.0 times faster and batches up to 36% slower at those dimensions.
Pass either option through `-DFASTCD_TUNING_DEFINITIONS=...` at configure
time.

`benchmarks/run_dedicated.sh` performs a complete run on a dedicated
machine: a fresh Release build, the C test suite, and the full benchmark
into a timestamped file under `benchmark-results/`, optionally followed by
the same steps for the `FASTCD_CC_DOUBLING=1` build (`--cc-doubling`).

## Running the benchmark

To build fastCD and run the benchmark, run these commands on a GNU/Linux system.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
cmake --build build --parallel
python3 benchmarks/benchmark.py --runner build/fastcd_benchmark \
    --output benchmark-results/results-current.json
```

The driver requires NumPy. It writes the results to the requested JSON file
and places a Markdown table beside it, including the hashes of the source,
executable, and input corpora, the build options, the raw timings, and the
verification results. Choose a new output path for each run, as the program
will not overwrite an existing JSON or Markdown report.

Rebuild before running the benchmark. CMake writes the build record to
`fastcd-build-Release.txt` beside the runner; for a multi-configuration build,
use `--build-metadata` to select that record. After each dimension the driver
saves the results so far beside the output as `<name>.partial.json`, and
removes that file when the run completes.

| Option | Default | Meaning |
| --- | --- | --- |
| `--dimensions` | `2,...,1024` | comma-separated powers of two to time |
| `--trials` | `10000` | pairs per dimension |
| `--blocks` | `24` | subprocess blocks; each yields one observation per task |
| `--warmup` | `2` | untimed corpus passes before each observation |
| `--min-time-ms` | `20` | calibration target per observation |
| `--seed` | `20260906` | master corpus seed |
| `--norm-min`, `--norm-max` | `0.5`, `2.0` | operand norm range |
| `--corpora` | `benchmark-corpora` | directory of `corpus-nN.f64` files |
| `--max-direct-dimension` | `1024` | largest dimension at which direct is timed |
| `--max-cc-dimension` | `1024` | largest dimension at which Cariow–Cariowa is timed (see above) |
| `--verify-trials` | all | pairs verified in the first block |
| `--cpu` | first allowed | logical CPU to pin (Linux) |
| `--subprocess-timeout` | `3600` | seconds allowed for one runner invocation |
| `--build-metadata` | beside runner | build record for a multi-config build |

The JSON records the CPU frequency governor, turbo and boost settings, and
SMT state when the kernel exposes them, so that two runs can be compared for
comparable machine states.

## Benchmark protocol

The default protocol uses seed `20260906` and 10,000 pairs at each dimension
`N = 2, 4, ..., 1024`. Both modes run with
two warmup passes, 24 subprocess blocks whose task order rotates, and a
20 ms calibration target for each observation. Setting `--blocks` to a
multiple of 12 places every task in each position equally often. You can
restrict the run with `--modes batch` or `--modes single`, or extend it with
`--dimensions`.

The inputs are generated by NumPy PCG64 with `SeedSequence([seed, N])`.
Their coordinates are independent uniform samples, which are normalized and
then scaled to give norms in `[0.5, 2)`. The corpora are kept in
`benchmark-corpora/` so that they can be reused.

On Linux, the driver and its child processes run on the first allowed CPU,
unless you select another with `--cpu`. A full run takes about forty minutes
on a server core, with most of that time spent on the two quadratic
algorithms and the oracle verification at `N = 1024`. Both modes time C calls
with storage already allocated, so Python object overhead is excluded.
If verification fails, the driver still writes both reports, marks the run
and each failing algorithm with the status `verification_failure` alongside
the mismatch counts and largest errors, and exits with a nonzero status.

The C and Python interfaces share the choices in `auto_single_by_level` and
`auto_batch_by_level`, the two tables at the start of the algorithm-selection
section in `src/fastcd.c`. Review these choices against the results of a new
run.

## Selecting a protocol

`--protocol` selects one of three experiments. The optional `flint`
protocol compares fastCD's finite-field arithmetic with
[FLINT](https://flintlib.org/)'s prime-field and polynomial-basis
extension-field arithmetic. FLINT has no Cayley–Dickson multiplication
routine, so for complete products the benchmark runs fastCD's quasilinear
schedule (`benchmarks/flint_schedule.h`) with FLINT's field operations in
place of fastCD's coefficient arithmetic.

| `--protocol` | Measurements | Native runner |
| --- | --- | --- |
| `real` (default) | The three algorithms over R, single and batch | `fastcd_benchmark` |
| `domains` | Direct vs quasilinear over C and odd finite fields | `fastcd_domains_benchmark` |
| `flint` | Scalar field multiplication and complete CD products, fastCD vs FLINT | `fastcd_flint_benchmark` |
| `all` | All three protocols, run sequentially with separate reports | All three runners |

For an individual protocol, `--runner` names the corresponding executable.
For `all`, it names `fastcd_benchmark`; the other executables are found
beside it, or named with `--domain-runner` and `--flint-runner`.
`--build-metadata` selects the runner's build record; an `all` run also
accepts `--domain-build-metadata` and `--flint-build-metadata` for
multi-configuration builds.

An `all` run checks every runner, build record and required dependency
before timing starts. With `--output results.json`, it writes separate
`results.real.json`, `results.domains.json` and `results.flint.json`
files, each with its own Markdown report, and `results.json` and
`results.md` index them. Existing output files are never overwritten, and
results from different protocols are not combined into one statistic.

## Complex and finite fields

A standard build also creates `fastcd_domains_benchmark`, which compares
direct and quasilinear multiplication over complex and finite-field
coefficients. Select it through the main driver:

```sh
python3 benchmarks/benchmark.py --protocol domains \
    --runner build/fastcd_domains_benchmark \
    --output benchmark-results/domains-current.json
```

The default fields are F_3, F_5, F_13, F_(3^5), F_(5^2), F_(13^5),
F_65521, F_(65521^2), F_2147483647 and C, at powers of two from 2 through
1024. Use `--fields char:deg,...` to choose fields, where `0:2` denotes C,
the characteristic is an odd prime at most 2147483647 and the extension
degree is 1 through 12. For example, `--fields 13:5,0:2 --dimensions 8,32,128`
selects two coefficient fields and three CD dimensions.

The protocol uses 10,000 deterministic pairs per field and dimension, 24
rotated blocks, two warmup passes and a 20 ms minimum observation. Each
algorithm and mode calibrates whole-corpus passes independently. `--modes
batch` or `--modes single` restricts the measurement; the block count must
be a multiple of the number of algorithm/mode tasks, four by default and
two with one mode. Native calls are timed, and Python object construction
is excluded.

Direct multiplication and the independent verification both perform O(N^2)
coefficient products per pair, and the verification computes each
extension-field product with about `deg`^2 residue products, so the largest
cells dominate the run time. With the default 10,000 pairs the campaign
takes more than a day; the archived run used `--trials 1000`. Each cell
runs in its own subprocess with a one-hour limit by default, which a
1,000-pair cell at dimension 1024 over an extension field can exceed, so
raise `--subprocess-timeout` (in seconds) for such runs. A smaller
`--trials` gives a quick pilot; keep that value in mind when reading its
results.

Each finite coefficient occupies `deg` canonical `uint32` residues in the
polynomial basis, and a record has N such coefficients. Setup chooses and
validates a deterministic monic irreducible defining polynomial, recorded
in the JSON. Complex records contain N `complex128` coefficients. Corpus
manifests bind the field, dimension, trial count, seed, dtype and byte
hash, and a mismatching existing corpus is rejected. These corpora are
separate from the real benchmark's normalized `float64` inputs.

Every pair is verified before timing: finite products against independent
schoolbook polynomial arithmetic with signed CD basis accumulation, complex
products against long-double accumulation with a norm-scaled tolerance.
Raw timings, exact moduli, corpus hashes, build/source hashes, verification
counts and paired direct/quasilinear ratios are retained; a ratio above 1
favors quasilinear. The quasilinear timings, and the direct timings from
dimension 16 on where a packed kernel serves the field, include the degree
specialization, the packed SIMD arithmetic and the field-specific maps that
each call builds to split extension-field products into prime-field
products, so the results measure the shipped implementations rather than
coefficient operation counts alone.

`benchmarks/domains.py` can also be run directly with the same options.
For complex coefficients, `--fields 0:2 --baseline` additionally times a
quasilinear implementation that splits each complex product into three
real products, with packing and reconstruction inside its timer.

## Comparing finite-field arithmetic with FLINT

The FLINT protocol answers two separate questions:

* **Scalar field multiplication:** `fastcd_ff_field_mul` versus FLINT's
  `nmod_mul` for prime fields or `fq_nmod_mul` for extensions. This
  measures the coefficient arithmetic alone.
* **Complete CD multiplication:** fastCD's quasilinear product versus the
  same quasilinear schedule evaluated with FLINT's field operations. Both
  multiply N field coefficients with the same basis convention and the
  same defining polynomial.

Scalar multiplication is measured once per field; CD multiplication uses
the selected `--dimensions`. Both default to the nine finite fields listed
above; C is not part of this protocol. `--workloads scalar` or `--workloads cd`
selects one workload, and `--fields` accepts any supported
characteristic/degree, for example `3:12,13:5,2147483647:12`. The default
grid is a representative experiment, not a survey of all primes or all
irreducible polynomials.

### Building and running

Install FLINT's development headers and library through your package
manager or the [FLINT installation instructions](https://flintlib.org/doc/install.html),
then build the extra executable:

```sh
cmake -S . -B build-flint -DCMAKE_BUILD_TYPE=Release \
    -DFASTCD_BUILD_FLINT_BENCHMARK=ON
cmake --build build-flint --parallel
ctest --test-dir build-flint --output-on-failure
python3 benchmarks/benchmark.py --protocol flint \
    --runner build-flint/fastcd_flint_benchmark \
    --output benchmark-results/flint-current.json
```

For a nonstandard installation, add `-DCMAKE_PREFIX_PATH=/path/to/flint`
or expose its `flint.pc` through `PKG_CONFIG_PATH`. Only the comparison
executable links FLINT; the library, the installed headers and the Python
package do not depend on it, and a missing FLINT is reported only when
`FASTCD_BUILD_FLINT_BENCHMARK=ON`. The driver checks that the compile-time
and runtime FLINT versions match; the archived comparison uses FLINT 3.6.0.
CTest exercises every supported extension degree against the independent
oracle and checks the comparison wrapper's workspace and overlap errors.

Run every protocol from that build with:

```sh
python3 benchmarks/benchmark.py --protocol all \
    --runner build-flint/fastcd_benchmark \
    --output benchmark-results/all-current.json
```

The corpus and timing options apply to every protocol. `--modes` applies to
the real and domain experiments; the FLINT comparison makes one call per
product because its adapters have no batch entry, and the non-real
protocols always verify every pair. In an `all` run, `--fields` selects the
domain fields, and its finite subset selects the FLINT fields, so at least
one finite field is required. Corpora go into separate subdirectories below
`--corpora`.

### Timing, verification and interpretation

Both implementations receive identical canonical operand coefficients and
the same defining polynomial. The extension comparison uses FLINT's
`fq_nmod` polynomial-basis representation, not Zech logarithms. Conversion
to FLINT objects, context construction, corpus I/O, allocation, warmup and
verification are outside the timers; work inside FLINT's arithmetic
routines is included. FLINT is restricted to one thread.

Each observation replays the full fixed corpus with a common calibrated
pass count, so that even the faster implementation reaches the target, and
the two implementations alternate their first/second positions across the
24 paired blocks. Wall-clock timings are primary; thread CPU times are
retained when available. An independent schoolbook polynomial
implementation checks every pair before timing and every final timed output
afterwards; mismatches, missing observations or changed build files prevent
a successful report. Reports retain raw paired observations, exact
polynomials, seeds, corpus hashes, source/build/executable hashes and FLINT
versions, and partial JSON files preserve completed cells after an
interrupted run. CMake records the linked FLINT binary and header hashes
but cannot recover how an externally installed FLINT was compiled, so
record FLINT's own build options alongside published results.

Tables show each implementation's mean ns/product and the paired geometric
mean of **FLINT time / fastCD time** with a 95% Student-t interval; ratios
above 1 favor fastCD. Scalar results isolate the coefficient multiply.
Complete CD results also include degree specialization, SIMD packing, the
field-specific maps that each fastCD call builds, layout differences and
buffer validation, so they do not isolate the scalar multiply's
contribution. Neither comparison
establishes a universal speedup or covers every representation FLINT
offers.

The archived comparison in
[`benchmark-results/t440-20261104/`](../benchmark-results/t440-20261104/README.md)
(`flint.md`) covers the nine default fields plus F_(3^12), F_(13^12) and
F_(2147483647^12) with 1,000 pairs per cell, and the README tables summarize
it. The run-time notes for the finite-field protocol apply here too: the
verification costs O(N^2) coefficient products per pair and FLINT's
extension arithmetic is slowest at degree 12, so the degree-12 cells at
dimension 1024 are the most expensive, and `--subprocess-timeout` may need
raising.
