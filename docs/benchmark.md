# Benchmark

The benchmark compares the three multiplication algorithms on the same
random inputs so that the default algorithms for a CPU can be selected and
checked in one reproducible run. It measures both batched multiplication,
with one validated batch call per pass, and single multiplication, with one
validated call per product. The product of every input pair is checked once
per algorithm and mode, in the first block, against an independent
implementation of the defining recursion.

The archived `benchmark-results/paper-results.json` and its Markdown report
measure revision `14bf305` for the paper's v1, before the current
quasilinear local-map fusion and complex octonion base case. Preserve those
results when making a new comparison; the current
[arithmetic counts](mathematics.md) describe the default quasilinear schedule.

The [243-operation-leaf T440 rerun](../benchmark-results/paper-v2-octonion243-20261023/results.md)
measures the revised schedule through dimension 1024 using the v1 paper run's
protocol and identical input corpus hashes; the paper's v2 reports this run. Its
[raw observations](../benchmark-results/paper-v2-octonion243-20261023/results.json),
[environment](../benchmark-results/paper-v2-octonion243-20261023/environment.json),
[commands](../benchmark-results/paper-v2-octonion243-20261023/commands.json), and
[validation record](../benchmark-results/paper-v2-octonion243-20261023/validation.json)
are archived together. The measured source is revision `7da5a5f` plus the
archived [source patch](../benchmark-results/paper-v2-octonion243-20261023/source.patch),
with exact source hashes in the build record. The README tables use this run.
All 600,000 algorithm/mode verification checks passed with zero mismatches and
finite outputs. The [instrumented arithmetic counts](../benchmark-results/paper-v2-octonion243-20261023/operation-counts.json)
confirm `(M, A, E) = (72, 156, 15)`, stored under the keys `M`, `A`, and `S`,
for the complex octonion leaf: 243
operations, saving one per leaf and `N/16` per full real product at `N >= 16`.

The [earlier full 2026-10-23 rerun](../benchmark-results/paper-v2-20261023/results.md)
measures the preceding 244-operation leaf at revision `f9a61b2`. It remains
archived with its original observations and metadata. The
[comparison of the two leaf versions](../benchmark-results/paper-v2-octonion243-20261023/comparison-previous-leaf.json)
uses descriptive ratios of means from separate runs, not paired cross-version
confidence intervals. Both v2 runs used GCC 16.2.1 and Linux 7.2.11; the
v1 paper run used different compiler and kernel versions. The host is
shared, so historical timing differences do not isolate source-code changes.

The earlier [2026-10-19 T440 comparison](../benchmark-results/t440-20261019/README.md)
measures the 244-operation-leaf schedule against fastCD 1.0.0 through dimension
256, with raw observations, build hashes, arithmetic counts and correctness checks.

The results give the mean time with a 95% confidence interval, paired ratios
against the quasilinear algorithm, and a paired single/batch ratio for each
algorithm. The intervals use Student's t distribution with degrees of freedom
capped at 30. The "fastest mean" column names the lowest verified mean, or
reports a tie when the 95% interval of the paired per-block ratio between the
two lowest means contains one.

The [AUTO assessment](../benchmark-results/paper-v2-octonion243-20261023/auto-assessment.json)
supports retaining the existing choices after the revised-leaf run. Singles
use direct at dimensions 8 and 16 and quasilinear otherwise; SIMD batches
use direct at 8, Cariow–Cariowa at 16 and quasilinear otherwise. Dimension-4
singles still have close means: quasilinear 18.12 ns and direct 18.18 ns,
with a paired quasilinear/direct ratio of 0.997 and a 95%
interval of 0.979–1.015. That interval contains one, so the table labels them
a tie. The quasilinear choice remains a tuning decision; the fraction of a
confidence interval below one is not a probability that one algorithm is faster.
The new report records the current defaults, while older reports preserve
the defaults of their measured builds.

Both modes measure the kernels as the library ships them. With the default
SIMD configuration on supported GCC/Clang targets, batch mode runs
two products per SIMD instruction for quasilinear and Cariow–Cariowa at
dimensions 16 through 512 and for direct at 64 through 512, with scalar
kernels elsewhere. In the archived Xeon build, the compiler vectorizes the
dense direct kernels two outputs per instruction in single mode. The
quasilinear product evaluates the two children of the top complex node in
the two lanes from dimension 32; the
Cariow–Cariowa product runs scalar code: adjacent outputs need different
subsets of its sign-correction terms, and this implementation does not
vectorize them within one product. Single mode includes per-call validation;
on the archived Xeon run this cost roughly 15 ns and dominated below dimension 8.

Cariow–Cariowa is timed at every dimension the library supports, through
1024, and its results above 64 should be read with the implementation in
mind. Through 64 it runs generated straight-line kernels; above 64 it runs a
table-driven kernel, because the unrolled form would be megabytes of code.
In the table-driven kernel, its sign corrections select about half of the
terms of each non-scalar output in an irregular pattern, requiring index
lookups and gathered loads. This implementation does not vectorize those
corrections within one product; through dimension 64 the generated kernels
use fixed indices instead. The compiler can vectorize direct's dense sums
two outputs per instruction. In the archived 1.0.0 run on the Xeon Gold 6148
this left Cariow–Cariowa
1.5 to 1.8 times slower than direct for single products at dimensions 128
to 1024 despite having 0.6 times the operations, and between parity and
1.5 times slower for batches, where the paired lanes amortize its index
work until pairing stops above 512. Those cells therefore measure how the algorithm's
irregular corrections map onto the hardware, not its operation count; the
algorithm is not recommended for use above 64, as the
[mathematics](mathematics.md) document explains. Pass
`--max-cc-dimension 64` to stop timing it there; the report then states the
cap in its header and marks the cells "not timed" (as opposed to
"unavailable" when the library does not support an algorithm at a
dimension).

Two build options change what is measured. `FASTCD_DISABLE_SIMD=1` removes
the explicit SIMD kernels but not the compiler's own vectorization, so it is
not a scalar-only comparison; add `-fno-tree-vectorize` to `CMAKE_C_FLAGS`
for that with GCC. On targets with the explicit SIMD kernels,
`FASTCD_CC_DOUBLING=1` evaluates Cariow–Cariowa products at
dimensions 16 through 64 as the four half-dimensional products of one
Cayley–Dickson doubling step, two per lane, in both modes. That is a
different arithmetic schedule from the paper's fixed-dimension procedure
(about the same multiplications, roughly twice the Walsh–Hadamard
additions), so it is off by default and reported separately. On the
development machine it made single Cariow–Cariowa products 1.3, 1.5, and
2.0 times faster at dimensions 16, 32, and 64, and batches 36% and 9%
slower at 16 and 32, because the packing and the extra additions outweigh
the paired half-dimensional kernels there. Pass either option through
`-DFASTCD_TUNING_DEFINITIONS=...` at configure time.

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

The 2026-10-19 T440 comparison instead used dimensions through 256, 48 blocks
and a 100 ms calibration target on CPU 11. Its [report](../benchmark-results/t440-20261019/README.md#reproduction-and-artifacts)
gives the exact commands and build settings for that run.

The inputs are generated by NumPy PCG64 with `SeedSequence([seed, N])`.
Their coordinates are independent uniform samples, which are normalized and
then scaled to give norms in `[0.5, 2)`. The corpora are kept in
`benchmark-corpora/` so that they can be reused.

On Linux, the driver and its child processes run on the first allowed CPU,
unless you select another with `--cpu`. A full run takes about forty minutes on a server core, with most of that time spent on the two quadratic algorithms and the oracle verification at `N = 1024`. Both modes time C calls
with storage already allocated, so Python object overhead is excluded.
If verification fails, the driver still writes both reports, marks the run
and each failing algorithm with the status `verification_failure` alongside
the mismatch counts and largest errors, and exits with a nonzero status.

The C and Python interfaces share the choices in `auto_single_by_level` and
`auto_batch_by_level`, the two tables at the start of the algorithm-selection
section in `src/fastcd.c`. Review these choices against the results of a new
run.
