# Benchmark run of 2026-11-20 on the T440

This directory holds the run that the README tables and
[`docs/benchmark.md`](../../docs/benchmark.md) report: products over real
and complex coefficients and finite fields, the comparison with FLINT, and
the fused sums with their accuracy study. The stability study of the same
commit is in [`stability-20261120`](../stability-20261120/README.md).

## Host and builds

* Dell PowerEdge T440, Intel Xeon Gold 6148 (80 logical CPUs, two sockets,
  two SMT threads per core), Linux 7.2.11, shared host; no CPU governor,
  turbo or SMT settings were changed.
* Source: commit `93c9bd8`, built from a clean checkout. Each JSON file
  records the commit, that the checkout was clean, and the source and
  executable hashes it measured.
* Builds: CMake Release (`-O3 -DNDEBUG`, `-fno-fast-math -ffp-contract=off`),
  GCC 16.2.1, finite-field SIMD kernels enabled. The runners, but not the
  library, were built with `-Wa,-mbranches-within-32B-boundaries`, recorded
  as `runner_options` in the build records. Five builds were used:
  * default options: the AVX-512 kernel of the fused sums;
  * `-DFASTCD_TUNING_DEFINITIONS=FASTCD_SUM_MAX_KERNEL=2`: the AVX2 kernel;
  * `-DFASTCD_ENABLE_SUM_SIMD=OFF`: the portable kernel, which without
    `-mfma` rounds each product and sum separately;
  * `-DBUILD_SHARED_LIBS=ON`: the library loaded by the accuracy study;
  * `-DFASTCD_BUILD_FLINT_BENCHMARK=ON`: the runners of the real, domains
    and FLINT protocols, linked with the system FLINT 3.6.0.
* All CTest tests passed on each of these builds and on a Clang build with
  the address and undefined-behavior sanitizers before timing, including
  the FLINT runner's self-test.
* Matrix product: `cblas_dgemm` of Intel MKL 2026.2.1 (`libmkl_rt.so.3`
  from the `mkl` wheel), loaded by the runner, sequential threading layer,
  one thread. It was given to the AVX-512 sums runs only.
* Python 3.14.8; NumPy 2.5.5 for the sums runs and the accuracy study, and
  NumPy 2.5.4 for the other protocols.

## Files

| File | Content | CPU | Wall time (UTC) |
|---|---|---:|---|
| `sums.json`, `sums.md` | sums protocol, AVX-512 kernel, 8,192 elements per cell in samples of 64, 1,024 and 4,096, dimensions 2 to 1024 | 26 | 11-20 23:26 to 23:33 |
| `sums-one-sample.json`, `sums-one-sample.md` | the same with one sample of 1,024 elements per cell | 26 | 11-20 23:33 to 23:35 |
| `sums-avx2.json`, `sums-avx2.md` | as `sums`, AVX2 kernel, no BLAS library | 26 | 11-20 23:35 to 23:40 |
| `sums-portable.json`, `sums-portable.md` | as `sums`, portable kernel, no BLAS library | 26 | 11-20 23:40 to 23:46 |
| `accuracy.json`, `accuracy.md` | accuracy study of the fused sums, full profile: 689 sums and 84 covariances, AVX-512 kernel, 16 worker processes | | 11-20 23:46 to 23:51 |
| `real.json`, `real.md` | real coefficients, default protocol, dimensions 2 to 1024, 10,000 pairs per cell | 2 | 11-20 23:52 to 11-21 00:29 |
| `domains.json`, `domains.md` | direct vs quasilinear over the nine default finite fields and C, batch and single, 1,000 pairs per cell | 4 | 11-21 00:29 to 01:23 |
| `flint.json`, `flint.md` | fastCD vs FLINT 3.6.0, scalar and complete CD workloads, twelve finite fields, 1,000 pairs per cell | 6 | 11-21 01:23 to 08:05 |
| `host-load.log` | busy fraction of the pinned CPUs, their SMT siblings and all of socket 0, every 5 s, in local time (UTC−5) | | 11-20 23:52 to 11-21 08:20 |

Driver invocations, run one after another from the checkout (the runner,
library and corpus paths point at the scratch builds):

```sh
python3 benchmarks/benchmark.py --protocol sums \
    --runner build/fastcd_sums_benchmark \
    --output benchmark-results/t440-20261120/sums.json \
    --counts 64,1024,4096 --cpu 26 \
    --blas $VENV/lib/libmkl_rt.so.3 --blas-name "Intel MKL 2026.2.1"
python3 benchmarks/benchmark.py --protocol sums \
    --runner build/fastcd_sums_benchmark \
    --output benchmark-results/t440-20261120/sums-one-sample.json \
    --trials 1024 --counts 1024 --cpu 26 \
    --blas $VENV/lib/libmkl_rt.so.3 --blas-name "Intel MKL 2026.2.1"
python3 benchmarks/benchmark.py --protocol sums \
    --runner build-avx2/fastcd_sums_benchmark \
    --output benchmark-results/t440-20261120/sums-avx2.json \
    --counts 64,1024,4096 --cpu 26
python3 benchmarks/benchmark.py --protocol sums \
    --runner build-portable/fastcd_sums_benchmark \
    --output benchmark-results/t440-20261120/sums-portable.json \
    --counts 64,1024,4096 --cpu 26
OPENBLAS_NUM_THREADS=1 python3 benchmarks/stability_sums.py \
    --library build-shared/libfastcd.so \
    --output benchmark-results/t440-20261120/accuracy.json --workers 16
python3 benchmarks/benchmark.py --runner build-flint/fastcd_benchmark \
    --output benchmark-results/t440-20261120/real.json --cpu 2 \
    --corpora corpora/real
python3 benchmarks/benchmark.py --protocol domains \
    --runner build-flint/fastcd_domains_benchmark \
    --output benchmark-results/t440-20261120/domains.json \
    --trials 1000 --cpu 4 --corpora corpora/domains --subprocess-timeout 14400
python3 benchmarks/benchmark.py --protocol flint \
    --runner build-flint/fastcd_flint_benchmark \
    --output benchmark-results/t440-20261120/flint.json \
    --trials 1000 --cpu 6 --corpora corpora/flint --subprocess-timeout 14400 \
    --fields 3:1,5:1,13:1,3:5,5:2,13:5,65521:1,65521:2,2147483647:1,3:12,13:12,2147483647:12
```

Every other option kept its default: 60 blocks for the sums protocol and 24
for the others, two warmup passes, a 20 ms minimum observation and the
master seed 20260906; the accuracy study has its own seeds. Each protocol
ran alone.

## Protocol notes

* The finite-field and FLINT protocols used 1,000 pairs per cell instead of
  the 10,000-pair default. The direct algorithm and the independent
  schoolbook verification both cost O(N^2) coefficient products per pair,
  and each extension-field coefficient product costs about deg^2 residue
  products, so the default campaign would take more than a day.
* The real protocol starts a new runner process for every observation. The
  finite-field, FLINT and sums protocols run all blocks of a cell in one
  process and give each block its own stack offset, recorded as
  `stack_offset` in every raw observation.
* All pairs were verified against the independent oracles outside the
  timers: 600,000 real-coefficient checks (10,000 pairs for each of three
  algorithms, two modes and ten dimensions), 100,000 finite-field and
  complex pairs (1,000 in each of 100 cells, checked for both algorithms),
  and 132,000 FLINT-comparison pairs (1,000 in each of 132 cells, checked
  for both libraries). No check failed.
* Every sums run verified the first sample of each cell against a long
  double contraction of the cross moments and every sample for agreement
  between the tasks; all checks passed. In the accuracy study the record
  and plane layouts returned identical bits in all 689 sums.
* `host-load.log` was written by a sampler pinned to socket 1 from the start
  of the real protocol; the sums runs record only the load average, which
  was 4.7 at their start and between 1.8 and 3.8 at the end of each run.
  During the real and domains protocols the SMT siblings of the pinned CPUs
  stayed below 3% busy, and socket 0 averaged 1.8 busy CPUs, at most 3.5.
  During the FLINT protocol the sibling of CPU 6 exceeded 10% in one
  sample, at 11%, and other users' jobs raised socket 0 above 4.5 busy CPUs
  in 6 of its 4,823 samples, at most 6.6. The log continues through the
  stability study, whose CPUs 18 and 20 and their siblings it also records.
* The README tables show a subset of the dimensions; the Markdown reports
  here hold every cell with its 95% interval, and the JSON files hold the
  raw per-block observations, corpus hashes, exact moduli and CPU state.

## Interpreting the intervals

The intervals describe variation between the timing blocks of one run. For
the finite-field, FLINT and sums protocols, all blocks of a cell share one
process, and with it one placement in physical memory, which affects cache
behavior once a cell no longer fits in the per-core L2 cache. In repeated
runs on this host, individual finite-field cells from dimension 64 upward
differed by up to 17%, and direct/quasilinear ratios by up to 8%. Small
differences between cells, and ties, should be read with this in mind.

On this processor family, branches that straddle a 32-byte boundary run
slower (Intel's JCC erratum). The runners are padded against this, but the
library is measured as built, so its code placement is part of what is
timed. Single products at dimension 16 and below, where the per-call path
is a large part of the time, are the most sensitive to it.

## Results in brief

* Real coefficients: quasilinear has the lowest mean from dimension 32
  upward in both modes, by about 17x over direct and 26x to 27x over
  Cariow–Cariowa at dimension 1024. Direct has the lowest mean at
  dimension 8 in both modes and for single products at 4 and 16, and
  Cariow–Cariowa for batch products at 16; quasilinear has it at
  dimension 2 in both modes and for batch products at 4. Automatic
  selection agrees with every cell.
* Finite fields and C: direct is faster at dimension 2 in every cell except
  single products over F(65521^2), a tie, and at dimension 4 over C, five of
  the nine fields and single products over F(13^5). From dimension 8 upward
  quasilinear has the lower mean in every cell except two ties, F(3^5) at
  dimension 64 and batch products over F(65521^2) at dimension 16; at
  dimension 1024 it is 8.4x to 71x faster over the finite fields and 44x
  faster over C. Automatic selection uses direct for N <= 4 and
  quasilinear otherwise.
* FLINT comparison: on single field multiplications fastCD is 2.0x to
  7.0x faster than FLINT 3.6.0 across the twelve fields. On complete CD
  products, where both sides evaluate the same quasilinear schedule and
  only the coefficient arithmetic differs, the ratio is 2.1x to 37x at
  dimension 16 and 49x to 465x at dimension 1024.
* Fused sums, AVX-512 kernel: a fused sum of records is 1.6x to 5.1x as
  fast as a batch of products followed by a sum, and faster than one
  `dgemm` and the contraction with the basis signs from dimension 32
  upward at every sample size. `AUTO` uses the direct sum through
  dimension 8 and the quasilinear sum above.
* Accuracy of the fused sums: over all nine input families and sample
  sizes the largest `eta` of the fused sum is 3.4, and the covariance's is
  at most 0.21; [`accuracy.md`](accuracy.md) has every table.
