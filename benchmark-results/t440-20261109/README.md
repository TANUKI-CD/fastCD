# Benchmark run of 2026-11-09 on the T440

This directory holds the run that the README tables and
[`docs/benchmark.md`](../../docs/benchmark.md) report.

## Host and build

* Dell PowerEdge T440, Intel Xeon Gold 6148 (80 logical CPUs, two sockets,
  two SMT threads per core), Linux 7.2.11, shared host; no CPU governor,
  turbo or SMT settings were changed.
* Source: commit `78228a7`, built from a clean checkout. Each JSON file
  records the commit, that the checkout was clean, and the source and
  executable hashes it measured.
* Build: CMake Release (`-O3 -DNDEBUG`, `-fno-fast-math -ffp-contract=off`),
  GCC 16.2.1, finite-field SIMD kernels enabled, with
  `-DFASTCD_BUILD_FLINT_BENCHMARK=ON`. The three runners, but not the
  library, were built with `-Wa,-mbranches-within-32B-boundaries`, recorded
  as `runner_options` in the build record. All CTest tests and the FLINT
  runner's self-test passed before timing. The FLINT comparison links the
  system FLINT 3.6.0.
* Each driver pinned itself to one logical CPU with `--cpu`. The three CPUs
  are on distinct physical cores of socket 0.

## Files

| File | Protocol | Pairs per cell | CPU | Wall time |
|---|---|---:|---:|---|
| `real.json`, `real.md` | real coefficients, default protocol, dimensions 2 to 1024 | 10,000 | 2 | 09:04 to 09:42 |
| `domains.json`, `domains.md` | direct vs quasilinear over the nine default finite fields and C, batch and single | 1,000 | 4 | 09:42 to 10:36 |
| `flint.json`, `flint.md` | fastCD vs FLINT 3.6.0, scalar and complete CD workloads, twelve finite fields | 1,000 | 6 | 10:36 to 17:20 |
| `host-load.log` | busy fraction of the pinned CPUs, their SMT siblings and all of socket 0, every 5 s | | | 09:04 to 17:20 |

Driver invocations, run one after another from the checkout (the runner
and corpus paths point at the scratch build):

```sh
python3 benchmarks/benchmark.py --runner build/fastcd_benchmark \
    --output benchmark-results/t440-20261109/real.json --cpu 2 \
    --corpora corpora/real
python3 benchmarks/benchmark.py --protocol domains \
    --runner build/fastcd_domains_benchmark \
    --output benchmark-results/t440-20261109/domains.json \
    --trials 1000 --cpu 4 --corpora corpora/domains --subprocess-timeout 14400
python3 benchmarks/benchmark.py --protocol flint \
    --runner build/fastcd_flint_benchmark \
    --output benchmark-results/t440-20261109/flint.json \
    --trials 1000 --cpu 6 --corpora corpora/flint --subprocess-timeout 14400 \
    --fields 3:1,5:1,13:1,3:5,5:2,13:5,65521:1,65521:2,2147483647:1,3:12,13:12,2147483647:12
```

Every other option kept its default: 24 blocks, two warmup passes, a 20 ms
minimum observation and the master seed 20260906. Each protocol ran alone.

## Protocol notes

* The finite-field and FLINT protocols used 1,000 pairs per cell instead of
  the 10,000-pair default. The direct algorithm and the independent
  schoolbook verification both cost O(N^2) coefficient products per pair,
  and each extension-field coefficient product costs about deg^2 residue
  products, so the default campaign would take more than a day.
* The real protocol starts a new runner process for every observation. The
  finite-field and FLINT protocols run all 24 blocks of a cell in one
  process and give each block its own stack offset, recorded as
  `stack_offset` in every raw observation.
* All pairs were verified against the independent oracles outside the
  timers: 600,000 real-coefficient checks (10,000 pairs for each of three
  algorithms, two modes and ten dimensions), 100,000 finite-field and
  complex pairs (1,000 in each of 100 cells, checked for both algorithms),
  and 132,000 FLINT-comparison pairs (1,000 in each of 132 cells, checked
  for both libraries). No check failed.
* `host-load.log` was written by a sampler pinned to socket 1. The SMT
  siblings of the pinned CPUs (42, 44 and 46) stayed below 10% busy in
  every sample except two on CPU 46: 69% for one 5-second sample as the
  FLINT protocol started, and 23% for one sample during its last cell. On
  average about one logical CPU of socket 0 besides the runner was busy.
  Other users' jobs raised socket 0 above 4.5 busy CPUs in 13 isolated
  samples, at most 8.9, and each covered less than 2% of the samples of
  the cell it fell in.
* The README tables show a subset of the dimensions; the Markdown reports
  here hold every cell with its 95% interval, and the JSON files hold the
  raw per-block observations, corpus hashes, exact moduli and CPU state.

## Interpreting the intervals

The intervals describe variation between the timing blocks of one run. For
the finite-field and FLINT protocols, all blocks of a cell share one
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
  Cariow–Cariowa for batch products at 16. Single products at dimension 2
  are a tie between direct and quasilinear. Automatic selection agrees
  with every cell.
* Finite fields and C: direct is faster at dimension 2 everywhere and at
  dimension 4 over C and five of the nine fields. From dimension 8 upward
  quasilinear has the lower mean in every cell except two ties, F(3^5) at
  dimension 64 and single products over F(65521^2) at dimension 16; at
  dimension 1024 it is 8.2x to 68x faster over the finite fields and 43x
  faster over C. Automatic selection uses direct for N <= 4 and
  quasilinear otherwise.
* FLINT comparison: on single field multiplications fastCD is 1.9x to
  7.0x faster than FLINT 3.6.0 across the twelve fields. On complete CD
  products, where both sides evaluate the same quasilinear schedule and
  only the coefficient arithmetic differs, the ratio is 2.1x to 37x at
  dimension 16 and 44x to 456x at dimension 1024.
