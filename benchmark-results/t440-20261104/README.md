# Benchmark run of 2026-11-04 on the T440

This directory holds the run that the README tables and
[`docs/benchmark.md`](../../docs/benchmark.md) report for fastCD 1.2.0.

## Host and build

* Dell PowerEdge T440, Intel Xeon Gold 6148 (80 logical CPUs, two sockets,
  two SMT threads per core), Linux 7.2.11, shared host; no CPU governor,
  turbo or SMT settings were changed.
* Source: commit `c500d40` with the uncommitted 1.2.0 changes (documentation
  and the version string). Each JSON file records the source and executable
  hashes it measured.
* Build: CMake Release (`-O3 -DNDEBUG`, `-fno-fast-math -ffp-contract=off`),
  GCC 16.2.1, finite-field SIMD kernels enabled, with
  `-DBUILD_TESTING=ON -DFASTCD_BUILD_FLINT_BENCHMARK=ON`; all CTest
  tests passed before timing. The FLINT comparison links the system
  FLINT 3.6.0.
* Each driver pinned itself to one logical CPU with `--cpu`; the three
  pinned CPUs are on distinct physical cores of socket 0.

## Files

| File | Protocol | Pairs per cell | CPU | Wall time |
|---|---|---:|---:|---|
| `real.json`, `real.md` | real coefficients, default protocol, dimensions 2 to 1024 | 10,000 | 0 | 13:27 to 14:04 |
| `domains.json`, `domains.md` | direct vs quasilinear over the nine default finite fields and C, batch and single | 1,000 | 2 | 16:13 to 20:27 |
| `flint.json`, `flint.md` | fastCD vs FLINT 3.6.0, scalar and complete CD workloads, twelve finite fields | 1,000 | 6 | 16:14 to 22:56 |

Driver invocations (the runner paths point at the scratch build):

```sh
python3 benchmarks/benchmark.py --runner build/fastcd_benchmark \
    --output benchmark-results/t440-20261104/real.json --cpu 0 \
    --corpora benchmark-corpora
python3 benchmarks/benchmark.py --protocol domains \
    --runner build/fastcd_domains_benchmark \
    --output benchmark-results/t440-20261104/domains.json \
    --trials 1000 --cpu 2 --corpora benchmark-corpora --subprocess-timeout 14400
python3 benchmarks/benchmark.py --protocol flint \
    --runner build/fastcd_flint_benchmark \
    --output benchmark-results/t440-20261104/flint.json \
    --trials 1000 --cpu 6 --corpora benchmark-corpora --subprocess-timeout 14400 \
    --fields 3:1,5:1,13:1,3:5,5:2,13:5,65521:1,65521:2,2147483647:1,3:12,13:12,2147483647:12
```

Every other option kept its default: 24 blocks, two warmup passes, a 20 ms
minimum observation and the master seed 20260906. The real run executed
alone. The finite-field and FLINT runs executed concurrently on separate
physical cores. Two earlier attempts at the finite-field and FLINT runs, launched with the
driver's default one-hour per-cell limit, were killed by that limit at
F(3^5), N = 1024 and at F(3^12), N = 1024 respectively; the drivers do
not resume, so both protocols were rerun from the start with
`--subprocess-timeout 14400`. The FLINT rerun was started as
`flint-retry.json` while the first attempt was still running and was
renamed afterwards, so its JSON records that output name.

## Protocol notes

* The finite-field and FLINT protocols used 1,000 pairs per cell instead of
  the 10,000-pair default. The direct algorithm and the independent
  schoolbook verification both cost O(N^2) coefficient products per pair,
  and each extension-field coefficient product costs about deg^2 residue
  products, so the default campaign would have taken more than a day.
* All pairs were verified against the independent oracles outside the
  timers: 600,000 real-coefficient checks (10,000 pairs for each of
  three algorithms, two modes and ten dimensions), 100,000 finite-field
  and complex pairs (1,000 in each of 100 cells, checked for both
  algorithms), and 132,000 FLINT-comparison pairs (1,000 in each of 132 cells,
  checked for both libraries). No check failed.
* The README tables show a subset of the dimensions; the Markdown reports
  here hold every cell with its 95% interval, and the JSON files hold the
  raw per-block observations, corpus hashes, exact moduli and CPU state.

## Results in brief

* Real coefficients: quasilinear has the lowest mean from dimension 32 upward in both modes, by 17.5x over direct and about 26x over Cariow-Cariowa at dimension 1024. Direct wins at dimension 8 (and single products at 16), Cariow-Cariowa wins batch products at 16, and the automatic selection agrees with every cell. All 600,000 verification checks passed with zero mismatches.
* Finite fields and C: direct is faster only at dimension 2, and at
  dimension 4 over the prime fields and C; from dimension 8 upward
  quasilinear has the lower mean in every cell, by 219x to 3,300x at
  dimension 1024 over the finite fields and 45x over C. Automatic
  selection uses quasilinear over every finite field.
* FLINT comparison: on single field multiplications fastCD is 2.0x to
  7.1x faster than FLINT 3.6.0 across the twelve fields. On complete CD
  products, where both sides evaluate the same quasilinear schedule and
  only the coefficient arithmetic differs, the ratio grows from 2x to
  36x at dimension 16 to 48x to 452x at dimension 1024.
