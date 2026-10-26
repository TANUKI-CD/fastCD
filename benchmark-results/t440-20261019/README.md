# T440 multiplication measurements, 2026-10-19

Comparison of fastCD 1.0.0 (`d001c1e7fe5096253d7628fc1f57cb126d766919`) with the default quasilinear schedule using local-map fusion and the complex octonion alternating-product base case. The baseline was built from a clean source archive; the updated tree was an uncommitted source snapshot. Source and executable SHA-256 hashes are recorded in the raw reports and build records. No change was made to direct or Cariow–Cariowa arithmetic, the public interfaces, or AUTO selection.

Documentation and source comments were corrected after measurement. The
[documentation audit](documentation-audit.json) maps the affected source hashes
and confirms identical preprocessed C, generated headers and operation counts.
The raw benchmark records retain the hashes of the measured snapshot.

## Protocol

- Dell PowerEdge T440; Intel Xeon Gold 6148 at nominal 2.40 GHz; Linux 7.2.11, GCC 16.2.1 20260810. Both builds use `-O3 -DNDEBUG -fno-fast-math -ffp-contract=off`, default leaf cutoffs and SIMD settings, and warnings as errors.
- Each run covers dimensions 2, 4, 8, 16, 32, 64, 128 and 256, all three algorithms, and single/batch modes. Each task uses 10,000 operand pairs, 48 subprocess blocks, two warmup corpus passes and at least 100 ms per timing observation. Corpus order and hashes match across builds; operand norms are in `[0.5,2)`.
- Both complete runs use logical CPU 11, whose SMT sibling is 51. The algorithm/mode order rotates within each block. The baseline run precedes the updated run; before/after ratios below are ratios of the means, not paired confidence intervals. Each individual mean has its own 95% Student t interval.
- The server remains a shared host: other services were left running, with no governor, turbo, SMT or service changes. A short preliminary dimension-16 run on CPU 12 overlapped part of the baseline run; it is archived separately and excluded from the main comparison. The unchanged algorithms provide controls for host/compiler variation.

Baseline run: `2026-10-19T19:12:48-0400` to `2026-10-19T19:25:54-0400`. Updated run: `2026-10-19T19:25:55-0400` to `2026-10-19T19:38:50-0400`.

## Quasilinear timing

Time per product in nanoseconds, mean [95% confidence interval]. A ratio above 1 means the updated implementation was faster. Dimensions 2 and 4 have unchanged arithmetic and serve as small-size controls.

| N | Single baseline | Single updated | Baseline / updated | Batch baseline | Batch updated | Baseline / updated |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 15.54 [15.41, 15.67] | 15.39 [15.26, 15.52] | 1.010 | 1.01 [1.01, 1.02] | 1.18 [1.16, 1.19] | 0.863 |
| 4 | 19.16 [19.02, 19.31] | 18.22 [18.07, 18.37] | 1.052 | 3.76 [3.73, 3.79] | 3.80 [3.76, 3.83] | 0.992 |
| 8 | 38.44 [37.86, 39.02] | 37.31 [36.89, 37.73] | 1.030 | 17.14 [17.00, 17.27] | 17.89 [17.75, 18.03] | 0.958 |
| 16 | 122.24 [121.31, 123.18] | 113.65 [112.73, 114.57] | 1.076 | 59.65 [59.20, 60.11] | 56.64 [56.19, 57.09] | 1.053 |
| 32 | 235.65 [233.77, 237.54] | 217.00 [215.27, 218.72] | 1.086 | 197.61 [196.08, 199.14] | 185.79 [184.26, 187.31] | 1.064 |
| 64 | 558.55 [554.61, 562.49] | 509.14 [505.66, 512.61] | 1.097 | 508.62 [505.37, 511.87] | 457.91 [454.48, 461.33] | 1.111 |
| 128 | 1342.93 [1330.37, 1355.48] | 1248.37 [1238.79, 1257.95] | 1.076 | 1335.65 [1325.65, 1345.65] | 1234.15 [1220.94, 1247.36] | 1.082 |
| 256 | 3535.75 [3504.44, 3567.07] | 3231.32 [3205.80, 3256.84] | 1.094 | 3321.41 [3296.51, 3346.32] | 3035.32 [3010.91, 3059.73] | 1.094 |

At dimensions 16 through 256, unchanged direct/CC mean times changed by -2.40% to +6.13% between runs. The raw measurements therefore include host variation and possible code-layout effects; the before/after ratios should not be interpreted as an isolated instruction-throughput experiment.

The lower arithmetic count does not improve every small kernel: dimension-8 batch QL is slightly slower in this run. AUTO continues to use direct there. Dimension-2 arithmetic is unchanged; its subnanosecond batch difference also illustrates the sensitivity of the smallest timings to layout and measurement conditions.

Quasilinear has the lowest mean at every tested dimension from 32 through 256 in both modes. At dimension 16, direct remains fastest for singles and Cariow–Cariowa for batches, so the AUTO choices are retained.

The complete three-algorithm comparisons, paired within-run ratios and all 48 observations are in [baseline.md](baseline.md), [updated.md](updated.md), [baseline.json](baseline.json) and [updated.json](updated.json).

## Arithmetic counts

The operator-instrumented scalar template and actual generated leaves match the manuscript schedule at every power of two through 256. These are arithmetic schedule counts, not hardware instruction counts. `M` counts input-dependent products, `A` additions/subtractions, `S` nontrivial fixed scalings and `T=M+A+S`; known-zero operations, copying, permutations and signs are free. SIMD lanes preserve this schedule per product.

| N | M | A | S | T |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 0 | 0 | 1 |
| 2 | 4 | 2 | 0 | 6 |
| 4 | 16 | 12 | 0 | 28 |
| 8 | 58 | 62 | 0 | 120 |
| 16 | 160 | 244 | 14 | 418 |
| 32 | 454 | 766 | 56 | 1276 |
| 64 | 1186 | 2114 | 172 | 3472 |
| 128 | 2938 | 5418 | 468 | 8824 |
| 256 | 7018 | 13242 | 1188 | 21448 |

The complex dimension-8 leaf independently counts `(M,A,S)=(72,158,14)`, total 244. See [operation-counts.json](operation-counts.json) for this measured schedule. The current [mathematics.md](../../docs/mathematics.md) describes the later 243-operation leaf.

## Correctness and build checks

Each build verified 480,000 products (10,000 pairs × 8 dimensions × 3 algorithms × 2 modes) against the independent defining recursion, with zero mismatches, finite results and stable checksums across timing blocks. The corpus is reused between algorithms and builds; this is 10,000 distinct pairs per dimension, not 480,000 distinct pairs. The acceptance bound is `128 eps N ||x||_2 ||y||_2` in the infinity norm.

| Build | Largest absolute error, all algorithms | Largest norm-scaled error, quasilinear |
| --- | ---: | ---: |
| baseline | 1.33227e-15 | 4.28214e-16 |
| updated | 1.33227e-15 | 5.47688e-16 |

The new leaf also passes exact comparisons for all 196 ordered real basis pairs in its seven complex coordinates, plus 256 dense dyadic cancellation cases. Local C tests pass in default, SIMD-disabled, cutoff-4 and cutoff-16 builds and under Clang AddressSanitizer/UndefinedBehaviorSanitizer (leak detection disabled). All 79 Python interface tests and 12 benchmark-driver tests pass; the four generated headers reproduce byte for byte. Both T440 CTest suites pass. This validates the tested inputs and contracts, rather than proving a general floating-point error bound.

## Reproduction and artifacts

Run the following for each source tree on the T440, using separate build/output paths and the same corpus directory:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFASTCD_BUILD_BENCHMARK=ON -DFASTCD_WARNINGS_AS_ERRORS=ON
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 benchmarks/benchmark.py \
  --runner build/fastcd_benchmark --dimensions 2,4,8,16,32,64,128,256 \
  --trials 10000 --blocks 48 --warmup 2 --min-time-ms 100 \
  --cpu 11 --corpora benchmark-corpora --output results.json
```

Run the arithmetic probe from the updated source tree:

```sh
python3 tools/check_quasilinear_counts.py --output operation-counts.json
```

- [Machine-readable comparison](summary.json) and [validation inventory](validation.json).
- [Environment](environment.json), [baseline build record](baseline-build/fastcd-build-Release.txt) and [updated build record](updated-build/fastcd-build-Release.txt).
- [Baseline CTest](baseline-ctest.log), [updated CTest](updated-ctest.log), local CTest logs and configure/build logs in this directory.
- [Preliminary dimension-16 pilot](pilot16.md), excluded from the main comparison.
