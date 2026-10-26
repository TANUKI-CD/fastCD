# fastCD multiplication benchmark

Host: Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz (x86_64); gcc 16.2.1 20260810; 48 blocks of 10000 pairs per dimension; each cell is the mean over blocks of ns per product with a two-sided 95% Student-t interval.

Batch mode times one validated batch call per corpus pass (two products per SIMD instruction where the build supports it); single mode times one validated single-product call per pair.  Python object overhead is excluded. Verification uses an independent float64 oracle, not an exact product. Fastest names the lowest verified mean, or a tie when the paired-ratio interval of the two lowest means contains one; default is the library's automatic selection for that mode.

Seed: 20260906; CPU: 11; status: ok. Commit: unknown (not a git checkout). Build options, source and executable hashes, corpus hashes, and raw observations are in the accompanying JSON.

## Batch products (ns per product)

| N | direct | cariow_cariowa | quasilinear | fastest mean | default |
| ---: | --- | --- | --- | --- | --- |
| 2 | 1.33 [1.32, 1.34] | 1.57 [1.55, 1.58] | 1.01 [1.01, 1.02] | quasilinear | quasilinear |
| 4 | 3.86 [3.83, 3.89] | 4.63 [4.59, 4.67] | 3.76 [3.73, 3.79] | quasilinear | quasilinear |
| 8 | 12.34 [12.23, 12.46] | 14.17 [14.06, 14.28] | 17.14 [17.00, 17.27] | direct | direct |
| 16 | 56.23 [55.71, 56.75] | 51.03 [50.56, 51.49] | 59.65 [59.20, 60.11] | cariow_cariowa | cariow_cariowa |
| 32 | 261.12 [259.02, 263.22] | 232.90 [230.80, 235.01] | 197.61 [196.08, 199.14] | quasilinear | quasilinear |
| 64 | 1137.71 [1124.30, 1151.13] | 1007.73 [997.99, 1017.46] | 508.62 [505.37, 511.87] | quasilinear | quasilinear |
| 128 | 4590.38 [4537.05, 4643.72] | 4930.62 [4879.23, 4982.01] | 1335.65 [1325.65, 1345.65] | quasilinear | quasilinear |
| 256 | 18222.80 [18063.78, 18381.81] | 17976.58 [17745.74, 18207.42] | 3321.41 [3296.51, 3346.32] | quasilinear | quasilinear |

## Single products (ns per product)

| N | direct | cariow_cariowa | quasilinear | fastest mean | default |
| ---: | --- | --- | --- | --- | --- |
| 2 | 16.12 [15.97, 16.28] | 17.07 [16.93, 17.21] | 15.54 [15.41, 15.67] | quasilinear | quasilinear |
| 4 | 18.18 [18.05, 18.32] | 21.09 [20.90, 21.28] | 19.16 [19.02, 19.31] | direct | direct |
| 8 | 28.02 [27.80, 28.23] | 40.49 [40.11, 40.87] | 38.44 [37.86, 39.02] | direct | direct |
| 16 | 80.03 [79.40, 80.67] | 109.16 [108.27, 110.04] | 122.24 [121.31, 123.18] | direct | direct |
| 32 | 297.03 [294.89, 299.16] | 456.02 [451.04, 461.00] | 235.65 [233.77, 237.54] | quasilinear | quasilinear |
| 64 | 1157.72 [1148.46, 1166.98] | 1896.64 [1879.24, 1914.03] | 558.55 [554.61, 562.49] | quasilinear | quasilinear |
| 128 | 4445.19 [4416.09, 4474.30] | 7948.83 [7790.39, 8107.26] | 1342.93 [1330.37, 1355.48] | quasilinear | quasilinear |
| 256 | 17622.96 [17500.67, 17745.26] | 29386.46 [28864.16, 29908.76] | 3535.75 [3504.44, 3567.07] | quasilinear | quasilinear |

## Paired ratios against quasilinear, batch mode

Geometric mean of per-block ratios (competitor / quasilinear) with 95% interval; values above 1 mean quasilinear was faster.

| N | direct / quasilinear | cariow_cariowa / quasilinear |
| ---: | --- | --- |
| 2 | 1.310 [1.296, 1.324] (0/48 blocks faster) | 1.542 [1.520, 1.564] (0/48 blocks faster) |
| 4 | 1.025 [1.015, 1.034] (13/48 blocks faster) | 1.229 [1.217, 1.242] (0/48 blocks faster) |
| 8 | 0.720 [0.711, 0.729] (48/48 blocks faster) | 0.827 [0.817, 0.837] (48/48 blocks faster) |
| 16 | 0.943 [0.930, 0.955] (40/48 blocks faster) | 0.855 [0.845, 0.866] (48/48 blocks faster) |
| 32 | 1.321 [1.306, 1.337] (0/48 blocks faster) | 1.178 [1.165, 1.192] (0/48 blocks faster) |
| 64 | 2.236 [2.204, 2.267] (0/48 blocks faster) | 1.981 [1.957, 2.004] (0/48 blocks faster) |
| 128 | 3.435 [3.386, 3.485] (0/48 blocks faster) | 3.690 [3.639, 3.743] (0/48 blocks faster) |
| 256 | 5.486 [5.425, 5.547] (0/48 blocks faster) | 5.409 [5.338, 5.481] (0/48 blocks faster) |

## Paired ratios against quasilinear, single mode

Geometric mean of per-block ratios (competitor / quasilinear) with 95% interval; values above 1 mean quasilinear was faster.

| N | direct / quasilinear | cariow_cariowa / quasilinear |
| ---: | --- | --- |
| 2 | 1.037 [1.023, 1.052] (12/48 blocks faster) | 1.099 [1.087, 1.110] (0/48 blocks faster) |
| 4 | 0.949 [0.939, 0.959] (46/48 blocks faster) | 1.100 [1.090, 1.111] (0/48 blocks faster) |
| 8 | 0.730 [0.718, 0.741] (48/48 blocks faster) | 1.054 [1.036, 1.072] (11/48 blocks faster) |
| 16 | 0.655 [0.647, 0.662] (48/48 blocks faster) | 0.893 [0.883, 0.903] (48/48 blocks faster) |
| 32 | 1.261 [1.246, 1.275] (0/48 blocks faster) | 1.935 [1.911, 1.958] (0/48 blocks faster) |
| 64 | 2.073 [2.048, 2.098] (0/48 blocks faster) | 3.395 [3.360, 3.430] (0/48 blocks faster) |
| 128 | 3.311 [3.278, 3.344] (0/48 blocks faster) | 5.910 [5.789, 6.033] (0/48 blocks faster) |
| 256 | 4.985 [4.935, 5.035] (0/48 blocks faster) | 8.302 [8.157, 8.450] (0/48 blocks faster) |

## Single-product time relative to batched time

Geometric mean of per-block ratios (single / batch) for each algorithm; values above 1 mean batching was faster.

| N | direct | cariow_cariowa | quasilinear |
| ---: | --- | --- | --- |
| 2 | 12.127 [11.994, 12.261] | 10.912 [10.764, 11.061] | 15.313 [15.095, 15.535] |
| 4 | 4.715 [4.662, 4.768] | 4.557 [4.508, 4.607] | 5.091 [5.040, 5.142] |
| 8 | 2.271 [2.245, 2.297] | 2.857 [2.821, 2.893] | 2.241 [2.201, 2.281] |
| 16 | 1.423 [1.407, 1.440] | 2.139 [2.116, 2.163] | 2.049 [2.029, 2.069] |
| 32 | 1.138 [1.126, 1.149] | 1.958 [1.929, 1.987] | 1.192 [1.181, 1.204] |
| 64 | 1.018 [1.004, 1.032] | 1.882 [1.858, 1.907] | 1.098 [1.089, 1.108] |
| 128 | 0.969 [0.956, 0.982] | 1.610 [1.578, 1.643] | 1.005 [0.992, 1.019] |
| 256 | 0.967 [0.957, 0.978] | 1.634 [1.606, 1.662] | 1.064 [1.053, 1.075] |

## Verification against the defining recursion

Maximum norm-scaled coordinate error |actual - oracle| / (||x|| ||y||) over the verified pairs, and mismatches against the tolerance 128 eps N, for each mode.

| N | direct | cariow_cariowa | quasilinear |
| ---: | --- | --- | --- |
| 2 | batch: 0.00e+00 (0/10000); single: 0.00e+00 (0/10000) | batch: 4.09e-16 (0/10000); single: 4.09e-16 (0/10000) | batch: 0.00e+00 (0/10000); single: 0.00e+00 (0/10000) |
| 4 | batch: 2.51e-16 (0/10000); single: 2.51e-16 (0/10000) | batch: 4.24e-16 (0/10000); single: 4.24e-16 (0/10000) | batch: 3.08e-16 (0/10000); single: 3.08e-16 (0/10000) |
| 8 | batch: 3.47e-16 (0/10000); single: 3.47e-16 (0/10000) | batch: 4.13e-16 (0/10000); single: 4.13e-16 (0/10000) | batch: 4.28e-16 (0/10000); single: 4.28e-16 (0/10000) |
| 16 | batch: 4.08e-16 (0/10000); single: 4.08e-16 (0/10000) | batch: 3.88e-16 (0/10000); single: 3.88e-16 (0/10000) | batch: 3.48e-16 (0/10000); single: 3.48e-16 (0/10000) |
| 32 | batch: 2.21e-16 (0/10000); single: 2.21e-16 (0/10000) | batch: 3.63e-16 (0/10000); single: 3.63e-16 (0/10000) | batch: 3.04e-16 (0/10000); single: 3.04e-16 (0/10000) |
| 64 | batch: 1.81e-16 (0/10000); single: 1.81e-16 (0/10000) | batch: 4.35e-16 (0/10000); single: 4.35e-16 (0/10000) | batch: 2.95e-16 (0/10000); single: 2.95e-16 (0/10000) |
| 128 | batch: 1.45e-16 (0/10000); single: 1.45e-16 (0/10000) | batch: 2.12e-16 (0/10000); single: 2.12e-16 (0/10000) | batch: 2.26e-16 (0/10000); single: 2.26e-16 (0/10000) |
| 256 | batch: 9.90e-17 (0/10000); single: 9.90e-17 (0/10000) | batch: 1.80e-16 (0/10000); single: 1.80e-16 (0/10000) | batch: 3.29e-16 (0/10000); single: 3.29e-16 (0/10000) |
