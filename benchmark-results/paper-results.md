# fastCD multiplication benchmark

Host: Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz (x86_64); gcc 16.1.1 20260728; 24 blocks of 10000 pairs per dimension; each cell is the mean over blocks of ns per product with a two-sided 95% Student-t interval.

Batch mode times one validated batch call per corpus pass (two products per SIMD instruction where the build supports it); single mode times one validated single-product call per pair.  Python object overhead is excluded. Verification uses an independent float64 oracle, not an exact product. Fastest names the lowest verified mean, or a tie when the paired-ratio interval of the two lowest means contains one; default is the library's automatic selection for that mode.

Seed: 20260906; CPU: 0; status: ok. Commit: 14bf30548487ccf35939de89cb7f7e98a1dab5c6 (clean tree). Build options, source and executable hashes, corpus hashes, and raw observations are in the accompanying JSON.

## Batch products (ns per product)

| N | direct | cariow_cariowa | quasilinear | fastest mean | default |
| ---: | --- | --- | --- | --- | --- |
| 2 | 1.34 [1.33, 1.35] | 1.53 [1.50, 1.56] | 1.02 [1.01, 1.03] | quasilinear | quasilinear |
| 4 | 3.87 [3.78, 3.97] | 4.58 [4.49, 4.66] | 3.80 [3.71, 3.89] | tie (quasilinear, direct) | quasilinear |
| 8 | 11.89 [11.68, 12.09] | 13.53 [13.25, 13.82] | 16.22 [15.96, 16.49] | direct | direct |
| 16 | 54.68 [53.91, 55.45] | 49.38 [48.60, 50.16] | 56.95 [56.04, 57.85] | cariow_cariowa | cariow_cariowa |
| 32 | 257.63 [252.67, 262.58] | 221.01 [217.27, 224.76] | 193.27 [191.00, 195.54] | quasilinear | quasilinear |
| 64 | 1124.05 [1101.38, 1146.72] | 947.05 [935.01, 959.09] | 492.65 [483.39, 501.92] | quasilinear | quasilinear |
| 128 | 4468.17 [4362.50, 4573.85] | 4908.83 [4771.93, 5045.73] | 1322.56 [1304.49, 1340.63] | quasilinear | quasilinear |
| 256 | 17757.92 [17535.51, 17980.33] | 17663.82 [17465.57, 17862.08] | 3149.78 [3101.79, 3197.78] | quasilinear | quasilinear |
| 512 | 69138.03 [68183.42, 70092.64] | 65786.36 [64987.60, 66585.12] | 7485.61 [7376.06, 7595.15] | quasilinear | quasilinear |
| 1024 | 273693.44 [270572.17, 276814.71] | 408225.87 [406200.31, 410251.42] | 17483.64 [17250.55, 17716.73] | quasilinear | quasilinear |

## Single products (ns per product)

| N | direct | cariow_cariowa | quasilinear | fastest mean | default |
| ---: | --- | --- | --- | --- | --- |
| 2 | 16.90 [15.83, 17.96] | 17.92 [16.97, 18.87] | 15.69 [15.45, 15.92] | quasilinear | quasilinear |
| 4 | 19.33 [18.06, 20.60] | 22.15 [20.77, 23.52] | 19.89 [18.76, 21.02] | tie (direct, quasilinear) | direct |
| 8 | 27.78 [27.35, 28.21] | 39.42 [38.68, 40.16] | 36.45 [35.45, 37.46] | direct | direct |
| 16 | 78.06 [76.52, 79.60] | 103.90 [102.22, 105.58] | 119.59 [117.61, 121.57] | direct | direct |
| 32 | 286.89 [282.76, 291.03] | 439.21 [430.02, 448.41] | 225.86 [221.96, 229.76] | quasilinear | quasilinear |
| 64 | 1121.00 [1105.23, 1136.76] | 1851.90 [1819.35, 1884.44] | 542.68 [534.57, 550.79] | quasilinear | quasilinear |
| 128 | 4390.47 [4334.93, 4446.01] | 7833.73 [7680.01, 7987.44] | 1334.23 [1316.20, 1352.27] | quasilinear | quasilinear |
| 256 | 17314.96 [17119.93, 17509.99] | 28830.79 [28140.17, 29521.40] | 3381.47 [3323.64, 3439.30] | quasilinear | quasilinear |
| 512 | 69029.59 [68325.12, 69734.07] | 107873.78 [106494.36, 109253.20] | 7913.86 [7851.24, 7976.47] | quasilinear | quasilinear |
| 1024 | 274511.31 [272437.03, 276585.59] | 411259.09 [406462.40, 416055.78] | 17525.01 [17348.44, 17701.58] | quasilinear | quasilinear |

## Paired ratios against quasilinear, batch mode

Geometric mean of per-block ratios (competitor / quasilinear) with 95% interval; values above 1 mean quasilinear was faster.

| N | direct / quasilinear | cariow_cariowa / quasilinear |
| ---: | --- | --- |
| 2 | 1.316 [1.301, 1.331] (0/24 blocks faster) | 1.502 [1.474, 1.531] (0/24 blocks faster) |
| 4 | 1.019 [0.984, 1.055] (11/24 blocks faster) | 1.205 [1.170, 1.241] (0/24 blocks faster) |
| 8 | 0.733 [0.718, 0.748] (24/24 blocks faster) | 0.834 [0.816, 0.852] (24/24 blocks faster) |
| 16 | 0.960 [0.936, 0.985] (17/24 blocks faster) | 0.867 [0.846, 0.889] (24/24 blocks faster) |
| 32 | 1.332 [1.307, 1.358] (0/24 blocks faster) | 1.143 [1.124, 1.162] (0/24 blocks faster) |
| 64 | 2.281 [2.225, 2.339] (0/24 blocks faster) | 1.923 [1.873, 1.975] (0/24 blocks faster) |
| 128 | 3.375 [3.280, 3.473] (0/24 blocks faster) | 3.706 [3.593, 3.822] (0/24 blocks faster) |
| 256 | 5.639 [5.541, 5.738] (0/24 blocks faster) | 5.610 [5.513, 5.708] (0/24 blocks faster) |
| 512 | 9.237 [9.088, 9.388] (0/24 blocks faster) | 8.790 [8.640, 8.943] (0/24 blocks faster) |
| 1024 | 15.656 [15.421, 15.895] (0/24 blocks faster) | 23.359 [23.069, 23.652] (0/24 blocks faster) |

## Paired ratios against quasilinear, single mode

Geometric mean of per-block ratios (competitor / quasilinear) with 95% interval; values above 1 mean quasilinear was faster.

| N | direct / quasilinear | cariow_cariowa / quasilinear |
| ---: | --- | --- |
| 2 | 1.070 [1.020, 1.122] (1/24 blocks faster) | 1.136 [1.088, 1.186] (1/24 blocks faster) |
| 4 | 0.970 [0.922, 1.020] (18/24 blocks faster) | 1.112 [1.057, 1.169] (2/24 blocks faster) |
| 8 | 0.763 [0.739, 0.788] (24/24 blocks faster) | 1.083 [1.051, 1.115] (3/24 blocks faster) |
| 16 | 0.653 [0.641, 0.664] (24/24 blocks faster) | 0.869 [0.847, 0.891] (24/24 blocks faster) |
| 32 | 1.271 [1.249, 1.292] (0/24 blocks faster) | 1.944 [1.892, 1.997] (0/24 blocks faster) |
| 64 | 2.066 [2.019, 2.114] (0/24 blocks faster) | 3.412 [3.339, 3.486] (0/24 blocks faster) |
| 128 | 3.291 [3.222, 3.362] (0/24 blocks faster) | 5.868 [5.710, 6.031] (0/24 blocks faster) |
| 256 | 5.123 [5.010, 5.238] (0/24 blocks faster) | 8.521 [8.286, 8.762] (0/24 blocks faster) |
| 512 | 8.722 [8.625, 8.819] (0/24 blocks faster) | 13.627 [13.444, 13.813] (0/24 blocks faster) |
| 1024 | 15.666 [15.478, 15.857] (0/24 blocks faster) | 23.465 [23.136, 23.799] (0/24 blocks faster) |

## Single-product time relative to batched time

Geometric mean of per-block ratios (single / batch) for each algorithm; values above 1 mean batching was faster.

| N | direct | cariow_cariowa | quasilinear |
| ---: | --- | --- | --- |
| 2 | 12.516 [11.884, 13.182] | 11.647 [11.086, 12.236] | 15.404 [15.219, 15.591] |
| 4 | 4.951 [4.661, 5.259] | 4.800 [4.547, 5.067] | 5.201 [4.987, 5.424] |
| 8 | 2.338 [2.283, 2.393] | 2.913 [2.837, 2.991] | 2.244 [2.176, 2.314] |
| 16 | 1.427 [1.390, 1.465] | 2.104 [2.061, 2.148] | 2.100 [2.052, 2.149] |
| 32 | 1.114 [1.090, 1.139] | 1.986 [1.930, 2.044] | 1.168 [1.144, 1.193] |
| 64 | 0.998 [0.977, 1.019] | 1.955 [1.922, 1.988] | 1.102 [1.076, 1.128] |
| 128 | 0.984 [0.959, 1.009] | 1.597 [1.545, 1.651] | 1.009 [0.990, 1.028] |
| 256 | 0.975 [0.962, 0.989] | 1.630 [1.594, 1.668] | 1.073 [1.046, 1.101] |
| 512 | 0.999 [0.985, 1.013] | 1.640 [1.621, 1.659] | 1.058 [1.043, 1.073] |
| 1024 | 1.003 [0.990, 1.016] | 1.007 [0.994, 1.020] | 1.003 [0.989, 1.016] |

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
| 512 | batch: 8.20e-17 (0/10000); single: 8.20e-17 (0/10000) | batch: 1.65e-16 (0/10000); single: 1.65e-16 (0/10000) | batch: 2.28e-16 (0/10000); single: 2.28e-16 (0/10000) |
| 1024 | batch: 6.15e-17 (0/10000); single: 6.15e-17 (0/10000) | batch: 1.65e-16 (0/10000); single: 1.65e-16 (0/10000) | batch: 2.90e-16 (0/10000); single: 2.90e-16 (0/10000) |
