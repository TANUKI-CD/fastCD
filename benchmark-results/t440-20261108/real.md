# fastCD multiplication benchmark

Host: Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz (x86_64); gcc 16.2.1 20260810; 24 blocks of 10000 pairs per dimension; each cell is the mean over blocks of ns per product with a two-sided 95% Student-t interval.

Batch mode times one validated batch call per corpus pass (two products per SIMD instruction where the build supports it); single mode times one validated single-product call per pair.  Python object overhead is excluded. Verification uses an independent float64 oracle, not an exact product. Fastest names the lowest verified mean, or a tie when the paired-ratio interval of the two lowest means contains one; default is the library's automatic selection for that mode.

Seed: 20260906; CPU: 2; status: ok. Commit: 0b1d124c4c22ea57d9b8f4e45dfb9226bcd61c6f (clean tree). Build options, source and executable hashes, corpus hashes, and raw observations are in the accompanying JSON.

## Batch products (ns per product)

| N | direct | cariow_cariowa | quasilinear | fastest mean | default |
| ---: | --- | --- | --- | --- | --- |
| 2 | 1.32 [1.31, 1.34] | 1.53 [1.49, 1.56] | 1.01 [1.00, 1.02] | quasilinear | quasilinear |
| 4 | 3.79 [3.74, 3.85] | 4.54 [4.45, 4.63] | 3.62 [3.57, 3.67] | quasilinear | quasilinear |
| 8 | 12.30 [12.08, 12.51] | 14.12 [13.96, 14.28] | 17.77 [17.58, 17.96] | direct | direct |
| 16 | 56.63 [55.76, 57.50] | 51.54 [50.42, 52.66] | 53.49 [52.70, 54.29] | cariow_cariowa | cariow_cariowa |
| 32 | 268.06 [265.66, 270.45] | 239.78 [236.46, 243.11] | 191.61 [189.65, 193.56] | quasilinear | quasilinear |
| 64 | 1175.85 [1154.35, 1197.36] | 1050.38 [1034.47, 1066.29] | 468.89 [461.90, 475.89] | quasilinear | quasilinear |
| 128 | 4661.33 [4575.78, 4746.88] | 5021.51 [4924.28, 5118.74] | 1247.30 [1222.27, 1272.32] | quasilinear | quasilinear |
| 256 | 17722.80 [17415.97, 18029.63] | 18296.21 [17885.40, 18707.02] | 2987.79 [2940.73, 3034.85] | quasilinear | quasilinear |
| 512 | 71605.70 [70978.80, 72232.60] | 68716.18 [67751.71, 69680.64] | 6991.64 [6896.81, 7086.48] | quasilinear | quasilinear |
| 1024 | 279025.26 [277099.70, 280950.83] | 434500.24 [421724.66, 447275.82] | 16304.11 [16061.72, 16546.49] | quasilinear | quasilinear |

## Single products (ns per product)

| N | direct | cariow_cariowa | quasilinear | fastest mean | default |
| ---: | --- | --- | --- | --- | --- |
| 2 | 16.13 [15.97, 16.30] | 17.01 [16.87, 17.15] | 16.44 [16.16, 16.71] | direct | quasilinear |
| 4 | 18.81 [17.31, 20.31] | 19.66 [19.18, 20.13] | 18.35 [18.05, 18.64] | tie (quasilinear, direct) | direct |
| 8 | 28.28 [27.47, 29.10] | 40.88 [40.29, 41.48] | 34.28 [33.44, 35.13] | direct | direct |
| 16 | 81.43 [80.08, 82.79] | 113.49 [111.32, 115.65] | 106.08 [104.46, 107.69] | direct | direct |
| 32 | 305.74 [303.12, 308.36] | 468.23 [461.24, 475.21] | 230.21 [228.35, 232.08] | quasilinear | quasilinear |
| 64 | 1177.37 [1156.36, 1198.38] | 1945.77 [1915.72, 1975.81] | 529.31 [523.63, 534.99] | quasilinear | quasilinear |
| 128 | 4484.06 [4411.25, 4556.88] | 7997.66 [7816.10, 8179.21] | 1321.73 [1299.05, 1344.40] | quasilinear | quasilinear |
| 256 | 17600.81 [17373.78, 17827.85] | 29126.42 [28120.72, 30132.11] | 3245.59 [3194.01, 3297.16] | quasilinear | quasilinear |
| 512 | 69125.15 [68537.59, 69712.70] | 109750.70 [108906.17, 110595.23] | 7303.68 [7212.12, 7395.25] | quasilinear | quasilinear |
| 1024 | 277562.64 [275981.92, 279143.35] | 428055.56 [425148.91, 430962.20] | 16267.63 [16117.20, 16418.06] | quasilinear | quasilinear |

## Paired ratios against quasilinear, batch mode

Geometric mean of per-block ratios (competitor / quasilinear) with 95% interval; values above 1 mean quasilinear was faster.

| N | direct / quasilinear | cariow_cariowa / quasilinear |
| ---: | --- | --- |
| 2 | 1.305 [1.289, 1.322] (0/24 blocks faster) | 1.505 [1.473, 1.539] (0/24 blocks faster) |
| 4 | 1.048 [1.027, 1.070] (2/24 blocks faster) | 1.254 [1.220, 1.289] (0/24 blocks faster) |
| 8 | 0.692 [0.677, 0.707] (24/24 blocks faster) | 0.795 [0.784, 0.806] (24/24 blocks faster) |
| 16 | 1.059 [1.037, 1.081] (3/24 blocks faster) | 0.963 [0.938, 0.988] (18/24 blocks faster) |
| 32 | 1.399 [1.382, 1.417] (0/24 blocks faster) | 1.251 [1.231, 1.272] (0/24 blocks faster) |
| 64 | 2.507 [2.450, 2.566] (0/24 blocks faster) | 2.240 [2.199, 2.282] (0/24 blocks faster) |
| 128 | 3.738 [3.623, 3.855] (0/24 blocks faster) | 4.026 [3.918, 4.137] (0/24 blocks faster) |
| 256 | 5.931 [5.784, 6.081] (0/24 blocks faster) | 6.120 [5.978, 6.264] (0/24 blocks faster) |
| 512 | 10.245 [10.107, 10.384] (0/24 blocks faster) | 9.828 [9.637, 10.023] (0/24 blocks faster) |
| 1024 | 17.122 [16.830, 17.418] (0/24 blocks faster) | 26.613 [26.055, 27.183] (0/24 blocks faster) |

## Paired ratios against quasilinear, single mode

Geometric mean of per-block ratios (competitor / quasilinear) with 95% interval; values above 1 mean quasilinear was faster.

| N | direct / quasilinear | cariow_cariowa / quasilinear |
| ---: | --- | --- |
| 2 | 0.982 [0.965, 0.999] (15/24 blocks faster) | 1.035 [1.020, 1.051] (4/24 blocks faster) |
| 4 | 1.014 [0.956, 1.075] (16/24 blocks faster) | 1.070 [1.038, 1.104] (4/24 blocks faster) |
| 8 | 0.825 [0.801, 0.849] (24/24 blocks faster) | 1.194 [1.167, 1.221] (0/24 blocks faster) |
| 16 | 0.768 [0.754, 0.782] (24/24 blocks faster) | 1.069 [1.045, 1.095] (3/24 blocks faster) |
| 32 | 1.328 [1.311, 1.345] (0/24 blocks faster) | 2.033 [1.995, 2.071] (0/24 blocks faster) |
| 64 | 2.223 [2.184, 2.263] (0/24 blocks faster) | 3.675 [3.614, 3.737] (0/24 blocks faster) |
| 128 | 3.393 [3.314, 3.473] (0/24 blocks faster) | 6.048 [5.859, 6.242] (0/24 blocks faster) |
| 256 | 5.424 [5.336, 5.514] (0/24 blocks faster) | 8.956 [8.645, 9.278] (0/24 blocks faster) |
| 512 | 9.467 [9.317, 9.618] (0/24 blocks faster) | 15.031 [14.833, 15.231] (0/24 blocks faster) |
| 1024 | 17.065 [16.867, 17.264] (0/24 blocks faster) | 26.316 [25.972, 26.665] (0/24 blocks faster) |

## Single-product time relative to batched time

Geometric mean of per-block ratios (single / batch) for each algorithm; values above 1 mean batching was faster.

| N | direct | cariow_cariowa | quasilinear |
| ---: | --- | --- | --- |
| 2 | 12.194 [12.044, 12.345] | 11.147 [10.892, 11.407] | 16.209 [15.933, 16.490] |
| 4 | 4.902 [4.606, 5.218] | 4.326 [4.173, 4.485] | 5.070 [4.953, 5.189] |
| 8 | 2.297 [2.230, 2.366] | 2.895 [2.839, 2.951] | 1.927 [1.874, 1.982] |
| 16 | 1.438 [1.404, 1.472] | 2.203 [2.146, 2.260] | 1.983 [1.950, 2.017] |
| 32 | 1.141 [1.126, 1.155] | 1.952 [1.909, 1.997] | 1.202 [1.186, 1.217] |
| 64 | 1.001 [0.978, 1.026] | 1.852 [1.813, 1.893] | 1.129 [1.109, 1.150] |
| 128 | 0.962 [0.939, 0.986] | 1.592 [1.552, 1.634] | 1.060 [1.031, 1.090] |
| 256 | 0.993 [0.974, 1.013] | 1.590 [1.525, 1.657] | 1.086 [1.063, 1.110] |
| 512 | 0.965 [0.954, 0.977] | 1.598 [1.574, 1.621] | 1.045 [1.025, 1.065] |
| 1024 | 0.995 [0.986, 1.004] | 0.987 [0.960, 1.014] | 0.998 [0.982, 1.015] |

## Verification against the defining recursion

Maximum norm-scaled coordinate error |actual - oracle| / (||x|| ||y||) over the verified pairs, and mismatches against the tolerance 128 eps N, for each mode.

| N | direct | cariow_cariowa | quasilinear |
| ---: | --- | --- | --- |
| 2 | batch: 0.00e+00 (0/10000); single: 0.00e+00 (0/10000) | batch: 4.09e-16 (0/10000); single: 4.09e-16 (0/10000) | batch: 0.00e+00 (0/10000); single: 0.00e+00 (0/10000) |
| 4 | batch: 2.51e-16 (0/10000); single: 2.51e-16 (0/10000) | batch: 4.24e-16 (0/10000); single: 4.24e-16 (0/10000) | batch: 3.08e-16 (0/10000); single: 3.08e-16 (0/10000) |
| 8 | batch: 3.47e-16 (0/10000); single: 3.47e-16 (0/10000) | batch: 4.13e-16 (0/10000); single: 4.13e-16 (0/10000) | batch: 4.48e-16 (0/10000); single: 4.48e-16 (0/10000) |
| 16 | batch: 4.08e-16 (0/10000); single: 4.08e-16 (0/10000) | batch: 3.88e-16 (0/10000); single: 3.88e-16 (0/10000) | batch: 5.48e-16 (0/10000); single: 5.48e-16 (0/10000) |
| 32 | batch: 2.21e-16 (0/10000); single: 2.21e-16 (0/10000) | batch: 3.63e-16 (0/10000); single: 3.63e-16 (0/10000) | batch: 3.63e-16 (0/10000); single: 3.63e-16 (0/10000) |
| 64 | batch: 1.81e-16 (0/10000); single: 1.81e-16 (0/10000) | batch: 4.35e-16 (0/10000); single: 4.35e-16 (0/10000) | batch: 2.29e-16 (0/10000); single: 2.29e-16 (0/10000) |
| 128 | batch: 1.45e-16 (0/10000); single: 1.45e-16 (0/10000) | batch: 2.12e-16 (0/10000); single: 2.12e-16 (0/10000) | batch: 1.69e-16 (0/10000); single: 1.69e-16 (0/10000) |
| 256 | batch: 9.90e-17 (0/10000); single: 9.90e-17 (0/10000) | batch: 1.80e-16 (0/10000); single: 1.80e-16 (0/10000) | batch: 1.38e-16 (0/10000); single: 1.38e-16 (0/10000) |
| 512 | batch: 8.20e-17 (0/10000); single: 8.20e-17 (0/10000) | batch: 1.24e-16 (0/10000); single: 1.24e-16 (0/10000) | batch: 1.12e-16 (0/10000); single: 1.12e-16 (0/10000) |
| 1024 | batch: 6.15e-17 (0/10000); single: 6.15e-17 (0/10000) | batch: 9.61e-17 (0/10000); single: 9.61e-17 (0/10000) | batch: 7.84e-17 (0/10000); single: 7.84e-17 (0/10000) |
