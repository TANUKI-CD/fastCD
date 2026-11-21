# fastCD multiplication benchmark

Host: Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz (x86_64); gcc 16.2.1 20260810; 24 blocks of 10000 pairs per dimension; each cell is the mean over blocks of ns per product with a two-sided 95% Student-t interval.

Batch mode times one validated batch call per corpus pass (two products per SIMD instruction where the build supports it); single mode times one validated single-product call per pair.  Python object overhead is excluded. Verification uses an independent float64 oracle, not an exact product. Fastest names the lowest verified mean, or a tie when the paired-ratio interval of the two lowest means contains one; default is the library's automatic selection for that mode.

Seed: 20260906; CPU: 2; status: ok. Commit: 93c9bd8ab1c3bcfdb16c76e5ad55fa487d741a18 (clean tree). Build options, source and executable hashes, corpus hashes, and raw observations are in the accompanying JSON.

## Batch products (ns per product)

| N | direct | cariow_cariowa | quasilinear | fastest mean | default |
| ---: | --- | --- | --- | --- | --- |
| 2 | 1.32 [1.30, 1.34] | 1.52 [1.49, 1.55] | 1.02 [1.00, 1.04] | quasilinear | quasilinear |
| 4 | 3.84 [3.80, 3.88] | 4.65 [4.60, 4.69] | 3.68 [3.62, 3.73] | quasilinear | quasilinear |
| 8 | 12.18 [11.99, 12.37] | 14.05 [13.85, 14.25] | 17.59 [17.34, 17.85] | direct | direct |
| 16 | 54.79 [53.87, 55.71] | 49.92 [48.69, 51.14] | 52.41 [51.06, 53.76] | cariow_cariowa | cariow_cariowa |
| 32 | 256.56 [252.77, 260.35] | 227.38 [221.50, 233.25] | 184.50 [182.24, 186.76] | quasilinear | quasilinear |
| 64 | 1133.91 [1109.34, 1158.48] | 973.51 [951.64, 995.38] | 465.99 [460.22, 471.75] | quasilinear | quasilinear |
| 128 | 4530.78 [4453.12, 4608.44] | 5006.53 [4926.07, 5086.99] | 1242.81 [1221.67, 1263.95] | quasilinear | quasilinear |
| 256 | 17665.04 [17305.84, 18024.25] | 17695.42 [17442.11, 17948.73] | 3046.46 [2999.09, 3093.83] | quasilinear | quasilinear |
| 512 | 69956.80 [69154.63, 70758.97] | 68578.92 [67922.14, 69235.70] | 6920.47 [6825.76, 7015.18] | quasilinear | quasilinear |
| 1024 | 275135.85 [273397.86, 276873.84] | 433524.98 [419774.76, 447275.19] | 16209.21 [16034.88, 16383.55] | quasilinear | quasilinear |

## Single products (ns per product)

| N | direct | cariow_cariowa | quasilinear | fastest mean | default |
| ---: | --- | --- | --- | --- | --- |
| 2 | 15.76 [15.52, 16.01] | 16.76 [16.56, 16.95] | 15.30 [15.12, 15.48] | quasilinear | quasilinear |
| 4 | 17.75 [17.48, 18.02] | 20.22 [19.97, 20.48] | 18.08 [17.91, 18.26] | direct | direct |
| 8 | 28.22 [27.86, 28.58] | 38.16 [37.71, 38.60] | 36.25 [35.84, 36.67] | direct | direct |
| 16 | 76.28 [74.68, 77.88] | 110.28 [108.18, 112.37] | 108.25 [106.07, 110.43] | direct | direct |
| 32 | 298.82 [295.63, 302.01] | 432.14 [423.85, 440.43] | 221.13 [217.79, 224.47] | quasilinear | quasilinear |
| 64 | 1163.54 [1146.73, 1180.35] | 1835.74 [1797.03, 1874.46] | 518.34 [511.99, 524.69] | quasilinear | quasilinear |
| 128 | 4495.22 [4430.30, 4560.14] | 7910.79 [7706.85, 8114.74] | 1333.57 [1308.25, 1358.88] | quasilinear | quasilinear |
| 256 | 17445.55 [17268.47, 17622.63] | 29214.67 [28321.53, 30107.80] | 3244.06 [3196.11, 3292.01] | quasilinear | quasilinear |
| 512 | 69310.72 [68887.00, 69734.44] | 108032.81 [107477.81, 108587.82] | 7427.95 [7327.56, 7528.34] | quasilinear | quasilinear |
| 1024 | 277013.17 [274746.54, 279279.79] | 419988.45 [417575.06, 422401.83] | 16258.80 [16093.16, 16424.44] | quasilinear | quasilinear |

## Paired ratios against quasilinear, batch mode

Geometric mean of per-block ratios (competitor / quasilinear) with 95% interval; values above 1 mean quasilinear was faster.

| N | direct / quasilinear | cariow_cariowa / quasilinear |
| ---: | --- | --- |
| 2 | 1.303 [1.273, 1.334] (0/24 blocks faster) | 1.495 [1.459, 1.531] (0/24 blocks faster) |
| 4 | 1.044 [1.026, 1.063] (4/24 blocks faster) | 1.263 [1.240, 1.287] (0/24 blocks faster) |
| 8 | 0.692 [0.676, 0.709] (24/24 blocks faster) | 0.799 [0.784, 0.813] (24/24 blocks faster) |
| 16 | 1.047 [1.014, 1.080] (6/24 blocks faster) | 0.953 [0.921, 0.985] (17/24 blocks faster) |
| 32 | 1.390 [1.367, 1.414] (0/24 blocks faster) | 1.231 [1.205, 1.257] (0/24 blocks faster) |
| 64 | 2.431 [2.373, 2.491] (0/24 blocks faster) | 2.087 [2.037, 2.138] (0/24 blocks faster) |
| 128 | 3.645 [3.558, 3.735] (0/24 blocks faster) | 4.029 [3.951, 4.108] (0/24 blocks faster) |
| 256 | 5.796 [5.673, 5.922] (0/24 blocks faster) | 5.809 [5.699, 5.922] (0/24 blocks faster) |
| 512 | 10.110 [9.932, 10.292] (0/24 blocks faster) | 9.912 [9.736, 10.092] (0/24 blocks faster) |
| 1024 | 16.977 [16.760, 17.197] (0/24 blocks faster) | 26.690 [25.868, 27.539] (0/24 blocks faster) |

## Paired ratios against quasilinear, single mode

Geometric mean of per-block ratios (competitor / quasilinear) with 95% interval; values above 1 mean quasilinear was faster.

| N | direct / quasilinear | cariow_cariowa / quasilinear |
| ---: | --- | --- |
| 2 | 1.030 [1.014, 1.046] (4/24 blocks faster) | 1.095 [1.079, 1.112] (0/24 blocks faster) |
| 4 | 0.981 [0.970, 0.993] (20/24 blocks faster) | 1.118 [1.100, 1.137] (0/24 blocks faster) |
| 8 | 0.778 [0.766, 0.791] (24/24 blocks faster) | 1.052 [1.035, 1.070] (2/24 blocks faster) |
| 16 | 0.705 [0.684, 0.726] (24/24 blocks faster) | 1.019 [0.988, 1.050] (9/24 blocks faster) |
| 32 | 1.352 [1.328, 1.376] (0/24 blocks faster) | 1.953 [1.899, 2.009] (0/24 blocks faster) |
| 64 | 2.244 [2.218, 2.271] (0/24 blocks faster) | 3.539 [3.451, 3.629] (0/24 blocks faster) |
| 128 | 3.372 [3.277, 3.470] (0/24 blocks faster) | 5.928 [5.746, 6.115] (0/24 blocks faster) |
| 256 | 5.379 [5.297, 5.464] (0/24 blocks faster) | 8.990 [8.723, 9.265] (0/24 blocks faster) |
| 512 | 9.335 [9.209, 9.462] (0/24 blocks faster) | 14.550 [14.351, 14.752] (0/24 blocks faster) |
| 1024 | 17.039 [16.826, 17.256] (0/24 blocks faster) | 25.836 [25.560, 26.115] (0/24 blocks faster) |

## Single-product time relative to batched time

Geometric mean of per-block ratios (single / batch) for each algorithm; values above 1 mean batching was faster.

| N | direct | cariow_cariowa | quasilinear |
| ---: | --- | --- | --- |
| 2 | 11.905 [11.708, 12.105] | 11.038 [10.819, 11.261] | 15.063 [14.798, 15.332] |
| 4 | 4.620 [4.532, 4.710] | 4.353 [4.282, 4.425] | 4.918 [4.827, 5.009] |
| 8 | 2.318 [2.279, 2.357] | 2.717 [2.670, 2.765] | 2.061 [2.019, 2.104] |
| 16 | 1.392 [1.351, 1.434] | 2.211 [2.153, 2.270] | 2.067 [2.001, 2.135] |
| 32 | 1.165 [1.143, 1.188] | 1.902 [1.835, 1.972] | 1.198 [1.181, 1.216] |
| 64 | 1.027 [0.998, 1.057] | 1.886 [1.830, 1.944] | 1.112 [1.096, 1.129] |
| 128 | 0.992 [0.969, 1.016] | 1.579 [1.536, 1.623] | 1.073 [1.045, 1.102] |
| 256 | 0.988 [0.967, 1.010] | 1.648 [1.601, 1.697] | 1.065 [1.039, 1.092] |
| 512 | 0.991 [0.979, 1.003] | 1.576 [1.558, 1.593] | 1.073 [1.050, 1.097] |
| 1024 | 1.007 [0.998, 1.015] | 0.971 [0.945, 0.998] | 1.003 [0.988, 1.018] |

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
