# fastCD multiplication benchmark

Host: Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz (x86_64); gcc 16.2.1 20260810; 24 blocks of 10000 pairs per dimension; each cell is the mean over blocks of ns per product with a two-sided 95% Student-t interval.

Batch mode times one validated batch call per corpus pass (two products per SIMD instruction where the build supports it); single mode times one validated single-product call per pair.  Python object overhead is excluded. Verification uses an independent float64 oracle, not an exact product. Fastest names the lowest verified mean, or a tie when the paired-ratio interval of the two lowest means contains one; default is the library's automatic selection for that mode.

Seed: 20260906; CPU: 2; status: ok. Commit: 33addc944b8cf47d07b9eec59541ffcbdda441e7 (clean tree). Build options, source and executable hashes, corpus hashes, and raw observations are in the accompanying JSON.

## Batch products (ns per product)

| N | direct | cariow_cariowa | quasilinear | fastest mean | default |
| ---: | --- | --- | --- | --- | --- |
| 2 | 1.33 [1.31, 1.35] | 1.59 [1.57, 1.62] | 1.03 [1.02, 1.04] | quasilinear | quasilinear |
| 4 | 3.94 [3.89, 3.98] | 4.68 [4.62, 4.73] | 3.81 [3.75, 3.86] | quasilinear | quasilinear |
| 8 | 12.74 [12.64, 12.85] | 14.57 [14.36, 14.79] | 18.28 [18.00, 18.56] | direct | direct |
| 16 | 57.40 [56.52, 58.27] | 52.74 [51.68, 53.80] | 55.10 [54.16, 56.04] | cariow_cariowa | cariow_cariowa |
| 32 | 268.23 [264.09, 272.37] | 237.26 [233.80, 240.73] | 190.71 [188.03, 193.39] | quasilinear | quasilinear |
| 64 | 1171.27 [1141.31, 1201.22] | 1013.51 [990.61, 1036.42] | 478.38 [468.63, 488.13] | quasilinear | quasilinear |
| 128 | 4659.72 [4568.53, 4750.90] | 5194.87 [5030.70, 5359.04] | 1282.23 [1255.74, 1308.72] | quasilinear | quasilinear |
| 256 | 18411.96 [18131.73, 18692.19] | 18509.70 [18092.51, 18926.89] | 3072.16 [3024.09, 3120.22] | quasilinear | quasilinear |
| 512 | 70453.07 [69518.40, 71387.74] | 70924.53 [69837.36, 72011.69] | 6940.28 [6843.35, 7037.21] | quasilinear | quasilinear |
| 1024 | 276258.74 [274098.13, 278419.34] | 426435.28 [423564.06, 429306.50] | 16196.47 [15993.55, 16399.39] | quasilinear | quasilinear |

## Single products (ns per product)

| N | direct | cariow_cariowa | quasilinear | fastest mean | default |
| ---: | --- | --- | --- | --- | --- |
| 2 | 15.95 [15.70, 16.19] | 16.52 [16.31, 16.73] | 16.15 [15.81, 16.50] | tie (direct, quasilinear) | quasilinear |
| 4 | 17.70 [17.47, 17.93] | 19.67 [19.28, 20.05] | 18.36 [18.07, 18.66] | direct | direct |
| 8 | 28.36 [26.88, 29.83] | 40.41 [39.93, 40.88] | 34.56 [33.87, 35.25] | direct | direct |
| 16 | 86.52 [83.22, 89.81] | 113.79 [111.14, 116.43] | 106.36 [104.37, 108.35] | direct | direct |
| 32 | 304.51 [301.10, 307.91] | 467.38 [457.54, 477.22] | 230.14 [226.99, 233.29] | quasilinear | quasilinear |
| 64 | 1174.60 [1136.91, 1212.29] | 1921.76 [1874.63, 1968.89] | 527.82 [519.25, 536.40] | quasilinear | quasilinear |
| 128 | 4577.54 [4510.54, 4644.55] | 8450.41 [8127.34, 8773.48] | 1341.54 [1319.44, 1363.65] | quasilinear | quasilinear |
| 256 | 17751.24 [17557.41, 17945.07] | 29374.01 [28802.52, 29945.50] | 3284.33 [3228.72, 3339.93] | quasilinear | quasilinear |
| 512 | 69656.86 [69120.36, 70193.36] | 113587.77 [112582.42, 114593.12] | 7423.90 [7345.57, 7502.23] | quasilinear | quasilinear |
| 1024 | 276236.76 [274697.43, 277776.09] | 435957.96 [423208.54, 448707.38] | 16073.92 [15903.01, 16244.83] | quasilinear | quasilinear |

## Paired ratios against quasilinear, batch mode

Geometric mean of per-block ratios (competitor / quasilinear) with 95% interval; values above 1 mean quasilinear was faster.

| N | direct / quasilinear | cariow_cariowa / quasilinear |
| ---: | --- | --- |
| 2 | 1.290 [1.273, 1.307] (0/24 blocks faster) | 1.547 [1.526, 1.568] (0/24 blocks faster) |
| 4 | 1.035 [1.017, 1.052] (4/24 blocks faster) | 1.229 [1.205, 1.254] (0/24 blocks faster) |
| 8 | 0.697 [0.683, 0.711] (24/24 blocks faster) | 0.797 [0.780, 0.815] (24/24 blocks faster) |
| 16 | 1.042 [1.021, 1.063] (4/24 blocks faster) | 0.957 [0.932, 0.982] (20/24 blocks faster) |
| 32 | 1.406 [1.383, 1.430] (0/24 blocks faster) | 1.244 [1.220, 1.268] (0/24 blocks faster) |
| 64 | 2.447 [2.356, 2.541] (0/24 blocks faster) | 2.118 [2.047, 2.192] (0/24 blocks faster) |
| 128 | 3.634 [3.549, 3.721] (0/24 blocks faster) | 4.045 [3.915, 4.180] (0/24 blocks faster) |
| 256 | 5.993 [5.870, 6.119] (0/24 blocks faster) | 6.021 [5.867, 6.178] (0/24 blocks faster) |
| 512 | 10.152 [9.946, 10.362] (0/24 blocks faster) | 10.218 [10.005, 10.436] (0/24 blocks faster) |
| 1024 | 17.061 [16.826, 17.300] (0/24 blocks faster) | 26.337 [25.934, 26.746] (0/24 blocks faster) |

## Paired ratios against quasilinear, single mode

Geometric mean of per-block ratios (competitor / quasilinear) with 95% interval; values above 1 mean quasilinear was faster.

| N | direct / quasilinear | cariow_cariowa / quasilinear |
| ---: | --- | --- |
| 2 | 0.988 [0.964, 1.012] (14/24 blocks faster) | 1.024 [1.001, 1.047] (4/24 blocks faster) |
| 4 | 0.964 [0.944, 0.985] (16/24 blocks faster) | 1.071 [1.040, 1.103] (4/24 blocks faster) |
| 8 | 0.817 [0.780, 0.855] (23/24 blocks faster) | 1.170 [1.144, 1.196] (0/24 blocks faster) |
| 16 | 0.811 [0.787, 0.836] (24/24 blocks faster) | 1.069 [1.037, 1.103] (5/24 blocks faster) |
| 32 | 1.323 [1.306, 1.341] (0/24 blocks faster) | 2.029 [1.990, 2.070] (0/24 blocks faster) |
| 64 | 2.222 [2.158, 2.287] (0/24 blocks faster) | 3.638 [3.550, 3.727] (0/24 blocks faster) |
| 128 | 3.413 [3.333, 3.495] (0/24 blocks faster) | 6.280 [6.043, 6.525] (0/24 blocks faster) |
| 256 | 5.407 [5.317, 5.499] (0/24 blocks faster) | 8.942 [8.716, 9.173] (0/24 blocks faster) |
| 512 | 9.384 [9.265, 9.505] (0/24 blocks faster) | 15.302 [15.133, 15.472] (0/24 blocks faster) |
| 1024 | 17.189 [16.967, 17.415] (0/24 blocks faster) | 27.078 [26.255, 27.927] (0/24 blocks faster) |

## Single-product time relative to batched time

Geometric mean of per-block ratios (single / batch) for each algorithm; values above 1 mean batching was faster.

| N | direct | cariow_cariowa | quasilinear |
| ---: | --- | --- | --- |
| 2 | 11.994 [11.730, 12.264] | 10.366 [10.181, 10.554] | 15.666 [15.355, 15.982] |
| 4 | 4.496 [4.437, 4.555] | 4.202 [4.123, 4.283] | 4.825 [4.699, 4.953] |
| 8 | 2.213 [2.125, 2.305] | 2.773 [2.711, 2.836] | 1.889 [1.837, 1.943] |
| 16 | 1.503 [1.445, 1.563] | 2.157 [2.095, 2.220] | 1.930 [1.887, 1.974] |
| 32 | 1.136 [1.116, 1.155] | 1.969 [1.929, 2.009] | 1.207 [1.183, 1.231] |
| 64 | 1.002 [0.966, 1.040] | 1.896 [1.848, 1.945] | 1.104 [1.078, 1.131] |
| 128 | 0.983 [0.964, 1.002] | 1.625 [1.555, 1.698] | 1.047 [1.022, 1.072] |
| 256 | 0.964 [0.949, 0.980] | 1.587 [1.548, 1.628] | 1.069 [1.046, 1.092] |
| 512 | 0.989 [0.975, 1.003] | 1.602 [1.576, 1.629] | 1.070 [1.053, 1.087] |
| 1024 | 1.000 [0.992, 1.009] | 1.020 [0.997, 1.044] | 0.993 [0.975, 1.010] |

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
