# fastCD multiplication benchmark

Host: Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz (x86_64); gcc 16.2.1 20260810; 24 blocks of 10000 pairs per dimension; each cell is the mean over blocks of ns per product with a two-sided 95% Student-t interval.

Batch mode times one validated batch call per corpus pass (two products per SIMD instruction where the build supports it); single mode times one validated single-product call per pair.  Python object overhead is excluded. Verification uses an independent float64 oracle, not an exact product. Fastest names the lowest verified mean, or a tie when the paired-ratio interval of the two lowest means contains one; default is the library's automatic selection for that mode.

Seed: 20260906; CPU: 0; status: ok. Commit: c500d40f4e72e60b28fcafca60d7aa831393c84f (with uncommitted changes to tracked files). Build options, source and executable hashes, corpus hashes, and raw observations are in the accompanying JSON.

## Batch products (ns per product)

| N | direct | cariow_cariowa | quasilinear | fastest mean | default |
| ---: | --- | --- | --- | --- | --- |
| 2 | 1.33 [1.31, 1.35] | 1.58 [1.54, 1.62] | 1.03 [1.02, 1.04] | quasilinear | quasilinear |
| 4 | 3.89 [3.83, 3.95] | 4.68 [4.59, 4.76] | 3.80 [3.73, 3.87] | quasilinear | quasilinear |
| 8 | 12.58 [12.37, 12.78] | 14.45 [14.25, 14.64] | 18.03 [17.72, 18.35] | direct | direct |
| 16 | 56.72 [55.77, 57.67] | 51.01 [50.22, 51.79] | 56.06 [55.37, 56.74] | cariow_cariowa | cariow_cariowa |
| 32 | 262.45 [257.07, 267.82] | 237.29 [234.06, 240.52] | 178.47 [174.59, 182.34] | quasilinear | quasilinear |
| 64 | 1145.54 [1119.18, 1171.90] | 1013.19 [994.89, 1031.49] | 453.45 [445.57, 461.33] | quasilinear | quasilinear |
| 128 | 4680.78 [4577.72, 4783.84] | 4978.12 [4879.90, 5076.33] | 1238.11 [1219.50, 1256.71] | quasilinear | quasilinear |
| 256 | 18671.60 [18492.21, 18850.98] | 17894.54 [17496.60, 18292.47] | 3005.67 [2962.63, 3048.72] | quasilinear | quasilinear |
| 512 | 71373.60 [70472.12, 72275.07] | 65361.92 [64656.19, 66067.66] | 6758.91 [6622.16, 6895.66] | quasilinear | quasilinear |
| 1024 | 282743.07 [280338.13, 285148.01] | 422651.68 [411684.98, 433618.38] | 16113.68 [15991.54, 16235.82] | quasilinear | quasilinear |

## Single products (ns per product)

| N | direct | cariow_cariowa | quasilinear | fastest mean | default |
| ---: | --- | --- | --- | --- | --- |
| 2 | 16.17 [15.93, 16.40] | 17.39 [16.88, 17.90] | 15.57 [15.29, 15.84] | quasilinear | quasilinear |
| 4 | 18.26 [18.04, 18.49] | 20.32 [20.00, 20.64] | 18.52 [18.17, 18.88] | tie (direct, quasilinear) | quasilinear |
| 8 | 28.75 [28.22, 29.29] | 41.40 [40.66, 42.13] | 37.63 [37.05, 38.20] | direct | direct |
| 16 | 82.05 [80.88, 83.23] | 114.32 [112.88, 115.75] | 113.73 [111.20, 116.27] | direct | direct |
| 32 | 300.37 [295.88, 304.86] | 464.58 [455.96, 473.20] | 215.83 [213.02, 218.64] | quasilinear | quasilinear |
| 64 | 1144.89 [1128.71, 1161.07] | 1918.87 [1879.66, 1958.08] | 500.59 [492.62, 508.57] | quasilinear | quasilinear |
| 128 | 4392.20 [4322.03, 4462.36] | 8313.62 [7963.52, 8663.73] | 1244.21 [1224.54, 1263.89] | quasilinear | quasilinear |
| 256 | 17607.86 [17418.09, 17797.63] | 29052.88 [28576.28, 29529.49] | 3227.60 [3173.17, 3282.03] | quasilinear | quasilinear |
| 512 | 69288.27 [68796.10, 69780.45] | 107763.51 [106749.81, 108777.20] | 7225.82 [7108.12, 7343.52] | quasilinear | quasilinear |
| 1024 | 285131.22 [283178.93, 287083.52] | 420241.56 [417302.94, 423180.17] | 16319.92 [16153.98, 16485.86] | quasilinear | quasilinear |

## Paired ratios against quasilinear, batch mode

Geometric mean of per-block ratios (competitor / quasilinear) with 95% interval; values above 1 mean quasilinear was faster.

| N | direct / quasilinear | cariow_cariowa / quasilinear |
| ---: | --- | --- |
| 2 | 1.291 [1.270, 1.313] (0/24 blocks faster) | 1.531 [1.495, 1.568] (0/24 blocks faster) |
| 4 | 1.025 [1.000, 1.050] (7/24 blocks faster) | 1.230 [1.206, 1.255] (0/24 blocks faster) |
| 8 | 0.697 [0.678, 0.718] (24/24 blocks faster) | 0.801 [0.785, 0.818] (24/24 blocks faster) |
| 16 | 1.012 [0.995, 1.029] (9/24 blocks faster) | 0.910 [0.893, 0.927] (23/24 blocks faster) |
| 32 | 1.471 [1.432, 1.510] (0/24 blocks faster) | 1.330 [1.305, 1.357] (0/24 blocks faster) |
| 64 | 2.525 [2.467, 2.584] (0/24 blocks faster) | 2.234 [2.181, 2.289] (0/24 blocks faster) |
| 128 | 3.778 [3.680, 3.879] (0/24 blocks faster) | 4.019 [3.923, 4.118] (0/24 blocks faster) |
| 256 | 6.214 [6.104, 6.326] (0/24 blocks faster) | 5.949 [5.806, 6.096] (0/24 blocks faster) |
| 512 | 10.566 [10.347, 10.790] (0/24 blocks faster) | 9.678 [9.465, 9.895] (0/24 blocks faster) |
| 1024 | 17.546 [17.363, 17.731] (0/24 blocks faster) | 26.192 [25.594, 26.805] (0/24 blocks faster) |

## Paired ratios against quasilinear, single mode

Geometric mean of per-block ratios (competitor / quasilinear) with 95% interval; values above 1 mean quasilinear was faster.

| N | direct / quasilinear | cariow_cariowa / quasilinear |
| ---: | --- | --- |
| 2 | 1.039 [1.022, 1.056] (3/24 blocks faster) | 1.116 [1.095, 1.138] (0/24 blocks faster) |
| 4 | 0.987 [0.966, 1.008] (12/24 blocks faster) | 1.097 [1.072, 1.124] (1/24 blocks faster) |
| 8 | 0.764 [0.747, 0.781] (24/24 blocks faster) | 1.100 [1.074, 1.127] (0/24 blocks faster) |
| 16 | 0.722 [0.705, 0.740] (24/24 blocks faster) | 1.006 [0.983, 1.030] (9/24 blocks faster) |
| 32 | 1.392 [1.371, 1.413] (0/24 blocks faster) | 2.152 [2.104, 2.200] (0/24 blocks faster) |
| 64 | 2.287 [2.256, 2.319] (0/24 blocks faster) | 3.831 [3.734, 3.931] (0/24 blocks faster) |
| 128 | 3.530 [3.472, 3.589] (0/24 blocks faster) | 6.658 [6.384, 6.943] (0/24 blocks faster) |
| 256 | 5.458 [5.333, 5.585] (0/24 blocks faster) | 9.002 [8.811, 9.196] (0/24 blocks faster) |
| 512 | 9.594 [9.438, 9.753] (0/24 blocks faster) | 14.920 [14.657, 15.189] (0/24 blocks faster) |
| 1024 | 17.474 [17.276, 17.674] (0/24 blocks faster) | 25.754 [25.446, 26.066] (0/24 blocks faster) |

## Single-product time relative to batched time

Geometric mean of per-block ratios (single / batch) for each algorithm; values above 1 mean batching was faster.

| N | direct | cariow_cariowa | quasilinear |
| ---: | --- | --- | --- |
| 2 | 12.167 [11.976, 12.362] | 11.024 [10.694, 11.365] | 15.125 [14.871, 15.383] |
| 4 | 4.692 [4.622, 4.763] | 4.346 [4.236, 4.459] | 4.873 [4.731, 5.019] |
| 8 | 2.286 [2.225, 2.349] | 2.864 [2.807, 2.923] | 2.087 [2.044, 2.131] |
| 16 | 1.447 [1.418, 1.476] | 2.242 [2.201, 2.284] | 2.027 [1.979, 2.076] |
| 32 | 1.145 [1.117, 1.174] | 1.957 [1.914, 2.001] | 1.210 [1.181, 1.240] |
| 64 | 1.000 [0.974, 1.028] | 1.893 [1.845, 1.943] | 1.104 [1.081, 1.127] |
| 128 | 0.939 [0.921, 0.958] | 1.665 [1.598, 1.734] | 1.005 [0.985, 1.026] |
| 256 | 0.943 [0.930, 0.957] | 1.624 [1.580, 1.671] | 1.074 [1.045, 1.103] |
| 512 | 0.971 [0.956, 0.987] | 1.649 [1.629, 1.669] | 1.069 [1.042, 1.097] |
| 1024 | 1.009 [1.000, 1.018] | 0.996 [0.974, 1.018] | 1.013 [1.003, 1.022] |

## Verification against the defining recursion

Maximum norm-scaled coordinate error |actual - oracle| / (||x|| ||y||) over the verified pairs, and mismatches against the tolerance 128 eps N, for each mode.

| N | direct | cariow_cariowa | quasilinear |
| ---: | --- | --- | --- |
| 2 | batch: 0.00e+00 (0/10000); single: 0.00e+00 (0/10000) | batch: 4.09e-16 (0/10000); single: 4.09e-16 (0/10000) | batch: 0.00e+00 (0/10000); single: 0.00e+00 (0/10000) |
| 4 | batch: 2.51e-16 (0/10000); single: 2.51e-16 (0/10000) | batch: 4.24e-16 (0/10000); single: 4.24e-16 (0/10000) | batch: 3.08e-16 (0/10000); single: 3.08e-16 (0/10000) |
| 8 | batch: 3.47e-16 (0/10000); single: 3.47e-16 (0/10000) | batch: 4.13e-16 (0/10000); single: 4.13e-16 (0/10000) | batch: 4.48e-16 (0/10000); single: 4.48e-16 (0/10000) |
| 16 | batch: 4.08e-16 (0/10000); single: 4.08e-16 (0/10000) | batch: 3.88e-16 (0/10000); single: 3.88e-16 (0/10000) | batch: 5.48e-16 (0/10000); single: 5.48e-16 (0/10000) |
| 32 | batch: 2.21e-16 (0/10000); single: 2.21e-16 (0/10000) | batch: 3.63e-16 (0/10000); single: 3.63e-16 (0/10000) | batch: 3.63e-16 (0/10000); single: 3.63e-16 (0/10000) |
| 64 | batch: 1.81e-16 (0/10000); single: 1.81e-16 (0/10000) | batch: 4.35e-16 (0/10000); single: 4.35e-16 (0/10000) | batch: 2.95e-16 (0/10000); single: 2.95e-16 (0/10000) |
| 128 | batch: 1.45e-16 (0/10000); single: 1.45e-16 (0/10000) | batch: 2.12e-16 (0/10000); single: 2.12e-16 (0/10000) | batch: 2.26e-16 (0/10000); single: 2.26e-16 (0/10000) |
| 256 | batch: 9.90e-17 (0/10000); single: 9.90e-17 (0/10000) | batch: 1.80e-16 (0/10000); single: 1.80e-16 (0/10000) | batch: 3.29e-16 (0/10000); single: 3.29e-16 (0/10000) |
| 512 | batch: 8.20e-17 (0/10000); single: 8.20e-17 (0/10000) | batch: 1.65e-16 (0/10000); single: 1.65e-16 (0/10000) | batch: 2.28e-16 (0/10000); single: 2.28e-16 (0/10000) |
| 1024 | batch: 6.15e-17 (0/10000); single: 6.15e-17 (0/10000) | batch: 1.65e-16 (0/10000); single: 1.65e-16 (0/10000) | batch: 2.90e-16 (0/10000); single: 2.90e-16 (0/10000) |
