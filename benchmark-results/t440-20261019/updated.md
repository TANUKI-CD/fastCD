# fastCD multiplication benchmark

Host: Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz (x86_64); gcc 16.2.1 20260810; 48 blocks of 10000 pairs per dimension; each cell is the mean over blocks of ns per product with a two-sided 95% Student-t interval.

Batch mode times one validated batch call per corpus pass (two products per SIMD instruction where the build supports it); single mode times one validated single-product call per pair.  Python object overhead is excluded. Verification uses an independent float64 oracle, not an exact product. Fastest names the lowest verified mean, or a tie when the paired-ratio interval of the two lowest means contains one; default is the library's automatic selection for that mode.

Seed: 20260906; CPU: 11; status: ok. Commit: unknown (not a git checkout). Build options, source and executable hashes, corpus hashes, and raw observations are in the accompanying JSON.

## Batch products (ns per product)

| N | direct | cariow_cariowa | quasilinear | fastest mean | default |
| ---: | --- | --- | --- | --- | --- |
| 2 | 1.33 [1.32, 1.34] | 1.57 [1.55, 1.58] | 1.18 [1.16, 1.19] | quasilinear | quasilinear |
| 4 | 3.89 [3.85, 3.93] | 4.62 [4.58, 4.65] | 3.80 [3.76, 3.83] | quasilinear | quasilinear |
| 8 | 12.51 [12.39, 12.64] | 14.35 [14.21, 14.48] | 17.89 [17.75, 18.03] | direct | direct |
| 16 | 56.98 [56.59, 57.37] | 51.17 [50.77, 51.57] | 56.64 [56.19, 57.09] | cariow_cariowa | cariow_cariowa |
| 32 | 262.65 [260.53, 264.77] | 231.78 [230.15, 233.42] | 185.79 [184.26, 187.31] | quasilinear | quasilinear |
| 64 | 1140.31 [1130.45, 1150.17] | 1014.33 [1005.99, 1022.67] | 457.91 [454.48, 461.33] | quasilinear | quasilinear |
| 128 | 4683.59 [4652.06, 4715.12] | 4995.86 [4940.95, 5050.77] | 1234.15 [1220.94, 1247.36] | quasilinear | quasilinear |
| 256 | 18449.79 [18318.06, 18581.51] | 17819.49 [17562.45, 18076.52] | 3035.32 [3010.91, 3059.73] | quasilinear | quasilinear |

## Single products (ns per product)

| N | direct | cariow_cariowa | quasilinear | fastest mean | default |
| ---: | --- | --- | --- | --- | --- |
| 2 | 16.52 [16.33, 16.71] | 17.21 [17.09, 17.33] | 15.39 [15.26, 15.52] | quasilinear | quasilinear |
| 4 | 18.79 [18.63, 18.95] | 20.37 [20.16, 20.58] | 18.22 [18.07, 18.37] | quasilinear | direct |
| 8 | 28.28 [27.93, 28.63] | 41.39 [40.98, 41.81] | 37.31 [36.89, 37.73] | direct | direct |
| 16 | 82.64 [82.05, 83.23] | 115.85 [114.72, 116.98] | 113.65 [112.73, 114.57] | direct | direct |
| 32 | 301.14 [298.93, 303.35] | 457.80 [454.56, 461.05] | 217.00 [215.27, 218.72] | quasilinear | quasilinear |
| 64 | 1134.50 [1124.25, 1144.75] | 1911.13 [1897.73, 1924.53] | 509.14 [505.66, 512.61] | quasilinear | quasilinear |
| 128 | 4542.97 [4510.16, 4575.79] | 7850.71 [7751.35, 7950.08] | 1248.37 [1238.79, 1257.95] | quasilinear | quasilinear |
| 256 | 17654.61 [17536.41, 17772.82] | 28682.34 [28464.50, 28900.18] | 3231.32 [3205.80, 3256.84] | quasilinear | quasilinear |

## Paired ratios against quasilinear, batch mode

Geometric mean of per-block ratios (competitor / quasilinear) with 95% interval; values above 1 mean quasilinear was faster.

| N | direct / quasilinear | cariow_cariowa / quasilinear |
| ---: | --- | --- |
| 2 | 1.128 [1.117, 1.140] (0/48 blocks faster) | 1.331 [1.310, 1.352] (0/48 blocks faster) |
| 4 | 1.026 [1.012, 1.040] (17/48 blocks faster) | 1.217 [1.202, 1.233] (0/48 blocks faster) |
| 8 | 0.699 [0.692, 0.707] (48/48 blocks faster) | 0.802 [0.792, 0.812] (48/48 blocks faster) |
| 16 | 1.006 [0.995, 1.017] (19/48 blocks faster) | 0.903 [0.893, 0.914] (48/48 blocks faster) |
| 32 | 1.414 [1.396, 1.431] (0/48 blocks faster) | 1.248 [1.232, 1.263] (0/48 blocks faster) |
| 64 | 2.490 [2.457, 2.523] (0/48 blocks faster) | 2.215 [2.188, 2.242] (0/48 blocks faster) |
| 128 | 3.796 [3.747, 3.847] (0/48 blocks faster) | 4.048 [3.981, 4.116] (0/48 blocks faster) |
| 256 | 6.079 [6.011, 6.148] (0/48 blocks faster) | 5.866 [5.778, 5.956] (0/48 blocks faster) |

## Paired ratios against quasilinear, single mode

Geometric mean of per-block ratios (competitor / quasilinear) with 95% interval; values above 1 mean quasilinear was faster.

| N | direct / quasilinear | cariow_cariowa / quasilinear |
| ---: | --- | --- |
| 2 | 1.073 [1.058, 1.088] (2/48 blocks faster) | 1.118 [1.108, 1.129] (0/48 blocks faster) |
| 4 | 1.031 [1.017, 1.046] (13/48 blocks faster) | 1.118 [1.102, 1.135] (0/48 blocks faster) |
| 8 | 0.758 [0.747, 0.769] (48/48 blocks faster) | 1.110 [1.092, 1.127] (1/48 blocks faster) |
| 16 | 0.727 [0.718, 0.736] (48/48 blocks faster) | 1.019 [1.006, 1.033] (16/48 blocks faster) |
| 32 | 1.388 [1.372, 1.404] (0/48 blocks faster) | 2.110 [2.089, 2.131] (0/48 blocks faster) |
| 64 | 2.228 [2.206, 2.250] (0/48 blocks faster) | 3.754 [3.718, 3.790] (0/48 blocks faster) |
| 128 | 3.639 [3.606, 3.673] (0/48 blocks faster) | 6.285 [6.210, 6.361] (0/48 blocks faster) |
| 256 | 5.464 [5.417, 5.512] (0/48 blocks faster) | 8.877 [8.771, 8.983] (0/48 blocks faster) |

## Single-product time relative to batched time

Geometric mean of per-block ratios (single / batch) for each algorithm; values above 1 mean batching was faster.

| N | direct | cariow_cariowa | quasilinear |
| ---: | --- | --- | --- |
| 2 | 12.442 [12.276, 12.611] | 10.998 [10.851, 11.147] | 13.086 [12.894, 13.280] |
| 4 | 4.829 [4.765, 4.894] | 4.410 [4.353, 4.468] | 4.801 [4.742, 4.861] |
| 8 | 2.260 [2.221, 2.300] | 2.885 [2.839, 2.932] | 2.085 [2.055, 2.114] |
| 16 | 1.450 [1.435, 1.466] | 2.263 [2.236, 2.291] | 2.007 [1.986, 2.027] |
| 32 | 1.147 [1.136, 1.157] | 1.975 [1.955, 1.995] | 1.168 [1.156, 1.180] |
| 64 | 0.995 [0.984, 1.006] | 1.884 [1.863, 1.906] | 1.112 [1.102, 1.122] |
| 128 | 0.970 [0.961, 0.979] | 1.571 [1.550, 1.593] | 1.012 [0.999, 1.025] |
| 256 | 0.957 [0.948, 0.966] | 1.611 [1.586, 1.636] | 1.065 [1.054, 1.076] |

## Verification against the defining recursion

Maximum norm-scaled coordinate error |actual - oracle| / (||x|| ||y||) over the verified pairs, and mismatches against the tolerance 128 eps N, for each mode.

| N | direct | cariow_cariowa | quasilinear |
| ---: | --- | --- | --- |
| 2 | batch: 0.00e+00 (0/10000); single: 0.00e+00 (0/10000) | batch: 4.09e-16 (0/10000); single: 4.09e-16 (0/10000) | batch: 0.00e+00 (0/10000); single: 0.00e+00 (0/10000) |
| 4 | batch: 2.51e-16 (0/10000); single: 2.51e-16 (0/10000) | batch: 4.24e-16 (0/10000); single: 4.24e-16 (0/10000) | batch: 3.08e-16 (0/10000); single: 3.08e-16 (0/10000) |
| 8 | batch: 3.47e-16 (0/10000); single: 3.47e-16 (0/10000) | batch: 4.13e-16 (0/10000); single: 4.13e-16 (0/10000) | batch: 4.48e-16 (0/10000); single: 4.48e-16 (0/10000) |
| 16 | batch: 4.08e-16 (0/10000); single: 4.08e-16 (0/10000) | batch: 3.88e-16 (0/10000); single: 3.88e-16 (0/10000) | batch: 5.48e-16 (0/10000); single: 5.48e-16 (0/10000) |
| 32 | batch: 2.21e-16 (0/10000); single: 2.21e-16 (0/10000) | batch: 3.63e-16 (0/10000); single: 3.63e-16 (0/10000) | batch: 4.01e-16 (0/10000); single: 4.01e-16 (0/10000) |
| 64 | batch: 1.81e-16 (0/10000); single: 1.81e-16 (0/10000) | batch: 4.35e-16 (0/10000); single: 4.35e-16 (0/10000) | batch: 2.95e-16 (0/10000); single: 2.95e-16 (0/10000) |
| 128 | batch: 1.45e-16 (0/10000); single: 1.45e-16 (0/10000) | batch: 2.12e-16 (0/10000); single: 2.12e-16 (0/10000) | batch: 2.26e-16 (0/10000); single: 2.26e-16 (0/10000) |
| 256 | batch: 9.90e-17 (0/10000); single: 9.90e-17 (0/10000) | batch: 1.80e-16 (0/10000); single: 1.80e-16 (0/10000) | batch: 3.29e-16 (0/10000); single: 3.29e-16 (0/10000) |
