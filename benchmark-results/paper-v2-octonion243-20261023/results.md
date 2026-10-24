# fastCD multiplication benchmark

Host: Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz (x86_64); gcc 16.2.1 20260810; 24 blocks of 10000 pairs per dimension; each cell is the mean over blocks of ns per product with a two-sided 95% Student-t interval.

Batch mode times one validated batch call per corpus pass (two products per SIMD instruction where the build supports it); single mode times one validated single-product call per pair.  Python object overhead is excluded. Verification uses an independent float64 oracle, not an exact product. Fastest names the lowest verified mean, or a tie when the paired-ratio interval of the two lowest means contains one; default is the library's automatic selection for that mode.

Seed: 20260906; CPU: 0; status: ok. Commit: 7da5a5f46c08ef41dfe98aaafa13be4643c3b647 (with uncommitted changes to tracked files). Build options, source and executable hashes, corpus hashes, and raw observations are in the accompanying JSON.

## Batch products (ns per product)

| N | direct | cariow_cariowa | quasilinear | fastest mean | default |
| ---: | --- | --- | --- | --- | --- |
| 2 | 1.31 [1.29, 1.32] | 1.57 [1.54, 1.60] | 1.01 [1.00, 1.02] | quasilinear | quasilinear |
| 4 | 3.84 [3.78, 3.89] | 4.58 [4.50, 4.65] | 3.77 [3.72, 3.81] | tie (quasilinear, direct) | quasilinear |
| 8 | 12.41 [12.20, 12.61] | 14.19 [13.82, 14.56] | 17.86 [17.57, 18.15] | direct | direct |
| 16 | 55.34 [54.18, 56.49] | 50.46 [49.49, 51.43] | 55.15 [54.24, 56.06] | cariow_cariowa | cariow_cariowa |
| 32 | 257.28 [252.91, 261.66] | 230.56 [227.19, 233.93] | 174.49 [172.83, 176.15] | quasilinear | quasilinear |
| 64 | 1152.26 [1133.60, 1170.93] | 1012.92 [993.30, 1032.54] | 446.49 [440.03, 452.95] | quasilinear | quasilinear |
| 128 | 4748.29 [4688.47, 4808.10] | 5136.17 [4962.06, 5310.28] | 1249.72 [1219.47, 1279.97] | quasilinear | quasilinear |
| 256 | 18377.31 [18145.54, 18609.07] | 18353.65 [17953.07, 18754.24] | 3059.44 [3007.34, 3111.55] | quasilinear | quasilinear |
| 512 | 70080.95 [69127.78, 71034.12] | 66708.04 [65482.11, 67933.97] | 6791.69 [6697.58, 6885.80] | quasilinear | quasilinear |
| 1024 | 272762.00 [270235.59, 275288.41] | 406124.61 [403922.66, 408326.56] | 15426.99 [15279.98, 15573.99] | quasilinear | quasilinear |

## Single products (ns per product)

| N | direct | cariow_cariowa | quasilinear | fastest mean | default |
| ---: | --- | --- | --- | --- | --- |
| 2 | 15.93 [15.71, 16.16] | 16.73 [16.52, 16.95] | 15.14 [14.85, 15.42] | quasilinear | quasilinear |
| 4 | 18.18 [17.88, 18.47] | 20.02 [19.67, 20.37] | 18.12 [17.85, 18.39] | tie (quasilinear, direct) | quasilinear |
| 8 | 27.72 [27.19, 28.26] | 40.65 [40.10, 41.20] | 36.63 [36.01, 37.24] | direct | direct |
| 16 | 81.65 [79.95, 83.34] | 115.26 [113.79, 116.74] | 113.28 [111.07, 115.48] | direct | direct |
| 32 | 299.20 [294.51, 303.89] | 454.81 [449.85, 459.78] | 210.52 [207.60, 213.43] | quasilinear | quasilinear |
| 64 | 1162.63 [1141.36, 1183.91] | 1873.65 [1833.21, 1914.10] | 508.11 [501.06, 515.17] | quasilinear | quasilinear |
| 128 | 4592.40 [4521.08, 4663.72] | 7850.82 [7698.55, 8003.09] | 1240.04 [1224.28, 1255.79] | quasilinear | quasilinear |
| 256 | 17641.62 [17449.49, 17833.76] | 29390.25 [28810.93, 29969.57] | 3259.17 [3214.80, 3303.53] | quasilinear | quasilinear |
| 512 | 69431.66 [68744.19, 70119.14] | 108745.08 [107367.31, 110122.86] | 7281.07 [7135.00, 7427.14] | quasilinear | quasilinear |
| 1024 | 270678.46 [269014.65, 272342.27] | 406446.84 [404362.16, 408531.51] | 15713.75 [15492.33, 15935.16] | quasilinear | quasilinear |

## Paired ratios against quasilinear, batch mode

Geometric mean of per-block ratios (competitor / quasilinear) with 95% interval; values above 1 mean quasilinear was faster.

| N | direct / quasilinear | cariow_cariowa / quasilinear |
| ---: | --- | --- |
| 2 | 1.295 [1.274, 1.317] (0/24 blocks faster) | 1.547 [1.520, 1.576] (0/24 blocks faster) |
| 4 | 1.018 [1.000, 1.037] (7/24 blocks faster) | 1.215 [1.192, 1.238] (0/24 blocks faster) |
| 8 | 0.695 [0.679, 0.710] (24/24 blocks faster) | 0.794 [0.776, 0.812] (24/24 blocks faster) |
| 16 | 1.003 [0.981, 1.026] (11/24 blocks faster) | 0.915 [0.894, 0.936] (23/24 blocks faster) |
| 32 | 1.474 [1.451, 1.497] (0/24 blocks faster) | 1.321 [1.302, 1.340] (0/24 blocks faster) |
| 64 | 2.580 [2.543, 2.618] (0/24 blocks faster) | 2.268 [2.219, 2.318] (0/24 blocks faster) |
| 128 | 3.804 [3.705, 3.905] (0/24 blocks faster) | 4.104 [3.968, 4.245] (0/24 blocks faster) |
| 256 | 6.009 [5.874, 6.146] (0/24 blocks faster) | 5.996 [5.835, 6.162] (0/24 blocks faster) |
| 512 | 10.319 [10.147, 10.494] (0/24 blocks faster) | 9.818 [9.633, 10.007] (0/24 blocks faster) |
| 1024 | 17.681 [17.468, 17.897] (0/24 blocks faster) | 26.330 [26.082, 26.580] (0/24 blocks faster) |

## Paired ratios against quasilinear, single mode

Geometric mean of per-block ratios (competitor / quasilinear) with 95% interval; values above 1 mean quasilinear was faster.

| N | direct / quasilinear | cariow_cariowa / quasilinear |
| ---: | --- | --- |
| 2 | 1.053 [1.033, 1.074] (4/24 blocks faster) | 1.106 [1.086, 1.127] (0/24 blocks faster) |
| 4 | 1.003 [0.985, 1.021] (16/24 blocks faster) | 1.105 [1.078, 1.132] (0/24 blocks faster) |
| 8 | 0.757 [0.741, 0.773] (24/24 blocks faster) | 1.110 [1.091, 1.130] (0/24 blocks faster) |
| 16 | 0.721 [0.704, 0.738] (24/24 blocks faster) | 1.018 [0.998, 1.039] (8/24 blocks faster) |
| 32 | 1.421 [1.393, 1.450] (0/24 blocks faster) | 2.161 [2.124, 2.198] (0/24 blocks faster) |
| 64 | 2.287 [2.239, 2.336] (0/24 blocks faster) | 3.685 [3.593, 3.779] (0/24 blocks faster) |
| 128 | 3.703 [3.638, 3.769] (0/24 blocks faster) | 6.328 [6.220, 6.437] (0/24 blocks faster) |
| 256 | 5.414 [5.324, 5.506] (0/24 blocks faster) | 9.013 [8.846, 9.183] (0/24 blocks faster) |
| 512 | 9.543 [9.369, 9.721] (0/24 blocks faster) | 14.945 [14.649, 15.247] (0/24 blocks faster) |
| 1024 | 17.233 [16.966, 17.504] (0/24 blocks faster) | 25.878 [25.534, 26.226] (0/24 blocks faster) |

## Single-product time relative to batched time

Geometric mean of per-block ratios (single / batch) for each algorithm; values above 1 mean batching was faster.

| N | direct | cariow_cariowa | quasilinear |
| ---: | --- | --- | --- |
| 2 | 12.163 [11.955, 12.374] | 10.692 [10.434, 10.956] | 14.957 [14.704, 15.216] |
| 4 | 4.738 [4.649, 4.828] | 4.373 [4.275, 4.473] | 4.810 [4.714, 4.908] |
| 8 | 2.234 [2.186, 2.284] | 2.869 [2.802, 2.937] | 2.051 [2.013, 2.090] |
| 16 | 1.475 [1.436, 1.516] | 2.285 [2.241, 2.331] | 2.053 [2.001, 2.107] |
| 32 | 1.163 [1.138, 1.189] | 1.973 [1.937, 2.010] | 1.206 [1.188, 1.224] |
| 64 | 1.009 [0.985, 1.033] | 1.849 [1.799, 1.901] | 1.138 [1.118, 1.159] |
| 128 | 0.967 [0.951, 0.983] | 1.532 [1.478, 1.587] | 0.993 [0.970, 1.018] |
| 256 | 0.960 [0.945, 0.975] | 1.602 [1.562, 1.643] | 1.066 [1.044, 1.088] |
| 512 | 0.991 [0.979, 1.003] | 1.631 [1.603, 1.660] | 1.071 [1.046, 1.098] |
| 1024 | 0.992 [0.982, 1.003] | 1.001 [0.993, 1.008] | 1.018 [1.000, 1.037] |

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
