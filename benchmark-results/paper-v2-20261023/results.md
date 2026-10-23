# fastCD multiplication benchmark

Host: Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz (x86_64); gcc 16.2.1 20260810; 24 blocks of 10000 pairs per dimension; each cell is the mean over blocks of ns per product with a two-sided 95% Student-t interval.

Batch mode times one validated batch call per corpus pass (two products per SIMD instruction where the build supports it); single mode times one validated single-product call per pair.  Python object overhead is excluded. Verification uses an independent float64 oracle, not an exact product. Fastest names the lowest verified mean, or a tie when the paired-ratio interval of the two lowest means contains one; default is the library's automatic selection for that mode.

Seed: 20260906; CPU: 0; status: ok. Commit: f9a61b28550cd01253d59115d1cefb47d4e3b991 (clean tree). Build options, source and executable hashes, corpus hashes, and raw observations are in the accompanying JSON.

## Batch products (ns per product)

| N | direct | cariow_cariowa | quasilinear | fastest mean | default |
| ---: | --- | --- | --- | --- | --- |
| 2 | 1.32 [1.31, 1.34] | 1.53 [1.49, 1.57] | 1.18 [1.16, 1.20] | quasilinear | quasilinear |
| 4 | 3.72 [3.57, 3.87] | 4.41 [4.35, 4.48] | 3.64 [3.49, 3.79] | tie (quasilinear, direct) | quasilinear |
| 8 | 12.44 [12.26, 12.62] | 14.35 [14.12, 14.57] | 17.79 [17.56, 18.01] | direct | direct |
| 16 | 55.70 [54.75, 56.66] | 50.26 [49.49, 51.04] | 54.72 [53.89, 55.55] | cariow_cariowa | cariow_cariowa |
| 32 | 260.03 [255.89, 264.16] | 228.57 [225.45, 231.69] | 182.12 [179.61, 184.64] | quasilinear | quasilinear |
| 64 | 1157.56 [1135.01, 1180.10] | 1030.75 [1017.72, 1043.79] | 460.16 [454.79, 465.53] | quasilinear | quasilinear |
| 128 | 4617.52 [4536.22, 4698.83] | 4915.14 [4791.39, 5038.89] | 1234.76 [1209.71, 1259.82] | quasilinear | quasilinear |
| 256 | 18115.11 [17890.99, 18339.23] | 17592.57 [17353.49, 17831.64] | 2993.77 [2949.73, 3037.82] | quasilinear | quasilinear |
| 512 | 69768.05 [68774.94, 70761.16] | 64511.31 [63946.62, 65076.00] | 6791.81 [6688.70, 6894.93] | quasilinear | quasilinear |
| 1024 | 272717.79 [270536.99, 274898.58] | 413297.56 [402950.61, 423644.51] | 15688.11 [15549.41, 15826.82] | quasilinear | quasilinear |

## Single products (ns per product)

| N | direct | cariow_cariowa | quasilinear | fastest mean | default |
| ---: | --- | --- | --- | --- | --- |
| 2 | 16.42 [16.21, 16.63] | 17.00 [16.78, 17.23] | 15.37 [15.20, 15.53] | quasilinear | quasilinear |
| 4 | 19.12 [17.56, 20.68] | 21.80 [19.50, 24.10] | 18.55 [17.77, 19.32] | tie (quasilinear, direct) | direct |
| 8 | 28.07 [27.66, 28.47] | 41.11 [40.56, 41.65] | 37.20 [36.69, 37.71] | direct | direct |
| 16 | 82.44 [80.56, 84.32] | 114.64 [113.13, 116.15] | 111.54 [109.99, 113.08] | direct | direct |
| 32 | 300.32 [296.59, 304.05] | 456.03 [450.17, 461.88] | 217.40 [212.48, 222.32] | quasilinear | quasilinear |
| 64 | 1148.56 [1133.29, 1163.83] | 1944.67 [1921.60, 1967.75] | 511.60 [504.11, 519.09] | quasilinear | quasilinear |
| 128 | 4516.08 [4453.89, 4578.27] | 7732.47 [7608.13, 7856.80] | 1227.89 [1206.02, 1249.75] | quasilinear | quasilinear |
| 256 | 17367.15 [17238.01, 17496.29] | 28262.75 [27874.95, 28650.56] | 3176.66 [3131.20, 3222.12] | quasilinear | quasilinear |
| 512 | 68849.27 [68178.55, 69519.99] | 106804.97 [105999.77, 107610.16] | 7307.34 [7191.12, 7423.57] | quasilinear | quasilinear |
| 1024 | 274053.28 [271828.85, 276277.71] | 408040.98 [406100.25, 409981.72] | 15924.07 [15659.90, 16188.25] | quasilinear | quasilinear |

## Paired ratios against quasilinear, batch mode

Geometric mean of per-block ratios (competitor / quasilinear) with 95% interval; values above 1 mean quasilinear was faster.

| N | direct / quasilinear | cariow_cariowa / quasilinear |
| ---: | --- | --- |
| 2 | 1.121 [1.098, 1.144] (0/24 blocks faster) | 1.293 [1.268, 1.319] (0/24 blocks faster) |
| 4 | 1.022 [0.976, 1.070] (7/24 blocks faster) | 1.216 [1.170, 1.263] (1/24 blocks faster) |
| 8 | 0.699 [0.687, 0.712] (24/24 blocks faster) | 0.806 [0.796, 0.817] (24/24 blocks faster) |
| 16 | 1.018 [0.996, 1.040] (11/24 blocks faster) | 0.919 [0.898, 0.940] (23/24 blocks faster) |
| 32 | 1.428 [1.407, 1.448] (0/24 blocks faster) | 1.255 [1.232, 1.279] (0/24 blocks faster) |
| 64 | 2.514 [2.472, 2.557] (0/24 blocks faster) | 2.240 [2.205, 2.275] (0/24 blocks faster) |
| 128 | 3.741 [3.662, 3.821] (0/24 blocks faster) | 3.979 [3.870, 4.090] (0/24 blocks faster) |
| 256 | 6.052 [5.940, 6.166] (0/24 blocks faster) | 5.877 [5.785, 5.970] (0/24 blocks faster) |
| 512 | 10.273 [10.049, 10.502] (0/24 blocks faster) | 9.502 [9.355, 9.652] (0/24 blocks faster) |
| 1024 | 17.384 [17.182, 17.589] (0/24 blocks faster) | 26.312 [25.646, 26.996] (0/24 blocks faster) |

## Paired ratios against quasilinear, single mode

Geometric mean of per-block ratios (competitor / quasilinear) with 95% interval; values above 1 mean quasilinear was faster.

| N | direct / quasilinear | cariow_cariowa / quasilinear |
| ---: | --- | --- |
| 2 | 1.069 [1.055, 1.082] (0/24 blocks faster) | 1.106 [1.088, 1.125] (0/24 blocks faster) |
| 4 | 1.023 [0.996, 1.050] (7/24 blocks faster) | 1.155 [1.088, 1.227] (0/24 blocks faster) |
| 8 | 0.754 [0.741, 0.768] (24/24 blocks faster) | 1.105 [1.083, 1.128] (0/24 blocks faster) |
| 16 | 0.739 [0.720, 0.758] (24/24 blocks faster) | 1.028 [1.009, 1.047] (6/24 blocks faster) |
| 32 | 1.383 [1.352, 1.414] (0/24 blocks faster) | 2.099 [2.043, 2.158] (0/24 blocks faster) |
| 64 | 2.245 [2.210, 2.282] (0/24 blocks faster) | 3.802 [3.736, 3.869] (0/24 blocks faster) |
| 128 | 3.679 [3.589, 3.771] (0/24 blocks faster) | 6.298 [6.130, 6.471] (0/24 blocks faster) |
| 256 | 5.469 [5.384, 5.556] (0/24 blocks faster) | 8.898 [8.732, 9.066] (0/24 blocks faster) |
| 512 | 9.426 [9.247, 9.608] (0/24 blocks faster) | 14.624 [14.390, 14.861] (0/24 blocks faster) |
| 1024 | 17.219 [16.923, 17.521] (0/24 blocks faster) | 25.641 [25.190, 26.101] (0/24 blocks faster) |

## Single-product time relative to batched time

Geometric mean of per-block ratios (single / batch) for each algorithm; values above 1 mean batching was faster.

| N | direct | cariow_cariowa | quasilinear |
| ---: | --- | --- | --- |
| 2 | 12.400 [12.200, 12.603] | 11.132 [10.831, 11.441] | 13.006 [12.800, 13.216] |
| 4 | 5.097 [4.754, 5.465] | 4.841 [4.477, 5.233] | 5.093 [4.991, 5.198] |
| 8 | 2.256 [2.205, 2.309] | 2.866 [2.799, 2.934] | 2.092 [2.052, 2.131] |
| 16 | 1.479 [1.437, 1.523] | 2.281 [2.237, 2.326] | 2.038 [2.005, 2.072] |
| 32 | 1.155 [1.130, 1.181] | 1.995 [1.960, 2.031] | 1.193 [1.160, 1.227] |
| 64 | 0.993 [0.972, 1.014] | 1.887 [1.850, 1.924] | 1.112 [1.095, 1.128] |
| 128 | 0.978 [0.958, 0.999] | 1.575 [1.532, 1.619] | 0.995 [0.970, 1.020] |
| 256 | 0.959 [0.946, 0.972] | 1.607 [1.575, 1.639] | 1.061 [1.035, 1.087] |
| 512 | 0.987 [0.973, 1.001] | 1.656 [1.639, 1.672] | 1.076 [1.051, 1.102] |
| 1024 | 1.005 [0.994, 1.016] | 0.989 [0.966, 1.012] | 1.015 [0.995, 1.035] |

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
| 512 | batch: 8.20e-17 (0/10000); single: 8.20e-17 (0/10000) | batch: 1.65e-16 (0/10000); single: 1.65e-16 (0/10000) | batch: 2.28e-16 (0/10000); single: 2.28e-16 (0/10000) |
| 1024 | batch: 6.15e-17 (0/10000); single: 6.15e-17 (0/10000) | batch: 1.65e-16 (0/10000); single: 1.65e-16 (0/10000) | batch: 2.90e-16 (0/10000); single: 2.90e-16 (0/10000) |
