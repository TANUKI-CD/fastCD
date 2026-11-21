Paired timings on one host: 12 blocks pinned to CPU 18. Times are medians in ns per product, and brackets are paired bootstrap 95% intervals. The direct control is the ratio of direct timings between the two builds, whose direct arithmetic is identical, so it shows the noise between builds.

| domain | kernels | mode | N | sequential ns | balanced ns | balanced / sequential [95% CI] | direct ns | direct / balanced | direct control |
|:---|:---|:---|---:|---:|---:|:---|---:|---:|---:|
| real | simd | single | 16 | 117.1 | 106.1 | 0.9105 [0.9003, 0.9322] | 80.33 | 0.7596 | 0.8855 |
| real | simd | batch | 16 | 53.55 | 51.52 | 0.9605 [0.9455, 0.9663] | 55.83 | 1.087 | 0.9866 |
| real | scalar | single | 16 | 119 | 111.9 | 0.9415 [0.936, 0.9561] | 81.52 | 0.7348 | 0.9187 |
| real | scalar | batch | 16 | 87.35 | 87.63 | 0.9997 [0.9621, 1.014] | 56.05 | 0.6492 | 1.004 |
| real | simd | single | 64 | 507.3 | 530.6 | 1.043 [1.022, 1.047] | 1150 | 2.165 | 0.9937 |
| real | simd | batch | 64 | 438.6 | 456.5 | 1.042 [1.038, 1.051] | 1084 | 2.366 | 0.996 |
| real | scalar | single | 64 | 654.8 | 724.9 | 1.098 [1.09, 1.111] | 1134 | 1.581 | 1.005 |
| real | scalar | batch | 64 | 625.4 | 685.5 | 1.094 [1.082, 1.104] | 1095 | 1.589 | 1.004 |
| real | simd | single | 256 | 2658 | 2737 | 1.021 [1.007, 1.039] | 1.772e+04 | 6.5 | 0.9939 |
| real | simd | batch | 256 | 2601 | 2649 | 1.013 [0.9987, 1.039] | 1.699e+04 | 6.431 | 0.9945 |
| real | scalar | single | 256 | 3392 | 3731 | 1.107 [1.077, 1.143] | 1.781e+04 | 4.765 | 0.9832 |
| real | scalar | batch | 256 | 3382 | 3680 | 1.098 [1.079, 1.125] | 1.788e+04 | 4.833 | 0.9911 |
| real | simd | single | 1024 | 1.434e+04 | 1.479e+04 | 1.023 [1.004, 1.051] | 2.831e+05 | 19.15 | 0.9851 |
| real | simd | batch | 1024 | 1.433e+04 | 1.49e+04 | 1.028 [1.015, 1.069] | 2.853e+05 | 19.17 | 0.9944 |
| real | scalar | single | 1024 | 1.762e+04 | 1.951e+04 | 1.101 [1.062, 1.127] | 2.89e+05 | 14.83 | 0.9799 |
| real | scalar | batch | 1024 | 1.768e+04 | 1.949e+04 | 1.1 [1.078, 1.129] | 2.882e+05 | 14.72 | 0.9568 |
| complex | simd | single | 16 | 303.3 | 322.5 | 1.056 [1.04, 1.084] | 544.4 | 1.711 | 1.076 |
| complex | simd | batch | 16 | 268.7 | 291 | 1.086 [1.058, 1.106] | 523.8 | 1.819 | 1.082 |
| complex | scalar | single | 16 | 301.3 | 318 | 1.063 [1.055, 1.08] | 541.1 | 1.687 | 1.062 |
| complex | scalar | batch | 16 | 265.8 | 286.8 | 1.082 [1.068, 1.101] | 518.3 | 1.792 | 1.075 |
| complex | simd | single | 64 | 1558 | 1654 | 1.072 [1.051, 1.114] | 8429 | 5.051 | 1.086 |
| complex | simd | batch | 64 | 1497 | 1610 | 1.078 [1.065, 1.106] | 8367 | 5.188 | 1.079 |
| complex | scalar | single | 64 | 1552 | 1674 | 1.076 [1.062, 1.081] | 8483 | 5.079 | 1.075 |
| complex | scalar | batch | 64 | 1503 | 1623 | 1.077 [1.063, 1.085] | 8412 | 5.184 | 1.073 |
| complex | simd | single | 256 | 8023 | 8836 | 1.098 [1.082, 1.127] | 1.351e+05 | 15.28 | 1.081 |
| complex | simd | batch | 256 | 7932 | 8762 | 1.115 [1.083, 1.134] | 1.35e+05 | 15.55 | 1.087 |
| complex | scalar | single | 256 | 7999 | 8687 | 1.081 [1.069, 1.102] | 1.349e+05 | 15.5 | 1.071 |
| complex | scalar | batch | 256 | 7896 | 8634 | 1.094 [1.072, 1.118] | 1.354e+05 | 15.71 | 1.071 |
