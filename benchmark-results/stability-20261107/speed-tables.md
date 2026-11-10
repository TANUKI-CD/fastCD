Paired timings on one host: 12 blocks pinned to CPU 18. Times are medians in ns per product, and brackets are paired bootstrap 95% intervals. The direct control is the ratio of direct timings between the two builds, whose direct arithmetic is identical, so it shows the noise between builds.

| domain | kernels | mode | N | sequential ns | balanced ns | balanced / sequential [95% CI] | direct ns | direct / balanced | direct control |
|:---|:---|:---|---:|---:|---:|:---|---:|---:|---:|
| real | simd | single | 16 | 117.9 | 113.7 | 0.9646 [0.9207, 1.01] | 88.96 | 0.7733 | 0.9753 |
| real | simd | batch | 16 | 54.75 | 51.96 | 0.9477 [0.9137, 0.982] | 55.87 | 1.071 | 0.982 |
| real | scalar | single | 16 | 112.2 | 105.4 | 0.938 [0.8782, 1.015] | 87.94 | 0.8258 | 1.016 |
| real | scalar | batch | 16 | 91.17 | 91.75 | 1.017 [1.002, 1.058] | 55.71 | 0.601 | 0.9719 |
| real | simd | single | 64 | 502 | 522.3 | 1.059 [1.001, 1.107] | 1154 | 2.217 | 1.005 |
| real | simd | batch | 64 | 424.6 | 446.4 | 1.035 [0.9973, 1.088] | 1124 | 2.442 | 1.06 |
| real | scalar | single | 64 | 651.2 | 709.4 | 1.098 [1.073, 1.118] | 1148 | 1.611 | 1.023 |
| real | scalar | batch | 64 | 617.8 | 677.4 | 1.09 [1.061, 1.129] | 1061 | 1.586 | 0.961 |
| real | simd | single | 256 | 2698 | 2682 | 1.01 [0.9818, 1.034] | 1.763e+04 | 6.433 | 0.982 |
| real | simd | batch | 256 | 2589 | 2673 | 1.031 [1.019, 1.054] | 1.697e+04 | 6.461 | 1.016 |
| real | scalar | single | 256 | 3401 | 3722 | 1.098 [1.062, 1.125] | 1.83e+04 | 4.929 | 0.9908 |
| real | scalar | batch | 256 | 3343 | 3722 | 1.107 [1.07, 1.152] | 1.837e+04 | 4.878 | 1.005 |
| real | simd | single | 1024 | 1.41e+04 | 1.434e+04 | 1.015 [0.9923, 1.062] | 2.771e+05 | 19.28 | 0.9774 |
| real | simd | batch | 1024 | 1.425e+04 | 1.468e+04 | 1.049 [0.9838, 1.081] | 2.795e+05 | 18.89 | 0.9905 |
| real | scalar | single | 1024 | 1.75e+04 | 1.945e+04 | 1.124 [1.09, 1.169] | 2.773e+05 | 14.31 | 0.9967 |
| real | scalar | batch | 1024 | 1.713e+04 | 1.933e+04 | 1.128 [1.091, 1.147] | 2.898e+05 | 15 | 1.011 |
| complex | simd | single | 16 | 300 | 306.3 | 1.021 [0.9805, 1.035] | 531.1 | 1.719 | 1.058 |
| complex | simd | batch | 16 | 271 | 274.6 | 1.008 [0.9867, 1.059] | 510 | 1.854 | 1.061 |
| complex | scalar | single | 16 | 302.3 | 314.5 | 1.036 [1.016, 1.056] | 544.1 | 1.736 | 1.084 |
| complex | scalar | batch | 16 | 265.8 | 283.8 | 1.04 [1.022, 1.069] | 510.9 | 1.814 | 1.063 |
| complex | simd | single | 64 | 1588 | 1676 | 1.059 [1.038, 1.1] | 8396 | 4.971 | 1.081 |
| complex | simd | batch | 64 | 1533 | 1592 | 1.053 [1.034, 1.063] | 8381 | 5.208 | 1.078 |
| complex | scalar | single | 64 | 1516 | 1639 | 1.068 [1.049, 1.082] | 8375 | 5.16 | 1.089 |
| complex | scalar | batch | 64 | 1477 | 1600 | 1.085 [1.035, 1.106] | 8287 | 5.114 | 1.086 |
| complex | simd | single | 256 | 7793 | 8690 | 1.115 [1.098, 1.122] | 1.329e+05 | 15.39 | 1.08 |
| complex | simd | batch | 256 | 7835 | 8519 | 1.111 [1.071, 1.122] | 1.334e+05 | 15.79 | 1.107 |
| complex | scalar | single | 256 | 8013 | 8623 | 1.058 [1.051, 1.094] | 1.332e+05 | 15.51 | 1.08 |
| complex | scalar | batch | 256 | 7810 | 8492 | 1.099 [1.085, 1.12] | 1.306e+05 | 15.47 | 1.059 |
