Cariow–Cariowa products, 12 paired blocks pinned to logical CPU 18. The baseline and measured columns are the builds of the baseline and measured commits; times are medians in ns per product, and brackets are paired bootstrap 95% intervals. The direct control is the measured / baseline ratio of direct timings, whose arithmetic is identical in both builds.

| kernels | mode | N | baseline ns | measured ns | measured / baseline [95% CI] | direct control |
|:---|:---|---:|---:|---:|:---|---:|
| simd | single | 16 | 120.43 | 118.69 | 0.9817 [0.8992, 1.03] | 0.9935 |
| simd | batch | 16 | 50.249 | 50.93 | 1.025 [0.9902, 1.054] | 0.9998 |
| scalar | single | 16 | 108.59 | 110.29 | 1.016 [0.9494, 1.073] | 0.9872 |
| scalar | batch | 16 | 90.982 | 92.107 | 1.002 [0.9865, 1.045] | 0.9831 |
| simd | single | 64 | 1949 | 1931.1 | 0.9939 [0.9804, 0.9996] | 0.984 |
| simd | batch | 64 | 1033.7 | 1017.4 | 0.9645 [0.9401, 0.9842] | 0.9955 |
| scalar | single | 64 | 1970.7 | 1965.3 | 1.009 [0.9803, 1.048] | 1.016 |
| scalar | batch | 64 | 1926.1 | 1919.3 | 0.9964 [0.9408, 1.078] | 1.01 |
| simd | single | 128 | 7835.9 | 7839 | 0.9864 [0.9454, 1.031] | 0.9938 |
| simd | batch | 128 | 4920.3 | 4690.5 | 0.948 [0.909, 1.061] | 0.9535 |
| scalar | single | 128 | 7967.2 | 7603 | 0.9714 [0.9264, 0.9811] | 1.055 |
| scalar | batch | 128 | 7471.7 | 7558.4 | 1.011 [0.9682, 1.075] | 1 |
| simd | single | 256 | 28144 | 27958 | 0.9564 [0.8774, 1.027] | 1.018 |
| simd | batch | 256 | 18061 | 17008 | 0.9484 [0.8945, 1.058] | 0.9741 |
| scalar | single | 256 | 27859 | 28233 | 1.024 [0.9897, 1.031] | 1.007 |
| scalar | batch | 256 | 28721 | 27921 | 0.9708 [0.9573, 1.018] | 1.001 |
| simd | single | 512 | 1.0515e+05 | 1.0804e+05 | 1.033 [1.01, 1.056] | 1.025 |
| simd | batch | 512 | 64546 | 69934 | 1.108 [1.068, 1.163] | 1.011 |
| scalar | single | 512 | 1.0221e+05 | 1.1243e+05 | 1.077 [1.055, 1.12] | 1.016 |
| scalar | batch | 512 | 1.0558e+05 | 1.0989e+05 | 1.024 [1.009, 1.136] | 0.9916 |
| simd | single | 1024 | 4.0953e+05 | 4.2515e+05 | 1.03 [0.9587, 1.069] | 1.009 |
| simd | batch | 1024 | 4.1248e+05 | 4.1765e+05 | 0.9972 [0.9644, 1.045] | 1.017 |
| scalar | single | 1024 | 4.0776e+05 | 4.2321e+05 | 1.049 [1.011, 1.111] | 1.015 |
| scalar | batch | 1024 | 4.0842e+05 | 4.293e+05 | 1.051 [1.031, 1.096] | 1.015 |
