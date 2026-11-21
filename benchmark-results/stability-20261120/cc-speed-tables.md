Cariow–Cariowa products, 12 paired blocks pinned to logical CPU 18. The baseline and measured columns are the builds of the baseline and measured commits; times are medians in ns per product, and brackets are paired bootstrap 95% intervals. The direct control is the measured / baseline ratio of direct timings, whose arithmetic is identical in both builds.

| kernels | mode | N | baseline ns | measured ns | measured / baseline [95% CI] | direct control |
|:---|:---|---:|---:|---:|:---|---:|
| simd | single | 16 | 119.82 | 116.16 | 0.9655 [0.9211, 0.9924] | 0.8837 |
| simd | batch | 16 | 50.374 | 50.492 | 1.005 [0.9924, 1.028] | 0.9924 |
| scalar | single | 16 | 109.25 | 116.69 | 1.064 [1.033, 1.094] | 0.9292 |
| scalar | batch | 16 | 90.906 | 92.845 | 1.028 [0.9757, 1.046] | 0.9961 |
| simd | single | 64 | 1830.5 | 1827.8 | 0.9845 [0.9774, 0.9943] | 0.9947 |
| simd | batch | 64 | 950.64 | 956.28 | 1.021 [0.9572, 1.045] | 1.004 |
| scalar | single | 64 | 1826.1 | 1834.5 | 1.02 [0.9907, 1.042] | 1.012 |
| scalar | batch | 64 | 1819.7 | 1822.5 | 1.002 [0.982, 1.011] | 1.001 |
| simd | single | 128 | 7745.9 | 7577.1 | 0.9858 [0.9449, 1.013] | 0.9846 |
| simd | batch | 128 | 4731.5 | 4774.4 | 1.007 [0.9629, 1.037] | 0.9916 |
| scalar | single | 128 | 7834.7 | 7776.3 | 0.9955 [0.9697, 1.017] | 0.9967 |
| scalar | batch | 128 | 7849 | 7772.2 | 0.9923 [0.9693, 1.028] | 0.9991 |
| simd | single | 256 | 28754 | 28402 | 0.9978 [0.9573, 1.013] | 1.006 |
| simd | batch | 256 | 17387 | 17946 | 1.029 [1.011, 1.047] | 1.027 |
| scalar | single | 256 | 27693 | 27590 | 0.9953 [0.9678, 1.037] | 0.9844 |
| scalar | batch | 256 | 27846 | 28084 | 1.013 [0.9738, 1.048] | 0.9775 |
| simd | single | 512 | 1.0829e+05 | 1.1026e+05 | 1.001 [0.9809, 1.032] | 1.017 |
| simd | batch | 512 | 64209 | 67309 | 1.04 [0.997, 1.104] | 0.9887 |
| scalar | single | 512 | 1.0522e+05 | 1.1032e+05 | 1.051 [1.031, 1.07] | 1.001 |
| scalar | batch | 512 | 1.0325e+05 | 1.0774e+05 | 1.026 [1.003, 1.09] | 1.008 |
| simd | single | 1024 | 4.0872e+05 | 4.3173e+05 | 1.057 [1.022, 1.093] | 1.004 |
| simd | batch | 1024 | 4.2159e+05 | 4.2772e+05 | 1.013 [0.9671, 1.041] | 1.006 |
| scalar | single | 1024 | 3.9751e+05 | 4.2098e+05 | 1.049 [1.031, 1.083] | 1.014 |
| scalar | batch | 1024 | 4.0254e+05 | 4.1546e+05 | 1.023 [0.9883, 1.049] | 0.9983 |
