# Fused-sum benchmark

Measured 1,024 elements per cell in samples of the stated count, 60 blocks at separate stack offsets, on Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz.

Times are means in ns per element. `products` is one batch of products followed by a sum; `records` and `planes` are fused sums with the automatic algorithm; `dgemm` is one matrix product and the contraction with the basis signs, with Intel MKL 2026.2.1.

| N | Count | Products | Records | Planes, direct | Planes, quasilinear | dgemm |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1024 | 3.95 | 2.20 | 0.35 | 0.41 | 1.42 |
| 4 | 1024 | 6.74 | 2.16 | 0.95 | 1.03 | 1.89 |
| 8 | 1024 | 16.28 | 4.35 | 2.21 | 2.79 | 2.88 |
| 16 | 1024 | 56.67 | 12.63 | 10.28 | 11.84 | 7.40 |
| 32 | 1024 | 197.51 | 33.31 | 437.41 | 26.35 | 34.66 |
| 64 | 1024 | 497.82 | 95.71 | 3449.51 | 83.96 | 130.90 |
| 128 | 1024 | 1228.75 | 252.20 |  | 243.61 | 590.43 |
| 256 | 1024 | 2814.78 | 601.10 |  | 600.34 | 1847.05 |
| 512 | 1024 | 6799.75 | 1518.39 |  | 1590.47 | 6880.14 |
| 1024 | 1024 | 16884.18 | 4620.59 |  | 5385.78 | 29063.64 |

Ratios are paired geometric means with 95% Student-t intervals. Each divides the time of the first task by that of the second, so values above 1 favor the second. In the last column, values above 1 favor quasilinear.

| N | Count | Products/records | Products/planes | dgemm/records | dgemm/planes | Direct/quasilinear (planes) |
|---:|---:|---|---|---|---|---|
| 2 | 1024 | 1.79 (1.76–1.82) | 11.27 (11.00–11.55) | 0.65 (0.63–0.66) | 4.06 (3.96–4.16) | 0.86 (0.84–0.88) |
| 4 | 1024 | 3.12 (3.06–3.18) | 7.08 (6.92–7.23) | 0.87 (0.85–0.90) | 1.98 (1.92–2.04) | 0.93 (0.90–0.95) |
| 8 | 1024 | 3.76 (3.67–3.84) | 7.37 (7.17–7.58) | 0.66 (0.64–0.68) | 1.30 (1.26–1.35) | 0.79 (0.78–0.81) |
| 16 | 1024 | 4.51 (4.41–4.61) | 4.78 (4.67–4.90) | 0.59 (0.57–0.60) | 0.62 (0.61–0.64) | 0.87 (0.84–0.89) |
| 32 | 1024 | 5.95 (5.81–6.09) | 7.52 (7.35–7.69) | 1.04 (1.00–1.08) | 1.31 (1.28–1.35) | 16.60 (16.24–16.97) |
| 64 | 1024 | 5.22 (5.09–5.34) | 5.94 (5.79–6.09) | 1.37 (1.32–1.41) | 1.56 (1.52–1.59) | 41.08 (40.31–41.85) |
| 128 | 1024 | 4.89 (4.78–5.00) | 5.06 (4.94–5.19) | 2.34 (2.27–2.41) | 2.42 (2.36–2.49) |  |
| 256 | 1024 | 4.70 (4.60–4.80) | 4.71 (4.59–4.82) | 3.07 (3.00–3.15) | 3.08 (3.01–3.15) |  |
| 512 | 1024 | 4.49 (4.41–4.58) | 4.28 (4.20–4.36) | 4.53 (4.42–4.65) | 4.32 (4.25–4.40) |  |
| 1024 | 1024 | 3.67 (3.59–3.74) | 3.14 (3.09–3.19) | 6.29 (6.14–6.45) | 5.39 (5.28–5.50) |  |

A long double contraction of the cross moments verified 10 of 10 samples. Every sample was checked for agreement between the tasks, outside timing. The two fused layouts returned identical bits wherever they ran the same algorithm.

Raw observations, corpus hashes, the runner, build and source hashes and the BLAS library are recorded in the adjacent JSON.
