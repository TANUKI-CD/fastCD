# Fused-sum benchmark

Measured 1,024 elements per cell in samples of the stated count, 60 blocks at separate stack offsets, on Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz.

Times are means in ns per element. `products` is one batch of products followed by a sum; `records` and `planes` are fused sums with the automatic algorithm; `dgemm` is one matrix product and the contraction with the basis signs, with Intel MKL 2026.2.1.

| N | Count | Products | Records | Planes, direct | Planes, quasilinear | dgemm |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1024 | 4.02 | 2.23 | 0.36 | 0.42 | 1.44 |
| 4 | 1024 | 6.73 | 2.33 | 0.97 | 1.02 | 1.91 |
| 8 | 1024 | 15.92 | 4.48 | 2.21 | 2.82 | 2.88 |
| 16 | 1024 | 57.21 | 12.87 | 10.57 | 12.38 | 7.74 |
| 32 | 1024 | 198.39 | 33.96 | 441.55 | 26.46 | 33.80 |
| 64 | 1024 | 500.96 | 95.47 | 3475.76 | 83.25 | 130.99 |
| 128 | 1024 | 1221.53 | 248.07 |  | 234.78 | 553.96 |
| 256 | 1024 | 2821.99 | 582.95 |  | 582.36 | 1846.63 |
| 512 | 1024 | 6843.76 | 1571.31 |  | 1681.28 | 6988.41 |
| 1024 | 1024 | 16741.27 | 4580.57 |  | 4974.84 | 29322.95 |

Ratios are paired geometric means with 95% Student-t intervals. Each divides the time of the first task by that of the second, so values above 1 favor the second. In the last column, values above 1 favor quasilinear.

| N | Count | Products/records | Products/planes | dgemm/records | dgemm/planes | Direct/quasilinear (planes) |
|---:|---:|---|---|---|---|---|
| 2 | 1024 | 1.80 (1.77–1.83) | 11.30 (10.99–11.62) | 0.64 (0.63–0.65) | 4.04 (3.94–4.14) | 0.85 (0.83–0.87) |
| 4 | 1024 | 2.90 (2.84–2.96) | 6.97 (6.82–7.13) | 0.82 (0.79–0.85) | 1.97 (1.90–2.03) | 0.95 (0.93–0.97) |
| 8 | 1024 | 3.56 (3.49–3.63) | 7.22 (7.06–7.37) | 0.64 (0.63–0.66) | 1.30 (1.27–1.34) | 0.79 (0.77–0.80) |
| 16 | 1024 | 4.46 (4.36–4.57) | 4.63 (4.53–4.74) | 0.60 (0.58–0.62) | 0.62 (0.61–0.64) | 0.85 (0.83–0.88) |
| 32 | 1024 | 5.86 (5.73–5.99) | 7.52 (7.32–7.73) | 1.00 (0.97–1.03) | 1.28 (1.24–1.32) | 16.72 (16.35–17.09) |
| 64 | 1024 | 5.26 (5.16–5.36) | 6.03 (5.89–6.18) | 1.37 (1.34–1.40) | 1.57 (1.54–1.61) | 41.78 (40.97–42.60) |
| 128 | 1024 | 4.94 (4.85–5.04) | 5.21 (5.12–5.31) | 2.23 (2.18–2.29) | 2.35 (2.31–2.40) |  |
| 256 | 1024 | 4.85 (4.77–4.94) | 4.86 (4.76–4.96) | 3.16 (3.08–3.25) | 3.17 (3.09–3.26) |  |
| 512 | 1024 | 4.37 (4.28–4.45) | 4.08 (3.99–4.18) | 4.45 (4.35–4.55) | 4.16 (4.07–4.25) |  |
| 1024 | 1024 | 3.66 (3.60–3.73) | 3.37 (3.31–3.44) | 6.38 (6.20–6.57) | 5.88 (5.74–6.02) |  |

The first 1 sample(s) of every cell were verified against a long double contraction of the cross moments, and every sample for agreement between the tasks, outside timing. The two fused layouts returned identical bits wherever they ran the same algorithm.

Raw observations, corpus hashes, the runner, build and source hashes and the BLAS library are recorded in the adjacent JSON.
