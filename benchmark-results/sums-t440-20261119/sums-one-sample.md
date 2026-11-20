# Fused-sum benchmark

Measured 1,024 elements per cell in samples of the stated count, 60 blocks at separate stack offsets, on Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz.

Times are means in ns per element. `products` is one batch of products followed by a sum; `records` and `planes` are fused sums with the automatic algorithm; `dgemm` is one matrix product and the contraction with the basis signs, with Intel MKL 2026.2.1.

| N | Count | Products | Records | Planes, direct | Planes, quasilinear | dgemm |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1024 | 4.02 | 2.94 | 0.35 | 0.42 | 1.46 |
| 4 | 1024 | 6.80 | 3.16 | 0.98 | 1.04 | 1.92 |
| 8 | 1024 | 15.87 | 5.30 | 2.14 | 2.76 | 2.79 |
| 16 | 1024 | 58.31 | 16.05 | 10.99 | 12.74 | 7.77 |
| 32 | 1024 | 203.17 | 35.99 | 463.26 | 27.83 | 35.69 |
| 64 | 1024 | 509.77 | 99.69 | 3609.27 | 85.86 | 136.31 |
| 128 | 1024 | 1254.17 | 254.32 |  | 241.55 | 598.18 |
| 256 | 1024 | 2870.06 | 606.83 |  | 595.27 | 1887.45 |
| 512 | 1024 | 7034.69 | 1685.01 |  | 1720.54 | 7522.37 |
| 1024 | 1024 | 16623.15 | 4459.04 |  | 4876.58 | 28874.97 |

Ratios are paired geometric means with 95% Student-t intervals. Each divides the time of the first task by that of the second, so values above 1 favor the second. In the last column, values above 1 favor quasilinear.

| N | Count | Products/records | Products/planes | dgemm/records | dgemm/planes | Direct/quasilinear (planes) |
|---:|---:|---|---|---|---|---|
| 2 | 1024 | 1.37 (1.35–1.39) | 11.57 (11.24–11.92) | 0.50 (0.48–0.51) | 4.20 (4.05–4.35) | 0.84 (0.81–0.87) |
| 4 | 1024 | 2.16 (2.10–2.22) | 6.97 (6.74–7.21) | 0.61 (0.58–0.63) | 1.96 (1.88–2.04) | 0.94 (0.91–0.97) |
| 8 | 1024 | 3.00 (2.93–3.08) | 7.46 (7.26–7.65) | 0.53 (0.51–0.54) | 1.31 (1.27–1.35) | 0.78 (0.76–0.79) |
| 16 | 1024 | 3.65 (3.56–3.74) | 5.33 (5.17–5.50) | 0.48 (0.47–0.50) | 0.71 (0.68–0.73) | 0.86 (0.84–0.89) |
| 32 | 1024 | 5.67 (5.54–5.80) | 7.34 (7.09–7.60) | 0.99 (0.96–1.02) | 1.28 (1.25–1.32) | 16.66 (16.19–17.15) |
| 64 | 1024 | 5.13 (5.03–5.23) | 5.95 (5.83–6.07) | 1.37 (1.33–1.40) | 1.59 (1.55–1.62) | 41.96 (41.00–42.94) |
| 128 | 1024 | 4.94 (4.85–5.03) | 5.20 (5.11–5.30) | 2.35 (2.28–2.41) | 2.47 (2.42–2.52) |  |
| 256 | 1024 | 4.74 (4.64–4.84) | 4.84 (4.72–4.95) | 3.11 (3.03–3.19) | 3.17 (3.09–3.26) |  |
| 512 | 1024 | 4.20 (4.10–4.29) | 4.10 (4.02–4.19) | 4.46 (4.34–4.59) | 4.36 (4.26–4.47) |  |
| 1024 | 1024 | 3.74 (3.67–3.80) | 3.41 (3.36–3.47) | 6.46 (6.29–6.64) | 5.91 (5.77–6.04) |  |

The first sample of every cell was verified against a long double contraction of the cross moments, and every sample for agreement between the tasks, outside timing. The two fused layouts returned identical bits wherever they ran the same algorithm.

Raw observations, corpus hashes, the runner, build and source hashes and the BLAS library are recorded in the adjacent JSON.
