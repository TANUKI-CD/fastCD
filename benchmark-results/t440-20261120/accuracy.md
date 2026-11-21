# Accuracy of the fused sums

fastCD 1.5.0, 689 sums and 84 covariances, on Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz.

Each cell is the median / maximum of `eta = ||S_hat - S||_2 / (u sum_p ||x_p||_2 ||y_p||_2)` with `u = 2^-53`. `fused` is the automatic algorithm; the two `products` columns sum a batch of products in sequence and with NumPy's pairwise `sum`; `cross moments` is the binary64 matrix product of the samples contracted with the basis signs.

## Gaussian and uniform samples of 1,024 elements

| N | Cases | fused | fused, direct | fused, quasilinear | products, sequential | products, pairwise | cross moments |
|---:|---:|---|---|---|---|---|---|
| 1 | 16 | 0.05 / 0.25 | 0.05 / 0.25 | 0.05 / 0.25 | 0.16 / 0.75 | 0.05 / 0.15 | 0.06 / 0.20 |
| 2 | 16 | 0.06 / 0.18 | 0.06 / 0.18 | 0.06 / 0.13 | 0.15 / 0.52 | 0.06 / 0.14 | 0.35 / 0.78 |
| 4 | 16 | 0.06 / 0.10 | 0.06 / 0.10 | 0.06 / 0.13 | 0.27 / 0.58 | 0.05 / 0.08 | 0.29 / 0.68 |
| 8 | 16 | 0.07 / 0.09 | 0.07 / 0.09 | 0.08 / 0.12 | 0.22 / 0.35 | 0.06 / 0.08 | 0.33 / 0.46 |
| 16 | 16 | 0.09 / 0.13 | 0.09 / 0.11 | 0.09 / 0.13 | 0.28 / 0.42 | 0.08 / 0.13 | 0.29 / 0.39 |
| 32 | 16 | 0.09 / 0.13 | 0.22 / 0.29 | 0.09 / 0.13 | 0.30 / 0.43 | 0.08 / 0.09 | 0.19 / 0.22 |
| 64 | 16 | 0.11 / 0.14 | 0.32 / 0.35 | 0.11 / 0.14 | 0.29 / 0.36 | 0.08 / 0.10 | 0.19 / 0.22 |
| 128 | 8 | 0.12 / 0.14 | 0.44 / 0.53 | 0.12 / 0.14 | 0.30 / 0.32 | 0.09 / 0.10 | 0.20 / 0.22 |
| 256 | 8 | 0.13 / 0.15 | 0.62 / 0.65 | 0.13 / 0.15 | 0.31 / 0.35 | 0.09 / 0.10 | 0.24 / 0.25 |
| 512 | 4 | 0.14 / 0.14 | 0.85 / 0.86 | 0.14 / 0.14 | 0.30 / 0.34 | 0.10 / 0.10 | 0.28 / 0.29 |
| 1024 | 2 | 0.15 / 0.15 | 1.19 / 1.20 | 0.15 / 0.15 | 0.30 / 0.31 | 0.10 / 0.10 | 0.35 / 0.35 |

## By family, samples of 1,024 elements, all dimensions

| Family | Cases | fused | fused, direct | fused, quasilinear | products, sequential | products, pairwise | cross moments |
|---|---:|---|---|---|---|---|---|
| gaussian | 67 | 0.09 / 0.25 | 0.10 / 1.17 | 0.10 / 0.25 | 0.30 / 0.75 | 0.08 / 0.13 | 0.22 / 0.78 |
| uniform | 67 | 0.09 / 0.15 | 0.08 / 1.20 | 0.10 / 0.15 | 0.26 / 0.59 | 0.08 / 0.15 | 0.24 / 0.68 |
| rademacher | 64 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| self | 64 | 0.35 / 2.60 | 0.61 / 7.37 | 0.34 / 2.60 | 3.44 / 14.97 | 0.23 / 1.40 | 1.44 / 14.44 |
| conjugate | 64 | 0.50 / 2.07 | 0.68 / 5.22 | 0.57 / 2.07 | 5.16 / 18.63 | 0.44 / 1.89 | 1.62 / 7.58 |
| cancelling | 64 | 0.03 / 0.05 | 0.04 / 0.44 | 0.04 / 0.05 | 0.21 / 0.64 | 0.00 / 0.00 | 0.17 / 0.42 |
| offset | 67 | 0.58 / 2.06 | 0.83 / 10.26 | 0.54 / 2.06 | 6.46 / 21.95 | 0.41 / 1.07 | 4.64 / 14.89 |
| graded | 64 | 0.46 / 0.87 | 0.50 / 4.10 | 0.45 / 0.87 | 1.19 / 2.50 | 0.41 / 0.75 | 0.87 / 2.03 |
| sparse | 56 | 0.07 / 0.12 | 0.06 / 0.12 | 0.08 / 0.12 | 0.10 / 0.73 | 0.05 / 0.09 | 0.07 / 0.27 |

## By sample size, Gaussian and offset samples, N <= 64

| Count | Cases | fused | fused, direct | fused, quasilinear | products, sequential | products, pairwise | cross moments |
|---:|---:|---|---|---|---|---|---|
| 64 | 56 | 0.31 / 3.44 | 0.32 / 1.18 | 0.31 / 3.44 | 0.55 / 5.02 | 0.28 / 1.01 | 0.45 / 3.48 |
| 1024 | 112 | 0.15 / 2.06 | 0.33 / 8.88 | 0.13 / 2.06 | 0.57 / 21.95 | 0.12 / 0.97 | 0.41 / 14.89 |
| 4096 | 56 | 0.12 / 2.04 | 0.29 / 5.17 | 0.09 / 2.04 | 0.82 / 25.25 | 0.07 / 0.88 | 0.54 / 19.82 |

## Covariance, samples of 1,024 elements, N <= 64

Here the scale is `sum_p ||x_p - mean(x)||_2 ||y_p - mean(y)||_2 / (count - 1)`. `centered products` and `centered moments` subtract the means with NumPy first.

| Family | Cases | covariance | centered products | centered moments |
|---|---:|---|---|---|
| gaussian | 28 | 0.09 / 0.14 | 0.30 / 0.64 | 0.22 / 0.87 |
| offset | 28 | 0.08 / 0.11 | 0.29 / 0.47 | 0.21 / 0.46 |
| far_offset | 28 | 0.11 / 0.21 | 4902.89 / 8165.83 | 4902.93 / 8165.83 |

The record and the plane layout returned identical bits in every sum.
