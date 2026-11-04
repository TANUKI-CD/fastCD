# Finite-field comparison with FLINT

FLINT (Fast Library for Number Theory) is an independent C library. This experiment compares its nmod_mul (prime fields) and fq_nmod_mul (polynomial-basis extensions) with fastCD's public arithmetic.

FLINT 3.6.0; one arithmetic thread; 32 pairs per cell; 24 paired blocks; 2 warmup passes; 0.5 ms calibration target; Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz.

Times are means in ns/product. FLINT/fastCD is the geometric mean of paired block ratios with a 95% Student-t interval. Values above 1 favor fastCD. Scalar and complete CD products are separate workloads.

## Scalar field multiplication

| Field | N | fastCD | FLINT | FLINT/fastCD (95% CI) |
|---|---:|---:|---:|---| 
| F(3^1) | 1 | 2.09 | 3.03 | 1.450 (1.430–1.470) |
| F(5^1) | 1 | 1.98 | 2.89 | 1.462 (1.448–1.476) |
| F(13^1) | 1 | 2.01 | 2.90 | 1.448 (1.429–1.467) |
| F(3^5) | 1 | 53.15 | 119.24 | 2.244 (2.211–2.278) |
| F(5^2) | 1 | 10.40 | 44.62 | 4.292 (4.257–4.328) |
| F(13^5) | 1 | 57.42 | 158.96 | 2.765 (2.708–2.823) |
| F(65521^1) | 1 | 2.12 | 2.91 | 1.369 (1.353–1.386) |
| F(65521^2) | 1 | 10.03 | 51.83 | 5.167 (5.090–5.245) |
| F(2147483647^1) | 1 | 2.20 | 3.20 | 1.458 (1.428–1.488) |
| F(3^12) | 1 | 259.41 | 410.05 | 1.580 (1.550–1.612) |
| F(13^12) | 1 | 320.45 | 393.92 | 1.229 (1.216–1.242) |
| F(2147483647^12) | 1 | 346.16 | 801.57 | 2.316 (2.287–2.344) |

## Complete CD multiplication

| Field | N | fastCD | FLINT | FLINT/fastCD (95% CI) |
|---|---:|---:|---:|---| 
| F(3^1) | 2 | 36.51 | 26.64 | 0.730 (0.718–0.742) |
| F(3^1) | 16 | 306.37 | 970.31 | 3.167 (3.137–3.198) |
| F(3^1) | 64 | 875.73 | 11217.01 | 12.904 (12.313–13.525) |
| F(3^1) | 256 | 889.36 | 67723.61 | 76.485 (73.776–79.294) |
| F(5^1) | 2 | 38.14 | 26.44 | 0.693 (0.684–0.702) |
| F(5^1) | 16 | 297.84 | 1004.81 | 3.374 (3.330–3.418) |
| F(5^1) | 64 | 776.00 | 11921.64 | 15.432 (14.867–16.019) |
| F(5^1) | 256 | 1228.66 | 74074.22 | 60.596 (58.396–62.879) |
| F(13^1) | 2 | 37.54 | 26.22 | 0.698 (0.689–0.707) |
| F(13^1) | 16 | 289.55 | 938.59 | 3.240 (3.209–3.272) |
| F(13^1) | 64 | 791.14 | 12356.33 | 15.717 (15.071–16.390) |
| F(13^1) | 256 | 1404.93 | 77566.62 | 55.256 (54.094–56.443) |
| F(3^5) | 2 | 245.79 | 584.86 | 2.379 (2.352–2.406) |
| F(3^5) | 16 | 997.48 | 30567.52 | 30.673 (30.205–31.150) |
| F(3^5) | 64 | 1729.32 | 229332.94 | 133.050 (128.285–137.992) |
| F(3^5) | 256 | 5519.84 | 1320023.30 | 239.896 (232.022–248.038) |
| F(5^2) | 2 | 60.39 | 241.01 | 3.994 (3.929–4.060) |
| F(5^2) | 16 | 567.90 | 14486.24 | 25.533 (25.023–26.054) |
| F(5^2) | 64 | 1841.75 | 107627.42 | 58.467 (57.170–59.793) |
| F(5^2) | 256 | 7611.36 | 651147.29 | 85.952 (82.631–89.406) |
| F(13^5) | 2 | 255.25 | 713.79 | 2.793 (2.729–2.858) |
| F(13^5) | 16 | 3224.10 | 34622.28 | 10.743 (10.580–10.908) |
| F(13^5) | 64 | 4642.58 | 261412.13 | 56.430 (54.957–57.942) |
| F(13^5) | 256 | 20665.88 | 1532881.23 | 74.912 (71.181–78.839) |
| F(65521^1) | 2 | 37.97 | 27.57 | 0.726 (0.716–0.736) |
| F(65521^1) | 16 | 312.71 | 1000.58 | 3.201 (3.165–3.237) |
| F(65521^1) | 64 | 884.28 | 12936.93 | 14.769 (14.058–15.516) |
| F(65521^1) | 256 | 2655.80 | 78561.81 | 29.654 (28.740–30.596) |
| F(65521^2) | 2 | 60.68 | 257.21 | 4.241 (4.184–4.299) |
| F(65521^2) | 16 | 1174.68 | 13612.36 | 11.623 (11.202–12.059) |
| F(65521^2) | 64 | 2906.89 | 99892.30 | 34.654 (32.966–36.429) |
| F(65521^2) | 256 | 13032.17 | 604398.10 | 46.691 (44.483–49.008) |
| F(2147483647^1) | 2 | 39.15 | 28.42 | 0.726 (0.712–0.740) |
| F(2147483647^1) | 16 | 308.20 | 1006.47 | 3.265 (3.223–3.309) |
| F(2147483647^1) | 64 | 790.65 | 12518.45 | 15.924 (15.231–16.648) |
| F(2147483647^1) | 256 | 2694.86 | 77310.01 | 28.920 (27.548–30.360) |
| F(3^12) | 2 | 1071.80 | 1710.78 | 1.596 (1.569–1.623) |
| F(3^12) | 16 | 4529.52 | 81095.19 | 17.908 (17.637–18.183) |
| F(3^12) | 64 | 4896.24 | 605775.11 | 123.950 (120.584–127.409) |
| F(3^12) | 256 | 18029.09 | 3521642.15 | 196.013 (189.374–202.886) |
| F(13^12) | 2 | 1319.53 | 1698.84 | 1.288 (1.270–1.306) |
| F(13^12) | 16 | 58562.53 | 80163.79 | 1.369 (1.352–1.386) |
| F(13^12) | 64 | 41073.87 | 581945.65 | 14.174 (13.920–14.432) |
| F(13^12) | 256 | 96322.22 | 3362360.82 | 35.031 (33.987–36.108) |
| F(2147483647^12) | 2 | 1405.67 | 3286.85 | 2.339 (2.310–2.368) |
| F(2147483647^12) | 16 | 14377.25 | 145855.38 | 10.179 (9.859–10.509) |
| F(2147483647^12) | 64 | 30330.48 | 1054773.43 | 35.149 (33.299–37.101) |
| F(2147483647^12) | 256 | 108350.68 | 6196004.32 | 57.632 (54.553–60.884) |

CD results compare fastCD's public quasilinear dispatcher with the same maintained algebraic recurrence using FLINT coefficients. FLINT does not provide the CD algorithm. Native degree specialization, packed SIMD backends, element layouts and normalization costs are part of this implementation comparison; scalar results measure coefficient multiplication separately.

Both backends use identical operands and defining polynomials, preallocated outputs/scratch and a common calibrated pass count. Setup and conversions are excluded; work inside FLINT's arithmetic routines is included. Every input pair and every actual final timed buffer were checked against an independent polynomial/basis oracle.

The adjacent JSON records exact moduli, corpora, raw observations, verification counts, versions, build/source hashes and resolved shared library hashes where available. This fixed selection of fields is not a guarantee for other fields, representations or machines.
