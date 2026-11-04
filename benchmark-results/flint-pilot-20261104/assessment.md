# FLINT comparison pilot: 2026-11-04

This is a 32-pair-per-cell pilot on the recorded Xeon Gold 6148 host. It is not the default 10,000-pair campaign. The calibration target is **0.5 ms for the faster backend**, with 24 paired blocks and two warmup passes. Every selected cell is reported.

[FLINT (Fast Library for Number Theory)](https://flintlib.org/) provides the independent prime-field and polynomial-basis extension arithmetic. The scalar experiment compares public field multiplication APIs. The CD experiment compares the native quasilinear dispatcher against the maintained CD recurrence using FLINT coefficients; FLINT does not provide the CD algorithm.

The fields were fixed before timing: the nine default finite fields plus degree 12 at characteristics 3, 13 and 2147483647. Each field has one scalar cell and CD dimensions 2, 16, 64 and 256. This covers several characteristic sizes and the maximum supported extension degree without selecting fields from observed wins.

## Results

All 60 cells passed. The independent oracle checked 1,920 input pairs; both implementations passed 3,840 initial product checks and 92,160 final timed-output pair checks across 2,880 observations. The sum of measured-cell wall times, including setup and verification, was 63.4 seconds.

Ratios are paired geometric means of **FLINT time / fastCD time**; values above 1 favor fastCD. The intervals are 95% Student-t intervals over paired blocks, not ratios of independently pooled means.

| Field | fastCD scalar ns | FLINT scalar ns | FLINT/fastCD (95% CI) |
|---|---:|---:|---|
| F(3^1) | 2.09 | 3.03 | 1.450 (1.430–1.470) |
| F(5^1) | 1.98 | 2.89 | 1.462 (1.448–1.476) |
| F(13^1) | 2.01 | 2.90 | 1.448 (1.429–1.467) |
| F(3^5) | 53.15 | 119.24 | 2.244 (2.211–2.278) |
| F(5^2) | 10.40 | 44.62 | 4.292 (4.257–4.328) |
| F(13^5) | 57.42 | 158.96 | 2.765 (2.708–2.823) |
| F(65521^1) | 2.12 | 2.91 | 1.369 (1.353–1.386) |
| F(65521^2) | 10.03 | 51.83 | 5.167 (5.090–5.245) |
| F(2147483647^1) | 2.20 | 3.20 | 1.458 (1.428–1.488) |
| F(3^12) | 259.41 | 410.05 | 1.580 (1.550–1.612) |
| F(13^12) | 320.45 | 393.92 | 1.229 (1.216–1.242) |
| F(2147483647^12) | 346.16 | 801.57 | 2.316 (2.287–2.344) |

Scalar ratios range from 1.229 to 5.167; every scalar interval lies above 1. These selected cases support the public coefficient kernel's performance on this build. Complete-CD ratios range from 0.693 to 239.896 across all measured dimensions, and from 28.920 to 239.896 at N=256. The [complete tables](results.md) retain all scalar and CD cells.

## Scope and reproduction

The 32-pair corpus has different cache behavior from a 10,000-pair campaign. The short target, fixed polynomial selection and shared host also limit generalization. Complete CD timings include native degree specialization, SIMD packing, per-call maps, representation layouts and validation; they do not isolate the scalar multiplication contribution or compare every FLINT representation. No universal speedup or real AUTO change is inferred.

Native code uses GCC 16.2.1 Release flags, without -march=native; runtime dispatch may select the separately compiled SIMD kernels. FLINT 3.6.0 uses -O3, its default assembly, AVX2/AVX-512 build options disabled, shared linkage and one arithmetic thread. The dependency build record was matched to the measured library SHA256. See [dependency-build.json](dependency-build.json) for the release hash and build options. Native compiler/source/executable hashes, loaded library hashes, exact moduli, corpora and every observation are in [results.json](results.json).

The [recorded command](commands.json) gives the full field selection and overrides. To reproduce from this repository after installing FLINT:

```sh
cmake -S . -B build-flint -DCMAKE_BUILD_TYPE=Release \
    -DFASTCD_BUILD_FLINT_BENCHMARK=ON -DFASTCD_WARNINGS_AS_ERRORS=ON
cmake --build build-flint --parallel
python3 benchmarks/benchmark.py --protocol flint \
    --runner build-flint/fastcd_flint_benchmark \
    --output benchmark-results/flint-pilot-new.json \
    --fields 3:1,5:1,13:1,3:5,5:2,13:5,65521:1,65521:2,2147483647:1,3:12,13:12,2147483647:12 \
    --dimensions 2,16,64,256 --trials 32 --blocks 24 --warmup 2 \
    --min-time-ms 0.5 --seed 20261104
```

Use a new output path. Add CMAKE_PREFIX_PATH for a nonstandard FLINT installation, as described in the [benchmark guide](../../docs/benchmark.md). The default protocol remains the real benchmark; this pilot does not change its parameters or archived results.
