# Fused sums: benchmark and accuracy, 2026-11-20 on the T440

This directory holds the measurements of the fused sums of products that
[`docs/benchmark.md`](../../docs/benchmark.md#fused-sums) and
[`docs/stability.md`](../../docs/stability.md#fused-sums) report. It follows
[`sums-t440-20261119`](../sums-t440-20261119/README.md), which measured an
earlier state of the source in which `AUTO` used the direct sum at
dimension 16.

## Host and builds

* Dell PowerEdge T440, Intel Xeon Gold 6148 (80 logical CPUs, two sockets,
  two SMT threads per core), Linux 7.2.11, shared host; no CPU governor,
  turbo or SMT settings were changed.
* Source: a working tree with uncommitted changes on commit `793c36f`. Each
  benchmark JSON file records that state and the SHA-256 of every measured
  source file, and the accuracy JSON the SHA-256 of the shared library it
  loaded.
* Builds: CMake Release (`-O3 -DNDEBUG`, `-fno-fast-math -ffp-contract=off`),
  GCC 16.2.1. The runners, but not the library, were built with
  `-Wa,-mbranches-within-32B-boundaries`. Three builds differ in the kernel
  of the fused sums that they select on this processor:
  * default options: the AVX-512 kernel;
  * `-DFASTCD_TUNING_DEFINITIONS=FASTCD_SUM_MAX_KERNEL=2`: the AVX2 kernel;
  * `-DFASTCD_ENABLE_SUM_SIMD=OFF`: the portable kernel, which without
    `-mfma` rounds each product and sum separately.
* Matrix product: `cblas_dgemm` of Intel MKL 2026.2.1 (`libmkl_rt.so.3`
  from the `mkl` wheel), loaded by the runner, sequential threading layer,
  one thread. It was given to the AVX-512 runs only.
* All CTest tests passed on the default build, a Clang build with the
  address and undefined-behavior sanitizers, a build with the portable
  kernel and a shared-library build before timing.

## Files

| File | Content | Wall time (UTC) |
|---|---|---|
| `sums.json`, `sums.md` | sums protocol, AVX-512 kernel, 8,192 elements per cell in samples of 64, 1,024 and 4,096, dimensions 2 to 1024 | 14:23 to 14:30 |
| `sums-one-sample.json`, `sums-one-sample.md` | the same with one sample of 1,024 elements per cell | 14:30 to 14:32 |
| `sums-avx2.json`, `sums-avx2.md` | as `sums`, AVX2 kernel, no BLAS library | 14:32 to 14:37 |
| `sums-portable.json`, `sums-portable.md` | as `sums`, portable kernel, no BLAS library | 14:37 to 14:43 |
| `accuracy.json`, `accuracy.md` | accuracy study, full profile: 689 sums and 84 covariances, AVX-512 kernel, 16 worker processes | 14:43 to 14:48 |

```sh
python3 benchmarks/benchmark.py --protocol sums \
    --runner build/fastcd_sums_benchmark \
    --output benchmark-results/sums-t440-20261120/sums.json \
    --counts 64,1024,4096 --cpu 26 \
    --blas $VENV/lib/libmkl_rt.so.3 --blas-name "Intel MKL 2026.2.1"
python3 benchmarks/benchmark.py --protocol sums \
    --runner build/fastcd_sums_benchmark \
    --output benchmark-results/sums-t440-20261120/sums-one-sample.json \
    --trials 1024 --counts 1024 --cpu 26 \
    --blas $VENV/lib/libmkl_rt.so.3 --blas-name "Intel MKL 2026.2.1"
python3 benchmarks/benchmark.py --protocol sums \
    --runner build-avx2/fastcd_sums_benchmark \
    --output benchmark-results/sums-t440-20261120/sums-avx2.json \
    --counts 64,1024,4096 --cpu 26
python3 benchmarks/benchmark.py --protocol sums \
    --runner build-portable/fastcd_sums_benchmark \
    --output benchmark-results/sums-t440-20261120/sums-portable.json \
    --counts 64,1024,4096 --cpu 26
OPENBLAS_NUM_THREADS=1 python3 benchmarks/stability_sums.py \
    --library build-shared/libfastcd.so \
    --output benchmark-results/sums-t440-20261120/accuracy.json --workers 16
```

Every other option kept its default: 60 blocks, two warmup passes, a 20 ms
minimum observation and the benchmark driver's seed 20260906; the accuracy
study has its own seeds. The runs followed one another on logical CPU 26.
The one-minute load average of the host was 7.9 at the start, while the
builds finished, and between 2.3 and 4.0 at the end of each run.

## Notes

* Every benchmark run verified the first sample of each cell against a long
  double contraction of the cross moments and every sample for agreement
  between the tasks; all checks passed, and the fused layouts returned
  identical bits wherever they ran the same algorithm.
* `planes_direct` above dimension 16 is the loop that serves an explicit
  request for the direct algorithm, not a generated kernel.
* The `dgemm` timings depend on the BLAS library. No other library was
  measured.
* The host is shared, so small differences between cells, between the three
  kernel runs and between this archive and others do not isolate source
  changes.
