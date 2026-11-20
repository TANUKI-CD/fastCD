# Fused sums: benchmark and accuracy, 2026-11-19 on the T440

This directory holds the measurements of the fused sums of products that
[`docs/benchmark.md`](../../docs/benchmark.md#fused-sums) and
[`docs/stability.md`](../../docs/stability.md#fused-sums) report.

## Host and build

* Dell PowerEdge T440, Intel Xeon Gold 6148 (80 logical CPUs, two sockets,
  two SMT threads per core), Linux 7.2.11, shared host; no CPU governor,
  turbo or SMT settings were changed.
* Source: a working tree with uncommitted changes on commit `0a84b06`; the
  fused sums were not yet committed. Each benchmark JSON file records that
  state and the SHA-256 of every measured source file, and the accuracy
  JSON the SHA-256 of the shared library it loaded.
* Build: CMake Release (`-O3 -DNDEBUG`, `-fno-fast-math -ffp-contract=off`),
  GCC 16.2.1, with the AVX-512 kernel of the fused sums selected at run
  time. The runner, but not the library, was built with
  `-Wa,-mbranches-within-32B-boundaries`. All CTest tests passed before
  timing.
* Matrix product: `cblas_dgemm` of Intel MKL 2026.2.1 (`libmkl_rt.so.3`
  from the `mkl` wheel), loaded by the runner, sequential threading layer,
  one thread.

## Files

| File | Content | Wall time |
|---|---|---|
| `sums.json`, `sums.md` | sums protocol, 8,192 elements per cell in samples of 64, 1,024 and 4,096, dimensions 2 to 1024, logical CPU 26 | 21:51 to 21:58 |
| `sums-one-sample.json`, `sums-one-sample.md` | the same with one sample of 1,024 elements per cell | 21:58 to 22:00 |
| `accuracy.json`, `accuracy.md` | accuracy study, full profile: 689 sums and 84 covariances, 16 worker processes | 22:00 to 22:05 |

```sh
python3 benchmarks/benchmark.py --protocol sums \
    --runner build/fastcd_sums_benchmark \
    --output benchmark-results/sums-t440-20261119/sums.json \
    --counts 64,1024,4096 --cpu 26 \
    --blas $VENV/lib/libmkl_rt.so.3 --blas-name "Intel MKL 2026.2.1"
python3 benchmarks/benchmark.py --protocol sums \
    --runner build/fastcd_sums_benchmark \
    --output benchmark-results/sums-t440-20261119/sums-one-sample.json \
    --trials 1024 --counts 1024 --cpu 26 \
    --blas $VENV/lib/libmkl_rt.so.3 --blas-name "Intel MKL 2026.2.1"
OPENBLAS_NUM_THREADS=1 python3 benchmarks/stability_sums.py \
    --library build-shared/libfastcd.so \
    --output benchmark-results/sums-t440-20261119/accuracy.json --workers 16
```

Every other option kept its default: 60 blocks, two warmup passes, a 20 ms
minimum observation and the seed 20261119. The two benchmark runs ran one
after the other, and the accuracy study after them. The one-minute load
average of the host was 5.0 when the benchmark started and 2.6 when it
ended.

## Notes

* Both benchmark runs verified the first sample of every cell against a long
  double contraction of the cross moments and every sample for agreement
  between the tasks; all checks passed, and the fused layouts returned
  identical bits wherever they ran the same algorithm.
* `planes_direct` above dimension 16 is the loop that serves an explicit
  request for the direct algorithm, not a generated kernel.
* The timings depend on the BLAS library. No other library was measured.
* The host is shared, so small differences between cells and between this
  archive and others do not isolate source changes.
