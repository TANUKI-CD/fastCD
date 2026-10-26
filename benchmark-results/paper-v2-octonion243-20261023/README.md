# Full T440 comparison with the 243-operation octonion leaf

This run repeats the original paper protocol on 2026-10-23 after replacing
the 244-operation complex octonion alternating-product leaf with the
243-operation schedule. The previous full v2 run remains in
[`../paper-v2-20261023/`](../paper-v2-20261023/results.md), and the original
paper run remains in [`../paper-results.json`](../paper-results.json).

All 600,000 algorithm/mode verification checks passed, with finite outputs
and zero mismatches. The paired comparisons support retaining every current
AUTO choice. Quasilinear has the lowest mean at every tested dimension from
32 onward in both modes; at dimension 4 it remains tied with direct.

The measured source is commit `7da5a5f46c08ef41dfe98aaafa13be4643c3b647`
plus [source.patch](source.patch). The working tree was dirty during
measurement, so the commit alone does not identify the timed code. The patch,
its SHA-256 in [environment.json](environment.json), and the individual
source hashes in [fastcd-build-Release.txt](fastcd-build-Release.txt) identify
the measured snapshot. The source patch includes the generator, generated
quasilinear header, arithmetic-count checker, and a source comment update.
It excludes unrelated working-tree changes and documentation.

The arithmetic convention counts input-dependent products (`M`), real
additions/subtractions (`A`), and fixed-scalar multiplications other than
`0, ±1` (`S`). The new complex leaf uses `(72, 156, 15)`, totaling 243,
versus the previous `(72, 158, 14)`, totaling 244. A full real product at
dimension `N >= 16` saves `N/16` operations. See
[operation-counts.json](operation-counts.json),
[operation-count-comparison.json](operation-count-comparison.json), and the
[mathematical description](../../docs/mathematics.md) for the instrumented
counts and the restricted addition-count argument for the fixed local-map
decomposition.

## Protocol and artifacts

The machine is a Dell PowerEdge T440 with an Intel Xeon Gold 6148. The
Release build uses GCC 16.2.1, `-O3 -DNDEBUG -fno-fast-math -ffp-contract=off`,
warnings as errors, default leaf cutoffs, and the default SIMD kernels.
The benchmark is pinned to logical CPU 0. This is a shared host; no
governor, turbo, SMT, or service settings were changed.

All three methods are measured in single and batch modes at dimensions
`2,4,8,16,32,64,128,256,512,1024`, with 10,000 operand pairs per dimension,
24 subprocess blocks, two warmup passes, and a 20 ms calibration target.
The norm range is `[0.5,2)`, with seed `20260906`. All corpus hashes match
the original paper. Every pair is checked for each method and mode against
the independent binary64 defining-recursion reference after that task's
timing in the first block, outside the timed region. The 600,000 checks
reuse 100,000 distinct pairs;
they are comparisons with a floating-point reference, not exact products.

- [results.json](results.json) contains raw observations and paired ratios;
  [results.md](results.md) is the complete generated report.
- [auto-assessment.json](auto-assessment.json) compares each alternative
  with the measured AUTO choice using paired per-block ratios.
- [comparison-previous-leaf.json](comparison-previous-leaf.json) compares
  means with the preceding 244-operation-leaf run;
  [comparison.json](comparison.json) compares with the original paper run.
- [validation.json](validation.json) records C tests, the installed-wheel
  Python suite, sanitizers, arithmetic counts, and benchmark integrity checks.
  The accompanying logs preserve the CTest output.
- [environment.json](environment.json), [commands.json](commands.json),
  [corpus-manifest.json](corpus-manifest.json), and the build logs record
  the environment, commands, inputs, and build.

Cross-version comparisons are descriptive ratios of means from separate
runs. They are not paired cross-version confidence intervals. The two v2
runs used the same compiler and kernel versions, but host load and CPU
frequency can vary. The original paper additionally used different compiler
and kernel versions. Changes in elapsed time do not isolate the leaf change;
the unchanged direct and Cariow–Cariowa kernels provide controls.

## Reproduction

For the exact measured source, use a separate clean checkout of the base
commit and apply `source.patch` there. The patch is already applied in the
updated source tree. With NumPy installed, run:

```sh
python3 tools/check_quasilinear_counts.py --output /tmp/fastcd-counts.json
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_TESTING=ON -DFASTCD_BUILD_BENCHMARK=ON \
    -DFASTCD_BUILD_PYTHON=OFF -DFASTCD_WARNINGS_AS_ERRORS=ON
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
python3 benchmarks/benchmark.py --runner build/fastcd_benchmark \
    --dimensions 2,4,8,16,32,64,128,256,512,1024 \
    --trials 10000 --blocks 24 --warmup 2 --min-time-ms 20 \
    --seed 20260906 --norm-min 0.5 --norm-max 2.0 \
    --verify-trials 10000 --max-direct-dimension 1024 \
    --max-cc-dimension 1024 --modes batch,single --cpu 0 \
    --corpora benchmark-corpora \
    --output benchmark-results/results-octonion243-reproduction.json
```

Use a new output filename for each run. The corpus directory can contain
the original files or be regenerated with the documented seed; compare its
SHA-256 hashes with `corpus-manifest.json` before comparing results.
`commands.json` preserves the original absolute paths used for this run.
