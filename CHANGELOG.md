# Changelog

All notable changes to fastCD are recorded in this file. The project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html); the format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The C library and the
`pyfastcd` package share one version number, defined in `src/fastcd.c`.

## [Unreleased]

### Added

- Drivers for an exact-reference accuracy study and paired timings
  (`benchmarks/stability*.py`), run end to end by
  `benchmarks/stability_study.py`; the C regression test
  `tests/test_stability.c`; and the study in
  `benchmark-results/stability-20261107/`, which `docs/stability.md`
  summarizes.
- `tools/check_stability_archives.py`, a read-only check of the published
  study files against their recorded hashes, of the tables in
  `docs/stability.md` against the study, and of the measured commit's source
  against the recorded source. With `--full` it also checks the raw data,
  which is not distributed.

### Changed

- Floating-point sums whose length grows with the dimension are accumulated
  in blocks of eight with a balanced tree between blocks: in the quasilinear
  kernels, in the complex-coefficient pairing shared by quadratic norms,
  squares, anticommutators and reciprocals, and in the Cariow–Cariowa
  correction sums above dimension 256. From dimension 16 upward,
  quasilinear products and the complex-coefficient operations that use the
  pairing can round differently from 1.2.0, as can Cariow–Cariowa results at
  dimensions 512 and 1024. Direct and finite-field results, the public API,
  workspace sizes and the published arithmetic counts are unchanged.
- Over C and finite fields, AUTO selects direct multiplication up to
  dimension 4. Finite-field direct multiplication uses the packed field
  arithmetic of the quasilinear kernels from dimension 16 on.
- For real coefficients, AUTO selects direct multiplication for single
  products at dimension 4; batches at dimension 4 still use quasilinear.
- The finite-field and FLINT benchmark runners time each block at its own
  stack offset, spread over a 4 KB page, so that the observations of a cell
  do not share one stack placement.
- The benchmark runners are built with branch padding where the compiler
  supports it, so that timing loops do not depend on 32-byte code placement
  on Skylake-family processors. The library is built as before.

## [1.2.0] - 2026-11-04

### Added

- `FiniteField(char=..., deg=...)`: immutable, cached finite fields for use
  as coefficients, with deterministic irreducible polynomial selection,
  and `FiniteField.from_modulus` for an explicit polynomial. Field scalars
  and batches use the `Element` and `ElementArray` classes with exact
  polynomial storage.
- `char`, `deg`, and `field` arguments on `Algebra` and `cd_element` select
  complex or odd-characteristic finite-field coefficients. Calls without
  them keep their real-coefficient behavior.
- Finite-field quasilinear CD multiplication with fixed-degree arithmetic
  and runtime-dispatched x86 kernels, an independent direct recursion,
  structural operations, field scalars, broadcasting, output buffers and
  buffer/NumPy/DLPack support.
- Complex CD arithmetic through the complex alternating recursion, alongside
  a direct implementation of the defining recursion.
- `quadratic_norm()` on all elements and arrays, returning the canonical
  coefficient-field quadratic form; `norm_squared()` returns the same form.
- C headers `<fastcd/arithmetic.h>`, `<fastcd/complex.h>` and
  `<fastcd/finite.h>` for finite and complex contexts, workspace queries,
  record operations, products, batches and quadratic norms. The real C ABI
  is unchanged.
- Benchmark protocols for complex and finite fields (`--protocol domains`)
  and for comparison with FLINT (Fast Library for Number Theory)
  (`--protocol flint`), with full-corpus verification, paired confidence
  intervals and reproducible corpus and build manifests. Only the FLINT
  benchmark executable links FLINT.
- C and Python tests for the new coefficient fields, exact workspace
  boundary checks, a portable reduction test, v1.1.0 pickle fixtures,
  `tools/check_v110_compat.py`, and type stubs that infer coefficient types.
- T440 benchmark data for the 1.2.0 source: the full real-coefficient
  protocol through dimension 1024, direct versus quasilinear over the
  default finite fields and C, and the FLINT comparison. The README tables
  use this run.

## [1.1.0] - 2026-10-26

### Changed

- Automatic single-product selection at dimension 4 now uses quasilinear
  instead of direct, including one-record batches and composite operations.
  This follows the lower measured mean in the earlier 2026-10-23 T440 run;
  the paired 95% interval still includes equal performance. AUTO results
  can round differently from the previous direct default.
- Quasilinear multiplication uses a fused nonrecursive bilinear map and a
  specialized complex octonion alternating-product base case; arithmetic
  counts and validation instructions are documented in
  `docs/mathematics.md`. Its floating-point rounding can differ from version
  1.0.0.
- The complex octonion base case of the quasilinear recursion uses 72
  products, 156 additions/subtractions and 15 halvings, 243 operations in
  total, so a full dimension-16 real product uses 417 operations. The
  arithmetic-count check covers dimensions through 1024, and the
  mathematics documentation gives the addition-count argument for the
  fixed nonrecursive-map decomposition.
- The C and Python stability documentation clarifies that AUTO choices,
  workspace lengths, and arithmetic schedules can change between compatible
  releases; bitwise floating-point reproducibility across versions is not
  guaranteed.

### Added

- Full T440 benchmark data through dimension 1024 with the paper's input
  corpora and protocol. The run passes all 600,000 verification checks and
  supports the current AUTO choices. Source distributions include the
  benchmark archives linked by the documentation.
- CI checks for quasilinear arithmetic counts and the complete source
  distribution to wheel to installed-package test path, including byte-for-byte
  verification of the bundled benchmark archives.

### Fixed

- Documentation matches scalar-comparison exceptions, NumPy return types,
  zero-copy layout rules, and the current arithmetic schedules. The
  nonrecursive-map addition-count argument is given in full, and the
  mathematics documentation follows the notation of the paper's v2.
- NumPy boolean scalars are rejected as dimensions, levels, and basis indices,
  matching Python booleans consistently across supported NumPy versions.
- README image and file links use absolute URLs on `git.tanuki-cd.com`, so the
  package description links to the canonical repository when rendered on PyPI.
- Scalar equality preserves the exact values of integers, decimals, fractions,
  and NumPy scalars instead of rounding comparison operands to binary64.
  Equal hashable scalars and `Element` values now remain interchangeable in
  dictionaries and sets. `ElementArray` uses the same comparison rules;
  supplying `__float__` alone does not imply numeric equality.
- Cariow–Cariowa batches return immediately when SIMD pairing processes every
  record, avoiding out-of-range tail pointers for minimally allocated padded
  batches. Multiplication arithmetic and the archived paper data are unchanged.
- The kernel generator's fixed-length `zip` traversals explicitly use
  `strict=True`, satisfying the configured lint checks without changing the
  generated kernels.
- CMake reconfigures when the canonical version changes, including builds
  with the benchmark disabled, keeping installed package metadata in sync.

## [1.0.0] - 2026-09-13

The first public release.

### Added

- C library `libfastcd`: direct, Cariow–Cariowa, and quasilinear
  Cayley–Dickson multiplication with benchmark-derived automatic selection,
  strided batch products with two-lane SIMD pairing, coordinate and structural
  operations, cross products, commutators, associators, indexed basis
  operations, and matrix-free operator commutators.
- Python package `pyfastcd`: `Algebra`, `Element`, `ElementArray`,
  `BasisProduct`, and `cd_element`, with NumPy ufunc, buffer, and DLPack
  interoperability and complete type stubs.
- Independent reference oracles for the C and Python test suites, and a
  reproducible benchmark driver that records source, executable, and corpus
  hashes with its results.
- Single quasilinear products above dimension 16 evaluate the two recursive
  calls of the top complex-kernel call in the two SIMD lanes and continue
  through the paired kernel, bitwise identical for finite results to the
  scalar and batched forms; the quasilinear workspace query grows
  accordingly. The `FASTCD_CC_DOUBLING=1` tuning option evaluates
  Cariow–Cariowa products through one doubling step with paired
  half-dimensional kernels, and `benchmarks/run_dedicated.sh` scripts a
  complete build, test, and benchmark run.
- Cariow–Cariowa multiplication is available through dimension 1024 via a
  table-driven kernel with the same operations as the generated kernels; it
  needs `2N` workspace doubles above dimension 64 and is not recommended
  there. The benchmark times it at every supported dimension by default;
  `--max-cc-dimension` restricts the timed range, and dimensions above the
  cap are marked "not timed" as opposed to "unavailable".
