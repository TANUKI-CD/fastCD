# Changelog

All notable changes to fastCD are recorded in this file. The project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html); the format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The C library and the
`pyfastcd` package share one version number, defined in `src/fastcd.c`.

## [Unreleased]

## [1.1.0] - 2026-10-26

### Changed

- Automatic single-product selection at dimension 4 now uses quasilinear
  instead of direct, including one-record batches and composite operations.
  This follows the lower measured mean in the earlier 2026-10-23 T440 run; the paired
  95% interval still includes equal performance. AUTO results can round
  differently from the previous direct default.
- Quasilinear multiplication uses a fused local bilinear map and a specialized
  complex octonion alternating-product base case; arithmetic counts and
  validation instructions are documented in `docs/mathematics.md`. Its
  floating-point rounding can differ from version 1.0.0.
- Joint reconstruction of the complex octonion leaf's `h` coordinate reduces
  its count to 72 products, 156 additions/subtractions and 15 halvings:
  243 operations, saving one per leaf. A full dimension-16 real quasilinear
  product now uses 417 operations, and the arithmetic-count check covers dimensions
  through 1024. The mathematics documentation gives the constrained
  addition-count argument for the fixed local-map decomposition.
- The C and Python stability documentation clarifies that AUTO choices,
  workspace lengths, and arithmetic schedules can change between compatible
  releases; bitwise floating-point reproducibility across versions is not
  guaranteed.

### Added

- Full T440 benchmark data through dimension 1024 for both the 244-operation
  and revised 243-operation leaf schedules, preserving the original input
  corpora and protocol. The revised-leaf run passes all 600,000 verification
  checks and supports retaining AUTO. Source distributions include the
  benchmark archives linked by the documentation.
- CI checks for quasilinear arithmetic counts and the complete source
  distribution to wheel to installed-package test path, including byte-for-byte
  verification of the bundled benchmark archives.

### Fixed

- Documentation matches scalar-comparison exceptions, NumPy return types,
  zero-copy layout rules, and the current arithmetic schedules. The local-map
  addition-count argument is given in full, and the mathematics documentation
  follows the notation of the paper's v2.
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
- Single quasilinear products above dimension 16 evaluate the two children
  of the top complex node in the two SIMD lanes and continue through the
  paired kernel, bitwise identical for finite results to the
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
