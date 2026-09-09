# Changelog

All notable changes to fastCD are recorded in this file. The project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html); the format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The C library and the
`pyfastcd` package share one version number, defined in `src/fastcd.c`.

## [1.0.0rc1] - 2026-09-08

Release candidate for the initial public release.

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
  of every complex node in the two SIMD lanes, bitwise identical to the
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
