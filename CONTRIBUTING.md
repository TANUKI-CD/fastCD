# Contributing to fastCD

The canonical repository is
[git.tanuki-cd.com/algebraity/fastCD](https://git.tanuki-cd.com/algebraity/fastCD).
Please open issues and pull requests on [GitHub](https://github.com/TANUKI-CD/fastCD).

## Building and testing

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug \
    -DFASTCD_ENABLE_SANITIZERS=ON -DFASTCD_WARNINGS_AS_ERRORS=ON
cmake --build build --parallel
cmake -E chdir build ctest --output-on-failure

python -m venv .venv && . .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
python -m mypy
python -m ruff check python benchmarks tools
```

The library must build without warnings under GCC, Clang, and MSVC, and the
test suite must pass under AddressSanitizer and UndefinedBehaviorSanitizer.
Please also run the suite once with `-DFASTCD_TUNING_DEFINITIONS=FASTCD_DISABLE_SIMD=1`
when touching the batch kernels.

## Generated kernels

The straight-line kernels in `src/generated/` are produced by
`tools/generate_kernels.py` and are committed. Never edit them by hand; change
the generator, regenerate with `python3 tools/generate_kernels.py src/generated`,
and commit both. Generation is deterministic, and CI checks that the committed
files match.

## Arithmetic checks

For quasilinear arithmetic changes, run
`python3 tools/check_quasilinear_counts.py`; for Cariow–Cariowa arithmetic
changes, run `python3 tools/check_cariow_counts.py`. These optional
development checks require a C++ compiler (`c++` by default, or `CXX`) and
instrument the scalar templates and generated kernels through dimension
1024. The library itself remains C11. CTest also checks the complex
octonion leaf against the independent defining-recursion reference on all
196 real basis pairs and 256 dense dyadic cases.

For changes to floating-point accumulation, keep `tests/test_stability.c`
passing: it checks the blocked reductions and the bitwise agreement of
scalar, SIMD, single and batched products. If the order of additions
changes, rerun the stability study with `benchmarks/stability_study.py` (see
`docs/stability.md`).

## Algorithm defaults

The automatic selection tables live at the top of the algorithm-selection
section of `src/fastcd.c` and are pinned by tests. Change them only on the
basis of a new benchmark run (see `docs/benchmark.md`), and update the
descriptions of the defaults in `README.md`, `docs/c-api.md`, and
`docs/mathematics.md` in the same commit.

## Benchmark archives

Never edit the archived results under `benchmark-results/`, including their
checksum and preservation records.

`python3 tools/check_stability_archives.py` checks each stability study
against its `SHA256SUMS`, checks that the tables in `docs/stability.md` match
the newest study, and, when the git history is available, checks that the
measured commit contains exactly the recorded source files. Add `--full` to
check the raw files under `raw/` as well; they are not distributed, and this
mode requires NumPy. Both modes are read-only.

## Release checks

Update the canonical version in `src/fastcd.c`, the README heading, the C API
version description, and the version and release date in `CITATION.cff`.
Move the pending changelog entries into a dated release section. CMake and
Python package metadata read the canonical version automatically; historical
versions and hashes in the benchmark archives describe the measured builds.

Build and check the distributions in a development environment:

```sh
python -m build
python -m twine check --strict dist/*
```

The default build creates a source distribution and then builds the wheel from
it. CI also compares the bundled benchmark archives with the checkout,
installs that wheel, and runs the tests from the extracted source distribution
outside the checkout. This checks the distributed package and its actual
version metadata rather than a staged extension or an editable install.

## Pull requests

- Keep the C library free of heap allocation and global mutable state.
- Every public behavior change needs a test, a documentation update, and a
  line in `CHANGELOG.md`.
- Error messages shared by the C extension and the Python layer are defined
  in `python/src/pyfastcd/_validation.py` and mirrored in `_core.c`; keep them
  identical.

## Coefficient-field kernels

Real-coefficient arithmetic is in the sources directly under `src/` and
their generated kernels in `src/generated/`. Complex arithmetic is in
`src/complex.c`; finite-field arithmetic is in `src/finite/`. Use
`tools/finite/generate_fields.py` and
`tools/finite/generate_finite_backends.py` to regenerate the committed
finite kernels; the latter also regenerates the shared maps and value
schedule from its checked-in templates. Do not edit generated outputs by
hand.

The complex and finite C sources follow the conventions of the real
kernels: four-space block indentation, eight-space continuations, pointers
attached to their types, and a short purpose comment before each function.
Explain recursive stages, workspace ownership, aliasing exceptions, and
arithmetic bounds where they occur, as in `src/direct_impl.h` and
`src/quasilinear_impl.h`. The root `.clang-format` records these
conventions, and finite generation uses `clang-format==18.1.8` (included
in the development extras) so generated C stays readable and
deterministic. No formatter is needed to build the committed sources. Do
not reformat the real-coefficient sources or the archived benchmark
snapshots.

Complex and finite-field arithmetic use caller-owned contexts and scratch,
with no heap allocation or global mutable state in the C library. Every
finite buffer contains canonical `uint32_t` residues; the Python boundary
validates this on every call, since callers may have mutated an array.

Run the coefficient-field tests (`tests/test_domains.c` and
`python/tests/test_domains.py`) in the default build, with
`-DFASTCD_ENABLE_FINITE_SIMD=OFF`, and with
`-DFASTCD_TUNING_DEFINITIONS=FASTCD_DISABLE_SIMD=1`. CTest includes
independent polynomial and basis-sign oracles, exact workspace canaries,
composite operations, zero-norm checks, and a reduction test that avoids
128-bit integers. The Python tests include pickle fixtures from v1.1.0.
`python -m mypy python/tests/typing_domains.py` checks the inferred
coefficient and result types, and `tools/check_v110_compat.py` compares
the installed package against a separately installed v1.1.0 to confirm
backward compatibility: every result must match exactly, except that
floating-point values may differ by rounding and `"auto"` may select a
different algorithm.

`benchmarks/benchmark.py` runs the real-coefficient experiment by default.
`--protocol domains` measures direct and quasilinear over C and finite
fields, `--protocol flint` compares scalar arithmetic and complete CD
products against FLINT (Fast Library for Number Theory), an independent
exact-arithmetic C library, and `--protocol all` runs all three. The
[benchmark guide](docs/benchmark.md) describes the timing, field matching,
verification and reporting rules.
