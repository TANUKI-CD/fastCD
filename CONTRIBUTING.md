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

For quasilinear arithmetic changes, run `python3 tools/check_quasilinear_counts.py`.
This optional development check requires a C++ compiler (`c++` by default,
or `CXX`) and instruments the scalar template and generated kernels through
dimension 1024. The library itself remains C11. CTest also checks the complex
octonion leaf against the independent defining-recursion reference on all
196 real basis pairs and 256 dense dyadic cases.

## Algorithm defaults

The automatic selection tables live at the top of the algorithm-selection
section of `src/fastcd.c` and are pinned by tests. Change them only on the
basis of a new benchmark run (see `docs/benchmark.md`), and update the tables
in `README.md`, `docs/c-api.md`, and `docs/mathematics.md` in the same commit.

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
