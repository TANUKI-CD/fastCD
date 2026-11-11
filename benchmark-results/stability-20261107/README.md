# Stability study, 2026-11-07

This directory holds the measurements behind [docs/stability.md](../../docs/stability.md), which defines the error measures and interprets the results.

## What was compared

- **Sequential accumulation:** commit `982b9daed40e61e59fd939dae61a0e358a35191a`, the baseline.
- **Balanced accumulation:** commit `558e609b499f5a46bd102c370e64d445031b85a8`, the measured commit.

Errors are measured against the exact product. Both commits contain the same direct multiplication, which is measured alongside for comparison, and the accuracy study checks that the two builds give bit-for-bit identical direct products on every stored operand pair. Direct timings are therefore the control for noise between the two builds. The study also compares the Cariow–Cariowa correction sums of the two commits and times the balanced-accumulation block sizes 4, 8 and 16 at the measured commit.

## Host and software

| | |
|:---|:---|
| CPU | Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz |
| Operating system | Linux-7.2.11-arch1-1-x86_64-with-glibc2.44 |
| Compiler | cc (GCC) 16.2.1 20260810 (`/usr/bin/cc`) |
| Library flags | `-O3 -DNDEBUG -fno-fast-math -ffp-contract=off -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wstrict-prototypes -Werror` |
| Python | 3.14.8 |
| NumPy, gmpy2, MPFR | 2.5.4, 2.3.2, MPFR 4.2.2 |
| Matplotlib | 3.11.2 |
| Logical CPU for the timings | 18 |
| Logical CPU for the accuracy study | 20 |

## How it was produced

`benchmarks/stability_study.py` built both commits from clean detached worktrees and ran every step below with the scripts of the measured commit.

```sh
python3 benchmarks/stability_study.py --baseline-commit 982b9da --commit 558e609 --output benchmark-results/stability-20261107 --cpu 18 --accuracy-cpu 20 --jobs 16 --document docs/stability.md
```

| Step | What it does | Duration |
|:---|:---|---:|
| build-baseline-simd | Build the baseline library. | 15 s |
| build-baseline-scalar | Build the baseline library without the explicit SIMD kernels. | 12 s |
| build-balanced-simd | Build the measured library. | 16 s |
| build-balanced-scalar | Build the measured library without the explicit SIMD kernels. | 12 s |
| build-block-4 | Build the measured library with block size 4. | 16 s |
| build-block-16 | Build the measured library with block size 16. | 16 s |
| accuracy | Measure every error against the exact product. | 5 min 04 s |
| accuracy-tables | Tabulate the accuracy results. | 1 s |
| ties | Compare exactly the cases whose stored errors are equal. | 32 s |
| agreement | Compare scalar, SIMD, single and batched products bit for bit. | 5 s |
| cc-accuracy | Compare the Cariow–Cariowa errors of the two builds. | 1 min 13 s |
| speed | Time sequential and balanced accumulation. | 1 min 54 s |
| speed-tables | Tabulate the timings. | 0 s |
| cc-speed | Time the Cariow–Cariowa products of the two builds. | 2 min 30 s |
| block-size | Time block sizes 4, 8 and 16. | 10 s |
| plots | Measure and draw the six charts. | 2 min 22 s |

## Published files

These files are part of the repository and the source distributions.

| File | Contents |
|:---|:---|
| `README.md` | This description, generated from the results. |
| `SHA256SUMS` | SHA-256 of every file in the study, including the raw files. |
| `study.json` | The two commits, the command, the host and software, the source hashes of the measured commit, and every step with its duration. |
| `builds.json` | Configure and build commands, build records, and the SHA-256 of every library and benchmark runner. |
| `key-figures.json` | The figures quoted in docs/stability.md, computed from the results. |
| `accuracy-tables.md` | The accuracy tables shown in docs/stability.md. |
| `accuracy-summary.csv` | Error statistics for every domain, dimension, family and algorithm. |
| `ties.json` | Exact comparison of the cases whose sequential and balanced errors round to the same binary64 value. |
| `agreement.json` | Bitwise comparison of scalar, SIMD, single and batched products on every stored operand pair. |
| `cc-accuracy-summary.json` | Cariow–Cariowa errors of the two builds: counts of improved, worsened and tied cases, and statistics by dimension and family. |
| `speed-tables.md` | The timing table shown in docs/stability.md. |
| `speed-summary.csv` | Medians and paired ratios of the sequential and balanced timings. |
| `cc-speed-tables.md` | Cariow–Cariowa timings of the two builds, with the direct control. |
| `cc-speed-summary.csv` | Medians and paired ratios of the Cariow–Cariowa, quasilinear and direct timings of the two builds. |
| `block-size-tables.md` | Quasilinear timings for balanced-accumulation block sizes 4, 8 and 16. |
| `block-size-summary.csv` | Medians and paired ratios of the block-size timings. |
| `PLOTS.md` | Description of the six charts and the command that drew them. |
| `plot-data.json` | Every plotted value, with sample keys, counts and bootstrap intervals. |
| `plot-data.csv` | Every plotted value in tabular form. |
| `relative-error-*.{svg,pdf,png}`, `speed-*.{svg,pdf,png}` | The six charts described in PLOTS.md, in three formats. |

## Raw files

Files under `raw/` stay on the machine that ran the study. They are not included in Git or in the source distributions, but SHA256SUMS lists them.

| Path | Contents |
|:---|:---|
| `raw/accuracy.json` | Every case of the accuracy study: operand hashes, seeds, and the error measures of quasilinear multiplication with sequential and with balanced accumulation, direct multiplication and Cariow–Cariowa multiplication. |
| `raw/accuracy-corpus.npz` | The stored operands of the accuracy study. |
| `raw/cc-accuracy.json` | Every compared Cariow–Cariowa case. |
| `raw/speed.json` | Every native timing observation of the sequential and balanced builds. |
| `raw/speed-corpora/` | Operands of the sequential and balanced timings. |
| `raw/cc-speed.json` | Every native timing observation of the Cariow–Cariowa comparison. |
| `raw/cc-speed-corpora/` | Operands of the Cariow–Cariowa timings. |
| `raw/block-size.json` | Every native timing observation of the block-size comparison. |
| `raw/block-size-corpora/` | Operands of the block-size timings. |
| `raw/plot-timings.json` | Every native timing observation behind the speed charts. |
| `raw/plot-corpora/` | Operands of the chart timings. |
| `raw/logs/` | The study log and the output of every step and build. |

## Verification

From the repository root:

```sh
python3 tools/check_stability_archives.py
```

The checker compares every published file with SHA256SUMS, checks that the tables in docs/stability.md are the tables in this directory, and, when the git history is available, checks that the measured commit contains exactly the source files recorded in study.json. With `--full` it also requires the raw files, checks their hashes and regenerates the tables from the raw results.
