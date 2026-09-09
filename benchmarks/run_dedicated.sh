#!/usr/bin/env bash
# Build fastCD from this checkout, run the C tests, and record one complete
# benchmark run on a dedicated machine.  Results go to benchmark-results/
# under a timestamped name, so nothing is overwritten; build directories
# and logs go to build-bench-*/ (ignored by git).
#
#   benchmarks/run_dedicated.sh [--cpu N] [--label TEXT] [--cc-doubling]
#                               [--blocks B] [--modes batch,single]
#                               [--dimensions 2,4,...] [--max-cc-dimension N]
#
# --cpu          logical CPU for the driver and runner (default: first allowed)
# --label        suffix for the output names (default: git short hash)
# --cc-doubling  also build and time the FASTCD_CC_DOUBLING=1 configuration
#                into a second result file for comparison
# --blocks       observation blocks per dimension (default 24; keep a
#                multiple of 12 for a balanced task order)
# --modes        subset of batch,single (default both)
# --dimensions   powers of two to time (default: the driver's 2..1024)
# --max-cc-dimension  largest dimension at which Cariow–Cariowa is timed
#                (default: every supported dimension, 1024)
#
# Run from any directory; paths are resolved relative to the repository.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cpu=""
label=""
cc_doubling=0
blocks=24
modes="batch,single"
dimensions=""
max_cc=""
while [ $# -gt 0 ]; do
    case "$1" in
        --cpu) cpu="$2"; shift 2 ;;
        --label) label="$2"; shift 2 ;;
        --cc-doubling) cc_doubling=1; shift ;;
        --blocks) blocks="$2"; shift 2 ;;
        --modes) modes="$2"; shift 2 ;;
        --dimensions) dimensions="$2"; shift 2 ;;
        --max-cc-dimension) max_cc="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
if [ -z "$label" ]; then
    label=$(git -C "$root" rev-parse --short HEAD 2>/dev/null || echo "nogit")
fi
stamp=$(date +%Y%m%d-%H%M%S)
results="$root/benchmark-results"
mkdir -p "$results"
driver_options=(--blocks "$blocks" --modes "$modes")
if [ -n "$cpu" ]; then driver_options+=(--cpu "$cpu"); fi
if [ -n "$dimensions" ]; then driver_options+=(--dimensions "$dimensions"); fi
if [ -n "$max_cc" ]; then driver_options+=(--max-cc-dimension "$max_cc"); fi

# One configuration: fresh Release build, tests, then the full benchmark.
run_configuration() {
    local name="$1"; shift
    local build="$root/build-bench-$name"
    local output="$results/results-$stamp-$label-$name.json"

    echo "== $name: configuring $build" >&2
    rm -rf "$build"
    mkdir -p "$build"
    cmake -S "$root" -B "$build" -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTING=ON \
        -DFASTCD_WARNINGS_AS_ERRORS=ON "$@" > "$build/configure.log" 2>&1 \
        || { cat "$build/configure.log" >&2; exit 1; }
    cmake --build "$build" --parallel > "$build/build.log" 2>&1 \
        || { tail -50 "$build/build.log" >&2; exit 1; }
    echo "== $name: running the C test suite" >&2
    ctest --test-dir "$build" --output-on-failure --timeout 1800 \
        > "$build/ctest.log" 2>&1 || { tail -30 "$build/ctest.log" >&2; exit 1; }
    echo "== $name: benchmarking into $output" >&2
    python3 "$root/benchmarks/benchmark.py" --runner "$build/fastcd_benchmark" \
        --output "$output" "${driver_options[@]}"
    echo "== $name: wrote $output and ${output%.json}.md" >&2
}

if ! git -C "$root" diff --quiet 2>/dev/null; then
    echo "note: the working tree has uncommitted changes; the JSON records" \
         "source hashes, not a commit" >&2
fi
run_configuration default
if [ "$cc_doubling" = 1 ]; then
    run_configuration cc-doubling \
        -DFASTCD_TUNING_DEFINITIONS=FASTCD_CC_DOUBLING=1
fi
