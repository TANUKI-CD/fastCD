#!/usr/bin/env python3
"""Compare real public-call results against an installed v1.1.0 baseline.

Build/install the old tag into a separate --target directory first. The
check runs both packages in isolated subprocesses using the same interpreter,
covering constructors, buffers, algorithms, scalar protocols, arrays,
structural operations, exceptions and pickle reconstruction. It covers the
calls that v1.1.0 accepts; the domain tests exercise the field selectors.

The comparison follows the compatibility rules of docs/python-api.md. Types,
parents, dtypes, shapes, strides, writeability, exceptions, hashes and pickles
must match exactly. Binary64 values may differ by rounding, and the algorithm
that "auto" selects may differ; both kinds of difference are counted and
reported without failing the check.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

PROBE = r"""
import base64, hashlib, json, pickle
import numpy as np
import pyfastcd as fcd

rng = np.random.default_rng(110)
results = []


def emit(value):
    if isinstance(value, (fcd.Element, fcd.ElementArray)):
        a = np.asarray(value)
        results.append(
            [
                type(value).__name__,
                repr(value.algebra),
                a.dtype.str,
                a.shape,
                a.strides,
                a.flags.writeable,
                a.tobytes().hex(),
            ]
        )
    elif isinstance(value, np.ndarray):
        results.append([value.dtype.str, value.shape, value.tobytes().hex()])
    elif isinstance(value, (float, np.floating)):
        results.append(["float", float(value).hex()])
    else:
        results.append(repr(value))


for level in range(11):
    A = fcd.Algebra(level=level)
    n = A.dimension
    x = A(rng.standard_normal(n))
    y = A(rng.standard_normal(n))
    z = A(rng.standard_normal(n))
    emit(A)
    emit(A is fcd.Algebra(dimension=n))
    results.append(["algorithm", A.default_algorithm])
    results.append(["algorithm", A.default_batch_algorithm])
    for v in (
        A(),
        A.zero(),
        A.one(),
        A.scalar(-0.0),
        A.basis(n - 1),
        +x,
        -x,
        x + y,
        x - y,
        x * y,
        x * 3,
        3 * x,
        x / 3,
        x**3,
        x.conjugate(),
        x.imag_part(),
        x.square(),
        x.imag_cross(y),
        x.commutator(y),
        x.anticommutator(y),
        x.associator(y, z),
        x.reciprocal(),
        x.normalized(),
        x.inner(y),
        x.norm_squared(),
        x.norm(),
        abs(x),
        A.left_basis_multiply(n - 1, x),
        A.right_basis_multiply(x, n - 1),
        A.left_operator_commutator(x, y, z),
        A.right_operator_commutator(x, y, z),
    ):
        emit(v)
    for alg in ("direct", "quasilinear", "cariow_cariowa", "auto"):
        emit(A.multiply(x, y, algorithm=alg))
    emit(memoryview(x).format)
    emit(memoryview(x).shape)
    emit(tuple(x))
    emit(hash(A.one()))
    emit(x == x.copy())
    a = A.array(rng.standard_normal((3, 2, n)))
    b = A.array(rng.standard_normal((2, n)))
    for v in (
        a,
        a[::-1],
        a[1],
        a + b,
        a - b,
        a * b,
        a * 3,
        3 * a,
        a / 3,
        a**3,
        a.conjugate(),
        a.imag_part(),
        a.real,
        a.inner(b),
        a.norm_squared(),
        a.norm(),
        a.square(),
        a.imag_cross(b),
        a.commutator(b),
        a.anticommutator(b),
        a.associator(b, x),
        a.reciprocal(),
        a.normalized(),
        np.asarray(a),
        np.multiply(a, b),
        a == b,
    ):
        emit(v)
    target = A.array(np.zeros(a.shape))
    emit(A.multiply(a, b, out=target))
    emit(target)
    for ctor, value in ((A, np.zeros(n + 1)), (A, 1j), (A.array, np.zeros((2, n + 1)))):
        try:
            ctor(value)
        except Exception as e:
            emit((type(e).__name__, str(e)))
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        for v in (A, x, a):
            emit(pickle.loads(pickle.dumps(v, protocol=protocol)))
print(json.dumps(results))
"""


ALGORITHMS = ("direct", "cariow_cariowa", "quasilinear")

# Largest accepted difference between two binary64 results, relative to one
# plus the largest magnitude in either result
ROUNDING_TOLERANCE = 1e-12


def probe(path: Path) -> list[object]:
    env = dict(os.environ, PYTHONPATH=str(path.resolve()))
    completed = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd="/tmp",
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(completed.stdout)


def binary64_values(result: object) -> tuple[list[object], tuple[float, ...]] | None:
    """Split a binary64 probe result into its exact part and its values.

    Elements, element arrays and ndarrays end with their coefficient bytes in
    hexadecimal, and a float is tagged ``"float"``. Any other result returns
    None and is compared exactly.
    """

    if not isinstance(result, list) or not result:
        return None
    if result[0] == "float" and len(result) == 2:
        return ["float"], (float.fromhex(result[1]),)
    dtype = result[2] if len(result) == 7 else result[0]
    if len(result) not in (3, 7) or dtype not in ("<f8", ">f8"):
        return None
    data = bytes.fromhex(result[-1])
    return result[:-1], struct.unpack(f"{dtype[0]}{len(data) // 8}d", data)


def rounding_difference(before: object, after: object) -> float | None:
    """Return the scaled difference of two binary64 results with equal layout.

    The result is None when the two results are not binary64 values of the
    same kind, parent, shape and layout.
    """

    old, new = binary64_values(before), binary64_values(after)
    if old is None or new is None or old[0] != new[0] or len(old[1]) != len(new[1]):
        return None
    scale = 1.0 + max((abs(value) for value in old[1] + new[1]), default=0.0)
    gap = max((abs(a - b) for a, b in zip(old[1], new[1], strict=True)), default=0.0)
    return gap / scale


def is_algorithm(result: object) -> bool:
    """Return whether a probe result names an automatically selected algorithm."""

    return (
        isinstance(result, list)
        and len(result) == 2
        and result[0] == "algorithm"
        and result[1] in ALGORITHMS
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--baseline-path", type=Path, required=True)
    parser.add_argument("--current-path", type=Path, required=True)
    args = parser.parse_args()
    before, after = probe(args.baseline_path), probe(args.current_path)
    if len(before) != len(after):
        raise SystemExit(f"different result counts: {len(before)} vs {len(after)}")
    mismatches = []
    selections = []
    rounded = 0
    largest = 0.0
    for i, (a, b) in enumerate(zip(before, after, strict=True)):
        if a == b:
            continue
        if is_algorithm(a) and is_algorithm(b):
            selections.append(i)
            continue
        difference = rounding_difference(a, b)
        # A NaN difference fails this comparison and is reported as a mismatch
        if difference is not None and difference <= ROUNDING_TOLERANCE:
            rounded += 1
            largest = max(largest, difference)
            continue
        mismatches.append(i)
    if mismatches:
        for i in mismatches[:10]:
            print(f"result {i}: baseline={before[i]!r}; current={after[i]!r}")
        raise SystemExit(f"{len(mismatches)} real compatibility mismatches")
    exact = len(before) - rounded - len(selections)
    print(f"Matched {len(before)} v1.1.0 real public-call results: {exact} exactly")
    if rounded:
        print(f"  {rounded} differ by rounding (largest scaled gap {largest:.3g})")
    for i in selections:
        print(f'  result {i}: "auto" selects {after[i][1]} (baseline {before[i][1]})')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
