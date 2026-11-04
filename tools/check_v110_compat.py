#!/usr/bin/env python3
"""Compare real public-call results against an installed v1.1.0 baseline.

Build/install the old tag into a separate --target directory first. The
check runs both packages in isolated subprocesses using the same interpreter,
covering constructors, buffers, algorithms, scalar protocols, arrays,
structural operations, exceptions and pickle reconstruction. It deliberately
compares existing calls; new field selectors are exercised in the domain tests.
"""

from __future__ import annotations

import argparse
import json
import os
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
        results.append(float(value).hex())
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
    emit(A.default_algorithm)
    emit(A.default_batch_algorithm)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-path", type=Path, required=True)
    parser.add_argument("--current-path", type=Path, required=True)
    args = parser.parse_args()
    before, after = probe(args.baseline_path), probe(args.current_path)
    if len(before) != len(after):
        raise SystemExit(f"different result counts: {len(before)} vs {len(after)}")
    mismatches = [
        i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b
    ]
    if mismatches:
        for i in mismatches[:10]:
            print(f"result {i}: baseline={before[i]!r}; current={after[i]!r}")
        raise SystemExit(f"{len(mismatches)} real compatibility mismatches")
    print(f"Matched {len(before)} v1.1.0 real public-call results exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
