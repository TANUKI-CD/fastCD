#!/usr/bin/env python3
"""Independent, reproducible forward-error study (optional numpy + gmpy2).

Every reference coefficient is an EXACT integer times a power of two, obtained
from stored binary64 operands and the signed-XOR multiplication table. MPFR is
used at >=256 bits for norms and metrics, and to independently re-evaluate
selected products at 256 and 512 bits. No fastCD arithmetic is used by the oracle.
"""

from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from fractions import Fraction
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

import numpy as np

try:
    import gmpy2 as gp
except ImportError as exc:
    raise SystemExit("Experiment only: install optional gmpy2 (python -m pip install gmpy2)") from exc

REAL_FAMILIES = (
    "gaussian", "uniform", "rademacher", "sparse_basis", "sparse_four",
    "sparse_eighth", "correlated_plus", "correlated_minus", "repeated",
    "scalar_cancellation", "dyadic_cancellation", "long_sum_cancellation", "alternating_magnitude",
    "scalar_only", "imaginary_only", "repeated_blocks", "recursive_blocks",
    "zero_divisor", "near_zero_divisor_20", "near_zero_divisor_40", "near_zero_divisor_52",
)
COMPLEX_FAMILIES = REAL_FAMILIES[:-4] + ("isotropic", "near_isotropic")


@cache
def sign_table(n: int) -> np.ndarray:
    """Derive basis signs from (a,b)(c,d)=(ac-d* b, da+b c*).

    Here * is the CD involution, fixing coefficient scalars. Index of each
    basis product is XOR. The off-diagonal blocks reverse multiplication.
    This routine neither imports nor calls fastCD's basis/sign functions.
    """
    if n < 1 or n & (n - 1):
        raise ValueError("dimension must be a positive power of two")
    if n == 1:
        return np.ones((1, 1), dtype=np.int8)
    s = sign_table(n // 2)
    conjugation = -np.ones(n // 2, dtype=np.int8)
    conjugation[0] = 1
    result = np.block([[s, s.T], [s * conjugation, -s.T * conjugation]])
    result.setflags(write=False)
    return result


def encode(values: np.ndarray) -> tuple[list[int], int]:
    """Convert binary64 components EXACTLY to integers * 2**(-exponent)."""
    ratios = [float(v).as_integer_ratio() for v in values]
    exponent = max((d.bit_length() - 1 for _, d in ratios), default=0)
    return [a << (exponent - (d.bit_length() - 1)) for a, d in ratios], exponent


def parts(values: np.ndarray) -> tuple[list[int], list[int], int]:
    components = np.column_stack((values.real, values.imag)).ravel()
    integers, exponent = encode(components)
    return integers[::2], integers[1::2], exponent


def exact_product(x: np.ndarray, y: np.ndarray) -> tuple[list[int], list[int], int]:
    """Signed-XOR product with exact dyadic components and integer summation."""
    n = len(x)
    xr, xi, ex = parts(x)
    yr, yi, ey = parts(y)
    rows = [row.tobytes() for row in sign_table(n)]
    left = [(j, a, b) for j, (a, b) in enumerate(zip(xr, xi, strict=True)) if a or b]
    right = [(j, a, b) for j, (a, b) in enumerate(zip(yr, yi, strict=True)) if a or b]
    zr, zi = [0] * n, [0] * n
    for j, a, b in left:
        row = rows[j]
        for ell, c, d in right:
            k = j ^ ell
            re, im = a * c - b * d, a * d + b * c
            if row[ell] == 1:
                zr[k] += re
                zi[k] += im
            else:
                zr[k] -= re
                zi[k] -= im
    return zr, zi, ex + ey


def squared_errors(outputs: list[np.ndarray], reference: tuple[list[int], list[int], int]) -> list[int]:
    """Exact squared Euclidean errors of several outputs on one dyadic scale.

    The returned integers share the denominator 2**(2E) for a common exponent E,
    so they compare exactly even when the binary64 norms round to the same value.
    """
    zr, zi, er = reference
    encoded = [parts(values) for values in outputs]
    exponent = max([er, *(e for _, _, e in encoded)])
    shift = exponent - er
    target_r, target_i = [v << shift for v in zr], [v << shift for v in zi]
    totals = []
    for real, imag, e in encoded:
        scale = exponent - e
        totals.append(sum(((a << scale) - b) ** 2 + ((c << scale) - d) ** 2
                          for a, b, c, d in zip(real, target_r, imag, target_i, strict=True)))
    return totals


def as_mp(reference: tuple[list[int], list[int], int]) -> list[Any]:
    re, im, exponent = reference
    return [gp.mpc(gp.mul_2exp(gp.mpfr(a), -exponent), gp.mul_2exp(gp.mpfr(b), -exponent))
            for a, b in zip(re, im, strict=True)]


def mp_product(x: np.ndarray, y: np.ndarray, precision: int) -> list[Any]:
    """Separate direct MPFR/MPC sum, for precision checks only."""
    with gp.context(precision=precision):
        xx = [gp.mpc(float(v.real), float(v.imag)) for v in x]
        yy = [gp.mpc(float(v.real), float(v.imag)) for v in y]
        s = sign_table(len(x))
        return [sum((int(s[j, j ^ k]) * xx[j] * yy[j ^ k]
                     for j in range(len(x))), gp.mpc(0)) for k in range(len(x))]


def defining_product(x: list[Fraction], y: list[Fraction]) -> list[Fraction]:
    """Literal rational doubling recursion, independent of signed-XOR oracle."""
    if len(x) == 1:
        return [x[0] * y[0]]
    h = len(x) // 2
    a, b, c, d = x[:h], x[h:], y[:h], y[h:]
    def conjugate(v: list[Fraction]) -> list[Fraction]:
        return v[:1] + [-a for a in v[1:]]
    ac, db = defining_product(a, c), defining_product(conjugate(d), b)
    da, bc = defining_product(d, a), defining_product(b, conjugate(c))
    return [u - v for u, v in zip(ac, db, strict=True)] + [u + v for u, v in zip(da, bc, strict=True)]


def validate_oracle() -> dict[str, Any]:
    # Quaternion i*j=k, j*k=i, k*i=j and standard doubling octonion triples
    known = {(1, 2): (3, 1), (2, 3): (1, 1), (3, 1): (2, 1),
             (1, 4): (5, 1), (2, 4): (6, 1), (3, 4): (7, 1),
             (1, 6): (7, -1), (2, 5): (7, 1)}
    for (j, k), (index, sign) in known.items():
        assert j ^ k == index and sign_table(8)[j, k] == sign
    count = 0
    for n in (1, 2, 4, 8, 16):
        for j in range(n):
            for k in range(n):
                x, y = [Fraction(0)] * n, [Fraction(0)] * n
                x[j], y[k] = Fraction(1), Fraction(1)
                result = defining_product(x, y)
                assert result[j ^ k] == int(sign_table(n)[j, k])
                assert sum(bool(v) for v in result) == 1
                count += 1
        x = np.arange(-(n // 2), n - n // 2, dtype=np.float64) / 8
        y = np.arange(n, dtype=np.float64)[::-1] / 16
        real, imag, exponent = exact_product(x, y)
        rational = defining_product([Fraction(float(a)) for a in x], [Fraction(float(a)) for a in y])
        assert all(Fraction(v, 1 << exponent) == q for v, q in zip(real, rational, strict=True))
        assert not any(imag)
    return {"basis_cases_against_rational_doubling": count, "dense_rational_dimensions": [1, 2, 4, 8, 16], "known_quaternion_octonion_cases": len(known)}


@lru_cache(maxsize=1)
def zero_divisor() -> tuple[np.ndarray, np.ndarray]:
    """Find and independently prove a sparse sedenion zero-divisor pair."""
    n, s = 16, sign_table(16)
    for a in range(1, n):
        for b in range(a + 1, n):
            for c in range(1, n):
                d = a ^ b ^ c
                if not c < d < n:
                    continue
                for t in (-1, 1):
                    if s[a, c] + t * s[b, d] == 0 and t * s[a, d] + s[b, c] == 0:
                        x, y = np.zeros(n), np.zeros(n)
                        x[a], x[b], y[c], y[d] = 1, 1, 1, t
                        zr, zi, _ = exact_product(x, y)
                        assert not any(zr) and not any(zi)
                        return x, y
    raise AssertionError("no independently verified zero divisor found")


def euclidean_norm(values: np.ndarray) -> float:
    """Correctly rounded Euclidean norm of the binary64 coefficients.

    The squares are summed with math.fsum, so the result is the same on every
    platform and does not depend on the BLAS library NumPy uses.
    """
    parts = np.concatenate([values.real, values.imag]) if np.iscomplexobj(values) else values
    return math.sqrt(math.fsum((parts * parts).tolist()))


def make_inputs(domain: str, n: int, family: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.Generator(np.random.PCG64(seed))
    complex_domain = domain == "complex"
    dtype = np.complex128 if complex_domain else np.float64

    def normal() -> np.ndarray:
        result = rng.standard_normal(n).astype(dtype)
        if complex_domain:
            result += 1j * rng.standard_normal(n)
        return result / euclidean_norm(result)

    x, y = normal(), normal()
    if family == "uniform":
        x, y = rng.uniform(-1, 1, (2, n)).astype(dtype)
        if complex_domain:
            x += 1j * rng.uniform(-1, 1, n)
            y += 1j * rng.uniform(-1, 1, n)
    elif family == "rademacher":
        x, y = (rng.choice([-1.0, 1.0], (2, n)) * 2.0 ** (-math.ceil(math.log2(n) / 2))).astype(dtype)
        if complex_domain:
            x += 1j * rng.choice([-1.0, 1.0], n) * 2.0 ** (-math.ceil(math.log2(n) / 2))
            y += 1j * rng.choice([-1.0, 1.0], n) * 2.0 ** (-math.ceil(math.log2(n) / 2))
    elif family.startswith("sparse_"):
        support = {"sparse_basis": 1, "sparse_four": min(4, n), "sparse_eighth": max(1, n // 8)}[family]
        for v in (x, y):
            keep = rng.choice(n, support, replace=False)
            v[np.setdiff1d(np.arange(n), keep)] = 0
    elif family in ("correlated_plus", "correlated_minus", "repeated"):
        y = x.copy() if family == "repeated" else (1 if family.endswith("plus") else -1) * x + 2.0 ** -40 * y
    elif family == "scalar_cancellation":
        x[0] = 1
        # bilinear pairing, no coefficient conjugation
        with gp.context(precision=256):
            value = sum((gp.mpc(float(a.real), float(a.imag)) * gp.mpc(float(b.real), float(b.imag)) for a, b in zip(x[1:], y[1:], strict=True)), gp.mpc(0))
        y[0] = complex(value) if complex_domain else float(value.real)
    elif family == "dyadic_cancellation":
        x.fill(0.125)
        y.fill(0.125)
        y[0] = (n - 1) * 0.125
    elif family == "long_sum_cancellation":
        x.fill(1.0 + (1j if complex_domain else 0.0))
        y.fill(1.0)
        x[0], y[0] = 0, 0
        if n >= 4:
            y[1], y[-1] = 2.0 ** 53, -2.0 ** 53
    elif family == "alternating_magnitude":
        magnitude = np.exp2(np.linspace(-24, 24, n).astype(int))
        x = (rng.choice([-1.0, 1.0], n) * magnitude).astype(dtype)
        y = (rng.choice([-1.0, 1.0], n) * magnitude[::-1]).astype(dtype)
        if complex_domain:
            x += 1j * magnitude[::-1]
            y -= 1j * magnitude
    elif family == "scalar_only":
        x[1:], y[1:] = 0, 0
    elif family == "imaginary_only":
        x[0], y[0] = 0, 0
    elif family in ("repeated_blocks", "recursive_blocks"):
        width = min(4, n)
        x = np.tile(x[:width], n // width)
        y = np.tile(y[:width], n // width)
        if family == "recursive_blocks":
            signs = np.array([(-1.0) ** j.bit_count() for j in range(n // width)])
            x *= np.repeat(signs, width)
    elif "zero_divisor" in family:
        if n < 16:
            raise ValueError("zero divisors require dimension >=16")
        a, b = zero_divisor()
        x, y = np.zeros(n), np.zeros(n)
        x[:16], y[:16] = a, b
        if family != "zero_divisor":
            scale = 2.0 ** -int(family.rsplit("_", 1)[1])
            x += scale * normal()
            y += scale * normal()
    elif family in ("isotropic", "near_isotropic"):
        # Q(x)=1+i**2=0, while the Euclidean norm is sqrt(2)
        x.fill(0)
        x[0], x[1] = 1, 1j
        if family == "near_isotropic":
            x += 2.0 ** -40 * normal()
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


class Workspace(ct.Structure):
    _fields_ = [("data", ct.POINTER(ct.c_double)), ("length", ct.c_size_t)]


class Library:
    """Only public C APIs are used; the comparison never calls an oracle kernel."""

    def __init__(self, path: Path):
        self.path = path.resolve()
        self.lib = ct.CDLL(str(self.path))
        self.lib.fastcd_mul_algorithm_workspace_size.argtypes = [ct.c_size_t, ct.c_int, ct.POINTER(ct.c_size_t)]
        self.lib.fastcd_workspace_size_c128.argtypes = [ct.c_size_t, ct.c_int, ct.POINTER(ct.c_size_t)]
        for suffix in ("f64", "c128"):
            function = getattr(self.lib, "fastcd_mul_with_algorithm_" + suffix)
            function.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_int, ct.POINTER(Workspace)]
            function.restype = ct.c_int

    def multiply(self, x: np.ndarray, y: np.ndarray, algorithm: int) -> np.ndarray:
        n, size = len(x), ct.c_size_t()
        complex_domain = np.iscomplexobj(x)
        status = (self.lib.fastcd_workspace_size_c128(n, 0, ct.byref(size)) if complex_domain
                  else self.lib.fastcd_mul_algorithm_workspace_size(n, algorithm, ct.byref(size)))
        if status:
            raise RuntimeError(f"workspace query returned {status}")
        scratch = np.empty(size.value, dtype=np.float64)
        workspace = Workspace(scratch.ctypes.data_as(ct.POINTER(ct.c_double)), size.value)
        result = np.empty_like(x)
        function = getattr(self.lib, "fastcd_mul_with_algorithm_" + ("c128" if complex_domain else "f64"))
        status = function(result.ctypes.data, x.ctypes.data, y.ctypes.data, n, algorithm, ct.byref(workspace))
        if status:
            raise RuntimeError(f"product returned status {status}")
        return result

    def multiply_batch(self, xs: np.ndarray, ys: np.ndarray, algorithm: int) -> np.ndarray:
        """Multiply the rows of xs and ys with one batched call."""
        count, n = xs.shape
        complex_domain = np.iscomplexobj(xs)
        size = ct.c_size_t()
        status = (self.lib.fastcd_workspace_size_c128(n, 0, ct.byref(size)) if complex_domain
                  else self.lib.fastcd_mul_algorithm_workspace_size(n, algorithm, ct.byref(size)))
        if status:
            raise RuntimeError(f"workspace query returned {status}")
        scratch = np.empty(size.value, dtype=np.float64)
        workspace = Workspace(scratch.ctypes.data_as(ct.POINTER(ct.c_double)), size.value)
        xs, ys = np.ascontiguousarray(xs), np.ascontiguousarray(ys)
        result = np.empty_like(xs)
        name = "fastcd_mul_batch_c128" if complex_domain else "fastcd_mul_batch_with_algorithm_f64"
        function = getattr(self.lib, name)
        function.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_size_t,
                             ct.c_size_t, ct.c_size_t, ct.c_int, ct.POINTER(Workspace)]
        function.restype = ct.c_int
        # Strides count doubles for real records and complex coefficients for complex records
        status = function(result.ctypes.data, n, xs.ctypes.data, n, ys.ctypes.data, n, count, n,
                          algorithm, ct.byref(workspace))
        if status:
            raise RuntimeError(f"batched product returned status {status}")
        return result


def metric(result: np.ndarray, reference: tuple[list[int], list[int], int], x: np.ndarray, y: np.ndarray, precision: int) -> dict[str, Any]:
    with gp.context(precision=precision):
        ref = as_mp(reference)
        xx = [gp.mpc(float(v.real), float(v.imag)) for v in x]
        yy = [gp.mpc(float(v.real), float(v.imag)) for v in y]
        zz = [gp.mpc(float(v.real), float(v.imag)) for v in result]
        def norm(values: list[Any]) -> Any:
            return gp.sqrt(sum((a.real * a.real + a.imag * a.imag for a in values), gp.mpfr(0)))
        differences = [a - b for a, b in zip(zz, ref, strict=True)]
        error, reference_norm = norm(differences), norm(ref)
        scale = norm(xx) * norm(yy)
        return {"absolute": float(error), "relative": float(error / reference_norm) if reference_norm else None,
                "normalized": float(error / (gp.mul_2exp(gp.mpfr(1), -53) * scale)) if scale else None,
                "maximum_coordinate": float(max(map(abs, differences))), "scalar": float(abs(differences[0])),
                "chi": float(scale / reference_norm) if reference_norm else None,
                "reference_norm": float(reference_norm), "zero_product": not bool(reference_norm)}


def ratio(numerator: float, denominator: float) -> dict[str, Any]:
    return {"value": numerator / denominator if denominator else None,
            "zero_case": None if denominator else ("both_zero" if not numerator else "positive_over_zero")}


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for algorithm, metrics in row["algorithms"].items():
            groups[row["domain"], row["dimension"], row["family"], algorithm].append({**metrics, "improved": row["improved"], "tied": row["tied"]})
    summaries = []
    for (domain, n, family, algorithm), data in sorted(groups.items()):
        eta = [r["normalized"] for r in data if r["normalized"] is not None]
        relative = [r["relative"] for r in data if r["relative"] is not None]
        improved = sum(r["improved"] for r in data)
        count = len(data)
        # The Wilson interval is descriptive, since samples from deterministic
        # families need not be exchangeable draws from a stochastic population
        center = (improved + 1.96 ** 2 / 2) / (count + 1.96 ** 2)
        radius = 1.96 / (count + 1.96 ** 2) * math.sqrt(improved * (count - improved) / count + 1.96 ** 2 / 4)
        summaries.append({"domain": domain, "dimension": n, "family": family, "algorithm": algorithm,
                          "samples": count, "median_normalized": float(np.median(eta)), "mean_normalized": float(np.mean(eta)),
                          "p95_normalized": float(np.percentile(eta, 95)), "p99_normalized": float(np.percentile(eta, 99)),
                          "maximum_normalized": max(eta), "median_relative": float(np.median(relative)) if relative else None,
                          "maximum_relative": max(relative) if relative else None, "median_scalar": float(np.median([r["scalar"] for r in data])),
                          "fraction_improved": improved / count, "fraction_tied": sum(r["tied"] for r in data) / count,
                          "fraction_improved_wilson95": [max(0, center - radius), min(1, center + radius)],
                          "tail_estimates_reliable": count >= 100,
                          "median_eta_over_sqrtN_log2N": float(np.median(eta)) / (math.sqrt(n) * math.log2(n)),
                          "median_eta_over_N": float(np.median(eta)) / n})
    return summaries


def source_metadata(path: Path) -> dict[str, Any]:
    files = sorted([*path.glob("src/**/*.c"), *path.glob("src/**/*.h"), *path.glob("include/**/*.h")])
    hashes = {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    record: dict[str, Any] = {"path": str(path.resolve()), "files_sha256": hashes,
                             "aggregate_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}
    try:
        record["commit"] = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        record["status"] = subprocess.check_output(["git", "-C", str(path), "status", "--short"], text=True).strip()
    except subprocess.CalledProcessError:
        provenance = path / "BASELINE.json"
        record["commit"] = json.loads(provenance.read_text()).get("starting_commit") if provenance.exists() else None
    return record


def sample_count(n: int, family: str, profile: str) -> int:
    dense = family in ("gaussian", "uniform")
    if profile == "smoke":
        return 1
    base = (32 if dense else 4) if n <= 32 else ((12 if dense else 2) if n <= 128 else (4 if dense else 1))
    if n >= 1024:
        base = 1
    return base * (8 if profile == "full" else 1)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.precision < 256:
        raise ValueError("precision must be >=256 bits")
    for maximum in (args.real_max, args.complex_max):
        if maximum < 2 or maximum & (maximum - 1):
            raise ValueError("maximum dimensions must be powers of two >=2")
    if args.cpu is not None and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {args.cpu})
    validation = validate_oracle()
    baseline, modified = Library(args.baseline), Library(args.modified)
    scalar_modified = Library(args.scalar_modified) if args.scalar_modified else None
    rows, arrays, checks = [], {}, []
    corpus_hash = hashlib.sha256()
    start = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for domain in ("real", "complex"):
        maximum = args.real_max if domain == "real" else args.complex_max
        for exponent in range(1, int(math.log2(maximum)) + 1):
            n = 1 << exponent
            for family_id, family in enumerate(REAL_FAMILIES if domain == "real" else COMPLEX_FAMILIES):
                if "zero_divisor" in family and n < 16:
                    continue
                for sample in range(sample_count(n, family, args.profile)):
                    seed = int(np.random.SeedSequence([args.seed, domain == "complex", n, family_id, sample]).generate_state(1, dtype=np.uint64)[0])
                    x, y = make_inputs(domain, n, family, seed)
                    key = f"{domain}_{n}_{family}_{sample}"
                    arrays[key + "_x"], arrays[key + "_y"] = x, y
                    payload = x.astype(x.dtype.newbyteorder("<")).tobytes() + y.astype(y.dtype.newbyteorder("<")).tobytes()
                    corpus_hash.update(key.encode() + payload)
                    reference = exact_product(x, y)
                    algorithms = {"original_quasilinear": metric(baseline.multiply(x, y, 3), reference, x, y, args.precision),
                                  "modified_quasilinear": metric(modified.multiply(x, y, 3), reference, x, y, args.precision),
                                  "direct": metric(modified.multiply(x, y, 1), reference, x, y, args.precision)}
                    if domain == "real" and n <= 1024:
                        algorithms["cariow_cariowa"] = metric(modified.multiply(x, y, 2), reference, x, y, args.precision)
                    # Both builds share the direct source, so their direct outputs must match
                    assert baseline.multiply(x, y, 1).tobytes() == modified.multiply(x, y, 1).tobytes(), f"direct baseline mismatch: {key}"
                    if scalar_modified:
                        assert scalar_modified.multiply(x, y, 3).tobytes() == modified.multiply(x, y, 3).tobytes(), f"scalar/SIMD mismatch: {key}"
                    old, new = algorithms["original_quasilinear"]["absolute"], algorithms["modified_quasilinear"]["absolute"]
                    rows.append({"key": key, "domain": domain, "dimension": n, "family": family, "sample": sample,
                                 "seed": seed, "operands_sha256": hashlib.sha256(payload).hexdigest(), "algorithms": algorithms,
                                 "improved": new < old, "tied": new == old,
                                 "modified_over_original": ratio(new, old),
                                 "modified_over_direct": ratio(new, algorithms["direct"]["absolute"])})
                    if family == "gaussian" and sample == 0 and n in (2, 16, 128, min(512, maximum)):
                        exact = None
                        for precision in (256, 512):
                            with gp.context(precision=precision):
                                exact = as_mp(reference)
                                candidate = mp_product(x, y, precision)
                                difference = max(abs(a - b) for a, b in zip(exact, candidate, strict=True))
                                assert difference == 0  # Controlled exponents fit exactly at 256 bits.
                                observed = metric(modified.multiply(x, y, 3), reference, x, y, precision)
                            checks.append({"key": key, "precision_bits": precision, "maximum_reference_discrepancy": float(difference),
                                           "normalized_error": observed["normalized"]})
            print(f"{domain} N={n}: {len(rows)} products; {time.perf_counter()-start:.1f}s", flush=True)
    a, b = zero_divisor()
    validation["zero_divisor"] = {"x_support": np.flatnonzero(a).tolist(), "x_values": a[a != 0].tolist(),
                                 "y_support": np.flatnonzero(b).tolist(), "y_values": b[b != 0].tolist(),
                                 "verified_exactly": True}
    corpus_path = args.output.with_name(args.output.stem + "-corpus.npz")
    np.savez_compressed(corpus_path, **arrays)
    document = {"schema": "fastcd-stability-v1", "profile": args.profile, "completed": True,
                "command": [sys.executable, *sys.argv], "seed": args.seed, "precision_bits": args.precision,
                "oracle": "exact binary64-to-integer dyadic signed-XOR, arbitrary-integer sums; MPFR norms",
                "oracle_validation": validation, "precision_checks": checks,
                "environment": {"python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
                                "numpy": np.__version__, "gmpy2": gp.version(), "mpfr": gp.mpfr_version(),
                                "cpu_model": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), None) if Path("/proc/cpuinfo").exists() else platform.processor(),
                                "cpu_flags": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("flags")), None) if Path("/proc/cpuinfo").exists() else None,
                                "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None},
                "baseline_library": {"path": str(args.baseline.resolve()), "sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest()},
                "modified_library": {"path": str(args.modified.resolve()), "sha256": hashlib.sha256(args.modified.read_bytes()).hexdigest()},
                "scalar_modified_library": {"path": str(args.scalar_modified.resolve()), "sha256": hashlib.sha256(args.scalar_modified.read_bytes()).hexdigest()} if args.scalar_modified else None,
                "baseline_source": source_metadata(args.baseline_source), "modified_source": source_metadata(args.modified_source),
                "tool_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), Path(__file__).with_name("stability_report.py"))},
                "build_records": {name: (path.parent / "fastcd-build-Release.txt").read_text() if (path.parent / "fastcd-build-Release.txt").exists() else None for name, path in (("baseline", args.baseline), ("modified", args.modified))},
                "corpus": {"path": corpus_path.name, "npz_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
                           "ordered_operands_sha256": corpus_hash.hexdigest(), "pairs": len(rows)},
                "agreement": {"direct_baseline_modified_cases": len(rows), "direct_mismatches": 0,
                              "modified_scalar_simd_cases": len(rows) if scalar_modified else 0, "modified_scalar_simd_mismatches": 0 if scalar_modified else None},
                "elapsed_seconds": time.perf_counter() - start,
                "limitations": ["Dimension-dependent finite samples: groups below100 samples do not support reliable population-tail inference.",
                                "Controlled finite exponent ranges; no overflow, underflow or exceptional-value study.",
                                "Finite samples describe only the measured corpus."],
                "samples": rows, "summaries": summarize(rows)}
    args.output.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    return document


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="immutable baseline shared library")
    parser.add_argument("--modified", type=Path, required=True)
    parser.add_argument("--scalar-modified", type=Path, help="optional scalar build for bitwise SIMD agreement checks")
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--modified-source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True, help="JSON results file; the corpus is written beside it")
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="pilot")
    parser.add_argument("--real-max", type=int, default=2048)
    parser.add_argument("--complex-max", type=int, default=512)
    parser.add_argument("--precision", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--cpu", type=int, help="optional CPU affinity for experiment process")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_arguments())
