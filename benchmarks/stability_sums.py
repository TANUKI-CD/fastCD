#!/usr/bin/env python3
"""Forward-error study of the fused sums of products and the covariance.

For samples of real-coefficient elements, the script measures the error of
the fused sums, of a batch of products followed by a sum, and of the matrix
of cross moments contracted with the basis signs, against a reference that
uses no fastCD arithmetic.

The reference of a sum is built from error-free transformations. Every term
``x_p[i] y_p[j]`` is split into its rounded product and the exact rounding
error (Dekker's product with Veltkamp splitting), and ``math.fsum`` adds the
terms of one output coordinate with a single rounding. A second pass adds
the same terms and the negated first result, which gives the remainder, so
each reference coordinate is known to about 106 bits. The covariance divides
by the count, so its reference is exact integer arithmetic: the operands
become integers times a power of two, and the cross moments, the sums and
their contraction are evaluated in Python integers. The same integer
arithmetic checks the sum reference at start.

Requires NumPy and a shared build of the library (``-DBUILD_SHARED_LIBS=ON``).
"""

from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import math
import os
import platform
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

U = 2.0**-53
SUM_FAMILIES = (
    "gaussian", "uniform", "rademacher", "self", "conjugate", "cancelling",
    "offset", "graded", "sparse",
)
LARGE_FAMILIES = ("gaussian", "uniform", "offset")
COVARIANCE_FAMILIES = ("gaussian", "offset", "far_offset")
SUM_METHODS = (
    "fused", "fused_direct", "fused_quasilinear", "products_sequential",
    "products_pairwise", "cross_moments",
)
COVARIANCE_METHODS = ("covariance", "centered_products", "centered_moments")
ALGORITHMS = {"auto": 0, "direct": 1, "quasilinear": 3}


@cache
def sign_table(n: int) -> np.ndarray:
    """Signs s[i, j] with e_i e_j = s e_(i xor j), from the doubling formula."""
    if n == 1:
        return np.ones((1, 1), dtype=np.int8)
    h = n // 2
    s = sign_table(h)
    eps = -np.ones(h, dtype=np.int8)
    eps[0] = 1
    out = np.empty((n, n), dtype=np.int8)
    out[:h, :h] = s
    out[:h, h:] = s.T
    out[h:, :h] = s * eps[None, :]
    out[h:, h:] = -(s.T * eps[None, :])
    return out


def split(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Veltkamp splitting: a = hi + lo with 26-bit halves, exactly."""
    c = 134217729.0 * a
    hi = c - (c - a)
    return hi, a - hi


def exact_sum(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (hi, lo) with sum_p x_p y_p = hi + lo to about 106 bits."""
    count, n = x.shape
    signs = sign_table(n)
    xh, xl = split(x)
    yh, yl = split(y)
    index = np.arange(n)
    hi = np.empty(n)
    lo = np.empty(n)
    for k in range(n):
        j = index ^ k
        s = signs[index, j].astype(np.float64)
        a, ah, al = x * s, xh * s, xl * s
        b, bh, bl = y[:, j], yh[:, j], yl[:, j]
        p = a * b
        e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
        terms = np.concatenate((p.ravel(), e.ravel())).tolist()
        hi[k] = math.fsum(terms)
        terms.append(-hi[k])
        lo[k] = math.fsum(terms)
    return hi, lo


def to_integers(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Return integers m and an exponent e with values = m 2^e exactly."""
    mantissa, exponent = np.frexp(values)
    low = int(exponent[values != 0].min()) - 53 if np.any(values != 0) else 0
    flat = [
        int(math.ldexp(m, 53)) << (int(e) - 53 - low) if m else 0
        for m, e in zip(mantissa.ravel().tolist(), exponent.ravel().tolist(),
                        strict=True)
    ]
    return np.array(flat, dtype=object).reshape(values.shape), low


def contract(moments: np.ndarray, conjugate: bool) -> list[int]:
    """Contract an integer matrix of cross moments with the basis signs."""
    n = moments.shape[0]
    signs = sign_table(n)
    out = [0] * n
    for i in range(n):
        for j in range(n):
            s = int(signs[i, j]) * (-1 if conjugate and j else 1)
            out[i ^ j] += s * moments[i, j]
    return out


def integer_sum(x: np.ndarray, y: np.ndarray) -> list[Fraction]:
    """sum_p x_p y_p in exact integer arithmetic."""
    xi, ex = to_integers(x)
    yi, ey = to_integers(y)
    scale = Fraction(2) ** (ex + ey)
    return [v * scale for v in contract(xi.T.dot(yi), False)]


def integer_covariance(x: np.ndarray, y: np.ndarray, ddof: int) -> list[Fraction]:
    """The covariance of two samples in exact rational arithmetic."""
    count = x.shape[0]
    xi, ex = to_integers(x)
    yi, ey = to_integers(y)
    sx, sy = xi.sum(axis=0), yi.sum(axis=0)
    # count * sum (x - mean)(y - mean)^T = count * X^T Y - sx sy^T
    moments = count * xi.T.dot(yi) - np.outer(sx, sy)
    scale = Fraction(2) ** (ex + ey) / (count * (count - ddof))
    return [v * scale for v in contract(moments, True)]


class Workspace(ct.Structure):
    _fields_ = [("data", ct.POINTER(ct.c_double)), ("length", ct.c_size_t)]


class Library:
    """The functions of the shared library that the study calls."""

    def __init__(self, path: Path) -> None:
        self.lib = ct.CDLL(str(path))
        self.lib.fastcd_version_string.restype = ct.c_char_p
        self.version = self.lib.fastcd_version_string().decode()
        self.scratch: dict[int, tuple[np.ndarray, Workspace]] = {}

    def workspace(self, n: int) -> Workspace:
        if n not in self.scratch:
            lengths = [ct.c_size_t(), ct.c_size_t(), ct.c_size_t()]
            self.lib.fastcd_mul_workspace_size(ct.c_size_t(n), ct.byref(lengths[0]))
            self.lib.fastcd_mul_sum_workspace_size(ct.c_size_t(n),
                                                   ct.byref(lengths[1]))
            self.lib.fastcd_covariance_workspace_size(ct.c_size_t(n),
                                                      ct.byref(lengths[2]))
            length = max(v.value for v in lengths) + 1
            data = np.zeros(length)
            self.scratch[n] = (data, Workspace(
                data.ctypes.data_as(ct.POINTER(ct.c_double)), length))
        return self.scratch[n][1]

    @staticmethod
    def _p(a: np.ndarray) -> Any:
        return a.ctypes.data_as(ct.POINTER(ct.c_double))

    def _check(self, status: int) -> None:
        if status:
            raise RuntimeError(f"fastCD returned status {status}")

    def fused(self, x: np.ndarray, y: np.ndarray, algorithm: str) -> np.ndarray:
        count, n = x.shape
        out = np.empty(n)
        self._check(self.lib.fastcd_mul_sum_with_algorithm_f64(
            self._p(out), self._p(x), ct.c_size_t(n), self._p(y), ct.c_size_t(n),
            ct.c_size_t(count), ct.c_size_t(n), None, ct.c_int(ALGORITHMS[algorithm]),
            ct.byref(self.workspace(n))))
        return out

    def planes(self, x: np.ndarray, y: np.ndarray, algorithm: str) -> np.ndarray:
        count, n = x.shape
        px, py = np.ascontiguousarray(x.T), np.ascontiguousarray(y.T)
        out = np.empty(n)
        self._check(self.lib.fastcd_mul_sum_planes_with_algorithm_f64(
            self._p(out), self._p(px), self._p(py), ct.c_size_t(count),
            ct.c_size_t(count), ct.c_size_t(n), ct.c_int(ALGORITHMS[algorithm]),
            ct.byref(self.workspace(n))))
        return out

    def products(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        count, n = x.shape
        out = np.empty((count, n))
        self._check(self.lib.fastcd_mul_batch_f64(
            self._p(out), ct.c_size_t(n), self._p(x), ct.c_size_t(n), self._p(y),
            ct.c_size_t(n), ct.c_size_t(count), ct.c_size_t(n),
            ct.byref(self.workspace(n))))
        return out

    def covariance(self, x: np.ndarray, y: np.ndarray, ddof: int) -> np.ndarray:
        count, n = x.shape
        out = np.empty(n)
        self._check(self.lib.fastcd_covariance_f64(
            self._p(out), self._p(x), ct.c_size_t(n), self._p(y), ct.c_size_t(n),
            ct.c_size_t(count), ct.c_size_t(n), ct.c_size_t(ddof),
            ct.byref(self.workspace(n))))
        return out


def moments_sum(x: np.ndarray, y: np.ndarray, conjugate: bool = False) -> np.ndarray:
    """The binary64 matrix of cross moments, contracted with the signs."""
    n = x.shape[1]
    signs = sign_table(n).astype(np.float64)
    if conjugate:
        signs = signs.copy()
        signs[:, 1:] *= -1.0
    weighted = (x.T @ y) * signs
    out = np.zeros(n)
    index = np.arange(n)
    for i in range(n):
        np.add.at(out, index ^ i, weighted[i])
    return out


def make_inputs(family: str, n: int, count: int,
                rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Draw one sample pair of the named family."""
    shape = (count, n)
    if family == "uniform":
        return rng.uniform(-1, 1, shape), rng.uniform(-1, 1, shape)
    if family == "rademacher":
        return (rng.integers(0, 2, shape) * 2.0 - 1.0,
                rng.integers(0, 2, shape) * 2.0 - 1.0)
    x = rng.standard_normal(shape)
    if family == "self":
        return x, x.copy()
    if family == "conjugate":
        y = -x
        y[:, 0] = x[:, 0]
        return x, y
    y = rng.standard_normal(shape)
    if family == "cancelling":
        # The second half repeats the first with y negated: the sum is zero
        half = count // 2
        x[half:2 * half], y[half:2 * half] = x[:half], -y[:half]
    elif family == "offset":
        x, y = x + 1.0e6, y + 1.0e6
    elif family == "far_offset":
        x, y = x + 1.0e9, y + 1.0e9
    elif family == "graded":
        # Elements of alternating magnitude over exponents [-24, 24]
        scale = np.ldexp(1.0, rng.integers(-24, 25, count))
        x, y = x * scale[:, None], y * scale[:, None]
    elif family == "sparse":
        keep = np.zeros(shape, dtype=bool)
        keep[np.arange(count), rng.integers(0, n, count)] = True
        x = np.where(keep, x, 0.0)
        keep = np.zeros(shape, dtype=bool)
        keep[np.arange(count), rng.integers(0, n, count)] = True
        y = np.where(keep, y, 0.0)
    elif family != "gaussian":
        raise ValueError(family)
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def pair_scale(x: np.ndarray, y: np.ndarray) -> float:
    """sum_p ||x_p|| ||y_p||, the scale of the rounding errors of a sum."""
    return float(np.sum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)))


def sum_case(library: Library, family: str, n: int, count: int,
             seed: list[int]) -> dict[str, Any]:
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    x, y = make_inputs(family, n, count, rng)
    hi, lo = exact_sum(x, y)
    scale = pair_scale(x, y)
    products = library.products(x, y)
    results = {
        "fused": library.fused(x, y, "auto"),
        "fused_direct": library.fused(x, y, "direct"),
        "fused_quasilinear": library.fused(x, y, "quasilinear"),
        "products_sequential": np.add.accumulate(products, axis=0)[-1],
        # NumPy sums pairwise along a contiguous axis
        "products_pairwise": np.ascontiguousarray(products.T).sum(axis=1),
        "cross_moments": moments_sum(x, y),
    }
    layouts_agree = all(
        np.array_equal(library.planes(x, y, a).view(np.uint64),
                       results[m].view(np.uint64))
        for a, m in (("auto", "fused"), ("direct", "fused_direct"),
                     ("quasilinear", "fused_quasilinear"))
    )
    exact_norm = float(np.linalg.norm(hi))
    row: dict[str, Any] = {
        "kind": "sum", "family": family, "dimension": n, "count": count,
        "seed": seed, "scale": scale, "exact_norm": exact_norm,
        "layouts_agree": layouts_agree, "eta": {},
    }
    for name, value in results.items():
        error = float(np.linalg.norm((value - hi) - lo))
        row["eta"][name] = error / (U * scale) if scale else 0.0
    return row


def covariance_case(library: Library, family: str, n: int, count: int,
                    seed: list[int]) -> dict[str, Any]:
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    x, y = make_inputs(family, n, count, rng)
    exact = integer_covariance(x, y, 1)
    xc, yc = x - x.mean(axis=0), y - y.mean(axis=0)
    conj = -yc
    conj[:, 0] = yc[:, 0]
    scale = pair_scale(xc, yc) / (count - 1)
    products = library.products(np.ascontiguousarray(xc), np.ascontiguousarray(conj))
    results = {
        "covariance": library.covariance(x, y, 1),
        "centered_products": np.add.accumulate(products, axis=0)[-1] / (count - 1),
        "centered_moments": moments_sum(xc, yc, conjugate=True) / (count - 1),
    }
    row: dict[str, Any] = {
        "kind": "covariance", "family": family, "dimension": n, "count": count,
        "seed": seed, "scale": scale, "eta": {},
    }
    for name, value in results.items():
        squared = sum((Fraction(float(v)) - e) ** 2
                      for v, e in zip(value.tolist(), exact, strict=True))
        row["eta"][name] = math.sqrt(float(squared)) / (U * scale)
    return row


def validate_oracle() -> dict[str, Any]:
    """Check the error-free sum reference against exact integer arithmetic."""
    checked = 0
    worst = 0.0
    for n in (1, 2, 4, 8, 16, 32):
        for family in SUM_FAMILIES:
            rng = np.random.Generator(np.random.PCG64([20261119, n, checked]))
            x, y = make_inputs(family, n, 40, rng)
            hi, lo = exact_sum(x, y)
            for h, low, e in zip(hi.tolist(), lo.tolist(), integer_sum(x, y),
                                 strict=True):
                # hi is the correctly rounded sum and lo the rounded remainder
                if e:
                    assert float(e) == h, (n, family)
                remainder = e - Fraction(h)
                assert float(remainder) == low, (n, family)
                if e:
                    worst = max(worst, abs(float((remainder - Fraction(low)) / e)))
            checked += 1
    return {"cases": checked, "max_relative_remainder": worst}


def plan(profile: str) -> list[tuple[str, str, int, int, list[int]]]:
    """List the cases: (kind, family, dimension, count, seed)."""
    quick = profile == "quick"
    cases = []
    for level in range(11):
        n = 1 << level
        if quick and n > 64:
            break
        repeats = 2 if quick else 8 if n <= 64 else 4 if n <= 256 else 2 if n == 512 else 1
        families = SUM_FAMILIES if n <= 256 else LARGE_FAMILIES
        for family in families:
            if family == "sparse" and n == 1:
                continue
            for r in range(repeats):
                cases.append(("sum", family, n, 1024,
                              [20261119, SUM_FAMILIES.index(family), n, 1024, r]))
        if n <= 64:
            for count in (64, 4096):
                for family in ("gaussian", "offset"):
                    for r in range(2 if quick else 4):
                        cases.append(("sum", family, n, count, [
                            20261119, SUM_FAMILIES.index(family), n, count, r]))
            for family in COVARIANCE_FAMILIES:
                for r in range(1 if quick else 4):
                    cases.append(("covariance", family, n, 1024, [
                        20261120, COVARIANCE_FAMILIES.index(family), n, 1024, r]))
    return cases


_LIBRARY: Library | None = None


def _start(path: str) -> None:
    global _LIBRARY
    _LIBRARY = Library(Path(path))


def _run(case: tuple[str, str, int, int, list[int]]) -> dict[str, Any]:
    kind, family, n, count, seed = case
    assert _LIBRARY is not None
    function = sum_case if kind == "sum" else covariance_case
    return function(_LIBRARY, family, n, count, seed)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Median and maximum eta by dimension, and maxima by family."""
    result: dict[str, Any] = {"sum_by_dimension": [], "sum_by_family": [],
                              "sum_by_count": [], "covariance": []}
    sums = [r for r in rows if r["kind"] == "sum"]
    main = [r for r in sums if r["count"] == 1024]
    for n in sorted({r["dimension"] for r in main}):
        dense = [r for r in main if r["dimension"] == n
                 and r["family"] in ("gaussian", "uniform")]
        result["sum_by_dimension"].append({
            "dimension": n, "cases": len(dense),
            "median": {m: statistics.median(r["eta"][m] for r in dense)
                       for m in SUM_METHODS},
            "max": {m: max(r["eta"][m] for r in dense) for m in SUM_METHODS},
        })
    for family in SUM_FAMILIES:
        chosen = [r for r in main if r["family"] == family]
        if chosen:
            result["sum_by_family"].append({
                "family": family, "cases": len(chosen),
                "median": {m: statistics.median(r["eta"][m] for r in chosen)
                           for m in SUM_METHODS},
                "max": {m: max(r["eta"][m] for r in chosen) for m in SUM_METHODS},
            })
    for count in sorted({r["count"] for r in sums}):
        chosen = [r for r in sums if r["count"] == count and r["dimension"] <= 64
                  and r["family"] in ("gaussian", "offset")]
        result["sum_by_count"].append({
            "count": count, "cases": len(chosen),
            "median": {m: statistics.median(r["eta"][m] for r in chosen)
                       for m in SUM_METHODS},
            "max": {m: max(r["eta"][m] for r in chosen) for m in SUM_METHODS},
        })
    covariances = [r for r in rows if r["kind"] == "covariance"]
    for family in COVARIANCE_FAMILIES:
        chosen = [r for r in covariances if r["family"] == family]
        if chosen:
            result["covariance"].append({
                "family": family, "cases": len(chosen),
                "median": {m: statistics.median(r["eta"][m] for r in chosen)
                           for m in COVARIANCE_METHODS},
                "max": {m: max(r["eta"][m] for r in chosen)
                        for m in COVARIANCE_METHODS},
            })
    result["layouts_agree"] = all(r["layouts_agree"] for r in sums)
    result["cases"] = {"sum": len(sums), "covariance": len(covariances)}
    return result


def _cells(entry: dict[str, Any], methods: tuple[str, ...]) -> str:
    return " | ".join(f"{entry['median'][m]:.2f} / {entry['max'][m]:.2f}"
                      for m in methods)


def report(path: Path, document: dict[str, Any]) -> None:
    summary = document["summary"]
    head = ("fused | fused, direct | fused, quasilinear | products, sequential "
            "| products, pairwise | cross moments")
    lines = [
        "# Accuracy of the fused sums",
        "",
        f"fastCD {document['version']}, {summary['cases']['sum']} sums and "
        f"{summary['cases']['covariance']} covariances, on {document['cpu']}.",
        "",
        "Each cell is the median / maximum of `eta = ||S_hat - S||_2 / (u sum_p "
        "||x_p||_2 ||y_p||_2)` with `u = 2^-53`. `fused` is the automatic "
        "algorithm; the two `products` columns sum a batch of products in "
        "sequence and with NumPy's pairwise `sum`; `cross moments` is the "
        "binary64 matrix product of the samples contracted with the basis "
        "signs.",
        "",
        "## Gaussian and uniform samples of 1,024 elements",
        "",
        f"| N | Cases | {head} |",
        "|---:|---:|---|---|---|---|---|---|",
    ]
    for entry in summary["sum_by_dimension"]:
        lines.append(f"| {entry['dimension']} | {entry['cases']} | "
                     f"{_cells(entry, SUM_METHODS)} |")
    lines += ["", "## By family, samples of 1,024 elements, all dimensions", "",
              f"| Family | Cases | {head} |", "|---|---:|---|---|---|---|---|---|"]
    for entry in summary["sum_by_family"]:
        lines.append(f"| {entry['family']} | {entry['cases']} | "
                     f"{_cells(entry, SUM_METHODS)} |")
    lines += ["", "## By sample size, Gaussian and offset samples, N <= 64", "",
              f"| Count | Cases | {head} |", "|---:|---:|---|---|---|---|---|---|"]
    for entry in summary["sum_by_count"]:
        lines.append(f"| {entry['count']} | {entry['cases']} | "
                     f"{_cells(entry, SUM_METHODS)} |")
    lines += [
        "", "## Covariance, samples of 1,024 elements, N <= 64", "",
        "Here the scale is `sum_p ||x_p - mean(x)||_2 ||y_p - mean(y)||_2 / "
        "(count - 1)`. `centered products` and `centered moments` subtract "
        "the means with NumPy first.",
        "",
        "| Family | Cases | covariance | centered products | centered moments |",
        "|---|---:|---|---|---|",
    ]
    for entry in summary["covariance"]:
        lines.append(f"| {entry['family']} | {entry['cases']} | "
                     f"{_cells(entry, COVARIANCE_METHODS)} |")
    lines += [
        "",
        "The record and the plane layout returned identical bits in "
        + ("every sum." if summary["layouts_agree"] else "NOT every sum."),
        "",
    ]
    path.write_text("\n".join(lines))


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True,
                        help="shared fastCD library")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("quick", "full"), default="full")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.output.suffix.lower() == ".md":
        parser.error("output and Markdown report paths must differ")
    if args.workers < 1:
        parser.error("workers must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    library_path = args.library.resolve()
    start = time.monotonic()
    oracle = validate_oracle()
    cases = plan(args.profile)
    # Largest cases first, so the pool stays busy to the end
    order = sorted(range(len(cases)),
                   key=lambda i: -cases[i][2] ** 2 * cases[i][3])
    rows: list[Any] = [None] * len(cases)
    with ProcessPoolExecutor(args.workers, initializer=_start,
                             initargs=(str(library_path),)) as pool:
        for i, row in zip(order, pool.map(_run, [cases[i] for i in order]),
                          strict=True):
            rows[i] = row
    library = Library(library_path)
    document = {
        "schema": "fastcd-sum-accuracy-1",
        "version": library.version,
        "library_sha256": hashlib.sha256(library_path.read_bytes()).hexdigest(),
        "profile": args.profile,
        "cpu": platform.processor() or platform.machine(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "oracle": oracle,
        "wall_seconds": time.monotonic() - start,
        "summary": summarize(rows),
        "rows": rows,
    }
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    document["cpu"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    report(args.output.with_suffix(".md"), document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
