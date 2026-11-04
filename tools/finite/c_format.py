"""Emit readable C using the conventions of fastCD's original real kernels.

The formatter is a development dependency only. Pinning its version makes
committed finite kernels reproducible; ordinary C and wheel builds use those
committed files and do not invoke this tool.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@cache
def _formatter() -> str:
    """Find the pinned formatter in the active environment or on PATH."""
    sibling = Path(sys.executable).with_name("clang-format")
    executable = str(sibling) if sibling.is_file() else shutil.which("clang-format")
    if executable is None:
        raise RuntimeError("finite generation requires clang-format==18.1.8")
    version = subprocess.run(
        [executable, "--version"], capture_output=True, text=True, check=True
    ).stdout
    if "clang-format version 18.1.8" not in version:
        raise RuntimeError("finite generation requires clang-format==18.1.8")
    return executable


def format_c(source: str) -> str:
    """Format emitted C without changing tokens or the arithmetic schedule."""
    return subprocess.run(
        [_formatter(), f"--style=file:{ROOT / '.clang-format'}"],
        input=source,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
