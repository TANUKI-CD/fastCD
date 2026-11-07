#!/usr/bin/env python3
"""Check the stability studies and the tables shown in docs/stability.md.

For every benchmark-results/stability-*/ directory with a SHA256SUMS file:

- every published file matches its recorded SHA-256, and every file present is
  recorded; files under raw/ may be absent, since they are not distributed;
- when the git history contains the measured commit, its src/ and include/
  files are exactly the ones recorded in study.json, with the same hashes.
  Differences between the working tree and the measured commit are reported
  but are not failures.

The delimited tables of docs/stability.md must equal the tables of the newest
study. With --full, the raw files must be present and match as well, and the
tables are regenerated from the raw results (this needs NumPy). The default
mode uses only the standard library. Nothing is ever written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def library_sources(names: list[str]) -> list[str]:
    """The recorded source file set: src/**/*.{c,h} and include/**/*.h."""
    return sorted(
        n for n in names
        if (n.startswith("src/") and n.endswith((".c", ".h")))
        or (n.startswith("include/") and n.endswith(".h"))
    )


def git(root: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    command = ["git", "-C", str(root), *arguments]
    return subprocess.run(command, capture_output=True, check=False)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check_commit(root: Path, study: dict[str, Any]) -> dict[str, Any]:
    commit = study["commit"]
    recorded: dict[str, str] = study["source"]["measured"]["files_sha256"]
    require(study["source"]["measured"]["commit"] == commit,
            "study.json records two different measured commits")
    if git(root, "rev-parse", "--git-dir").returncode:
        return {"status": "skipped", "reason": "no git repository"}
    if git(root, "cat-file", "-e", f"{commit}^{{commit}}").returncode:
        return {"status": "skipped",
                "reason": f"commit {commit} is not in the local history"}
    listing = git(root, "ls-tree", "-r", "--name-only", commit, "--", "src", "include")
    require(listing.returncode == 0, f"cannot list the files of {commit}")
    names = library_sources(listing.stdout.decode().splitlines())
    require(names == sorted(recorded),
            f"the source files of {commit} differ from those in study.json")
    for name in names:
        blob = git(root, "cat-file", "blob", f"{commit}:{name}")
        require(blob.returncode == 0, f"cannot read {name} at {commit}")
        require(sha256(blob.stdout) == recorded[name],
                f"{name} at {commit} differs from study.json")
    files = [*root.glob("src/**/*.c"), *root.glob("src/**/*.h"),
             *root.glob("include/**/*.h")]
    working = {p.relative_to(root).as_posix(): p for p in files}
    changed = sorted(
        n for n in set(working) | set(recorded)
        if n not in working or n not in recorded
        or sha256(working[n].read_bytes()) != recorded[n]
    )
    return {"status": "ok", "commit": commit, "files": len(names),
            "working_tree_differs_from_measured_commit": changed}


def check_archive(root: Path, directory: Path, full: bool) -> dict[str, Any]:
    import stability_archive as archive

    entries = archive.read_checksums(directory)
    present = {p.relative_to(directory).as_posix() for p in directory.rglob("*")
               if p.is_file() and p.name != "SHA256SUMS"}
    unrecorded = sorted(present - set(entries))
    require(not unrecorded,
            f"files not recorded in {directory}/SHA256SUMS: {unrecorded}")
    missing_raw = []
    for name, digest in entries.items():
        path = directory / name
        if not path.is_file():
            require(archive.is_raw(name) and not full, f"missing file: {path}")
            missing_raw.append(name)
            continue
        require(archive.sha256(path) == digest, f"checksum differs: {path}")
    require("study.json" in entries, f"{directory} has no study.json")
    study = json.loads((directory / "study.json").read_text())
    raw = sum(archive.is_raw(n) for n in entries)
    report = {"directory": str(directory), "files": len(entries),
              "published_files": len(entries) - raw,
              "raw_files_present": raw - len(missing_raw),
              "raw_files_absent": len(missing_raw), "profile": study["profile"],
              "baseline_commit": study["baseline_commit"], "commit": study["commit"],
              "measured_source": check_commit(root, study)}
    if full:
        report["tables_regenerated"] = regenerate(directory)
    return report


def regenerate(directory: Path) -> list[str]:
    """Regenerate every table from the raw results and compare it with the archive."""
    import stability_block_size
    import stability_cc_performance
    import stability_report
    import stability_speed_report

    renderers = {
        "accuracy-tables.md": ("raw/accuracy.json", stability_report.render),
        "speed-tables.md": ("raw/speed.json", stability_speed_report.render),
        "cc-speed-tables.md": ("raw/cc-speed.json", stability_cc_performance.render),
        "block-size-tables.md": ("raw/block-size.json", stability_block_size.render),
    }
    checked = []
    for table, (source, render) in renderers.items():
        rendered = render(json.loads((directory / source).read_text()))
        require(rendered == (directory / table).read_text(),
                f"{directory / table} differs from {source}")
        checked.append(table)
    return checked


def check_document(document: Path, newest: Path | None) -> dict[str, Any]:
    import stability_archive as archive

    if not document.exists():
        return {"status": "skipped", "reason": f"{document} does not exist"}
    text = document.read_text()
    labels = [label for label in archive.SECTION_FILES
              if archive.markers(label)[0] in text]
    if not labels:
        return {"status": "skipped", "reason": "the document has no delimited tables"}
    require(newest is not None,
            f"{document} shows stability tables, but there is no stability study")
    assert newest is not None
    for label in labels:
        table = newest / archive.SECTION_FILES[label]
        require(archive.extract_section(text, label)
                == table.read_text().strip("\n") + "\n",
                f"the {label} section of {document} differs from {table}")
    return {"status": "ok", "study": str(newest), "sections": labels}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true",
                        help="also require and check the raw files")
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="repository root (default: this checkout)")
    parser.add_argument("--archive", type=Path, action="append",
                        help="study directory to check (default: every "
                             "benchmark-results/stability-*/ with SHA256SUMS)")
    parser.add_argument("--document", type=Path,
                        help="document with the delimited tables "
                             "(default: docs/stability.md)")
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / "benchmarks"))
    results = root / "benchmark-results"
    archives = sorted(args.archive
                      or [p.parent for p in results.glob("stability-*/SHA256SUMS")])
    document = (args.document or root / "docs" / "stability.md").resolve()
    try:
        studies = [check_archive(root, d.resolve(), args.full) for d in archives]
        shown = check_document(document, archives[-1].resolve() if archives else None)
    except ValueError as error:
        print(json.dumps({"status": "failed", "error": str(error)}, indent=2))
        raise SystemExit(1) from error
    scope = "full" if args.full else "published files"
    print(json.dumps({"status": "ok", "scope": scope, "studies": studies,
                      "document": shown}, indent=2))


if __name__ == "__main__":
    main()
