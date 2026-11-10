"""Checks of tools/check_stability_archives.py on small synthetic studies."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import check_stability_archives as checker  # noqa: E402
import stability_archive as archive  # noqa: E402

TABLE = "| N | value |\n|---:|---:|\n| 2 | 1 |\n"


def git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *arguments], text=True,
                                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.org",
                                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.org",
                                        "PATH": "/usr/bin:/bin"}).strip()


@pytest.fixture
def repository(tmp_path: Path) -> tuple[Path, Path]:
    """A repository with one source file, a measured commit and a study of it."""
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "include").mkdir()
    # LF bytes, committed unconverted, so the blobs match the recorded hashes on Windows
    (root / "src" / "kernel.c").write_text("int kernel(void) { return 1; }\n", newline="\n")
    (root / "include" / "api.h").write_text("int kernel(void);\n", newline="\n")
    git(root, "init", "-q")
    git(root, "config", "core.autocrlf", "false")
    git(root, "add", ".")
    git(root, "commit", "-qm", "measured")
    commit = git(root, "rev-parse", "HEAD")
    study = root / "benchmark-results" / "stability-20990101"
    (study / "raw").mkdir(parents=True)
    (study / "raw" / "speed.json").write_text("{}\n")
    (study / "accuracy-tables.md").write_text(TABLE)
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ("include/api.h", "src/kernel.c")}
    (study / "study.json").write_text(json.dumps({
        "profile": "full", "baseline_commit": "0" * 40, "commit": commit,
        "source": {"measured": {"commit": commit, "files_sha256": hashes}},
    }) + "\n")
    archive.write_checksums(study)
    (root / "docs").mkdir()
    (root / "docs" / "stability.md").write_text(
        archive.replace_section("Intro\n\n<!-- BEGIN STABILITY ACCURACY -->\n<!-- END STABILITY ACCURACY -->\n",
                                archive.ACCURACY_SECTION, TABLE))
    return root, study


def test_a_complete_study_passes_and_reports_the_measured_commit(repository):
    root, study = repository
    report = checker.check_archive(root, study, full=False)
    assert report["measured_source"]["status"] == "ok"
    assert report["measured_source"]["working_tree_differs_from_measured_commit"] == []
    assert checker.check_document(root / "docs" / "stability.md", study)["status"] == "ok"


def test_raw_files_may_be_absent_except_in_full_mode(repository):
    root, study = repository
    (study / "raw" / "speed.json").unlink()
    assert checker.check_archive(root, study, full=False)["raw_files_absent"] == 1
    with pytest.raises(ValueError, match="missing file"):
        checker.check_archive(root, study, full=True)


def test_changed_missing_and_unrecorded_published_files_fail(repository):
    root, study = repository
    (study / "accuracy-tables.md").write_text(TABLE + "| 4 | 2 |\n")
    with pytest.raises(ValueError, match="checksum differs"):
        checker.check_archive(root, study, full=False)
    archive.write_checksums(study)
    (study / "notes.txt").write_text("extra\n")
    with pytest.raises(ValueError, match="not recorded"):
        checker.check_archive(root, study, full=False)
    (study / "notes.txt").unlink()
    (study / "accuracy-tables.md").unlink()
    with pytest.raises(ValueError, match="missing file"):
        checker.check_archive(root, study, full=False)


def test_working_tree_changes_are_reported_not_failures(repository):
    root, study = repository
    (root / "src" / "kernel.c").write_text("int kernel(void) { return 2; }\n")
    report = checker.check_archive(root, study, full=False)
    assert report["measured_source"]["working_tree_differs_from_measured_commit"] == ["src/kernel.c"]


def test_a_measured_commit_with_different_sources_fails(repository):
    root, study = repository
    record = json.loads((study / "study.json").read_text())
    record["source"]["measured"]["files_sha256"]["src/kernel.c"] = "0" * 64
    (study / "study.json").write_text(json.dumps(record) + "\n")
    archive.write_checksums(study)
    with pytest.raises(ValueError, match="differs from study.json"):
        checker.check_archive(root, study, full=False)


def test_a_commit_missing_from_the_history_is_skipped(repository):
    root, study = repository
    record = json.loads((study / "study.json").read_text())
    record["commit"] = record["source"]["measured"]["commit"] = "1" * 40
    (study / "study.json").write_text(json.dumps(record) + "\n")
    archive.write_checksums(study)
    assert checker.check_archive(root, study, full=False)["measured_source"]["status"] == "skipped"


def test_document_tables_must_equal_the_newest_study(repository):
    root, study = repository
    document = root / "docs" / "stability.md"
    document.write_text(archive.replace_section(document.read_text(), archive.ACCURACY_SECTION, "| other |\n"))
    with pytest.raises(ValueError, match="differs"):
        checker.check_document(document, study)
    with pytest.raises(ValueError, match="no stability study"):
        checker.check_document(document, None)


def test_malformed_checksum_lines_and_unsafe_paths_are_rejected(tmp_path):
    (tmp_path / "SHA256SUMS").write_text(f"{'0' * 64}  ../outside.txt\n")
    with pytest.raises(ValueError, match="malformed"):
        archive.read_checksums(tmp_path)
