"""Run the finite-field and FLINT protocols, alone or with the real-coefficient run."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from . import benchmark as common
    from . import domains, flint
except ImportError:
    import benchmark as common
    import domains
    import flint


def commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    """Keep all outputs distinct, and make omitted sibling runners explicit."""
    directory = Path(__file__).resolve().parent
    selected = (
        ("real", "domains", "flint") if args.protocol == "all" else (args.protocol,)
    )
    suffix = args.runner.suffix
    runners = {
        "real": args.runner,
        "domains": args.domain_runner
        or (args.runner.parent / f"fastcd_domains_benchmark{suffix}"),
        "flint": args.flint_runner
        or (args.runner.parent / f"fastcd_flint_benchmark{suffix}"),
    }
    if args.protocol != "all":
        runners[args.protocol] = (
            args.flint_runner if args.protocol == "flint" else args.domain_runner
        ) or args.runner
    result = []
    for protocol in selected:
        output = (
            args.output.with_name(f"{args.output.stem}.{protocol}.json")
            if args.protocol == "all"
            else args.output
        )
        script = "benchmark.py" if protocol == "real" else f"{protocol}.py"
        command = [
            sys.executable,
            str(directory / script),
            "--runner",
            str(runners[protocol]),
            "--output",
            str(output),
            "--dimensions",
            args.dimensions,
            "--trials",
            str(args.trials),
            "--blocks",
            str(args.blocks),
            "--warmup",
            str(args.warmup),
            "--min-time-ms",
            f"{args.min_time_ms:g}",
            "--seed",
            str(args.seed),
            "--corpora",
            str(args.corpora if protocol == "real" else args.corpora / protocol),
            "--subprocess-timeout",
            str(args.subprocess_timeout),
        ]
        if args.cpu is not None:
            command += ["--cpu", str(args.cpu)]
        metadata = (
            args.build_metadata
            if args.protocol != "all" or protocol == "real"
            else args.domain_build_metadata
            if protocol == "domains"
            else args.flint_build_metadata
        )
        if metadata is not None:
            command += ["--build-metadata", str(metadata)]
        if protocol == "real":
            command += [
                "--modes",
                ",".join(args.modes),
                "--norm-min",
                str(args.norm_min),
                "--norm-max",
                str(args.norm_max),
                "--max-direct-dimension",
                str(args.max_direct_dimension),
                "--max-cc-dimension",
                str(args.max_cc_dimension),
            ]
            if args.verify_trials is not None:
                command += ["--verify-trials", str(args.verify_trials)]
        elif protocol == "domains":
            command += [
                "--fields",
                args.fields or domains.DEFAULT_FIELDS,
                "--modes",
                ",".join(args.modes),
            ]
        else:
            fields = args.fields or flint.DEFAULT_FIELDS
            if args.protocol == "all":
                fields = ",".join(
                    f for f in fields.split(",") if int(f.split(":")[0]) != 0
                )
                if not fields:
                    raise ValueError(
                        "all requires at least one finite field for the FLINT comparison"
                    )
            command += ["--fields", fields, "--workloads", args.workloads]
        result.append((protocol, command))
    return result


def preflight(args: argparse.Namespace, selected: list[tuple[str, list[str]]]) -> None:
    """Check every runner/dependency/output before an all run spends time timing."""
    if args.protocol == "all":
        for path in (
            args.output,
            args.output.with_suffix(".md"),
            args.output.with_name(args.output.stem + ".partial.json"),
        ):
            if path.exists():
                raise ValueError(f"refusing to overwrite {path}")
    for protocol, command in selected:
        runner = Path(command[command.index("--runner") + 1])
        if not runner.is_file():
            extra = (
                "; configure with -DFASTCD_BUILD_FLINT_BENCHMARK=ON"
                if protocol == "flint"
                else ""
            )
            raise ValueError(f"{protocol} runner is missing: {runner}{extra}")
        options = command[2:]
        if protocol == "flint":
            flint.preflight(flint.parse_arguments(options))
        elif protocol == "domains":
            domain_args = domains.parse_arguments(options)
            metadata = common.run_json(
                [str(runner.resolve()), "--metadata"], args.subprocess_timeout
            )
            if metadata.get("schema") != "fastcd-domains-1":
                raise ValueError("domain runner must name fastcd_domains_benchmark")
            common.provenance(domain_args)
        else:
            metadata_path = None
            if "--build-metadata" in command:
                metadata_path = Path(command[command.index("--build-metadata") + 1])
            common.provenance(
                argparse.Namespace(runner=runner, build_metadata=metadata_path)
            )
            output = Path(command[command.index("--output") + 1])
            for path in (
                output,
                output.with_suffix(".md"),
                output.with_name(output.stem + ".partial.json"),
            ):
                if path.exists():
                    raise ValueError(f"refusing to overwrite {path}")


def run(args: argparse.Namespace) -> int:
    selected = commands(args)
    preflight(args, selected)
    document: dict[str, Any] = {
        "schema": "fastcd-benchmark-suite-1",
        "status": "incomplete",
        "protocol": args.protocol,
        "reports": [],
    }
    partial = args.output.with_name(args.output.stem + ".partial.json")
    if args.protocol == "all":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        partial.write_text(json.dumps(document, indent=2) + "\n")
    for protocol, command in selected:
        result = subprocess.run(command, check=False)
        if result.returncode:
            if args.protocol == "all":
                document.update({"status": "failed", "failed_protocol": protocol})
                partial.write_text(json.dumps(document, indent=2) + "\n")
            return result.returncode
        output = Path(command[command.index("--output") + 1])
        document["reports"].append(
            {
                "protocol": protocol,
                "json": output.name,
                "markdown": output.with_suffix(".md").name,
                "sha256": common.sha256_file(output),
                "command": command,
            }
        )
        if args.protocol == "all":
            partial.write_text(json.dumps(document, indent=2) + "\n")
    if args.protocol == "all":
        document["status"] = "ok"
        args.output.write_text(json.dumps(document, indent=2) + "\n")
        lines = [
            "# Benchmark protocols",
            "",
            "Each protocol has its own JSON observations and Markdown report.",
            "",
            "The real protocol preserves the real direct/Cariow–Cariowa/quasilinear "
            "experiment. Domains compares direct and quasilinear over complex and "
            "finite fields. FLINT (Fast Library for Number Theory) supplies the "
            "independent coefficient arithmetic used in the optional scalar and "
            "complete CD comparisons.",
            "",
            "| Protocol | Results | Raw data |",
            "|---|---|---|",
        ]
        for entry in document["reports"]:
            lines.append(
                f"| {entry['protocol']} | [Report]({entry['markdown']}) | [JSON]({entry['json']}) |"
            )
        args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
        partial.unlink()
    return 0
