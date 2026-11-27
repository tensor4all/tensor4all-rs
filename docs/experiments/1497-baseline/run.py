#!/usr/bin/env python3
"""Record the predeclared serial diagnostic; build both binaries first.

The adjacent profiling.patch is applied only in a disposable measurement
checkout. This controller never builds, edits library source, or changes a
numerical tolerance. It retains failed and contaminated processes too.
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prime_exec(argv):
    """Use identical priming, then replace this process with a finished binary."""
    print("T4A_LAUNCH " + json.dumps({
        "affinity": sorted(os.sched_getaffinity(0)),
        "thread_environment": {key: os.environ.get(key) for key in [
            "RAYON_NUM_THREADS", "BLAS_NUM_THREADS", "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
            "BLIS_NUM_THREADS", "T4A_MERGE_REFINE_PROFILE",
        ]},
    }), file=sys.stderr, flush=True)
    until = time.monotonic() + 2.0
    value = 0
    while time.monotonic() < until:
        value = (value * 1664525 + 1013904223) & 0xFFFFFFFF
    os.execvpe(argv[0], argv, os.environ)


def summarize(case, directory, code):
    stderr = (directory / "stderr.txt").read_text()
    stdout = (directory / "stdout.txt").read_text()
    events = [json.loads(line.removeprefix("T4A_PROFILE "))
              for line in stderr.splitlines() if line.startswith("T4A_PROFILE ")]
    totals = [e for e in events if e["scope"] == "schedule" and e["phase"] == "total"]
    total = totals[-1] if totals else None
    leaves = [e for e in events if e["scope"] == "prepare" and e["phase"] == "leaf"]
    regions = [e for e in events if e["scope"] == "schedule" and e["phase"] == "region"]
    levels = {}
    for event in regions:
        levels.setdefault(str(event["level"]), []).append(event["elapsed_ns"])
    phases = {}
    for event in events:
        key = event["scope"] + "/" + event["phase"]
        phases[key] = phases.get(key, 0) + event["elapsed_ns"]
    report_path = directory / "ai-run.json"
    guard = json.loads(report_path.read_text()) if report_path.exists() else {}
    result = {
        "case": case["id"], "exit_code": code,
        "guard_verdict": guard.get("verdict"),
        "schedule_completed": bool(total and total["completed"]),
        "phases_seconds": {k: v / 1e9 for k, v in phases.items()},
        "leaf_seconds": [e["elapsed_ns"] / 1e9 for e in leaves],
        "region_seconds_by_level": {k: [v / 1e9 for v in values] for k, values in levels.items()},
        "largest_region_count": max(map(len, levels.values()), default=0),
        "max_observed_threads": max((e["threads"] or 0 for e in events), default=None),
        "max_boundary_rss_kib": max((e["rss_kib"] or 0 for e in events), default=None),
        "max_process_hwm_kib": max((e["hwm_kib"] or 0 for e in events), default=None),
    }
    if total and total["completed"]:
        ns = total["elapsed_ns"]
        result["leaf_share"] = sum(e["elapsed_ns"] for e in leaves) / ns
        result["region_share"] = sum(e["elapsed_ns"] for e in regions) / ns
    for line in stdout.splitlines():
        if line.startswith("{"):
            item = json.loads(line)
            if item.get("kind") == "result":
                result["m6a_result"] = item
    ir_report = directory / "ir-output" / "ft_probe.csv"
    if ir_report.exists():
        with ir_report.open() as handle:
            result["ir_rows"] = list(csv.DictReader(handle))
    frontend = result.get("m6a_result", {}).get("seconds")
    if frontend is None and result.get("ir_rows"):
        frontend = float(result["ir_rows"][0]["ft_seconds"])
    if frontend and total and total["completed"]:
        result["frontend_seconds"] = frontend
        result["leaf_frontend_share"] = sum(e["elapsed_ns"] for e in leaves) / 1e9 / frontend
        result["region_frontend_share"] = sum(e["elapsed_ns"] for e in regions) / 1e9 / frontend
    (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "raw")
    parser.add_argument("--prime-exec", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.prime_exec:
        prime_exec(args.prime_exec)
        return
    os.chdir(ROOT)
    protocol = json.loads((HERE / "protocol.json").read_text())
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != protocol["baseline_commit"]:
        raise SystemExit("Baseline HEAD differs from the registered revision")
    binaries = {
        "m6a": ROOT / "target/release/examples/benchmark_m6a_fourier",
        "ir": ROOT / "target/release/ft_probe",
    }
    for binary in binaries.values():
        if not binary.is_file():
            raise SystemExit(f"Build first: missing {binary}")
    source_files = [
        "Cargo.lock", "benchmarks/rust/benchmark_m6a_fourier.rs",
        "crates/tensor4all-partitionedtreetn/src/reconstruction/mod.rs",
        "crates/tensor4all-partitionedtreetn/src/reconstruction/schedule.rs",
        "crates/tensor4all-partitionedtreetn/src/reconstruction/target.rs",
        "crates/tensor4all-partitionedtreetn/src/reconstruction/profile_1497.rs",
        "target/1497-ir-harness/Cargo.toml", "target/1497-ir-harness/Cargo.lock",
        "docs/experiments/1497-baseline/protocol.json",
        "docs/experiments/1497-baseline/profiling.patch",
        "docs/experiments/1497-baseline/lazy-reader.patch",
        "docs/experiments/1497-baseline/run.py",
    ]
    source_files += [str(p.relative_to(ROOT)) for p in
                     sorted((ROOT / "target/1497-ir-harness/src").rglob("*.rs"))]
    sources = {path: digest(ROOT / path) for path in source_files}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "baseline_commit": head, "host": platform.platform(),
        "machine": platform.machine(), "source_sha256": sources,
        "binaries": {key: {"path": str(value.relative_to(ROOT)), "sha256": digest(value)}
                     for key, value in binaries.items()},
        "protocol_sha256": digest(HERE / "protocol.json"),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    env = dict(os.environ, **protocol["thread_environment"], T4A_MERGE_REFINE_PROFILE="1")
    attempts = []
    for attempt in range(1, protocol["max_complete_attempts"] + 1):
        summaries = []
        clean = True
        for case in protocol["cases"]:
            for repetition in range(case["repeats"]):
                directory = output / f"attempt-{attempt}" / case["id"] / str(repetition)
                directory.mkdir(parents=True, exist_ok=False)
                if case["family"] == "m6a":
                    params = protocol["m6a_common_args"] + case["args"]
                else:
                    c = protocol["ir_common"]
                    params = [str(directory / "ir-output"), str(c["beta"]), str(case["r"]),
                              case["cap"], str(c["goal"]), str(c["ft_rtol"]), c["application"],
                              c["output_depth"], str(c["input_rtol"]), str(c["ir_eps"]),
                              str(c["wmax"]), str(c["mu"]), str(c["degree"]), c["domain"],
                              c["route"], str(c["src_cap"])]
                command = [
                    "ai-run", "measure", "--mem", "8G", "--cpus", protocol["measurement_cpus"],
                    "--report", str(directory / "ai-run.json"), "--",
                    "/usr/bin/time", "-v", "-o", str(directory / "time.txt"),
                    "timeout", "--kill-after=5", str(protocol["per_process_wall_cap_seconds"]),
                    "taskset", "-c", "8",
                    sys.executable, str(Path(__file__).resolve()), "--prime-exec",
                    str(binaries[case["family"]]), *params,
                ]
                (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
                print(f"Starting attempt {attempt}, {case['id']}, repetition {repetition}", flush=True)
                with (directory / "stdout.txt").open("w") as out, (directory / "stderr.txt").open("w") as err:
                    process = subprocess.run(command, env=env, stdout=out, stderr=err)
                summary = summarize(case, directory, process.returncode)
                summaries.append(summary)
                clean = clean and summary["guard_verdict"] == "CLEAN"
                print(json.dumps({k: summary[k] for k in [
                    "case", "exit_code", "guard_verdict", "schedule_completed",
                    "largest_region_count",
                ]}), flush=True)
        for path, expected in sources.items():
            if digest(ROOT / path) != expected:
                raise SystemExit(f"Source changed during diagnostic: {path}")
        attempts.append({"attempt": attempt, "all_clean": clean, "summaries": summaries})
        (output / "attempts.json").write_text(json.dumps(attempts, indent=2) + "\n")
        if clean:
            print(f"Complete diagnostic matrix: CLEAN attempt {attempt}", flush=True)
            return
        print("Discarding timing claims for this entire attempt; retaining every file.", flush=True)
    raise SystemExit("INCONCLUSIVE: both complete attempts contain non-CLEAN measurements")


if __name__ == "__main__":
    main()
