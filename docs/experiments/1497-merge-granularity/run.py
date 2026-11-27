#!/usr/bin/env python3
"""Build once, then retain the complete declared serial granularity diagnostic."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prime_exec(argv):
    until = time.monotonic() + 2
    value = 0
    while time.monotonic() < until:
        value = (value * 1664525 + 1013904223) & 0xFFFFFFFF
    os.execvpe(argv[0], argv, os.environ)


def parse(directory, exit_code):
    regions, candidates, probes = [], [], []
    emission = 0
    for line in (directory / "stderr.txt").read_text().splitlines():
        if line.startswith("T4A_PARENT_PROBE "):
            probe = json.loads(line.split(" ", 1)[1])
            assert sum(p["items"] for p in probe["parents"]) == probe["input_items"]
            assert sum(p["additions"] for p in probe["parents"]) == probe["additions"]
            assert sum(p["ns"] for p in probe["parents"]) <= probe["ns"]
            probes.append(probe)
        elif line.startswith("T4A_CANDIDATE "):
            candidate = json.loads(line.split(" ", 1)[1])
            assert sum(p["ns"] for p in probes) <= candidate["ns"]
            candidate["probes"] = probes
            candidates.append(candidate)
            probes = []
        elif line.startswith("T4A_REGION "):
            region = json.loads(line.split(" ", 1)[1])
            assert not probes
            assert sum(c["ns"] for c in candidates) <= region["ns"]
            region["candidates"] = candidates
            regions.append(region)
            candidates = []
        elif line.startswith("T4A_EMIT "):
            emission += json.loads(line.split(" ", 1)[1])["ns"]
    result = None
    for line in (directory / "stdout.txt").read_text().splitlines():
        if line.startswith("{"):
            item = json.loads(line)
            if item.get("kind") == "result":
                result = item
    if result is not None and regions:
        total_additions = sum(p["additions"] for r in regions for c in r["candidates"] for p in c["probes"])
        assert total_additions == result["merge_refine"]["additions"], (total_additions, result)
    guard = json.loads((directory / "ai-run.json").read_text())
    summary = {"exit_code": exit_code, "guard_verdict": guard["verdict"],
               "result": result, "regions": regions, "emission_ns": emission,
               "unclosed_probes": probes, "unclosed_candidates": candidates}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path)
    parser.add_argument("--target-dir", type=Path)
    parser.add_argument("--prime-exec", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.prime_exec:
        prime_exec(args.prime_exec)
        return
    if args.checkout is None or args.target_dir is None:
        parser.error("--checkout and --target-dir are required")
    checkout, target = args.checkout.resolve(), args.target_dir.resolve()
    raw = HERE / "raw"
    if raw.exists():
        raise SystemExit("Keep existing evidence; use a fresh result directory")
    protocol = json.loads((HERE / "protocol.json").read_text())
    git = lambda *a: subprocess.check_output(["git", "-C", str(checkout), *a], text=True)
    assert git("rev-parse", "HEAD").strip() == protocol["baseline_commit"]
    relative = "crates/tensor4all-partitionedtreetn/src/reconstruction/schedule"
    sources = ["Cargo.toml", "Cargo.lock", "benchmarks/rust/Cargo.toml",
               "benchmarks/rust/benchmark_m6a_fourier.rs", relative + ".rs",
               relative + "/tolerance.rs", relative + "/profiling1497.rs",
               "crates/tensor4all-quanticstransform/tests/qft_reconstruction/output_selection.rs"]
    hashes = {name: sha(checkout / name) for name in sources}
    ai_run_path = shutil.which("ai-run")
    if ai_run_path is None:
        raise SystemExit("Install ai-run and add it to PATH before reproducing this campaign")
    environment = dict(os.environ, CARGO_BUILD_JOBS="2", CARGO_TARGET_DIR=str(target))
    build = [ai_run_path, "build", "--mem", "6G", "--", "cargo"]
    raw.mkdir()
    with (raw / "build.log").open("w") as log:
        subprocess.run(build + ["build", "--locked", "--release", "-p", "tensor4all-benchmarks",
                               "--features", "patched-fourier", "--example", "benchmark_m6a_fourier"],
                       cwd=checkout, env=environment, stdout=log, stderr=log, check=True)
    with (raw / "test-build.log").open("w") as log:
        compiled = subprocess.run(build + ["test", "--locked", "--release", "-p", "tensor4all-quanticstransform",
                                           "--features", "partitioned", "--test", "qft_reconstruction", "--no-run",
                                           "--message-format=json"], cwd=checkout, env=environment,
                                  stdout=subprocess.PIPE, stderr=log, text=True, check=True)
    (raw / "test-build.jsonl").write_text(compiled.stdout)
    artifacts = [json.loads(line) for line in compiled.stdout.splitlines() if line.startswith("{")]
    tests = [Path(a["executable"]) for a in artifacts if a.get("reason") == "compiler-artifact"
             and a.get("target", {}).get("name") == "qft_reconstruction" and a.get("executable")]
    assert len(tests) == 1
    binaries = {"m6a": target / "release/examples/benchmark_m6a_fourier", "tiny": tests[0]}
    compiler = subprocess.check_output([ai_run_path, "build", "--mem", "1G", "--", "rustc", "-Vv"], text=True)
    for name, digest in hashes.items():
        assert sha(checkout / name) == digest, name
    patch = git("diff", "--", "crates", "benchmarks")
    helper = subprocess.run(["git", "diff", "--no-index", "--", "/dev/null", relative + "/profiling1497.rs"],
                            cwd=checkout, capture_output=True, text=True)
    assert helper.returncode == 1
    (HERE / "profiling.patch").write_text(patch + helper.stdout)
    manifest = {"at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "baseline": protocol["baseline_commit"], "platform": platform.platform(), "compiler": compiler,
                "source_sha256": hashes, "protocol_sha256": sha(HERE / "protocol.json"),
                "controller_sha256": sha(Path(__file__)), "instrument_sha256": sha(HERE / "instrument.py"),
                "profiler_sha256": sha(HERE / "profiler.rs"), "patch_sha256": sha(HERE / "profiling.patch"),
                "ai_run_sha256": sha(Path(ai_run_path)),
                "binaries": {k: {"path": str(p), "sha256": sha(p)} for k, p in binaries.items()}}
    (raw / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    env = dict(os.environ, **protocol["thread_environment"])
    env.pop("T4A_MERGE_GRANULARITY", None)

    def execute(directory, family, params, tracing=True):
        directory.mkdir(parents=True, exist_ok=False)
        execution_env = dict(env)
        if tracing:
            execution_env["T4A_MERGE_GRANULARITY"] = "1"
        binary = [str(binaries[family]), *params]
        if family == "m6a":
            binary = [sys.executable, str(Path(__file__).resolve()), "--prime-exec", *binary]
        command = [ai_run_path, "measure", "--mem", "8G", "--cpus", "8-15", "--report", str(directory / "ai-run.json"),
                   "--", "timeout", "--kill-after=5", "300", "taskset", "-c", "8", *binary]
        (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        with (directory / "stdout.txt").open("w") as out, (directory / "stderr.txt").open("w") as err:
            p = subprocess.run(command, env=execution_env, stdout=out, stderr=err)
        summary = parse(directory, p.returncode)
        print(directory.relative_to(raw), summary["exit_code"], summary["guard_verdict"], flush=True)
        return summary

    validation = []
    params = protocol["common_args"] + protocol["cases"][0]["args"]
    for mode in [False, True]:
        validation.append(execute(raw / f"validation-{int(mode)}", "m6a", params, mode))
    signatures = [{k: v for k, v in s["result"].items() if k not in {"seconds", "g0_seconds"}} for s in validation]
    assert all(s["exit_code"] == 0 and s["guard_verdict"] == "CLEAN" for s in validation)
    assert signatures[0] == signatures[1]
    tiny = execute(raw / "tiny", "tiny", ["--exact", "output_selection::minimal_two_bit_exact_fanout_has_two_second_level_region_jobs", "--nocapture", "--test-threads=1"])
    assert tiny["exit_code"] == 0 and tiny["guard_verdict"] == "CLEAN"
    widths = [sum(r["level"] == level for r in tiny["regions"]) for level in [1, 2]]
    assert widths == [1, 2], widths
    attempts = []
    for attempt in range(1, 3):
        samples = []
        for case in protocol["cases"]:
            for repetition in range(case["repeats"]):
                name = f"attempt-{attempt}/{case['id']}/{repetition}"
                sample = execute(raw / name, "m6a", protocol["common_args"] + case["args"])
                samples.append({"case": case["id"], "directory": name, "exit": sample["exit_code"], "verdict": sample["guard_verdict"]})
        clean = all(s["verdict"] == "CLEAN" for s in samples)
        success = all(s["exit"] == 0 for s in samples)
        attempts.append({"attempt": attempt, "all_clean": clean, "success": success, "samples": samples})
        (raw / "attempts.json").write_text(json.dumps(attempts, indent=2) + "\n")
        for name, digest in hashes.items():
            assert sha(checkout / name) == digest, name
        for key, binary in binaries.items():
            assert sha(binary) == manifest["binaries"][key]["sha256"], key
        if clean:
            if not success:
                raise SystemExit("Complete CLEAN experiment failed; failures retained")
            return
    raise SystemExit("INCONCLUSIVE: both complete attempts contaminated")


if __name__ == "__main__":
    main()
