#!/usr/bin/env python3
"""Measure the original enumeration workload, not a full interpolation."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent / "rss"
BUILD = ROOT / "target/1496-rss"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True)


def main():
    protocol = json.loads((HERE / "protocol.json").read_text())
    BUILD.mkdir(parents=True, exist_ok=True)
    source = "crates/tensor4all-treetci/src/globalpivot.rs"
    original = git("show", protocol["baseline"] + ":" + source)
    start = original.index("fn enumeration_points(")
    end = original.index("\n}\n", start) + 3
    (BUILD / "baseline.rs").write_text(original[start:end])
    candidate = git("show", protocol["candidate"] + ":crates/tensor4all-treetci/src/globalpivot/enumeration.rs")
    (BUILD / "candidate.rs").write_text(candidate)
    (BUILD / "main.rs").write_text((HERE / "main.rs").read_text())
    subprocess.run([
        "ai-run", "build", "--mem", "1G", "--", "rustc", "--edition=2021",
        "-C", "opt-level=3", "-C", "debuginfo=0", str(BUILD / "main.rs"),
        "-o", str(BUILD / "enumeration"),
    ], check=True)
    compiler = subprocess.check_output([
        "ai-run", "build", "--mem", "1G", "--", "rustc", "-Vv",
    ], text=True)
    manifest = {
        "protocol": protocol,
        "compiler": compiler,
        "source_sha256": {p.name: digest(p) for p in BUILD.glob("*.rs")},
        "controller_sha256": digest(Path(__file__)),
        "binary_sha256": digest(BUILD / "enumeration"),
        "ai_run_sha256": digest(Path("/root/.local/bin/ai-run")),
    }
    raw = HERE / "raw"
    if raw.exists():
        raise SystemExit("Use a fresh result directory; existing observations are retained")
    raw.mkdir()
    (raw / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    env = dict(os.environ, RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", BLAS_NUM_THREADS="1")
    attempts = []
    for attempt in range(1, 3):
        samples = []
        for pair in range(protocol["pairs"]):
            arms = ["baseline", "candidate"] if pair % 2 == 0 else ["candidate", "baseline"]
            for arm in arms:
                directory = raw / f"attempt-{attempt}" / f"{pair}-{arm}"
                directory.mkdir(parents=True)
                command = [
                    "ai-run", "measure", "--mem", "2G", "--cpus", "8-15",
                    "--report", str(directory / "ai-run.json"), "--",
                    "/usr/bin/time", "-v", "-o", str(directory / "time.txt"),
                    "timeout", "--kill-after=5", "120", "taskset", "-c", "8",
                    str(BUILD / "enumeration"), arm,
                ]
                (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
                with (directory / "stdout.txt").open("w") as out, (directory / "stderr.txt").open("w") as err:
                    p = subprocess.run(command, env=env, stdout=out, stderr=err)
                guard = json.loads((directory / "ai-run.json").read_text())
                stdout = (directory / "stdout.txt").read_text()
                observations = json.loads(stdout) if p.returncode == 0 else None
                peak = next(int(line.split(":")[-1]) for line in
                            (directory / "time.txt").read_text().splitlines()
                            if "Maximum resident set size" in line)
                sample = {"arm": arm, "pair": pair, "exit": p.returncode,
                          "verdict": guard["verdict"], "peak_rss_kib": peak,
                          "observations": observations}
                samples.append(sample)
                print(json.dumps(sample), flush=True)
        clean = all(s["verdict"] == "CLEAN" for s in samples)
        success = all(s["exit"] == 0 for s in samples)
        same = len({json.dumps(s["observations"], sort_keys=True) for s in samples}) == 1
        attempts.append({"attempt": attempt, "clean": clean, "success": success,
                         "same_observations": same, "samples": samples})
        (raw / "attempts.json").write_text(json.dumps(attempts, indent=2) + "\n")
        if clean:
            if not success or not same:
                raise SystemExit("Complete experiment failed; no selective retry")
            return
    raise SystemExit("INCONCLUSIVE: both complete attempts contaminated")


if __name__ == "__main__":
    main()
