#!/usr/bin/env python3
"""Record bounded Cargo diagnostics, build/post-build time, and the suite budget."""

import argparse
from collections import deque
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

# Nextest's own measure of the suite it ran: "Summary [  79.365s] 3662 tests run".
SUITE_SUMMARY = re.compile(r"\s*Summary \[\s*([0-9.]+)s\]")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-seconds", type=float,
                        help="fail when the reported suite execution time exceeds this")
    parser.add_argument("name")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a command is required")
    directory = Path(os.environ.get("CI_BUILD_DIAGNOSTICS", "target/ci-build-diagnostics"))
    directory.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("CARGO_LOG", "cargo::core::compiler::fingerprint=info")
    rustc = subprocess.run(["rustc", "-Vv"], capture_output=True, text=True, check=True).stdout
    cpu_model = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu_model = next((line.partition(":")[2].strip()
                          for line in Path("/proc/cpuinfo").read_text().splitlines()
                          if line.startswith("model name")), cpu_model)
    started = time.monotonic()
    build_finished = None
    suite_seconds = None
    tail = deque(maxlen=200)
    fingerprints = []
    compiled = []
    fresh = []
    with (directory / f"{args.name}.log").open("w") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, env=env)
        for line in process.stdout:
            elapsed = time.monotonic() - started
            clean = re.sub(r"\x1b\[[0-9;]*m", "", line).rstrip()
            log.write(f"[{elapsed:.3f}s] {clean}\n")
            tail.append(clean)
            if "Finished" in clean and "target(s) in" in clean:
                build_finished = elapsed
            summary_match = SUITE_SUMMARY.match(clean)
            if summary_match:
                suite_seconds = float(summary_match.group(1))
            if "fingerprint" in clean and len(fingerprints) < 60:
                fingerprints.append(clean)
            if clean.lstrip().startswith("Compiling "):
                compiled.append(clean.strip())
            if clean.lstrip().startswith("Fresh "):
                fresh.append(clean.strip())
        returncode = process.wait()
    elapsed = time.monotonic() - started
    budget_exceeded = (args.budget_seconds is not None and suite_seconds is not None
                       and suite_seconds > args.budget_seconds)
    summary = {"command": command, "elapsed_seconds": elapsed,
               "build_finished_seconds": build_finished,
               "post_build_seconds": None if build_finished is None else elapsed - build_finished,
               "suite_seconds": suite_seconds, "budget_seconds": args.budget_seconds,
               "budget_exceeded": budget_exceeded,
               "returncode": returncode, "compiled": compiled, "fresh": fresh,
               "fingerprints_first_60": fingerprints,
               "rustc": rustc, "cpu_model": cpu_model, "cpu_count": os.cpu_count(),
               "parallelism": {key: env.get(key) for key in
                               ("CARGO_BUILD_JOBS", "NEXTEST_TEST_THREADS", "OMP_NUM_THREADS",
                                "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS")}}
    (directory / f"{args.name}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in ("compiled", "fresh", "fingerprints_first_60")}, indent=2))
    print(f"Observed Cargo Compiling lines: {len(compiled)}; Fresh lines: {len(fresh)}")
    print("\n".join(fingerprints))
    print("\n".join(list(tail)[-200 if returncode else -30:]))
    if returncode != 0:
        # The command's own failure is the primary signal; the budget verdict is
        # still recorded in the JSON next to it.
        return returncode
    if args.budget_seconds is None:
        return 0
    if suite_seconds is None:
        print("error: a suite budget was requested but no nextest summary line was found",
              file=sys.stderr)
        return 2
    if budget_exceeded:
        print(f"error: the suite ran for {suite_seconds:.1f}s, above its "
              f"{args.budget_seconds:.0f}s budget; move the slow test into the scheduled "
              f"heavy-tests workflow (CONTRIBUTING.md, CI time budget)", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
