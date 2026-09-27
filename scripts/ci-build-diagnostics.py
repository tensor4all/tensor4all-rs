#!/usr/bin/env python3
"""Record bounded Cargo diagnostics and build/post-build elapsed time in CI."""

import argparse
from collections import deque
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
            if "fingerprint" in clean and len(fingerprints) < 60:
                fingerprints.append(clean)
            if clean.lstrip().startswith("Compiling "):
                compiled.append(clean.strip())
            if clean.lstrip().startswith("Fresh "):
                fresh.append(clean.strip())
        returncode = process.wait()
    elapsed = time.monotonic() - started
    summary = {"command": command, "elapsed_seconds": elapsed,
               "build_finished_seconds": build_finished,
               "post_build_seconds": None if build_finished is None else elapsed - build_finished,
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
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
