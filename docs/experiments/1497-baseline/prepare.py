#!/usr/bin/env python3
"""Prepare a disposable, detached baseline checkout; no builds or runs."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent


def git(checkout, *args):
    return subprocess.check_output(["git", "-C", str(checkout), *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--ir-source", type=Path, required=True)
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    ir_source = args.ir_source.resolve()
    protocol = json.loads((HERE / "protocol.json").read_text())
    if git(checkout, "rev-parse", "HEAD") != protocol["baseline_commit"]:
        raise SystemExit("The disposable checkout must be at the registered baseline")
    branch = subprocess.run(["git", "-C", str(checkout), "symbolic-ref", "-q", "HEAD"],
                            capture_output=True)
    if branch.returncode == 0:
        raise SystemExit("Use a detached disposable worktree, not an active branch")
    if git(checkout, "status", "--porcelain", "--", "crates", "benchmarks"):
        raise SystemExit("Numerical sources in the disposable checkout are not clean")
    if git(ir_source, "rev-parse", "HEAD") != protocol["upstream_ir_commit"]:
        raise SystemExit("The IR source checkout differs from the registered revision")
    if git(ir_source, "status", "--porcelain", "--", "g_ir_patching"):
        raise SystemExit("The pinned IR package contains local modifications")
    package = checkout / "target/1497-ir-harness"
    if package.exists():
        raise SystemExit("The diagnostic harness directory already exists; use a fresh worktree")
    patches = [HERE / "profiling.patch", HERE / "lazy-reader.patch"]
    subprocess.run(["git", "-C", str(checkout), "apply", "--check",
                    *map(str, patches)], check=True)
    destination = checkout / "docs/experiments/1497-baseline"
    destination.mkdir(parents=True, exist_ok=True)
    for name in ["protocol.json", "profiling.patch", "lazy-reader.patch", "run.py",
                 "ir-Cargo.toml", "ir-Cargo.lock"]:
        shutil.copy2(HERE / name, destination / name)
    package.mkdir(parents=True)
    shutil.copytree(ir_source / "g_ir_patching/src", package / "src")
    shutil.copy2(HERE / "ir-Cargo.toml", package / "Cargo.toml")
    shutil.copy2(HERE / "ir-Cargo.lock", package / "Cargo.lock")
    subprocess.run(["git", "-C", str(checkout), "apply", *map(str, patches)], check=True)
    print("Prepared temporary diagnostic sources; build through ai-run before measuring.")


if __name__ == "__main__":
    main()
