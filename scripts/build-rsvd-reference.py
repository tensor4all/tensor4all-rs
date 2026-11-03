#!/usr/bin/env python3
"""Build the standalone Lanczos oracle from the benchmark's exact Cargo artifacts.

No tensor4all library or workspace manifest gains a direct faer dependency.
The reference executable consumes the identical column-major matrix emitted
by benchmark_randomized_factorization --decompositions.
"""
import json
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    command = ["cargo", "build", "--locked", "--release", "-p", "tensor4all-benchmarks",
               "--example", "benchmark_randomized_factorization", "--message-format=json"]
    result = subprocess.run(command, cwd=root, text=True, stdout=subprocess.PIPE, check=True)
    artifacts = {}
    for line in result.stdout.splitlines():
        item = json.loads(line)
        if item.get("reason") != "compiler-artifact":
            continue
        name = item["target"]["name"]
        if name in ("faer", "dyn_stack"):
            paths = [Path(p) for p in item["filenames"] if p.endswith(".rlib")]
            if len(paths) != 1 or name in artifacts:
                raise RuntimeError(f"ambiguous reference artifact: {name}")
            artifacts[name] = (paths[0], item["package_id"])
    if set(artifacts) != {"faer", "dyn_stack"}:
        raise RuntimeError("Cargo did not report the complete Lanczos reference dependency closure")
    deps = artifacts["faer"][0].parent
    output = deps.parent / "rsvd-lanczos-reference"
    command = ["rustc", "--edition=2021", "-O", str(root / "benchmarks/rust/reference_lanczos.rs"),
               "-L", f"dependency={deps}", "-o", str(output)]
    for name, (path, _) in artifacts.items():
        command.extend(["--extern", f"{name}={path}"])
    subprocess.run(command, cwd=root, check=True)
    print(json.dumps({"binary": str(output), "artifacts": {
        k: {"path": str(v[0]), "package_id": v[1]} for k, v in artifacts.items()}}, indent=2))


if __name__ == "__main__":
    main()
