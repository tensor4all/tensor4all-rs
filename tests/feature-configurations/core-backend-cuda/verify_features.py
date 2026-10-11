#!/usr/bin/env python3
"""Prove the mixed feature configuration of #869 from `cargo metadata` output.

Reads `cargo metadata --format-version 1 --locked` JSON on stdin and fails
unless

* this package is its own workspace root (so that no other member's features can
  unify core onto a CUDA-enabled build, and so ordinary workspace tests never
  need the CUDA toolkit),
* the single `tensor4all-core` is built without `tenferro-cuda` and without its
  dependency defaults, and
* the single `tensor4all-tensorbackend` is built with `tenferro-cuda`.

Effective features are read from `resolve.nodes`, not from the declared feature
lists. See README.md for the invocation and the negative case.
"""

import json
import pathlib
import sys

CORE = "tensor4all-core"
BACKEND = "tensor4all-tensorbackend"
CUDA = "tenferro-cuda"
PROVIDER = "tenferro-cpu-faer"


def fail(message: str) -> None:
    print(f"verify_features: FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def unique_package(data: dict, name: str) -> dict:
    matches = [package for package in data["packages"] if package["name"] == name]
    if len(matches) != 1:
        fail(f"expected exactly one {name}, found {len(matches)}")
    return matches[0]


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError as error:
        fail(f"stdin is not cargo metadata JSON: {error}")

    here = pathlib.Path(__file__).resolve().parent
    root = pathlib.Path(data["workspace_root"]).resolve()
    if root != here:
        fail(
            f"workspace_root is {root}, expected {here}: this package joined another "
            "workspace, so feature unification is no longer isolated"
        )

    core = unique_package(data, CORE)
    backend = unique_package(data, BACKEND)
    expected_core = here / ".." / ".." / ".." / "crates" / CORE / "Cargo.toml"
    if pathlib.Path(core["manifest_path"]).resolve() != expected_core.resolve():
        fail(f"{CORE} resolves to {core['manifest_path']}, expected {expected_core}")

    resolved = {
        node["id"]: set(node.get("features") or []) for node in data["resolve"]["nodes"]
    }
    core_features = resolved.get(core["id"], set())
    backend_features = resolved.get(backend["id"], set())

    if PROVIDER not in core_features:
        fail(f"{CORE} is missing the {PROVIDER} feature: {sorted(core_features)}")
    if CUDA in core_features:
        fail(f"{CORE} has {CUDA} enabled, so the mixed configuration is not tested")
    if "default" in core_features:
        fail(f"{CORE} was built with dependency defaults: {sorted(core_features)}")
    if CUDA not in backend_features:
        fail(f"{BACKEND} is missing {CUDA}: {sorted(backend_features)}")

    print("verify_features: OK")
    print(f"  workspace_root:    {root}")
    print(f"  {CORE}: {sorted(core_features)}")
    print(f"  {BACKEND}: {sorted(backend_features)}")


if __name__ == "__main__":
    main()
