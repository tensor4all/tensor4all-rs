#!/usr/bin/env python3
"""Resolve downstream M6 package AND feature graphs without workspace unification.

Library fixtures are independent Cargo roots. Benchmark queries select only the
actual benchmark package, including its harness dev-dependencies in a second
graph. No Rust compilation or numerical benchmark is performed here.
"""

import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent.parent
NATIVE = (
    "tensor4all-treetn",
    "tensor4all-partitionedtreetn",
    "tensor4all-treeaci",
    "tensor4all-treetci",
    "tensor4all-treersi",
)
LEGACY = {"tensor4all-simplett", "tensor4all-tensorci", "tensor4all-aci"}
PACKAGE = re.compile(r"^(tensor4all-[\w-]+) v\S+.* features=\[([^]]*)\]")


def verify_tree(tree, required, *, bridge=False, fourier=False, comparisons=False,
                itensorlike=False):
    """Fail on missing roots, unwanted packages, or leaked optional features."""
    packages = {}
    for line in tree.splitlines():
        match = PACKAGE.match(line.strip())
        if match:
            packages.setdefault(match[1], set()).update(filter(None, match[2].split(",")))
    missing = set(required) - packages.keys()
    if missing:
        raise ValueError(f"missing expected packages: {sorted(missing)}")
    allowed = {"tensor4all-simplett"} if bridge else set()
    if comparisons:
        allowed.add("tensor4all-tensorci")
    unwanted = packages.keys() & (LEGACY - allowed)
    if not fourier and "tensor4all-quanticstransform" in packages:
        unwanted.add("tensor4all-quanticstransform")
    if not itensorlike and "tensor4all-itensorlike" in packages:
        unwanted.add("tensor4all-itensorlike")
    if "tensor4all-interpolativeqtt" in packages:
        unwanted.add("tensor4all-interpolativeqtt")
    if unwanted:
        raise ValueError(f"unexpected dependencies: {sorted(unwanted)}")
    bridge_enabled = "simplett-bridge" in packages["tensor4all-treetn"]
    if bridge_enabled != bridge:
        raise ValueError(f"TreeTN simplett-bridge enabled={bridge_enabled}, expected={bridge}")
    if "simplett-transforms" in packages.get("tensor4all-quanticstransform", set()):
        raise ValueError("Fourier benchmark enabled legacy simplett-transforms")
    if bridge and "tensor4all-simplett" not in packages:
        raise ValueError("explicit bridge did not resolve SimpleTT")
    if comparisons and "tensor4all-tensorci" not in packages:
        raise ValueError("legacy comparisons did not resolve TCI")


def check_graph(label, manifest, required, cargo_args=(), **expectations):
    # Check normal/build packages and their activated features. For benchmarks,
    # also include dev edges: examples link the package's dev-dependencies.
    edges = ["normal,build,features"]
    if "tensor4all-benchmarks" in required:
        edges.append("normal,build,dev,features")
    for edge_set in edges:
        command = [
            "cargo", "tree", "--manifest-path", str(manifest), *cargo_args,
            "--edges", edge_set, "--prefix", "none", "--color", "never",
            "--format", "{p} features=[{f}]",
        ]
        tree = subprocess.check_output(command, cwd=ROOT, text=True)
        try:
            verify_tree(tree, required, **expectations)
        except ValueError as error:
            raise ValueError(f"{label} ({edge_set}): {error}\n{tree}") from error
    print(f"check-m6-dependencies: {label} passed", flush=True)


def consumer(directory, dependencies, *, no_default_features=False, bridge=False):
    """Seed a disposable downstream lock from the workspace's pinned resolution."""
    directory.mkdir()
    (directory / "src").mkdir()
    (directory / "src/lib.rs").write_text("// Graph-only downstream fixture.\n")
    shutil.copyfile(ROOT / "Cargo.lock", directory / "Cargo.lock")
    lines = [
        '[package]', 'name = "m6-consumer"', 'version = "0.0.0"',
        'edition = "2024"', 'publish = false', '[workspace]', 'resolver = "2"',
        '[dependencies]',
    ]
    for package in dependencies:
        options = [f"path = {json.dumps(str(ROOT / 'crates' / package))}"]
        features = []
        if no_default_features:
            options.append("default-features = false")
            if package == "tensor4all-treersi":
                features.append("global-defaults")
        if bridge:
            features.append("simplett-bridge")
        if features:
            options.append(f"features = {json.dumps(features)}")
        lines.append(f"{package} = {{ {', '.join(options)} }}")
    manifest = directory / "Cargo.toml"
    manifest.write_text("\n".join(lines) + "\n")
    return manifest


def main():
    (ROOT / "target").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="m6-consumers-", dir=ROOT / "target") as temporary:
        directory = Path(temporary)
        # In particular, the direct TreeTN fixture leaves defaults ON.
        for package in NATIVE:
            manifest = consumer(directory / package, [package])
            check_graph(f"default {package} consumer", manifest, [package, "tensor4all-treetn"])
        manifest = consumer(directory / "combined", NATIVE)
        check_graph("combined M6 consumer", manifest, NATIVE)
        manifest = consumer(directory / "combined-no-default", NATIVE, no_default_features=True)
        check_graph("combined M6 consumer (no default features)", manifest, NATIVE)
        manifest = consumer(directory / "bridge", ["tensor4all-treetn"], bridge=True)
        check_graph("explicit legacy bridge consumer", manifest, ["tensor4all-treetn"], bridge=True)

    for features, expectations in (
        ("", {}),
        ("diagnostics", {}),
        ("fourier", {"fourier": True}),
        ("itensorlike", {"itensorlike": True}),
        ("hdf5", {"itensorlike": True}),
        ("legacy-comparisons", {"bridge": True, "comparisons": True}),
        ("fourier,legacy-comparisons,itensorlike",
         {"fourier": True, "bridge": True, "comparisons": True, "itensorlike": True}),
    ):
        args = ["--locked", "--package", "tensor4all-benchmarks"]
        required = ["tensor4all-benchmarks", *NATIVE]
        if features:
            args += ["--features", features]
        if expectations.get("fourier"):
            required.append("tensor4all-quanticstransform")
        if expectations.get("itensorlike"):
            required.append("tensor4all-itensorlike")
        check_graph(f"benchmarks ({features or 'default M6 kernels'})", ROOT / "Cargo.toml",
                    required, args, **expectations)


if __name__ == "__main__":
    main()
