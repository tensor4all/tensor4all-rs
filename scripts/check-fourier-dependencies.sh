#!/usr/bin/env bash
# Check the graph seen by a downstream user. Pass --run to also run the
# Fourier layout tests and the small patched example in that configuration.
set -euo pipefail

cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
mkdir -p target
consumer=$(mktemp -d "$PWD/target/fourier-consumer.XXXXXX")
trap 'rm -rf -- "$consumer"' EXIT

# Seed registry and git resolutions from the workspace lock. Cargo updates
# this disposable copy for the consumer package and its selected dependencies.
cp Cargo.lock "$consumer/Cargo.lock"
cat > "$consumer/Cargo.toml" <<'EOF'
[package]
name = "fourier-consumer"
version = "0.0.0"
edition = "2024"
publish = false

[workspace]
resolver = "2"

[profile.dev]
debug = 0

[features]
default = ["partitioned"]
partitioned = ["tensor4all-quanticstransform/partitioned"]

[[bin]]
name = "patched_fourier"
path = "../../crates/tensor4all-quanticstransform/examples/patched_fourier.rs"

[[test]]
name = "multiaxis_fourier"
path = "../../crates/tensor4all-quanticstransform/tests/multiaxis_fourier.rs"

[dependencies]
num-complex = "0.4"
tensor4all-core = { path = "../../crates/tensor4all-core", default-features = false, features = ["tenferro-cpu-platform"] }
tensor4all-treetn = { path = "../../crates/tensor4all-treetn", default-features = false, features = ["tenferro-cpu-platform"] }
tensor4all-partitionedtreetn = { path = "../../crates/tensor4all-partitionedtreetn", default-features = false, features = ["tenferro-cpu-platform"] }
tensor4all-quanticstransform = { path = "../../crates/tensor4all-quanticstransform", default-features = false, features = ["tenferro-cpu-platform"] }
EOF

manifest="$consumer/Cargo.toml"
tree=$(cargo tree --manifest-path "$manifest" --edges normal,build,features --prefix none --color never)
if grep -E '^(tensor4all-simplett v|tensor4all-treetn feature "simplett-bridge")' <<< "$tree"; then
    echo "check-fourier-dependencies: SimpleTT or its TreeTN bridge is enabled" >&2
    exit 1
fi
echo "check-fourier-dependencies: the patched Fourier build has no SimpleTT or TreeTN bridge"

if [[ ${1:-} == --run ]]; then
    build=(--locked --manifest-path "$manifest" --target-dir "${CARGO_TARGET_DIR:-target}")
    cargo test "${build[@]}" --test multiaxis_fourier
    cargo run "${build[@]}" --bin patched_fourier
fi
