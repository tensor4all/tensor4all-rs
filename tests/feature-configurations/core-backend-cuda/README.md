# core + backend CUDA feature-configuration check (#869)

Isolated verification package for the configuration of
[tensor4all-rs#869](https://github.com/tensor4all/tensor4all-rs/issues/869):
`tensor4all-tensorbackend` is built **with** `tenferro-cuda` while
`tensor4all-core` is built **without** it, so `ExecutionContext::Cuda` exists but
core has no CUDA support compiled in.

core must build in that configuration, keep serving CPU contexts, and reject a
CUDA context with `IdxTensorError::UnsupportedExecutionContext` before any
runtime initialisation, transfer, upload, reduction, or output mutation.

## Why it lives here

This package declares its own `[workspace]` (resolver 2) and is deliberately not
a member of the repository workspace:

- an ordinary `cargo build`/`cargo test` of the workspace must not acquire a CUDA
toolkit requirement, and `tensor4all-core`'s dev-dependencies must not enable
  `tenferro-cuda`;
- another member's features could unify `tensor4all-core` onto a CUDA-enabled
  build within the same workspace, which would stop this package from testing the
  mixed configuration.

No repository workspace manifest, feature list, or dev-dependency is changed by
this package.

## Requirements

- A machine with a visible CUDA device and a CUDA toolkit (`nvcc`).
- The pinned tenferro revision from the repository `Cargo.toml` (see this
  package's `Cargo.toml`).
- Rust toolchain of the workspace.

## Run

```bash
cd tests/feature-configurations/core-backend-cuda
export CARGO_TARGET_DIR="$PWD/../../../target/issue-869-consumer"

# 1. Effective feature set: core without CUDA, backend with CUDA.
cargo metadata --format-version 1 --locked > /tmp/869-metadata.json
python3 verify_features.py < /tmp/869-metadata.json

# 2. The mixed build compiles.
cargo check -j 16 --locked --tests

# 3. core rejects a real CUDA context with the typed error.
cargo test -j 16 --locked --test rejection -- --test-threads=1 --nocapture
```

The verifier exits non-zero, and prints the offending feature set, if the
workspace root is not this directory, if `tensor4all-core` resolves to another
checkout, if core has `tenferro-cuda` or dependency defaults enabled, or if the
backend lacks `tenferro-cuda`. Its negative case can be checked by feeding a
metadata file in which core has an extra CUDA feature:

```bash
python3 - <<'PY' > /tmp/869-metadata-negative.json
import json
data = json.load(open("/tmp/869-metadata.json"))
for node in data["resolve"]["nodes"]:
    if node["id"].startswith("path+file:///") and "tensor4all-core" in node["id"]:
        node["features"].append("tenferro-cuda")
json.dump(data, open("/tmp/869-metadata-negative.json", "w"))
PY
python3 verify_features.py < /tmp/869-metadata-negative.json || echo "verifier rejected the negative case"
```

## Coverage

`tests/rejection.rs` covers materialized-storage and eager-storage validation,
decision readback, scaling (`0`, `1`, `2`), norm, `from_dense_in` (f64 and
`Complex64`), `ones_in` (including empty and zero-sized requests), `factorize_in`,
`factorize_full_rank_in`, `src_error_estimate_in`,
`factorize_probe_batch_incremental_in`, `qr_with_in`, `svd_with_in`, the typed
cause on the `FactorizeError`/`QrError`/`SvdError` source chains, the absence of
CUDA eager-runtime initialisation after every rejection group, unchanged operands
after a rejection, and the same build's CPU context still reconstructing a
factorization.

Moving the support check below `validate_context`'s storage branch makes five of
the seven tests fail (they then observe the generic residency error instead of the
typed one), so the fixtures and assertions pin the rejection boundary rather than
passing vacuously.

CPU CI cannot build this configuration, so these results are evidence from a
machine with a CUDA device; the README records what must be re-recorded
(toolchain, driver, toolkit, exact commands) when the check is repeated.
