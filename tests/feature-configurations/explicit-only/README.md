# Explicit-only feature-configuration check (#1403)

Isolated test package for the configuration a downstream crate gets from the
default features since
[tensor4all-rs#1403](https://github.com/tensor4all/tensor4all-rs/issues/1403):
`tensor4all-aci`, `tensor4all-core`, `tensor4all-itensorlike` and `tensor4all-tensorbackend` built
with the platform CPU provider and **without** `global-defaults`, so there is no
process-global execution context.

## Why it lives here

Inside the repository workspace every library's dev-dependencies enable
`tensor4all-core/global-defaults` (tests, doctests and benches still use the
context-free API), and Cargo unifies features across one build. A
`#[cfg(not(feature = "global-defaults"))]` test in a workspace crate therefore
never compiles in any test build. This package declares its own `[workspace]`
(resolver 2), so only the dependencies listed in its `Cargo.toml` decide the
features, exactly as for a downstream crate.

No repository workspace manifest, feature list, or dev-dependency is changed by
this package. Its `Cargo.lock` must resolve every registry and git package to an
entry of the root lock (`scripts/check-single-versions.sh`).

## Run

```bash
cd tests/feature-configurations/explicit-only
CARGO_TARGET_DIR="$PWD/../../../target/explicit-only" cargo test --locked --test explicit_only
```

CI runs the same command in the Lint job of `.github/workflows/CI_rs.yml`.

## Coverage

`tests/explicit_only.rs` checks that

- the linked `tensor4all-core` has no global route (`legacy::GLOBAL_DEFAULTS`
  is `false`), so the other tests cannot pass on the wrong configuration;
- core's `legacy` bridges and `with_runtime_engine` on an owner-less runtime
  refuse with `ExecutionContextEntryError::NoContext` without running anything;
- a context-free construction from host data refuses with the typed
  `IdxTensorError::EngineEntry { source: NoContext }`, and a context-free
  contraction works on values whose context is alive and refuses once it is gone;
- `IdxTensor::with_owning_cpu_engine` lends the creating context's engine;
- `TensorTrain::norm_squared`/`norm` of a chain built in a context run on that
  context and match a hand-computed value, and a chain whose context was dropped
  refuses with `NoContext` instead of falling back to a host loop;
- a chain of host-only storage (`IdxTensor::from_storage`) refuses already at the
  read step that packs its sites (`to_vec` needs a runtime), so there is no host
  computation a norm fallback could use.
