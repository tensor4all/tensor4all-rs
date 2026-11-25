# Runtime diagnostics configuration (#1387)

## Decisions

- Backend native-einsum diagnostics and core profiling each own one immutable
  configuration, initialized on first use. The core snapshot also includes the
  already cached pairwise-contraction flag. Flags keep their original presence
  and Unicode semantics; malformed numeric limits keep their defaults.
- Remove the undocumented per-einsum engine/pool/allocator cleanup switches.
  Resource owners already have explicit context reset and allocator-release
  APIs; a process-global diagnostic variable should not decide their lifecycle.
  Keeping test-only copies of the removed production branches would serve no
  existing caller or test.
- Unit tests parse supplied settings directly and enable core profiling through
  scoped thread-local overrides. They no longer mutate the process environment;
  nested overrides and unwinding restore the prior state.

## Verification conclusions and constraints

- The intended guarantee is one environment snapshot per crate and explicit
  cleanup ownership. No kernel, numerical tolerance, or contraction algorithm
  changes, and no measured speedup is claimed.
- Removed cleanup branches were reviewed for coverage impact. The underlying
  context reset and allocator-release APIs retain their own tests in the
  backend's context and memory modules. Trace-limit checks move from the bridge
  helper smoke test to configuration tests with explicit inputs and defaults.
- Focused core configuration, Krylov, contraction, and tensor tests pass, as do
  the backend unit tests (one pre-existing measurement-only test stays ignored).
  Fresh-process checks also pass with tracing/profiling enabled and with the
  removed cleanup variables set. Clippy, explicit-only and no-feature build
  checks, formatting, and deterministic repository checks pass. The scoped
  performance audit finds no new hot-loop allocation or direct environment
  access; remaining backend thread-environment reads belong to construction.
