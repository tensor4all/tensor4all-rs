# Explicit execution engine migration (#1376)

Design and inventory only; the durable record is
[docs/design/explicit-engine-migration.md](../design/explicit-engine-migration.md).
This log records the decisions and what remains unverified.

## Decisions

- **The deliverable is a removal plan, not a context abstraction.** `CpuExecutionContext`
  and `explicit::Session` already exist; the gap is that the implicit twin is still the
  advertised entry point everywhere. The document is therefore organised around four
  testable properties (mandatory parameter, no process-global resolution, no owner-less
  value adoption, one parameter shape) rather than around new types.
- **`ExecutionContext` stays the public token; `explicit::Session` is the inner handle.**
  Rejected: making every function generic over `E: Engine`. It duplicates tenferro's trait
  and cannot carry the CPU/CUDA union (#869) or the context-owned graph/plan cache.
  Rejected: leaving `default_context()` in place and only adding `_in` twins, because that
  is exactly the state the tree is already in after the #859 consumer pass, and it does not
  satisfy "an operation cannot silently choose a backend".
- **The smallest first PR is the fallback removal, not the API addition.** Turning
  `crates/tensor4all-tensorbackend/src/context.rs:1950`'s silent substitution into a typed `NoLiveOwner` is one function, makes
  the contract true for callers that already passed an owned value, and is the only step
  that can reveal a latent bug rather than a compile error. It is issue #1375's finding 2.
- **The inventory is grouped into eight families, not one grep.** A single
  `grep … None` search finds 53 sites and misses five other families (an
  `Option<&..Context>` parameter whose `None` is implicit, implicit entries that construct
  the global context themselves, five implicit twins with no context parameter at all, the
  bindings, and the owner-less value adoption). Each family has its own published command.
- **`GlobalDefault` is explicitly out of scope** and named as such in the document, because
  the issue says "remove the global-defaults surface" and that names a Cargo feature while
  a same-named type in `tensor4all-core` is an unrelated `rtol` default.

## Verification conclusions and constraints

- Every table's count was reproduced at `origin/main@20b42b0d` from a fresh worktree. The
  counts are published next to the commands that regenerate them, so a later reader can
  detect drift instead of trusting them.
- **The first published revision mis-cited §3.9, and the correction method is now a rule
  (§10 rule 3).** Sixteen cited lines in that one table were enclosing-function
  declarations rather than the `with_default_session` call sites they were claimed to be
  (`backend.rs` alone: eight published lines, twelve real sites), one row
  (`native_tensor_from_vec`) was a false positive from a `use` clause, one `matrix.rs` row
  attributed `with_default_session` to `submatrix_argmax`, which is host-only, and eight
  tenferro citations were missing the `crates/` prefix so they did not resolve at the pinned
  revision. All are fixed and re-checked symbol-by-symbol. The earlier claim that every
  citation had been "content checked against the claim" was therefore wrong when written.
- **Unverified: the number of tests and doctests that currently depend on the
  `with_runtime_engine` fallback.** Stage 1 must measure that before flipping the branch;
  the document says so and gives the split-into-two-commits escape hatch if the number is
  large.
- **Unverified: numeric parity once the `is_global_default_cpu` branch disappears.** The
  document specifies the parity harness and the matched-thread-count requirement but the
  measurement belongs to stage 3.
- **One structural blocker is named that the first pass missed:**
  `crates/tensor4all-treersi/src/engine.rs:118` routes the explicit CUDA LUCI arm through the *global* CPU
  context, so `global-defaults` removal would silently regress an `_in` entry point. It is
  a prerequisite work item for stage 3, not a follow-up.
- `tensor4all-core` cannot currently be compiled against an explicit-only
  `tensorbackend` (`crates/tensor4all-core/Cargo.toml:58-61` hard-requires `global-defaults`), so stage 3
  starts by moving those re-exports, not by deleting the feature.

## Review

review: glm-5.3-flash changes (pre-implementation plan review, `ollama-cloud/glm-5.3-flash`);
findings adopted, all re-verified against source. The blocker was that the inventory
covered one family and would have passed its own acceptance bar while failing the issue.
No post-implementation review is required for this change: the diff adds only documentation
(`AGENTS.md` "Exempt without asking" — a diff made only of documentation).
