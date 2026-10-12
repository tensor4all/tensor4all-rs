# F4: the adaptive-interpolation worker path on the phase/lane execution seam (#857 / #859 Phase 4)

`#857` is closed as *not planned*, superseded by `#859` Phase 4: "child sessions from tenferro-rs `#2044` plus a
phase scheduler on the context's own pool". This is revision 2 of the design; revision 1 proposed a new
`with_child_context` bridge over `CpuExecSession::child_execution()` and an external-domain fan-out, and an
independent pre-review falsified that premise on the source (see §7). The corrected target is the **existing
phase/lane seam**: `HeldSession::phase` / `PhaseLane`, which tenferro-rs `#2055` added and U2-concrete released.

## 1. What is true in this checkout (verified)

- **The worker path is off by deliberate default.**
  `crates/tensor4all-partitionedtt/Cargo.toml` keeps `default = ["tenferro-cpu-faer"]` and documents why:
  "its worker callbacks enter the tenferro CPU session from Rayon workers, which tenferro-rs #2004 rejects as
  `SessionEntryError::Contended` until an execution can hand its child ticket to workers". The feature compiles
  and its tests sit under `#[cfg(feature = "adaptive-hataori-rayon")]`.
- **The substrate exists and is ours.** `crates/tensor4all-tensorbackend/src/explicit.rs` exposes
  `HeldSession::phase` with worker-local `PhaseLane` session views; upstream `held_session/phase.rs` broadcasts a
  multi-worker phase onto the backend's **existing pool**, gives the one-worker case an inline lane under the
  parent admission, and lends the parent's pool to the phase.
- **The interpolation fans out through an external Hataori domain**
  (`hataori::map_in(domain, LocalMode::Outer, wave, |patch| process_patch(...))`), and `process_patch` reaches the
  numbers through `crossinterpolate2_with_rng` (tensor4all-tensorci) plus the core LUCI/solve path.
- **`tensor4all-tensorci` uses no explicit session at all**; its numerical work goes through the compatibility
  path, so a worker callback is either rejected or rerouted onto the legacy inline fallback.
- **Measured: today's worker path does not use the global session at all.** With the process-global backend
  held, a 1-worker hataori domain running the split-forcing interpolation fixture (the one that actually reaches
  TCI) completed in 6.9 s: the worker callbacks therefore run on the *compatibility frontend's* Rayon-worker
  fallback, which constructs an unrelated one-thread backend and discards the inherited owner. The existing
  regression test is green for that reason - it verifies values, not routing - so the acceptance test has to
  observe execution, exactly as `#857` says.
- **`CpuExecSession::child_execution()` alone is not a fix.** `child_execution().backend()` preserves the
  inherited owner, but every route our context offers then re-blocks or reroutes: `with_concrete_session` rejects
  **any** Rayon worker before consulting the backend, the canonical `with_session` selects an unrelated inline
  backend on a worker, and the F1/F3 context entries reach that rejecting entry. A wrapped child backend is
  therefore not worker-capable, and one shared child context cannot even run two callbacks at once (its backend
  mutex spans the session callback, and eager execution takes a single owner lock).

## 2. Design

### 2.1 Execution seam: the phase, not a new child-context bridge

The fan-out runs inside **one phase** of the caller's held session. Each pool worker takes a `PhaseLane` and its
session view, reusing that lane across the work items it runs, so worker-local execution state is per lane and not
per callback. A worker's numerical work receives the lane's session, not a context wrapper: the F1 operations
already take an explicit session, and the new boundaries in §2.2 take one too.

Scheduling is the requirement `#859` states - **a work queue with one loop per pool worker**, work items instead
of spawned Rayon jobs, and no second pool. The phase lends the backend's existing pool; the queue decides what a
lane runs next. `hataori::map_in` keeps its external domain for the legacy path (§2.4) and is not the mechanism
of the new path.

Entry rules, from the review's deadlock analysis: for a fresh root entry, check reentry **before** locking, clone
the configured backend briefly, release the context mutex, then open issuance on the clone. For an existing
session, derive the phase from that session instead of opening a second root. "Reentrant under the same owner"
never means permission for a same-thread nested canonical session.

### 2.2 Numerical boundaries in tensor4all-tensorci

A TCI2 twin that takes the lane's session, threaded to the operations that really execute:

| Work | Where |
| --- | --- |
| Full-search LUCI factor assembly | `tensorci2.rs` (dense assembly path) |
| Rook factor reconstruction and solve (lazy blocks) | `tensorci2.rs`, `core/src/matrix_luci.rs`, `core/src/matrixluci/factors.rs` |
| Per-sweep site interpolation solve | `tensorci2.rs` |
| Final one-site cleanup LUCI | `tensorci2.rs` |

The existing context-taking LUCI entry covers dense assembly only, so the **rook/lazy-block factor solve must get
its own session-taking form** or be excluded explicitly.

**Landed in slice 1:** the core seam `Linalg` gained a `Session(&mut Session)`
variant, its helpers take `&mut Linalg` (a session cannot be copied the way a context can), and
`matrix_luci_factors_from_matrix_owned_on_session` is the session-taking twin of the dense owned entry. The
lazy/rook path is **not** threaded yet, so it stays on the global route. `tensor4all-tensorci` now carries an
internal `TciLinalg` route through its optimizer, sweeps and site-tensor fill, with the public entries delegating
to the global route as before and a new `crossinterpolate2_with_rng_on_session` entry taking a session the caller
already entered. Tensor4all's own tests (core 509, tensorbackend 278+185, tensorci 92+integration) pass, and the
new session entry completes while the process-global backend is held.

**Two structural obstacles the consumer threading has to solve first** (found while attempting it):

1. The dense factor assembly in the pivot-search path lives in `update_pivots`, which is reached through the
   **public `GlobalPivotFinder` trait** - so a session cannot be added to its signature without either changing
   that trait or carrying the session in the finder value itself.
2. `TensorCI2::sweep1site`, `sweep2site` and `make_canonical` are **public methods**. Threading a session through
   them either adds public twins (API bloat) or needs private `_on_session` twins that the public methods call
   with the global route - the second is the intended answer, and the finder case is the same shape: a session
   carried by the finder value rather than by an ambient selector, so no public trait or method signature moves.

Deliberately **out of scope** because they are host-only: `to_tensor_train`'s clone of host site tensors,
`tensor3_from_data`'s typed host construction, the default search's `TTCache` environment products (direct scalar
loops), and projector/cache bookkeeping. Error adaptation must preserve typed failures: the current LUCI adapters
stringify entry failures, and the site solve flattens errors, so an end-to-end typed-rejection assertion needs
those adapted rather than inherited.

### 2.3 Publication, after issuance closes

Results must be usable under the issuing context once the fan-out has ended. The remaining global publication is
outside TCI2: the interpolation's final assembly, the simplett bridge's `IdxTensor::from_dense` publication, and
the dense embedding / structured-selector construction, all of which adopt into `default_eager_ctx`. The design
therefore publishes **after** the issuance closes, as portable concrete storage, or adopts explicitly into the
parent context outside the held region - decided per site, with the structured-selector case named rather than
implied.

### 2.4 Callback capability, legacy and MPI paths

The callbacks currently receive indices only, so passing a context into `process_patch` would not give the user's
callback the execution capability. The context-taking entry therefore carries **context-aware scalar and batch
callback signatures that name the supplied capability**; no ambient/TLS selector, and the capability comes from
the caller, not from Hataori worker ids (Hataori owns scheduling, not the tenferro witness).

The existing entries keep their signatures and their external-domain scheduling; the legacy path continues through
a no-session adapter, so the MPI entry stays unchanged and its callbacks cross ranks as host core data. An
explicit MPI route later must issue per rank (no child handle crosses the wire) and converge entry failures
collectively before any blocking collective.

"A callback error" is defined as the paths that already exist: a malformed batch length is an error, while the
current evaluators return `T`/`Vec<T>` and cannot fail.

### 2.5 Feature restoration

Restoring `adaptive-hataori-rayon` to `tensor4all-partitionedtt`'s defaults is an **implementation step gated on
worker evidence**, not a current fact, and the same change must cover more than one and two workers.

### 2.6 Landed: the phase consumer (slice 3a)

`adaptiveinterpolate_on_phase` runs each wave's patches as work items on the lanes of a held session: one loop per
pool worker, one session view per work item, per-patch seeds derived from the path as the Hataori entry derives
them, and outcomes written back per index so the wave's result does not depend on scheduling. `process_patch`
carries a `PatchRoute` (global or session) instead of a context, and the patch's TCI runs through the
session-taking entry from slice 1. Every existing entry - sequential, Hataori, MPI - keeps the global route and
its signature.

Evidence in `crates/tensor4all-partitionedtt/tests/phase_execution.rs`: parity of projectors and dense values with
the sequential entry for one and two lanes, the wave's evaluations distributed across **more than one** pool
worker, and the phase completing **while the process-global backend is held** (a bounded hold of 20 s against a
~1 s phase, so a wrong route fails the test instead of hanging it).

Not covered, and therefore open: the design's third oracle (an unrelated *non-waiting* entry must be refused while
the phase holds its reservation). In this configuration an unrelated entry **waits**, so a refusal test would
assert the wrong behaviour; the arbiter's non-waiting path is what that oracle needs. The callback-capability
signatures of §2.4 are also not in this slice: the callbacks still receive only indices, and the lane session is
what carries the capability.

## 3. Contracts

- **Per-lane execution state**: one held phase per fan-out, one lane/session per pool worker, reused across work
  items; simultaneous worker entry is enforced, not merely observed.
- **Inherited admission**: a lane's session is admitted under the issuing owner and never waits for the parent
  reservation; an unrelated non-waiting entry fails while the parent is held.
- **Lifecycle**: a callback error, a panic, or a cancelled phase releases admission; a stale lane or child handle
  is rejected typed rather than admitted as a fresh owner.
- **Unchanged**: public signatures of the existing entries, the compatibility frontend and its worker fallback,
  the MPI path, and every host-only operation listed in §2.2.

## 4. Slices (each merged and verified before the next)

1. **tensor4all-tensorci numerical sessions** (§2.2): the TCI2 twin plus the rook/lazy-block factor solve, with
   dispatch evidence (the same held-global-backend pattern F3 used) and typed error adaptation. Needs no
   scheduler, so it is the first independently verifiable slice.
2. **Publication after issuance** (§2.3), including the structured-selector path.
3. **The phase consumer and work-queue scheduler** (§2.1, §2.4) with the context-aware callback signatures.
4. **Feature restoration and regression coverage** (§2.5): one, two and more workers; the routing evidence below.

## 5. Evidence plan (with the corrected routing oracle)

1. **Routing**: hold the parent reservation; require an **unrelated non-waiting entry to fail** while inherited
   lanes successfully execute. An overlapping unrelated *active* reservation cannot coexist with the parent under
   normal admission, so revision 1's oracle was impossible. A queued unrelated waiter, if used, must verify that
   lanes bypass it without the parent being released.
2. **Simultaneous entry** enforced before either worker is released, asserting numerical execution while the
   issuance is active - not callback completion, and not two results that a serialized shared context could also
   produce.
3. **Negative controls**: a non-child clone that must fail, and a stale child/lane that must be rejected typed.
4. **Global routing** in a separate test with the global backend held (F3's pattern), accounting for the retained
   eager publication of §2.3.
5. **Returned tensors** exercised only after the issuance ends, through the advertised parent-context operation,
   not merely `to_vec` or a legacy contraction.
6. **Coverage of the numerical paths**: full and rook paths, search-time and final solves, and the zero/exact
   shortcuts.
7. **Fixtures that actually reach TCI**: the existing one-site worker test never reaches it, and the split
   regression test checks results rather than routing, so both can pass with the compatibility fallback.
8. **Gates**: fmt, clippy with the repository flags, workspace tests with the feature enabled (the command CI
   runs), affected doctests, the MPI feature compiles, and the repository's own scripts.

## 6. Non-goals

- Zero-copy value transfer at the boundary, and transporting AD-tracked tensors across it.
- Deleting the legacy compatibility fallback, removing the global defaults, or migrating other applications.
- Promising dense F1 support for arbitrary structured/diagonal `IdxTensor` contraction: the F1 session routes
  reject structured operands and narrow dtypes, and `IdxTensor::validate_context` requires exact runtime identity
  only for eager storage, so mismatch means incompatible allocation/provider/readiness or eager ownership, not a
  universal context-id rule.

## 7. What the pre-review corrected (revision 1 -> 2)

- **Target**: the phase/lane seam exists; `child_execution()` is not the mechanism, and keeping `hataori::map_in`
  as the new path would have been a scope change rather than a shortcut.
- **Seam**: a wrapped child backend is not worker-capable; the lane's session is what operations must receive.
- **Concurrency**: one shared child context cannot run callbacks simultaneously (backend mutex spanning the
  session callback; a single eager owner lock), so the design needs per-lane state.
- **Numerical scope**: the TCI2 twin alone is neither sufficient (rook/lazy-block solve, site solve, cleanup) nor
  entirely necessary (host-only operations).
- **Publication**: final publication happens after issuance closes, and the interpolation's assembly, the simplett
  bridge and the structured-selector construction are part of it.
- **API**: the callback capability must be named in the new signatures; MPI keeps a no-session adapter.
- **Evidence**: the routing oracle had to change (an active unrelated reservation cannot overlap), and the
  fixture/coverage gaps above were listed.
- **Facts**: `adaptive-hataori-rayon` is *not* currently a default feature; revision 1 said it was, from a grep
  snippet rather than the file.

## 8. Open questions

1. Rook/lazy-block factor solve: a session-taking form in core, or excluded with the reason recorded? The slice-1
   dispatch test decides which paths actually execute in the split-interpolation workload. (Still open: the dense
   path is threaded, the lazy path is not.)
2. The work-queue scheduler's shape: a small crate-local queue over the phase's pool, or an extension of the
   phase API upstream. Upstream `#2055` already lends the pool, so the queue is expected to be ours.
3. Whether the context-aware callbacks should be a new entry beside the old one (the F1-F3 precedent) or a
   trait/`Fn` adapter; the precedent favours a new entry.
