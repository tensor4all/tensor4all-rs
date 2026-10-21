# F4 residual: a callback capability for the phase entry (#857 / #859 Phase 4)

Status: superseded 2026-10-21 by tenferro's caller-held engine model (tenferro-rs #2206); the session/held-session APIs described here were deleted. See explicit-cpu-execution-context.md for the current ownership model.


`#859`'s F4 row asks to "connect scalar/batch callbacks, TCI factor/solve/output routes and the phase
scheduler". The routes (#923) and the scheduler (#924) are in; what is missing is the callbacks: the user's
scalar and batch functions still receive only indices, so a caller whose *evaluation* wants to run numerical work
has to reach for the process-global route from inside a worker lane. This is the design for that last piece.

## 1. Where the capability has to appear

- `PatchEvaluator` (`crates/tensor4all-partitionedtt/src/adaptive_interpolation.rs`) is the only place the user's
  functions are called (`(self.f)(&full_index)`, `batch(&full_indices)`), and it is created inside
  `process_patch`, which already carries the `PatchRoute` (global or lane session).
- Its methods take `&self` and use `RefCell` caches, so a session can be passed **per call** rather than stored:
  `eval(&self, pivot, session)` is expressible, while a session *field* would collide with the route's own
  borrow of the same session.
- Upstream offers no second execution surface from a lane (`tenferro_cpu::PhaseLane` exposes only `session()` and
  `cancelled()`), so the callback shares the lane's session. That is sound because the lane's own numerical calls
  and the callback's are sequential: the pipeline evaluates, then factorizes, then evaluates again.
- In `tensor4all-tensorci` the evaluator is invoked between operations of the session-taking entry, where the
  session is reachable through the route it already threads (`TciLinalg::Session`), so the same reborrow works
  there.

## 2. Design

### 2.1 tensor4all-tensorci: session-aware callbacks on the session route only

The session-taking entry gains session-aware callback variants:

```rust
pub fn crossinterpolate2_with_rng_on_session_with<F, B>(
    f: F,                      // Fn(&MultiIndex, &mut Session<'_>) -> T
    batched_f: Option<B>,     // Fn(&[MultiIndex], &mut Session<'_>) -> Vec<T>
    ...,
    session: &mut Session<'_>,
) -> Result<TCI2OptimizationResult<T>>;
```

Internally one adapter type covers both shapes (`Evaluator::{Plain, Scoped}`), whose `call(index, session)`
either ignores the session or passes the reborrow. Every existing entry keeps `Fn(&MultiIndex) -> T` and calls
through the `Plain` adapter, so no public signature and no behaviour changes.

### 2.2 The interpolation: the callback signature names the capability

`adaptiveinterpolate_on_phase` gains the session-aware forms beside the current one (the F1-F4 precedent: a new
entry rather than a changed signature), and `PatchEvaluator` grows the session-aware path. The session handed to
a callback is the lane's own, so the callback runs in the issuing execution; the plain callbacks keep working
unchanged and are the default for every existing caller.

No ambient or thread-local selector is involved: the capability is a parameter of the call the user writes.

### 2.3 What stays out

- No child execution per callback: upstream exposes none from a lane, and the sequential sharing above is sound.
- No zero-copy or AD-tracked value crossing at the boundary (the issue's non-goals).
- The legacy Hataori entry keeps its callbacks and its route.

## 3. Contracts

- **Sequential, not reentrant**: a callback may use the session freely; it must not expect a *second* concurrent
  view of it, and the session is entered by the phase, not by the callback.
- **Same results**: a session-aware callback that ignores its session must produce exactly what the plain
  callback produces; parity is asserted by the existing projector/value comparison.
- **Worker rules unchanged**: a callback that calls back into the process-global route from a lane still blocks
  there, which is what the dispatch test observes.

## 4. Evidence plan

1. **Capability, not just signature**: a scalar and a batch callback that each run one session operation (a
   contraction of the caller's own tensors through the supplied session) and return a value derived from it, on a
   fixture that splits into several patches. The run must produce the same result as the plain callbacks computing
   the same values without the session.
2. **Dispatch**: the whole run completes **while the process-global backend is held** by another thread - a
   callback that had used the global route would block there. This is the evidence shape F3 and slices 1/3a
   already use, and it is the one that distinguishes "the callback received a session" from "the callback
   received a wrapper it ignored".
3. **Both callback shapes**: scalar-only, batch-only and both supplied, since the batch path is the one that
   bypasses the per-index call.
4. **Error and cancellation**: a callback that fails surfaces as the patch error through the existing path, and a
   lane that returns after `cancelled()` still yields the phase's typed error.
5. **Gates**: fmt, clippy with the repository flags for the touched crates, their whole-crate tests, rustdoc
   `-D warnings`, `check-public-error-docs.py`, the panic audit (the new code must not add panic paths),
   `test-mdbook.sh` and `repository-rules-review.py`.

## 5. Open questions

1. Should the session-aware entry live in `tensor4all-tensorci` at all, or is the interpolation the only consumer
   that needs it? If the latter, the tensorci variant can stay `pub(crate)`-adjacent and the public surface stays
   smaller.
2. Naming: `..._with_session_callbacks` is explicit but long; the repository's convention for the earlier twins
   is a suffix that names the *resource* (`_in`, `_on_session`), which here would read `_on_session_with_scalars`.

## 6. Pre-review of revision 1, and the revision-2 plan

An independent pre-review of this design (before any implementation) accepted the target and rejected the design
as written. It found no unsound operation on the current session route - the dense path really evaluates before it
factorizes - but it falsified the blanket claim that "the same reborrow works there", and it corrected the
evidence plan. The required changes, to be applied before implementing:

1. **Two `Fn`-only boundaries need an explicit bridge**, and they are the reason the design is not mechanical:
   the global pivot search (`GlobalPivotFinder::find_global_pivots` requires `Fn`) and the rook/lazy block
   callback (`matrix_luci_factors_from_blocks` requires `Fn`). A closure that captures a `&mut TciLinalg` and
   invokes a scoped callback is `FnMut`, not `Fn`. Choose either a patch-local `RefCell` borrow of the route with
   each borrow released after the callback, or private context-passing helpers. `fill_tensor` also needs the route
   threaded into it. No public finder-trait change and no second execution is needed.
   Also: `PatchRoute::interpolate` lends its session to tensorci for the whole call, so the interpolation's local
   closures cannot separately capture that session - tensorci must supply the callback's reborrow from its own
   route.
2. **Specify the adapter concretely**: plain and scoped callbacks have different `Fn` signatures, so the internal
   shape is statically dispatched plain/scoped wrappers with an interface taking the route (or the optional
   session), not one generic bound covering both. `TciLinalg::Global` has no session, which the universal
   `call(index, session)` interface assumed. The scoped bounds are higher-ranked in the *per-call* borrow but not
   in the backend lifetime (`Fn(&MultiIndex, &mut Session<'_>) -> T`, never `&'a mut Session<'a>`). `Send + Sync`
   belongs to the phase-facing user callbacks only; taking `&mut Session` as an argument does not cost `Fn + Sync`.
3. **The dispatch oracle is wrong as written.** The compatibility session entry checks active CPU execution
   *before* locking, and an eligible Rayon worker takes the unrelated inline backend, so a lane callback using the
   global route rejects or falls back rather than blocking: "the run completes while the global backend is held"
   proves nothing here, and the existing phase test already passes with a host-only oracle. The revised evidence
   is: both callback shapes must call a supplied-session operation and return a value derived from its output;
   assert the operations really ran (cache hits must not make it vacuous); pair it with a **negative control**
   using the corresponding legacy global operation, asserting the observed rejection rather than an assumed
   timeout; replace the holder's sleep-based readiness with an acknowledgement sent after the lock is acquired;
   and in a multi-worker fixture record that callbacks ran on a worker thread.
4. **Keep the callbacks infallible.** `pending_error` captures a malformed batch length, not a failed numerical
   operation; the parent design already states that current callbacks cannot fail. Define the error test as
   malformed batch output, document that cancellation is observed only between patches (a `Session` has no
   cancellation accessor, and `lane.cancelled()` cannot be reached while `lane.session()` borrows the lane), and
   test the phase primitive's typed cancellation separately. A fallible callback API is a separate contract.
5. **tensorci is a real required consumer seam.** The callbacks must receive the session during the sweeps, site
   fill and global search, not only during pre-TCI sampling, and `partitionedtt` cannot call a `pub(crate)`
   entry from another crate. So exactly one deliberately narrow tensorci entry is exposed; extra scoped public
   sweep methods or finder variants would be speculative.
6. **Rook stays a global numerical route.** Passing the callback its session does not move the surrounding
   rook/block-join numerical work onto that session. Say so, and make the dispatch fixture explicitly
   `PivotSearchStrategy::Full`.
7. **A pre-existing mismatch to handle, not to inherit silently**: the MPI caller
   (`adaptive_interpolation.rs`, in the pmap path) calls `process_patch` without the route argument the sequential
   and Hataori callers pass. Whatever the callbacks do, the "MPI unchanged" promise needs this addressed and an
   MPI-feature compile check.

Nothing here changes the target: a callback capability that lets a caller's evaluation run in the issuing lane's
execution is still the right last piece of F4. It changes how the capability is threaded and how it is proven.
