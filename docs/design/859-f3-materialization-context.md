# F3 step 3d: a caller-owned execution context for the TCI materialization solve (#670 / #859 F3)

Short design for the target the corrected inventory named: after `docs/design/859-f3-g0-stage-budget.md` §2
withdrew the readout claim, the one call on this stage's path with a demonstrated caller that still reaches the
process-global canonical session is the materialization's nonzero full-pivot solve.

## 1. The call site (verified)

- `tensor4all-treetci/src/materialize.rs`: `site_tensor_with_parent` ends in
  `T::solve_right_full_piv_lu(&pi1_values, rows, cols, &p_values, p_rows, cols)`; the zero-pivot early return
  above it (`materialize.rs:287-295`) is **materialization** logic, not trait logic.
- That trait method is implemented by the `impl_full_piv_lu_scalar!` macro
  (`tensor4all-tensorbackend/src/backend.rs`) for `f32`, `f64`, `Complex32`, `Complex64`, and its body ends in
  `with_default_session(|session| pivot_tensor.full_piv_lu_solve(&lhs_tensor, session))` - the
  **process-global canonical session**. That is the resource this step moves.
- The materialization runs **once per pivot search** (`globalpivot.rs`, `search_with_readout` → `to_treetn`) and
  **once for the final tree** (`api.rs`).
- **Retained, not fixed here:** publishing the site tensors (`IdxTensor::from_dense` →
  `from_native_with_axis_classes` → `default_eager_ctx`) consults the process-global eager runtime for adoption.
  That is retention/wrapping rather than another solve or contraction, and it stays out of scope; this step
  therefore does **not** advertise whole-materialization independence, and the dispatch test warms that runtime
  before holding the global backend.

## 2. Scope

1. **A context-taking solve on the trait.** `FullPivLuScalar` gains a **required**

   ```rust
   fn solve_right_full_piv_lu_in(
       context: &CpuExecutionContext,
       lhs_values: &[Self], lhs_rows: usize, lhs_cols: usize,
       pivot_values: &[Self], pivot_rows: usize, pivot_cols: usize,
   ) -> Result<Vec<Self>, BackendLinalgError>;
   ```

   implemented in the macro's four expansions. No default: `BackendLinalgScalar` is bounded by the upstream
   **sealed** `TensorScalar`, so no downstream implementor can exist and a delegating default would only
   establish a silent context-ignoring path for a capability that has no possible user. The macro body is
   factored so the only difference between the two routes is how the session is obtained, preserving the
   legacy sequence exactly: non-square pivot validation → LHS/pivot compatibility validation → the two ordinary
   (non-conjugating) column-major transposes → pivot tensor → LHS tensor → session → dtype extraction → result
   transpose. Preprocessing stays outside session entry so invalid inputs keep validation precedence.
2. **Error adaptation.** The legacy adapter keeps its present behaviour (stringified backend failure). The new
   route preserves `CpuExecutionContextError::SessionEntry` and its tenferro cause inside
   `BackendLinalgError`'s source rather than flattening it to text, so a worker rejection stays typed. Callback
   solve errors remain errors; no retry and no fallback. `Session::backend()` is the accessor for the session
   the backend trait expects.
3. **Internal plumbing.** `Option<&CpuExecutionContext>` is threaded through
   `crossinterpolate2_with_streams` → `optimize_with_streams` → `find_global_pivots_erased` →
   `search_with_readout` → `to_treetn_chunked` → `to_named_treetn_chunked` → `site_tensor_with_parent`, and to
   the final materialization in `api.rs`. The two internal `to_treetn` calls are replaced by the chunked
   context-taking form. Public entries keep their signatures and pass `None` (`to_treetn`, `to_named_treetn` -
   which also serves the interpolator - `find_global_pivots`, `find_global_pivots_with_rng`). No new public API
   in treetci; the context-taking path is reached only from `crossinterpolate2_in` /
   `crossinterpolate2_with_rng_in`, as in step 3a.
4. **Documentation of the entries.** The context-taking entries' rustdoc stops implying factorization is the
   only stage that moved.

## 3. Contracts

- **Same algorithm and acceptance rules; numerical equivalence within tolerances.** This moves the solve that
  produces the values the search reads, and the source documents that approximation rounding affects pivot
  selection, so identical pivot sets and error histories are **fixture-specific evidence, not a guarantee**.
  Numerical equivalence is a fixed-input solve/materialization contract, demonstrated on fixtures.
- **Worker rejection applies when a nonzero solve is reached**, not unconditionally to materialization (the
  zero-pivot early return takes no session).
- **No new resource**, and no change to any public signature.

## 4. Evidence plan

1. **Solve parity and validation**: `solve_right_full_piv_lu_in` with a caller context agrees with the legacy
   method for `f64` and `Complex64` on a nonzero nonsingular fixture with a nontrivial complex pivot and a valid
   rectangular LHS (exercising transposition rather than identity fixtures); the two validation errors keep
   their variants, including a both-invalid case that establishes square-pivot precedence.
2. **All four overrides**: `f32` and `Complex32` also reach a nonzero solve through the context route with the
   same values as their legacy route.
3. **Dispatch at the boundary**: a multi-site state with nonzero nonsingular pivots is materialized through the
   context-taking path **while the process-global backend is held**, with the eager runtime warmed first and the
   lock released in a controlled way so a wrong (global) route fails the test instead of hanging the suite.
   Analytic comparisons that need global work happen after release.
4. **Propagation**: a focused check that the caller context reaches **both** materializations on the TCI path -
   search-time (global search enabled and actually reached) and final - independent of value parity, which the
   existing end-to-end test already passes with the global route.
5. **Worker rejection**: the context-taking materialization entered from a Rayon worker errors typed - asserted
   by walking the source chain to `CpuExecutionContextError::SessionEntry`, not only by text - on a fixture where
   the solve is reached, paired with the zero-pivot fixture that needs no session at all.
6. **End-to-end parity**: the seeded TCI comparison at non-training points, with ranks/error histories reported;
   identical pivots are stated as fixture evidence if they hold.
7. **Paired measurement, step 3a versus 3d**: the `g0_stage_budget` protocol's `T4A_G0_CALLER_THREADS > 0`
   already selects step 3a's factorization route, so a legacy-versus-new comparison would measure 3a and 3d
   together. The measurement is therefore a paired comparison between the pre-3d revision and this one with
   identical global/caller configurations, including explicit global-1T/caller-1T and the same caller scaling
   cases, recording elapsed time, peak threads and CPU ticks whether or not the curve improves.
8. Gates: fmt, clippy with the repository flags for the two touched crates, their whole-crate tests, rustdoc
   `-D warnings`, `scripts/check-public-error-docs.py`, the panic audit and
   `scripts/repository-rules-review.py`.

## 5. Non-goals and non-claims

- The readout, the cached evaluator, the bridge's einsum and the value traits are **not** touched.
- Publication's global eager-runtime dependency is retained and named, not fixed here.
- No CUDA path, no AD interaction, no downstream `R=10` reproduction.
- No claim that the solve dominates the stage's cost; an unchanged stage-wide measurement is reported as such.

## 6. Measured: relocating the solve shows no clear end-to-end benefit

The paired comparison of §4.7, `crates/tensor4all-treetci/benches/g0_stage_budget.rs` at its default workload
(8-site chain, local dimension 4, `max_bond_dim = 16`, `max_iter = 30`, global pivots enabled, three
repetitions, release), all three repetitions shown, on the local AMD EPYC 7713P. The revisions are
`origin/main` before this change (step 3a only) and this branch (step 3a + 3d), with identical
`RAYON_NUM_THREADS` / `T4A_G0_CALLER_THREADS`:

| global / caller | before 3d | with 3d | peak threads (both) |
| --- | --- | --- | --- |
| 1 / legacy | 6.743, 5.833, 6.083 ms | 6.372, 6.072, 5.640 ms | 2 |
| 1 / 1 | 7.428, 6.274, 6.206 ms | 7.390, 6.254, 6.234 ms | 3 |
| 4 / 1 | 7.867, 6.607, 6.602 ms | 7.320, 6.314, 6.258 ms | 7 |
| 4 / 4 | 8.765, 7.506, 7.538 ms | 8.890, 7.709, 7.741 ms | 10 |

Ranks (`[2, 2, 2]`), the error history and the termination are identical in every configuration, and CPU ticks
stay at or below tick resolution for the single-caller-worker configurations. The 4/4 case is the only one whose
spread separates in the same direction as its medians, and there the relocation is **slower** (+2.7% median).

**What this shows:** at equal or reduced caller budgets, moving the solve onto the caller's context gives no
clear end-to-end benefit on this workload, and the worker penalty the diagnosis measured is unchanged by the
relocation. **What it does not show:** the size of the solve's share, or that the remaining cost is inside the
readout. A relocation experiment of this shape cannot separate those - an equally expensive solve would look the
same under both placements, and matrix preparation, oracle evaluation, tensor publication and search overhead
all remain candidates. Ticks are too coarse here for the differences involved, and peak thread counts are not a
schedule measurement.

Attributing the stage would need accumulated per-component times (solve, preparation, publication, oracle
evaluation, search/readout) rather than a placement comparison, and a budget question would need a wider sweep
of global/caller combinations than the four above. **That attribution has since been made**:
`859-f3-g0-stage-budget.md` §3 measures the components and finds the search's *readout* to be ~67-80% of the
stage while the session-bound materialization machinery is ~8-13%, with the worker penalty living in that
machinery (it doubles) rather than in the readout (which is flat). So this step's neutral result is explained:
the cost this step moved was never the dominant one, and the penalty follows the work into any context.

Together with step 3a this leaves two verified negative results and one falsified hypothesis: the factorization
is not where the gap is (step 3a); relocating the solve does not move the stage either (here); and the readout
does not reach the session seam the withdrawn design proposed, because for this shape it takes raw message
kernels and warm cached columns. The readout does still reach the global session in general - larger grouped
messages, narrow scalars and generic layouts do - so this step's dispatch evidence is about the paths it
executes, not about whole-run independence.

## 7. Settled decisions

1. `Session::backend()` supplies the session object the backend trait expects; no coercion question remains.
2. No public context-taking materialization entry: nothing on this deliverable's path needs one.
