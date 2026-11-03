# Redundancy audit: double implementations and dead surface

Read-only audit of `origin/main` at `56c25d4a` (2026-11-04). **No code was changed.**
Nothing here was found by grep alone: every entry below was confirmed by reading
both bodies at the cited lines. Where the survey found nothing, that is recorded
too (`Reviewed and clean`), because a clean result is evidence for the next step.

## Why this audit exists

The tree has a documented history of "collapse duplicate types" being claimed
done while duplicates remained (#566 Phase 4 item 31, post-closure gap #659,
closed). This audit re-measures the same class of problem from the current
source instead of from the issue tracker, and it separates two things that the
earlier sweeps mixed:

- **Double implementations** (category A): the same algorithm or the same
  contract written twice, where fixing one side silently leaves the other
  wrong. These are the priority.
- **Other redundancy** (category B): dead public surface, copy-pasted test and
  example harnesses, thin pass-throughs.

## Method

1. `git fetch origin main`, branch `docs/audit-redundancy-2026-11-04` from
   `origin/main`, work in a separate worktree.
2. Mechanical sweeps over `crates/*/src` (no `target/`, no generated output):
   - exact 6- and 8-line duplicate windows across files, then reading the
     matched regions;
   - duplicate string literals in `src/`, which is how the duplicated
     validation contracts (A10) were found;
   - a census of function names defined in three or more source files;
   - parallel `fn` definition sites across example and test directories;
   - dead-surface checks: every free `pub fn`, `pub struct`, `pub enum`,
     `pub trait`, `pub type` against all occurrences in `crates/**/*.rs`.
3. For each candidate, read both bodies, decide identical vs already diverging,
   and only then write it down.
4. Categories A and B are ranked together by (removable lines × certainty);
   every A item is included even where its line count is small.

## Summary

| Category | Items | Removable lines (estimate) |
|---|---|---|
| A. Double implementations | 13 | ~890 |
| B. Other redundancy | 6 | ~4,840 |
| Total (cap 25 items) | 19 | ~5,700 |
| Reviewed and clean (no entry) | 5 areas | — |
| Do not touch (deliberate) | 7 items | — |

Largest single cluster: ~2,800 duplicated lines in `crates/tensor4all-treetn/examples/`
(15 helper functions copied into 3–9 files each). Largest single-contract risk:
two hand-maintained copies of the TCI2 convergence criterion (A1).

---

## A. Double implementations

### A1. The TCI2 convergence criterion is maintained twice by hand

- Left: `crates/tensor4all-tensorci/src/tensorci2.rs:1611-1645` (`convergence_criterion`)
- Right: `crates/tensor4all-treetci/src/optimize.rs:948-963` (inline in `optimize_with_streams`), window size `NCHECK_HISTORY` at `:831`
- What is duplicated: the trailing-window stop rule — three recent iterations with
  error below tolerance, no newly injected global pivots in the window, and a
  stable rank (`min(window) == last`), plus the `max_bond_dim` saturation stop.
- Behaviour: **identical today.** Both compute
  `errors.iter().all(|e| e < tolerance) && nglobal.iter().all(|n| n == 0) && min(ranks) == ranks.last()`.
  The duplication is explicit: `treetci/src/optimize.rs:824-828` says the
  constant "Mirrors `tensor4all-tensorci`'s `TensorCI2Options::ncheck_history`
  default", and `:949-950` points at `docs/design/treetci-termination.md`.
- Action: keep the criterion in one place. The natural home is
  `tensor4all-treetci`'s own stopping contract (documented in
  `docs/design/treetci-termination.md`) with `tensor4all-tensorci` delegating,
  or the reverse if `tensor4all-tensorci` stays authoritative.
- Removable: ~35 lines of duplicated predicate plus the mirrored constant.
- Risk: **high** — this is stopping behaviour for two public APIs; a divergence
  changes returned ranks and errors, not just code shape.
- Why it is like this: the two engines keep separate state layouts
  (`i_set`/`j_set` + `SimpleTensorTrain` vs `HashMap<SubtreeKey, ColMajorArray>`
  + `TreeTN`), so the predicate was copied rather than shared across the layout
  boundary. No crate depends on `tensor4all-tensorci` from inside the workspace
  (`crates/*/Cargo.toml`: `treetci`, `quanticstci`, `partitionedtreetn`,
  `tensor4all-py` depend on `treetci`; nothing depends on `tensorci`), so the
  chain engine is a public leaf crate and the tree engine cannot reuse it
  without a dependency edge.

### A2. `tensor3_to_left_matrix` / `tensor3_to_right_matrix` / `qr_decomp` exist three times

- Left: `crates/tensor4all-simplett/src/canonical.rs:18-31` (`qr_decomp`), `:40-56`, `:59-75`
- Right: `crates/tensor4all-simplett/src/vidal.rs:27-39`, `:126-142`, `:145-161`; `crates/tensor4all-simplett/src/compression.rs:127-143`, `:146-162`
- What is duplicated: the two `Tensor3` → `Matrix` reshapes (left: `l*S+s` rows;
  right: `s*R+r` columns) and the QR-via-rank-revealing-LU helper.
- Behaviour: **identical.** `tensor3_to_left_matrix` and
  `tensor3_to_right_matrix` are byte-identical after whitespace normalisation in
  all three files; `qr_decomp` differs only by one trailing comment
  (`rel_tol: 0.0, // No truncation`).
- Action: one `pub(crate)` module (or move into `types.rs`) and delete the other
  copies; `compression.rs` already imports from `canonical`.
- Removable: ~80 lines (two reshape copies + one `qr_decomp` copy); deleting
  `vidal.rs` (B1) removes the third copy for free.
- Risk: **low** — private helpers inside one crate, byte-identical.
- Why it is like this: `vidal.rs` and `canonical.rs` were written as parallel
  representations, each self-contained; `compression.rs` then grew its own copy.

### A3. `ScalarValue` and the `ScalarTensorElement` trait are defined in two crates

- Left: `crates/tensor4all-core/src/any_scalar.rs:24-113` (`ScalarValue` + 4 `From` + `impl ScalarValue`), `:115-146` (`trait ScalarTensorElement` + 4 impls)
- Right: `crates/tensor4all-tensorbackend/src/any_scalar.rs:14-113` (same enum, 7 variants, same accessors), `:198-233` (same trait + 4 impls)
- What is duplicated: the dynamic scalar value type, its dtype-dispatch
  accessors (`real`/`imag`/`abs`/`is_complex`/`is_zero`/`into_complex`), and the
  private one-method trait that maps a concrete scalar type to a variant.
- Behaviour: **already diverges.** (a) `abs()` — core uses
  `f64::from(re).hypot(im)` for `C32` (`core/src/any_scalar.rs:74-84`), while
  tensorbackend uses `Complex::norm()` in `f32` (`tensorbackend/src/any_scalar.rs:64-73`);
  so `AnyScalar::abs()` and `BackendScalar`-derived magnitudes can differ in the
  last bits for complex-f32 data. (b) core's enum has no `I32`/`I64`/`Bool`, so
  `is_zero`/`as_f64` cover fewer dtypes than the backend's. (c) core's
  `ScalarTensorElement` impls are the exact 4 arms tensorbackend already has.
- Action: expose the backend's value view (or `BackendScalar::value()`), keep
  core's `AnyScalar` for the AD-tracked wrapper only, delete core's
  enum/trait/impls.
- Removable: ~140 lines.
- Risk: **medium** — `AnyScalar::abs` is visible behaviour; unifying changes
  f32-complex rounding. Also already tracked (closed) as part of #659, re-verified
  here at `origin/main`.
- Why it is like this: `tensor4all-core::AnyScalar` wraps `BackendScalar` plus
  eager-session/AD fields, and was ported from `tensor4all-tensorbackend`'s
  value type rather than exported from it (the layering rule forbids core from
  reaching into backend internals, and the backend never made the value type
  public).

### A4. `move_center_to_region_full_rank` is copy-pasted between GSE and TDVP

- Left: `crates/tensor4all-treetn/src/gse.rs:1395-1459`
- Right: `crates/tensor4all-treetn/src/tdvp/mod.rs:1322-1394`
- What is duplicated: the whole "pick the shortest canonical-region → target-region
  path, then sweep QR full-rank left along it" routine, including the error text
  and the `sweep_edge_full_rank(.., FactorizeAlg::QR, Canonical::Left, ..)` call.
- Behaviour: **already diverges.** The bodies differ only in the `"GSE"`/`"TDVP"`
  prefixes and log label — *except* for the signature: gse's version is
  `fn move_center_to_region_full_rank<V>(state: &mut TreeTN<IdxTensor, V>, ..)`
  while tdvp's is generic over `T`. GSE is therefore locked to `IdxTensor`.
- Action: move one generic implementation into the treetn core (next to
  `sweep_edge_full_rank`) and call it from both.
- Removable: ~65 lines, and it un-hardcodes GSE's tensor type.
- Risk: **low** for the move, **medium** if the un-hardcoding exposes a generic
  path GSE never exercised.
- Why it is like this: TDVP and GSE were developed as separate algorithms over
  the same tree; the center-move helper was copied when GSE was added.

### A5. `MPOError` and `SimpleTensorTrainError` are two error enums with six shared variants

- Left: `crates/tensor4all-simplett/src/mpo/error.rs:9-94`
- Right: `crates/tensor4all-simplett/src/error.rs:39-83`
- What is duplicated: `DimensionMismatch`, `IndexOutOfBounds`, `DataLengthMismatch`,
  `Empty`, `InvalidOperation` and `MatrixCI(#[from] MatrixCIError)` — same variant
  shape, same `#[error]` string for five of six.
- Behaviour: **already diverges.** `Empty` reads `"MPO is empty"` in one and
  `"Tensor train is empty"` in the other; `MPOError` adds
  `BondDimensionMismatch`/`SharedDimensionMismatch`/`LengthMismatch`/
  `InvalidBoundary`/`FactorizationError`. A caller matching on one enum cannot
  reach the other's arms.
- Action: drop `MPOError` and return `SimpleTensorTrainError` (early development,
  no compatibility obligation); or keep one enum and alias.
- Removable: ~94 lines plus the `?`-conversion sites in `mpo/*` (10 files use
  `MPOError`).
- Risk: **medium** — public error type of `simplett::mpo`.
- Why it is like this: the MPO module was developed after the TT module and
  needed extra variants; the shared ones were copied instead of reused.

### A6. `concatenate_along_new_index` is both a trait default and an `IdxTensor` override, and the two differ in semantics

- Left: `crates/tensor4all-core/src/tensor_like.rs:1999-2065` (trait
  `TensorConstructionLike::concatenate_along_new_index`, default body)
- Right: `crates/tensor4all-core/src/defaults/idx_tensor.rs:7747-7821`
  (`impl TensorConstructionLike for IdxTensor`, override)
- What is duplicated: the whole validation block — empty input, count mismatch,
  source-index presence, source-axis position, "must match away from the source
  axis", dimension-sum overflow, output-dimension match — with the same
  `anyhow!` messages in the same order.
- Behaviour: **already diverges, in a way that matters.** The `IdxTensor`
  override calls `ensure_shape_packing_preserves_ad(..)` and
  `try_materialized_inner()` and concatenates through the eager session; the
  trait default builds the result by slicing via `select_indices`. Any other
  implementor of the trait silently gets the non-AD-aware path.
- Action: put the AD-safe body in one place (the trait default), or make the
  trait require an explicit implementation and delete the default.
- Removable: ~75 lines.
- Risk: **medium** — the default body is what future implementations inherit.
- Why it is like this: the default was written first for the generic
  `TensorConstructionLike` surface; `IdxTensor` later needed the session/AD path
  and its override copied the validation so the error messages would not change.

### A7. `InnerCache` hand-writes the same five-arm match in seven methods

- Left: `crates/tensor4all-core/src/cached_function/mod.rs:143-266`
  (`key_bytes` :144, `new` :155, `get` :191, `insert` :203, `contains` :223,
  `len` :235, `clear` :245, `key_type_name` :255)
- Right: same file, each method repeats the `Self::U64/U128/U256/U512/U1024`
  match with identical arms.
- What is duplicated: the variant dispatch, not the payload logic — every method
  matches the five key widths and runs one two-line operation.
- Behaviour: **identical** by construction.
- Action: a `macro_rules!` that expands the match for a given operation, or a
  small trait implemented once per key width.
- Removable: ~120 lines (eight matches × ~15 lines each, minus one macro).
- Risk: **low** (a macro expands to the same match, so no indirection appears).
  The plausible reason for the current shape is that the author wanted each
  method readable on its own, or worried a macro would obscure the hot `get`/
  `insert` path — a macro leaves the generated code identical.
- Why it is like this: the five key widths were added incrementally; each new
  method copied the existing match.

### A8. The C API hand-rolls the `require_tree` error preamble in five functions

- Left: `crates/tensor4all-capi/src/treetn.rs:957-962`, `:1095-1100`,
  `:1621-1627`, `:1628-1634`, `:2081-2087`
- Right: 21 other call sites in the same file use `require_tree(..)?` /
  `require_tree_mut(..)?` inside `run_catching`/`run_status`, e.g. `:990`.
- What is duplicated: the six-line
  `match require_tree(..) { Err((code, msg)) => { set_last_error(&msg); return code; } }`
  preamble, which is exactly what the `CapiResult`/`?` route already does.
- Behaviour: **identical** — `run_catching`/`run_status` map the error and set
  the same last-error text. `t4a_treetn_set_tensor` (`:990`) proves the shorter
  route works for a mutable handle.
- Action: hoist the five call sites into their existing `run_catching`/
  `run_status` closure and use `?`.
- Removable: ~30 lines (5 × 6).
- Risk: **low** — ABI and observably identical errors (`t4a_last_error_message`
  is documented in `docs/CAPI_DESIGN.md` and keeps the same text).
- Why it is like this: `require_tree` returns `CapiResult`, and the functions
  written before the `run_catching` refactor still match on it explicitly;
  later functions use `?`.

### A9. `GseError` and `TdvpError` are two enums with seven shared variants and a duplicated `From` impl

- Left: `crates/tensor4all-treetn/src/gse.rs:177-247` (enum), `:249-266` (`From<SquareSiteMappingError>`)
- Right: `crates/tensor4all-treetn/src/tdvp/mod.rs:42-128` (enum), `:130-147` (same impl)
- What is duplicated: `InvalidOption`, `MissingCenter`, `TopologyMismatch`,
  `UnsupportedStateSiteCount`, `UnsupportedMultipleSiteMappings`,
  `MissingMapping`, `InvalidMapping`, `Algorithm`, and the four-arm
  `SquareSiteMappingError` translation, which differs only in the target type
  name.
- Behaviour: **identical** (same payloads, same `#[error]` strings for the shared
  variants). `GseError` additionally has a `Tdvp(TdvpError)` variant, because GSE
  runs TDVP internally — which is itself a hint that the two types should be one.
- Action: one `SquareSiteMappingError`-carrying error type shared by both
  modules, or a `From` written once in terms of a common trait.
- Removable: ~40 lines (the `From` impl plus the shared variant declarations).
- Risk: **medium** — public error enums of `treetn::gse` and `treetn::tdvp`.
- Why it is like this: TDVP came first; GSE reuses the TDVP machinery but wanted
  its own error namespace.

### A10. "finite and non-negative threshold" is re-implemented 14 times against 4 error types

- Shared validator: `crates/tensor4all-core/src/truncation.rs:211-216`
  (`validate_threshold_value`, returning `InvalidThresholdError`)
- Independent copies of the same predicate `!x.is_finite() || x < 0.0`:
  - `crates/tensor4all-core/src/defaults/qr.rs:478` + `:486-488`
    (`options_rtol_is_valid`, returning `QrError::InvalidRtol`)
  - `crates/tensor4all-core/src/global_default.rs:58-63`
    (`GlobalDefault::set`, returning `InvalidRtolError`)
  - `crates/tensor4all-core/src/floating_zone.rs:254`
  - `crates/tensor4all-itensorlike/src/options.rs:16-23`
    (`validate_svd_truncation_options`, `TensorTrainError::OperationError`)
  - `crates/tensor4all-itensorlike/src/linsolve.rs:92`
  - `crates/tensor4all-aci/src/validation.rs:32-39` (`AciError::InvalidOptions`, two fields)
  - `crates/tensor4all-interpolativeqtt/src/interpolation.rs:324`, `:682`
  - `crates/tensor4all-quanticstransform/src/fourier.rs:118`
  - `crates/tensor4all-partitionedtreetn/src/patching.rs:1168`,
    `src/subdomain_tree_tn.rs:945`, `:954`
- What is duplicated: the same validity predicate and message
  (`"Invalid rtol value: {0}. rtol must be finite and non-negative."` appears
  verbatim in `qr.rs:23`, `tensor_like.rs:76` and `global_default.rs:11` — three
  public error types in one crate).
- Behaviour: **identical for the shared predicate**; the field sets and error
  types differ (`<= 0.0` in `linsolve.rs:70` and `capi/src/treetn.rs:1995` is a
  deliberately different contract for a solver tolerance).
- Action: the three in-crate copies in `core` (`truncation`, `qr`,
  `global_default`) should route through `validate_threshold_value` and one of
  the three error types; the cross-crate copies should at least share the
  message and the predicate. `core/defaults/factorize.rs:297` already does the
  conversion correctly and shows the pattern.
- Removable: ~40 lines (the local predicates), plus the three duplicate messages.
- Risk: **low** for `core`'s internal three; **medium** across crates, because
  each crate owns its public error type.
- Why it is like this: each crate grew its own options-validation entry point,
  and `core` has three independent public error types (`InvalidThresholdError`,
  `QrError`, `InvalidRtolError`) created at different times.

### A11. `descending_order` is implemented twice, with the same test twice

- Left: `crates/tensor4all-core/src/defaults/svd.rs:641-648` (`&[f64]`),
  test at `crates/tensor4all-core/src/defaults/svd/tests/mod.rs:323-329`
- Right: `crates/tensor4all-simplett/src/compression.rs:235-255`
  (generic over `S`, with a `value: impl Fn(S) -> f64` reader),
  test at `crates/tensor4all-simplett/src/compression/tests/mod.rs:316-325`
- What is duplicated: the "is the sequence already non-increasing? otherwise
  return the descending permutation" helper, plus its unit test, under the same
  test name (`descending_order_is_none_for_ordered_values_and_sorts_the_rest`).
- Behaviour: **identical.** `compression.rs`'s own doc comment says so: "The
  `tensor4all-core` SVD wrappers keep the same check."
- Action: keep the generic one (it subsumes the `f64` version) in `core` and call
  it from `simplett`; delete the duplicate test.
- Removable: ~30 lines including the duplicated test.
- Risk: **low** — pure function, already covered by two identical tests.
- Why it is like this: the backend-agnostic singular-value ordering fix (faer#82)
  was needed in both crates and duplicated rather than exported.

### A12. Thirteen `Options` types hand-write the same truncation builders

- Left: `crates/tensor4all-treetn/src/options.rs:112-115`, `:181-184`;
  `crates/tensor4all-treetn/src/treetn/fit.rs:1505-1508`;
  `crates/tensor4all-treetn/src/treetn/contraction.rs:2434`, `:2462`;
  `crates/tensor4all-treetn/src/dmrg/mod.rs:265-268`, `:281`;
  `crates/tensor4all-treetn/src/tdvp/mod.rs:392-395`, `:407`;
  `crates/tensor4all-treetn/src/operator/apply.rs:238-241`, `:244`;
  `crates/tensor4all-treetn/src/treetn/swap.rs:81`
- Right: `crates/tensor4all-core/src/tensor_like.rs:396`, `:426-429`;
  `core/src/defaults/svd.rs:100-103`; `itensorlike/src/options.rs:68`, `:74-77`,
  `:281-284`, `:287`, `:528-531`, `:534-537`; `quanticstci/src/options.rs:172-175`;
  `interpolativeqtt/src/options.rs:80-83`;
  `treetn/src/linsolve/common/options.rs:88-91`, `:94`
- What is duplicated: `with_max_bond_dim` (13 definitions) and
  `with_svd_policy` (13 definitions). Twelve of the `with_max_bond_dim` bodies
  are the byte-identical `self.max_bond_dim = Some(..); self`; the treetn
  variants write `self.truncation.max_bond_dim`, and
  `linsolve/common/options.rs:88-91` delegates.
- Behaviour: **identical** for the field assignment; the *validation* around it
  is not uniform (see A10 — `itensorlike` validates the policy, `treetn`'s
  `TruncationOptions` does not), so today "the same builder call" does not imply
  "the same validation".
- Action: a macro that generates the builder for a named field, or a small
  `TruncationOptions` type shared by the crates that already speak the same
  vocabulary. Start with the seven in `tensor4all-treetn`, where the field lives
  in `TruncationOptions` for two of them only.
- Removable: ~50 lines mechanically; the larger gain is removing the
  option-vocabulary drift (one builder, one validation).
- Risk: **low** mechanically; **medium** if the missing validation is changed as
  part of the same step.
- Why it is like this: each sub-algorithm (DMRG, TDVP, GSE, fit, contraction
  sweeps, operator apply, linsolve) owns its own options struct so its
  documentation can list only the knobs it reads; the builder boilerplate then
  repeats.

### A13. The C API writes the opaque-handle boilerplate once per wrapper type

- Left: `crates/tensor4all-capi/src/types.rs:36-70` (`t4a_index`: struct, `new`,
  `inner`, `Clone`, `Drop`, `unsafe impl Send`)
- Right: `:125-190` (`t4a_tensor`), `:196-235` (`t4a_treetn`), `:240-280`
  (`t4a_treetn_evaluator`), `:741-779` (`t4a_qtt_layout`)
- What is duplicated: `#[repr(C)] struct { _private: *const c_void }`, the
  `Box::into_raw`/`Box::from_raw` pair, `Clone` via `new(inner().clone())`, the
  null check in `Drop`, and `unsafe impl Send`/`Sync`.
- Behaviour: **identical** except `t4a_treetn_evaluator`/`t4a_qtt_layout`, which
  also carry non-pointer fields.
- Action: a `macro_rules!` defining the opaque wrapper for an inner type, with
  the extra fields as macro input.
- Removable: ~120 lines (5 × ~24 lines core boilerplate, minus one macro).
- Risk: **medium** — this is `unsafe` code; a macro hides the `Box::from_raw`
  pairing from a reviewer. The pairing and the `Send`/`Sync` justification must
  stay visible, e.g. one documented macro definition instead of five copies.
- Why it is like this: wrappers were added one C API endpoint group at a time,
  each copying the previous one; the `SAFETY` comments are already the only
  per-type difference.

---

## B. Other redundancy

### B1. `simplett/src/vidal.rs` has no production caller

- `crates/tensor4all-simplett/src/vidal.rs:208` (`pub struct VidalTensorTrain`),
  `:548` (`pub struct InverseTensorTrain`); tests at
  `crates/tensor4all-simplett/src/vidal/tests/mod.rs` (335 lines)
- What is redundant: 801 lines of implementation plus 335 lines of tests for two
  canonical forms that nothing in the workspace uses. `rg 'VidalTensorTrain|InverseTensorTrain'`
  outside `vidal.rs`/`vidal/tests/` finds only `lib.rs:94` (re-export),
  `prelude.rs:20-22`, `traits.rs:44`, `tensortrain.rs:40` (doc links),
  `lib.rs:23` and `README.md:9` (doc prose).
- Behaviour: not a duplicate — **dead public surface**. It *also* carries the
  third copies of A2's helpers.
- Action: delete the module, both exports, the prelude entries and the README
  line, per `AGENTS.md` "Early development: no backward compatibility".
- Removable: ~1,136 lines.
- Risk: **medium** — it is public API. The README advertises it as a feature, so
  removal is a product decision, not a pure cleanup. No user-facing guide uses
  it (`docs/book/src` has no `VidalTensorTrain`).

### B2. `simplett/src/canonical.rs` is reachable only from its own tests and one integration test

- `crates/tensor4all-simplett/src/canonical.rs:103` (`pub struct SiteTensorTrain`),
  `:449` (`pub fn center_canonicalize`), 558 lines total
- What is redundant: `SiteTensorTrain` appears outside `canonical.rs` only in
  `crates/tensor4all-simplett/tests/final_bond_guards.rs:23`,
  `lib.rs:71`, `prelude.rs:21`, and doc prose. `center_canonicalize` is called
  only from `canonical/tests/mod.rs` and `crates/tensor4all-simplett/tests/`.
- Behaviour: **same caveat as B1** — dead production surface, not a duplicate.
- Action: decide explicitly: keep as documented public API (then add a runnable
  guide that uses it) or delete. Do not leave it as an export nobody exercises.
- Removable: up to 558 lines if deleted.
- Risk: **medium** — public API.

### B3. `treetn/examples/`: 15 helper functions copied into 3–9 files (~2,800 lines)

- Left/right (all in `crates/tensor4all-treetn/examples/`, 22 binaries, 10,613 lines):
  `create_n_site_pauli_x_mpo_with_internal_indices` ×7 (e.g. `compare_mps_mpo_pauli.rs:40-109`,
  `test_linsolve_mpo_pauli_operator.rs:73-142`), `unique_dyn_index` ×9
  (`test_linsolve_mpo_identity.rs:31-38` and
  `minimal_linsolve_identity_cases.rs:28-35` are byte-identical; the
  `benchmark_linsolve_mpo.rs:35-42` copy has already drifted — it takes an RNG
  and uses `Index::new` instead of `DynIndex::new_dyn`), `make_node_name` ×9,
  `scale_treetn` ×8, `print_bond_dims` ×7,
  `create_n_site_index_mappings` ×7, `compute_residual` ×6, `run_test_case` ×5,
  `create_random_mps_with_same_sites` ×5, `mpo_node_indices` ×5,
  `create_n_site_mps` ×4, `create_index_mappings` ×4, `bond_indices` ×4,
  `compute_state_error` ×3.
- What is redundant: the same MPS/MPO construction, residual computation and
  bond-dimension printing harness, copied per repro program.
- Behaviour: **identical** for the ones spot-checked byte-for-byte
  (`create_n_site_pauli_x_mpo_with_internal_indices`, two of the
  `unique_dyn_index` copies), **already diverged** for at least one
  (`unique_dyn_index` in `benchmark_linsolve_mpo.rs`); the
  file-level duplicate windows reach 308 shared 8-line windows between a single
  pair (`test_linsolve_mpo_pauli_imaginary.rs` vs
  `test_linsolve_mpo_pauli_operator.rs`).
- Action: one `examples/common/mod.rs` (or `#[path]` module) with these helpers;
  each example keeps only its scenario. Some are issue repros
  (`test_linsolve_identity_operator_issue.rs`, 894 lines) that should arguably
  become `#[ignore]` regression tests instead of examples.
- Removable: ~2,800 lines (sum of the duplicate copies, measured per function).
- Risk: **low** — examples are not part of the API; each stays runnable.
- Why it is like this: each of these files was written as a standalone repro for
  a specific linsolve defect, copied from the previous one so it could run alone.

### B4. `itensorlike/examples/`: benchmark and GMRES harness copied per binary

- `crates/tensor4all-itensorlike/examples/`: `create_random_mpo` ×6
  (`benchmark_contract.rs:28-59`, `benchmark_contract_profiled.rs:11`,
  `benchmark_contract_one_shot.rs:8`, `benchmark_contract_fit.rs:28`,
  `benchmark_contract_detailed.rs:11`, `test_gmres_mps.rs:831`),
  `create_identity_mpo` ×4 (`test_gmres_mps.rs:290`, `test_gmres_mpo.rs:555`,
  `test_restart_gmres_mps.rs:193`, `test_fit_vs_zipup.rs:1368-1404`).
- What is redundant: the same random/identity MPO builders across the five
  `benchmark_contract*` binaries and the GMRES repro binaries.
- Behaviour: **identical** (32 and 37 lines respectively, same bodies).
- Action: same `examples/common` module as B3.
- Removable: ~300 lines (`create_random_mpo` 5 copies = 169,
  `create_identity_mpo` 3 copies = 136).
- Risk: **low**.
- Why it is like this: the `benchmark_contract*` family are variants of one
  benchmark split by contract algorithm.

### B5. Test-module setup helpers duplicated 3–4 times

- `chain` ×4 (`crates/tensor4all-treetn/tests/context_adoption.rs:22-46`,
  `dmrg.rs`, `tdvp.rs`, `gse.rs`), `dense` ×4
  (`tests/cuda_component_transfer.rs:14-27`, `cuda_pending_download.rs`,
  `cuda_component_upload.rs`, `cuda_tree_contraction.rs`), `cpu_context` ×4,
  `col_major_offset` ×3, plus 22 duplicate windows between
  `crates/tensor4all-core/src/block_tensor/tests/mod.rs` and
  `crates/tensor4all-core/src/krylov/tests/mod.rs` (both build the same
  `GmresOptions` fixture at `block_tensor/tests/mod.rs:129-134` /
  `krylov/tests/mod.rs:968-973`).
- What is redundant: small per-file tree/context fixtures and solver options.
- Behaviour: **identical**.
- Action: a `tests/common/mod.rs` per crate.
- Removable: ~140 lines.
- Risk: **low**, but see the do-not-touch note: independent test harnesses can be
  deliberate. Here they are byte-identical fixtures, not independent oracles.

### B6. `itensorlike::LinsolveOptions` re-declares treetn's options as pass-through getters

- `crates/tensor4all-itensorlike/src/options.rs:555-580`
  (`with_coefficients`, `gmres_tol`, `gmres_max_restarts`, `gmres_restart_dim`
  forwarding) vs `crates/tensor4all-treetn/src/linsolve/common/options.rs:120-144`
- What is redundant: `itensorlike::linsolve` already converts to
  `tensor4all_treetn::LinsolveOptions` at
  `crates/tensor4all-itensorlike/src/linsolve.rs:103-107`; the wrapper struct
  keeps its own copies of the same six fields and forwards them one by one.
- Behaviour: **diverges in validation** — `itensorlike/src/linsolve.rs:70-90`
  validates `gmres_tol`/`gmres_max_restarts`/`gmres_restart_dim` itself, and
  `treetn`'s copy validates nothing, so the same field is checked in one crate
  only.
- Action: keep one options type (the treetn one) and re-export or `Deref`; or
  keep the wrapper but delete the duplicated validation and let treetn own it.
- Removable: ~40 lines.
- Risk: **medium** — public options type; the treetn side would have to start
  validating, which can reject input that is accepted today.
- Why it is like this: `itensorlike` offers a TT-shaped facade over the tree API
  and mirrors the knobs for discoverability.

---

## Reviewed and clean (checked, nothing to report)

- **Dead free `pub fn`**: all 349 free `pub fn` in `crates/**/src` have at least
  one occurrence outside their definition (whole-workspace count, including
  tests). Zero dead free functions.
- **Dead public types**: of 243 `pub struct`, 113 `pub enum`, 46 `pub trait`,
  19 `pub type`, only the `ReadmeDoctests` markers are definition-only, and
  those are intentional rustdoc anchors. Type-level deadness therefore shows up
  only as B1/B2 (exported, tested, but never used by another crate).
- **`PivotKernel` kernels**: `matrixluci/dense.rs` (dense full-pivoting LU via the
  tensor backend) and `matrixluci/block_rook.rs` (lazy rook search over a
  candidate source) are two different pivot strategies behind one trait
  (`matrixluci/kernel.rs`), not two copies of one algorithm.
- **`itensorlike::linsolve`**: delegates to `tensor4all_treetn::square_linsolve`
  (`linsolve.rs:103`); it does not re-implement GMRES. Its
  `infer_index_mappings` (`:163`) has no treetn counterpart — treetn requires
  explicit `IndexMapping` — so this is a facade, not a duplicate.
- **`matrixlu.rs` / `matrixaca.rs` / `matrix_luci.rs`** implement rrLU, ACA and
  LU cross-interpolation respectively; shared helpers
  (`validate_luci_options`, `validate_owned_luci_input`) are already imported
  rather than copied.
- **`quanticstci`'s `windows(2)` checks** (`quantics_tci.rs:1024-1060`) validate
  a strictly increasing grid; unrelated to A11's descending-order helper.

## Do not touch (deliberate duplication)

1. **Rustdoc `# Examples` and `ReadmeDoctests` anchors** — one per crate by
   design; they are the doctest entry points, not duplicated code.
2. **`tensor4all-capi` thin wrappers around core/treetn calls** (e.g.
   `types.rs` accessors, `treetn.rs` per-property getters) — the FFI boundary is
   exempt from the "no thin wrapper" rule (AGENTS.md, `docs/CAPI_DESIGN.md`), and
   each one carries the `unsafe`/ownership contract the Rust API does not.
3. **`tensor4all-py` wrappers** — the PyO3 boundary is a prototype outside the
   workspace and mirrors the Rust API on purpose.
4. **`capi`'s `catch_unwind`/status mapping** (`run_catching`, `run_status`,
   `capi_error`, `err_status`) — error-preservation contract from
   `docs/CAPI_DESIGN.md`; A8 only removes the five sites that bypass it.
5. **Benchmark and test harnesses that measure different backends** (e.g.
   `crates/tensor4all-treetn/benches/cached_evaluator.rs`, the
   `cuda_*` integration tests) — separate measurement setups are intentional;
   B5 is limited to byte-identical fixtures.
6. **`normal` / per-type `#[inline]` scalar helpers** in
   `tensor4all-tensorbackend`'s `storage.rs` arithmetic impls — written per
   concrete type so the hot loops monomorphise without a dispatch match; sharing
   them would reintroduce dynamic dispatch. Not reported as redundancy.
7. **`feature`-gated mutually exclusive implementations** (`global-defaults`,
   `tenferro-cpu-*`, `hdf5/link`, `hataori-mpi`) — exactly one side is compiled
   in any configuration; duplication is not reachable.

## Known / already tracked

- The two public `Scalar`-family and `AnyScalar` surfaces (A3) are the subject of
  **#659**, which was **closed** while the duplicates remained: re-verified at
  `origin/main` = `56c25d4a`, `core::AnyScalar` still exists next to
  `tensorbackend::BackendScalar`, and `ScalarValue` is still defined twice. This
  audit adds the detail that they already diverge numerically
  (`hypot` vs `Complex::norm`).
- Hand-rolled contraction loops and `_f64`/`_c64` twins were swept in **#624**
  (closed); the remaining items found here are a different class (error types,
  validation predicates, stop criteria), not contraction loops.
- Evaluator key/layout dedup (**#710**, closed) holds: `treetn`'s
  `cached_evaluator` shows no cross-file duplicate windows against its own
  modules beyond the test helpers counted in B5.
- No open issue currently tracks any of A1, A2, A4, A5, A6, A7, A8, A9, A10,
  A11, A12, A13, B1–B6.

## Verification limits

- Local gates were not run: this change adds one document and no code, so no
  build, test or doctest was executed (`cargo` was not invoked at all).
- No GPU or release-mode evidence was needed.
- Line counts are the duplicated copies measured by function span; "removable"
  excludes the shared helper that replaces them, so the real diff will be
  slightly smaller (a macro or module of ~10–40 lines each).
- The audit covers `crates/**/src`, `crates/*/examples`, `crates/*/tests`,
  `crates/*/benches` and non-tutorial `docs/`. `docs/tutorial-code`,
  `docs/book` snippets and `target/` were not audited.
- Findings on `tensor4all-tensorci`'s relation to `treetci` (A1) were limited to
  the stopping criterion and the dependency graph; the full two-site sweep
  implementations were not compared line by line, because a whole-crate
  comparison is a design decision rather than a cleanup diff.
