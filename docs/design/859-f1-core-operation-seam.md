# F1 slice 1 — the core operation seam: pairwise contraction on an explicit session

Short design for the first F1 package of [tensor4all-rs#859](https://github.com/tensor4all/tensor4all-rs/issues/859)
("core operation/value seam"). **Revision 3**, after an independent pre-review rejected revision 1 (the
operand bridge it proposed - `try_materialized_inner` + `explicit::detach` - can enter the operand's owning
eager/default runtime from inside the held session, and its result carrier cannot preserve the promised
dtype coverage) and an independent post-review rejected revision 2's diff on four points: a repeated index
reaching the planner through public API (`replace_indices`) underflows instead of being refused, mixed
`f64`/`c64` operands pass adaptation but are not promoted on the session route, a retained deferred storage
error is masked by the generic tracked-value refusal, and the evidence did not cover the promised
representations, dtypes or the helper bodies the boundary guard must include. Revision 3 answers all four.

## 1. What B1 left, and what F1 has to add

B1 (#888) put the concrete route in `tensor4all-tensorbackend::explicit`: a held `Session` with primitive,
einsum and linalg routes, `HeldSession`/`PhaseLane` lifecycle and the `lift`/`detach` boundaries. It did
not let an *app-layer* value use that route: app operations still run through the eager runtime, so a
region cannot hold one explicit session and run core operations on it.

F1's seam: numerical/materializing core operations get an opt-in entry point taking the caller's
`explicit::Session`. Existing public calls and `Index` behavior stay; pure metadata operations keep their
current signatures and take no session. Numerical permutation (`permute`/`permute_indices` with a
nonidentity permutation) is *not* metadata - it materializes and executes an eager transpose today - and is
a later seam, not part of this slice. (Revision 1 called every numerical `IdxTensor` operation eager; that
was too broad: an untracked `sum` already has a storage route.)

## 2. Slice 1 (this change)

```rust
// tensor4all-core, defaults::contract_session, re-exported at the crate root
/// Contract two dense host tensors on a caller-held explicit concrete session.
pub fn contract_pair_on_session(
    session: &mut tensor4all_tensorbackend::explicit::Session<'_>,
    lhs: &IdxTensor,
    rhs: &IdxTensor,
    options: PairwiseContractionOptions,
) -> Result<IdxTensor, IdxTensorError>;
```

### 2.1 Supported inputs (everything else is a typed refusal)

| Input | Decision |
| --- | --- |
| Materialized dense host storage, `f64`/`c64` | Supported |
| Untracked eager dense host value, `f64`/`c64` | Supported, through its borrowed read representation |
| Tracked value (dense or not) | **Refused**: a plain route must not drop AD implicitly |
| Non-dense storage (diagonal, repeated axis class) | Refused: the compatibility route contracts these through its structured-payload path, which this slice does not reproduce |
| CUDA-resident operand | Refused: that needs the resident plan route |
| Integer, `bool`, `f32`/`c32`, or `External` operand | Refused: the result carrier below serves `f64`/`c64` only, and the compatibility route's integer promotion is *not* the session route's (it promotes integers through `common_dtype`), so executing them here would not be parity |
| Mixed `f64`/`c64` operands | Supported: `Session::contraction` promotes them on the caller's session, and the parity tests check it against the compatibility route |

`f32`/`c32` are refused rather than widened. Widening through the existing snapshot bridge would silently
change precision, and `Storage::from_dense_col_major` is only implemented for `f64`/`c64`.

### 2.2 How it runs

- **Planning is the existing body.** Operand indices, expected dims and the contraction specification come
  from the compatibility route's own preparation (`operand_indices_for_contraction`, `prepare_contraction`).
  No second planner. The result's axis classes are reconstructed by dense publication, which is correct
  here precisely because non-dense payloads are refused: this slice does not call
  `binary_contraction_axis_classes`.
- **Malformed input is refused before planning.** A repeated index inside one operand makes the planner's
  result-length arithmetic underflow, and `replace_indices` can produce one through public API, so
  uniqueness is checked first and refused typed.
- **Operand adaptation is representation-based, never eager-materializing.** Materialized storage goes
  through the existing host/native bridge; an untracked eager value goes through its borrowed read
  representation; nothing calls `try_materialized_inner`, `explicit::detach`, `IdxTensor::storage`, or any
  default-context entry, so the held session cannot be re-entered behind the caller's back. Chained calls
  stay on the session for the same reason: the result is materialized host storage, not an eager value.
- **Execution is the caller's session**, through `Session::contraction` with the planned axis pairs. That
  route builds the same label order the planner produces (uncontracted lhs axes in original order, then
  uncontracted rhs axes) and promotes heterogeneous operands on the session, so mixed `f64`/`c64` is
  served exactly as the compatibility route serves it. Conjugation, when the options ask for it, is applied
  on the session before the contraction; no post-contraction permutation is needed.
- **A retained deferred storage error wins over the representation refusals.** It is the failure that
  actually happened, and the metadata checks (`tracks_grad`, `is_cuda_resident`) read through the retained
  source, so returning a generic refusal would lose its typed origin.
- **Publication is a concrete `IdxTensor`.** `IdxTensor::from_storage` over dense column-major `Storage`,
  so the new route does not adopt the value into any eager runtime and no graph is compiled.
- **Pure metadata still takes no session.**

## 3. Where the code lives

`crates/tensor4all-core/src/defaults/`, next to the compatibility route and re-exported at the crate root.
`tensor4all-core` always enables `tensor4all-tensorbackend/global-defaults`, and its `defaults` module is
not cfg-gated today, so this is a *dependency* consequence rather than a new feature gate. The new body
stays free of default-context access so that stage R can move or gate it without changing behavior.

## 4. Evidence plan

1. **An ordering oracle**, independent of the compatibility route: lhs `[a,k,b]`, rhs `[c,k,d]` with
   unequal dims and asymmetric data; expect the values of `[a,b,c,d]`.
2. **Parity with the compatibility route** for dense `f64`/`c64` (values and full index lists: id, prime
   level, tags), including outer products, contractions over every index, rank-0 operands and a second
   contraction chained through one held session.
3. **Representation coverage**: uncached `from_storage` operands, already-cached storage, an untracked
   eager operand, and a non-contiguous untracked eager operand (a nonidentity eager permutation), in both
   operand positions.
4. **Refusals**, each asserting the operation name and that the session is still usable afterwards: tracked
   (dense and structured), diagonal, repeated-axis-class, CUDA-resident (behind the CUDA feature), integer
   and `f32` operands, and a shape mismatch.
5. **Boundary observation**: a direct-source guard over the whole session module, the entry-point body and
   the operand helper, against the default-entry symbols (`try_materialized_inner`, `default_eager_ctx`,
   `default_cpu_execution_context`, `with_default_session`, `explicit::detach`, `.storage()`). It is a
   text guard, not a transitive zero-entry proof; identity comparison (`is_global_default_cpu`) is *not*
   used, because calling it initializes the default runtime itself.
6. **Malformed and deferred input**: a repeated index (through `replace_indices`) refused before planning,
   and the existing injected-failure fixture's deferred conjugation error returned as the typed storage
   error it is.
6. Repository gates: fmt, clippy with the repository flags, rustdoc `-D warnings`, doctests,
   `scripts/repository-rules-review.py`, unit tests for the touched crate.

## 5. Non-goals and explicit non-claims

- No change to `IdxTensor`'s representation, no new storage kind, no change to the compatibility route's
  numerics or routing.
- Not F1 complete: `Matrix`, structured payloads, reductions and the remaining operation families are
  later slices with their own parity evidence.
- No AD, no CUDA device-resident contraction, no session-taking variant for metadata operations.
- No benchmark and no speedup claim: this slice is the seam. F2 measures.
- If a later slice replaces the `IdxTensor` result with a concrete carrier, that is an API change and will
  be described as one.

## 6. Open questions

1. Should `Matrix`/structured coverage land before or after F2's first measurement? F2 only needs the
   dense `f64`/`c64` seam, so this slice serves it either way.
2. Should the deferred-error precedence also apply to the compatibility route? This slice only changes its
   own entry point; the same masking exists there and is not addressed here.
