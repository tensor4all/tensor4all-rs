# F1 slice 1 — the core operation seam: pairwise contraction on an explicit session

Short design for the first F1 package of [tensor4all-rs#859](https://github.com/tensor4all/tensor4all-rs/issues/859)
("core operation/value seam"). **Revision 9**, adding slice 7. **Revision 8**, adding slice 6. **Revision 7**, adding slice 5. **Revision 6**, adding slice 4. **Revision 5**, adding slice 3. **Revision 4**, after an independent pre-review rejected revision 1 (the
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

## 4.1 Slice 2: the inner product on a session (delivered after slice 1)

`inner_product_on_session(session, lhs, rhs) -> AnyScalar` is the reduction companion slice 1 needed for
F2 (#510), and it reuses slice 1's operand adaptation untouched - same representations, same typed refusals,
same deferred-error precedence, same uniqueness refusal before planning. It mirrors
`IdxTensor::inner_product`: when both operands carry the same index set, the right operand is aligned to the
left one's order with `Session::permute` and contracted over every axis; otherwise the shared axes are
contracted and the remaining axes are reduced with `Session::sum`. The left operand is conjugated on the
session, as the compatibility route conjugates it. The result is published as a rank-zero `IdxTensor` and
returned as an `AnyScalar` built from the concrete value, *not* through `AnyScalar::from_tensor`: that
wrapper extracts its scalar through the tensor's eager representation, which would adopt a freshly
materialized value into the process-global eager runtime - the exact boundary this seam exists to keep.
Both operands are adapted, and therefore refused, before either is materialized, and the planning that the
unaligned branch needs happens before any numerical session work.

Evidence for slice 2 (`crates/tensor4all-core/tests/f1_session_inner_product.rs`): parity with
`IdxTensor::inner_product` for `f64` across the same-order, permuted, subset and disjoint cases; `c64`
parity, where the left-operand conjugation is observable; and refusals for a diagonal and a tracked operand
inside one held session that keeps working for the next reduction and contraction. The source guard covers
the new entry point automatically, because it checks the whole session module.

## 4.2 Slice 3: the reduction (delivered after slices 1 and 2)

`sum_on_session(session, tensor) -> AnyScalar` is the next reduction the app families need, so it is the
next seam. It reuses slice 1's operand adaptation unchanged - same
representations, same typed refusals, same deferred-error precedence, same uniqueness-free path - and runs
`Session::sum`, which reduces every axis and duplicates an already rank-zero value, so one call covers the
rank-zero and the general case. The scalar is published from the concrete value through the same
`AnyScalar::from_published_*` route as slice 2, never through `AnyScalar::from_tensor`.

**A zero-length axis is a documented non-parity**: the compatibility route accumulates from zero over the
empty payload and answers `0`, while the session's reduction rejects a zero-length axis. The session keeps
its backend's refusal rather than special-casing it, and a regression test covers it (including that the
held session survives that session-side failure).

**The dtype domain is the shared one: `f64` and `c64`.** `f32`/`c32` are refused by the same adaptation the
contraction and inner-product slices use. The compatibility `sum` serves them (it routes through
`AnyScalar`'s `f32`/`c32` variants), so this is a deliberate narrowing: widening that adaptation is one
change with its own parity evidence for all three operations, not a side effect of adding a reduction. The
refusal is typed and names the served domain, and the tests assert it.

`norm`/`norm_squared` were deferred here and landed in §4.5 with their own entry points and evidence. They
were *algebraically* derivable from `inner_product_on_session` plus a square root, but that is not numerical
parity: the compatibility route uses a scaled accumulator, and a derived value would not necessarily round
the same way. §4.5 records what the accumulator does and does not make observable through this API.

Evidence for slice 3 (`crates/tensor4all-core/tests/f1_session_sum.rs`, 5 tests): parity with
`IdxTensor::sum` and with an input-data sum independent of it, for a `from_dense` operand, a fresh
`from_storage` operand and a rank-zero operand; complex parity through all three of those representations,
asserting both components and that a public operation routing through the scalar's carried tensor sees the
same numbers; the empty-payload regression above; the `f32`/`c32` refusals with their typed message; and
refusals for a diagonal and a tracked operand with successful reductions inside the same held region. Each
failure is asserted by operation name - `sum_on_session` for this entry point, `contract_pair_on_session`
for the shared contraction one - so a sum failure is not reported as a contraction.

## 4.3 Slice 4: the nonidentity permutation (delivered after slices 1-3)

`permute_on_session(session, tensor, perm) -> IdxTensor` is the numerical counterpart of
`IdxTensor::permute`: the compatibility entry point materializes an eager transpose, so this is work an app
holding a region open cannot do on the compatibility route. The convention is the compatibility one
(`perm[new_axis] = old_axis`), the validation is the compatibility route's own three checks with its
messages, and an identity permutation returns the tensor unchanged with no session work - the one case that
is metadata only. The nonidentity case adapts the operand through the shared admission (same refusals, same
deferred-error precedence), runs a single-operand einsum whose output labels are the requested order (which
materializes a contiguous host result rather than leaving a transposed view), and publishes it with the
permuted index list.

**The identity shortcut is not an admission bypass by accident.** Because it returns before admission,
nonidentity permutations of a narrower dtype are refused while identity permutations of the same tensor
succeed, and identity also succeeds for a tracked, diagonal, device-resident or deferred operand. That is the
compatibility route's behaviour as well - it clones after validating, and only a nonidentity permutation
materializes - so the record states it instead of moving admission ahead of the shortcut and breaking
compatibility. The shared validation itself is one private helper used by both entry points, so the three
refusals cannot drift apart.

Evidence (`crates/tensor4all-core/tests/f1_session_permute.rs`, 6 tests): parity with `IdxTensor::permute`
and an independent stride oracle in which each result element is checked against the old element it came
from, for two representations and three three-axis permutations, one of which is neither a reversal nor a
rotation; complex values against an explicit expected vector; identity on a rank-one vector, rank zero and a
square matrix; **`f32` and `c32` nonidentity refusals naming the served domain, with the identity case
succeeding on the same tensors**; identity succeeding on a diagonal and on a tracked operand; the three
malformed permutations with their exact messages and operation name; a non-contiguous eager operand
(a compatibility permutation) fed to a further nonidentity session permutation and compared against the
compatibility route on the same operand, plus its value multiset; and a permutation applied twice inside one
held region returning the original.

## 4.4 Slice 5: the index-order alignment (delivered after slice 4)

`permute_indices_on_session(session, tensor, new_indices) -> IdxTensor` is the second numerical permutation
form the app crates use: the one that aligns an operand to a caller-supplied index order before a
contraction. The permutation is derived with the compatibility route's own helper
(`compute_permutation_from_indices`), so an index list that is not a permutation of the tensor's own indices is
refused with the same message, and a nonidentity alignment executes through slice 4's route, which means the
same dense `f64`/`c64` domain and the same operand refusals. An identity order is metadata only: it returns the
same storage under the supplied index list *for any representation*, including a diagonal or structured
payload, exactly as the compatibility route does - which is why the seam keeps a crate-private index-metadata
setter rather than materializing.

**Full index equality ignores the dimension**, so a supplied list can carry a different shape than the
tensor's matching index. The compatibility route rejects that when it builds a nonidentity result, and this
route refuses it *before either branch* rather than silently keeping the tensor's own dimensions. For an
identity order that is deliberately stricter than compatibility, which accepts malformed dimension metadata
there; the record states the tightening instead of reproducing the hole.

**The identity case keeps the representation, the storage and the eager cache.** It returns a crate-private
metadata view (`aligned_identity_metadata`) that clones the storage enum value and the cache `Arc`, exactly
what the compatibility route returns for an identity order - including for tracked, deferred, diagonal and
structured payloads, which never reach admission.

Evidence (`crates/tensor4all-core/tests/f1_session_permute_indices.rs`, 5 tests, plus one crate-owned unit
test): parity with `IdxTensor::permute_indices` on indices, values and per-axis dimensions for a three-axis
alignment, the identity order and an axis swap, including that the dimension multiset is preserved; the
identity order keeping a diagonal payload diagonal and its values unchanged; a complex alignment against the
compatibility values; a supplied index carrying a mismatched dimension refused with this entry point's
message, in both the nonidentity and the identity order; the refusal reasons (length, foreign index, same-id
different-prime) preserved with this entry point's operation name and a numerical alignment succeeding
afterwards; and a nonidentity alignment of a diagonal and of a tracked operand refused under the permutation
route's name, because the execution is delegated. The unit test asserts that the identity alignment shares
the storage kind, the axis classes and the **eager cache `Arc`** (with a populated cache, via a compact
payload) for dense, diagonal, tracked and deferred operands, and that a deferred operand keeps its retained
diagnostic.

## 4.5 Slice 6: `norm_squared` and `norm` (delivered after slice 5)

`norm_squared_on_session` and `norm_on_session` compute the compatibility route's values, using its own
scaled accumulator (`Lassq`) over the values the session materialized, so the two routes agree exactly -
including on fixtures whose direct squares underflow or overflow. The session's part is the materialization:
the operand is adapted and materialized on the caller's session instead of through the compatibility route.

**An earlier roadmap note in §4.2 said the scaled accumulation is why this needed its own slice, and that
claim is narrower than `1e-200`-style examples suggested.** The public API squares the accumulator's norm, so
for a tensor whose norm is representable but whose square is not, `norm_squared` returns `0` in the
compatibility route as well; and `norm` is defined as the square root of that same squared value, so it
returns `0` too. The accumulator still matters for rounding and for in-range magnitudes, and this route
reproduces it exactly rather than summing squares itself, but the record no longer claims a visible
representability difference.

The dtype domain was the shared narrow one when this slice landed; §4.6 widens it for the norms, because their
whole result is a plain `f64`. The refusal is the shared adaptation's rather than the compatibility route's
`ScalarTypeMismatch`, and diagonal or structured payloads are refused for the same reason as in the other
slices.

Evidence (`crates/tensor4all-core/tests/f1_session_norm.rs`, 4 tests): exact parity with `norm_squared` and
`norm` for `f64` and `c64` dense operands, plus independently known results (`[3, 4]` and `[3 + 4i]` both 25);
**a discriminating fixture**: for `[1.0, 1.0]` the accumulator squares `sqrt(2)`, which is not exactly the
naive sum `2`, and the session route reproduces the accumulator's value while sixteen copies of `1e-162` -
whose individual squares underflow - give a representable non-zero result, so this is not a comparison
between two implementations that both sum squares; representation and nonfinite coverage: a fresh
`from_storage` operand, rank-zero real and complex operands, an empty payload, a zero-length axis, a
non-contiguous eager operand, several infinities and a complex infinity (all agreeing with compatibility), a
NaN in a complex component refused through both entry points, and `f32` refusals under this entry point's
name through both; the diagonal and tracked refusals with the same operation name; and repeated calls inside
one held session agreeing.

## 4.6 Slice 7: which dtypes a route can serve follows from its result carrier

The shared operand adaptation now takes a dtype policy, and the norms use the wider one: `f32` and `c32`
operands are served there, promoted into the same `f64` accumulation the compatibility route already uses for
them. Everything else keeps the narrow policy, because the result is published into an `IdxTensor` over
compact storage, which holds only `f64` and `c64`:

| route | result carrier | dtypes its numerical path admits |
| --- | --- | --- |
| `norm_squared_on_session`, `norm_on_session` | a plain `f64` | `f32`, `f64`, `c32`, `c64`, the narrower ones promoted exactly as compatibility promotes them |
| `sum_on_session` | an `AnyScalar` over a rank-zero `IdxTensor` on compact storage | `f64`, `c64` |
| `inner_product_on_session` | an `AnyScalar` over a rank-zero `IdxTensor` on compact storage | `f64`, `c64` |
| `contract_pair_on_session` | a dense `IdxTensor` over compact storage, with the planned result indices | `f64`, `c64` |
| `permute_on_session`, `permute_indices_on_session` | a dense `IdxTensor` over compact storage | `f64`, `c64` (an *identity* permutation stays metadata only and accepts any dtype, as §4.3 and §4.4 record) |

**This is conservative per-operand admission for compact-storage publication, not "accept whenever the
resulting dtype would fit".** A mixed `f32`/`f64` pair would contract to `f64` on the backend, which compact
storage can hold, and this slice still refuses the narrow operand rather than admitting mixed widths on those
routes. The refusal therefore says what a caller can act on - "this numerical route accepts f64 and c64
operands for compact-storage publication, not `f32`" - instead of implying that the public result type cannot
represent the dtype. Widening the carriers themselves, or admitting mixed widths, is **not** part of this
slice, and §4.5's earlier statement of the narrower norm domain is superseded here.

Evidence (`crates/tensor4all-core/tests/f1_session_narrow_floats.rs`, 4 tests): the norms serve `f32` and
`c32` with exact parity against `norm_squared`/`norm` (whose compatibility values come from the same promotion);
`sum` refuses `f32`/`c32` under its own operation name although compatibility serves them, so the refusal is
this seam's carrier limit and not the input's; the tensor-publishing routes refuse them too, for the same
reason, while a scalar route with an `f64` carrier keeps working afterwards; and `c64` still takes the wide
path unchanged.

## 5. Non-goals and explicit non-claims

- No change to `IdxTensor`'s representation, no new storage kind, no change to the compatibility route's
  numerics or routing.
- Not F1 complete: `Matrix`, structured payloads, *remaining* reductions and the other operation families
  are later slices with their own parity evidence.
- No AD, no CUDA device-resident contraction, no session-taking variant for metadata operations.
- No benchmark and no speedup claim: this slice is the seam. F2 measures.
- If a later slice replaces the `IdxTensor` result with a concrete carrier, that is an API change and will
  be described as one.

## 6. Open questions

1. Should `Matrix`/structured coverage land before or after F2's first measurement? F2 only needs the
   dense `f64`/`c64` seam, so this slice serves it either way.
2. Should the deferred-error precedence also apply to the compatibility route? This slice only changes its
   own entry point; the same masking exists there and is not addressed here.
