# F2 — the first measured non-AD workload: simplett's TT inner product on a held session

Design for the F2 package of [tensor4all-rs#859](https://github.com/tensor4all/tensor4all-rs/issues/859)
(paired with [#510](https://github.com/tensor4all/tensor4all-rs/issues/510)). It is the first app-track
package after F1 slices 1 and 2 (#902, #903), and it exists to show what holding one explicit session across
a real algorithm region buys - or that it buys nothing.

**Revision 3**, adding the measurement §4.1. **Revision 2**, after an independent pre-review: revision 1's measurement plan covered only two arms while
F2's recorded identity asks for direct, bridge-inclusive and eager comparisons separately; it promised a
rank-zero finish the route's equations never produce (they end at a 1x1 environment); it named a borrowed-read
API that is not the pinned one; and its parity tests could pass with axis-order or conjugation bugs. Revision 2
answers all four. The pre-review's own findings are the basis for §3 and §4.

## 1. The target, and what it does today

`tensor4all_simplett::SimpleTensorTrain::inner_product` walks the two trains site by site:

- site 0: `einsum_tensors("asr,ast->rt", &[a0.as_inner(), b0.as_inner()])`, whose result is converted into a
  `Matrix` (`contraction.rs:82-120`);
- each later site: the `Matrix` is converted **back** into a `TypedTensor` with
  `typed_tensor_from_col_major_slice`, then `einsum_tensors("ij,isk,jsl->kl", &[&env, a, b])`, whose result is
  converted into a `Matrix` again (`contraction.rs:131-160`).

So the loop performs `2N - 1` matrix/tensor conversions for a length-`N` train: `N` conversions of a
contraction result into a `Matrix`, and `N - 1` conversions of that matrix back into a `TypedTensor` for the
next site. `einsum_tensors` additionally duplicates each operand before every contraction - the duplication is
simplett's own helper's choice, not a requirement of the backend signature, which accepts borrowed operands.
Nothing in the loop holds a session: each einsum enters and leaves the compatibility route on its own.

## 2. What F2 adds

One opt-in entry point in the same module, with the existing one untouched:

```rust
impl<T: TTScalar + Scalar + Default + EinsumScalar> SimpleTensorTrain<T> {
    /// Compute the inner product with another train on a caller-held explicit session.
    pub fn inner_product_with_session(
        &self,
        other: &Self,
        session: &mut tensor4all_tensorbackend::explicit::Session<'_>,
    ) -> Result<T>;
}
```

Shape of the change:

- **One session across the whole site loop.** The caller opens it (or holds it across a wider region), and the
  entry point performs no session entry of its own.
- **Borrowed operands, no frontend payload duplication.** The site tensors are read through the pinned
  `TensorScalar::tensor_read` on the wrapped `TypedTensor<T>`, which builds a view over existing storage, and
  the running environment is read through `TensorRead::from_tensor`. "No copy" means no frontend payload
  duplication: output allocation, metadata, packing and backend scratch are unaffected.
- **No per-site matrix bridge.** The environment stays a concrete `Tensor` between sites; the `Matrix`
  conversion happens only at the API's edges, if at all, so the two conversions per site disappear.
- **The scalar is read once at the end, from the 1x1 environment the equations produce.** The last
  contraction keeps both right-bond axes (`"ij,isk,jsl->kl"`), so the loop ends with a 1x1 result and the
  compatibility route reads `[0, 0]`. This design keeps that shape and reads the single element through
  checked host access (`T::tensor_read`-free `as_slice::<T>()` plus a length check), not through a final
  reduction: `Session::sum` would sum every entry of a malformed environment where the compatibility route
  reads the first, so the two would disagree on invalid input. The 1x1 condition itself holds because the
  train constructors require unit boundary bonds; a core replaced through `site_tensors_mut` can break it,
  and the checked length is what turns that into a typed error instead of a silent difference.
- **The same validation, in the same order**: train-length mismatch and site-dimension mismatch are refused
  with the same `SimpleTensorTrainError` variants and messages as the compatibility entry point.

## 3. Implementation contract

- The dtypes are the compatibility entry point's: `f64`, `f32`, `Complex64` and `Complex32`. The scalar is
  extracted through `Tensor::as_slice::<T>()` behind a length check, so every one of them is preserved rather
  than routed through a wider type.
- **The semantics are simplett's, not F1's.** This is a *bilinear* product with no conjugation of either
  operand, unlike `tensor4all_core::inner_product_on_session`, which conjugates the left operand. The tests
  below use genuinely complex data precisely because they must not share the Hermitian oracle.
- Validation happens in the compatibility order and with its error variants: train-length mismatch, then
  (for a non-empty train) the site-dimension mismatch at site 0 and at each later site.

## 4. Evidence plan

1. **An independent dense oracle**, not route parity. For small trains whose dense expansion is cheap to
   build in the test, compute the bilinear product sum over all site indices and compare. Cases: one site
   (the first-site contraction is the whole computation), two sites, three or more sites, unequal bond
   dimensions between the two operands, and asymmetric non-constant data. One-site complex vectors with
   non-zero imaginary parts make an accidental conjugation visible (`[1+i, 2-i]` against `[3+2i, -1+i]` is
   `8i`, while conjugating the left one would give `2`).
2. **Route parity with `inner_product`** on the same cases, exactly for `f64`/`f32` and within tolerance for
   the complex dtypes, plus the empty train (`T::zero()`) and a length mismatch.
3. **Session behaviour**: several inner products inside one held region equal the separate calls, a refusal
   (different lengths, then a site-dimension mismatch at site 0, then at a later site) leaves the session
   usable, and the values are identical to the compatibility entry point's.
4. **Measurement**, in `crates/tensor4all-simplett/benches/`, criterion, with the arms and boundaries named
   before the run:
   - **(a) compatibility, eager and bridge-inclusive**: the existing `inner_product`, which the compatibility
     route runs as an untracked eager computation and which pays the `2N - 1` matrix/tensor conversions.
     This arm *is* the untracked eager arm; it is named as such rather than omitted.
   - **(b) held explicit session**: `inner_product_with_session` with the session opened **around** the timed
     region by the caller, so the number is the repeated computation inside a held region, not session entry
     plus one computation. Session entry is reported separately where it is measured at all.
   - **(c) direct concrete**: the same site loop inlined in the bench, calling `Session::einsum_reads` without
     the TT-level method, so the difference between (b) and (c) is the TT-level wrapper and nothing else.
   - **Payload-copy bridges on the new route: none.** Both operands are already concrete `TypedTensor`
     values in the train, so the new route has no input adaptation boundary to time. What it removes is the
     old route's `2N - 1` conversions, and (a) versus (b) is where that shows; the design does not label that
     difference as an isolated bridge cost.
   - **Protocol**: both arms configured to one effective worker and the thread count verified before
     measuring (the compatibility context otherwise follows the machine's affinity); paired runs over the same
     trains; several lengths and small bond dimensions so fixed entry cost and per-site cost are
     distinguishable; warm-up treated as warm state, and "cold" not claimed for ordinary criterion samples.
     One `Session::einsum_reads` call builds its own plan, so "warm" here means engine and resources, not a
     cached plan.
   - **Comparator**: the independent oracle above for small sizes, and the compatibility entry point for the
     sizes where an oracle is impractical. Outputs are validated outside the timed region.
5. **Worker sweep and first call**: the held arm is measured at 1, 2 and 4 workers, and one first call is
   reported next to a warmed call (§4.1.1). What remains out of scope is named in §4.2.
6. Repository gates: fmt, clippy with the repository flags, rustdoc `-D warnings`, the crate's tests and
   doctests, `scripts/repository-rules-review.py`, and the panic audit (F1 slice 2 was failed by an
   `unreachable!` there).

## 4.1 Measured (this machine, one worker per arm)

`cargo bench -p tensor4all-simplett --bench inner_product_session`, criterion, profile `bench` on
`feat/859-f2-prerequisites` (commit 2ec68cd9's tree plus this slice's corrections), AMD EPYC 7713P,
Linux 6.8, one worker per arm: the explicit arms on a `CpuBackend::with_threads(1)` and the compatibility
arm on the process-global context, which the bench forces to one worker before it initializes and verifies
before measuring (`compatibility workers=1, explicit workers=1`). Three trains of physical dimension 4 with
bond dimension 4; the two operands of each pair differ. Every arm is validated against the compatibility
value outside the timed region. Displayed central estimates (criterion's 50-sample estimates, not medians):

| sites | compatibility (compiled-graph route) | held session | direct session |
| ---: | ---: | ---: | ---: |
| 4 | 273.2 us | 87.4 us | 88.3 us |
| 8 | 531.9 us | 172.3 us | 172.5 us |
| 16 | 1.0572 ms | 340.3 us | 341.3 us |

What the table supports:

- **The explicit route is about 3.1-3.2x faster per call** than the compatibility route at these sizes
  (3.13x, 3.09x, 3.11x for the run above), and the ratio is flat across lengths, so the difference is
  per-site work rather than a fixed entry cost. An earlier run of the same arms agreed within a few percent.
- **The TT-level method and the same loop inlined are within about one percent** of each other at every size
  (88.3 against 87.4 us, 172.5 against 172.3 us, 341.3 against 340.3 us). The win is the route, not the
  wrapper.
- **Naming what is being compared.** The compatibility arm is the compatibility route: it traces and runs a
  compiled graph on the process-global context, duplicates each operand per contraction, converts a
  contraction result into a `Matrix` and back at each site (`2N - 1` conversions) and builds a plan per
  contraction. The explicit arms run the caller's concrete session, which also builds one plan per
  `einsum_reads` call, reads the site tensors through borrowed views and keeps the environment concrete. The
  measured difference covers all of that together.

What the table does **not** support, and the record does not claim:

- It does not isolate the benefit of *holding* a session: both explicit arms hold one, and there is no
  scoped-explicit arm holding everything else constant.
- It does not isolate a session-entry cost, a planning cost or a bridge cost. Two whole-route timings cannot
  separate them; the inlined-loop arm only bounds the wrapper's own overhead.
- It is not an untracked-eager-frontend comparison. An `EagerTensor`-frontend arm at the same sizes is
  **deferred**, not delivered here.
- It is not a scaling result: §4.1.1 shows this workload does not scale with workers at these sizes, and
  reports the first call next to the steady state without attributing either.

### 4.1.1 Threads and the first call

Measured on `feat/859-f2-thread-and-cold-arms`, the same machine and one-worker setup as §4.1 for the
compatibility arm; criterion's displayed central estimates for the worker sweep, and single `Instant`
observations for the first call. Two further groups in the same bench answer the rest of F2's recorded
identity.

**Workers** (`simplett_inner_product_threads`, 16 sites, bond 4, held session, worker count verified per
context):

| workers | held session |
| ---: | ---: |
| 1 | 340.8 us |
| 2 | 341.9 us |
| 4 | 343.8 us |

**No benefit from additional workers was observed for this size**: with 16 sites and bond 4 the per-site
contraction is a 4 x bond x 4 tensor operation, and the two- and four-worker numbers are within about one
percent of the one-worker one (one worker 340.8 us, two 341.9 us, four 343.8 us). This is one size, and it is a
*configured worker budget* that was verified, not proof that no worker performs useful work. Read together
with the source - sequentially dependent site contractions, small per-site operands, and one plan and one
operand adaptation per `einsum_reads` call - it is consistent with substantial planning and dispatch overhead
in this workload, which the sweep does not decompose. The compatibility arm cannot be measured at other
worker counts in this bench, because the process-global context fixes its worker count when it first
initializes; the one-worker comparison above is the only like-for-like one.

**First call versus steady state** (`inner_product_first_call`, one worker, fresh context per size, printed
rather than benchmarked so that criterion cannot average the first call away):

| sites | first call on a fresh context | steady state |
| ---: | ---: | ---: |
| 4 | 141 us | 91 us |
| 16 | 391 us | 349 us |

These are two single observations, not a repeatable cold-start measurement: the first call is taken once,
with backend construction and session opening excluded from the timer and process-wide allocator, code and
cache state not reset, and the second number follows three warm-up calls. The first call costs about 1.55x
(four sites) and 1.12x (16 sites) the warmed call here - an earlier run of the same code gave 1.65x and
1.11x - and whether that is a stable first-call penalty, and
whether it comes from plan construction, allocator growth or kernel/engine warm-up, is not established.

## 4.2 Explicitly partial scope

F2's recorded identity mentions 1-thread and multi-thread runs, cold/warm pairing and a reference
comparator. This slice delivers the one-worker route comparison, the worker-count sweep and the first-call
measurement above, with the independent dense oracle as the comparator for small sizes. What remains out of
scope, and is named rather than implied: an **untracked-eager-frontend arm** (the workload's loop has no
eager-frontend path without restructuring it; the compatibility arm is a compiled-graph route, not that
arm), a warm/cold *allocation-pool* protocol beyond the first-call number, and any comparison against the
Julia reference implementation. The parity and oracle evidence in §4 supports the values; the timing tables
support the route and worker-count comparisons they were designed for.

## 5. Non-goals and explicit non-claims

- `inner_product` and every other public entry point keep their signatures, routing and numerics. No new
  free-function wrapper: the existing `contraction::inner_product` stays the compatibility spelling.
- No new feature flag and no new scalar-conversion abstraction: the crate's dependency features already
  provide the explicit context, and `TensorScalar` covers extraction.
- **No optional-AD comparison**: the new route is the plain concrete one and creates no tape. An untracked
  eager baseline *is* measured, as arm (a); the tracked/differentiated comparison belongs to the U2-AD gate
  and is not approximated here.
- No migration of `tensorci`, `treetn` or any other consumer: F2 is one workload, migrated and measured.
- No new backend capability: `Session::einsum_reads` and `Session::sum` already exist, and no kernel changes.
- Not a claim that the removed conversions are the dominant cost: the record reports the measured difference
  and the mechanism the source shows, not a causal explanation the measurement cannot support.

## 6. Post-fix status, the before/after envelope, and the regression note

[#510](https://github.com/tensor4all/tensor4all-rs/issues/510)'s acceptance criterion is a reproducible
tensor4all benchmark that shows the small-chi TT inner-product gap before and after the tenferro-side fix
([tenferro-rs#872](https://github.com/tensor4all/tenferro-rs/issues/872), closed 2026-05-21). This section
records where each half of that criterion is measured, and the envelope a future change should be held to.

**The pin is post-fix, so §4.1 measures the after state.** The workspace pins tenferro at
`3f802fa1e0d6f008a22e264c1122fd1f7a24ac44`, committed 2026-10-10 - about five months after #872 closed - so
every number in §4.1 is measured on the fixed backend.

**The before/after is recorded upstream, where the kernels are.** The issue's table is tenferro's pairwise
`site_update` benchmark, and #510's comment of 2026-05-20 records the same points after the fix (one thread,
OpenBLAS backend):

| chi | ITensors median (before) | tenferro normal (before) | tenferro normal (after) | tenferro cached (after) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 3.26 us | 20.38 us | 2.24 us | 0.98 us |
| 2 | 3.72 us | 20.65 us | - | - |
| 4 | 3.78 us | 19.21 us | - | - |

The small-`chi` gap the issue tracked is closed on the tenferro side (roughly 9x at `chi = 1`, and the cached
figure is below the ITensors median). What that leaves on the tensor4all side is **not** the backend entry any
more: it is the route, which is what §4.1 measures.

**The graph/traced route does not avoid the per-call cost - measured, not assumed.** The issue's proposed
work items 4 and 5 ask whether tensor4all's graph compilation or traced execution avoids repeated backend
installs in realistic TT workloads. §4.1 answers it for this workload: the compatibility arm *is* the
compiled-graph route, and it is **3.1x slower** per call than the explicit session (273.2 us against 87.4 us
at four sites), flat across lengths. The source explains the direction - that route traces and runs a compiled
graph on the process-global context, duplicates each operand per contraction, converts a contraction result
into a `Matrix` and back at each site (`2N - 1` conversions) and builds a plan per contraction - so more graph
compilation is not the lever here; the session-held route is.

**The regression envelope to hold.** The explicit route's per-site cost is flat on this workload:
87.4/4, 172.3/8 and 340.3/16 sites give 21.9, 21.5 and 21.3 us per site (physical dimension 4, bond dimension
4, one worker, AMD EPYC 7713P). A future change should be investigated if, on this workload and machine:

- the explicit route's per-site time exceeds **~30 us** (about 40% above the measured envelope), or
- the explicit-to-compatibility ratio falls below **2x** (measured 3.09-3.13x).

This is a note for whoever changes the route, not a CI threshold: the bench is `criterion`-based and
machine-dependent, and §4.2 already names the untracked-eager arm and the Julia comparison as deferred. The
ITensors.jl comparison itself stays upstream, because that is where the same kernels are called and where the
before/after table above lives; a tensor4all-side re-implementation of the Julia reference would measure a
different loop, not this one.

