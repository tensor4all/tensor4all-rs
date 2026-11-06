# F5 slice 1: component-wise transfer and per-node readiness for TreeTN (#553 / #859 F5)

`#859`'s F5 row asks to "migrate the existing #675 vertical slice and component-wise network transfer with full
Index/structural readiness preservation ... and separate compute/upload/download/wait/overlap measurements". The
feature builds and its phases are measured (#926, #928); what is missing on this route is the *component-wise*
part: today a caller can only upload or download a whole network, and cannot ask where one node lives.

## 1. What exists (verified in this checkout)

- `TreeTN::upload_cuda` / `TreeTN::download` already walk the nodes one by one internally, stage every
  replacement, and then rewrite the cloned metadata graph, with typed per-node errors
  (`CudaTreeTNError::{MissingTensor, Upload { node, source }, Replacement { node, source }}`).
- `IdxTensor::validate_cuda_residency(&CudaExecutionContext)` answers "is this value resident in that exact
  context" with typed outcomes (`Resident`, `ForeignContext`, `MissingEagerInner`), and the existing tests assert
  that upload/download preserve the network's indices, bonds and node names.
- Nothing exposes a *per-node* transfer or a *per-node* readiness query, and the route has no stream or
  asynchronous-transfer surface, so "wait" cannot be told apart from "work" and "overlap" cannot be measured at
  all (recorded in the tutorial).

## 2. Design

### 2.1 Component-wise upload (revision 2, after pre-review)

```rust
impl<V: Clone + Eq + std::hash::Hash + std::fmt::Debug + Ord + Send + Sync> TreeTN<IdxTensor, V> {
    /// Upload the named nodes into `context`, leaving every other node untouched.
    pub fn upload_nodes_cuda(&self, nodes: &[V], context: &CudaExecutionContext)
        -> Result<Self, CudaTreeTNError>;
}
```

Selection rules, all resolved **before** any transfer starts: an empty selection returns a clone of the input; a
name with no node is a new typed outcome `CudaTreeTNError::UnknownNode { name: String }` (the existing
`MissingTensor { node }` needs a `NodeIndex`, and `MissingRoot` means something else); duplicate names are
deduplicated; a node already resident in **this** context is left as it is (the goal is residency, and uploading it
again is rejected by the backend's host-source boundary); a node resident in a **foreign** context is an error
rather than a silent re-upload, because it needs an explicit download first.

The contract is deliberately **upload-only staged assembly**. A partly uploaded network is metadata-accessible -
structure, names and untouched nodes are all readable - but it is **not CUDA-contractible**: the existing
`contract_to_tensor_cuda` validates every node and must keep rejecting mixed host/device networks. Selective
download is deliberately not added: no caller on this path needs it, and `node_index` → `tensor` →
`IdxTensor::download` → `replace_tensor` already expresses it.

All-or-nothing holds for **values and network structure**, not for every observable detail: the source's tensors
can gain a lazily initialized CPU eager cache during a failed call (the upload boundary materializes a host value
first), and because `IdxTensor`'s cache is an `Arc<OnceLock<…>>` shared by clones, cloning earlier would not
prevent it. That is stated rather than papered over; nothing of the caller's values or graph is changed.

### 2.2 No public readiness query

Revision 1 proposed `node_placement`. The pre-review showed it is redundant - per-node residency is already
reachable through `node_index` → `tensor` → `IdxTensor::validate_cuda_residency` - and that its proposed states
were wrong: a value built by `from_dense` is adopted into the process-global CPU eager runtime, so an untouched
fixture node reports `ForeignContext`, and `MissingEagerInner` means "no eager representation", not "not resident".
The method is dropped; tests assert residency through the existing validator and describe it as **eager-context
residency**, not physical placement.

### 2.3 What is preserved, and why

`replace_tensor` does not leave metadata untouched in general: it accepts changed site indices and rewrites the
site-index network. What makes this upload preserve structure is narrower and has to be stated as the actual
invariant: `IdxTensor::upload_cuda` **reconstructs the value with the original ordered indices**, and the
replacement then re-registers exactly those indices. So structural preservation is a property of the upload path,
not of `replace_tensor`. The tests assert it directly - ordered full-index equality (including indices that share
a dimension but differ in prime level or tags), site membership and reverse lookup, edge endpoints and bond
indices, and node names - rather than arguing from "only tensors are replaced".

## 3. Contracts

- **Upload-only, staged assembly**: values and network structure preserved; partly resident networks stay
  metadata-accessible and are refused by the contraction entry, as before.
- **Selection is validated before transfer**: unknown names, duplicates and foreign-resident selections have the
  described outcomes, so a call cannot fail halfway through selection.
- **Atomicity for values and structure**: a failure leaves the caller's values and graph untouched; the sole
  documented side effect is lazy CPU eager-cache initialization inside the source tensor.
- **No change to existing entries**: `upload_cuda` and `download` keep their signatures and behaviour; the
  whole-network entry may be expressed in terms of this one internally.
- **Preparatory, not the async migration**: this slice uses the compatibility eager transfer route. It is not
  evidence that F5's async/session transfer migration is complete.

## 4. Evidence plan

1. **Component-wise transfer and staging**: upload nodes one at a time on a fixture whose tensors were built so
   that some are plain host values and some are adopted CPU-eager values; after each step assert that the selected
   nodes validate as resident in this context, that the untouched ones are unchanged (their eager-context
   residency is whatever it was, asserted through the existing validator), and that the selection rules hold
   (empty selection, unknown name, duplicate name, already-resident node).
2. **Structure**: ordered full-index equality including an index pair that differs only in prime level or tags,
   site membership and reverse lookup, edge endpoints and bond indices, and node names, for both the partial and
   the whole-network route.
3. **Equivalence**: uploading every node through the partial entry gives the same structure and the same
   contraction result as `upload_cuda`, still matching the CPU reference within `1e-10`; a partly uploaded network
   is still refused by the contraction entry.
4. **Measurements** (example + tutorial): report the **individual component upload times and their sum**, the
   whole-network upload as the comparison, and the cold-versus-warmed component-call cost, and say which
   initialisation and bookkeeping each figure includes. **Measured, and it changed the record:** on the A100 the
   first per-node upload carries 427.42 ms of first-use initialisation while the remaining five cost 0.05-0.09 ms
   each, a warmed pass costs 0.32 ms in total, and the whole-network upload costs 0.36 ms then 0.28 ms. So the
   hundreds of milliseconds the tutorial previously attributed to "upload" are initialisation, not data movement;
   the tutorial now says so, and says that each per-node figure also contains a network clone, native duplication,
   session entry and eager wrapping. Readiness assertions stay outside the timed regions, and
   where a first-use initialisation can move into a reported phase, say so rather than attributing the time to
   DMA. Pin and record an explicit single-threaded CPU configuration for the reference row.
5. **Gates**: fmt, clippy with the repository flags, the crate's tests with and without the feature, rustdoc
   `-D warnings`, `check-public-error-docs.py`, the panic audit, `test-mdbook.sh` and the repository-rules review;
   the CUDA tests run on the local A100 with model, driver and toolkit recorded.
6. **Tutorial correction**: `cuda-tree-contraction.md` currently implies component-wise transfer is what would
   make wait/overlap measurable. It is not - this slice adds no stream or pending surface - so the text is
   corrected to distinguish *synchronized compute latency* from *separately observable asynchronous wait and
   overlap*, and the F5 async migration stays named as the step that would provide the latter.

## 5. Non-goals

- Streams, pending handles, readiness events and therefore wait/overlap measurement (needs the upstream surface
  to be migrated first; recorded, not claimed). This slice must not be cited as evidence that F5's async transfer
  migration is done.
- Migrating the #675 vertical slice's application-level route, which is a separate F5 step.
- Any change to the CPU routes or to the compatibility entries.
