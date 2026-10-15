# F5 slice 3: the TreeTN-level pending download surface (#553 / #859 F5)

F5's row asks for component-wise network transfer **and** separated upload/download/wait/overlap measurements.
Slice 1 gave the TreeTN the component-wise *synchronous* upload with its per-node readiness validator
(`docs/design/859-f5-component-transfer.md`, merged in #929). Slice 2 gave the *tensor* level its pending routes
(`download_pending` in #932, `upload_pending` in #973, with the overlap figures those tests print). What is still
missing is a route a caller can use to submit a whole network's device-to-host transfer and wait for it
separately, so at the network level wait is still not separable from work.

**Revision 2, after an independent pre-review (openai/gpt-6.1-sol) rejected revision 1.** Revision 1 proposed two
public routes (a pending upload and a pending download at the TreeTN level) plus a destination-ownership argument
and a "no public device-allocation entry" claim. The review rejected the upload half as unnecessary for this row,
rejected the allocation claim as falsifiable, and required the error mapping, the timing boundaries and the
readiness evidence to be stated rather than assumed. Revision 2 is the download-only surface, with those
corrections applied; the review's findings are the basis for §2 to §5.

## 1. What exists (verified in this checkout)

At the TreeTN level, all synchronous: `upload_nodes_cuda` (component-wise, transactional, skips resident nodes),
`upload_cuda` (whole network), `download` (whole network, back to host), `contract_to_tensor_cuda`.

At the tensor level (slice 2), pending and separable: `CudaExecutionContext::{allocate_pinned, download_pending,
upload_pending}` with `PendingCudaDownload` / `PendingCudaUpload` (`is_ready`, `wait`), and the `IdxTensor`-level
entries in `tensor4all-core` that the synchronous TreeTN routes already go through.

## 2. Scope: one route, and why not two

The F5 row requires component-wise network transfer, preserved structure, the retained example and features, the
tutorial, and separated compute/upload/download/wait/overlap measurements. It does not require two new public
TreeTN pending APIs, and the pre-review was explicit that a resident-only pending re-upload route would not supply
evidence the row is missing: a pending upload can only overwrite a node that is *already* device-resident, so it
adds no transfer capability the synchronous component-wise upload does not already have, and the upload half's
wait/overlap figures already exist at the tensor level (#973).

So this slice adds **one** route - the whole-network pending download - and records the rest as it is:

- the component-wise *synchronous* upload stays as slice 1 left it;
- the tensor-level pending upload and its figures stay as slice 2 left them;
- **no** TreeTN-level pending upload, and therefore no destination-ownership rule, no allocation claim, and no
  resident-only restriction. If acceptance later demands an asynchronous *network* host-to-device route, it is
  remaining work to be designed on its own evidence, not something this slice quietly claims.

## 3. Design

```rust
impl<V> TreeTN<IdxTensor, V> {
    /// Submit a pending device-to-host transfer of every node, without waiting.
    pub fn download_pending(&self, context: &CudaExecutionContext)
        -> Result<CudaTreeTNPendingDownload, CudaTreeTNError>;
}

impl CudaTreeTNPendingDownload {
    /// Whether every submitted transfer's event has completed.
    pub fn is_ready(&mut self) -> Result<bool, CudaTreeTNError>;
    /// Wait for all of them and return the host network.
    pub fn wait(self) -> Result<TreeTN<IdxTensor, V>, CudaTreeTNError>;
}
```

Contracts:

- **No partial network is published.** Every node is preflighted (topology, tensor presence, device placement in
  this exact context) before the first submission, and `wait` either returns a complete host network or fails -
  it never returns a network with some nodes still device-resident. This is the honest version of "transactional":
  once submissions have started, the *submissions* cannot be undone, only the network publication can be withheld.
- **Errors reuse the existing node-local variants.** `CudaTreeTNError::Download { node, source }` already carries
  the failing node and an `IdxTensorCudaError`; the pending route maps its failures through the same field, so no
  new TreeTN variant is introduced for a condition the synchronous route already reports. The limitations are
  stated rather than implied: a **submission** failure surfaces as the tensor-level route's diagnostic-only
  `Transfer` error (the recoverable package is neither `Send` nor `Sync`, so it is already discarded below this
  level and is not promised back), while a **wait** failure preserves `CompletionUnproven` as a typed source.
- **`wait` returns the network, not a partial one** (above): a node whose completion is unproven makes `wait` fail
  with that typed source, and the caller discards the buffers rather than reading them.
- **Drop is not wait, with the runtime-teardown exception stated.** Dropping neither proves a copy nor
  synchronizes *handle detachment*; reclamation additionally needs the copy to have completed and the runtime's
  bounded retirement queue to admit it, and releasing the **last** runtime owner is a different event that does
  synchronize initialized streams. That is the wording the tensor-level handles already use; this slice does not
  add another backend clone, because the contained pending handles already retain the runtime.
- **No readiness granularity beyond "all"**, and no per-node waiting: nothing in this row needs them.
- **Synchronous entries unchanged**: `upload_nodes_cuda`, `upload_cuda` and `download` keep their signatures and
  behaviour.

## 4. Evidence plan

1. **Parity**: the pending route's host network equals `download`'s, structurally - the node's graph index and
   the **ordered `DynIndex` identity** of every index on its tensor, so a relabelling between two indices that
   share a dimension and prime level is visible - and numerically, for the float and complex dtypes with unequal
   real and imaginary parts and an index order that differs between nodes.
2. **Submission is separable from completion**: measure submission and wait separately on a network large enough
   to be visible, and record a `is_ready == false` observation as **hardware evidence, not a correctness gate** -
   a nonblocking copy can complete before the host is scheduled to query it, so a bounded retry cannot guarantee
   the observation even when the implementation is correct.
3. **Two networks submitted before either is waited on**, with identical timing boundaries on both sides of the
   comparison (preparation, submission, wait and host reconstruction named separately; the same work excluded
   from both).
4. **Refusals**: an empty network, a missing tensor, and a node resident in a *different* context are refused
   with the existing typed errors. The placement and tensor checks run before the first submission, so such a
   call transfers nothing; a submission failure part-way through the loop is reported with its node and leaves the
   caller's network untouched.
5. **Gates**: fmt, clippy with the repository flags, the crate's tests with and without `tenferro-cuda`, rustdoc
   `-D warnings`, `test-mdbook.sh`, the API dump, and the CUDA tests on the local A100 with driver and toolkit
   recorded.

## 5. Non-goals

- A TreeTN-level pending **upload**, per §2, and any device-allocation entry it would need.
- Per-node readiness or per-node waiting.
- Any change to the synchronous routes, the CPU routes, or the contraction entry.
- A claim about how much a real workload gains from the overlap before the measurement in §4 exists.
- Re-stating what the tutorial says today: it already documents both pending tensor-level directions, so §4's
  measurement section updates it with the *network* figures rather than correcting a stale claim.
