# F5 slice 2: an asynchronous transfer surface, so wait and overlap become measurable (#553 / #859 F5)

F5's row asks for "separate compute/upload/download/**wait**/**overlap** measurements". Until now every transfer on
this route was synchronous, so wait could not be told apart from work and overlap could not be measured at all
(`docs/book/src/tutorials/gpu/cuda-tree-contraction.md` says so). The substrate for this is now pinned: tenferro-rs
`3f802fa1e` carries U4's pending transfers (#2061, #2065, #2073), and the workspace is pinned there (#931).

## 1. What the pinned substrate provides (verified)

`tenferro_gpu::cuda` exports, in the pinned revision:

- `download_pending(runtime: &CudaRuntime, source: Tensor, buffer: PinnedHostBuffer) -> Result<PendingDownload, TransferSubmitError>`
  with `PendingDownload::{is_ready(&mut self) -> Result<bool>, wait(self) -> Result<(Tensor, PinnedHostBuffer), CompletionUnproven>}`;
- `upload_pending<'dst>(runtime: &CudaRuntime, source: PinnedHostBuffer, destination: &'dst mut Tensor) -> Result<PendingUpload<'dst>>`
  with `PendingUpload::{is_ready, wait -> Result<PinnedHostBuffer>}`;
- `PinnedHostBuffer`, `TransferRejected`, `TransferSubmitError`, `CompletionUnproven`, and
  `CudaRuntime::{drain_transfer_retirements, pending_transfer_retirements}`.

`tensor4all_tensorbackend::CudaExecutionContext` already reaches that runtime (`backend.runtime()`) and wraps the
eager runtime for adoption, so no new dependency is needed.

## 2. Design

Three additions, all on the CUDA context and the CUDA tensor path; no CubeCL type reaches any signature.

```rust
impl CudaExecutionContext {
    /// Allocate a pageable-pinned host buffer for a pending transfer.
    pub fn pinned_host_buffer(&self, bytes: usize) -> Result<PinnedCudaBuffer, CudaExecutionContextError>;
    /// Submit a device-to-host copy without waiting for it.
    pub fn download_pending(&self, tensor: &IdxTensor) -> Result<PendingCudaDownload, IdxTensorCudaError>;
    /// Submit a host-to-device copy without waiting for it.
    pub fn upload_pending(&self, source: PinnedCudaBuffer) -> Result<PendingCudaUpload, IdxTensorCudaError>;
}

impl PendingCudaDownload {
    pub fn is_ready(&mut self) -> Result<bool, IdxTensorCudaError>;
    pub fn wait(self) -> Result<(IdxTensor, PinnedCudaBuffer), IdxTensorCudaError>;
}

impl PendingCudaUpload {
    pub fn is_ready(&mut self) -> Result<bool, IdxTensorCudaError>;
    /// Wait for the copy and adopt the now-populated device value into this context's runtime.
    pub fn wait(self) -> Result<IdxTensor, IdxTensorCudaError>;
}
```

- **The destination is allocated by the wrapper**, not the caller: `upload_pending` owns the device destination
  for the whole pending lifetime, so no caller can observe a half-written device tensor (the upstream entry takes
  the destination by `&mut`; the wrapper allocates it and keeps it).
- **Wait returns an `IdxTensor`**, host-side for a download and device-side (adopted into this context) for an
  upload, exactly as the synchronous entries return them. Adoption happens in `wait`, after the transfer is
  proven complete, so nothing half-transferred can be wrapped.
- **Typed failures keep their sources**: a rejected submission keeps `TransferRejected`, an unproven completion
  keeps `CompletionUnproven`, and both are exposed in `IdxTensorCudaError`'s source chain rather than stringified.
- **Drop is not wait**: dropping a pending transfer leaves the upstream retirement behaviour in charge (the
  runtime drains what detachments left behind), and the wrapper documents that dropping neither waits nor proves
  the copy.

## 3. Contracts

- **One destination per upload**, owned by the pending value; waiting consumes it.
- **Readiness is a query, not a promise**: `is_ready` answers "the submission's event has completed", and `wait`
  is the only way to obtain the value.
- **Synchronous entries unchanged**: `upload_cuda` / `download` keep their signatures and behaviour; this is an
  additional route.
- **No claim about the *shape* of the overlap**: the wrapper exposes submission and completion; whether an overlap
  pays off is the measurement's business.

## 4. Evidence plan

1. **Round-trip parity**: submit an upload, `is_ready` until true (bounded), wait, and check the device value
   against the host original; then submit a download of that value, wait, and check the host value and the
   returned buffer's length. Same numbers as the synchronous route within `1e-12`.
2. **Wait is separable from submit**: measure submission time and wait time separately on a transfer large enough
   to be visible, and assert that the submission returns before the copy completes (the point of a pending API) -
   by observing `is_ready == false` at least once on a large transfer, with a bounded retry budget rather than a
   fixed sleep.
3. **Overlap**: submit two downloads of distinct tensors, then wait both; report the wall time against the sum of
   two separate submit-and-wait round trips, which is what makes "overlap" measurable rather than asserted.
4. **Failure paths**: a byte-count mismatch is a typed rejection with the original source; a wrong-context source
   for `download_pending` is refused before submission.
5. **Measurement + docs**: an A100 run recorded in the tutorial, which also stops saying that wait and overlap are
   *not measurable* - it will say what the measured overlap is, and keep the distinction from synchronised compute
   latency.
6. **Gates**: fmt, clippy with the repository flags, the crate's tests with and without the feature, rustdoc
   `-D warnings`, `check-public-error-docs.py`, the panic audit, `test-mdbook.sh`, the repository-rules review; the
   CUDA tests run on the local A100 with model, driver and toolkit recorded.

## 5. Non-goals

- Wiring the pending route into `TreeTN`'s component-wise upload (a follow-up slice; this one is the surface and
  its evidence).
- Changing the synchronous route, the eager adoption rules, or the CPU routes.
- Any statement about how much overlap a real workload gains before the measurement above exists.

## 6. Pre-review of revision 1, and the revision-2 plan

The independent pre-review rejected revision 1 as written. It confirmed the substrate exists and is reachable, and
rejected the wrapper shape on three counts, with four more corrections. The required changes, to be applied before
implementing:

1. **The owning upload wrapper is self-referential.** Upstream `PendingUpload<'dst>` stores
   `Option<&'dst mut Tensor>`, so a wrapper holding both a freshly allocated destination and a pending upload that
   borrows it cannot be built in safe Rust; `Box`/pinning does not fix the lifetime. Choose one concrete model -
   a caller-owned destination with a borrowed pending handle, a scoped submission/callback API, or an upstream
   owned-destination pending upload - and do not manufacture it with unsafe lifetime extension. The upstream
   owned-destination variant does not exist in the pinned revision, so the caller-owned/scoped shapes are the
   available ones.
2. **Method placement crosses the dependency direction.** `CudaExecutionContext` lives in
   `tensor4all-tensorbackend`; `IdxTensor` and `IdxTensorCudaError` live in downstream `tensor4all-core`. Native
   transfer and buffer primitives therefore belong in tensorbackend, and the indexed-tensor methods and host/device
   reconstruction in core, as the synchronous entries already are.
3. **Upload needs dtype, shape and index identity.** A byte count alone cannot say what a buffer holds, which
   destination to allocate, or which indices the value carries; and pinned allocation rejects zero bytes. The
   core-level entry must take a host `IdxTensor` (or explicit validated metadata), with checked byte-count
   arithmetic and a defined zero-length case.
4. **"Drop is not wait" is wrong.** Upstream `PendingUpload::drop` synchronizes its event before releasing the
   destination borrow, and on failure abandons the pinned source and forgets allocation retention; dropping a
   download instead moves its package into the retirement queue, whose queries never wait. The wrapper must retain
   the destination until upstream destruction finishes, discard rather than adopt a destination after failed
   completion, and document the failure-retention behaviour instead of implying "never waits". Buffers hold only a
   weak runtime reference, and releasing the final runtime owner synchronizes initialized streams, so the guarantee
   is about handle detachment, not about runtime teardown.
5. **The rejected submission package does not fit the existing error fields.** `TransferRejected` contains a
   `PinnedHostBuffer` (`!Send + !Sync`, raw pointer) and implements neither `Display` nor `Error`, while the
   existing generic source fields need `Send + Sync`. Decide explicitly whether a rejection preserves the
   recoverable package (a deliberate backend-owned representation plus an explicit core carrier) or only its typed
   diagnostic (the existing `Operation`/`Context::Transfer` machinery). `CompletionUnproven` fits as a typed
   source field as it is. Also: the upload route's submission and wait return ordinary upstream errors, so the
   design must not claim `TransferRejected`/`CompletionUnproven` for it, and note that upstream's own `source()`
   chain stops at those types.
6. **Download's host reconstruction is unspecified.** Upstream `wait` returns the original **device** tensor plus
   the filled pinned buffer, so the host payload must be built from the filled bytes while preserving metadata, and
   decoding/copying/host eager construction have to be accounted for separately from the event wait. Calling the
   synchronous download again would perform a second transfer.
7. **Timing boundaries must be explicit.** `EagerTensor::from_tensor_in` is not metadata-only adoption - it
   creates a semantic contiguous value through a backend session - so it must happen after proven completion, must
   retain the right runtime or borrow the context, and must be measured separately from the event wait.

None of this changes the target: making wait and overlap measurable on this route is still what F5 owes. It
changes the wrapper's ownership model, where the primitives live, and what a rejection preserves.

## 7. API findings from the first implementation attempt (revision 2 must start here)

I started revision 2 as a tensorbackend-only slice (a caller-owned pinned buffer, a caller-owned device buffer,
and pending submit/is-ready/wait wrappers) and stopped when the pinned revision's API turned out not to offer what
that shape needs. The findings are the useful part, so they are recorded instead of a half-built wrapper:

1. **There is no way to allocate the destination of a pending upload.** Upstream `upload_pending` takes
   `destination: &mut Tensor`, so the caller must already own a device tensor, and the pinned revision exposes no
   device-allocation entry at this level (`CudaRuntime` has no `allocate_device_like`; the synchronous
   `upload_cuda` allocates its own destination internally). A pending upload therefore needs either an upstream
   allocation entry, or the wrapper has to obtain the destination from a synchronous upload first - which defeats
   the point. This is the concrete blocker for the upload half.
2. **`Tensor` is not a host-data carrier here.** The pinned `Tensor` (tenferro-runtime) has no `host_data` accessor
   and is not `Clone`, so a wrapper cannot copy a host value's identity and bytes from it directly; the
   synchronous entries go through their own materialization path, which is what a revision-2 design has to reuse
   rather than reinvent.
3. **The pending packages are not `Send`.** `PinnedHostBuffer` (and therefore the rejected submission package and
   the pending handles) carry raw pointers and are neither `Send` nor `Sync`, matching the pre-review's finding
   that a rejection cannot be retained in an existing `Send + Sync` error source. A wrapper may keep a
   `CudaBackend` clone alive for the handle's lifetime (that part works: the pending value does not borrow the
   runtime, only the destination), but the handle itself must stay on its thread.
4. **What does work, verified while trying:** `CudaBackend::runtime()` is reachable through the context's backend
   guard, `PinnedHostBuffer::new(runtime, bytes)` allocates with `as_slice`/`as_mut_slice` accessors,
   `upload_pending`/`download_pending` submit without waiting, `PendingDownload`/`PendingUpload` expose
   `is_ready`/`wait`, and a `CudaBackend` clone can keep the runtime alive for the handle. The download half is
   therefore implementable today; the upload half needs finding 1 resolved first.

So revision 2 should either start with the **download-only** surface (measurable wait and overlap on the
device-to-host side, which is where the transfer is visible at these sizes) or wait for an upstream allocation
entry for the upload side. Starting with download-only is the smaller and independently useful step.
