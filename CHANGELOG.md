# Changelog

## Unreleased (next minor)

### Behavior changes

- TreeTCI's default stop rule is the per-bond rule again (#1405), the one of
  TensorCrossInterpolation.jl, TreeTCI.jl and xfac: the recorded error is the largest
  bond pivot error, and the global search accepts a pivot only above
  `tolerance * tol_margin_global_search` (scaled by the sample magnitude when
  `scale_tolerance` is on). The sampled stop of #1357, which also records the residuals
  the global search measured and keeps observed errors above `tolerance`, stays available
  as `TreeTciOptions { stop_rule: StopRule::Sampled, .. }` (and
  `QtciOptions::with_stop_rule`). On the 16-site quantics chain of the treetci
  full-run golden the sampled stop takes 1.74x the oracle evaluations of the per-bond
  stop (22073 against 12720, six iterations against four); on the 16-site Gaussian
  chain of #1371 it took 3.14x, 3.37x and 2.18x at tolerances 1e-4, 1e-6 and 1e-8
  (six seeds summed), for the same true error. The unit-endpoint probes and subtree
  restarts of #1357, the third search stage of #1433 and the #1434 proposer change stay
  on both rules; under the per-bond rule their finds are accepted only above the
  margin. Runs with the global search enabled can therefore stop earlier, with
  different `errors`, than on the previous release; to keep the previous behaviour, set
  `StopRule::Sampled`. The public `find_global_pivots*` functions keep the sampled
  acceptance.
- `tensor4all-hdf5` no longer sets `HDF5_USE_FILE_LOCKING=FALSE` on first use (#1386).
  The library never touches the process environment now. If you save and then re-open
  the same path in one process (or do so from several threads), set
  `HDF5_USE_FILE_LOCKING=FALSE` yourself before the first HDF5 call in the process
  (shell, service unit, Cargo `[env]`, or `main` before threads start); otherwise the
  reopen can fail with `errno = 35`. Leaving it unset keeps HDF5's cross-process write
  protection.
- The tenferro pin moves to tenferro-rs `5df83672` (tenferro 0.13.0 on threadteam 0.6.1,
  colmat 0.34, cpueinsum 0.13.0, tlinalg 0.12.0, tcontract 0.13.0; #1428). tenferro now
  has one faer-backed CPU backend (tenferro-rs#2609), so the CPU backend features
  `tenferro-cpu-platform`, `tenferro-cpu-faer`, `tenferro-system-blas` and
  `tenferro-provider-inject` are removed from every crate, together with the
  `tensor4all-cpu-platform` crate; there is no alias. macOS no longer uses Apple
  Accelerate. Drop these features from dependency lines (and the
  `default-features = false` that only served them). The direct dense dependency is
  `colmat` 0.34 (the renamed `t4a-faer`). `tensor4all_tensorbackend::explicit::detach`
  takes `&mut dyn tenferro::ExecutionEngine` instead of `&mut dyn Engine`, because
  tenferro materializes an eager value through an eager session.
- `tensor4all-core` no longer pins `tensor4all-tensorbackend/global-defaults` and no
  longer enables the process-global route by default (#1403). It has its own opt-in
  `global-defaults` feature (`backend-tenferro` is an alias); its default is the
  platform CPU provider only, and the provider features of the workspace crates no
  longer imply it. Without it, a context-free operation that would have adopted an
  owner-less value into the process-global eager runtime, or run on the process-global
  engine, returns the new `ExecutionContextEntryError::NoContext` (inside
  `IdxTensorError::EngineEntry` or the operation's own error) instead of choosing an
  engine; values created through a caller-owned context keep running on that context's
  engine. With `global-defaults` nothing changes. `tensor4all-treersi` drops
  `global-defaults` from its defaults (so `hadamard_many` needs it; `hadamard_many_in`
  does not); `tenferro-cuda` (in core, and so in every crate forwarding it) still implies it, because
  CUDA execution keeps host steps with no caller CPU context (core's CUDA-resident SRC
  readback, treersi's rank decision). `tensor4all-hdf5` enables it in its
  library, because its load API takes no context. Callers of the context-free API enable
  `tensor4all-core/global-defaults`. Explicit callers of
  `IdxTensor::src_error_estimate_in` and of structured contractions with host-only
  operands now run the estimator solve and the native einsum on their own context's
  engine instead of the process-global one. New: `src_error_estimate_general_in`.
  `TensorTrain::norm`/`norm_squared` run the packed-site backend kernel on the CPU
  context that owns the chain (new `IdxTensor::with_owning_cpu_engine`) in every build;
  only a chain with no CPU-owned value uses the process-global context (or refuses
  without `global-defaults`). `IdxTensor::scale` places an untracked `AnyScalar` that has
  no tensor of its own in the operand's runtime, so scaling context-owned values works
  without `global-defaults`. In every build, `From<anyhow::Error> for IdxTensorError`
  recovers a wrapped `ExecutionContextEntryError` as `IdxTensorError::EngineEntry`
  instead of `IdxTensorError::Operation`.
- `tensor4all-tensorbackend`: `global-defaults` is no longer enabled by the default
  features (#1377). The default is now `explicit-context` plus the platform CPU
  provider, which also exposes `Matrix`, `Storage`, `TensorElement`, the native
  tensor bridge and the `*_in` linalg entries. Direct dependents that call a
  process-global function (`mat_mul`, `solve_matrix`, `svd_backend`,
  `einsum_native_tensors`, ...) enable `global-defaults` (or `backend-tenferro`).
  `tensor4all-core` still enables it for its dependents. New caller-context entries:
  `hermitian_eigendecomposition_in`, `lowest_hermitian_eigenpair_in`,
  `hermitian_exponential_first_column_in`, `einsum_native_tensors_in`,
  `einsum_native_tensors_owned_in`, `einsum_native_tensor_reads_in`,
  `NativeEinsumPlan::execute_reads_in`.
- `tensor4all-partitionedtreetn`: a certified operator application refused by its
  memory gate is reported as the new `PartitionedTreeTNError::CertifiedMemoryLimit
  { operation, estimated_bytes, limit_bytes }` (#1374). It was a `ResourceLimit`
  whose `value` held the estimate and whose message called that estimate the
  limit, so a 9.7 GB estimate against the 4 GiB default read as "limit of
  9705853056". Callers matching `ResourceLimit { limit: "memory_limit (bytes)", .. }`
  match the new variant instead.
- `tensor4all-partitionedtreetn`: `PartitionedTreeTN::iter`, `values`, `projectors`
  and both `IntoIterator` impls yield patches in canonical patch order (#1366):
  by index signature, then by the patch's projector coordinates. The order no
  longer depends on insertion order or hash-map state, so an order-dependent
  step on the patches, such as a truncated pairwise sum, is reproducible across
  runs. The `IntoIterator` impls now require `V: Ord` and return
  `std::vec::IntoIter`; each call sorts the patch list once.
- `tensor4all-treetn` no longer enables `simplett-bridge` by default (#1335).
  Enable it explicitly for SimpleTT conversions and the bridge's chain helpers.
  Native M6 benchmark builds also exclude legacy comparisons and Fourier
  construction unless their benchmark features are explicitly selected.
- `tensor4all-interpolativeqtt` returns TreeTN chains with ordered physical indices
  for ordinary and patched interpolation (#1334). Pass the tree and its site
  indices to `invert_qtt::<T>` for binary chains. The interpolation crate no longer
  depends on SimpleTT.
- `tensor4all-partitionedtreetn`: `MergeRefineOptions::target_bond_dim` decides
  one coordinate axis at a time (#1328). A level previously probed the joint
  product of every refining axis and then fixed an output bit on all of them or
  on none; it now probes each axis on its own and fixes output bits only on the
  axes whose own children store no more parameters in total than the region's
  merged, unsplit candidate (the #1327 guard, applied per axis; a candidate that
  already fits the goal stops every axis without a probe). A level of
  an `N`-axis schedule therefore splits into `2^(refining axes)` children instead
  of `2^N`, and the stopped axes keep their region's shorter prefix while the
  region's input siblings keep merging on every axis. Single-axis runs are
  unaffected; `MergeRefineReport::stopped_regions` now counts one region-axis
  decision per stopped axis instead of one region, which is the same value for a
  single axis. A region with several axes can now stop every axis at a coarse level
  even though only the joint split would have paid, because no single axis pays on
  its own.
- `tensor4all-simplett`: `evaluate`, `sum`, `norm2`, `norm`, `log_norm`,
  `inner_product`, and `inner_product_with_session` now refuse a tensor train
  whose first core has a non-unit leading bond (#1303, #1340), the
  left-boundary sibling of the trailing-bond guard from #913. They previously
  read only the `l == 0` component and returned its projection; the
  inner-product routes summed every leading component instead.
- `tensor4all-partitionedtreetn`: `ErrorNorm::sampled_max()` pins its
  reference to the largest `|f|` that a greedy coordinate ascent reaches from
  the root patch's candidate samples (#1274), as `optfirstpivot` does in
  TensorCrossInterpolation.jl. It used to take the largest candidate sample,
  which for a peaked function was up to 2711 times too small. This changes
  the tolerance of `sampled_max()` runs.
  - A root whose candidate samples are all zero still fails as before.
  - `sampled_max_with_reference` is unchanged.
- `PivotSearchStrategy::Rook` runs two more rook walks, from columns spread
  over the remaining block, before it accepts a tolerance stop, and takes the
  first pivot they find above the threshold (#1257). On the #1256 ridge and a
  4-site ridge without global pivots, single-walk stops returned `Converged`
  with true errors of 1.3e-6 and 0.39; both now reach the tolerance. The stop
  is still heuristic. A rook walk also moves only to a strictly larger
  magnitude now: on a roundoff-level residual it used to cycle until its step
  limit (rows + columns + 1 steps), because the same entry computed from its
  row and from its column differs in the last bits.
- `PivotSearchStrategy::Rook` is now documented and implemented as a heuristic
  strategy (#1271, #1282). A bond update stops when the current rook pivot is
  below tolerance, and the trailing `pivot_errors` entry (and so the TCI2 bond
  error) is that last rook pivot, not an upper bound on the remaining residual;
  `TCI2Termination::Converged` under `Rook` does not prove the tolerance. The
  exact remaining-residual scan that v0.3.0 ran before every `Rook` tolerance
  stop and at a rank cap (#1258, #1090) is removed, and `Rook` walks now start
  from the bond's previous pivots. `PivotSearchStrategy::Full` stays the
  default and remains certified; use it when a bound is required. The same
  contract applies to `matrix_luci_factors_from_blocks`.
- `tensor4all-treetn` composes operators in one scalar type (#1298).
  `compose_exclusive_linear_operators`, and the extension of a partial operator
  inside `apply_linear_operator`, built their identity gaps and bridges with
  `T::delta`, which is real. A complex operator therefore became a network
  mixing `f64` and `Complex64` tensors, which `PartitionedTreeTN` rejects. When
  any operator tensor is complex, the identities and the real operator tensors
  are now promoted to complex, an exact conversion; real operators are
  unchanged.
- `tensor4all-partitionedtreetn`: `contract_adaptive` and `add_with_patching`
  check each exact group sum before forming it (#1309). A sum whose estimated
  size exceeds `PatchingOptions::group_sum_limit_bytes` (default 4 GiB) returns
  the new `PartitionedTreeTNError::GroupSumLimit` instead of being allocated.
  A global operator, such as a Fourier transform on patches of the transformed
  variable, puts every pair of `contract_adaptive` into one group whose summed
  bond is the sum of all contribution bonds; it previously ran out of memory.
  Such an operator is applied with `schedule_merge_refine` instead.
- `tensor4all-core`: `direct_sum`, `direct_sum_many` and, through
  `TensorContractionLike`, `TreeTN::add` / `add_many` promote operands of
  different dense scalar types instead of refusing them. The lattice is the one
  the eager arithmetic already uses: `f32 + f64 -> f64`, `f32 + Complex32 ->
  Complex32`, `f64 + Complex64 -> Complex64` and `f64 + Complex32 -> Complex64`
  (so a real network and a complex-scaled copy of it add). Operands are cast
  before the block copy, so each block stays exact; a same-type call allocates
  nothing new. Tracked operands are still rejected (#1426).
- `tensor4all-treetci`: `DefaultProposer` (and through it
  `TruncatedDefaultProposer`) expands a vertex of local dimension 1 whose
  candidates are locked (#1352). Such a vertex has no index to expand, so its
  side of an edge offered only the product of its other bonds' pivots, and
  two bonds through it could cap each other below the true rank. A frozen
  three-branch table then stopped `Converged` at tolerance 1e-10 with a dense
  error of 6.3e-10 (seed 4) and 2.5e-8 (seed 6). When that product is below
  twice the edge's current rank, the largest-rank other bond is now replaced
  by its neighbour's own candidates, expanded over the neighbour's local
  index. Above that threshold nothing changes. On the #1153 comb the
  oracle-point cost at matched accuracy is 0.88–1.07x the previous one, and
  the reopened table needs about 25% fewer points. Trajectories on trees with
  such a vertex change, including the pinned #1153 junction ranks.
- `tensor4all-treetci`: the global pivot search has a third stage (#1433).
  When the uniform walks, the retained tail-pair probes and the subtree
  restarts all find nothing, it evaluates, around the same retained tail
  pivots, every configuration of each maximal edge side with at most 32 local
  configurations, without random draws (in batches of at most 2^20 index
  entries, which does not change the selected pivots). On binary quantics
  chains that is the aligned block of width at most 32 around a pivot's value
  of one variable, so a run can no longer stop `Converged` while a residual
  above the tolerance lies in such a block in each of its three trailing
  searches. Simon's box at `R = 7`, which stopped `Converged` with a dense
  error of 0.75–0.875 on 9 of 11 seeds, is now exact on all 11. The budget is
  fixed: the width-64 box at `R = 8` is still missed, and this is not a fix of
  the zero-fiber defect class. Trees without a vertex of local dimension 1 are
  unaffected, and so are runs with `nsearch = 0` or `max_nglobal_pivot = 0`,
  where the stage does not run. On the #1153 comb, accuracy, ranks and termination are
  unchanged and the oracle points grow by 1.32% at `delta = 1` (53,153 to
  53,855) and 0.02% at `delta = 4` (1,125,824 to 1,126,042).

### Added

- `tensor4all_tensorbackend::explicit::Session::batched_mat_mul_same_shape_owned`: the session
  twin of the process-global batched same-shape GEMM, one einsum on the session's engine
  (#1382).

- `tensor4all-tensorbackend` adds `Runtime`, an engine-free runtime that several
  engines share (#1393, stage A2 of #1376). It holds the eager AD runtime, the graph
  compiler and the compiled-program cache, and no engine or engine lock; clones are
  the same runtime. `Runtime::with_eager(exec, f)` and
  `Runtime::run_graph(graph, inputs, exec)` borrow the caller's
  `exec: &mut dyn ExecutionEngine` for one call, so a value made on one `CpuEngine`
  runs on any other `CpuEngine` the runtime supports. `Runtime::check_engine` refuses
  an engine with another runtime id, engine type or allocation domain with
  `RuntimeError::UnsupportedEngine` instead of registering it lazily. With
  `tenferro-cuda`, `Runtime::for_cuda_engine` builds a CUDA runtime (no graph path),
  and `CudaEngine`/`CudaDeviceId` are re-exported. `ExecutionEngine` is re-exported.
  `CpuExecutionContext` and `CudaExecutionContext` are now built on a private
  `Runtime` each, with the same numerics and error messages. One difference: a CPU
  context now builds its graph and eager runtimes together on first use. Any first
  use (`compile_graph`, `run_graph`, `graph_cache_stats`, `eager_runtime`) can
  therefore report an initialization failure of either half (for example
  `eager_runtime` can report `graph CPU engine`), and the `Debug` output reports
  `graph_initialized` and `eager_initialized` together.
- `tensor4all-treetn` adds `TreeTN::estimate_exact_sum_bytes` (#1309), the
  size estimate of `TreeTN::exact_sum` taken before the sum is built.
- `tensor4all-treetn` adds `LinearOperator::to_state_network` and
  `StateOperatorNetwork` (#1298): an operator on any subset of a state's nodes,
  written as a plain `TreeTN` on the state's whole tree, with true input
  indices, identity gaps, bonds routed along the state's paths, and one scalar
  type. It is the operand a `PartitionedTreeTN` contraction such as
  `contract_adaptive` needs, without rewriting indices or promoting dtypes by
  hand.
- `tensor4all-treetn` adds `apply_linear_operator_certified` and
  `apply_linear_operator_certified_to_indices`, with `CertifiedApplyOptions`,
  `CertifiedApplication` and `CertifiedApplyError` (#1286). On a chain state
  they truncate during the zip-up sweep and return a proved bound on
  `||A x - y||`, at most the caller's absolute budget, without forming the exact
  image. The exact factors the certificate needs are estimated before they are
  built, and an estimate above the memory limit (4 GiB by default) is refused.
- `tensor4all-partitionedtreetn` adds `OperatorApplication::Certified`
  (#1286): `schedule_merge_refine`, and so `patched_fourier_transform`, can apply
  the operator to each input leaf with the certified truncation, given a
  required share of the allowance, and charge the proved bounds instead of a
  measured difference. The exact image is never formed, so two interleaved
  axes at R = 20 now fit in memory.

### Breaking Rust API changes

- `tensor4all-aci`: `AciError` has a new `Backend { operation, source }` variant (#1382).
  A matrix product of the ACI sweep that the execution context refuses (for example a
  reentrant session), that the engine rejects, or whose operand shapes disagree is now
  reported as `AciError::Backend` with the original error as `source`, instead of
  `AciError::InvalidInitialGuess`, which is kept for problems with the initial guess.
  An exhaustive `match` on `AciError` needs the new arm. `elementwise_batched_with_rng_in`
  now runs every factorization and matrix product of the workload on the caller's
  context, including the batched frame updates, local block setup and global-pivot
  environments that previously took the process-global route; the context-free entries
  run the same code path on the process-global context. In a build without
  `global-defaults` they return `AciError::Backend` for a run that needs a sweep, after
  input and option validation; the single-site case needs no context and still succeeds.

- `tensor4all-core`: `IdxTensorError` has two new variants, `RuntimeMismatch` and
  `PlacementMismatch` (#1393). The enum is not `#[non_exhaustive]`, so exhaustive
  matches must add both arms. They are reported by the engine-taking operation entry
  that stage B2 of #1376 makes public; no current public operation returns them yet.
- `tensor4all-partitionedtreetn`: `PartitionedContractionOptions` has the new
  field `group_sum_limit_bytes` (#1316). Complete struct literals must add it.
  `contract` and `contract_selected` return `GroupSumLimit` before building a
  sum whose estimate exceeds the limit. The default is 4 GiB, including when
  passing plain `ContractionOptions`. Use `usize::MAX` to disable the limit.
- `tensor4all-partitionedtreetn`: the default group-sum limit is the crate-level
  `DEFAULT_GROUP_SUM_LIMIT_BYTES`, shared by `PartitionedContractionOptions` and
  `PatchingOptions`; `PatchingOptions::DEFAULT_GROUP_SUM_LIMIT_BYTES` is removed.
  Where `usize` cannot represent 4 GiB, the default is `usize::MAX`.
  `contract`, `contract_selected` and `add_with_patching` check every group
  before building any group sum. The `GroupSumLimit` message also suggests
  forming only the needed outputs.
- `tensor4all-core`: `floating_zone_walk` returns `FloatingZoneError<Err>`
  instead of the callback's own `Err`, and validates its arguments before
  evaluating the starting point, as `floating_zone_walk_with_initial_error`
  already did (#1320). A starting point whose length does not match
  `local_dims` used to panic on an internal index; it now returns
  `FloatingZoneError::StartingPointLength` without calling the evaluator. Call
  sites recover the callback error from `FloatingZoneError::Evaluation`.
- `tensor4all-partitionedtreetn`: `PatchingOptions` has the new field
  `group_sum_limit_bytes` (#1309). A complete struct literal must add it; one
  that ends in `..PatchingOptions::default()` is unaffected.
- `tensor4all-partitionedtreetn`: `MergeRefineOptions::apply_options:
  Option<ApplyOptions<V>>` is replaced by `MergeRefineOptions::application:
  OperatorApplication<V>` (#1286). `None` becomes `OperatorApplication::Exact`
  (the default) and `Some(options)` becomes
  `OperatorApplication::Measured(options)`; the new `Certified` variant is the
  third choice, so the three cannot be combined.
- `tensor4all_core::MatrixLuciScalar` is sealed and
  `matrix_luci_factors_from_blocks_seeded` is a required method (#1300). The
  trait dispatches to crate-internal kernels, so a foreign implementation could
  not produce MatrixLUCI factors; the supported scalar types stay `f32`, `f64`,
  `Complex32`, and `Complex64`. Generic `T: MatrixLuciScalar` code is
  unaffected.
- `tensor4all-partitionedtreetn`: `MaxReferenceSource::MaxOfRootCandidates`
  is replaced by `MaxReferenceSource::SearchedRootMaximum` (#1274).
- `tensor4all-core` adds the required method `TensorVectorSpace::is_complex`
  (#1298), the complex counterpart of `is_single_precision`. An implementor
  outside this workspace must add it; a container reports `true` when any part
  is complex.
- `tensor4all-simplett` removes the `EinsumScalar` marker (#1283). Use
  `TTScalar` in generic tensor-train code. The supported scalar types remain
  `f32`, `f64`, `Complex32`, and `Complex64`.
- `tensor4all-interpolativeqtt` constructors, `estimate_interpolation_error[_nd]`
  and `invert_qtt` are generic over the function's value type
  `T: InterpolativeScalar` (`f64` or `Complex64`) and return
  `SimpleTensorTrain<T>` / `Matrix<T>` (#1276). Real results are bit-for-bit
  unchanged and ordinary calls compile as before; only a call that names the
  closure type with a turbofish (`::<F>`) has to add `T` (`::<f64, F>`).

### Diagnostics

- `schedule_merge_refine` and `patched_fourier_transform` accept differing
  spectator constraints (#1299). Disjoint groups are retained directly;
  overlapping constraints use a common spectator grid checked against work and
  term limits before tensor operations. Each original patch is transformed
  once. The obsolete `PartitionedTreeTNError::SpectatorMismatch` is removed.
- Exact zip-up identity skips promote real spectator tensors when the identity
  operand is complex, matching ordinary contraction's scalar promotion.

### Performance

- `tensor4all-partitionedtreetn` finds overlapping patch projectors through a
  trie instead of pairwise loops (#1293). This covers partition validation,
  `append`, `add`, `Projector::are_disjoint`, and the overlap check and pair
  enumeration of `contract`. Results and pair order are unchanged. Two
  operands whose patches split on different indices, contracted into 16,384
  output patches, take 36 s instead of 106 s in `contract` (release, one
  thread); at about 100 patches nothing changes measurably.
- `tensor4all-partitionedtreetn` validates merge-refine input leaves through
  that same projector trie instead of comparing every pair of selected prefixes
  (#1325). The check, its `InvalidOptions` message, and the error order are
  unchanged, and the cost drops from `O(N^2 d)` to `O(N d)` for `N` leaves of
  `d` selected bits, which is linear in the geometry the caller already supplies.
  16,384 full-depth dyadic leaves validate in under 120 ms instead of about
  720 ms (release profile, one core; two pinned runs), and each doubling costs
  2.3x-2.7x instead of 4.0x.

## v0.3.0

### Patched Fourier transforms (#1275)

- `tensor4all-quanticstransform` adds `AxisBit` and
  `quantics_fourier_operator_multiaxis` for explicit axis/bit layouts. Layouts
  whose axes are monotone along the chain (grouped, interleaved) are built
  exactly without site swaps; their bond is the product of the per-axis bonds.
  Its optional `partitioned` feature adds `patched_fourier_transform`,
  `PatchedFourierOptions`, and the reconstruction option/result re-exports.
  Input patches stay lazy output-region terms; inverse calls reverse each
  axis's index order. Reconstruction error excludes operator construction
  error and backend roundoff.
- `schedule_merge_refine` and `ReconstructionTarget::from_subset_operator`
  accept an independent `OperatorV` generic parameter. Explicit turbofish
  call sites need an additional inferred `_` (for example,
  `schedule_merge_refine::<V, _>`).
- `schedule_merge_refine` and `ReconstructionTarget::from_subset_operator`
  apply the operator exactly by zip-up with a zero SVD threshold, centred at
  the node that balances the site bits, instead of the naive local product
  (#1279). The naive product's bonds were input bond × operator bond, 7200 for
  two interleaved axes at R = 10, which ran out of memory; the zip-up keeps
  the images' bonds at their physical bound, and that case now completes with
  a 4.0 GiB peak. Results are unchanged up to roundoff. On a tree that is not a
  chain, a patch whose zip-up is estimated to form a block above `2^28`
  elements is refused with `ResourceLimit` before any application. Two
  interleaved axes at R = 20 remain out of reach.
- An exact zip-up request in `tensor4all-treetn` (`CanonicalForm::Unitary`,
  `svd_policy = Some(SvdTruncationPolicy::new(0.0))`, no `max_bond_dim`) now
  factorizes every block by untruncated QR instead of SVD (#1285). Bonds and
  results are unchanged up to roundoff; no singular values are computed. The
  exact subset-operator application uses this path: two interleaved axes at
  R = 10 take 364 s instead of 837 s with a 2.4 GiB instead of 4.0 GiB peak
  (release, one thread).
- `QuanticsTransformError` is now `#[non_exhaustive]`; exhaustive matches must
  add a wildcard arm. The `partitioned` feature adds the typed `Reconstruction`
  variant. Later feature unification or new variants do not require another
  exhaustive-match migration.

### Breaking Rust API changes

Fixed-rank randomized factorization (#1108) and the partitioned TreeTN error
enum are breaking pre-1.0 Rust API changes designated for the next minor
release. The following changes break exhaustive matches, complete struct
literals, or named public types:

- `tensor4all-treetn`: `TreeTNOperationError` is an enum, and its public
  `source` field is gone. It mirrors `TreeTciError` from
  `tensor4all-treetci`:
  `InvalidConfiguration`, `InvalidGraph`, `IndexOutOfBounds` and
  `ScalarKindMismatch` are typed, and `Operation { source: anyhow::Error }` is
  the catch-all. A caller matches the variant instead of downcasting a public
  field, and reads the cause through `std::error::Error::source()`. The
  `Display` of every variant renders one level (`{source}`, as `Operation` did
  on `main`), so repeated propagation does not grow the message; the rest of
  the chain is read through `source()`. When
  `From<anyhow::Error>` recovers a classified variant that is the `anyhow` error
  object, the context layers around it are re-rendered onto the variant as text, so a typed
  context cannot be downcast after that conversion; the `Operation` catch-all
  keeps the typed chain. Only an error that *is* a classified variant is recovered:
  a variant attached with `.context(..)` is a note about another failure, so it is
  not taken as the failure. The first classified error in the source chain keeps
  its variant whether a typed context or a foreign wrapper sits above it, since
  `anyhow` cannot tell the two apart; such a variant holds the received chain
  unchanged, except `ScalarKindMismatch`, which has no chain and keeps only its
  own payload and context (the links above it are dropped, not re-rendered, so
  a wrapper that renders the mismatch does not repeat it). Failures that
  reach the boundary through an internal `anyhow` helper - tensor, storage and
  backend failures, the option checks the factorization runs in
  `tensor4all-core`, and every diagnostic an internal helper raises - keep the
  `Operation` variant with their typed cause in the chain.
- `tensor4all-treetci`, `tensor4all-core`: the batch callbacks of
  `crossinterpolate2`, `crossinterpolate2_in`, `crossinterpolate2_with_rng`,
  `crossinterpolate2_with_rng_in` and `MultiIndexCache::evaluate_batched` no
  longer require `anyhow::Result`. Each takes `E: Into<anyhow::Error>`, so a
  callback returns its own error type and the caller recovers it from the source
  chain without boxing it into `anyhow` itself. Two source changes follow. A
  callback that only ever returns `Ok` now needs its error type written
  (`Ok::<_, anyhow::Error>(..)`, `anyhow::Ok(..)`, or a
  `-> anyhow::Result<Vec<T>>` closure signature), because nothing else
  constrains `E`; a callback that already returns an explicitly typed
  `anyhow::Result` compiles unchanged. The `crossinterpolate2*` entries gain one
  generic parameter, so an explicit turbofish adds one `_`:
  `crossinterpolate2::<f64, _, _, _>` becomes
  `crossinterpolate2::<f64, _, _, _, _>`.

- `tensor4all-partitionedtreetn`: `PartitionedTreeTNError` gained the
  `RefinementTooLarge { atoms, limit }` variant (#1143) and is now
  `#[non_exhaustive]`. Both land in this release, so an exhaustive match breaks
  once: add a wildcard arm, and later variants stay additive.

- `tensor4all-core`: `FactorizeAlg::RSVD` and
  `FactorizeError::UnsupportedTruncationPolicy` are new enum variants.
  `FactorizeOptions` adds `randomized` and `rsvd_seed`. Update exhaustive
  matches and use the options constructors or `..Default::default()` in
  struct literals.
- `tensor4all-treetn`: `ContractionOptions` and `FitOptions` add `randomized`
  and `rsvd_seed`. `FitUpdater` gains a lifetime parameter for the borrowed
  execution strategy and RNG. Update named types and complete struct literals.
- `tensor4all-simplett`: remove the unimplemented `FactorizeMethod::RSVD`
  variant and its unused `rsvd_p`/`rsvd_q` options. New randomized consumers
  use the core and TreeTN APIs.
- `tensor4all-capi`: remove the unused Rust `From<FactorizeAlg>` conversion
  to the C enum. The C ABI and C enum values are unchanged.
- `tensor4all-tensorbackend`: `ScalarValue` (the dynamic scalar value) and
  `ScalarTensorElement` (its construction trait) are now public here and are the
  only definitions; `tensor4all-core` dropped its private copies and uses these.
  `BackendScalar` and `AnyScalar` remain the two wrappers. The new public surface
  is additive; the removed core items were `pub(crate)`, so no external caller
  loses a path.
- `tensor4all-treeaci`: `TreeAciResult` adds `max_sample_magnitude`, the
  largest `|f|` among the samples the interpolation was built from (the global
  guard's search samples excluded). Update complete struct literals.

### Fixed

- `abs()` of a `c32` scalar is now formed in `f64` (`hypot` after widening the
  parts) instead of computing the modulus in `f32` and widening the result, so
  `BackendScalar::abs` and `AnyScalar::abs` agree bit-for-bit. Values change only
  by the last bits of the former `f32` norm.
- `abs()` of an integer scalar widens before taking the absolute value, so
  `i32::MIN` and `i64::MIN` no longer overflow (debug panic / negative result in
  release).

### Added

- Dense fixed-rank RSVD, Gaussian range finding, caller-owned RNG entries,
  and RSVD selection in TreeTN Fit and Zipup. Supported scalar types are
  `f64` and `Complex64`; `f32` and `Complex32` are unsupported.
- `MergeRefineOptions::output_regions` names the output regions a
  `schedule_merge_refine` run must compute. `None`, the default, keeps the
  previous behaviour of refining every region at `output_depth`; `Some(regions)`
  prunes every output prefix that intersects none of them, so the result, the
  report counters and `error_bound` cover the selected part of the output.
- `tensor4all-treetn`: `contraction::contract_with_outcome` and
  `contract_with_outcome_in` return a `ContractionOutcome`: a completed
  contraction with a `ContractionReport` of its edge ranks (`EdgeRank`), or the
  first edge that reached a rank threshold (`Saturated`).
- `tensor4all-treetn`: the element-wise interpolation contract
  `interpolation::ElementwiseInterpolator`, with `ElementwiseOperator`, the
  binary `Product`, and `PointwiseElementwise`, which runs any
  `TreeInterpolator` on point evaluations of the inputs.
- `tensor4all-treeaci`: `TreeAciInterpolator`, tree ACI as an element-wise
  engine. Its kernel runs at the problem's tolerance divided by a tolerance
  factor (`with_tolerance_factor`, default 3). A `RankLimited` run is
  `BondCapReached` at the cap and `IterationLimit` below it.
- `tensor4all-treersi`: `TreeRsiInterpolator`, tree RSI as an element-wise
  engine for products.
- `tensor4all-partitionedtreetn`: `adaptive_interpolation::patched_elementwise`,
  `patched_elementwise_on_lanes` and, with the MPI features,
  `patched_elementwise_mpi` compute `f(x_1, ..., x_k)` of partitions on one
  site space as a partition, under the L2 contract of `patched_interpolate`.
- `tensor4all-partitionedtreetn`: `adaptive_interpolation::merge_siblings`
  merges complete groups of sibling patches of a patched interpolation result
  without breaking its L2 error contract (`SiblingMergeOptions`,
  `SiblingMergeReport`).
- `tensor4all-partitionedtreetn`: `contract_adaptive` groups compatible pairs by
  predicted output projector before any numerical work and, with a cap and
  `PatchSplitStrategy::Sequential`, stops a group at its first saturated
  contribution instead of contracting every pair first. Layouts accepted
  before give the same partition and values. Output projectors that overlap
  without being equal, which used to return `OverlappingProjectors`, are now
  refined onto a disjoint grid; a grid above `AUTOMATIC_REFINEMENT_LIMIT`
  returns `RefinementTooLarge`. `contract_adaptive_selected` forms only the
  pairs whose output key equals a wanted key, so a pair whose key overlaps a
  wanted key without being equal to it is not formed.
- `tensor4all-partitionedtreetn`: under the L2 error norm, the adaptive
  interpolation driver now measures an engine run that ends with
  `ErrorNotMeasured` below the cap, and can accept or retry it like a
  converged run, where it used to split it. Besides tree RSI this covers
  TreeTCI runs that end with `DegenerateCandidateSet` in
  `patched_interpolate`.

### Public-API renames and removals since v0.2.0

The sections above record release impact; this one is the porting list. It names
the public Rust API and C API items that existed at `v0.2.0` and no longer exist
at this tag, grouped by crate. Each entry cites the commit that made the change
(12 hex digits) and the pull request. Additions are not listed here — read the
rustdoc for those. Nothing is published to crates.io; depend on the tag as
described in [README.md](README.md).

#### Crate-level changes

- `tensor4all-tcicore` was dissolved into `tensor4all-core` — `5396d0d25bbd`,
  #639, #642. The public modules `cached_function`, `indexset`, `matrixaca`,
  `matrixlu`, `matrixluci` (with `cache_key`, `error`, `index_int`, `scalar`,
  `traits`), the types `CachedFunction`, `IndexSet`, `MultiIndex`, `LocalIndex`,
  `MatrixACA`, `MatrixLUCI`, `MatrixLuciFactors`, `RrLU`, `RrLUOptions`, the
  traits `AbstractMatrixCI`, `CacheKey`, `IndexInt`, `Scalar`, and the error
  enums `CacheKeyError` and `MatrixCIError` moved to `tensor4all-core`; their
  inherent methods moved with them. The crate itself no longer exists. Two
  other removals landed in the crate before it was dissolved: `rrlu_inplace` —
  `7d263fadb7ef`, #595; `PivotKernelOptions::max_rank` and
  `RrLUOptions::max_rank` — `9a77ac651643`, #599.
- `tensor4all-partitionedtt` was replaced by the new `tensor4all-partitionedtreetn`
  crate — `8ee92420e528`, #915 D3, #1005. `PartitionedTT`, `SubDomainTT`,
  `Projector`, `PatchingOptions` (fields `max_bond_dim`, `patch_order`, `rtol`),
  `PartitionedTTError` with its seven variants, `Result`, and their inherent
  methods are gone. Before that removal: `data_mut`, `get_mut`, `values_mut` —
  `2b11f7fe564a`, #649; `SubDomainTT::siteinds` — `7d263fadb7ef`, #595;
  `PartitionedTTError::TensorTrainError` — `928c567b922e`, #589.

#### tensor4all-core

- `7d263fadb7ef`, #595 — `replaceinds` renamed to `replace_indices` and
  `replaceinds_in_place` to `replace_indices_mut`; removed `as_slice_f64` and
  `as_slice_c64` (use the generic `to_vec::<f64>()` / `to_vec::<Complex64>()`).
- `a93f582c202f`, #612 — `TensorDynLen` renamed to `IdxTensor`, the module
  `tensordynlen` to `idx_tensor`, and the constructor `diag_tensor_dyn_len` to
  `diag_idx_tensor`. The `indices` field is a separate change, below.
- `6e0d55800018`, #1024 — the `indices` field of `IdxTensor` (public at
  `v0.2.0`) became `pub(crate)`; read it through `IdxTensor::indices()`.
- `9a77ac651643`, #599 — `SvdOptions::max_rank` and `FactorizeOptions::max_rank`
  renamed to `max_bond_dim`; `with_max_rank` renamed to `with_max_bond_dim`.
- `2c972983e594`, #616, #643 — `index_ops::hasinds` renamed to `has_inds` and
  `hascommoninds` to `has_common_inds` (the old spellings survive only as
  `#[doc(alias)]`).
- `928c567b922e`, #589 — `try_norm_squared` removed; `norm_squared` now returns
  `Result` instead of `f64`, as do `norm` and `maxabs`.
- `53763172f65c`, #593 — `build_diag_union` removed (now internal to the
  contraction planner), and `AxisUnionFind`, `remap_tensor_ids`,
  `remap_output_ids` and `collect_sizes` in `defaults::contract` became
  `pub(crate)`.

#### tensor4all-itensorlike

- `7d263fadb7ef`, #595 — `siteinds` renamed to `site_indices`, `linkinds` to
  `link_indices`, `sim_linkinds` to `sim_link_indices`, `maxbonddim` to
  `max_bond_dim`.
- `2c972983e594`, #616, #643 — `isortho` renamed to `is_ortho`, `orthocenter` to
  `ortho_center`.
- `9a77ac651643`, #599 — `max_rank` renamed to `max_bond_dim` and
  `with_max_rank` to `with_max_bond_dim` on the contraction and truncation
  option types.
- `1e0ecea48b73`, #999 — `tensor_checked`, `tensor_mut_checked`,
  `tensors_mut_checked` and `set_tensor_checked` removed; `tensor`, `tensor_mut`
  and `tensors_mut` return `Result` on their own.
- `f04320320afb`, #650 — `site_range` and `with_site_range` removed from the
  contraction options.

#### tensor4all-quanticstci

- `2c972983e594`, #616, #643 — `QtciOptions::maxiter` renamed to `max_iter`,
  `nrandominitpivot` to `n_random_init_pivot`, `unfoldingscheme` to
  `unfolding_scheme`.
- `7d263fadb7ef`, #595 — `QtciOptions::maxbonddim` renamed to `max_bond_dim`,
  `with_maxbonddim` to `with_max_bond_dim`.
- `97b0df907f6b`, #1000 — `QtciOptions::normalize_error` renamed to
  `scale_tolerance`.
- `f04320320afb`, #650 — `QtciOptions::nsearch`, `nsearchglobalpivot`,
  `with_nsearch` and `with_nsearchglobalpivot` removed; the fields were never
  wired to the quantics TCI run.
- `b881f39d9e72`, #828 — `cachedata` and `cachedata_origcoord` removed. The
  run no longer exports the points it evaluated; `cache_stats()` / `CacheStats`
  report counters only.

#### tensor4all-quanticstransform

- `928c567b922e`, #589 — the public fields of `AffineParams` (`a`, `b`, `m`,
  `n`) and `UnfusedTensorInfo` (`m`, `n`, `num_physical_dims`, `physical_dim`)
  became accessor methods of the same names.
- `7d263fadb7ef`, #595 — `FourierOptions::maxbonddim` renamed to `max_bond_dim`.

#### tensor4all-simplett

- `7d263fadb7ef`, #595 — `TensorTrain` renamed to `SimpleTensorTrain`; `dot`
  renamed to `inner_product`, `fulltensor` to `full_tensor`. Scaling was split:
  the old mutating `scale(&mut self, factor)` is now `scale_mut`, and the old
  copy-producing `scaled(&self, factor)` is now `scale(&self, factor) -> Self`,
  so an uncorrected `tt.scale(f)` call silently drops the scaled copy.
- `6c71bccb09a9`, #681 — `TensorTrainError` renamed to `SimpleTensorTrainError`
  with unchanged variants, so it no longer collides with
  `tensor4all_itensorlike::TensorTrainError`.
- `9a77ac651643`, #599 — `FactorizeOptions::max_rank` renamed to `max_bond_dim`.
- `ad56049ab28f`, #1003 — the retired MPO surface is deleted: the modules
  `contract_fit`, `inverse_mpo`, `site_mpo`, `vidal_mpo`, the types
  `Contraction`, `FitOptions`, `InverseMPO`, `SiteMPO`, `VidalMPO`, the
  `ContractionAlgorithm::Fit` variant, and their methods (`contract_fit`,
  `from_mpo`, `into_mpo`, `from_parts_unchecked`, `with_transform`,
  `result_site_dims`, `gamma`, `gamma_mut`, `gammas`, `lambda`, `lambdas`,
  `inv_lambda`, `inv_lambdas`, plus the `FitOptions` fields `tolerance`,
  `convergence_tol`, `max_sweeps`, `max_bond_dim`, `factorize_method`).
- `23285e70e229`, #1116 — `FactorizeOptions::rsvd_p` and `rsvd_q` removed and
  `FactorizeMethod::RSVD` dropped (it was never implemented); randomized SVD is
  now the explicit `RsvdOptions` / `randomized_svd` surface.
- `39b1884b0513`, #631 — `Tensor::as_inner_mut` removed.

#### tensor4all-tensorbackend

- `7d263fadb7ef`, #595 — `mindim` renamed to `min_dim`;
  `native_tensor_primal_to_dense_f64_col_major` and
  `native_tensor_primal_to_dense_c64_col_major` replaced by the generic
  `native_tensor_primal_to_dense_col_major<T>`;
  `native_tensor_primal_to_diag_f64` and `native_tensor_primal_to_diag_c64`
  replaced by `native_tensor_primal_to_diag<T>`.
- `2c972983e594`, #616, #643 — `Storage::from_dense_f64_col_major` and
  `Storage::from_dense_c64_col_major` replaced by
  `Storage::from_dense_col_major<T>`.
- `8ee92420e528`, #915 D3, #1005 — `Storage::from_diag_f64_col_major` and
  `Storage::from_diag_c64_col_major` replaced by `Storage::from_diag_col_major<T>`;
  the `StorageError::RealScalarRequired` variant is gone.
- `39b1884b0513`, #631 — the public result fields became accessors:
  `SvdResult::{s, u, vt}` and `FullPivLuResult::{p, l, u, q}` are now methods of
  the same names, with `into_parts` for the owning form.
- `6c71bccb09a9`, #681 — the struct `Scalar` renamed to `BackendScalar`; the
  `AnyScalar` type alias is removed.

#### tensor4all-treetci

- `97b0df907f6b`, #1000 — `TreeTciOptions::normalize_error` renamed to
  `scale_tolerance`.
- `6e0d55800018`, #1024 — `TreeTCI2` no longer exposes its state as fields:
  `graph`, `local_dims`, `bond_errors`, `pivot_errors` and `max_sample_value`
  are accessor methods (`max_sample_value` also has `set_max_sample_value`), and
  the `ijset` field is replaced by `pivot_set(key)`.
- `70d4148f333b`, #842 — `TreeTCI2::ijset_history` removed.
- `9a77ac651643`, #599 — `TreeTCI2::max_rank` renamed to `max_bond_dim`.
- `b1bb828da13d`, #837 — the `TreeTciRunResult` type alias became a struct of
  the same name carrying `treetn`, `ranks`, `errors` and `termination`.
- `3bc704e399a5`, #808 — `OwnedGlobalIndexBatch` (with `new`, `as_view`,
  `into_vec`), `assemble_global_point` and `assemble_points_column_major`
  removed from the batch and assembly surface.
- `53763172f65c`, #593 — the `FullPivLuScalar` trait removed.

#### tensor4all-treetn

- `TreeTNOperationError` is an enum (`InvalidConfiguration`, `InvalidGraph`,
  `IndexOutOfBounds`, `ScalarKindMismatch`, `Operation`) instead of a struct
  with a public `source: anyhow::Error` field, and `std::error::Error::source()`
  replaces that field — #1023. `crossinterpolate2*` and
  `MultiIndexCache::evaluate_batched` live in the crates above, but their
  relaxed callback bound is the same change.
- `7d263fadb7ef`, #595 — the public modules `algorithm`, `dmrg`, `error`,
  `link_index_network`, `linsolve`, `named_graph`, `node_name_network`,
  `operator`, `options`, `random`, `site_index_network`, `tdvp` and `treetn`
  became private; reach their items through the crate root, except
  `random::DefaultIndex`, which is not re-exported: use
  `tensor4all_core::DynIndex`, the same `Index<DynId, TagSet>`.
  `DefaultSiteIndexNetwork` removed. The `site_index_network` re-export alias
  `CanonicalizeEdgesType` is gone with the module; the type is
  `tensor4all_treetn::CanonicalizeEdges` (`lib.rs:73`).
  `evaluate_batch` renamed to
  `evaluate_batched`, `sim_linkinds` and `sim_linkinds_mut` to
  `sim_link_indices` and `sim_link_indices_mut`, and the `evaluate_at` alias
  removed (use `evaluate`).
- `9a77ac651643`, #599 — `max_rank` renamed to `max_bond_dim` on `ApplyOptions`,
  `ContractionOptions`, `FitContractionOptions`, `SwapOptions`,
  `TruncateUpdater` and on the option accessor.
- `97b0df907f6b`, #1000 — `SwapOptions::rtol` replaced by the explicit
  `svd_policy: Option<SvdTruncationPolicy>`; `with_svd_policy` is the setter.
- `89d78b9c26cf`, #992 — `fix_and_remove_site_from_treetn_chain` renamed to
  `fix_and_remove_site`.
- `efd88be287fc`, #1002 — `TreeTN::site_index_network_mut` removed.
- `7d263fadb7ef`, #595 — `TreeTN::scale(&mut self)` became the fallible
  mutating `scale_mut`, and `scale(&self)` now returns a scaled copy. An
  uncorrected `tn.scale(x)?;` call discards the copy; keep the old behaviour by
  renaming the call to `scale_mut`.

#### tensor4all-aci, tensor4all-interpolativeqtt, tensor4all-tensorci

- `7d263fadb7ef`, #595 — `AciError::TensorTrain` renamed to
  `AciError::SimpleTensorTrain`; `InterpolativeQttError::TensorTrain` renamed to
  `InterpolativeQttError::SimpleTensorTrain`.
- `6c71bccb09a9`, #681 — `TCIError::TensorTrainError` renamed to
  `TCIError::SimpleTensorTrain`.

#### tensor4all-capi

- `7d263fadb7ef`, #595 — `t4a_treetn_siteinds` renamed to
  `t4a_treetn_site_indices`. Tensor4all.jl and other C callers must rebuild
  against the new header.

## Verifying an entry

The lists above were produced from the public-item surface at `v0.2.0` and at
the release commit. To reproduce a claim, look the item up at the old tag and
find the commit that changed it:

```bash
git grep -n 'fn linkinds' v0.2.0 -- crates/tensor4all-itensorlike
git log -S'fn linkinds' --oneline v0.2.0..v0.3.0 -- crates/tensor4all-itensorlike
```
