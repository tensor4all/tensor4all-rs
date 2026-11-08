# Changelog

## v0.3.0

### Breaking Rust API changes

Fixed-rank randomized factorization (#1108) and the partitioned TreeTN error
enum are breaking pre-1.0 Rust API changes designated for the next minor
release. The following changes break exhaustive matches, complete struct
literals, or named public types:

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
    relative_linear_system_residual, square_linsolve, EnvironmentCache, GmresToleranceMode,
    LinsolveOptions,
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

- `7d263fadb7ef`, #595 — the public modules `algorithm`, `dmrg`, `error`,
  `link_index_network`, `linsolve`, `named_graph`, `node_name_network`,
  `operator`, `options`, `random`, `site_index_network`, `tdvp` and `treetn`
  became private; reach their items through the crate root, except
  `random::DefaultIndex`, which is not re-exported: use
  `tensor4all_core::DynIndex`, the same `Index<DynId, TagSet>`.
  `DefaultSiteIndexNetwork` removed. `evaluate_batch` renamed to
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
