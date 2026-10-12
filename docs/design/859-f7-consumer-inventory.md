# Consumer and feature inventory for the #859 migration (F7)

`#859`'s F7 row asks to "inventory the current crate/feature map, not only this historical list" before migrating
the remaining consumers. This is that inventory, taken from the checkout at `6b006188` (the merge of #926). Every
table below names the command that produced it, so the next reader can re-derive it instead of trusting it.

## 1. Workspace crates, roles and default features

| crate | role | default features |
| --- | --- | --- |
| `tensor4all-tensorbackend` | the backend boundary: contexts, sessions, tenferro bridge, storage | `backend-tenferro`, `tenferro-cpu-faer` |
| `tensor4all-core` | index/tensor values, MatrixLUCI, matrix helpers | `backend-tenferro`, `tenferro-cpu-faer` |
| `tensor4all-simplett` | simple TT (MPS-shaped) values | `backend-tenferro`, `tenferro-cpu-faer` |
| `tensor4all-treetn` | general tree tensor networks | `tenferro-cpu-faer`, `simplett-bridge` |
| `tensor4all-tensorci` | TCI2 (two-site) cross interpolation | `tenferro-cpu-faer` |
| `tensor4all-treetci` | tree-TCI | `tenferro-cpu-faer` |
| `tensor4all-treersi` | SRC / rank-revealing sketches | `tenferro-cpu-faer`, `global-defaults` |
| `tensor4all-treeaci` | adaptive cross interpolation on trees | `tenferro-cpu-faer` |
| `tensor4all-aci` | adaptive CI driver over tree structures | `tenferro-cpu-faer` |
| `tensor4all-partitionedtt` | adaptive interpolation over projected patches | `tenferro-cpu-faer` |
| `tensor4all-partitionedtreetn` | partitioned tree-TN (QTT/QTCI drivers) | `tenferro-cpu-faer` |
| `tensor4all-quanticstci` | quantics TCI | `tenferro-cpu-faer` |
| `tensor4all-quanticstransform` | quantics operators/transforms | `tenferro-cpu-faer` |
| `tensor4all-interpolativeqtt` | interpolative QTT | `tenferro-cpu-faer` |
| `tensor4all-itensorlike` | TensorTrain/MPS-shaped values | `tenferro-cpu-faer` |
| `tensor4all-hdf5` | HDF5 input/output | `link`, `tenferro-cpu-faer` |
| `tensor4all-capi` | C ABI | `tenferro-cpu-faer` |
| `tensor4all-py` | Python bindings | — |

Reproduce: parse each `crates/*/Cargo.toml` and print `[features].default`.

Two feature families matter for this program: the **provider** features (`tenferro-cpu-faer`,
`tenferro-system-blas`, `tenferro-provider-inject`, `tenferro-cuda`) select the numerical backend, and the
**route** features (`backend-tenferro` = the compatibility process-global surface, `explicit-context` = the
caller-supplied one, `global-defaults` = legacy operations on one global context) select which entry points
exist. `tensor4all-treetci` and `tensor4all-treetn` also carry application features (`adaptive-hataori-rayon`
lives in `partitionedtt`, and is deliberately not default).

## 2. Dependency layers (who inherits whose route)

```
tensor4all-tensorbackend            (chooses the route: contexts, sessions, compatibility entries)
  └── tensor4all-core               (session-taking twins + compatibility entries)
        ├── tensor4all-simplett     (session-taking inner product)
        ├── tensor4all-treetn       (context-taking contraction dispatch: SRC / zip-up / fit / naive)
        │     ├── tensor4all-treetci      (context-taking TCI entries)
        │     ├── tensor4all-treeaci
        │     ├── tensor4all-treersi
        │     └── tensor4all-partitionedtreetn
        └── tensor4all-tensorci     (session-taking TCI2 entry)
              ├── tensor4all-partitionedtt   (phase entry: lanes on a held session)
              └── tensor4all-quanticstci
tensor4all-capi / tensor4all-py / tensor4all-hdf5   (bindings and I/O on top of the above)
```

Reproduce: `grep -hoE 'tensor4all-[a-z0-9]+ *= *\{ *path' crates/*/Cargo.toml`.

A crate that appears with no session-taking entry of its own is **not** necessarily on the global route: it
inherits whatever its dependencies do. The distinction that matters is who *chooses* the route (only
`tensorbackend`), who *threads* it (core, simplett, treetn, treetci, tensorci, partitionedtt, and - since the
F7 consumer pass - aci, treeaci, itensorlike, quanticstci and partitionedtreetn, whose engine carries the
context) and who has never been touched.

## 3. Session/explicit public surface per crate

| crate | session/context-taking public entries (this checkout) |
| --- | --- |
| `tensor4all-core` | `contract_pair_on_session`, `inner_product_on_session`, `sum_on_session`, `permute_on_session`, `permute_indices_on_session`, `norm_squared_on_session`, `norm_on_session`, `matrix_luci_factors_from_matrix_owned_in`, `matrix_luci_factors_from_matrix_owned_on_session`, `matrix_luci_row_interpolation_owned_in`, `factorize_in`, `factorize_full_rank_in`, `from_dense_in`, `ones_in`, `norm_in`, `svd_with_in`, `qr_with_in`, `scale_in` |
| `tensor4all-simplett` | `SimpleTensorTrain::inner_product_with_session` |
| `tensor4all-treetn` | `contract_in` (dispatch over SRC / zip-up / fit / naive), `contract_src_with_rng_in`, `contract_zipup_with_in`, `contract_fit_in`, `new_in` |
| `tensor4all-treetci` | `crossinterpolate2_in`, `crossinterpolate2_with_rng_in` (both cover the factor assembly **and** the materialization solve), `optimize_with_proposer_in`, `to_treetn_in` |
| `tensor4all-tensorci` | `crossinterpolate2_with_rng_on_session` (dense factor assembly and site solves) |
| `tensor4all-partitionedtt` | `adaptiveinterpolate_in` (Hataori domain, compatibility route), `adaptiveinterpolate_on_phase` (held-session lanes) |
| `tensor4all-quanticstci` | `quanticscrossinterpolate_discrete_batch_in`, `quanticscrossinterpolate_batch_in` (thread the caller's context to `treetci`'s entries) |
| `tensor4all-aci` | `elementwise_batched_with_rng_in` (LUCI and frame products in the caller's context) |
| `tensor4all-treeaci` | `tree_elementwise_batched_with_rng_in` (LUCI and candidate-matrix products in the caller's context) |
| `tensor4all-itensorlike` | `contract_in` (SRC / Zip-up / Fit / Naive through treetn's twins) |
| `tensor4all-treersi` | `hadamard_many_in`, `hadamard_many_with_rng_in` |
| everything else | none yet (see §5) |

Reproduce: `grep -rhoE "pub fn [a-z_0-9]*(_in|_on_session|_on_phase|_with_context|_in_context)\b" crates/*/src`.

Indicative file counts of the two route families, to show where work remains rather than to score crates:
`grep -rl "Session<'_>\|CpuExecutionContext\|HeldSession\|PhaseLane" crates/<crate>/src` versus
`grep -rl "default_eager_ctx\|with_default_session\|default_cpu_execution_context\|with_default_backend"
crates/<crate>/src`. In this checkout (after the F7 consumer pass, #940-#947) the first is non-zero for `core`
(12), `treetci` (8), `treeaci` (5), `treersi` (3), `simplett` (2), `treetn` (2), `aci` (2), `tensorci` (1),
`partitionedtt` (1), `quanticstci` (1), and zero for `itensorlike`, `quanticstransform`, `interpolativeqtt`,
`partitionedtreetn`, `capi`. The second is non-zero only for `core` (3) and `treetn` (4) and `treetci` (2) and
`treersi` (1): the other crates reach the global route transitively, which is why a per-crate count cannot
decide whether a crate is migrated.

## 4. Compatibility surface intentionally retained (declared, not advertised as migrated)

These are the places where the program has deliberately kept the compatibility route, each with its recorded
reason; the F7 migration must not silently claim them:

- **The lazy-block / rook pivot path** in the TCI2 assembly and in core's `matrixluci` block joins still solves
  and multiplies on the compatibility route; the dense path is the one that is session-taking
  (`docs/design/859-f4-materialization-context.md` §2.2, `859-f4-interpolation-child-executions.md` §2.2).
- **Publication** adopts host values into the process-global eager runtime; the phase entry documents that its
  returned tensors are usable after the session closes but are not bound to the issuing execution
  (`859-f4-interpolation-child-executions.md` §2.3).
- **The Hataori worker entry** (`adaptiveinterpolate_in`) keeps the compatibility route and its Rayon-worker
  fallback; the supported worker path is `adaptiveinterpolate_on_phase`, and the feature stays opt-in
  (same document, §2.5).
- **The CUDA SRC estimator** reads the resident factor back to the host once, because the pinned eager API has no
  general solve (PR #926).
- **The global defaults themselves** (`backend-tenferro`, `global-defaults`) are intact by design; #859's R stage
  retires them only after this inventory migrates and maintainers coordinate.

## 5. What the untouched crates need

| crate | first step | blocking on |
| --- | --- | --- |
| `tensor4all-aci` | wired (#935, #941): `elementwise_batched_with_rng_in` threads a caller-owned context to its LUCI and frame products | done; the batched frame updates stay on their existing route (no session-taking batched-GEMM twin) |
| `tensor4all-treeaci` | wired (this PR): `tree_elementwise_batched_with_rng_in` threads a caller-owned context to its LUCI and candidate-matrix products | done; its own frame contractions stay on their existing route for now |
| `tensor4all-itensorlike` | done (#943/#945/#946/#947): `contract_in` routes all four methods (SRC, Zip-up, Fit, Naive) through treetn's context twins | — |
| `tensor4all-quanticstci` | done (#940): both batch entries thread the caller's context to `treetci`'s entries | — |
| `tensor4all-quanticstransform` | mostly operator construction (metadata + host assembly) | nothing but the work |
| `tensor4all-interpolativeqtt` | thread to `simplett`/`treetn` session entries | nothing but the work |
| `tensor4all-partitionedtreetn` | done (#944): the engine layer carries the context - `TreeTciInterpolatorIn` in `tensor4all-treetci` runs `patched_interpolate` (and any `TreeInterpolator` driver) in a caller-owned context; the patched driver and its trait need no change | — |
| `tensor4all-capi` | new ABI surface would be required (F6), not a threaded parameter | F6's ABI decision |
| `tensor4all-py` | as `capi` | F6 |
| `tensor4all-hdf5` | I/O only; no numerical route | — |

## 6. Per-crate verdicts, first pass

A first read of the untouched crates, so the migration order rests on what each crate actually does rather than
on its position in the graph. Every verdict names the call site that decides it.

| crate | verdict | evidence |
| --- | --- | --- |
| `tensor4all-interpolativeqtt` | **host-only; no session surface needed at this layer** | its API is Lagrange-basis evaluation and `interpolation_tensor`, which builds a host `Tensor3<f64>`; it contains no contraction, solve or TCI call at all, so whatever consumes its values decides the route. |
| `tensor4all-quanticstransform` | **host operator construction; the numerics belong to whoever applies it** | `QuanticsOperator = LinearOperator<IdxTensor, usize>` and the operators are built from host `IdxTensor`s; applying one goes through `tensor4all-treetn`'s entries, which already have session-taking twins. A session surface is needed at the *apply* call site, not in the transform. |
| `tensor4all-itensorlike` | **thread-through; all four methods wired** | `contract` delegates to `tensor4all-treetn`'s `contract` (`contract.rs:112`); the SRC method now has the pass-through (`contract_in` -> treetn's `contract_src_with_rng_in`), Zip-up now routes through treetn's `contract_zipup_with_in` (context-scoped `contract_pair_on_session` + `factorize_in`), Fit routes through treetn's `contract_fit_in` (context-scoped initial zip-up, environment contractions and per-sweep factorizations) and Naive through `contract_naive_to_treetn_with_leaves` (context-scoped dense product, pairwise product and decomposition factorizations), so every method has a caller-owned route; the dense-reference limit contract is unchanged. SRC requires inputs materialized in the supplied context (`IdxTensor::from_dense_in`), matching its provenance validation; Zip-up validates inputs with `TreeTN::validate_context` (host-only CPU tensors validate against any CPU context; resident or foreign-eager inputs are refused). SRC parity is exact on a seeded run; Zip-up parity is tolerance-based (the session route factorizes directly because the `factorize_auto` Gram shortcut has no context twin). The release-mode, one-thread bench measures SRC plain 0.912 ms vs context 1.103 ms, Zip-up plain 0.103 ms vs context 0.539 ms, Fit plain 4.024 ms vs context 11.495 ms and Naive plain 0.168 ms vs context 0.616 ms (6 sites, chi 2; the context figures include the per-leaf adoption round-trip that keeps operands in one eager AD context). |
| `tensor4all-quanticstci` | **thread-through; both entry families wired** | it drives the TCI itself rather than calling `crossinterpolate2`, so its pass-through needed the context-taking optimizer (`optimize_with_proposer_in`, #937) and materialization (`to_treetn_in`, #939) entries. Both `quanticscrossinterpolate_discrete_batch_in` and `quanticscrossinterpolate_batch_in` forward to them, so a run's session-bound work takes the caller's context; the caller's function and the readout keep their routes. |
| `tensor4all-partitionedtreetn` | **engine-provider; the engine carries the context** | `patched_interpolate` holds no numerics of its own: the engine trait (`TreeInterpolator`) is the numeric provider, so the caller-owned route belongs to the engine. `TreeTciInterpolatorIn` (treetci, this PR) implements the same trait with both the optimizer (`optimize_with_proposer_in`, #937) and the final materialization (`to_treetn_chunked` with a context, #939 wiring) on the supplied context; the patched driver and trait are untouched. Parity is dense-equal to 1e-12 on a seeded run (`tests/adaptive_interpolation_engines.rs`); the release-mode, one-thread bench (`benches/f7_engine_context_overhead.rs`) measures plain 0.697 ms vs context 0.705 ms (2-node tree, 4x4, parity within noise). |
| `tensor4all-aci` | **has numerics of its own; LUCI and frame products wired** | it calls core's LUCI directly (`local_update`, `initialize_right_frames`) and multiplies the input frames itself (`frame_matmul_checked`, `matmul_checked_owned`). `elementwise_batched_with_rng_in` now threads a caller-owned context to both kinds, with parity and a measurement. The **batched** frame updates (`batched_mat_mul_same_shape_owned`) stay on their existing route because the backend has no session-taking batched-GEMM twin, and the caller's own operator is untouched. |
| `tensor4all-treeaci` | **has numerics of its own; LUCI and candidate-matrix products wired** | its local update factorizes through core's LUCI and multiplies the candidate matrices itself (`materialize_and_factor_edge`: core's `matrix_luci_factors_from_matrix_owned` at `local_update.rs:400` and `mat_mul`/`mat_mul_owned` at `:347`-`:352`). `tree_elementwise_batched_with_rng_in` now threads a caller-owned context to both kinds, with parity (bit-for-bit on a seeded run) and a release-mode, one-thread benchmark (`benches/f7_context_overhead.rs`: plain 1.390 ms vs context 1.433 ms per run, n_sites=4, chi=3, both routes pinned via `RAYON_NUM_THREADS`/`BLAS_NUM_THREADS`/`OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS=1` and `CpuBackend::with_threads(1)`; ~3% spread, parity within noise). The crate's own **frame contractions** (`frames.rs:3244`, `:3260`) stay on their existing route for now, and the caller's own operator is untouched. |
| `tensor4all-hdf5` | **no route to choose** | input/output only. |
| `tensor4all-capi`, `tensor4all-py` | **F6's business** | a new ABI surface, not a threaded parameter, so they are not part of this pass. |

The verdict pass is now complete: every workspace crate that carries numerical work has a named call site deciding
whether it needs a session surface of its own (`aci`, `treeaci`), a pass-through entry (`itensorlike`, `quanticstci`), an engine-provider whose engine carries the context
(`partitionedtreetn`), or nothing at all (`interpolativeqtt`, `quanticstransform`, `hdf5`), with
`capi`/`py` left to F6.

## 7. Keeping this honest

Re-run the three commands in §1-§3 after any landing that adds a session-taking entry, changes a default
feature, or moves a compatibility path; the tables are only as good as the checkout they were taken from. When a
row changes, say what changed and cite the PR, so that "migrated" keeps meaning "measured", not "planned".

Read this file from `origin/main` or a fresh worktree, never from a long-lived checkout: during this pass a stale
main checkout made a merged entry point look absent, and only a fresh worktree showed the real state. A verdict
taken from the wrong tree is worse than no verdict.

Re-verified on a fresh `origin/main` worktree after the consumer pass (#940 quanticstci, #941 aci, #942 treeaci,
#943 itensorlike SRC, #944 partitionedtreetn, #945 treetn zip-up + itensorlike Zip-up, #946 treetn fit +
itensorlike Fit, #947 treetn naive + itensorlike Naive): §1 default features are unchanged; §2's dependency
layers are unchanged (the tree's treetn node now names the full contraction dispatch, and the threading list
gained the five consumer crates); §3's entry table and route-family counts above are from that checkout.
