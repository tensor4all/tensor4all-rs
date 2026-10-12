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
        ├── tensor4all-treetn       (session-taking SRC contraction, evaluator)
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
`tensorbackend`), who *threads* it (core, simplett, treetn, treetci, tensorci, partitionedtt) and who has never
been touched.

## 3. Session/explicit public surface per crate

| crate | session/context-taking public entries (this checkout) |
| --- | --- |
| `tensor4all-core` | `contract_pair_on_session`, `inner_product_on_session`, `sum_on_session`, `permute_on_session`, `permute_indices_on_session`, `norm_squared_on_session`, `norm_on_session`, `matrix_luci_factors_from_matrix_owned_in`, `matrix_luci_factors_from_matrix_owned_on_session`, `matrix_luci_row_interpolation_owned_in`, `factorize_in`, `factorize_full_rank_in`, `from_dense_in`, `ones_in`, `norm_in` |
| `tensor4all-simplett` | `SimpleTensorTrain::inner_product_with_session` |
| `tensor4all-treetn` | `contract_src_with_rng_in`, `new_in` |
| `tensor4all-treetci` | `crossinterpolate2_in`, `crossinterpolate2_with_rng_in` (both cover the factor assembly **and** the materialization solve) |
| `tensor4all-tensorci` | `crossinterpolate2_with_rng_on_session` (dense factor assembly and site solves) |
| `tensor4all-partitionedtt` | `adaptiveinterpolate_in` (Hataori domain, compatibility route), `adaptiveinterpolate_on_phase` (held-session lanes) |
| `tensor4all-treersi` | `hadamard_many_in`, `hadamard_many_with_rng_in` |
| everything else | none yet (see §5) |

Reproduce: `grep -rhoE "pub fn [a-z_0-9]*(_in|_on_session|_on_phase|_with_context|_in_context)\b" crates/*/src`.

Indicative file counts of the two route families, to show where work remains rather than to score crates:
`grep -rl "Session<'_>\|CpuExecutionContext\|HeldSession\|PhaseLane" crates/<crate>/src` versus
`grep -rl "default_eager_ctx\|with_default_session\|default_cpu_execution_context\|with_default_backend"
crates/<crate>/src`. In this checkout the first is non-zero for `core` (12), `treetci` (7), `treersi` (3),
`simplett` (2), `treetn` (2), `tensorci` (1), `partitionedtt` (1), and zero for `aci`, `treeaci`,
`itensorlike`, `quanticstci`, `quanticstransform`, `interpolativeqtt`, `partitionedtreetn`, `capi`. The second is
non-zero only for `core` (3) and `treetn` (4) and `treetci` (2) and `treersi` (1): the other crates reach the
global route transitively, which is why a per-crate count cannot decide whether a crate is migrated.

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
| `tensor4all-aci` | thread a session through its tree-ACI driver to the `treetci`/`treetn` entries | nothing but the work |
| `tensor4all-treeaci` | as above, plus its own adaptive driver entries | nothing but the work |
| `tensor4all-itensorlike` | metadata/structure only; needs an inventory of which operations materialize | nothing but the work |
| `tensor4all-quanticstci` | thread to `treetci`'s context entries | nothing but the work |
| `tensor4all-quanticstransform` | mostly operator construction (metadata + host assembly) | nothing but the work |
| `tensor4all-interpolativeqtt` | thread to `simplett`/`treetn` session entries | nothing but the work |
| `tensor4all-partitionedtreetn` | thread to `treetci`/`treetn` | nothing but the work |
| `tensor4all-capi` | new ABI surface would be required (F6), not a threaded parameter | F6's ABI decision |
| `tensor4all-py` | as `capi` | F6 |
| `tensor4all-hdf5` | I/O only; no numerical route | — |

## 6. Keeping this honest

Re-run the three commands in §1-§3 after any landing that adds a session-taking entry, changes a default
feature, or moves a compatibility path; the tables are only as good as the checkout they were taken from. When a
row changes, say what changed and cite the PR, so that "migrated" keeps meaning "measured", not "planned".
