# Design Documents

## Architecture & Backend

| Document | Description |
|----------|-------------|
| [t4a_unified_tensor_backend.md](./t4a_unified_tensor_backend.md) | Unified tensor backend design (tenferro-rs integration) |
| [tenferro-main-session-migration.md](./tenferro-main-session-migration.md) | First issue #623 slice: pin current tenferro main and migrate internal CPU calls to its canonical session API |
| [explicit-cpu-execution-context.md](./explicit-cpu-execution-context.md) | Issue #663 explicit plain, graph, eager-AD, and logical reconstruction context |
| [tensorbackend-session-entry.md](./tensorbackend-session-entry.md) | Issue #623 slice: centralize concrete CPU sessions on explicit contexts before CUDA dispatch |
| [cuda-tree-contraction.md](./cuda-tree-contraction.md) | Issues #623/#553 single-CUDA explicit transfer and TreeTN contraction vertical slice |
| [context-scoped-src-contraction.md](./context-scoped-src-contraction.md) | Issue #720 caller-owned context construction, factorization, adaptive decisions, and CUDA-resident SRC |
| [torch_backend.md](./torch_backend.md) | PyTorch backend design exploration |
| [tenferro_ad_scalar_operator_extension_note.md](./tenferro_ad_scalar_operator_extension_note.md) | Tenferro AD scalar operator extension notes |
| [build-profiles.md](./build-profiles.md) | Debug-free ordinary Cargo profiles and the opt-in full-debug release profile |

## Tensor Networks

| Document | Description |
|----------|-------------|
| [adaptive-tci-interpolation.md](./adaptive-tci-interpolation.md) | Adaptive TCI patching, convergence, pivot recycling, and structured embedding |
| [adaptive-tci-parallel-execution.md](./adaptive-tci-parallel-execution.md) | Optional Hataori/Rayon/MPI patch scheduling and one-pass cache projection |
| [partitionedtt-projector-invariants.md](./partitionedtt-projector-invariants.md) | Issue #634 design for coherent projector identity, validation, and transactional PartitionedTT mutation |
| [partitioned-treetn.md](./partitioned-treetn.md) | Issue #648 migration design for TreeTN-native eager partitioning and adaptive patching |
| [tree-adaptive-patching-roadmap.md](./tree-adaptive-patching-roadmap.md) | Milestone roadmap for adaptive patched interpolation, contraction, and parallel execution on arbitrary trees |
| [treetn-contraction-outcome.md](./treetn-contraction-outcome.md) | Proposal: TreeTN contraction outcome report and rank-threshold early abort (tree patching M6) |
| [tree-interpolation-engine-seam.md](./tree-interpolation-engine-seam.md) | Engine-agnostic tree interpolation contract in treetn and its TreeTCI implementation (tree patching M1, implemented) |
| [tree-interpolation-edge-pivots.md](./tree-interpolation-edge-pivots.md) | Proposal: optional selected edge pivots through the interpolation engine seam for a future recursive split-selection study; not implemented |
| [tree-pqtci-driver.md](./tree-pqtci-driver.md) | Sequential adaptive patched interpolation driver in partitionedtreetn (tree patching M2, implemented) |
| [tree-pqtci-split-selection.md](./tree-pqtci-split-selection.md) | M5 literature review and open decisions for split-site selection, patch-size bounds, and sibling merging |
| [tree-pqtci-patch-size-bounds.md](./tree-pqtci-patch-size-bounds.md) | Implementation plan for the minimum patch size and the capped-patch size bound of tree pQTCI (tree patching M5 open questions 2 and 4; not implemented) |
| [tree-patching-error-contract.md](./tree-patching-error-contract.md) | Measured L2 error contract (certified where exact or exhaustive, estimated where sampled), user-selectable error norm, and per-patch budget for tree adaptive patching (tree patching M3; interpolation side implemented, open questions 1, 2, 4, 5, 8, and 9 decided) |
| [tree-patching-findings.md](./tree-patching-findings.md) | Verified facts about current code used by the tree patching milestones: contraction reporting, element-wise product, fixed sites, patch representation, avoidable overhead, sparse storage decision |
| [orthogonal-target-reconstruction.md](./orthogonal-target-reconstruction.md) | Fixed global L2 target, gain-driven reconstruction, superpositions, subset-QFT integration, and the level-coupled merge-refine schedule |
| [gse-chain-mps-algorithm.md](./gse-chain-mps-algorithm.md) | Chain MPS global subspace expansion analysis for TreeTN GSE-TDVP planning |
| [itensormps-compatible-zipup.md](./itensormps-compatible-zipup.md) | ITensorMPS-compatible chain zip-up contraction schedule and policy-aware decomposition follow-up |
| [fit-sum.md](./fit-sum.md) | Variational fitting of compatible TreeTN sums without exact direct-sum materialization |

## Automatic Differentiation

| Document | Description |
|----------|-------------|
| [three_mode_ad_design.md](./three_mode_ad_design.md) | Three-mode automatic differentiation design |

## Julia Compatibility

| Document | Description |
|----------|-------------|
| [quanticstransform_julia_comparison.md](./quanticstransform_julia_comparison.md) | Quantics transform Julia compatibility analysis |
