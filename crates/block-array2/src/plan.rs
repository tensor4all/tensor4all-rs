//! Contraction order optimization and contraction plan generation.

use std::collections::{HashMap, HashSet};

use crate::error::{BlockArray2Error, Result};
use crate::partition::BlockPartition;
use crate::structure::BlockStructure;

/// Index (edge) label.
pub type IndexLabel = String;

/// An atom used to name tensors.
///
/// Tensor names are represented as `Vec<TensorLabelAtom>` and normalized to a sorted order.
pub type TensorLabelAtom = u32;

/// Tensor name/label (normalized).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorLabel(pub Vec<TensorLabelAtom>);

impl TensorLabel {
    pub fn new(mut atoms: Vec<TensorLabelAtom>) -> Self {
        atoms.sort_unstable();
        Self(atoms)
    }

    pub fn merged(lhs: &TensorLabel, rhs: &TensorLabel) -> TensorLabel {
        let mut out = lhs.0.clone();
        out.extend_from_slice(&rhs.0);
        out.sort_unstable();
        TensorLabel(out)
    }
}

#[derive(Debug, Clone)]
pub struct PairContractionPlan {
    pub lhs_tensor: TensorLabel,
    pub rhs_tensor: TensorLabel,
    pub out_tensor: TensorLabel,

    pub lhs_labels: Vec<IndexLabel>,
    pub rhs_labels: Vec<IndexLabel>,
    pub out_labels: Vec<IndexLabel>,

    pub lhs_axes: Vec<usize>,
    pub rhs_axes: Vec<usize>,

    pub estimated_cost: u64,
    pub out_structure: BlockStructure,
}

#[derive(Debug, Clone)]
pub struct ContractionPathPlan {
    pub steps: Vec<PairContractionPlan>,
    pub outputs: Vec<TensorLabel>,
}

type TensorId = usize;

#[derive(Debug, Clone)]
struct EdgeInfo {
    partition: BlockPartition,
    connections: Vec<(TensorId, usize)>,
}

#[derive(Debug, Clone)]
struct NetworkTensor {
    tensor_label: TensorLabel,
    structure: BlockStructure,
    index_labels: Vec<IndexLabel>,
}

/// A planner for tensor-network contraction order.
///
/// This planner enforces a simple (graph-like) index structure:
/// - An index label may connect at most two tensors (degree <= 2).
/// - Duplicate labels within the same tensor are rejected.
#[derive(Debug, Default, Clone)]
pub struct TensorNetworkPlanner {
    tensors: HashMap<TensorId, NetworkTensor>,
    next_id: TensorId,
    edges: HashMap<IndexLabel, EdgeInfo>,
}

impl TensorNetworkPlanner {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_tensor(
        &mut self,
        tensor_label: TensorLabel,
        structure: BlockStructure,
        index_labels: Vec<IndexLabel>,
    ) -> Result<TensorId> {
        if structure.rank() != index_labels.len() {
            return Err(BlockArray2Error::RankMismatch {
                rank: structure.rank(),
                labels_len: index_labels.len(),
            });
        }

        let mut seen = HashSet::with_capacity(index_labels.len());
        for label in &index_labels {
            if !seen.insert(label.clone()) {
                return Err(BlockArray2Error::DuplicateIndexLabelWithinTensor {
                    label: label.clone(),
                });
            }
        }

        let id = self.next_id;
        self.next_id += 1;

        // Update edge information + validate hyperedges/partition mismatches.
        for (axis, label) in index_labels.iter().enumerate() {
            let partition = structure.partitions()[axis].clone();
            if let Some(edge) = self.edges.get_mut(label) {
                if edge.connections.len() >= 2 {
                    return Err(BlockArray2Error::HyperedgeNotSupported {
                        label: label.clone(),
                    });
                }
                if edge.partition != partition {
                    return Err(BlockArray2Error::PartitionMismatch {
                        label: label.clone(),
                    });
                }
                edge.connections.push((id, axis));
            } else {
                self.edges.insert(
                    label.clone(),
                    EdgeInfo {
                        partition,
                        connections: vec![(id, axis)],
                    },
                );
            }
        }

        self.tensors.insert(
            id,
            NetworkTensor {
                tensor_label,
                structure,
                index_labels,
            },
        );
        Ok(id)
    }

    /// Compute a greedy contraction path plan.
    ///
    /// This only contracts along *internal* indices (labels with degree == 2).
    /// If the network cannot be fully contracted by internal indices (disconnected),
    /// this returns `DisconnectedNetwork`.
    pub fn plan_greedy(&self) -> Result<ContractionPathPlan> {
        let mut working: TensorNetworkPlanner = self.clone();
        let mut steps = Vec::new();

        while working.tensors.len() > 1 {
            let Some((id1, id2, plan)) = working.find_best_pair()? else {
                return Err(BlockArray2Error::DisconnectedNetwork);
            };
            working.apply_pair_contraction(id1, id2, &plan)?;
            steps.push(plan);
        }

        let outputs = working
            .tensors
            .values()
            .map(|t| t.tensor_label.clone())
            .collect::<Vec<_>>();

        Ok(ContractionPathPlan { steps, outputs })
    }

    fn internal_shared_labels(&self, id1: TensorId, id2: TensorId) -> Vec<IndexLabel> {
        let t1 = &self.tensors[&id1];
        let t2 = &self.tensors[&id2];
        let labels1: HashSet<_> = t1.index_labels.iter().cloned().collect();
        let labels2: HashSet<_> = t2.index_labels.iter().cloned().collect();

        let mut shared: Vec<_> = labels1
            .intersection(&labels2)
            .filter(|lab| {
                self.edges
                    .get(*lab)
                    .is_some_and(|e| e.connections.len() == 2)
            })
            .cloned()
            .collect();
        shared.sort();
        shared
    }

    fn axes_for_labels(labels: &[IndexLabel], shared: &HashSet<IndexLabel>) -> Vec<usize> {
        labels
            .iter()
            .enumerate()
            .filter_map(|(i, l)| shared.contains(l).then_some(i))
            .collect()
    }

    fn compute_out_labels(
        lhs: &[IndexLabel],
        rhs: &[IndexLabel],
        shared: &HashSet<IndexLabel>,
    ) -> Vec<IndexLabel> {
        lhs.iter()
            .filter(|l| !shared.contains(*l))
            .cloned()
            .chain(rhs.iter().filter(|l| !shared.contains(*l)).cloned())
            .collect()
    }

    fn find_best_pair(&self) -> Result<Option<(TensorId, TensorId, PairContractionPlan)>> {
        let ids: Vec<TensorId> = self.tensors.keys().copied().collect();
        let mut best: Option<(u64, (TensorId, TensorId), PairContractionPlan)> = None;

        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                let id1 = ids[i];
                let id2 = ids[j];

                let shared = self.internal_shared_labels(id1, id2);
                if shared.is_empty() {
                    continue;
                }
                let shared_set: HashSet<_> = shared.iter().cloned().collect();

                let t1 = &self.tensors[&id1];
                let t2 = &self.tensors[&id2];

                let lhs_axes = Self::axes_for_labels(&t1.index_labels, &shared_set);
                let rhs_axes = Self::axes_for_labels(&t2.index_labels, &shared_set);

                let estimated_cost =
                    t1.structure
                        .estimate_tensordot_cost(&t2.structure, &lhs_axes, &rhs_axes);

                let out_labels =
                    Self::compute_out_labels(&t1.index_labels, &t2.index_labels, &shared_set);
                let out_partitions: Vec<BlockPartition> = t1
                    .index_labels
                    .iter()
                    .enumerate()
                    .filter_map(|(axis, l)| {
                        (!shared_set.contains(l)).then_some(t1.structure.partitions()[axis].clone())
                    })
                    .chain(t2.index_labels.iter().enumerate().filter_map(|(axis, l)| {
                        (!shared_set.contains(l)).then_some(t2.structure.partitions()[axis].clone())
                    }))
                    .collect();

                let out_nnz = BlockStructure::estimate_nnz_after_contraction(
                    &t1.structure,
                    &t2.structure,
                    shared.len(),
                );
                let out_structure = BlockStructure::new(out_partitions, out_nnz);

                let lhs_tensor = t1.tensor_label.clone();
                let rhs_tensor = t2.tensor_label.clone();
                let out_tensor = TensorLabel::merged(&lhs_tensor, &rhs_tensor);

                let plan = PairContractionPlan {
                    lhs_tensor,
                    rhs_tensor,
                    out_tensor,
                    lhs_labels: t1.index_labels.clone(),
                    rhs_labels: t2.index_labels.clone(),
                    out_labels,
                    lhs_axes,
                    rhs_axes,
                    estimated_cost,
                    out_structure,
                };

                let (a, b) = if id1 < id2 { (id1, id2) } else { (id2, id1) };
                let tie_key = (plan.out_tensor.clone(), a, b);

                match &best {
                    None => best = Some((estimated_cost, (a, b), plan)),
                    Some((best_cost, (ba, bb), best_plan)) => {
                        if estimated_cost < *best_cost {
                            best = Some((estimated_cost, (a, b), plan));
                        } else if estimated_cost == *best_cost {
                            let best_key = (best_plan.out_tensor.clone(), *ba, *bb);
                            if tie_key < best_key {
                                best = Some((estimated_cost, (a, b), plan));
                            }
                        }
                    }
                }
            }
        }

        Ok(best.map(|(_, (a, b), plan)| (a, b, plan)))
    }

    fn apply_pair_contraction(
        &mut self,
        id1: TensorId,
        id2: TensorId,
        plan: &PairContractionPlan,
    ) -> Result<TensorId> {
        let _t1 = self.tensors.remove(&id1).expect("tensor exists");
        let _t2 = self.tensors.remove(&id2).expect("tensor exists");

        // Remove old connections and insert the new tensor connections.
        // For simplicity, we rebuild the edge map deterministically from remaining tensors + new tensor.
        let mut tensors: Vec<(TensorLabel, BlockStructure, Vec<IndexLabel>)> = self
            .tensors
            .values()
            .map(|t| {
                (
                    t.tensor_label.clone(),
                    t.structure.clone(),
                    t.index_labels.clone(),
                )
            })
            .collect();
        tensors.push((
            plan.out_tensor.clone(),
            plan.out_structure.clone(),
            plan.out_labels.clone(),
        ));

        *self = TensorNetworkPlanner::new();
        for (lbl, st, inds) in tensors {
            let _ = self.add_tensor(lbl, st, inds)?;
        }

        // Return new tensor id is not stable; but callers don't use it.
        Ok(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::partition::BlockPartition;
    use crate::structure::BlockStructure;

    fn bs1(rank: usize) -> BlockStructure {
        let parts = (0..rank).map(|_| BlockPartition::trivial(2)).collect();
        BlockStructure::new(parts, 1)
    }

    #[test]
    fn test_reject_duplicate_label_within_tensor() {
        let mut p = TensorNetworkPlanner::new();
        let err = p
            .add_tensor(
                TensorLabel::new(vec![1]),
                bs1(2),
                vec!["i".into(), "i".into()],
            )
            .unwrap_err();
        matches!(
            err,
            BlockArray2Error::DuplicateIndexLabelWithinTensor { .. }
        );
    }

    #[test]
    fn test_reject_hyperedge() {
        let mut p = TensorNetworkPlanner::new();
        p.add_tensor(TensorLabel::new(vec![1]), bs1(1), vec!["i".into()])
            .unwrap();
        p.add_tensor(TensorLabel::new(vec![2]), bs1(1), vec!["i".into()])
            .unwrap();
        let err = p
            .add_tensor(TensorLabel::new(vec![3]), bs1(1), vec!["i".into()])
            .unwrap_err();
        matches!(err, BlockArray2Error::HyperedgeNotSupported { .. });
    }

    #[test]
    fn test_plan_chain_has_two_steps() {
        let mut p = TensorNetworkPlanner::new();
        p.add_tensor(
            TensorLabel::new(vec![1]),
            bs1(2),
            vec!["i".into(), "j".into()],
        )
        .unwrap();
        p.add_tensor(
            TensorLabel::new(vec![2]),
            bs1(2),
            vec!["j".into(), "k".into()],
        )
        .unwrap();
        p.add_tensor(
            TensorLabel::new(vec![3]),
            bs1(2),
            vec!["k".into(), "l".into()],
        )
        .unwrap();

        let plan = p.plan_greedy().unwrap();
        assert_eq!(plan.steps.len(), 2);

        // Final tensor name should contain all three atoms, sorted.
        assert_eq!(plan.outputs.len(), 1);
        assert_eq!(plan.outputs[0], TensorLabel::new(vec![1, 2, 3]));

        // Each step should have deterministic, normalized out_tensor.
        for step in &plan.steps {
            let atoms = step.out_tensor.0.clone();
            let mut sorted = atoms.clone();
            sorted.sort_unstable();
            assert_eq!(atoms, sorted);
        }
    }
}
