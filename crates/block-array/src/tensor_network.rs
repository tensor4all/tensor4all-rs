//! Tensor network representation and contraction optimization.
//!
//! This module provides:
//! - `TensorNetwork`: A collection of tensors with labeled edges
//! - Greedy contraction order optimization
//! - Multi-tensor contraction execution

use std::collections::{HashMap, HashSet};

use crate::block_array::BlockArray;
use crate::partition::BlockPartition;
use crate::scalar::Scalar;

/// Unique identifier for a tensor in the network.
pub type TensorId = usize;

/// Label for tensor edges (indices).
/// Tensors sharing the same edge label will be contracted along that edge.
pub type EdgeLabel = String;

/// Information about an edge in the tensor network.
#[derive(Debug, Clone)]
pub struct EdgeInfo {
    /// The partition for this edge.
    pub partition: BlockPartition,
    /// Tensors connected to this edge: (tensor_id, axis_index).
    pub connections: Vec<(TensorId, usize)>,
}

/// A tensor in the network with labeled edges.
#[derive(Debug, Clone)]
struct NetworkTensor<T: Scalar> {
    /// The actual blocked array data.
    data: BlockArray<T>,
    /// Edge labels for each axis.
    edge_labels: Vec<EdgeLabel>,
}

/// A tensor network: collection of tensors with labeled edges.
///
/// Tensors sharing the same edge label will be contracted along that edge.
/// The contraction order is optimized using a greedy algorithm.
#[derive(Debug)]
pub struct TensorNetwork<T: Scalar> {
    /// Tensors in the network.
    tensors: HashMap<TensorId, NetworkTensor<T>>,
    /// Next available tensor ID.
    next_id: TensorId,
    /// Edge information indexed by label.
    edges: HashMap<EdgeLabel, EdgeInfo>,
}

impl<T: Scalar> TensorNetwork<T> {
    /// Create an empty tensor network.
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
            next_id: 0,
            edges: HashMap::new(),
        }
    }

    /// Add a tensor to the network with labeled edges.
    ///
    /// # Arguments
    /// * `tensor` - The blocked array to add
    /// * `edge_labels` - Labels for each axis (length must match tensor rank)
    ///
    /// # Returns
    /// The tensor ID assigned to this tensor.
    ///
    /// # Panics
    /// - If edge_labels length doesn't match tensor rank
    /// - If an edge label appears more than once within the same tensor
    /// - If an edge label would end up connected to 3 or more tensors (hyperedge not supported)
    /// - If an edge label is reused with incompatible partition
    pub fn add_tensor(&mut self, tensor: BlockArray<T>, edge_labels: Vec<EdgeLabel>) -> TensorId {
        assert_eq!(
            tensor.rank(),
            edge_labels.len(),
            "Edge labels length {} must match tensor rank {}",
            edge_labels.len(),
            tensor.rank()
        );

        // This TensorNetwork implementation assumes a simple graph-like index structure:
        // each edge label may appear on at most two tensors.
        let mut seen = HashSet::with_capacity(edge_labels.len());
        for label in &edge_labels {
            assert!(
                seen.insert(label),
                "Edge label '{}' appears more than once within the same tensor. \
Trace/self-contraction is not supported by TensorNetwork.",
                label
            );
        }

        let id = self.next_id;
        self.next_id += 1;

        // Update edge information
        for (axis, label) in edge_labels.iter().enumerate() {
            let partition = tensor.partitions()[axis].clone();

            if let Some(edge_info) = self.edges.get_mut(label) {
                assert!(
                    edge_info.connections.len() < 2,
                    "Edge label '{}' would connect to 3 or more tensors (hyperedge). \
TensorNetwork only supports labels with degree <= 2. Existing connections: {:?}",
                    label,
                    edge_info.connections
                );
                // Verify partition compatibility
                assert_eq!(
                    edge_info.partition, partition,
                    "Edge '{}' partition mismatch: existing {:?} vs new {:?}",
                    label, edge_info.partition, partition
                );
                edge_info.connections.push((id, axis));
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
                data: tensor,
                edge_labels,
            },
        );

        id
    }

    /// Get the number of tensors in the network.
    pub fn num_tensors(&self) -> usize {
        self.tensors.len()
    }

    /// Get edge labels that connect two specific tensors.
    fn shared_edges(&self, id1: TensorId, id2: TensorId) -> Vec<EdgeLabel> {
        let t1 = self.tensors.get(&id1).expect("Tensor not found");
        let t2 = self.tensors.get(&id2).expect("Tensor not found");

        let labels1: HashSet<_> = t1.edge_labels.iter().collect();
        let labels2: HashSet<_> = t2.edge_labels.iter().collect();

        labels1.intersection(&labels2).map(|&s| s.clone()).collect()
    }

    /// Find axes to contract between two tensors.
    fn contraction_axes(&self, id1: TensorId, id2: TensorId) -> (Vec<usize>, Vec<usize>) {
        let shared = self.shared_edges(id1, id2);
        let t1 = self.tensors.get(&id1).unwrap();
        let t2 = self.tensors.get(&id2).unwrap();

        let mut axes1 = Vec::new();
        let mut axes2 = Vec::new();

        for label in &shared {
            for (axis, l) in t1.edge_labels.iter().enumerate() {
                if l == label {
                    axes1.push(axis);
                }
            }
            for (axis, l) in t2.edge_labels.iter().enumerate() {
                if l == label {
                    axes2.push(axis);
                }
            }
        }

        (axes1, axes2)
    }

    /// Estimate the cost of contracting two tensors.
    pub fn estimate_contraction_cost(&self, id1: TensorId, id2: TensorId) -> u64 {
        let t1 = self.tensors.get(&id1).expect("Tensor not found");
        let t2 = self.tensors.get(&id2).expect("Tensor not found");
        let (axes1, axes2) = self.contraction_axes(id1, id2);

        t1.data
            .structure()
            .estimate_tensordot_cost(t2.data.structure(), &axes1, &axes2)
    }

    /// Find all pairs of tensors that share at least one edge.
    fn contractable_pairs(&self) -> Vec<(TensorId, TensorId)> {
        let mut ids: Vec<_> = self.tensors.keys().copied().collect();
        ids.sort(); // Ensure deterministic ordering
        let mut pairs = Vec::new();

        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                if !self.shared_edges(ids[i], ids[j]).is_empty() {
                    pairs.push((ids[i], ids[j]));
                }
            }
        }

        pairs
    }

    /// Find the best contraction pair using greedy selection (minimum cost).
    pub fn find_best_contraction(&self) -> Option<(TensorId, TensorId, u64)> {
        let pairs = self.contractable_pairs();
        if pairs.is_empty() {
            return None;
        }

        pairs
            .into_iter()
            .map(|(id1, id2)| {
                let cost = self.estimate_contraction_cost(id1, id2);
                (id1, id2, cost)
            })
            .min_by_key(|&(_, _, cost)| cost)
    }

    /// Contract two tensors and update the network.
    ///
    /// Returns the ID of the resulting tensor.
    fn contract_pair(&mut self, id1: TensorId, id2: TensorId) -> TensorId {
        let t1 = self.tensors.remove(&id1).expect("Tensor not found");
        let t2 = self.tensors.remove(&id2).expect("Tensor not found");

        let (axes1, axes2) = self.contraction_axes_from_tensors(&t1, &t2);
        let shared_labels: HashSet<_> = self.shared_edges_from_tensors(&t1, &t2);

        // Compute result edge labels (free edges from both tensors)
        let free_labels1: Vec<_> = t1
            .edge_labels
            .iter()
            .enumerate()
            .filter(|(axis, _)| !axes1.contains(axis))
            .map(|(_, label)| label.clone())
            .collect();
        let free_labels2: Vec<_> = t2
            .edge_labels
            .iter()
            .enumerate()
            .filter(|(axis, _)| !axes2.contains(axis))
            .map(|(_, label)| label.clone())
            .collect();

        let result_labels: Vec<_> = free_labels1.into_iter().chain(free_labels2).collect();

        // Perform contraction
        let result_data = t1.data.tensordot(&t2.data, &axes1, &axes2);

        // Update edge information
        for label in &shared_labels {
            if let Some(edge_info) = self.edges.get_mut(label) {
                edge_info
                    .connections
                    .retain(|&(tid, _)| tid != id1 && tid != id2);
                // Shared labels are contracted away; if nothing remains, drop the edge entry.
                if edge_info.connections.is_empty() {
                    // Defer removal to avoid mutable borrow issues by removing after loop.
                }
            }
        }

        // Remove fully-contracted edges (no remaining connections).
        for label in &shared_labels {
            let remove = self
                .edges
                .get(label)
                .map(|e| e.connections.is_empty())
                .unwrap_or(false);
            if remove {
                self.edges.remove(label);
            }
        }

        // Add result tensor
        let result_id = self.next_id;
        self.next_id += 1;

        // Update edge connections for result tensor
        for (axis, label) in result_labels.iter().enumerate() {
            if let Some(edge_info) = self.edges.get_mut(label) {
                edge_info
                    .connections
                    .retain(|&(tid, _)| tid != id1 && tid != id2);
                edge_info.connections.push((result_id, axis));
            }
        }

        self.tensors.insert(
            result_id,
            NetworkTensor {
                data: result_data,
                edge_labels: result_labels,
            },
        );

        result_id
    }

    fn contraction_axes_from_tensors(
        &self,
        t1: &NetworkTensor<T>,
        t2: &NetworkTensor<T>,
    ) -> (Vec<usize>, Vec<usize>) {
        let shared = self.shared_edges_from_tensors(t1, t2);

        let mut axes1 = Vec::new();
        let mut axes2 = Vec::new();

        for label in &shared {
            for (axis, l) in t1.edge_labels.iter().enumerate() {
                if l == label {
                    axes1.push(axis);
                }
            }
            for (axis, l) in t2.edge_labels.iter().enumerate() {
                if l == label {
                    axes2.push(axis);
                }
            }
        }

        (axes1, axes2)
    }

    fn shared_edges_from_tensors(
        &self,
        t1: &NetworkTensor<T>,
        t2: &NetworkTensor<T>,
    ) -> HashSet<EdgeLabel> {
        let labels1: HashSet<_> = t1.edge_labels.iter().cloned().collect();
        let labels2: HashSet<_> = t2.edge_labels.iter().cloned().collect();
        labels1.intersection(&labels2).cloned().collect()
    }

    /// Contract all tensors in the network using greedy order.
    ///
    /// Returns the final tensor (or None if network is empty).
    ///
    /// If the network is disconnected (no shared edges between any remaining tensors),
    /// this continues contracting by taking an outer product between an arbitrary pair.
    pub fn contract_all(&mut self) -> Option<BlockArray<T>> {
        while self.tensors.len() > 1 {
            if let Some((id1, id2, _cost)) = self.find_best_contraction() {
                self.contract_pair(id1, id2);
            } else {
                // No more contractable pairs - tensors are disconnected.
                // Continue by contracting an arbitrary pair via outer product (no shared edges).
                let mut ids: Vec<_> = self.tensors.keys().copied().collect();
                ids.sort();
                let id1 = ids[0];
                let id2 = ids[1];
                self.contract_pair(id1, id2);
            }
        }

        // Return the single remaining tensor (if any)
        self.tensors.values().next().map(|t| t.data.clone())
    }

    /// Contract all tensors and also return the remaining edge labels of the final tensor.
    ///
    /// This is useful when the caller needs to map result axes back to original labels.
    pub fn contract_all_with_labels(&mut self) -> Option<(BlockArray<T>, Vec<EdgeLabel>)> {
        while self.tensors.len() > 1 {
            if let Some((id1, id2, _cost)) = self.find_best_contraction() {
                self.contract_pair(id1, id2);
            } else {
                let mut ids: Vec<_> = self.tensors.keys().copied().collect();
                ids.sort();
                let id1 = ids[0];
                let id2 = ids[1];
                self.contract_pair(id1, id2);
            }
        }

        self.tensors
            .values()
            .next()
            .map(|t| (t.data.clone(), t.edge_labels.clone()))
    }

    /// Get the contraction order as a sequence of (id1, id2) pairs.
    ///
    /// This returns the order in which tensors would be contracted,
    /// without actually performing the contractions.
    pub fn get_contraction_order(&self) -> Vec<(TensorId, TensorId)> {
        let mut network = TensorNetwork {
            tensors: self
                .tensors
                .iter()
                .map(|(&id, t)| {
                    (
                        id,
                        NetworkTensor {
                            data: t.data.clone(),
                            edge_labels: t.edge_labels.clone(),
                        },
                    )
                })
                .collect(),
            next_id: self.next_id,
            edges: self.edges.clone(),
        };

        let mut order = Vec::new();

        while network.tensors.len() > 1 {
            if let Some((id1, id2, _cost)) = network.find_best_contraction() {
                order.push((id1, id2));
                network.contract_pair(id1, id2);
            } else {
                break;
            }
        }

        order
    }

    /// Get total estimated cost for contracting the entire network.
    pub fn estimate_total_cost(&self) -> u64 {
        let mut network = TensorNetwork {
            tensors: self
                .tensors
                .iter()
                .map(|(&id, t)| {
                    (
                        id,
                        NetworkTensor {
                            data: t.data.clone(),
                            edge_labels: t.edge_labels.clone(),
                        },
                    )
                })
                .collect(),
            next_id: self.next_id,
            edges: self.edges.clone(),
        };

        let mut total_cost = 0u64;

        while network.tensors.len() > 1 {
            if let Some((id1, id2, cost)) = network.find_best_contraction() {
                total_cost += cost;
                network.contract_pair(id1, id2);
            } else {
                break;
            }
        }

        total_cost
    }
}

impl<T: Scalar> Default for TensorNetwork<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_data::BlockData;
    use mdarray::Tensor;

    fn make_block_data(shape: &[usize], start_val: f64) -> BlockData<f64> {
        let tensor = Tensor::from_fn(shape, |idx| {
            let linear: usize = idx.iter().enumerate().fold(0, |acc, (i, &x)| {
                let stride: usize = shape[i + 1..].iter().product();
                acc + x * stride.max(1)
            });
            start_val + linear as f64
        });
        BlockData::from_tensor(tensor)
    }

    #[test]
    fn test_tensor_network_basic() {
        let mut network = TensorNetwork::<f64>::new();

        // Create tensors A[i,j] and B[j,k]
        let part_i = BlockPartition::trivial(2);
        let part_j = BlockPartition::trivial(3);
        let part_k = BlockPartition::trivial(4);

        let mut a = BlockArray::new(vec![part_i.clone(), part_j.clone()]);
        a.set_block(vec![0, 0], make_block_data(&[2, 3], 1.0));

        let mut b = BlockArray::new(vec![part_j, part_k]);
        b.set_block(vec![0, 0], make_block_data(&[3, 4], 1.0));

        let id_a = network.add_tensor(a, vec!["i".to_string(), "j".to_string()]);
        let id_b = network.add_tensor(b, vec!["j".to_string(), "k".to_string()]);

        assert_eq!(network.num_tensors(), 2);

        // Check shared edges
        let shared = network.shared_edges(id_a, id_b);
        assert_eq!(shared.len(), 1);
        assert!(shared.contains(&"j".to_string()));

        // Contract
        let result = network.contract_all().unwrap();
        assert_eq!(result.shape(), vec![2, 4]);
    }

    #[test]
    fn test_tensor_network_chain() {
        let mut network = TensorNetwork::<f64>::new();

        // Create chain: A[i,j] @ B[j,k] @ C[k,l]
        let part_i = BlockPartition::trivial(2);
        let part_j = BlockPartition::trivial(3);
        let part_k = BlockPartition::trivial(4);
        let part_l = BlockPartition::trivial(5);

        let mut a = BlockArray::new(vec![part_i, part_j.clone()]);
        a.set_block(vec![0, 0], make_block_data(&[2, 3], 1.0));

        let mut b = BlockArray::new(vec![part_j, part_k.clone()]);
        b.set_block(vec![0, 0], make_block_data(&[3, 4], 1.0));

        let mut c = BlockArray::new(vec![part_k, part_l]);
        c.set_block(vec![0, 0], make_block_data(&[4, 5], 1.0));

        network.add_tensor(a, vec!["i".to_string(), "j".to_string()]);
        network.add_tensor(b, vec!["j".to_string(), "k".to_string()]);
        network.add_tensor(c, vec!["k".to_string(), "l".to_string()]);

        assert_eq!(network.num_tensors(), 3);

        // Get contraction order
        let order = network.get_contraction_order();
        assert_eq!(order.len(), 2);

        // Contract all
        let result = network.contract_all().unwrap();
        // Result has free edges i and l, dimensions [2, 5]
        // Order depends on contraction sequence, so just check dimensions are correct
        let mut shape = result.shape();
        shape.sort();
        assert_eq!(shape, vec![2, 5]);
    }

    #[test]
    fn test_tensor_network_triangle() {
        let mut network = TensorNetwork::<f64>::new();

        // Create triangle: A[i,j], B[j,k], C[k,i] -> scalar
        let part = BlockPartition::trivial(3);

        let mut a = BlockArray::new(vec![part.clone(), part.clone()]);
        a.set_block(vec![0, 0], make_block_data(&[3, 3], 1.0));

        let mut b = BlockArray::new(vec![part.clone(), part.clone()]);
        b.set_block(vec![0, 0], make_block_data(&[3, 3], 1.0));

        let mut c = BlockArray::new(vec![part.clone(), part]);
        c.set_block(vec![0, 0], make_block_data(&[3, 3], 1.0));

        network.add_tensor(a, vec!["i".to_string(), "j".to_string()]);
        network.add_tensor(b, vec!["j".to_string(), "k".to_string()]);
        network.add_tensor(c, vec!["k".to_string(), "i".to_string()]);

        // Contract all
        let result = network.contract_all().unwrap();
        // Triangle contraction gives scalar
        assert_eq!(result.shape().iter().product::<usize>(), 1);
    }

    #[test]
    fn test_contraction_cost_estimate() {
        let mut network = TensorNetwork::<f64>::new();

        // A[i,j] = [10, 100], B[j,k] = [100, 10], C[k,l] = [10, 10]
        // A@B: contracts j (100), output [10, 10], cost = 2 * 10 * 100 * 10 = 20000
        // B@C: contracts k (10), output [100, 10], cost = 2 * 100 * 10 * 10 = 20000
        // Let's use asymmetric dimensions to create different costs
        //
        // A[i,j] = [5, 100], B[j,k] = [100, 20], C[k,l] = [20, 3]
        // A@B: contracts j (100), output [5, 20], cost = 2 * 5 * 100 * 20 = 20000
        // B@C: contracts k (20), output [100, 3], cost = 2 * 100 * 20 * 3 = 12000
        let part_5 = BlockPartition::trivial(5);
        let part_100 = BlockPartition::trivial(100);
        let part_20 = BlockPartition::trivial(20);
        let part_3 = BlockPartition::trivial(3);

        let mut a = BlockArray::new(vec![part_5.clone(), part_100.clone()]);
        a.set_block(vec![0, 0], make_block_data(&[5, 100], 1.0));

        let mut b = BlockArray::new(vec![part_100, part_20.clone()]);
        b.set_block(vec![0, 0], make_block_data(&[100, 20], 1.0));

        let mut c = BlockArray::new(vec![part_20, part_3]);
        c.set_block(vec![0, 0], make_block_data(&[20, 3], 1.0));

        let id_a = network.add_tensor(a, vec!["i".to_string(), "j".to_string()]);
        let id_b = network.add_tensor(b, vec!["j".to_string(), "k".to_string()]);
        let id_c = network.add_tensor(c, vec!["k".to_string(), "l".to_string()]);

        // A@B (contracts over 100) should be more expensive than B@C (contracts over 20)
        let cost_ab = network.estimate_contraction_cost(id_a, id_b);
        let cost_bc = network.estimate_contraction_cost(id_b, id_c);

        assert!(
            cost_ab > cost_bc,
            "A@B ({}) should cost more than B@C ({})",
            cost_ab,
            cost_bc
        );
    }

    #[test]
    #[should_panic(expected = "degree <= 2")]
    fn test_reject_hyperedge_degree_ge_3() {
        let mut network = TensorNetwork::<f64>::new();

        let part_i = BlockPartition::trivial(2);
        let part_j = BlockPartition::trivial(3);

        let mut a = BlockArray::new(vec![part_i.clone(), part_j.clone()]);
        a.set_block(vec![0, 0], make_block_data(&[2, 3], 1.0));

        let mut b = BlockArray::new(vec![part_i.clone(), part_j.clone()]);
        b.set_block(vec![0, 0], make_block_data(&[2, 3], 1.0));

        let mut c = BlockArray::new(vec![part_i, part_j]);
        c.set_block(vec![0, 0], make_block_data(&[2, 3], 1.0));

        // Label "j" would appear on three tensors.
        network.add_tensor(a, vec!["i".to_string(), "j".to_string()]);
        network.add_tensor(b, vec!["i2".to_string(), "j".to_string()]);
        network.add_tensor(c, vec!["i3".to_string(), "j".to_string()]);
    }

    #[test]
    #[should_panic(expected = "appears more than once within the same tensor")]
    fn test_reject_duplicate_label_within_tensor() {
        let mut network = TensorNetwork::<f64>::new();

        let part = BlockPartition::trivial(3);
        let mut a = BlockArray::new(vec![part.clone(), part]);
        a.set_block(vec![0, 0], make_block_data(&[3, 3], 1.0));

        // Duplicate label "i" on the same tensor is not supported.
        network.add_tensor(a, vec!["i".to_string(), "i".to_string()]);
    }
}
