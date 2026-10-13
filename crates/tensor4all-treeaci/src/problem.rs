//! Validated topology and physical-index preparation for tree ACI.

use std::collections::{HashMap, VecDeque};

use tensor4all_core::{DynIndex, IdxTensor, IndexLike};
use tensor4all_treetn::TreeTN;

use crate::{
    path_cover::SweepPlan, Result, TreeAciError, TreeAciNode, TreeAciOptions, TreeAciScalar,
};

pub(crate) type DirectedEdgeId = usize;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DirectedEdge<V> {
    pub(crate) id: DirectedEdgeId,
    pub(crate) from: V,
    pub(crate) to: V,
    pub(crate) reverse: DirectedEdgeId,
    pub(crate) incoming_to_from: Vec<DirectedEdgeId>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct LocalPhysicalPlan {
    pub(crate) indices: Vec<DynIndex>,
    pub(crate) dims: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    pub(crate) local_dim: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedTreeProblem<V> {
    pub(crate) node_order: Vec<V>,
    pub(crate) node_positions: HashMap<V, usize>,
    pub(crate) directed_edges: Vec<DirectedEdge<V>>,
    /// Directed cuts in dependency-first order. Every edge's incoming cuts
    /// occur before it, so frame and sample construction can be iterative.
    pub(crate) directed_dependency_order: Vec<DirectedEdgeId>,
    pub(crate) physical: Vec<LocalPhysicalPlan>,
    pub(crate) root: V,
    pub(crate) schedule: SweepPlan,
    pub(crate) max_sample_arena_bytes: usize,
    /// Resource ceilings resolved once, for the scalar type the run uses.
    ///
    /// Every ceiling below is the value
    /// [`TreeAciOptions::resolved_max_core_elements`] and its siblings return
    /// for that scalar, so a ceiling left unset by the caller has already been
    /// derived from `max_working_bytes` here and no later site re-derives it.
    pub(crate) max_frame_elements: usize,
    pub(crate) max_frame_bytes: usize,
    pub(crate) max_core_elements: usize,
    pub(crate) max_local_matrix_elements: usize,
    pub(crate) max_working_bytes: usize,
}

/// Validates the inputs and options of one run and resolves its ceilings.
///
/// Scalar `T` is the type the run's cores and local matrices are materialized
/// in, and is what the byte-derived element ceilings are charged against; see
/// [`TreeAciOptions::max_working_bytes`].
pub(crate) fn prepare_problem<T: TreeAciScalar, V: TreeAciNode>(
    inputs: &[TreeTN<IdxTensor, V>],
    options: &TreeAciOptions<V>,
) -> Result<PreparedTreeProblem<V>> {
    validate_options(options)?;
    let max_core_elements = options.resolved_max_core_elements::<T>();
    let max_local_matrix_elements = options.resolved_max_local_matrix_elements::<T>();
    let reference = inputs.first().ok_or(TreeAciError::NoInputs)?;
    if reference.node_count() == 0 {
        return Err(TreeAciError::EmptyTree { input: 0 });
    }
    reference.validate_tree()?;
    for (input, tree) in inputs.iter().enumerate().skip(1) {
        if tree.node_count() == 0 {
            return Err(TreeAciError::EmptyTree { input });
        }
        tree.validate_tree()?;
        if !reference.same_topology(tree) {
            return Err(TreeAciError::TopologyMismatch { input });
        }
    }

    let mut node_order = reference.node_names();
    node_order.sort();
    let node_positions = node_order
        .iter()
        .cloned()
        .enumerate()
        .map(|(position, node)| (node, position))
        .collect::<HashMap<_, _>>();

    let requested_root = match &options.root {
        Some(root) if node_positions.contains_key(root) => Some(root.clone()),
        Some(_) => {
            return Err(TreeAciError::InvalidOption {
                option: "root",
                message: "the requested root is not a node in the input tree",
            });
        }
        None => None,
    };

    let mut physical = Vec::with_capacity(node_order.len());
    for node in &node_order {
        let reference_space =
            reference
                .site_space(node)
                .ok_or(TreeAciError::InternalInvariant {
                    message: "a validated node is missing its physical-index space",
                })?;
        for (input, tree) in inputs.iter().enumerate().skip(1) {
            if tree.site_space(node) != Some(reference_space) {
                return Err(TreeAciError::PhysicalIndexMismatch {
                    input,
                    node: format!("{node:?}"),
                });
            }
        }

        let node_index = reference
            .node_index(node)
            .ok_or(TreeAciError::InternalInvariant {
                message: "a validated node name has no graph index",
            })?;
        let tensor = reference
            .tensor(node_index)
            .ok_or(TreeAciError::InternalInvariant {
                message: "a validated node has no tensor",
            })?;
        let indices = tensor
            .indices()
            .iter()
            .filter(|index| reference_space.contains(*index))
            .cloned()
            .collect::<Vec<_>>();
        if indices.len() != reference_space.len() {
            return Err(TreeAciError::InternalInvariant {
                message: "physical-index metadata disagrees with tensor axes",
            });
        }
        let dims = indices.iter().map(IndexLike::dim).collect::<Vec<_>>();
        if dims.contains(&0) {
            return Err(TreeAciError::PhysicalIndexMismatch {
                input: 0,
                node: format!("{node:?}"),
            });
        }
        let mut strides = Vec::with_capacity(dims.len());
        let mut local_dim = 1usize;
        for dim in &dims {
            strides.push(local_dim);
            local_dim = local_dim
                .checked_mul(*dim)
                .ok_or(TreeAciError::SizeOverflow {
                    context: "local physical dimension",
                })?;
        }
        enforce_limit("candidate rows", local_dim, options.max_candidate_rows)?;
        enforce_limit("candidate columns", local_dim, options.max_candidate_cols)?;
        physical.push(LocalPhysicalPlan {
            indices,
            dims,
            strides,
            local_dim,
        });
    }

    let mut edges = Vec::with_capacity(reference.edge_count());
    for (from_position, from) in node_order.iter().enumerate() {
        let from_index = reference
            .node_index(from)
            .ok_or(TreeAciError::InternalInvariant {
                message: "a validated node name has no graph index",
            })?;
        for (_, neighbor_index) in reference.edges_for_node(from_index) {
            let neighbor = reference
                .site_index_network()
                .node_name(neighbor_index)
                .ok_or(TreeAciError::InternalInvariant {
                    message: "a validated edge endpoint has no node name",
                })?;
            let to_position =
                *node_positions
                    .get(neighbor)
                    .ok_or(TreeAciError::InternalInvariant {
                        message: "a validated edge endpoint has no stable position",
                    })?;
            if from_position < to_position {
                edges.push((from_position, to_position));
            }
        }
    }
    edges.sort_unstable();

    // The byte budget is checked before any element ceiling derived from it,
    // so a run that the budget alone cannot admit is reported against the
    // budget the caller set rather than against a ceiling the caller never
    // saw. A derived ceiling then only ever narrows a run the budget would
    // have admitted. The minimum local matrix is charged at the widest
    // supported scalar, because it is a topology-only lower bound.
    for &(left, right) in &edges {
        let elements = physical[left]
            .local_dim
            .checked_mul(physical[right].local_dim)
            .ok_or(TreeAciError::SizeOverflow {
                context: "minimum local matrix",
            })?;
        let bytes = elements
            .checked_mul(std::mem::size_of::<num_complex::Complex64>())
            .ok_or(TreeAciError::SizeOverflow {
                context: "minimum local matrix bytes",
            })?;
        enforce_limit("working bytes", bytes, options.max_working_bytes)?;
        enforce_limit("local matrix elements", elements, max_local_matrix_elements)?;
    }

    for tree in inputs {
        for node in &node_order {
            let node_index = tree
                .node_index(node)
                .ok_or(TreeAciError::InternalInvariant {
                    message: "topology-compatible input is missing a node",
                })?;
            let tensor = tree
                .tensor(node_index)
                .ok_or(TreeAciError::InternalInvariant {
                    message: "topology-compatible input is missing a tensor",
                })?;
            let elements = checked_product(tensor.indices().iter().map(IndexLike::dim), "core")?;
            enforce_limit("core elements", elements, max_core_elements)?;
        }
    }

    let requested_start = requested_root
        .as_ref()
        .and_then(|root| node_positions.get(root).copied());
    let schedule = SweepPlan::for_strategy(
        options.traversal_strategy,
        node_order.len(),
        &edges,
        requested_start,
    )
    .map_err(|_| TreeAciError::InternalInvariant {
        message: "validated topology could not be lowered to a sweep plan",
    })?;
    let root = node_order[schedule.start].clone();
    let directed_edges = build_directed_edges(&node_order, &edges);
    let directed_dependency_order = dependency_order(&directed_edges)?;
    Ok(PreparedTreeProblem {
        node_order,
        node_positions,
        directed_edges,
        directed_dependency_order,
        physical,
        root,
        schedule,
        max_sample_arena_bytes: options.max_sample_arena_bytes,
        max_frame_elements: options.resolved_max_frame_elements::<T>(),
        max_frame_bytes: options.max_frame_bytes,
        max_core_elements,
        max_local_matrix_elements,
        max_working_bytes: options.max_working_bytes,
    })
}

/// Returns a dependency-first order for directed component construction.
///
/// A directed cut depends on the other cuts entering its source node. The
/// validated tree makes this dependency graph acyclic, so Kahn's algorithm
/// visits every dependency before its consumer without recursive traversal.
pub(crate) fn dependency_order<V>(
    directed_edges: &[DirectedEdge<V>],
) -> Result<Vec<DirectedEdgeId>> {
    let edge_count = directed_edges.len();
    let mut remaining = directed_edges
        .iter()
        .map(|edge| edge.incoming_to_from.len())
        .collect::<Vec<_>>();
    let mut dependents = vec![Vec::<DirectedEdgeId>::new(); edge_count];

    for (edge, directed) in directed_edges.iter().enumerate() {
        for &dependency in &directed.incoming_to_from {
            if dependency >= edge_count {
                return Err(TreeAciError::InternalInvariant {
                    message: "directed component dependency references an unknown edge",
                });
            }
            dependents[dependency].push(edge);
        }
    }

    let mut ready = VecDeque::new();
    for (edge, &count) in remaining.iter().enumerate() {
        if count == 0 {
            ready.push_back(edge);
        }
    }

    let mut order = Vec::with_capacity(edge_count);
    while let Some(edge) = ready.pop_front() {
        order.push(edge);
        for &dependent in &dependents[edge] {
            let count = remaining
                .get_mut(dependent)
                .ok_or(TreeAciError::InternalInvariant {
                    message: "directed component dependency has no indegree entry",
                })?;
            *count = count
                .checked_sub(1)
                .ok_or(TreeAciError::InternalInvariant {
                    message: "directed component dependency indegree underflowed",
                })?;
            if *count == 0 {
                ready.push_back(dependent);
            }
        }
    }

    if order.len() != edge_count {
        return Err(TreeAciError::InternalInvariant {
            message: "directed component dependency graph contains a cycle",
        });
    }
    Ok(order)
}

fn validate_options<V: TreeAciNode>(options: &TreeAciOptions<V>) -> Result<()> {
    if options.min_sweeps == 0 || options.max_sweeps < options.min_sweeps {
        return Err(TreeAciError::InvalidOption {
            option: "min_sweeps/max_sweeps",
            message: "sweep bounds must be nonzero and ordered",
        });
    }
    if !options.tolerance.is_finite() || options.tolerance <= 0.0 {
        return Err(TreeAciError::InvalidOption {
            option: "tolerance",
            message: "tolerance must be finite and positive",
        });
    }
    if !options.global_tolerance_margin.is_finite() || options.global_tolerance_margin < 0.0 {
        return Err(TreeAciError::InvalidOption {
            option: "global_tolerance_margin",
            message: "the margin must be finite and nonnegative",
        });
    }
    if options.max_bond_dim == Some(0) {
        return Err(TreeAciError::InvalidOption {
            option: "max_bond_dim",
            message: "a configured bond limit must be positive",
        });
    }
    for (name, limit) in [
        ("max_candidate_rows", Some(options.max_candidate_rows)),
        ("max_candidate_cols", Some(options.max_candidate_cols)),
        (
            "max_local_matrix_elements",
            options.max_local_matrix_elements,
        ),
        ("max_core_elements", options.max_core_elements),
        ("max_frame_elements", options.max_frame_elements),
        ("max_frame_bytes", Some(options.max_frame_bytes)),
        (
            "max_sample_arena_bytes",
            Some(options.max_sample_arena_bytes),
        ),
        ("max_working_bytes", Some(options.max_working_bytes)),
    ] {
        // An unset element ceiling is derived from `max_working_bytes`, which
        // is checked in this same loop, so only an explicit zero is rejected.
        if limit == Some(0) {
            return Err(TreeAciError::InvalidOption {
                option: name,
                message: "resource limits must be positive",
            });
        }
    }
    Ok(())
}

/// Rejects a request that exceeds its configured ceiling.
///
/// Shared so every allocation site reports the same `ResourceLimit` shape, and
/// so a new site cannot quietly grow its own ad hoc check.
pub(crate) fn enforce_limit(resource: &'static str, requested: usize, limit: usize) -> Result<()> {
    if requested > limit {
        return Err(TreeAciError::ResourceLimit {
            resource,
            requested,
            limit,
        });
    }
    Ok(())
}

fn checked_product(
    values: impl IntoIterator<Item = usize>,
    context: &'static str,
) -> Result<usize> {
    values.into_iter().try_fold(1usize, |product, value| {
        product
            .checked_mul(value)
            .ok_or(TreeAciError::SizeOverflow { context })
    })
}

fn build_directed_edges<V: TreeAciNode>(
    nodes: &[V],
    edges: &[(usize, usize)],
) -> Vec<DirectedEdge<V>> {
    let mut arcs = Vec::with_capacity(2 * edges.len());
    let mut endpoints = Vec::with_capacity(2 * edges.len());
    let mut incoming_by_target = vec![Vec::new(); nodes.len()];
    for (edge, &(left, right)) in edges.iter().enumerate() {
        let forward = 2 * edge;
        let reverse = forward + 1;
        endpoints.push((left, right));
        endpoints.push((right, left));
        incoming_by_target[right].push(forward);
        incoming_by_target[left].push(reverse);
        arcs.push(DirectedEdge {
            id: forward,
            from: nodes[left].clone(),
            to: nodes[right].clone(),
            reverse,
            incoming_to_from: Vec::new(),
        });
        arcs.push(DirectedEdge {
            id: reverse,
            from: nodes[right].clone(),
            to: nodes[left].clone(),
            reverse: forward,
            incoming_to_from: Vec::new(),
        });
    }
    for (id, arc) in arcs.iter_mut().enumerate() {
        let (from, to) = endpoints[id];
        let mut incoming = incoming_by_target[from]
            .iter()
            .copied()
            .filter(|incoming| endpoints[*incoming].0 != to)
            .collect::<Vec<_>>();
        incoming.sort_unstable();
        arc.incoming_to_from = incoming;
    }
    arcs
}

#[cfg(test)]
mod tests;
