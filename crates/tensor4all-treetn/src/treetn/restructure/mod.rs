//! Public scaffolding for multi-phase TreeTN restructuring.
//!
//! The approved B2a design is plan-first:
//! 1. Build a pure restructure plan from the current and target site-index networks.
//! 2. Execute that plan through split, move, and fuse phases.
//!
//! The initial implementation currently supports fuse-only, split-only,
//! swap-only, a conservative path-based swap-then-fuse mixed path, and a
//! conservative split-then-fuse mixed path.
//!
//! Unsupported patterns are reported explicitly. In particular, mixed cases
//! that require both splitting a node into multiple cross-node fragments and a
//! subsequent swap/move phase may still remain staged behind placeholder
//! errors.

use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use crate::named_graph::NamedGraph;
use crate::node_name_network::NodeNameNetwork;
use anyhow::{bail, Context, Result};
use petgraph::stable_graph::NodeIndex;
use tensor4all_core::{IndexLike, TensorLike};

use super::TreeTN;
use crate::{RestructureOptions, SiteIndexNetwork};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct FragmentNode<CurrentV, TargetV> {
    current: CurrentV,
    split_rank: usize,
    target: TargetV,
}

type SplitThenFuseTarget<CurrentV, TargetV, I> =
    SiteIndexNetwork<FragmentNode<CurrentV, TargetV>, I>;

#[derive(Debug, Clone)]
enum RestructurePlanKind<CurrentV, TargetV, I>
where
    CurrentV: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    TargetV: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    I: IndexLike,
    I::Id: Eq + Hash,
{
    FuseOnly,
    SplitOnly,
    SwapOnly {
        target_assignment: HashMap<I::Id, CurrentV>,
    },
    SwapThenFuse {
        target_assignment: HashMap<I::Id, CurrentV>,
    },
    SplitThenFuse {
        split_target: Box<SplitThenFuseTarget<CurrentV, TargetV, I>>,
    },
}

#[derive(Debug, Clone)]
struct RestructurePlan<CurrentV, TargetV, I>
where
    CurrentV: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    TargetV: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    I: IndexLike,
    I::Id: Eq + Hash,
{
    kind: RestructurePlanKind<CurrentV, TargetV, I>,
}

fn collect_site_targets<T, TargetV>(
    target: &SiteIndexNetwork<TargetV, T::Index>,
) -> Result<HashMap<<T::Index as IndexLike>::Id, TargetV>>
where
    T: TensorLike,
    TargetV: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
{
    let mut site_to_target = HashMap::new();
    for target_node_name in target.node_names() {
        let site_space = target.site_space(target_node_name).ok_or_else(|| {
            anyhow::anyhow!(
                "restructure_to: target node {:?} has no registered site space",
                target_node_name
            )
        })?;
        for site_idx in site_space {
            let existing = site_to_target.insert(site_idx.id().clone(), target_node_name.clone());
            if let Some(previous_target) = existing {
                bail!(
                    "restructure_to: site index {:?} appears in both target nodes {:?} and {:?}",
                    site_idx.id(),
                    previous_target,
                    target_node_name
                );
            }
        }
    }
    Ok(site_to_target)
}

fn collect_current_site_ids<T, CurrentV>(
    current: &SiteIndexNetwork<CurrentV, T::Index>,
) -> Result<HashSet<<T::Index as IndexLike>::Id>>
where
    T: TensorLike,
    CurrentV: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
{
    let mut site_ids = HashSet::new();
    for current_node_name in current.node_names() {
        let site_space = current.site_space(current_node_name).ok_or_else(|| {
            anyhow::anyhow!(
                "restructure_to: current node {:?} has no registered site space",
                current_node_name
            )
        })?;
        for site_idx in site_space {
            site_ids.insert(site_idx.id().clone());
        }
    }
    Ok(site_ids)
}

fn current_nodes_map_uniquely_to_targets<T, CurrentV, TargetV>(
    current: &SiteIndexNetwork<CurrentV, T::Index>,
    site_to_target: &HashMap<<T::Index as IndexLike>::Id, TargetV>,
) -> Result<bool>
where
    T: TensorLike,
    CurrentV: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    TargetV: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
{
    for current_node_name in current.node_names() {
        let site_space = current.site_space(current_node_name).ok_or_else(|| {
            anyhow::anyhow!(
                "restructure_to: current node {:?} has no registered site space",
                current_node_name
            )
        })?;
        let target_names: HashSet<_> = site_space
            .iter()
            .map(|site_idx| {
                site_to_target
                    .get(site_idx.id())
                    .cloned()
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "restructure_to: site index {:?} is present in the current network but missing from the target",
                            site_idx.id()
                        )
                    })
            })
            .collect::<Result<_>>()?;
        if target_names.len() > 1 {
            return Ok(false);
        }
    }
    Ok(true)
}

fn collect_site_currents<T, CurrentV>(
    current: &SiteIndexNetwork<CurrentV, T::Index>,
) -> Result<HashMap<<T::Index as IndexLike>::Id, CurrentV>>
where
    T: TensorLike,
    CurrentV: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
{
    let mut site_to_current = HashMap::new();
    for current_node_name in current.node_names() {
        let site_space = current.site_space(current_node_name).ok_or_else(|| {
            anyhow::anyhow!(
                "restructure_to: current node {:?} has no registered site space",
                current_node_name
            )
        })?;
        for site_idx in site_space {
            let existing = site_to_current.insert(site_idx.id().clone(), current_node_name.clone());
            if let Some(previous_current) = existing {
                bail!(
                    "restructure_to: site index {:?} appears in both current nodes {:?} and {:?}",
                    site_idx.id(),
                    previous_current,
                    current_node_name
                );
            }
        }
    }
    Ok(site_to_current)
}

fn target_nodes_map_uniquely_to_currents<T, CurrentV, TargetV>(
    target: &SiteIndexNetwork<TargetV, T::Index>,
    site_to_current: &HashMap<<T::Index as IndexLike>::Id, CurrentV>,
) -> Result<bool>
where
    T: TensorLike,
    CurrentV: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    TargetV: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
{
    for target_node_name in target.node_names() {
        let site_space = target.site_space(target_node_name).ok_or_else(|| {
            anyhow::anyhow!(
                "restructure_to: target node {:?} has no registered site space",
                target_node_name
            )
        })?;
        let current_names: HashSet<_> = site_space
            .iter()
            .map(|site_idx| {
                site_to_current
                    .get(site_idx.id())
                    .cloned()
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "restructure_to: site index {:?} is present in the target but missing from the current network",
                            site_idx.id()
                        )
                    })
            })
            .collect::<Result<_>>()?;
        if current_names.len() > 1 {
            return Ok(false);
        }
    }
    Ok(true)
}

fn target_nodes_span_connected_currents<T, CurrentV, TargetV>(
    current: &SiteIndexNetwork<CurrentV, T::Index>,
    target: &SiteIndexNetwork<TargetV, T::Index>,
    site_to_current: &HashMap<<T::Index as IndexLike>::Id, CurrentV>,
    full_graph: &NamedGraph<CurrentV, T, T::Index>,
) -> Result<bool>
where
    T: TensorLike,
    CurrentV: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    TargetV: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
{
    for target_node_name in target.node_names() {
        let site_space = target.site_space(target_node_name).ok_or_else(|| {
            anyhow::anyhow!(
                "restructure_to: target node {:?} has no registered site space",
                target_node_name
            )
        })?;
        let current_names: HashSet<CurrentV> = site_space
            .iter()
            .map(|site_idx| {
                site_to_current.get(site_idx.id()).cloned().ok_or_else(|| {
                    anyhow::anyhow!(
                        "restructure_to: site index {:?} is present in the target but missing from the current network",
                        site_idx.id()
                    )
                })
            })
            .collect::<Result<_>>()?;
        // Check connectivity through the full graph. Nodes that are connected
        // through internal nodes (nodes with empty site space) are considered
        // connected — those internals will be pulled in during contraction.
        let full_node_indices: HashSet<NodeIndex> = current_names
            .iter()
            .filter_map(|n| full_graph.node_index(n))
            .collect();
        if full_node_indices.is_empty() {
            return Ok(false);
        }
        // Compute Steiner tree spanning these indices in the full graph.
        let terms: Vec<NodeIndex> = full_node_indices.iter().copied().collect();
        let root = terms[0];
        let g = full_graph.graph();
        let mut steiner = full_node_indices.clone();
        for &term in &terms[1..] {
            if let Some((_, path)) =
                petgraph::algo::astar(g, root, |n| n == term, |_| 1usize, |_| 0usize)
            {
                steiner.extend(path);
            }
        }
        // Any extra node beyond the original set must be internal
        // (empty site space in the SiteIndexNetwork).
        for &idx in &steiner {
            if !full_node_indices.contains(&idx) {
                if let Some(name) = full_graph.node_name(idx) {
                    if current.site_space(name).is_some_and(|s| !s.is_empty()) {
                        return Ok(false);
                    }
                }
            }
        }
    }

    Ok(true)
}

fn collect_shared_targets<T, CurrentV, TargetV>(
    target: &SiteIndexNetwork<TargetV, T::Index>,
    site_to_current: &HashMap<<T::Index as IndexLike>::Id, CurrentV>,
) -> Result<HashSet<TargetV>>
where
    T: TensorLike,
    CurrentV: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    TargetV: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
{
    let mut shared_targets = HashSet::new();
    for target_node_name in target.node_names() {
        let site_space = target.site_space(target_node_name).ok_or_else(|| {
            anyhow::anyhow!(
                "restructure_to: target node {:?} has no registered site space",
                target_node_name
            )
        })?;
        let current_names: HashSet<_> = site_space
            .iter()
            .map(|site_idx| {
                site_to_current
                    .get(site_idx.id())
                    .cloned()
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "restructure_to: site index {:?} is present in the target but missing from the current network",
                            site_idx.id()
                        )
                    })
            })
            .collect::<Result<_>>()?;
        if current_names.len() > 1 {
            shared_targets.insert(target_node_name.clone());
        }
    }
    Ok(shared_targets)
}

fn build_split_then_fuse_target<T, CurrentV, TargetV>(
    current: &SiteIndexNetwork<CurrentV, T::Index>,
    target: &SiteIndexNetwork<TargetV, T::Index>,
    site_to_target: &HashMap<<T::Index as IndexLike>::Id, TargetV>,
    site_to_current: &HashMap<<T::Index as IndexLike>::Id, CurrentV>,
    full_graph: &NamedGraph<CurrentV, T, T::Index>,
) -> Result<Option<SplitThenFuseTarget<CurrentV, TargetV, T::Index>>>
where
    T: TensorLike,
    CurrentV: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    TargetV: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
{
    if !target_nodes_span_connected_currents::<T, CurrentV, TargetV>(
        current,
        target,
        site_to_current,
        full_graph,
    )? {
        return Ok(None);
    }

    let shared_targets = collect_shared_targets::<T, CurrentV, TargetV>(target, site_to_current)?;
    let mut split_target = SiteIndexNetwork::with_capacity(current.node_count(), 0);
    let mut current_node_names: Vec<_> = current.node_names().into_iter().cloned().collect();
    current_node_names.sort();

    for current_node_name in current_node_names {
        let site_space = current.site_space(&current_node_name).ok_or_else(|| {
            anyhow::anyhow!(
                "restructure_to: current node {:?} has no registered site space",
                current_node_name
            )
        })?;
        let mut fragments: HashMap<TargetV, HashSet<T::Index>> = HashMap::new();
        for site_idx in site_space {
            let target_node_name = site_to_target.get(site_idx.id()).cloned().ok_or_else(|| {
                anyhow::anyhow!(
                    "restructure_to: site index {:?} is present in the current network but missing from the target",
                    site_idx.id()
                )
            })?;
            fragments
                .entry(target_node_name)
                .or_default()
                .insert(site_idx.clone());
        }

        let shared_targets_here: Vec<_> = fragments
            .keys()
            .filter(|target_name| shared_targets.contains(*target_name))
            .cloned()
            .collect();
        if shared_targets_here.len() > 1 {
            return Ok(None);
        }
        let boundary_target = shared_targets_here.first().cloned();

        let mut fragments: Vec<_> = fragments.into_iter().collect();
        fragments.sort_by(|(left_name, _), (right_name, _)| {
            let left_is_boundary = boundary_target.as_ref() == Some(left_name);
            let right_is_boundary = boundary_target.as_ref() == Some(right_name);
            left_is_boundary
                .cmp(&right_is_boundary)
                .then_with(|| left_name.cmp(right_name))
        });

        for (split_rank, (target_node_name, fragment_site_space)) in
            fragments.into_iter().enumerate()
        {
            split_target
                .add_node(
                    FragmentNode {
                        current: current_node_name.clone(),
                        split_rank,
                        target: target_node_name,
                    },
                    fragment_site_space,
                )
                .map_err(anyhow::Error::msg)?;
        }
    }

    Ok(Some(split_target))
}

fn ordered_path_nodes<V>(topology: &NodeNameNetwork<V>) -> Option<Vec<V>>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    let graph = topology.graph();
    match topology.node_count() {
        0 => return Some(Vec::new()),
        1 => {
            let node = graph.node_indices().next()?;
            return Some(vec![topology.node_name(node)?.clone()]);
        }
        _ => {}
    }

    let mut leaves = Vec::new();
    for node in graph.node_indices() {
        let degree = graph.neighbors(node).count();
        if degree > 2 {
            return None;
        }
        if degree == 1 {
            leaves.push(node);
        }
    }
    if leaves.len() != 2 {
        return None;
    }

    leaves.sort_by_key(|node| topology.node_name(*node).cloned());
    let mut ordered = Vec::with_capacity(topology.node_count());
    let mut previous = None;
    let mut current = *leaves.first()?;

    loop {
        ordered.push(topology.node_name(current)?.clone());
        let next = graph
            .neighbors(current)
            .find(|neighbor| Some(*neighbor) != previous);
        let Some(next) = next else {
            break;
        };
        previous = Some(current);
        current = next;
    }

    if ordered.len() == topology.node_count() {
        Some(ordered)
    } else {
        None
    }
}

fn build_path_swap_then_fuse_assignment<T, CurrentV, TargetV>(
    current: &SiteIndexNetwork<CurrentV, T::Index>,
    target: &SiteIndexNetwork<TargetV, T::Index>,
    site_to_target: &HashMap<<T::Index as IndexLike>::Id, TargetV>,
) -> Result<Option<HashMap<<T::Index as IndexLike>::Id, CurrentV>>>
where
    T: TensorLike,
    CurrentV: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    TargetV: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
{
    let current_path = match ordered_path_nodes(current.topology()) {
        Some(path) => path,
        None => return Ok(None),
    };
    let target_path = match ordered_path_nodes(target.topology()) {
        Some(path) => path,
        None => return Ok(None),
    };

    let mut contributor_counts: HashMap<TargetV, usize> = HashMap::new();
    for current_node_name in &current_path {
        let site_space = current.site_space(current_node_name).ok_or_else(|| {
            anyhow::anyhow!(
                "restructure_to: current node {:?} has no registered site space",
                current_node_name
            )
        })?;
        let mut target_names: Vec<_> = site_space
            .iter()
            .map(|site_idx| {
                site_to_target
                    .get(site_idx.id())
                    .cloned()
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "restructure_to: site index {:?} is present in the current network but missing from the target",
                            site_idx.id()
                        )
                    })
            })
            .collect::<Result<_>>()?;
        target_names.sort();
        target_names.dedup();
        if target_names.len() != 1 {
            return Ok(None);
        }
        let target_name = target_names.into_iter().next().ok_or_else(|| {
            anyhow::anyhow!(
                "restructure_to: current node {:?} has no target mapping",
                current_node_name
            )
        })?;
        *contributor_counts.entry(target_name).or_default() += 1;
    }

    let total_contributors: usize = contributor_counts.values().sum();
    if total_contributors != current_path.len() {
        return Ok(None);
    }

    let mut contiguous_blocks: HashMap<TargetV, Vec<CurrentV>> = HashMap::new();
    let mut cursor = 0usize;
    for target_node_name in &target_path {
        let block_len = *contributor_counts.get(target_node_name).unwrap_or(&0);
        if block_len == 0 || cursor + block_len > current_path.len() {
            return Ok(None);
        }
        contiguous_blocks.insert(
            target_node_name.clone(),
            current_path[cursor..cursor + block_len].to_vec(),
        );
        cursor += block_len;
    }
    if cursor != current_path.len() {
        return Ok(None);
    }

    let mut target_assignment = HashMap::new();
    for target_node_name in &target_path {
        let block = contiguous_blocks.get(target_node_name).ok_or_else(|| {
            anyhow::anyhow!(
                "restructure_to: missing contiguous block for target {:?}",
                target_node_name
            )
        })?;
        let mut site_ids: Vec<_> = target
            .site_space(target_node_name)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "restructure_to: target node {:?} has no registered site space",
                    target_node_name
                )
            })?
            .iter()
            .map(|site_idx| site_idx.id().clone())
            .collect();
        site_ids.sort();
        if site_ids.len() < block.len() {
            return Ok(None);
        }

        for (position, site_id) in site_ids.into_iter().enumerate() {
            let block_index = position.min(block.len() - 1);
            target_assignment.insert(site_id, block[block_index].clone());
        }
    }

    Ok(Some(target_assignment))
}

fn tree_children<V>(
    topology: &NodeNameNetwork<V>,
    node: NodeIndex,
    parent: Option<NodeIndex>,
) -> Vec<NodeIndex>
where
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    topology
        .graph()
        .neighbors(node)
        .filter(|neighbor| Some(*neighbor) != parent)
        .collect()
}

fn rooted_signature<V>(
    topology: &NodeNameNetwork<V>,
    node: NodeIndex,
    parent: Option<NodeIndex>,
    cache: &mut HashMap<(usize, Option<usize>), String>,
) -> String
where
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    let key = (node.index(), parent.map(|p| p.index()));
    if let Some(signature) = cache.get(&key) {
        return signature.clone();
    }

    let mut child_signatures: Vec<String> = tree_children(topology, node, parent)
        .into_iter()
        .map(|child| rooted_signature(topology, child, Some(node), cache))
        .collect();
    child_signatures.sort();

    let signature = format!("({})", child_signatures.concat());
    cache.insert(key, signature.clone());
    signature
}

#[derive(Default)]
struct IsomorphicMatchState {
    current_cache: HashMap<(usize, Option<usize>), String>,
    target_cache: HashMap<(usize, Option<usize>), String>,
    mapping: HashMap<NodeIndex, NodeIndex>,
}

fn match_isomorphic_subtrees<CurrentV, TargetV>(
    current_topology: &NodeNameNetwork<CurrentV>,
    target_topology: &NodeNameNetwork<TargetV>,
    current_node: NodeIndex,
    current_parent: Option<NodeIndex>,
    target_node: NodeIndex,
    target_parent: Option<NodeIndex>,
    state: &mut IsomorphicMatchState,
) -> bool
where
    CurrentV: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
    TargetV: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    if let Some(existing) = state.mapping.insert(target_node, current_node) {
        if existing != current_node {
            return false;
        }
    }

    let current_children = tree_children(current_topology, current_node, current_parent);
    let target_children = tree_children(target_topology, target_node, target_parent);
    if current_children.len() != target_children.len() {
        return false;
    }

    let mut current_groups: HashMap<String, Vec<NodeIndex>> = HashMap::new();
    for child in current_children {
        let signature = rooted_signature(
            current_topology,
            child,
            Some(current_node),
            &mut state.current_cache,
        );
        current_groups.entry(signature).or_default().push(child);
    }

    let mut target_groups: HashMap<String, Vec<NodeIndex>> = HashMap::new();
    for child in target_children {
        let signature = rooted_signature(
            target_topology,
            child,
            Some(target_node),
            &mut state.target_cache,
        );
        target_groups.entry(signature).or_default().push(child);
    }

    if current_groups.len() != target_groups.len() {
        return false;
    }

    let mut signatures: Vec<String> = current_groups.keys().cloned().collect();
    signatures.sort();
    for signature in signatures {
        let mut current_bucket = match current_groups.remove(&signature) {
            Some(bucket) => bucket,
            None => return false,
        };
        let mut target_bucket = match target_groups.remove(&signature) {
            Some(bucket) => bucket,
            None => return false,
        };
        if current_bucket.len() != target_bucket.len() {
            return false;
        }

        current_bucket.sort_by_key(|node| node.index());
        target_bucket.sort_by_key(|node| node.index());

        for (current_child, target_child) in current_bucket.into_iter().zip(target_bucket) {
            if !match_isomorphic_subtrees(
                current_topology,
                target_topology,
                current_child,
                Some(current_node),
                target_child,
                Some(target_node),
                state,
            ) {
                return false;
            }
        }
    }

    true
}

fn match_tree_topologies<CurrentV, TargetV>(
    current_topology: &NodeNameNetwork<CurrentV>,
    target_topology: &NodeNameNetwork<TargetV>,
) -> Option<HashMap<NodeIndex, NodeIndex>>
where
    CurrentV: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
    TargetV: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    if current_topology.node_count() != target_topology.node_count() {
        return None;
    }
    if current_topology.edge_count() != target_topology.edge_count() {
        return None;
    }

    let mut current_roots: Vec<(String, NodeIndex)> = current_topology
        .graph()
        .node_indices()
        .map(|node| {
            (
                rooted_signature(current_topology, node, None, &mut HashMap::new()),
                node,
            )
        })
        .collect();
    current_roots.sort_by_key(|(signature, node)| (signature.clone(), node.index()));

    let mut target_roots: Vec<(String, NodeIndex)> = target_topology
        .graph()
        .node_indices()
        .map(|node| {
            (
                rooted_signature(target_topology, node, None, &mut HashMap::new()),
                node,
            )
        })
        .collect();
    target_roots.sort_by_key(|(signature, node)| (signature.clone(), node.index()));

    for (target_signature, target_root) in &target_roots {
        for (current_signature, current_root) in &current_roots {
            if current_signature != target_signature {
                continue;
            }

            let mut state = IsomorphicMatchState::default();
            if match_isomorphic_subtrees(
                current_topology,
                target_topology,
                *current_root,
                None,
                *target_root,
                None,
                &mut state,
            ) && state.mapping.len() == target_topology.node_count()
            {
                return Some(state.mapping);
            }
        }
    }

    None
}

fn build_swap_assignment<T, CurrentV, TargetV>(
    current: &SiteIndexNetwork<CurrentV, T::Index>,
    target: &SiteIndexNetwork<TargetV, T::Index>,
) -> Result<Option<HashMap<<T::Index as IndexLike>::Id, CurrentV>>>
where
    T: TensorLike,
    CurrentV: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    TargetV: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
{
    let topology_mapping = match match_tree_topologies(current.topology(), target.topology()) {
        Some(mapping) => mapping,
        None => return Ok(None),
    };

    let mut assignment = HashMap::new();
    for target_node in target.graph().node_indices() {
        let target_name = target.node_name(target_node).ok_or_else(|| {
            anyhow::anyhow!(
                "restructure_to: target topology mapping referenced a missing node index {:?}",
                target_node
            )
        })?;
        let current_node = *topology_mapping.get(&target_node).ok_or_else(|| {
            anyhow::anyhow!(
                "restructure_to: target topology mapping did not assign a current node to {:?}",
                target_name
            )
        })?;
        let current_name = current.node_name(current_node).ok_or_else(|| {
            anyhow::anyhow!(
                "restructure_to: current topology mapping referenced a missing node index {:?}",
                current_node
            )
        })?;
        let site_space = target.site_space(target_name).ok_or_else(|| {
            anyhow::anyhow!(
                "restructure_to: target node {:?} has no registered site space",
                target_name
            )
        })?;
        for site_idx in site_space {
            assignment.insert(site_idx.id().clone(), current_name.clone());
        }
    }

    Ok(Some(assignment))
}

fn clone_tree<T, V>(tree: &TreeTN<T, V>) -> Result<TreeTN<T, V>>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    Ok(TreeTN {
        graph: tree.graph.clone(),
        canonical_region: tree.canonical_region.clone(),
        canonical_form: tree.canonical_form,
        site_index_network: tree.site_index_network.clone(),
        link_index_network: tree.link_index_network.clone(),
        ortho_towards: tree.ortho_towards.clone(),
    })
}

fn apply_final_truncation<T, V>(
    tree: TreeTN<T, V>,
    options: &RestructureOptions,
) -> Result<TreeTN<T, V>>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
{
    let Some(final_truncation) = options.final_truncation else {
        return Ok(tree);
    };
    let center = tree
        .node_names()
        .into_iter()
        .min()
        .ok_or_else(|| anyhow::anyhow!("restructure_to: cannot truncate an empty network"))?;
    tree.truncate([center], final_truncation)
        .context("restructure_to: final truncation")
}

fn build_plan<T, CurrentV, TargetV>(
    current: &SiteIndexNetwork<CurrentV, T::Index>,
    target: &SiteIndexNetwork<TargetV, T::Index>,
    full_graph: &NamedGraph<CurrentV, T, T::Index>,
) -> Result<RestructurePlan<CurrentV, TargetV, T::Index>>
where
    T: TensorLike,
    CurrentV: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    TargetV: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
{
    let site_to_target = collect_site_targets::<T, TargetV>(target)?;
    let site_to_current = collect_site_currents::<T, CurrentV>(current)?;
    let current_site_ids = collect_current_site_ids::<T, CurrentV>(current)?;
    let target_site_ids: HashSet<_> = site_to_target.keys().cloned().collect();

    if current_site_ids != target_site_ids {
        bail!("restructure_to: current and target must contain the same site index ids");
    }

    if current_nodes_map_uniquely_to_targets::<T, CurrentV, TargetV>(current, &site_to_target)? {
        if target_nodes_span_connected_currents::<T, CurrentV, TargetV>(
            current,
            target,
            &site_to_current,
            full_graph,
        )? {
            return Ok(RestructurePlan {
                kind: RestructurePlanKind::FuseOnly,
            });
        }

        if let Some(target_assignment) = build_path_swap_then_fuse_assignment::<T, CurrentV, TargetV>(
            current,
            target,
            &site_to_target,
        )? {
            return Ok(RestructurePlan {
                kind: RestructurePlanKind::SwapThenFuse { target_assignment },
            });
        }
    }

    if target_nodes_map_uniquely_to_currents::<T, CurrentV, TargetV>(target, &site_to_current)? {
        return Ok(RestructurePlan {
            kind: RestructurePlanKind::SplitOnly,
        });
    }

    if let Some(target_assignment) = build_swap_assignment::<T, CurrentV, TargetV>(current, target)?
    {
        return Ok(RestructurePlan {
            kind: RestructurePlanKind::SwapOnly { target_assignment },
        });
    }

    if let Some(split_target) = build_split_then_fuse_target::<T, CurrentV, TargetV>(
        current,
        target,
        &site_to_target,
        &site_to_current,
        full_graph,
    )? {
        return Ok(RestructurePlan {
            kind: RestructurePlanKind::SplitThenFuse {
                split_target: Box::new(split_target),
            },
        });
    }

    bail!(
        "restructure_to: planner placeholder only; split/move/mixed restructure planning is not implemented yet"
    )
}

fn execute_plan<T, CurrentV, TargetV>(
    tree: &TreeTN<T, CurrentV>,
    plan: RestructurePlan<CurrentV, TargetV, T::Index>,
    target: &SiteIndexNetwork<TargetV, T::Index>,
    options: &RestructureOptions,
) -> Result<TreeTN<T, TargetV>>
where
    T: TensorLike,
    CurrentV: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    TargetV: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
{
    let result = match plan.kind {
        RestructurePlanKind::FuseOnly => tree.fuse_to(target),
        RestructurePlanKind::SplitOnly => tree.split_to(target, &options.split),
        RestructurePlanKind::SwapOnly { target_assignment } => {
            let mut working = clone_tree(tree)?;
            working
                .swap_site_indices(&target_assignment, &options.swap)
                .context("restructure_to: swap phase")?;
            Ok(working
                .fuse_to(target)
                .context("restructure_to: finalize after swap")?)
        }
        RestructurePlanKind::SwapThenFuse { target_assignment } => {
            let mut working = clone_tree(tree)?;
            working
                .swap_site_indices(&target_assignment, &options.swap)
                .context("restructure_to: swap phase")?;
            Ok(working
                .fuse_to(target)
                .context("restructure_to: finalize after swap")?)
        }
        RestructurePlanKind::SplitThenFuse { split_target } => {
            let split = tree
                .split_to(split_target.as_ref(), &options.split)
                .context("restructure_to: split phase")?;
            split.fuse_to(target).context("restructure_to: fuse phase")
        }
    }?;

    apply_final_truncation(result, options)
}

impl<T, V> TreeTN<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    /// Restructure this TreeTN to match a target site-index network.
    ///
    /// This is the plan-first public entry point for Issue #423 B2a.
    ///
    /// The current staged implementation already handles:
    /// - fuse-only restructures, where each current node maps to exactly one
    ///   target node;
    /// - split-only restructures, where each target node maps to exactly one
    ///   current node;
    /// - swap-only restructures, where the current and target topologies are
    ///   tree-isomorphic and only the site assignments differ;
    /// - conservative path-based swap-then-fuse restructures, where the
    ///   current nodes already map uniquely to target nodes but their target
    ///   groups must be rearranged into contiguous path blocks before fusing;
    /// - conservative mixed split-then-fuse restructures, where each current
    ///   node has at most one cross-node target fragment that must retain the
    ///   original external bonds.
    ///
    /// Unsupported patterns are reported explicitly. In particular, mixed
    /// cases that require both split planning and a subsequent move/swap phase
    /// may still remain intentionally staged behind placeholder errors while
    /// the pure planner is expanded.
    ///
    /// Related types:
    /// - [`RestructureOptions`] configures the split, transport, and optional
    ///   final truncation phases.
    /// - [`SiteIndexNetwork`] describes the desired final topology plus site
    ///   grouping.
    /// - [`TreeTN::split_to`](crate::treetn::TreeTN::split_to) and
    ///   [`TreeTN::swap_site_indices`](crate::treetn::TreeTN::swap_site_indices)
    ///   remain the lower-level building blocks that the executor will use.
    ///
    /// # Arguments
    /// * `target` - Desired final topology and site grouping.
    /// * `options` - Phase-specific options for split, transport, and optional
    ///   final truncation.
    ///
    /// # Returns
    /// A new `TreeTN` with the target node naming and target site-index
    /// network.
    ///
    /// # Errors
    /// Returns an error when the target is structurally incompatible with the
    /// current network, or when the requested restructure still needs the
    /// staged planner paths for mixed split/move/fuse execution.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashSet;
    ///
    /// use tensor4all_core::{DynIndex, TensorDynLen, TensorLike};
    /// use tensor4all_treetn::{RestructureOptions, SiteIndexNetwork, TreeTN};
    ///
    /// # fn main() -> anyhow::Result<()> {
    /// let left = DynIndex::new_dyn(2);
    /// let right = DynIndex::new_dyn(2);
    /// let bond = DynIndex::new_dyn(1);
    /// let t0 = TensorDynLen::from_dense(vec![left.clone(), bond.clone()], vec![1.0, 0.0])?;
    /// let t1 = TensorDynLen::from_dense(vec![bond, right.clone()], vec![1.0, 0.0])?;
    /// let treetn = TreeTN::<TensorDynLen, String>::from_tensors(
    ///     vec![t0, t1],
    ///     vec!["A".to_string(), "B".to_string()],
    /// )?;
    ///
    /// let mut target: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    /// assert!(target
    ///     .add_node("AB".to_string(), HashSet::from([left.clone(), right.clone()]))
    ///     .is_ok());
    ///
    /// let result = treetn.restructure_to(&target, &RestructureOptions::default())?;
    ///
    /// assert_eq!(result.node_count(), 1);
    /// let dense = result.contract_to_tensor()?;
    /// let expected = treetn.contract_to_tensor()?;
    /// assert!((&dense - &expected).maxabs() < 1e-12);
    /// # Ok::<(), anyhow::Error>(())
    /// # }
    /// ```
    pub fn restructure_to<TargetV>(
        &self,
        target: &SiteIndexNetwork<TargetV, T::Index>,
        options: &RestructureOptions,
    ) -> Result<TreeTN<T, TargetV>>
    where
        TargetV: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
        <T::Index as IndexLike>::Id:
            Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    {
        let plan = build_plan::<T, V, TargetV>(self.site_index_network(), target, &self.graph)
            .context("restructure_to: build plan")?;
        execute_plan(self, plan, target, options).context("restructure_to: execute plan")
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use tensor4all_core::{DynIndex, IndexLike, TensorDynLen};

    use super::*;

    type FourSiteChainCase = (
        TreeTN<TensorDynLen, String>,
        DynIndex,
        DynIndex,
        DynIndex,
        DynIndex,
    );

    fn two_node_chain() -> anyhow::Result<(TreeTN<TensorDynLen, String>, DynIndex, DynIndex)> {
        let left = DynIndex::new_dyn(2);
        let right = DynIndex::new_dyn(2);
        let bond = DynIndex::new_dyn(1);
        let t0 = TensorDynLen::from_dense(vec![left.clone(), bond.clone()], vec![1.0, 0.0])?;
        let t1 = TensorDynLen::from_dense(vec![bond, right.clone()], vec![1.0, 0.0])?;
        let treetn = TreeTN::<TensorDynLen, String>::from_tensors(
            vec![t0, t1],
            vec!["A".to_string(), "B".to_string()],
        )?;
        Ok((treetn, left, right))
    }

    fn two_node_groups_of_two() -> anyhow::Result<FourSiteChainCase> {
        let x0 = DynIndex::new_dyn(2);
        let x1 = DynIndex::new_dyn(2);
        let y0 = DynIndex::new_dyn(2);
        let y1 = DynIndex::new_dyn(2);
        let bond = DynIndex::new_dyn(2);
        let left_tensor = TensorDynLen::from_dense(
            vec![x0.clone(), x1.clone(), bond.clone()],
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        )?;
        let right_tensor = TensorDynLen::from_dense(
            vec![bond, y0.clone(), y1.clone()],
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        )?;
        let treetn = TreeTN::<TensorDynLen, String>::from_tensors(
            vec![left_tensor, right_tensor],
            vec!["Left".to_string(), "Right".to_string()],
        )?;
        Ok((treetn, x0, x1, y0, y1))
    }

    fn three_node_chain_for_swap() -> anyhow::Result<FourSiteChainCase> {
        let s0 = DynIndex::new_dyn(2);
        let s1 = DynIndex::new_dyn(2);
        let s2 = DynIndex::new_dyn(2);
        let s3 = DynIndex::new_dyn(2);
        let b01 = DynIndex::new_dyn(2);
        let b12 = DynIndex::new_dyn(2);
        let t0 = TensorDynLen::from_dense(
            vec![s0.clone(), s1.clone(), b01.clone()],
            vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0],
        )?;
        let t1 = TensorDynLen::from_dense(
            vec![b01.clone(), s2.clone(), b12.clone()],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )?;
        let t2 = TensorDynLen::from_dense(vec![b12, s3.clone()], vec![1.0, 2.0, 3.0, 4.0])?;
        let treetn = TreeTN::<TensorDynLen, String>::from_tensors(
            vec![t0, t1, t2],
            vec!["A".to_string(), "B".to_string(), "C".to_string()],
        )?;
        Ok((treetn, s0, s1, s2, s3))
    }

    fn four_node_interleaved_chain() -> anyhow::Result<FourSiteChainCase> {
        let x0 = DynIndex::new_dyn(2);
        let x1 = DynIndex::new_dyn(2);
        let y0 = DynIndex::new_dyn(2);
        let y1 = DynIndex::new_dyn(2);
        let b01 = DynIndex::new_dyn(2);
        let b12 = DynIndex::new_dyn(2);
        let b23 = DynIndex::new_dyn(2);
        let t0 = TensorDynLen::from_dense(vec![x0.clone(), b01.clone()], vec![1.0, 0.0, 0.0, 1.0])?;
        let t1 = TensorDynLen::from_dense(
            vec![b01.clone(), x1.clone(), b12.clone()],
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        )?;
        let t2 = TensorDynLen::from_dense(
            vec![b12.clone(), y0.clone(), b23.clone()],
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        )?;
        let t3 = TensorDynLen::from_dense(vec![b23, y1.clone()], vec![1.0, 0.0, 0.0, 1.0])?;
        let treetn = TreeTN::<TensorDynLen, String>::from_tensors(
            vec![t0, t1, t2, t3],
            vec![
                "0".to_string(),
                "1".to_string(),
                "2".to_string(),
                "3".to_string(),
            ],
        )?;
        Ok((treetn, x0, x1, y0, y1))
    }

    #[test]
    fn test_restructure_to_fuse_only_matches_target_structure() -> anyhow::Result<()> {
        let (treetn, left, right) = two_node_chain()?;

        let mut target: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
        target
            .add_node("AB".to_string(), HashSet::from([left, right]))
            .map_err(anyhow::Error::msg)?;

        let result = treetn.restructure_to(&target, &RestructureOptions::default())?;
        let dense_expected = treetn.contract_to_tensor()?;
        let dense_actual = result.contract_to_tensor()?;

        assert_eq!(result.node_count(), 1);
        assert_eq!(result.site_index_network().node_count(), 1);
        assert!((&dense_actual - &dense_expected).maxabs() < 1e-12);

        Ok(())
    }

    #[test]
    fn test_restructure_to_split_only_matches_target_structure() -> anyhow::Result<()> {
        let (treetn, left, right) = two_node_chain()?;

        let mut fused_target: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
        fused_target
            .add_node(
                "AB".to_string(),
                HashSet::from([left.clone(), right.clone()]),
            )
            .map_err(anyhow::Error::msg)?;
        let fused = treetn.restructure_to(&fused_target, &RestructureOptions::default())?;

        let mut split_target: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
        split_target
            .add_node("Left".to_string(), HashSet::from([left]))
            .map_err(anyhow::Error::msg)?;
        split_target
            .add_node("Right".to_string(), HashSet::from([right]))
            .map_err(anyhow::Error::msg)?;
        split_target
            .add_edge(&"Left".to_string(), &"Right".to_string())
            .map_err(anyhow::Error::msg)?;

        let result = fused.restructure_to(&split_target, &RestructureOptions::default())?;
        let dense_expected = fused.contract_to_tensor()?;
        let dense_actual = result.contract_to_tensor()?;

        assert_eq!(result.node_count(), 2);
        assert!((&dense_actual - &dense_expected).maxabs() < 1e-12);

        Ok(())
    }

    #[test]
    fn test_restructure_to_swap_only_matches_target_structure() -> anyhow::Result<()> {
        let (treetn, s0, s1, s2, s3) = three_node_chain_for_swap()?;

        let mut target: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
        target
            .add_node("X".to_string(), HashSet::from([s0.clone()]))
            .map_err(anyhow::Error::msg)?;
        target
            .add_node("Y".to_string(), HashSet::from([s1.clone(), s2.clone()]))
            .map_err(anyhow::Error::msg)?;
        target
            .add_node("Z".to_string(), HashSet::from([s3.clone()]))
            .map_err(anyhow::Error::msg)?;
        target
            .add_edge(&"X".to_string(), &"Y".to_string())
            .map_err(anyhow::Error::msg)?;
        target
            .add_edge(&"Y".to_string(), &"Z".to_string())
            .map_err(anyhow::Error::msg)?;

        let result = treetn.restructure_to(&target, &RestructureOptions::default())?;
        let dense_expected = treetn.contract_to_tensor()?;
        let dense_actual = result.contract_to_tensor()?;

        assert_eq!(
            result
                .site_index_network()
                .find_node_by_index_id(s0.id())
                .map(|name| name.as_str()),
            Some("X")
        );
        assert_eq!(
            result
                .site_index_network()
                .find_node_by_index_id(s1.id())
                .map(|name| name.as_str()),
            Some("Y")
        );
        assert_eq!(
            result
                .site_index_network()
                .find_node_by_index_id(s2.id())
                .map(|name| name.as_str()),
            Some("Y")
        );
        assert_eq!(
            result
                .site_index_network()
                .find_node_by_index_id(s3.id())
                .map(|name| name.as_str()),
            Some("Z")
        );
        assert!((&dense_actual - &dense_expected).maxabs() < 1e-10);

        Ok(())
    }

    #[test]
    fn test_restructure_to_split_then_fuse_mixed_case() -> anyhow::Result<()> {
        let (treetn, x0, x1, y0, y1) = two_node_groups_of_two()?;

        let mut target: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
        target
            .add_node("X".to_string(), HashSet::from([x0.clone()]))
            .map_err(anyhow::Error::msg)?;
        target
            .add_node("Y".to_string(), HashSet::from([x1.clone(), y0.clone()]))
            .map_err(anyhow::Error::msg)?;
        target
            .add_node("Z".to_string(), HashSet::from([y1.clone()]))
            .map_err(anyhow::Error::msg)?;
        target
            .add_edge(&"X".to_string(), &"Y".to_string())
            .map_err(anyhow::Error::msg)?;
        target
            .add_edge(&"Y".to_string(), &"Z".to_string())
            .map_err(anyhow::Error::msg)?;

        let result = treetn.restructure_to(&target, &RestructureOptions::default())?;
        let dense_expected = treetn.contract_to_tensor()?;
        let dense_actual = result.contract_to_tensor()?;

        assert_eq!(result.node_count(), 3);
        assert_eq!(result.edge_count(), 2);
        assert!(result
            .site_index_network()
            .share_equivalent_site_index_network(&target));
        assert_eq!(
            result
                .site_index_network()
                .find_node_by_index_id(x0.id())
                .map(|name| name.as_str()),
            Some("X")
        );
        assert_eq!(
            result
                .site_index_network()
                .find_node_by_index_id(x1.id())
                .map(|name| name.as_str()),
            Some("Y")
        );
        assert_eq!(
            result
                .site_index_network()
                .find_node_by_index_id(y0.id())
                .map(|name| name.as_str()),
            Some("Y")
        );
        assert_eq!(
            result
                .site_index_network()
                .find_node_by_index_id(y1.id())
                .map(|name| name.as_str()),
            Some("Z")
        );
        assert!((&dense_actual - &dense_expected).maxabs() < 1e-10);

        Ok(())
    }

    #[test]
    fn test_restructure_to_swap_then_fuse_mixed_case() -> anyhow::Result<()> {
        let (treetn, x0, x1, y0, y1) = four_node_interleaved_chain()?;

        let mut target: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
        target
            .add_node("X".to_string(), HashSet::from([x0.clone(), y0.clone()]))
            .map_err(anyhow::Error::msg)?;
        target
            .add_node("Y".to_string(), HashSet::from([x1.clone(), y1.clone()]))
            .map_err(anyhow::Error::msg)?;
        target
            .add_edge(&"X".to_string(), &"Y".to_string())
            .map_err(anyhow::Error::msg)?;

        let result = treetn.restructure_to(&target, &RestructureOptions::default())?;
        let dense_expected = treetn.contract_to_tensor()?;
        let dense_actual = result.contract_to_tensor()?;

        assert_eq!(result.node_count(), 2);
        assert_eq!(result.edge_count(), 1);
        assert!(result
            .site_index_network()
            .share_equivalent_site_index_network(&target));
        assert_eq!(
            result
                .site_index_network()
                .find_node_by_index_id(x0.id())
                .map(|name| name.as_str()),
            Some("X")
        );
        assert_eq!(
            result
                .site_index_network()
                .find_node_by_index_id(y0.id())
                .map(|name| name.as_str()),
            Some("X")
        );
        assert_eq!(
            result
                .site_index_network()
                .find_node_by_index_id(x1.id())
                .map(|name| name.as_str()),
            Some("Y")
        );
        assert_eq!(
            result
                .site_index_network()
                .find_node_by_index_id(y1.id())
                .map(|name| name.as_str()),
            Some("Y")
        );
        assert!((&dense_actual - &dense_expected).maxabs() < 1e-10);

        Ok(())
    }

    #[test]
    fn test_restructure_to_two_node_swap_only_cross_pairing() -> anyhow::Result<()> {
        let (treetn, x0, x1, y0, y1) = two_node_groups_of_two()?;

        let mut target: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
        target
            .add_node("X".to_string(), HashSet::from([x0.clone(), y0.clone()]))
            .map_err(anyhow::Error::msg)?;
        target
            .add_node("Y".to_string(), HashSet::from([x1.clone(), y1.clone()]))
            .map_err(anyhow::Error::msg)?;
        target
            .add_edge(&"X".to_string(), &"Y".to_string())
            .map_err(anyhow::Error::msg)?;

        let result = treetn.restructure_to(&target, &RestructureOptions::default())?;
        let dense_expected = treetn.contract_to_tensor()?;
        let dense_actual = result.contract_to_tensor()?;

        assert_eq!(result.node_count(), 2);
        assert_eq!(result.edge_count(), 1);
        assert!(result
            .site_index_network()
            .share_equivalent_site_index_network(&target));
        assert_eq!(
            result
                .site_index_network()
                .find_node_by_index_id(x0.id())
                .map(|name| name.as_str()),
            Some("X")
        );
        assert_eq!(
            result
                .site_index_network()
                .find_node_by_index_id(y0.id())
                .map(|name| name.as_str()),
            Some("X")
        );
        assert_eq!(
            result
                .site_index_network()
                .find_node_by_index_id(x1.id())
                .map(|name| name.as_str()),
            Some("Y")
        );
        assert_eq!(
            result
                .site_index_network()
                .find_node_by_index_id(y1.id())
                .map(|name| name.as_str()),
            Some("Y")
        );
        assert!((&dense_actual - &dense_expected).maxabs() < 1e-10);

        Ok(())
    }

    // ========================================================================
    // Y-shape branching topology tests for restructure_to
    // ========================================================================

    /// Create a Y-shape TreeTN with site indices on every node:
    ///     A (site_a)
    ///     |
    ///     B (site_b)
    ///    / \
    ///   C   D
    /// (site_c) (site_d)
    #[allow(clippy::type_complexity)]
    fn y_shape_treetn() -> anyhow::Result<(
        TreeTN<TensorDynLen, String>,
        DynIndex,
        DynIndex,
        DynIndex,
        DynIndex,
    )> {
        let mut tn = TreeTN::<TensorDynLen, String>::new();
        let site_a = DynIndex::new_dyn(2);
        let site_b = DynIndex::new_dyn(2);
        let site_c = DynIndex::new_dyn(2);
        let site_d = DynIndex::new_dyn(2);
        let bond_ab = DynIndex::new_dyn(3);
        let bond_bc = DynIndex::new_dyn(3);
        let bond_bd = DynIndex::new_dyn(3);
        let tensor_a =
            TensorDynLen::from_dense(vec![site_a.clone(), bond_ab.clone()], vec![1.0; 6])?;
        tn.add_tensor("A".to_string(), tensor_a).unwrap();
        let tensor_b = TensorDynLen::from_dense(
            vec![
                bond_ab.clone(),
                bond_bc.clone(),
                bond_bd.clone(),
                site_b.clone(),
            ],
            vec![1.0; 54],
        )?;
        tn.add_tensor("B".to_string(), tensor_b).unwrap();
        let tensor_c =
            TensorDynLen::from_dense(vec![bond_bc.clone(), site_c.clone()], vec![1.0; 6])?;
        tn.add_tensor("C".to_string(), tensor_c).unwrap();
        let tensor_d =
            TensorDynLen::from_dense(vec![bond_bd.clone(), site_d.clone()], vec![1.0; 6])?;
        tn.add_tensor("D".to_string(), tensor_d).unwrap();
        let n_a = tn.node_index(&"A".to_string()).unwrap();
        let n_b = tn.node_index(&"B".to_string()).unwrap();
        let n_c = tn.node_index(&"C".to_string()).unwrap();
        let n_d = tn.node_index(&"D".to_string()).unwrap();
        tn.connect(n_a, &bond_ab, n_b, &bond_ab).unwrap();
        tn.connect(n_b, &bond_bc, n_c, &bond_bc).unwrap();
        tn.connect(n_b, &bond_bd, n_d, &bond_bd).unwrap();
        Ok((tn, site_a, site_b, site_c, site_d))
    }

    #[test]
    fn test_restructure_to_y_shape_fuse_subtree() -> anyhow::Result<()> {
        let (tn, site_a, site_b, site_c, site_d) = y_shape_treetn()?;

        let mut target: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
        target
            .add_node("A".to_string(), HashSet::from([site_a]))
            .map_err(anyhow::Error::msg)?;
        target
            .add_node("BCD".to_string(), HashSet::from([site_b, site_c, site_d]))
            .map_err(anyhow::Error::msg)?;
        target
            .add_edge(&"A".to_string(), &"BCD".to_string())
            .map_err(anyhow::Error::msg)?;

        let result = tn.restructure_to(&target, &RestructureOptions::default())?;
        assert_eq!(result.node_count(), 2);
        assert_eq!(result.edge_count(), 1);
        assert!(
            result
                .site_index_network()
                .share_equivalent_site_index_network(&target),
            "Y-shape fused to chain must match target"
        );
        let dense_expected = tn.contract_to_tensor()?;
        let dense_actual = result.contract_to_tensor()?;
        assert!((&dense_actual - &dense_expected).maxabs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_restructure_to_y_shape_swap_only() -> anyhow::Result<()> {
        // Build a Y-shape with all bonds visible
        let mut tn = TreeTN::<TensorDynLen, String>::new();
        let site_a = DynIndex::new_dyn(2);
        let site_b = DynIndex::new_dyn(2);
        let site_c = DynIndex::new_dyn(2);
        let site_d = DynIndex::new_dyn(2);
        let bond_ab = DynIndex::new_dyn(3);
        let bond_bc = DynIndex::new_dyn(3);
        let bond_bd = DynIndex::new_dyn(3);

        let tensor_a =
            TensorDynLen::from_dense(vec![site_a.clone(), bond_ab.clone()], vec![1.0; 6])?;
        tn.add_tensor("A".to_string(), tensor_a)?;
        let tensor_b = TensorDynLen::from_dense(
            vec![
                bond_ab.clone(),
                bond_bc.clone(),
                bond_bd.clone(),
                site_b.clone(),
            ],
            vec![1.0; 54],
        )?;
        tn.add_tensor("B".to_string(), tensor_b)?;
        let tensor_c =
            TensorDynLen::from_dense(vec![bond_bc.clone(), site_c.clone()], vec![1.0; 6])?;
        tn.add_tensor("C".to_string(), tensor_c)?;
        let tensor_d =
            TensorDynLen::from_dense(vec![bond_bd.clone(), site_d.clone()], vec![1.0; 6])?;
        tn.add_tensor("D".to_string(), tensor_d)?;

        let n_a = tn.node_index(&"A".to_string()).unwrap();
        let n_b = tn.node_index(&"B".to_string()).unwrap();
        let n_c = tn.node_index(&"C".to_string()).unwrap();
        let n_d = tn.node_index(&"D".to_string()).unwrap();
        tn.connect(n_a, &bond_ab, n_b, &bond_ab)?;
        tn.connect(n_b, &bond_bc, n_c, &bond_bc)?;
        tn.connect(n_b, &bond_bd, n_d, &bond_bd)?;

        // SwapOnly: reassign site a→X, b→Y, c→Z, d→W, same Y topology
        let mut target: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
        target
            .add_node("X".to_string(), HashSet::from([site_a]))
            .map_err(anyhow::Error::msg)?;
        target
            .add_node("Y".to_string(), HashSet::from([site_b]))
            .map_err(anyhow::Error::msg)?;
        target
            .add_node("Z".to_string(), HashSet::from([site_c]))
            .map_err(anyhow::Error::msg)?;
        target
            .add_node("W".to_string(), HashSet::from([site_d]))
            .map_err(anyhow::Error::msg)?;
        target
            .add_edge(&"X".to_string(), &"Y".to_string())
            .map_err(anyhow::Error::msg)?;
        target
            .add_edge(&"Y".to_string(), &"Z".to_string())
            .map_err(anyhow::Error::msg)?;
        target
            .add_edge(&"Y".to_string(), &"W".to_string())
            .map_err(anyhow::Error::msg)?;

        let result = tn.restructure_to(&target, &RestructureOptions::default())?;
        assert_eq!(result.node_count(), 4);
        assert_eq!(result.edge_count(), 3);
        assert!(
            result
                .site_index_network()
                .share_equivalent_site_index_network(&target),
            "Y-shape swapped must match target"
        );
        let dense_expected = tn.contract_to_tensor()?;
        let dense_actual = result.contract_to_tensor()?;
        assert!((&dense_actual - &dense_expected).maxabs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_restructure_to_y_shape_internal_center() -> anyhow::Result<()> {
        // Y-shape with internal center (B has no site indices after connection):
        //     A(site_a)
        //       |
        //     B(no site)
        //      / \
        // C(site_c) D(site_d)
        let mut tn = TreeTN::<TensorDynLen, String>::new();
        let site_a = DynIndex::new_dyn(2);
        let site_c = DynIndex::new_dyn(2);
        let site_d = DynIndex::new_dyn(2);
        let bond_ab = DynIndex::new_dyn(3);
        let bond_bc = DynIndex::new_dyn(3);
        let bond_bd = DynIndex::new_dyn(3);

        let tensor_a =
            TensorDynLen::from_dense(vec![site_a.clone(), bond_ab.clone()], vec![1.0; 6])?;
        tn.add_tensor("A".to_string(), tensor_a)?;
        let tensor_b = TensorDynLen::from_dense(
            vec![bond_ab.clone(), bond_bc.clone(), bond_bd.clone()],
            vec![1.0; 27],
        )?;
        tn.add_tensor("B".to_string(), tensor_b)?;
        let tensor_c =
            TensorDynLen::from_dense(vec![bond_bc.clone(), site_c.clone()], vec![1.0; 6])?;
        tn.add_tensor("C".to_string(), tensor_c)?;
        let tensor_d =
            TensorDynLen::from_dense(vec![bond_bd.clone(), site_d.clone()], vec![1.0; 6])?;
        tn.add_tensor("D".to_string(), tensor_d)?;

        let n_a = tn.node_index(&"A".to_string()).unwrap();
        let n_b = tn.node_index(&"B".to_string()).unwrap();
        let n_c = tn.node_index(&"C".to_string()).unwrap();
        let n_d = tn.node_index(&"D".to_string()).unwrap();
        tn.connect(n_a, &bond_ab, n_b, &bond_ab)?;
        tn.connect(n_b, &bond_bc, n_c, &bond_bc)?;
        tn.connect(n_b, &bond_bd, n_d, &bond_bd)?;

        // Restructure: fuse C+D into one node (B is internal, pulled in)
        let mut target: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
        target
            .add_node("A".to_string(), HashSet::from([site_a]))
            .map_err(anyhow::Error::msg)?;
        target
            .add_node("CD".to_string(), HashSet::from([site_c, site_d]))
            .map_err(anyhow::Error::msg)?;
        target
            .add_edge(&"A".to_string(), &"CD".to_string())
            .map_err(anyhow::Error::msg)?;

        let result = tn.restructure_to(&target, &RestructureOptions::default())?;
        assert_eq!(result.node_count(), 2);
        let dense_expected = tn.contract_to_tensor()?;
        let dense_actual = result.contract_to_tensor()?;
        assert!((&dense_actual - &dense_expected).maxabs() < 1e-12);
        Ok(())
    }
}
