// Portions of this file are derived from ITensorNetworks.jl
// (https://github.com/ITensor/ITensorNetworks.jl),
// Copyright 2021 The Simons Foundation, Inc. - All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (see LICENSE-APACHE in this
// crate). Changes: translated from Julia to Rust and adapted to tensor4all-rs
// tensor and graph types.

use std::fmt::Debug;
use std::hash::Hash;

use num_complex::Complex64;

use crate::node_name_network::NodeNameNetwork;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TdvpRegionKind {
    OneSite,
    TwoSite,
    SiteCorrection,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TdvpRegionStep<V> {
    pub(crate) nodes: Vec<V>,
    pub(crate) new_center: V,
    pub(crate) exponent_step: Complex64,
    pub(crate) kind: TdvpRegionKind,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TdvpRegionPlan<V> {
    pub(crate) steps: Vec<TdvpRegionStep<V>>,
    pub(crate) nsite: usize,
    pub(crate) order: usize,
}

impl<V> TdvpRegionPlan<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    pub(crate) fn new(
        network: &NodeNameNetwork<V>,
        root: &V,
        nsite: usize,
        order: usize,
        exponent_step: Complex64,
    ) -> Option<Self> {
        if nsite != 1 && nsite != 2 {
            return None;
        }
        let weights = applyexp_sub_steps(order)?;
        let base_sweep = first_order_sweep(network, root, nsite, exponent_step)?;
        let mut steps = Vec::new();
        for (substep, weight) in weights.iter().enumerate() {
            let mut region_plan = base_sweep
                .iter()
                .map(|step| {
                    let mut step = step.clone();
                    step.exponent_step *= *weight;
                    step
                })
                .collect::<Vec<_>>();
            if (substep + 1) % 2 == 0 {
                region_plan = reverse_regions(&region_plan)?;
            }
            steps.extend(region_plan);
        }
        Some(Self {
            steps,
            nsite,
            order,
        })
    }
}

pub(crate) fn applyexp_sub_steps(order: usize) -> Option<Vec<f64>> {
    match order {
        1 => Some(vec![1.0]),
        2 => Some(vec![0.5, 0.5]),
        4 => {
            let s = (2.0 - 2.0_f64.powf(1.0 / 3.0)).recip();
            Some(vec![s / 2.0, s / 2.0, 0.5 - s, 0.5 - s, s / 2.0, s / 2.0])
        }
        _ => None,
    }
}

fn first_order_sweep<V>(
    network: &NodeNameNetwork<V>,
    root: &V,
    nsite: usize,
    exponent_step: Complex64,
) -> Option<Vec<TdvpRegionStep<V>>>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    match nsite {
        1 => {
            let vertices = network.post_order_dfs(root)?;
            Some(
                vertices
                    .into_iter()
                    .map(|node| TdvpRegionStep {
                        nodes: vec![node.clone()],
                        new_center: node,
                        exponent_step,
                        kind: TdvpRegionKind::OneSite,
                    })
                    .collect(),
            )
        }
        2 => {
            // Root-edge-first outward walk (pre-order DFS edges). Starting each
            // half-sweep at the edge containing the sweep root keeps every gauge
            // move inside already-evolved regions, which is required for the
            // symmetric projector-splitting integrator; the old post-order
            // (root-edge-last) ordering transported the gauge across unevolved
            // edges at the start and turning point of each half-sweep. Matches
            // ITensorNetworks.jl at NamedGraphs >= 0.11 (see benchmarks/results/
            // 2026-07-22-treetn-tdvp-itensornetworks-1t.md).
            let edges = pre_order_dfs_edges_by_name(network, root)?;
            let mut steps = Vec::new();
            let last_edge = edges.len().saturating_sub(1);
            for (j, (parent, child)) in edges.iter().enumerate() {
                // The center after a two-site step sits on the vertex shared
                // with the next region so the following site correction acts on
                // the overlap; the final edge parks the center on its far side.
                let center = match edges.get(j + 1) {
                    Some((next_parent, next_child)) => {
                        if parent == next_parent || parent == next_child {
                            parent.clone()
                        } else {
                            child.clone()
                        }
                    }
                    None => child.clone(),
                };
                let other = if center == *parent {
                    child.clone()
                } else {
                    parent.clone()
                };
                steps.push(TdvpRegionStep {
                    nodes: vec![other, center.clone()],
                    new_center: center.clone(),
                    exponent_step,
                    kind: TdvpRegionKind::TwoSite,
                });
                if j < last_edge {
                    steps.push(TdvpRegionStep {
                        nodes: vec![center.clone()],
                        new_center: center,
                        exponent_step: -exponent_step,
                        kind: TdvpRegionKind::SiteCorrection,
                    });
                }
            }
            Some(steps)
        }
        _ => None,
    }
}

fn reverse_regions<V>(steps: &[TdvpRegionStep<V>]) -> Option<Vec<TdvpRegionStep<V>>>
where
    V: Clone,
{
    let mut reversed = Vec::with_capacity(steps.len());
    for step in steps.iter().rev() {
        let mut nodes = step.nodes.clone();
        nodes.reverse();
        let new_center = nodes.last()?.clone();
        reversed.push(TdvpRegionStep {
            nodes,
            new_center,
            exponent_step: step.exponent_step,
            kind: step.kind,
        });
    }
    Some(reversed)
}

/// Tree edges as `(parent, child)` pairs in a parents-before-children order
/// (reverse post-order), so the edge containing `root` comes first and each
/// subsequent edge attaches to an already-visited vertex.
fn pre_order_dfs_edges_by_name<V>(network: &NodeNameNetwork<V>, root: &V) -> Option<Vec<(V, V)>>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    let root_idx = network.node_index(root)?;
    let post_order = network.post_order_dfs_by_index(root_idx);
    let mut edges = Vec::new();
    for &child in post_order.iter().rev() {
        if child == root_idx {
            continue;
        }
        let path = network.path_between(child, root_idx)?;
        let parent = *path.get(1)?;
        let child_name = network.node_name(child)?.clone();
        let parent_name = network.node_name(parent)?.clone();
        edges.push((parent_name, child_name));
    }
    Some(edges)
}

#[cfg(test)]
mod tests;
