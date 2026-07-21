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
                region_plan = reverse_regions(&region_plan);
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
            let edges = post_order_dfs_edges_by_name(network, root)?;
            let mut steps = Vec::new();
            let last_edge = edges.len().saturating_sub(1);
            for (j, (src, dst)) in edges.into_iter().enumerate() {
                steps.push(TdvpRegionStep {
                    nodes: vec![src, dst.clone()],
                    new_center: dst.clone(),
                    exponent_step,
                    kind: TdvpRegionKind::TwoSite,
                });
                if j < last_edge {
                    steps.push(TdvpRegionStep {
                        nodes: vec![dst.clone()],
                        new_center: dst,
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

fn reverse_regions<V>(steps: &[TdvpRegionStep<V>]) -> Vec<TdvpRegionStep<V>>
where
    V: Clone,
{
    steps
        .iter()
        .rev()
        .map(|step| {
            let mut nodes = step.nodes.clone();
            nodes.reverse();
            let new_center = nodes
                .last()
                .expect("TDVP region steps are never empty")
                .clone();
            TdvpRegionStep {
                nodes,
                new_center,
                exponent_step: step.exponent_step,
                kind: step.kind,
            }
        })
        .collect()
}

fn post_order_dfs_edges_by_name<V>(network: &NodeNameNetwork<V>, root: &V) -> Option<Vec<(V, V)>>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    let root_idx = network.node_index(root)?;
    let post_order = network.post_order_dfs_by_index(root_idx);
    let mut edges = Vec::new();
    for child in post_order {
        if child == root_idx {
            continue;
        }
        let path = network.path_between(child, root_idx)?;
        let parent = *path.get(1)?;
        let child_name = network.node_name(child)?.clone();
        let parent_name = network.node_name(parent)?.clone();
        edges.push((child_name, parent_name));
    }
    Some(edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain_abc() -> NodeNameNetwork<&'static str> {
        let mut network = NodeNameNetwork::new();
        network.add_node("A").unwrap();
        network.add_node("B").unwrap();
        network.add_node("C").unwrap();
        network.add_edge(&"A", &"B").unwrap();
        network.add_edge(&"B", &"C").unwrap();
        network
    }

    fn star_abcd() -> NodeNameNetwork<&'static str> {
        let mut network = NodeNameNetwork::new();
        for node in ["A", "B", "C", "D"] {
            network.add_node(node).unwrap();
        }
        network.add_edge(&"A", &"B").unwrap();
        network.add_edge(&"A", &"C").unwrap();
        network.add_edge(&"A", &"D").unwrap();
        network
    }

    #[test]
    fn applyexp_sub_steps_match_itensornetworks_orders() {
        assert_eq!(applyexp_sub_steps(1).unwrap(), vec![1.0]);
        assert_eq!(applyexp_sub_steps(2).unwrap(), vec![0.5, 0.5]);

        let fourth = applyexp_sub_steps(4).unwrap();
        assert_eq!(fourth.len(), 6);
        let total: f64 = fourth.iter().sum();
        assert!((total - 1.0).abs() < 1.0e-14);
        assert!(applyexp_sub_steps(3).is_none());
    }

    #[test]
    fn two_site_plan_inserts_negative_single_site_corrections_except_last_edge() {
        let network = chain_abc();
        let exponent = Complex64::new(0.0, -0.2);
        let plan = TdvpRegionPlan::new(&network, &"B", 2, 1, exponent).unwrap();

        let regions: Vec<_> = plan
            .steps
            .iter()
            .map(|step| (step.kind, step.nodes.clone(), step.exponent_step))
            .collect();
        assert_eq!(
            regions,
            vec![
                (TdvpRegionKind::TwoSite, vec!["A", "B"], exponent),
                (TdvpRegionKind::SiteCorrection, vec!["B"], -exponent),
                (TdvpRegionKind::TwoSite, vec!["C", "B"], exponent),
            ]
        );
    }

    #[test]
    fn second_order_plan_reverses_even_substep_regions() {
        let network = chain_abc();
        let exponent = Complex64::new(0.0, -0.2);
        let plan = TdvpRegionPlan::new(&network, &"B", 2, 2, exponent).unwrap();

        let regions: Vec<_> = plan
            .steps
            .iter()
            .map(|step| (step.kind, step.nodes.clone(), step.exponent_step))
            .collect();
        let half = exponent * 0.5;
        assert_eq!(
            regions,
            vec![
                (TdvpRegionKind::TwoSite, vec!["A", "B"], half),
                (TdvpRegionKind::SiteCorrection, vec!["B"], -half),
                (TdvpRegionKind::TwoSite, vec!["C", "B"], half),
                (TdvpRegionKind::TwoSite, vec!["B", "C"], half),
                (TdvpRegionKind::SiteCorrection, vec!["B"], -half),
                (TdvpRegionKind::TwoSite, vec!["B", "A"], half),
            ]
        );
    }

    #[test]
    fn one_site_plan_uses_post_order_vertices() {
        let network = star_abcd();
        let exponent = Complex64::new(0.0, -0.2);
        let plan = TdvpRegionPlan::new(&network, &"A", 1, 1, exponent).unwrap();

        assert_eq!(plan.nsite, 1);
        assert_eq!(plan.order, 1);
        assert_eq!(
            plan.steps
                .iter()
                .map(|step| (step.kind, step.nodes.clone()))
                .collect::<Vec<_>>(),
            vec![
                (TdvpRegionKind::OneSite, vec!["B"]),
                (TdvpRegionKind::OneSite, vec!["C"]),
                (TdvpRegionKind::OneSite, vec!["D"]),
                (TdvpRegionKind::OneSite, vec!["A"]),
            ]
        );
    }
}
