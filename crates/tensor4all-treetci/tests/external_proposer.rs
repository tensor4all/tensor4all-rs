//! A pivot candidate proposer written outside the crate, from the public accessors of the state only.
//!
//! The fields of `TreeTCI2` are crate-private, so a proposer of a downstream crate reads the state through
//! `graph()`, `local_dims()` and `pivot_set()`. `NeighborProduct` below is `DefaultProposer` rewritten that way; it
//! must give the same candidates on every edge and, run through `optimize_with_proposer`, the same result.

use std::collections::HashSet;

use anyhow::{anyhow, Result};
use rand::Rng;
use tensor4all_treetci::{
    optimize_with_proposer, AllEdges, DefaultProposer, GlobalIndexBatch, MultiIndex,
    PivotCandidateProposer, SubtreeKey, TreeTCI2, TreeTciEdge, TreeTciGraph, TreeTciOptions,
    TreeTciResult,
};

const N_SITES: usize = 5;

/// `DefaultProposer` against the public accessors: the neighbours' pivots merged into the subtree's positions,
/// expanded over the endpoint's local dimension, and united with the pivots already stored for the subtree.
struct NeighborProduct;

fn position(key: &SubtreeKey, site: usize) -> Result<usize> {
    key.as_slice()
        .iter()
        .position(|&candidate| candidate == site)
        .ok_or_else(|| anyhow!("site {site} is not in the subtree {:?}", key.as_slice()))
}

/// The columns of the pivot set stored for `key`.
fn columns<T>(state: &TreeTCI2<T>, key: &SubtreeKey) -> Result<Vec<MultiIndex>> {
    let pivots = state
        .pivot_set(key)
        .ok_or_else(|| anyhow!("no pivot set for the subtree {:?}", key.as_slice()))?;
    let count = pivots
        .ncols()
        .ok_or_else(|| anyhow!("a pivot set must be two-dimensional"))?;
    Ok((0..count)
        .filter_map(|j| pivots.column(j))
        .map(<[usize]>::to_vec)
        .collect())
}

/// Candidates of the side of `edge` that ends at `site` and whose subtree is `key`.
fn side<T>(
    state: &TreeTCI2<T>,
    edge: TreeTciEdge,
    site: usize,
    key: &SubtreeKey,
) -> Result<Vec<MultiIndex>> {
    let graph = state.graph();
    let incoming = graph.edge_in_ij_keys(site, &graph.adjacent_edges(site, &[edge]))?;

    // The neighbours' pivots, merged into the positions of `key`.
    let mut merged: Vec<MultiIndex> = vec![vec![0; key.as_slice().len()]];
    for in_key in &incoming {
        let positions = in_key
            .as_slice()
            .iter()
            .map(|&neighbour| position(key, neighbour))
            .collect::<Result<Vec<_>>>()?;
        let neighbours = columns(state, in_key)?;
        merged = merged
            .iter()
            .flat_map(|base| {
                neighbours.iter().map(|pivot| {
                    let mut next = base.clone();
                    for (&at, &value) in positions.iter().zip(pivot) {
                        next[at] = value;
                    }
                    next
                })
            })
            .collect();
    }

    // The endpoint's own index runs over its local dimension.
    let at = position(key, site)?;
    let expanded = merged.iter().flat_map(|pivot| {
        (0..state.local_dims()[site]).map(move |value| {
            let mut candidate = pivot.clone();
            candidate[at] = value;
            candidate
        })
    });

    // First occurrence wins; the pivots already stored for the subtree come last.
    let mut seen = HashSet::new();
    Ok(expanded
        .chain(columns(state, key)?)
        .filter(|candidate| seen.insert(candidate.clone()))
        .collect())
}

impl PivotCandidateProposer for NeighborProduct {
    fn candidates_with_rng<T, R: Rng + ?Sized>(
        &self,
        state: &TreeTCI2<T>,
        edge: TreeTciEdge,
        _rng: &mut R,
    ) -> TreeTciResult<(Vec<MultiIndex>, Vec<MultiIndex>)> {
        let graph = state.graph();
        let (left_site, right_site) = graph.separate_vertices(edge)?;
        let (left_key, right_key) = graph.subregion_vertices(edge)?;
        Ok((
            side(state, edge, left_site, &left_key)?,
            side(state, edge, right_site, &right_key)?,
        ))
    }
}

/// A branched tree, so that some sites have two neighbours: 0 - 1 - 2, 1 - 3 - 4.
fn graph() -> TreeTciGraph {
    let edges = [(0, 1), (1, 2), (1, 3), (3, 4)].map(|(a, b)| TreeTciEdge::new(a, b));
    TreeTciGraph::new(N_SITES, &edges).unwrap()
}

fn state() -> TreeTCI2<f64> {
    let mut state = TreeTCI2::<f64>::new(vec![2; N_SITES], graph()).unwrap();
    state.add_global_pivots(&[vec![0; N_SITES]]).unwrap();
    state.set_max_sample_value(1.0);
    state
}

/// `1 / (1 + sum_i (i + 1) x_i)`: smooth, not separable, so the pivot sets grow.
fn target(batch: GlobalIndexBatch<'_>) -> Result<Vec<f64>> {
    Ok((0..batch.n_points())
        .map(|point| {
            let sum: usize = (0..N_SITES)
                .map(|site| (site + 1) * batch.get(site, point).unwrap())
                .sum();
            1.0 / (1.0 + sum as f64)
        })
        .collect())
}

fn options(max_iter: usize) -> TreeTciOptions {
    TreeTciOptions {
        tolerance: 1e-10,
        max_iter,
        // `None` would seed the global pivot search from OS entropy.
        seed: Some(7),
        ..Default::default()
    }
}

#[test]
fn the_external_proposer_gives_the_default_candidates_on_every_edge() {
    let mut state = state();
    // After two iterations the pivot sets are no longer the single seed pivot.
    optimize_with_proposer(
        &mut state,
        target,
        &options(2),
        &DefaultProposer,
        &AllEdges,
        &mut (),
    )
    .unwrap();
    let edges = state.graph().edges();
    assert_eq!(edges.len(), 4);
    for edge in edges {
        let external = NeighborProduct.candidates(&state, edge).unwrap();
        let builtin = DefaultProposer.candidates(&state, edge).unwrap();
        assert_eq!(external, builtin, "edge {edge:?}");
        // A leaf side has at most `local_dim` candidates; the larger side shows that the pivot sets grew.
        assert!(
            external.0.len().max(external.1.len()) > 2,
            "edge {edge:?} has grown pivot sets"
        );
    }
}

#[test]
fn an_external_proposer_runs_through_optimize_with_proposer() {
    let run = |proposer: &dyn Fn(&mut TreeTCI2<f64>) -> _| {
        let mut state = state();
        proposer(&mut state)
    };
    let with_default = run(&|state| {
        optimize_with_proposer(
            state,
            target,
            &options(6),
            &DefaultProposer,
            &AllEdges,
            &mut (),
        )
        .unwrap()
    });
    let with_external = run(&|state| {
        optimize_with_proposer(
            state,
            target,
            &options(6),
            &NeighborProduct,
            &AllEdges,
            &mut (),
        )
        .unwrap()
    });
    assert_eq!(with_external.ranks, with_default.ranks);
    assert_eq!(
        with_external
            .errors
            .iter()
            .map(|e| e.to_bits())
            .collect::<Vec<_>>(),
        with_default
            .errors
            .iter()
            .map(|e| e.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(with_external.termination, with_default.termination);
    assert_eq!(with_external.evaluation, with_default.evaluation);
}
