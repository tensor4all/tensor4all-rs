use crate::{assemble::MultiIndex, SimpleTreeTci, SubtreeKey, TreeTciEdge};
use anyhow::{ensure, Result};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Generates candidate pivot sets for one edge bipartition.
pub trait PivotCandidateProposer {
    /// Return `(I_candidates, J_candidates)` for the requested edge.
    fn candidates<T>(
        &self,
        state: &SimpleTreeTci<T>,
        edge: TreeTciEdge,
    ) -> Result<(Vec<MultiIndex>, Vec<MultiIndex>)>;
}

/// Default neighbor-product proposer that mirrors `TreeTCI.jl`.
#[derive(Clone, Copy, Debug, Default)]
pub struct DefaultProposer;

impl PivotCandidateProposer for DefaultProposer {
    fn candidates<T>(
        &self,
        state: &SimpleTreeTci<T>,
        edge: TreeTciEdge,
    ) -> Result<(Vec<MultiIndex>, Vec<MultiIndex>)> {
        let (vp, vq) = state.graph.separate_vertices(edge)?;
        let (ikey, jkey) = state.graph.subregion_vertices(edge)?;

        let adjacent_vp = state.graph.adjacent_edges(vp, &[edge]);
        let in_ikeys = state.graph.edge_in_ij_keys(vp, &adjacent_vp)?;
        let ipivots = pivot_set(&state.ijset, &in_ikeys, &ikey)?;
        let isite_index = subtree_position(&ikey, vp)?;
        let iset = kronecker(&ipivots, isite_index, state.local_dims[vp]);

        let adjacent_vq = state.graph.adjacent_edges(vq, &[edge]);
        let in_jkeys = state.graph.edge_in_ij_keys(vq, &adjacent_vq)?;
        let jpivots = pivot_set(&state.ijset, &in_jkeys, &jkey)?;
        let jsite_index = subtree_position(&jkey, vq)?;
        let jset = kronecker(&jpivots, jsite_index, state.local_dims[vq]);

        let history = state.ijset_history.last();
        let icombined = union_with_history(iset, history, &ikey);
        let jcombined = union_with_history(jset, history, &jkey);
        Ok((icombined, jcombined))
    }
}

/// Simple random proposer that mirrors `TreeTCI.jl`'s
/// `SimplePivotCandidateProposer`.
#[derive(Clone, Copy, Debug)]
pub struct SimpleProposer {
    seed: u64,
}

impl SimpleProposer {
    /// Construct a proposer with a deterministic base seed.
    pub const fn seeded(seed: u64) -> Self {
        Self { seed }
    }

    fn rng_for_edge<T>(&self, state: &SimpleTreeTci<T>, edge: TreeTciEdge) -> SmallRng {
        let (ikey, jkey) = state
            .graph
            .subregion_vertices(edge)
            .expect("edge must belong to the tree graph");
        let mut hasher = DefaultHasher::new();
        self.seed.hash(&mut hasher);
        edge.hash(&mut hasher);
        state.ijset_history.len().hash(&mut hasher);
        state
            .ijset
            .get(&ikey)
            .map_or(0usize, Vec::len)
            .hash(&mut hasher);
        state
            .ijset
            .get(&jkey)
            .map_or(0usize, Vec::len)
            .hash(&mut hasher);
        SmallRng::seed_from_u64(hasher.finish())
    }
}

impl Default for SimpleProposer {
    fn default() -> Self {
        Self::seeded(0)
    }
}

impl PivotCandidateProposer for SimpleProposer {
    fn candidates<T>(
        &self,
        state: &SimpleTreeTci<T>,
        edge: TreeTciEdge,
    ) -> Result<(Vec<MultiIndex>, Vec<MultiIndex>)> {
        let (vp, vq) = state.graph.separate_vertices(edge)?;
        let (ikey, jkey) = state.graph.subregion_vertices(edge)?;
        let mut rng = self.rng_for_edge(state, edge);

        let ichi = state.local_dims[vp]
            * state
                .ijset
                .get(&ikey)
                .ok_or_else(|| anyhow::anyhow!("missing pivot set for subtree key {:?}", ikey))?
                .len();
        let jchi = state.local_dims[vq]
            * state
                .ijset
                .get(&jkey)
                .ok_or_else(|| anyhow::anyhow!("missing pivot set for subtree key {:?}", jkey))?
                .len();

        let iset = random_candidates(&mut rng, state.local_dims.as_slice(), &ikey, ichi);
        let jset = random_candidates(&mut rng, state.local_dims.as_slice(), &jkey, jchi);

        let history = state.ijset_history.last();
        let icombined = union_with_history(iset, history, &ikey);
        let jcombined = union_with_history(jset, history, &jkey);
        Ok((icombined, jcombined))
    }
}

fn subtree_position(key: &SubtreeKey, site: usize) -> Result<usize> {
    key.as_slice()
        .iter()
        .position(|&value| value == site)
        .ok_or_else(|| anyhow::anyhow!("site {} not found in subtree key {:?}", site, key))
}

fn union_with_history(
    values: Vec<MultiIndex>,
    history: Option<&HashMap<SubtreeKey, Vec<MultiIndex>>>,
    key: &SubtreeKey,
) -> Vec<MultiIndex> {
    let mut unique = Vec::with_capacity(values.len());
    for candidate in values {
        if !unique.contains(&candidate) {
            unique.push(candidate);
        }
    }
    if let Some(extra) = history.and_then(|history| history.get(key)) {
        for candidate in extra {
            if !unique.contains(candidate) {
                unique.push(candidate.clone());
            }
        }
    }
    unique
}

fn pivot_set(
    ijset: &HashMap<SubtreeKey, Vec<MultiIndex>>,
    in_keys: &[SubtreeKey],
    out_key: &SubtreeKey,
) -> Result<Vec<MultiIndex>> {
    let out_len = out_key.as_slice().len();
    let mut pivots = vec![vec![0; out_len]];

    for in_key in in_keys {
        let incoming = ijset
            .get(in_key)
            .ok_or_else(|| anyhow::anyhow!("missing pivot set for subtree key {:?}", in_key))?;
        let mut next = Vec::new();
        for base in &pivots {
            for index in incoming {
                ensure!(
                    index.len() == in_key.as_slice().len(),
                    "pivot length {} does not match subtree key length {}",
                    index.len(),
                    in_key.as_slice().len()
                );
                let mut merged = base.clone();
                for (&site, &value) in in_key.as_slice().iter().zip(index.iter()) {
                    let out_pos = subtree_position(out_key, site)?;
                    merged[out_pos] = value;
                }
                next.push(merged);
            }
        }
        pivots = next;
    }

    Ok(pivots)
}

fn kronecker(pivots: &[MultiIndex], site_index: usize, local_dim: usize) -> Vec<MultiIndex> {
    let mut result = Vec::with_capacity(pivots.len() * local_dim);
    for pivot in pivots {
        for value in 0..local_dim {
            let mut candidate = pivot.clone();
            candidate[site_index] = value;
            result.push(candidate);
        }
    }
    result
}

fn random_candidates(
    rng: &mut SmallRng,
    local_dims: &[usize],
    key: &SubtreeKey,
    size: usize,
) -> Vec<MultiIndex> {
    (0..size)
        .map(|_| {
            key.as_slice()
                .iter()
                .map(|&site| rng.random_range(0..local_dims[site]))
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests;
