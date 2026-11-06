//! Corner and edge start points of a patch, used by the residual-TT
//! estimator.
//!
//! A feature that enters a patch only through a corner or an edge is a flat
//! zero fiber from most starting points, so the estimator passes explicit
//! corners and edges to the engine's global pivot search. The points are in
//! active-site coordinates, in the order of the patch's active sites.
//!
//! # Groups
//!
//! [`PatchedInterpolationOptions::variables`](super::PatchedInterpolationOptions::variables)
//! names the sites of each variable. A patch uses the active sites of every
//! named group, plus one group of its active sites that no group names; an
//! empty option is one group of all its active sites. Only active sites
//! appear, so a group may be empty or absent in a patch.
//!
//! # Corners and edges
//!
//! A corner sets every site of a group to `0` or to `dim - 1`. All
//! `2^groups` corners are generated up to [`MAX_CORNERS`]; beyond that,
//! [`MAX_CORNERS`] corners are drawn from the estimator's stream. An edge is
//! a corner whose groups are drawn randomly, with one randomly chosen group
//! set to uniform random coordinates. [`K_EDGE`] edges follow the corners.
//! Every point is kept once, in generation order.

use std::collections::HashSet;

use tensor4all_core::DynIndex;

use super::sampling::SplitMix64;

/// Corner starts generated before the remainder is drawn.
pub(super) const MAX_CORNERS: usize = 64;
/// Edge starts after the corners.
pub(super) const K_EDGE: usize = 4;

/// The corner and edge start points of a patch, in active coordinates, each
/// point once, in generation order. `seed` drives the draws when `2^groups`
/// exceeds [`MAX_CORNERS`] and the [`K_EDGE`] edges.
pub(super) fn corner_and_edge_starts(
    sites: &[DynIndex],
    active: &[usize],
    active_dims: &[usize],
    variables: &[Vec<DynIndex>],
    seed: u64,
) -> Vec<Vec<usize>> {
    let groups = active_groups(sites, active, variables);
    let mut starts = Vec::new();
    let mut seen = HashSet::new();
    let mut rng = SplitMix64::new(seed);
    add_corners(&mut starts, &mut seen, &groups, active_dims, &mut rng);
    add_edges(&mut starts, &mut seen, &groups, active_dims, &mut rng);
    starts
}

fn push_unique(starts: &mut Vec<Vec<usize>>, seen: &mut HashSet<Vec<usize>>, point: Vec<usize>) {
    if seen.insert(point.clone()) {
        starts.push(point);
    }
}

/// Active slots of every variable group, then the active slots in no group.
/// An empty `variables` option is one group of every active slot.
fn active_groups(
    sites: &[DynIndex],
    active: &[usize],
    variables: &[Vec<DynIndex>],
) -> Vec<Vec<usize>> {
    if variables.is_empty() {
        return vec![(0..active.len()).collect()];
    }
    let slot_of: std::collections::HashMap<&DynIndex, usize> = active
        .iter()
        .enumerate()
        .map(|(slot, &position)| (&sites[position], slot))
        .collect();
    let mut assigned = vec![false; active.len()];
    let mut groups = Vec::new();
    for variable in variables {
        let mut group: Vec<usize> = variable
            .iter()
            .filter_map(|site| slot_of.get(site).copied())
            .filter(|&slot| !std::mem::replace(&mut assigned[slot], true))
            .collect();
        group.sort_unstable();
        if !group.is_empty() {
            groups.push(group);
        }
    }
    let ungrouped: Vec<usize> = (0..active.len()).filter(|&slot| !assigned[slot]).collect();
    if !ungrouped.is_empty() {
        groups.push(ungrouped);
    }
    groups
}

/// Add the corners of the groups: all `2^groups` combinations up to
/// [`MAX_CORNERS`], otherwise that many combinations drawn from `rng`.
fn add_corners(
    starts: &mut Vec<Vec<usize>>,
    seen: &mut HashSet<Vec<usize>>,
    groups: &[Vec<usize>],
    active_dims: &[usize],
    rng: &mut SplitMix64,
) {
    let n_groups = groups.len();
    if n_groups < usize::BITS as usize && (1usize << n_groups) <= MAX_CORNERS {
        for mask in 0..(1usize << n_groups) {
            let bits: Vec<bool> = (0..n_groups)
                .map(|group| mask & (1 << group) != 0)
                .collect();
            push_unique(starts, seen, corner(groups, active_dims, &bits));
        }
    } else {
        for _ in 0..MAX_CORNERS {
            let bits: Vec<bool> = (0..n_groups).map(|_| rng.below(2) == 1).collect();
            push_unique(starts, seen, corner(groups, active_dims, &bits));
        }
    }
}

/// Add edge starts: a random corner with one random group set uniformly.
fn add_edges(
    starts: &mut Vec<Vec<usize>>,
    seen: &mut HashSet<Vec<usize>>,
    groups: &[Vec<usize>],
    active_dims: &[usize],
    rng: &mut SplitMix64,
) {
    if groups.is_empty() {
        return;
    }
    for _ in 0..K_EDGE {
        let bits: Vec<bool> = (0..groups.len()).map(|_| rng.below(2) == 1).collect();
        let mut point = corner(groups, active_dims, &bits);
        let group = rng.below(groups.len() as u64) as usize;
        for &slot in &groups[group] {
            point[slot] = rng.below(active_dims[slot] as u64) as usize;
        }
        push_unique(starts, seen, point);
    }
}

/// The corner selected by `bits`: group `g` at its high extreme when
/// `bits[g]` is set, at zero otherwise.
fn corner(groups: &[Vec<usize>], active_dims: &[usize], bits: &[bool]) -> Vec<usize> {
    let mut point = vec![0usize; active_dims.len()];
    for (group, slots) in groups.iter().enumerate() {
        if bits[group] {
            for &slot in slots {
                point[slot] = active_dims[slot] - 1;
            }
        }
    }
    point
}

#[cfg(test)]
mod tests;
