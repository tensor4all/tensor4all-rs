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
mod tests {
    use super::*;

    fn sites(dims: &[usize]) -> Vec<DynIndex> {
        dims.iter().map(|&dim| DynIndex::new_dyn(dim)).collect()
    }

    #[test]
    fn an_empty_variable_option_uses_one_group_of_all_active_sites() {
        let sites = sites(&[2, 3]);
        let active = [0, 1];
        let dims = [2, 3];
        let starts = corner_and_edge_starts(&sites, &active, &dims, &[], 0);
        // One group: the two corners (all zero, all high) and up to K_EDGE
        // edges, each point once.
        assert_eq!(starts[0], vec![0, 0]);
        assert_eq!(starts[1], vec![1, 2]);
        assert!((2..=2 + K_EDGE).contains(&starts.len()));
        // Every point is a valid point of the patch.
        assert!(starts.iter().all(|p| p[0] < 2 && p[1] < 3));
    }

    #[test]
    fn variable_groups_mix_whole_variables() {
        // Two variables of two bits each: the corners set both bits of a
        // variable together, so the four corners are (0,0), (0,1), (1,0),
        // (1,1) as two-bit coordinates of x = s0 + 2 s1 and y = s2 + 2 s3.
        let sites = sites(&[2, 2, 2, 2]);
        let variables = vec![
            vec![sites[0].clone(), sites[1].clone()],
            vec![sites[2].clone(), sites[3].clone()],
        ];
        let starts = corner_and_edge_starts(&sites, &[0, 1, 2, 3], &[2, 2, 2, 2], &variables, 7);
        assert_eq!(
            &starts[..4],
            &[
                vec![0, 0, 0, 0],
                vec![1, 1, 0, 0],
                vec![0, 0, 1, 1],
                vec![1, 1, 1, 1],
            ]
        );
        // The single group of all four bits (the default) would also contain
        // the mixed corners [1, 0, 0, 0] and [0, 1, 0, 0].
        let one_group = corner_and_edge_starts(&sites, &[0, 1, 2, 3], &[2, 2, 2, 2], &[], 7);
        assert!(one_group.contains(&vec![1, 0, 0, 0]));
        assert!(!starts.contains(&vec![1, 0, 0, 0]));
    }

    #[test]
    fn ungrouped_active_sites_form_one_more_group() {
        let sites = sites(&[2, 2, 2]);
        let variables = vec![vec![sites[0].clone()]];
        let starts = corner_and_edge_starts(&sites, &[0, 1, 2], &[2, 2, 2], &variables, 0);
        // Two groups: site 0 and sites 1, 2 together.
        assert_eq!(starts[0], vec![0, 0, 0]);
        assert!(starts.contains(&vec![0, 1, 1]));
        assert!(starts.contains(&vec![1, 0, 0]));
        assert!(starts.contains(&vec![1, 1, 1]));
    }

    #[test]
    fn the_corner_list_is_a_fixed_function_of_the_stream() {
        // More groups than MAX_CORNERS allows, so the corners are drawn.
        let dims = vec![2usize; 7];
        let sites: Vec<DynIndex> = dims.iter().map(|&dim| DynIndex::new_dyn(dim)).collect();
        let variables: Vec<Vec<DynIndex>> = sites.iter().map(|site| vec![site.clone()]).collect();
        let active: Vec<usize> = (0..7).collect();
        let first = corner_and_edge_starts(&sites, &active, &dims, &variables, 11);
        let second = corner_and_edge_starts(&sites, &active, &dims, &variables, 11);
        assert_eq!(first, second);
        // Duplicate draws collapse, so the list is at most the drawn count.
        assert!(first.len() > MAX_CORNERS / 2);
        assert!(first.len() <= MAX_CORNERS + K_EDGE);
        let other = corner_and_edge_starts(&sites, &active, &dims, &variables, 12);
        assert_ne!(first, other);
        assert!(first.iter().all(|point| point.len() == 7));
    }
}
