//! A trie over a set of projectors that answers overlap queries (#1293).
//!
//! Two projectors overlap when they agree on every index both of them fix
//! ([`Projector::is_compatible_with`]). Callers used to answer "does any member
//! overlap `q`?" and "which members overlap `q`?" by testing every member, which
//! is quadratic when it runs for every member of a set and for every patch pair
//! of a contraction.
//!
//! The trie has one level per index that some member fixes. Levels are ordered
//! by how many members fix the index, most first, with ties broken by the
//! canonical index order, so a layout produced by splitting along one
//! `patch_order` puts its split indices at the top. At each level a member that
//! fixes the index goes into the child for its coordinate and a member that
//! does not goes into the wildcard child. A member is stored at the node after
//! its deepest fixed level: everything below that node is free for it.
//!
//! A query `q` collects the members at every node it reaches. At a level that
//! `q` fixes it descends into the child for `q`'s coordinate and the wildcard
//! child; at a level `q` leaves free it descends into every child. A member is
//! reached exactly when it agrees with `q` on every index both fix, so the
//! result is the brute-force one. Indices `q` fixes that no member fixes play
//! no role. In the worst case, members fixing unrelated index sets, a query
//! still visits every member; it never does more than a constant factor over
//! the pairwise loop it replaces.

use std::collections::HashMap;
use std::ops::ControlFlow;

use tensor4all_core::DynIndex;

use super::{canonical_index_cmp, Projector};

/// A trie over a fixed set of projectors, addressed by their position in the
/// slice it was built from.
pub(crate) struct ProjectorIndex {
    order: Vec<DynIndex>,
    nodes: Vec<Node>,
}

#[derive(Default)]
struct Node {
    children: HashMap<usize, usize>,
    wildcard: Option<usize>,
    members: Vec<usize>,
}

impl ProjectorIndex {
    /// Index `projectors`; a query reports members by their position here.
    pub(crate) fn new(projectors: &[&Projector]) -> Self {
        let mut counts: HashMap<&DynIndex, usize> = HashMap::new();
        for projector in projectors {
            for index in projector.projected_indices() {
                *counts.entry(index).or_insert(0) += 1;
            }
        }
        let mut order: Vec<(&DynIndex, usize)> = counts.into_iter().collect();
        order.sort_by(|(left, left_count), (right, right_count)| {
            right_count
                .cmp(left_count)
                .then_with(|| canonical_index_cmp(left, right))
        });
        let order: Vec<DynIndex> = order.into_iter().map(|(index, _)| index.clone()).collect();
        let level: HashMap<&DynIndex, usize> = order
            .iter()
            .enumerate()
            .map(|(depth, index)| (index, depth))
            .collect();

        let mut nodes = vec![Node::default()];
        for (member, projector) in projectors.iter().enumerate() {
            let deepest = projector
                .projected_indices()
                .map(|index| level[index] + 1)
                .max()
                .unwrap_or(0);
            let mut node = 0;
            for index in &order[..deepest] {
                let next = nodes.len();
                let child = match projector.get(index) {
                    Some(value) => *nodes[node].children.entry(value).or_insert(next),
                    None => *nodes[node].wildcard.get_or_insert(next),
                };
                if child == next {
                    nodes.push(Node::default());
                }
                node = child;
            }
            nodes[node].members.push(member);
        }
        Self { order, nodes }
    }

    /// Call `visit` with the position of every member that overlaps `query`,
    /// in no particular order, until it breaks.
    pub(crate) fn for_each_overlapping(
        &self,
        query: &Projector,
        mut visit: impl FnMut(usize) -> ControlFlow<()>,
    ) -> ControlFlow<()> {
        let mut stack = vec![(0usize, 0usize)];
        while let Some((node, depth)) = stack.pop() {
            let node = &self.nodes[node];
            for &member in &node.members {
                visit(member)?;
            }
            let Some(index) = self.order.get(depth) else {
                continue;
            };
            match query.get(index) {
                Some(value) => {
                    if let Some(&child) = node.children.get(&value) {
                        stack.push((child, depth + 1));
                    }
                }
                None => stack.extend(node.children.values().map(|&child| (child, depth + 1))),
            }
            if let Some(child) = node.wildcard {
                stack.push((child, depth + 1));
            }
        }
        ControlFlow::Continue(())
    }

    /// Positions of the members that overlap `query`, ascending.
    pub(crate) fn overlapping(&self, query: &Projector) -> Vec<usize> {
        let mut found = Vec::new();
        let _ = self.for_each_overlapping(query, |member| {
            found.push(member);
            ControlFlow::Continue(())
        });
        found.sort_unstable();
        found
    }
}

/// Whether two members at different positions of `projectors` overlap.
///
/// With `allow_equal`, members with equal keys do not count as overlapping,
/// as in a partition, where an equal key replaces the stored patch. Without
/// it, equal keys overlap, as [`Projector::are_disjoint`] defines.
pub(crate) fn has_overlapping_pair(projectors: &[&Projector], allow_equal: bool) -> bool {
    let index = ProjectorIndex::new(projectors);
    projectors.iter().enumerate().any(|(position, projector)| {
        index
            .for_each_overlapping(projector, |member| {
                if member == position || (allow_equal && projectors[member] == *projector) {
                    ControlFlow::Continue(())
                } else {
                    ControlFlow::Break(())
                }
            })
            .is_break()
    })
}

/// Whether a member of `left` overlaps a member of `right`; `allow_equal` as in
/// [`has_overlapping_pair`].
pub(crate) fn sets_overlap(left: &[&Projector], right: &[&Projector], allow_equal: bool) -> bool {
    let (indexed, queries) = if left.len() >= right.len() {
        (left, right)
    } else {
        (right, left)
    };
    let index = ProjectorIndex::new(indexed);
    queries.iter().any(|query| {
        index
            .for_each_overlapping(query, |member| {
                if allow_equal && indexed[member] == *query {
                    ControlFlow::Continue(())
                } else {
                    ControlFlow::Break(())
                }
            })
            .is_break()
    })
}

#[cfg(test)]
mod tests;
