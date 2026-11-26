//! Sorted union of small coordinate blocks without retaining their points.

use crate::MultiIndex;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

/// One mixed-radix digit, ordered by its site in the full multi-index.
#[derive(Clone, Debug)]
struct Coordinate {
    site: usize,
    dimension: usize,
    stride: usize,
}

/// A sorted block cursor. Only the varied coordinates are owned; all other
/// coordinates are borrowed from the retained anchor.
#[derive(Debug)]
struct Cursor<'a> {
    anchor: &'a [usize],
    coordinates: Vec<Coordinate>,
    code: usize,
    configurations: usize,
}

impl Cursor<'_> {
    fn values(&self) -> impl Iterator<Item = usize> + '_ {
        let mut coordinates = self.coordinates.iter().peekable();
        self.anchor.iter().enumerate().map(move |(site, &value)| {
            if let Some(coordinate) = coordinates.next_if(|coordinate| coordinate.site == site) {
                (self.code / coordinate.stride) % coordinate.dimension
            } else {
                value
            }
        })
    }
}

impl Ord for Cursor<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.values().cmp(other.values())
    }
}

impl PartialOrd for Cursor<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Cursor<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Cursor<'_> {}

/// Lexicographic, deduplicated union of every anchor/side block.
///
/// A min-heap holds one sparse cursor per block, with no full point or seen
/// set per cursor. Equal heads advance together, so duplicates need no stored
/// history. The caller retains only its current evaluation batch.
pub(super) struct EnumerationPoints<'a> {
    heads: BinaryHeap<Reverse<Cursor<'a>>>,
}

impl<'a> EnumerationPoints<'a> {
    /// Build cursors for sides already bounded by `maximal_enumeration_sides`.
    pub(super) fn new(
        anchors: &'a [MultiIndex],
        sides: &[Vec<usize>],
        local_dims: &[usize],
    ) -> Self {
        let mut heads = Vec::new();
        if !anchors.is_empty() {
            for side in sides {
                let mut coordinates: Vec<_> = side
                    .iter()
                    .filter(|&&site| local_dims[site] > 1)
                    .map(|&site| Coordinate {
                        site,
                        dimension: local_dims[site],
                        stride: 0,
                    })
                    .collect();
                coordinates.sort_unstable_by_key(|coordinate| coordinate.site);
                let mut configurations = 1;
                // INVARIANT: maximal_enumeration_sides checked the product
                // against 32. There are at most five nonunit digits, and
                // assigning the last site stride 1 makes each block lexical.
                for coordinate in coordinates.iter_mut().rev() {
                    coordinate.stride = configurations;
                    configurations *= coordinate.dimension;
                }
                heads.extend(anchors.iter().map(|anchor| {
                    Reverse(Cursor {
                        anchor,
                        coordinates: coordinates.clone(),
                        code: 0,
                        configurations,
                    })
                }));
            }
        }
        Self {
            heads: BinaryHeap::from(heads),
        }
    }

    /// Consume a head and put its successor back without allocating a point.
    fn advance(&mut self) {
        if let Some(Reverse(mut head)) = self.heads.pop() {
            head.code += 1;
            if head.code < head.configurations {
                self.heads.push(Reverse(head));
            }
        }
    }
}

impl Iterator for EnumerationPoints<'_> {
    type Item = MultiIndex;

    fn next(&mut self) -> Option<Self::Item> {
        // The sole full-point allocation is the result handed to the
        // caller's bounded evaluation batch, never a retained heap key.
        let point: MultiIndex = self.heads.peek()?.0.values().collect();
        self.advance();
        while self
            .heads
            .peek()
            .is_some_and(|head| head.0.values().eq(point.iter().copied()))
        {
            self.advance();
        }
        Some(point)
    }
}

/// Retain only the strongest accepted enumeration points, with stable ties.
/// The stream is already distinct and lexical; earlier equal residuals win,
/// exactly as in the search's stable final sort. Every residual is still
/// accounted for by the caller before admission here.
pub(super) fn retain_pivot(
    best: &mut Vec<(f64, MultiIndex)>,
    error: f64,
    point: &MultiIndex,
    limit: usize,
) {
    let position = best.partition_point(|(prior, _)| prior.total_cmp(&error).is_ge());
    if position < limit {
        if best.len() == limit {
            best.pop();
        }
        best.insert(position, (error, point.clone()));
    }
}

#[cfg(test)]
mod tests;
