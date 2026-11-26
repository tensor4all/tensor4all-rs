//! Third Guard stage: every configuration of each small edge side around the
//! call's random starts (#1466, the zero-output case of #1435).
//!
//! The Guard's first two stages are the revalidated retained Guard pivots and
//! one floating-zone walk from each of the call's `nsearch` uniform starts. A
//! walk moves one site at a time and only to a strictly larger residual, so
//! from an output that is exactly zero it cannot cross a plateau of exactly
//! zero residuals to a support it does not touch. On the #1435 table (three
//! branches of 5 binary sites joined at a centre of local dimension 1, target
//! nonzero only on the plane `y = 16`) a start whose `y` is two or more bit
//! flips away from 16 sees only zero residuals, and a run whose first two Guard
//! calls draw only such starts stopped `Converged` with the zero output.
//!
//! When the first two stages of a call return nothing, this stage evaluates,
//! around each of those same starts, every configuration of each maximal edge
//! side with 2 to [`SIDE_ENUMERATION_BUDGET`] configurations, with every other
//! node held at the start. That is TreeTCI's third stage (#1433, #1440), with
//! the call's random starts as anchors instead of retained pivots. On the #1435
//! table the sides are the three branches, so enumerating the `y` branch puts
//! every start's `(x, z)` on the plane.
//!
//! - **Trigger.** Only on a tree with a vertex of local dimension 1
//!   ([`has_unit_vertex`]), as #1440 restricts TreeTCI's stage; the caller runs
//!   it only when the first two stages found nothing.
//! - **No new randomness, no history.** The anchors are the starts the walks
//!   already used; no retained pivot is read and nothing is replayed (#1372).
//! - **Bounded.** At most `A * sum(c(s))` points, with `A <= nsearch` distinct
//!   anchors and `c(s) <= SIDE_ENUMERATION_BUDGET` configurations for each of
//!   at most `2 (n - 1)` maximal sides of an `n`-node tree.
//! - **Streamed.** Points are generated in a fixed order and evaluated in
//!   chunks of at most [`enumeration_chunk_points`]; a point is evaluated only
//!   from the first (anchor, side) pair that generates it, which needs no record
//!   of the points already evaluated (#1441).
//!
//! A side is represented by its nodes of local dimension above 1. A vertex of
//! local dimension 1 has the single coordinate 0 in every point, so it never
//! changes which points a side generates.

use std::collections::HashSet;
use std::mem::size_of;

use crate::{problem::PreparedTreeProblem, Result, TreeAciError, TreeAciNode};

/// Configuration budget of one edge side in the third Guard stage: the #1440
/// bound.
///
/// This is the value of TreeTCI's `LEAF_SIDE_ENUMERATION_BUDGET` (#1433,
/// #1440), carried over unchanged; it was not tuned for tree ACI. On a tree
/// whose variables are binary chains with the least significant bit at the
/// leaf, a side within this budget is at most the low 5 bits of one variable.
/// A side within the budget has at most `log2(32) = 5` nodes of local
/// dimension above 1.
pub(crate) const SIDE_ENUMERATION_BUDGET: usize = 32;

/// Index entries (`points * nodes`) in one evaluation chunk of the stage.
///
/// The same value as TreeTCI's `ENUMERATION_CHUNK_ENTRIES`: the chunk's point
/// buffer holds at most 2^20 coordinates (8 MiB) whatever the tree.
const ENUMERATION_CHUNK_ENTRIES: usize = 1 << 20;

/// Points per evaluation chunk on a tree of `node_count` nodes: the largest
/// count whose coordinates fit [`ENUMERATION_CHUNK_ENTRIES`], and at least one.
pub(crate) fn enumeration_chunk_points(node_count: usize) -> usize {
    (ENUMERATION_CHUNK_ENTRIES / node_count.max(1)).max(1)
}

/// Whether the third Guard stage applies to a tree with these local
/// dimensions: it has a vertex of local dimension 1.
pub(crate) fn has_unit_vertex(local_dims: &[usize]) -> bool {
    local_dims.contains(&1)
}

/// One edge side the stage enumerates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct EnumeratedSide {
    /// The side's nodes of local dimension above 1, ascending.
    nodes: Vec<usize>,
    /// Product of their local dimensions, in `2..=budget`.
    configurations: usize,
}

impl EnumeratedSide {
    fn contains(&self, node: usize) -> bool {
        self.nodes.contains(&node)
    }
}

/// The third Guard stage of one call, planned: its anchors and its sides.
#[derive(Debug)]
pub(crate) struct SideEnumeration<'a> {
    anchors: Vec<&'a [usize]>,
    sides: Vec<EnumeratedSide>,
    local_dims: &'a [usize],
}

/// What one run of the stage found.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct SideEnumerationReport {
    /// Points whose residual exceeds the threshold, strongest first, at most
    /// `max_pivots`. Equal residuals keep evaluation order.
    pub(crate) found: Vec<(f64, Vec<usize>)>,
    /// Points sent to the evaluator.
    pub(crate) evaluated_points: usize,
}

impl<'a> SideEnumeration<'a> {
    /// Plans the stage around `anchors` with sides of at most `budget`
    /// configurations.
    ///
    /// Repeated anchors are kept once. The sides are the maximal edge sides
    /// of `problem` (see [`maximal_sides`]).
    ///
    /// # Errors
    ///
    /// [`TreeAciError::InternalInvariant`] when `local_dims` or an anchor does
    /// not match the problem's node count, or when the problem's edges name
    /// unknown nodes.
    pub(crate) fn new<V: TreeAciNode>(
        problem: &PreparedTreeProblem<V>,
        local_dims: &'a [usize],
        anchors: &'a [Vec<usize>],
        budget: usize,
    ) -> Result<Self> {
        if local_dims.len() != problem.node_order.len() {
            return Err(TreeAciError::InternalInvariant {
                message: "side enumeration local dimensions differ from the node count",
            });
        }
        let mut distinct: Vec<&'a [usize]> = Vec::with_capacity(anchors.len());
        for anchor in anchors {
            if anchor.len() != local_dims.len() {
                return Err(TreeAciError::InternalInvariant {
                    message: "side enumeration anchor differs from the node count",
                });
            }
            if !distinct.contains(&anchor.as_slice()) {
                distinct.push(anchor);
            }
        }
        Ok(Self {
            anchors: distinct,
            sides: maximal_sides(problem, local_dims, budget)?,
            local_dims,
        })
    }

    /// Upper bound on the points the stage evaluates: anchors times the sum of
    /// the sides' configurations.
    ///
    /// # Errors
    ///
    /// [`TreeAciError::SizeOverflow`] when the bound does not fit `usize`.
    pub(crate) fn point_bound(&self) -> Result<usize> {
        self.sides
            .iter()
            .try_fold(0usize, |total, side| total.checked_add(side.configurations))
            .and_then(|per_anchor| per_anchor.checked_mul(self.anchors.len()))
            .ok_or(TreeAciError::SizeOverflow {
                context: "side enumeration point bound",
            })
    }

    /// Bytes the stage keeps besides its evaluation chunk: the anchor list,
    /// the sides, and, per (anchor, side) pair, the earlier pairs that can
    /// generate its points and the anchors' differing nodes.
    ///
    /// # Errors
    ///
    /// [`TreeAciError::SizeOverflow`] when the estimate does not fit `usize`.
    pub(crate) fn bookkeeping_bytes(&self) -> Result<usize> {
        let anchors = self.anchors.len();
        let node_count = self.local_dims.len();
        let side_nodes = self
            .sides
            .iter()
            .try_fold(0usize, |total, side| total.checked_add(side.nodes.len()));
        let side_bytes = self
            .sides
            .len()
            .checked_mul(size_of::<EnumeratedSide>())
            .zip(side_nodes.and_then(|nodes| nodes.checked_mul(size_of::<usize>())))
            .and_then(|(headers, nodes)| headers.checked_add(nodes));
        let anchor_bytes = anchors.checked_mul(size_of::<&[usize]>());
        // Per anchor: the nodes where each earlier anchor differs.
        let difference_bytes = anchors
            .checked_mul(node_count)
            .and_then(|entries| entries.checked_mul(size_of::<usize>()))
            .zip(anchors.checked_mul(size_of::<Vec<usize>>()))
            .and_then(|(payload, headers)| payload.checked_add(headers));
        // Per (anchor, side): every earlier pair at most.
        let generator_bytes = anchors
            .checked_mul(self.sides.len())
            .and_then(|pairs| pairs.checked_mul(size_of::<(usize, usize)>()));
        [side_bytes, anchor_bytes, difference_bytes, generator_bytes]
            .into_iter()
            .try_fold(0usize, |total, bytes| total.checked_add(bytes?))
            .ok_or(TreeAciError::SizeOverflow {
                context: "side enumeration bookkeeping bytes",
            })
    }

    /// Points in the largest chunk a run with `chunk_points` sends to the
    /// evaluator: `chunk_points` (at least one), or fewer when the whole stage
    /// has fewer points.
    ///
    /// # Errors
    ///
    /// [`TreeAciError::SizeOverflow`] as [`Self::point_bound`].
    pub(crate) fn largest_chunk(&self, chunk_points: usize) -> Result<usize> {
        Ok(self.point_bound()?.min(chunk_points.max(1)))
    }

    /// Evaluates the stage.
    ///
    /// For each anchor in order and each side in order, every configuration
    /// of the side (first side node fastest) is combined with the anchor's
    /// coordinates elsewhere. A point that an earlier (anchor, side) pair also
    /// generates is skipped, so every distinct point is evaluated exactly
    /// once, without storing the points already evaluated. `evaluate` receives
    /// the points in chunks of at most `chunk_points` (at least one) and
    /// returns one residual per point. The chunking changes neither which
    /// points are evaluated, nor their order, nor the result.
    ///
    /// # Arguments
    ///
    /// * `chunk_points` -- points per evaluation chunk; production passes
    ///   [`enumeration_chunk_points`].
    /// * `threshold` -- a point is found when its residual exceeds this.
    /// * `max_pivots` -- at most this many found points are kept, strongest
    ///   first.
    /// * `evaluate` -- the residual of every point of a chunk.
    ///
    /// # Errors
    ///
    /// Any error of `evaluate`; [`TreeAciError::InternalInvariant`] when it
    /// returns the wrong number of residuals; [`TreeAciError::SizeOverflow`]
    /// when a count overflows.
    pub(crate) fn run<E>(
        &self,
        chunk_points: usize,
        threshold: f64,
        max_pivots: usize,
        mut evaluate: E,
    ) -> Result<SideEnumerationReport>
    where
        E: FnMut(&[Vec<usize>]) -> Result<Vec<f64>>,
    {
        let chunk_points = self.largest_chunk(chunk_points)?;
        let mut report = SideEnumerationReport::default();
        if chunk_points == 0 || max_pivots == 0 {
            return Ok(report);
        }
        let mut chunk: Vec<Vec<usize>> = Vec::with_capacity(chunk_points);
        let mut flush =
            |chunk: &mut Vec<Vec<usize>>, report: &mut SideEnumerationReport| -> Result<()> {
                if chunk.is_empty() {
                    return Ok(());
                }
                let residuals = evaluate(chunk)?;
                if residuals.len() != chunk.len() {
                    return Err(TreeAciError::InternalInvariant {
                        message: "side enumeration evaluator returned the wrong residual count",
                    });
                }
                report.evaluated_points = report.evaluated_points.checked_add(chunk.len()).ok_or(
                    TreeAciError::SizeOverflow {
                        context: "side enumeration evaluated point count",
                    },
                )?;
                report.found.extend(
                    residuals
                        .into_iter()
                        .zip(chunk.drain(..))
                        .filter(|(residual, _)| *residual > threshold),
                );
                // Stable: equal residuals keep evaluation order.
                report
                    .found
                    .sort_by(|(left, _), (right, _)| right.total_cmp(left));
                report.found.truncate(max_pivots);
                Ok(())
            };
        for (anchor_index, &anchor) in self.anchors.iter().enumerate() {
            let differences = self.anchors[..anchor_index]
                .iter()
                .map(|earlier| differing_nodes(anchor, earlier))
                .collect::<Vec<_>>();
            for (side_index, side) in self.sides.iter().enumerate() {
                let generators = self.earlier_generators(anchor_index, side_index, &differences);
                for code in 0..side.configurations {
                    let mut point = anchor.to_vec();
                    let mut rest = code;
                    for &node in &side.nodes {
                        let dimension = self.local_dims[node];
                        point[node] = rest % dimension;
                        rest /= dimension;
                    }
                    if self.generated_earlier(&point, side, &generators) {
                        continue;
                    }
                    chunk.push(point);
                    if chunk.len() == chunk_points {
                        flush(&mut chunk, &mut report)?;
                    }
                }
            }
        }
        flush(&mut chunk, &mut report)?;
        Ok(report)
    }

    /// The (anchor, side) pairs before (`anchor_index`, `side_index`) that can
    /// generate a point of that pair.
    ///
    /// A point of the pair equals its anchor outside its side. An earlier pair
    /// generates it only if the two anchors agree outside both sides, which
    /// `differences` (the nodes where each earlier anchor differs from this
    /// one) decides without looking at a point. The same anchor agrees with
    /// itself, so every earlier side of the same anchor is a candidate.
    fn earlier_generators(
        &self,
        anchor_index: usize,
        side_index: usize,
        differences: &[Vec<usize>],
    ) -> Vec<(usize, usize)> {
        let side = &self.sides[side_index];
        let mut generators = Vec::new();
        for (earlier_anchor, difference) in differences.iter().enumerate() {
            for (earlier_side_index, earlier_side) in self.sides.iter().enumerate() {
                if difference
                    .iter()
                    .all(|&node| side.contains(node) || earlier_side.contains(node))
                {
                    generators.push((earlier_anchor, earlier_side_index));
                }
            }
        }
        generators.extend((0..side_index).map(|earlier_side| (anchor_index, earlier_side)));
        generators
    }

    /// Whether one of `generators` also generates `point`, a point of `side`:
    /// the point agrees with that pair's anchor on every node of `side`
    /// outside that pair's side. Outside `side` the generators already agree.
    fn generated_earlier(
        &self,
        point: &[usize],
        side: &EnumeratedSide,
        generators: &[(usize, usize)],
    ) -> bool {
        generators.iter().any(|&(anchor_index, side_index)| {
            let anchor = self.anchors[anchor_index];
            let earlier_side = &self.sides[side_index];
            side.nodes
                .iter()
                .all(|&node| earlier_side.contains(node) || point[node] == anchor[node])
        })
    }
}

/// Nodes where two points differ.
fn differing_nodes(left: &[usize], right: &[usize]) -> Vec<usize> {
    left.iter()
        .zip(right)
        .enumerate()
        .filter_map(|(node, (left, right))| (left != right).then_some(node))
        .collect()
}

/// The edge sides the third Guard stage enumerates, in directed-edge order.
///
/// A side is the component on the `from` end of a directed edge. Takes every
/// side whose product of local dimensions lies in `2..=budget` and keeps those
/// not strictly contained in another such side, as TreeTCI's
/// `maximal_enumeration_sides` does. In a tree the sides strictly containing
/// the side of `d` are exactly the sides of the edges `d` feeds into
/// (transitively), and the product grows with the side, so a fitting side is
/// maximal exactly when no edge it feeds directly has a fitting side. A side
/// made only of dimension-1 vertices has one configuration and is never kept;
/// sides with the same nodes of local dimension above 1 generate the same
/// points, so only the first is kept.
///
/// # Errors
///
/// [`TreeAciError::InternalInvariant`] when an edge names a node without a
/// position or an incoming edge is visited before its dependency.
pub(crate) fn maximal_sides<V: TreeAciNode>(
    problem: &PreparedTreeProblem<V>,
    local_dims: &[usize],
    budget: usize,
) -> Result<Vec<EnumeratedSide>> {
    let edge_count = problem.directed_edges.len();
    // `None`: the side has more than `budget` configurations, or is not
    // computed yet.
    let mut fitting: Vec<Option<EnumeratedSide>> = vec![None; edge_count];
    let mut computed = vec![false; edge_count];
    for &directed in &problem.directed_dependency_order {
        let edge = problem
            .directed_edges
            .get(directed)
            .ok_or(TreeAciError::InternalInvariant {
                message: "side enumeration visits an unknown directed edge",
            })?;
        let node =
            *problem
                .node_positions
                .get(&edge.from)
                .ok_or(TreeAciError::InternalInvariant {
                    message: "side enumeration edge has no node position",
                })?;
        let dimension = *local_dims
            .get(node)
            .ok_or(TreeAciError::InternalInvariant {
                message: "side enumeration node has no local dimension",
            })?;
        let mut side = Some(EnumeratedSide {
            nodes: if dimension > 1 {
                vec![node]
            } else {
                Vec::new()
            },
            configurations: dimension,
        })
        .filter(|side| side.configurations <= budget);
        for &incoming in &edge.incoming_to_from {
            if !computed.get(incoming).copied().unwrap_or(false) {
                return Err(TreeAciError::InternalInvariant {
                    message: "side enumeration visits an edge before its dependency",
                });
            }
            side = side
                .zip(fitting[incoming].as_ref())
                .and_then(|(side, child)| {
                    let configurations = side
                        .configurations
                        .checked_mul(child.configurations)
                        .filter(|&count| count <= budget)?;
                    let mut nodes = side.nodes;
                    nodes.extend_from_slice(&child.nodes);
                    Some(EnumeratedSide {
                        nodes,
                        configurations,
                    })
                });
        }
        if let Some(side) = side.as_mut() {
            side.nodes.sort_unstable();
        }
        fitting[directed] = side;
        computed[directed] = true;
    }
    if computed.iter().any(|&done| !done) {
        return Err(TreeAciError::InternalInvariant {
            message: "side enumeration dependency order misses a directed edge",
        });
    }
    let mut feeds_fitting = vec![false; edge_count];
    for (directed, edge) in problem.directed_edges.iter().enumerate() {
        if fitting[directed].is_some() {
            for &incoming in &edge.incoming_to_from {
                feeds_fitting[incoming] = true;
            }
        }
    }
    let mut seen = HashSet::new();
    let mut sides: Vec<EnumeratedSide> = Vec::new();
    for (side, feeds) in fitting.into_iter().zip(feeds_fitting) {
        let Some(side) = side else { continue };
        if feeds || side.configurations < 2 || !seen.insert(side.nodes.clone()) {
            continue;
        }
        sides.push(side);
    }
    Ok(sides)
}

#[cfg(test)]
mod tests;
