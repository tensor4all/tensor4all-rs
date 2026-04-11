use crate::{TreeTCI2, TreeTciEdge};

/// Determines the order in which tree edges are visited during optimization.
///
/// The optimizer calls [`visit_order`](EdgeVisitor::visit_order) once per
/// inner pass to decide which edges to update.
pub trait EdgeVisitor {
    /// Return the ordered list of edges to visit for the current state.
    fn visit_order<T>(&self, state: &TreeTCI2<T>) -> Vec<TreeTciEdge>;
}

/// Visit all edges in canonical graph order.
///
/// This is the default (and currently only) edge visitor. It visits every
/// edge in the tree in sorted order.
///
/// # Examples
///
/// ```
/// use tensor4all_treetci::{AllEdges, EdgeVisitor, TreeTCI2, TreeTciEdge, TreeTciGraph};
///
/// let graph = TreeTciGraph::linear_chain(3).unwrap();
/// let state = TreeTCI2::<f64>::new(vec![2, 2, 2], graph).unwrap();
/// let visitor = AllEdges;
/// let order = visitor.visit_order(&state);
///
/// assert_eq!(order.len(), 2);
/// assert_eq!(order[0], TreeTciEdge::new(0, 1));
/// assert_eq!(order[1], TreeTciEdge::new(1, 2));
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct AllEdges;

impl EdgeVisitor for AllEdges {
    fn visit_order<T>(&self, state: &TreeTCI2<T>) -> Vec<TreeTciEdge> {
        state.graph.edges()
    }
}
