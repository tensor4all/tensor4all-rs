use crate::{TreeTCI2, TreeTciEdge};

/// Determines the order in which tree edges are visited during optimization.
pub trait EdgeVisitor {
    /// Return the ordered list of edges to visit for the current state.
    fn visit_order<T>(&self, state: &TreeTCI2<T>) -> Vec<TreeTciEdge>;
}

/// Visit all edges in canonical graph order.
#[derive(Clone, Copy, Debug, Default)]
pub struct AllEdges;

impl EdgeVisitor for AllEdges {
    fn visit_order<T>(&self, state: &TreeTCI2<T>) -> Vec<TreeTciEdge> {
        state.graph.edges()
    }
}
