use crate::{TreeTCI2, TreeTciEdge};

/// Determines the order in which tree edges are visited during optimization.
///
/// The optimizer calls [`visit_order`](EdgeVisitor::visit_order) once per
/// inner pass to decide which edges to update. Every result must be a
/// permutation of the graph's edges: each edge exactly once per pass, with no
/// missing, duplicated, or unknown edge. The optimizer validates this before a
/// pass updates any state and rejects a violation with
/// [`TreeTciError::InvalidConfiguration`](crate::TreeTciError::InvalidConfiguration).
///
/// # Examples
///
/// ```
/// use tensor4all_treetci::{AllEdges, EdgeVisitor, TreeTCI2, TreeTciEdge, TreeTciGraph};
///
/// let graph = TreeTciGraph::linear_chain(3).unwrap();
/// let state = TreeTCI2::<f64>::new(vec![2, 2, 2], graph).unwrap();
/// let order = AllEdges.visit_order(&state);
/// assert_eq!(order, vec![TreeTciEdge::new(0, 1), TreeTciEdge::new(1, 2)]);
/// ```
pub trait EdgeVisitor {
    /// Return the ordered list of edges to visit for the current state.
    fn visit_order<T>(&self, state: &TreeTCI2<T>) -> Vec<TreeTciEdge>;

    /// Return the edge order for inner pass `pass` (0-based) of an iteration.
    ///
    /// The optimizer calls this method; the default returns
    /// [`visit_order`](EdgeVisitor::visit_order) for every pass, so a visitor
    /// that implements only `visit_order` keeps one order on every pass.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetci::{EdgeVisitor, TreeTCI2, TreeTciEdge, TreeTciGraph};
    ///
    /// struct Reversed;
    /// impl EdgeVisitor for Reversed {
    ///     fn visit_order<T>(&self, state: &TreeTCI2<T>) -> Vec<TreeTciEdge> {
    ///         state.graph().edges().into_iter().rev().collect()
    ///     }
    /// }
    ///
    /// let graph = TreeTciGraph::linear_chain(3).unwrap();
    /// let state = TreeTCI2::<f64>::new(vec![2, 2, 2], graph).unwrap();
    /// let reversed = vec![TreeTciEdge::new(1, 2), TreeTciEdge::new(0, 1)];
    /// assert_eq!(Reversed.visit_order_in_pass(&state, 0), reversed);
    /// assert_eq!(Reversed.visit_order_in_pass(&state, 1), reversed);
    /// ```
    fn visit_order_in_pass<T>(&self, state: &TreeTCI2<T>, pass: usize) -> Vec<TreeTciEdge> {
        let _ = pass;
        self.visit_order(state)
    }
}

/// Visit all edges in canonical graph order, reversed on odd passes.
///
/// This is the default edge visitor. [`visit_order`](EdgeVisitor::visit_order)
/// returns the edges in sorted order; the optimizer's even inner passes use
/// that order and its odd passes the reverse
/// ([`visit_order_in_pass`](EdgeVisitor::visit_order_in_pass)), so the two
/// passes of an iteration sweep in opposite directions, like a forward and a
/// backward half-sweep of a chain.
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
///
/// // The second pass of an iteration goes back.
/// let backward = visitor.visit_order_in_pass(&state, 1);
/// assert_eq!(backward, vec![TreeTciEdge::new(1, 2), TreeTciEdge::new(0, 1)]);
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct AllEdges;

impl EdgeVisitor for AllEdges {
    fn visit_order<T>(&self, state: &TreeTCI2<T>) -> Vec<TreeTciEdge> {
        state.graph.edges()
    }

    fn visit_order_in_pass<T>(&self, state: &TreeTCI2<T>, pass: usize) -> Vec<TreeTciEdge> {
        let mut order = self.visit_order(state);
        if pass % 2 == 1 {
            order.reverse();
        }
        order
    }
}
