//! The optimizer asks the visitor for each inner pass's order (#899).
//!
//! `optimize_with_proposer` calls `EdgeVisitor::visit_order_in_pass` with the
//! pass index of every inner pass, and with `AllEdges` the second pass of an
//! iteration visits the sorted edges backward.

use std::cell::RefCell;

use tensor4all_treetci::{
    optimize_with_proposer, AllEdges, DefaultProposer, EdgeVisitor, GlobalIndexBatch, TreeTCI2,
    TreeTciEdge, TreeTciGraph, TreeTciOptions,
};

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

/// Delegates to `AllEdges` and records every pass index and order it returns.
#[derive(Default)]
struct Recording {
    calls: RefCell<Vec<(usize, Vec<TreeTciEdge>)>>,
}

impl EdgeVisitor for Recording {
    fn visit_order<T>(&self, state: &TreeTCI2<T>) -> Vec<TreeTciEdge> {
        AllEdges.visit_order(state)
    }

    fn visit_order_in_pass<T>(&self, state: &TreeTCI2<T>, pass: usize) -> Vec<TreeTciEdge> {
        let order = AllEdges.visit_order_in_pass(state, pass);
        self.calls.borrow_mut().push((pass, order.clone()));
        order
    }
}

fn initial_state() -> TestResult<TreeTCI2<f64>> {
    let graph = TreeTciGraph::linear_chain(4)?;
    let mut state = TreeTCI2::<f64>::new(vec![2; 4], graph)?;
    state.add_global_pivots(&[vec![0; 4]])?;
    Ok(state)
}

fn evaluate(batch: GlobalIndexBatch<'_>) -> anyhow::Result<Vec<f64>> {
    Ok((0..batch.n_points())
        .map(|point| {
            (0..4)
                .map(|site| batch.get(site, point).unwrap_or(0) as f64 * (site as f64 + 1.0))
                .sum::<f64>()
                .cos()
                + 2.0
        })
        .collect())
}

fn options() -> TreeTciOptions {
    TreeTciOptions {
        tolerance: 1e-14,
        max_iter: 2,
        enable_global_pivots: false,
        ..TreeTciOptions::default()
    }
}

#[test]
fn optimizer_requests_each_pass_and_all_edges_alternates() -> TestResult {
    let visitor = Recording::default();
    let mut state = initial_state()?;
    let recorded = optimize_with_proposer(
        &mut state,
        evaluate,
        &options(),
        &DefaultProposer,
        &visitor,
        &mut (),
    )?;

    let forward = vec![
        TreeTciEdge::new(0, 1),
        TreeTciEdge::new(1, 2),
        TreeTciEdge::new(2, 3),
    ];
    let backward: Vec<TreeTciEdge> = forward.iter().rev().copied().collect();
    let calls = visitor.calls.into_inner();
    let expected = vec![
        (0, forward.clone()),
        (1, backward.clone()),
        (0, forward),
        (1, backward),
    ];
    assert_eq!(calls, expected);

    // Recording only observes: the run is the one `AllEdges` gives.
    let mut plain_state = initial_state()?;
    let plain = optimize_with_proposer(
        &mut plain_state,
        evaluate,
        &options(),
        &DefaultProposer,
        &AllEdges,
        &mut (),
    )?;
    assert_eq!(recorded.ranks, plain.ranks);
    assert_eq!(recorded.errors, plain.errors);
    Ok(())
}
