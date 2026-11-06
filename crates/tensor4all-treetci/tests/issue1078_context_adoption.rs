//! #1078 acceptance: a tree the caller did not build shortens in a context.
//!
//! Reported downstream in #859: a 4-site chain materialised by TCI is refused
//! by the context-scoped shortening on the *same* context, because a TCI
//! materialisation publishes runtime-free compact host storage. These tests
//! hold the result to the context-free result, not merely to "no error".

use std::sync::Arc;

use tensor4all_core::{ExecutionContext, IdxTensor};
use tensor4all_tensorbackend::{CpuBackend, CpuExecutionContext};
use tensor4all_treetci::{
    optimize_with_proposer_in, to_treetn, to_treetn_in, DefaultProposer, GlobalIndexBatch,
    TreeTCI2, TreeTciGraph, TreeTciOptions,
};
use tensor4all_treetn::{TreeTN, TruncationOptions};

const SITES: usize = 4;

fn oracle(batch: GlobalIndexBatch<'_>) -> anyhow::Result<Vec<f64>> {
    Ok((0..batch.n_points())
        .map(|point| {
            (0..SITES).fold(1.0, |total, site| {
                total + batch.get(site, point).expect("site in batch") as f64 * 0.5 + 0.3
            })
        })
        .collect())
}

fn cpu() -> Arc<CpuExecutionContext> {
    Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::with_threads(2).expect("CPU backend"),
    ))
}

/// A converged 4-site TCI state with `max_bond_dim = 2`, as in the report.
fn converged_state(context: &CpuExecutionContext) -> TreeTCI2<f64> {
    let graph = TreeTciGraph::linear_chain(SITES).expect("graph");
    let mut state = TreeTCI2::<f64>::new(vec![2; SITES], graph).expect("state");
    state
        .add_global_pivots(&[vec![0usize; SITES]])
        .expect("pivot");
    let options = TreeTciOptions {
        seed: Some(20_261_101),
        tolerance: 1e-10,
        max_iter: 8,
        max_bond_dim: Some(2),
        enable_global_pivots: true,
        nsearch: 2,
        max_nglobal_pivot: 2,
        ..TreeTciOptions::default()
    };
    optimize_with_proposer_in(
        &mut state,
        oracle,
        &options,
        &DefaultProposer,
        &mut (),
        context,
    )
    .expect("optimize");
    assert_eq!(
        state.max_bond_dim(),
        2,
        "the reported tree carries bond dimension 2"
    );
    state
}

fn options(max_bond_dim: usize) -> TruncationOptions {
    TruncationOptions::default().with_max_bond_dim(max_bond_dim)
}

fn assert_same_values(left: &TreeTN<IdxTensor, usize>, right: &TreeTN<IdxTensor, usize>) {
    let left = left.to_dense().expect("dense left");
    let right = right.to_dense().expect("dense right");
    assert!(
        left.isapprox(&right, 1e-12, 1e-12).expect("compare"),
        "the two dense readouts disagree"
    );
}

#[test]
fn to_treetn_in_then_truncate_in_on_one_context_matches_the_context_free_result() {
    let cpu = cpu();
    let context = ExecutionContext::Cpu(Arc::clone(&cpu));
    let state = converged_state(&cpu);

    let tree = to_treetn_in(&state, oracle, None, &cpu).expect("materialize in the context");
    let in_context = tree
        .clone()
        .truncate_in([0], options(1), &context)
        .expect("context-scoped truncation of the materialised tree");
    let plain = tree
        .truncate([0], options(1))
        .expect("context-free truncation");

    assert_eq!(in_context.link_dims(), vec![1; SITES - 1]);
    assert_eq!(in_context.link_dims(), plain.link_dims());
    assert_same_values(&in_context, &plain);
}

#[test]
fn to_treetn_global_then_truncate_in_adopts_the_implicit_tensors() {
    let cpu = cpu();
    let context = ExecutionContext::Cpu(Arc::clone(&cpu));
    let state = converged_state(&cpu);

    let tree = to_treetn(&state, oracle, None).expect("materialize on the global route");
    let in_context = tree
        .clone()
        .truncate_in([0], options(1), &context)
        .expect("context-scoped truncation of a globally materialised tree");
    let plain = tree
        .truncate([0], options(1))
        .expect("context-free truncation");

    assert_eq!(in_context.link_dims(), vec![1; SITES - 1]);
    assert_eq!(in_context.link_dims(), plain.link_dims());
    assert_same_values(&in_context, &plain);
}
