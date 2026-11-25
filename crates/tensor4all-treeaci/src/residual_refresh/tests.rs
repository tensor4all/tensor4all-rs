use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_treetn::TreeTN;

use super::refresh_stale_residuals;
use crate::{
    local_update::enumerate_candidates,
    schedule::{run_directional_pass, PassDirection},
    state::TreeAciState,
    test_support::{branched_product_inputs, product_of_two, BRANCHED_EDGES},
    TreeAciError, TreeAciOptions,
};

const EDGES: [(usize, usize); 6] = BRANCHED_EDGES;

fn inputs() -> (Vec<DynIndex>, Vec<TreeTN<IdxTensor, usize>>) {
    branched_product_inputs()
}

fn product(batch: crate::TreeElementwiseBatch<'_, f64>, output: &mut [f64]) -> crate::Result<()> {
    product_of_two(batch, output)
}

fn options() -> TreeAciOptions<usize> {
    TreeAciOptions {
        tolerance: 1.0e-3,
        scale_tolerance: false,
        max_bond_dim: Some(3),
        ..TreeAciOptions::default()
    }
}

/// Dense values of `tree`, indexed by `sum_node coordinate[node] * 2^node`.
fn dense(tree: &TreeTN<IdxTensor, usize>, sites: &[DynIndex]) -> Vec<f64> {
    tree.to_dense()
        .unwrap()
        .permute_indices(sites)
        .unwrap()
        .to_vec::<f64>()
        .unwrap()
}

/// `(max |f - output|, max |f|, entries)` over the current local matrix of
/// `edge_number`, from dense tensors and independently materialized points.
fn brute_force_residual(
    state: &TreeAciState<'_, f64, usize>,
    sites: &[DynIndex],
    edge_number: usize,
) -> (f64, f64, usize) {
    let forward = 2 * edge_number;
    let reverse = state.problem.directed_edges[forward].reverse;
    let rows = enumerate_candidates(
        &state.problem,
        &state.candidates,
        forward,
        "rows",
        usize::MAX,
    )
    .unwrap();
    let cols = enumerate_candidates(
        &state.problem,
        &state.candidates,
        reverse,
        "cols",
        usize::MAX,
    )
    .unwrap();
    let mut arena = state.sample_arena.clone();
    let row_ids = rows
        .into_iter()
        .map(|sample| arena.intern_component(&state.problem, forward, sample))
        .collect::<crate::Result<Vec<_>>>()
        .unwrap();
    let col_ids = cols
        .into_iter()
        .map(|sample| arena.intern_component(&state.problem, reverse, sample))
        .collect::<crate::Result<Vec<_>>>()
        .unwrap();
    let a = dense(&state.inputs[0], sites);
    let b = dense(&state.inputs[1], sites);
    let output = dense(&state.output, sites);
    let (mut error, mut scale) = (0.0_f64, 0.0_f64);
    for &row in &row_ids {
        for &col in &col_ids {
            let point = arena
                .materialize_global_point(&state.problem, forward, row, col)
                .unwrap();
            let flat = point
                .iter()
                .enumerate()
                .map(|(position, &coordinate)| coordinate << state.problem.node_order[position])
                .sum::<usize>();
            let target = a[flat] * b[flat];
            error = error.max((target - output[flat]).abs());
            scale = scale.max(target.abs());
        }
    }
    (error, scale, row_ids.len() * col_ids.len())
}

#[test]
fn refresh_remeasures_exactly_the_stale_edges_on_the_current_output() {
    let (sites, inputs) = inputs();
    let options = options();
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    for direction in [PassDirection::Forward, PassDirection::Reverse] {
        let report = run_directional_pass(
            &mut state,
            &options,
            direction,
            false,
            &mut product,
            &mut crate::exec::Exec::Global,
        )
        .unwrap();
        // The walk continues from each commit's target, so only the edge of
        // the pass's last commit holds a residual of the current output.
        let last = report.updated_edges.last().copied().unwrap() / 2;
        for edge in 0..EDGES.len() {
            assert_eq!(state.residual_is_stale(edge).unwrap(), edge != last);
        }
    }
    let fresh = (0..EDGES.len())
        .find(|&edge| !state.residual_is_stale(edge).unwrap())
        .unwrap();
    let fresh_record = (state.edge_errors[fresh], state.edge_scales[fresh]);
    let expected = (0..EDGES.len())
        .map(|edge| brute_force_residual(&state, &sites, edge))
        .collect::<Vec<_>>();

    let report = refresh_stale_residuals(
        &mut state,
        &options,
        &mut product,
        &mut crate::exec::Exec::Global,
    )
    .unwrap();

    assert_eq!(report.refreshed_edges, EDGES.len() - 1);
    let expected_points = (0..EDGES.len())
        .filter(|&edge| edge != fresh)
        .map(|edge| expected[edge].2 as u64)
        .sum::<u64>();
    assert_eq!(report.evaluated_points, expected_points);
    // The fixture needs more rank than the cap of 3 allows, so the output is
    // inexact and some refreshed residual exceeds the tolerance.
    let mut largest: f64 = 0.0;
    for (edge, &(error, scale, _)) in expected.iter().enumerate() {
        if edge == fresh {
            assert_eq!(
                (state.edge_errors[edge], state.edge_scales[edge]),
                fresh_record
            );
            continue;
        }
        assert!((state.edge_errors[edge] - error).abs() <= 1.0e-12 * scale.max(1.0));
        assert!((state.edge_scales[edge] - scale).abs() <= 1.0e-12 * scale.max(1.0));
        largest = largest.max(error);
    }
    assert!(
        largest > options.tolerance,
        "largest refreshed residual {largest:e}"
    );
    for edge in 0..EDGES.len() {
        assert!(!state.residual_is_stale(edge).unwrap());
    }

    // Nothing is stale any more: a second refresh evaluates nothing.
    let again = refresh_stale_residuals(
        &mut state,
        &options,
        &mut product,
        &mut crate::exec::Exec::Global,
    )
    .unwrap();
    assert_eq!(again.refreshed_edges, 0);
    assert_eq!(again.evaluated_points, 0);
}

fn state_after_forward_pass<'a>(
    inputs: &'a [TreeTN<IdxTensor, usize>],
    options: &TreeAciOptions<usize>,
) -> TreeAciState<'a, f64, usize> {
    let mut state = TreeAciState::<f64, usize>::initialize(inputs, options).unwrap();
    run_directional_pass(
        &mut state,
        options,
        PassDirection::Forward,
        false,
        &mut product,
        &mut crate::exec::Exec::Global,
    )
    .unwrap();
    state
}

#[test]
fn refused_refresh_leaves_every_residual_as_recorded() {
    let (_, inputs) = inputs();
    let options = options();
    let mut state = state_after_forward_pass(&inputs, &options);
    let before = (
        state.edge_errors.clone(),
        state.edge_scales.clone(),
        state.residual_revisions.clone(),
    );
    let tight = TreeAciOptions {
        max_working_bytes: 64,
        ..options
    };

    let error = refresh_stale_residuals(
        &mut state,
        &tight,
        &mut product,
        &mut crate::exec::Exec::Global,
    )
    .unwrap_err();

    assert!(matches!(
        error,
        TreeAciError::ResourceLimit {
            resource: "working bytes",
            ..
        }
    ));
    assert_eq!(
        (
            state.edge_errors.clone(),
            state.edge_scales.clone(),
            state.residual_revisions.clone()
        ),
        before
    );
}

#[test]
fn non_finite_refreshed_residual_is_an_error_and_records_nothing() {
    let (_, inputs) = inputs();
    let options = options();
    let mut state = state_after_forward_pass(&inputs, &options);
    let before = (state.edge_errors.clone(), state.residual_revisions.clone());
    let mut not_a_number = |_: crate::TreeElementwiseBatch<'_, f64>, output: &mut [f64]| {
        output.fill(f64::NAN);
        Ok(())
    };

    let error = refresh_stale_residuals(
        &mut state,
        &options,
        &mut not_a_number,
        &mut crate::exec::Exec::Global,
    )
    .unwrap_err();

    assert!(matches!(
        error,
        TreeAciError::NonFiniteValue {
            context: "refreshed edge residual"
        }
    ));
    assert_eq!(
        (state.edge_errors.clone(), state.residual_revisions.clone()),
        before
    );
}
