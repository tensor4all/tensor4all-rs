use super::*;
use std::collections::BTreeMap;
use tensor4all_tensorbackend::{explicit::Session, CpuBackend};
use tensor4all_treetn::interpolation::Parallelism;
use tensor4all_treetn::NodeNameNetwork;

/// Run `f` with an evaluation context over one caller-owned engine, so unit tests of
/// evaluator handling use the same explicit capability production runs use.
fn with_context<R>(f: impl FnOnce(&mut EvalContext<'_>) -> R) -> R {
    let mut backend = CpuBackend::with_threads(1).expect("1T CPU backend");
    f(&mut EvalContext::new(
        Session::new(backend.engine_mut()),
        Parallelism::Serial,
    ))
}

#[test]
fn map_termination_applies_the_cap_precedence() {
    let cap = NonZeroUsize::new(4);
    let cases = [
        // The criterion met strictly below the cap is accepted.
        (
            TreeTciTermination::Converged,
            3,
            cap,
            InterpolationTermination::Converged,
        ),
        (
            TreeTciTermination::Converged,
            9,
            None,
            InterpolationTermination::Converged,
        ),
        // The criterion met at a rank equal to the cap is not.
        (
            TreeTciTermination::Converged,
            4,
            cap,
            InterpolationTermination::BondCapReached,
        ),
        (
            TreeTciTermination::MaxBondDimension,
            4,
            cap,
            InterpolationTermination::BondCapReached,
        ),
        (
            TreeTciTermination::MaxIterations,
            2,
            cap,
            InterpolationTermination::IterationLimit,
        ),
        (
            TreeTciTermination::MaxIterations,
            2,
            None,
            InterpolationTermination::IterationLimit,
        ),
    ];
    for (reason, rank, cap, expected) in cases {
        assert_eq!(
            map_termination(reason, rank, cap),
            expected,
            "{reason:?} at rank {rank} with cap {cap:?}"
        );
    }
}

#[test]
fn classify_separates_evaluator_failures_from_engine_failures() {
    // Marker directly under TreeTciError::Operation: the original error is kept.
    let wrapped = TreeTciError::from(anyhow::Error::new(EvaluatorFailure {
        source: anyhow::anyhow!("user failure"),
    }));
    match classify(wrapped) {
        InterpolationError::Evaluator { source } => assert_eq!(source.to_string(), "user failure"),
        other => panic!("expected Evaluator, got {other:?}"),
    }

    // Marker under extra context: still an evaluator failure.
    let with_context = TreeTciError::from(
        anyhow::Error::new(EvaluatorFailure {
            source: anyhow::anyhow!("deep failure"),
        })
        .context("while filling a site tensor"),
    );
    match classify(with_context) {
        InterpolationError::Evaluator { source } => {
            assert!(format!("{source:#}").contains("deep failure"))
        }
        other => panic!("expected Evaluator, got {other:?}"),
    }

    // No marker: an engine failure.
    assert!(matches!(
        classify(TreeTciError::from(anyhow::anyhow!("singular pivot matrix"))),
        InterpolationError::Engine { .. }
    ));
    assert!(matches!(
        classify(TreeTciError::InvalidGraph {
            message: "cycle".to_string()
        }),
        InterpolationError::Engine { .. }
    ));
    assert!(matches!(
        classify_anyhow(anyhow::anyhow!("shape mismatch")),
        InterpolationError::Engine { .. }
    ));
}

/// Chain "a" - "b" - "c" (max degree 2): "a" has sites of dimensions 2 and 3
/// (fused, local dimension 6), "b" none, "c" one site of dimension 2.
fn fused_chain_problem() -> InterpolationProblem<String> {
    let mut topology = NodeNameNetwork::new();
    for node in ["a", "b", "c"] {
        topology.add_node(node.to_string()).unwrap();
    }
    topology
        .add_edge(&"a".to_string(), &"b".to_string())
        .unwrap();
    topology
        .add_edge(&"b".to_string(), &"c".to_string())
        .unwrap();
    let node_sites = BTreeMap::from([
        (
            "a".to_string(),
            vec![DynIndex::new_dyn(2), DynIndex::new_dyn(3)],
        ),
        ("b".to_string(), vec![]),
        ("c".to_string(), vec![DynIndex::new_dyn(2)]),
    ]);
    let pivots = ColMajorArray::new(vec![1, 2, 1], vec![3, 1]).unwrap();
    InterpolationProblem::new(topology, node_sites, pivots, 0.0, None, 0).unwrap()
}

#[test]
fn layout_fuses_column_major_and_round_trips() {
    let problem = fused_chain_problem();
    let layout = VertexLayout::new(&problem).unwrap();
    assert_eq!(layout.local_dims, vec![6, 1, 2]);
    assert_eq!(layout.site_offsets, vec![0, 2, 2, 3]);

    // (a0, a1) = (1, 2) fuses to 1 + 2 * 2 = 5; "b" has coordinate 0.
    let vertices = layout
        .site_columns_to_vertices(problem.initial_pivots())
        .unwrap();
    assert_eq!(vertices, vec![vec![5, 0, 1]]);

    // Every vertex point splits back to the site point it came from.
    let mut sites = vec![usize::MAX; 3];
    for fused in 0..6 {
        for c in 0..2 {
            layout.split_point(&[fused, 0, c], &mut sites).unwrap();
            assert_eq!(sites, vec![fused % 2, fused / 2, c]);
            let column = ColMajorArray::new(sites.clone(), vec![3, 1]).unwrap();
            assert_eq!(
                layout.site_columns_to_vertices(&column).unwrap(),
                vec![vec![fused, 0, c]]
            );
        }
    }

    // A batch of two vertex points becomes a [3, 2] site batch.
    let data = [5, 0, 1, 0, 0, 0];
    let batch = GlobalIndexBatch::new(&data, 3, 2).unwrap();
    assert_eq!(
        layout.vertex_batch_to_sites(batch).unwrap(),
        vec![1, 2, 1, 0, 0, 0]
    );
}

#[test]
fn layout_rejects_malformed_vertex_points() {
    let problem = fused_chain_problem();
    let layout = VertexLayout::new(&problem).unwrap();
    let mut sites = vec![0usize; 3];
    let error = layout.split_point(&[6, 0, 0], &mut sites).unwrap_err();
    assert!(error.to_string().contains("out of range"));
    let error = layout.split_point(&[0, 0], &mut sites).unwrap_err();
    assert!(error.to_string().contains("expected 3 and 3"));

    let data = [0, 0];
    let batch = GlobalIndexBatch::new(&data, 2, 1).unwrap();
    let error = layout.vertex_batch_to_sites(batch).unwrap_err();
    assert!(error.to_string().contains("expected 3"));
}

/// Chain "a" - "b" - "c" with one site per node of the given dimensions, or
/// no site where the dimension is `None`.
fn single_site_chain_problem(dims: [Option<usize>; 3]) -> InterpolationProblem<String> {
    let mut topology = NodeNameNetwork::new();
    for node in ["a", "b", "c"] {
        topology.add_node(node.to_string()).unwrap();
    }
    topology
        .add_edge(&"a".to_string(), &"b".to_string())
        .unwrap();
    topology
        .add_edge(&"b".to_string(), &"c".to_string())
        .unwrap();
    let node_sites: BTreeMap<String, Vec<DynIndex>> = ["a", "b", "c"]
        .into_iter()
        .zip(dims)
        .map(|(node, dim)| {
            (
                node.to_string(),
                dim.map(DynIndex::new_dyn).into_iter().collect(),
            )
        })
        .collect();
    let n_sites = dims.iter().flatten().count();
    let pivots = ColMajorArray::new(vec![0; n_sites], vec![n_sites, 1]).unwrap();
    InterpolationProblem::new(topology, node_sites, pivots, 0.0, None, 0).unwrap()
}

#[test]
fn layout_passes_one_site_vertices_through_without_a_copy() {
    let problem = single_site_chain_problem([Some(2), Some(3), Some(2)]);
    let layout = VertexLayout::new(&problem).unwrap();
    assert_eq!(layout.site_vertices, Some(vec![0, 1, 2]));

    let data = [1, 2, 0, 0, 0, 1];
    let batch = GlobalIndexBatch::new(&data, 3, 2).unwrap();
    let sites = layout.vertex_batch_to_sites(batch).unwrap();
    assert!(matches!(sites, Cow::Borrowed(_)));
    assert_eq!(sites, vec![1, 2, 0, 0, 0, 1]);
    // The general path gives the same rows.
    let mut split = vec![0usize; 6];
    for (vertices, out) in data.chunks(3).zip(split.chunks_mut(3)) {
        layout.split_point(vertices, out).unwrap();
    }
    assert_eq!(*sites, *split);

    // Malformed batches still fail: a coordinate out of range anywhere in
    // the batch, and a wrong vertex count.
    let data = [1, 2, 0, 0, 3, 1];
    let batch = GlobalIndexBatch::new(&data, 3, 2).unwrap();
    let error = layout.vertex_batch_to_sites(batch).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("vertex 1 coordinate 3 is out of range for local dimension 3"),
        "{error}"
    );
    let data = [0, 0];
    let batch = GlobalIndexBatch::new(&data, 2, 1).unwrap();
    let error = layout.vertex_batch_to_sites(batch).unwrap_err();
    assert!(error.to_string().contains("expected 3"), "{error}");
}

#[test]
fn layout_gathers_site_rows_past_site_free_vertices() {
    let problem = single_site_chain_problem([Some(2), None, Some(3)]);
    let layout = VertexLayout::new(&problem).unwrap();
    assert_eq!(layout.local_dims, vec![2, 1, 3]);
    assert_eq!(layout.site_vertices, Some(vec![0, 2]));

    let data = [1, 0, 2, 0, 0, 1];
    let batch = GlobalIndexBatch::new(&data, 3, 2).unwrap();
    let sites = layout.vertex_batch_to_sites(batch).unwrap();
    assert!(matches!(sites, Cow::Owned(_)));
    assert_eq!(sites, vec![1, 2, 0, 1]);
    let mut split = vec![0usize; 4];
    for (vertices, out) in data.chunks(3).zip(split.chunks_mut(2)) {
        layout.split_point(vertices, out).unwrap();
    }
    assert_eq!(*sites, *split);

    // The site-free vertex must have coordinate 0.
    let data = [1, 1, 2];
    let batch = GlobalIndexBatch::new(&data, 3, 1).unwrap();
    let error = layout.vertex_batch_to_sites(batch).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("vertex 1 coordinate 1 is out of range for local dimension 1"),
        "{error}"
    );
}

#[test]
fn layout_with_a_fused_vertex_takes_the_general_path() {
    let problem = fused_chain_problem();
    let layout = VertexLayout::new(&problem).unwrap();
    assert_eq!(layout.site_vertices, None);
    let data = [5, 0, 1];
    let batch = GlobalIndexBatch::new(&data, 3, 1).unwrap();
    let sites = layout.vertex_batch_to_sites(batch).unwrap();
    assert!(matches!(sites, Cow::Owned(_)));
    assert_eq!(sites, vec![1, 2, 1]);
    let data = [6, 0, 1];
    let batch = GlobalIndexBatch::new(&data, 3, 1).unwrap();
    let error = layout.vertex_batch_to_sites(batch).unwrap_err();
    assert!(error.to_string().contains("out of range"), "{error}");
}

#[test]
fn call_evaluator_marks_errors_and_wrong_lengths() {
    with_context(|ctx| {
        let data = [0usize, 1, 1, 0];
        let mut short = |_: ColMajorArrayRef<'_, usize>,
                         _: &mut EvalContext<'_>|
         -> anyhow::Result<Vec<f64>> { Ok(vec![1.0]) };
        let error = call_evaluator(&mut short, &data, 2, ctx).unwrap_err();
        assert!(has_evaluator_failure(error.as_ref()));
        assert!(format!("{error:#}").contains("returned 1 values for 2 points"));

        let mut failing = |_: ColMajorArrayRef<'_, usize>,
                           _: &mut EvalContext<'_>|
         -> anyhow::Result<Vec<f64>> { anyhow::bail!("nope") };
        let error = call_evaluator(&mut failing, &data, 2, ctx).unwrap_err();
        assert!(has_evaluator_failure(error.as_ref()));

        let mut echo = |batch: ColMajorArrayRef<'_, usize>,
                        _: &mut EvalContext<'_>|
         -> anyhow::Result<Vec<f64>> {
            assert_eq!(batch.shape(), &[2, 2]);
            Ok(batch
                .data()
                .chunks(2)
                .map(|p| (p[0] * 10 + p[1]) as f64)
                .collect())
        };
        assert_eq!(
            call_evaluator(&mut echo, &data, 2, ctx).unwrap(),
            vec![1.0, 10.0]
        );
    });
}

#[test]
fn call_evaluator_rejects_non_finite_complex_components_and_magnitudes() {
    use num_complex::Complex64;
    with_context(|ctx| {
        for bad in [
            Complex64::new(f64::NAN, 1.0),
            Complex64::new(1.0, f64::INFINITY),
            Complex64::new(f64::MAX, f64::MAX),
        ] {
            let mut evaluate = |_: ColMajorArrayRef<'_, usize>,
                                _: &mut EvalContext<'_>|
             -> anyhow::Result<Vec<Complex64>> {
                Ok(vec![Complex64::new(1.0, 0.0), bad])
            };
            let error = call_evaluator(&mut evaluate, &[0, 1], 1, ctx).unwrap_err();
            assert!(has_evaluator_failure(error.as_ref()));
            let message = format!("{error:#}");
            assert!(
                message.contains("batch point 1, coordinates [1]"),
                "{message}"
            );
        }
    });
}

// ---------------------------------------------------------------------------
// Problem-supplied global search starts (issue #1033).
// ---------------------------------------------------------------------------

/// Chain of six binary nodes, one binary site per node.
const CORNER_NODES: usize = 6;

/// f = 1 + 10 * delta at the all-ones corner: a flat field except one corner.
fn corner_function(point: &[usize]) -> f64 {
    if point.iter().all(|&coordinate| coordinate == 1) {
        11.0
    } else {
        1.0
    }
}

fn corner_evaluate(
    batch: ColMajorArrayRef<'_, usize>,
    _ctx: &mut EvalContext<'_>,
) -> anyhow::Result<Vec<f64>> {
    Ok(batch
        .data()
        .chunks(batch.shape()[0])
        .map(corner_function)
        .collect())
}

/// The chain problem with its only initial pivot at the all-zero corner and,
/// when given, extra global search starts in site order.
fn corner_chain_problem(starts: Option<ColMajorArray<usize>>) -> InterpolationProblem<usize> {
    let mut topology = NodeNameNetwork::new();
    for node in 0..CORNER_NODES {
        topology.add_node(node).unwrap();
    }
    for node in 0..CORNER_NODES - 1 {
        topology.add_edge(&node, &(node + 1)).unwrap();
    }
    let node_sites = (0..CORNER_NODES)
        .map(|node| (node, vec![DynIndex::new_dyn(2)]))
        .collect();
    let pivots = ColMajorArray::new(vec![0; CORNER_NODES], vec![CORNER_NODES, 1]).unwrap();
    // Seed 34: the ten random starts of each of the first three searches keep
    // at least two zero coordinates, so the corner stays unreachable for the
    // unstarted run below.
    let problem = InterpolationProblem::new(topology, node_sites, pivots, 1e-10, None, 34).unwrap();
    match starts {
        Some(starts) => problem.with_global_search_starts(starts).unwrap(),
        None => problem,
    }
}

#[test]
fn interpolator_forwards_problem_global_search_starts() {
    let corner = vec![1usize; CORNER_NODES];
    with_context(|ctx| {
        // Without a start, the seeded random search misses the corner and the
        // run converges to the constant field 1, which is the initial pivot's
        // cross-section everywhere.
        let problem = corner_chain_problem(None);
        let without = TreeTciInterpolator::default()
            .interpolate(&problem, &mut *ctx, corner_evaluate)
            .unwrap();
        assert_eq!(without.termination, InterpolationTermination::Converged);
        let value = without
            .network
            .evaluate_point(problem.site_order(), &corner)
            .unwrap()
            .real();
        assert!(
            (value - 1.0).abs() < 1e-12,
            "the corner feature should be missed without a start, got {value}"
        );

        // Forwarding the problem's start through the run options lets the
        // engine's global search find the corner, and the rank-2 result is
        // exact.
        let starts = ColMajorArray::new(corner.clone(), vec![CORNER_NODES, 1]).unwrap();
        let problem = corner_chain_problem(Some(starts));
        let with = TreeTciInterpolator::default()
            .interpolate(&problem, &mut *ctx, corner_evaluate)
            .unwrap();
        assert_eq!(with.termination, InterpolationTermination::Converged);
        let value = with
            .network
            .evaluate_point(problem.site_order(), &corner)
            .unwrap()
            .real();
        assert!(
            (value - 11.0).abs() < 1e-8,
            "the forwarded start should recover the corner, got {value}"
        );
        let interior = [0, 1, 0, 1, 0, 1];
        let value = with
            .network
            .evaluate_point(problem.site_order(), &interior)
            .unwrap()
            .real();
        assert!(
            (value - 1.0).abs() < 1e-8,
            "a non-corner point should stay at 1, got {value}"
        );
    });
}
