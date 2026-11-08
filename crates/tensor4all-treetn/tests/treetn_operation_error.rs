//! Each typed variant of `TreeTNOperationError` is reached by a real path (#1023).
//!
//! The enum mirrors `TreeTciError`: a variant for each failure a caller can act
//! on, plus `Operation` as the catch-all. These tests pin the variant each path
//! reports, so a path that stops classifying fails here instead of silently
//! degrading to the catch-all.

use std::error::Error as _;

use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor};
use tensor4all_treetn::{
    fix_and_remove_site, CachedEvaluatorOptions, EvaluationHint, NamedGraph, TreeTN,
    TreeTNCachedEvaluator, TreeTNOperationError,
};

fn one_site_f64_tree() -> (TreeTN<IdxTensor, usize>, DynIndex) {
    let site = DynIndex::new_dyn(2);
    let tree = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![site.clone()], vec![1.0_f64, 2.0]).unwrap()],
        vec![0usize],
    )
    .unwrap();
    (tree, site)
}

#[test]
fn invalid_configuration_covers_invalid_argument_and_option_values() {
    // Too few terms: an argument value the caller chose.
    let error = TreeTN::<IdxTensor, usize>::add_many(&[]).unwrap_err();
    assert!(
        matches!(error, TreeTNOperationError::InvalidConfiguration { .. }),
        "an empty term list must be an invalid configuration, got {error:?}"
    );

    // A non-positive batch limit is an invalid option value.
    let (tree, site) = one_site_f64_tree();
    let error = match TreeTNCachedEvaluator::new(
        &tree,
        &[site],
        CachedEvaluatorOptions {
            max_batch_points: Some(0),
            ..Default::default()
        },
    ) {
        Err(error) => error,
        Ok(_) => panic!("a zero batch limit must be rejected"),
    };
    assert!(
        matches!(error, TreeTNOperationError::InvalidConfiguration { .. }),
        "a zero batch limit must be an invalid configuration, got {error:?}"
    );
}

#[test]
fn invalid_graph_covers_unknown_and_duplicate_node_names() {
    let (mut tree, _site) = one_site_f64_tree();
    let error = tree.set_canonical_region([404usize]).unwrap_err();
    assert!(
        matches!(error, TreeTNOperationError::InvalidGraph { .. }),
        "an unknown node must be an invalid graph, got {error:?}"
    );
    // The offending name is still readable through the source chain.
    assert!(error.to_string().contains("does not exist"), "{error}");
    assert!(error.source().is_some());

    // A duplicate node name is the same class: the network cannot serve it.
    let mut graph: NamedGraph<usize, (), ()> = NamedGraph::new();
    graph.add_node(0usize, ()).unwrap();
    let error = graph.add_node(0usize, ()).unwrap_err();
    assert!(
        matches!(error, TreeTNOperationError::InvalidGraph { .. }),
        "an existing node name must be an invalid graph, got {error:?}"
    );
}

#[test]
fn index_out_of_bounds_covers_out_of_range_coordinates() {
    // A coordinate outside the site dimension: the canonical out-of-range case.
    let (tree, _site) = one_site_f64_tree();
    let error = fix_and_remove_site(tree, &0usize, 2).unwrap_err();
    assert!(
        matches!(error, TreeTNOperationError::IndexOutOfBounds { .. }),
        "a coordinate outside the site dimension must be out of bounds, got {error:?}"
    );

    // A coordinate list outside the index list is the same class.
    let (tree, site) = one_site_f64_tree();
    let error = tree
        .evaluate_point(std::slice::from_ref(&site), &[])
        .unwrap_err();
    assert!(
        matches!(error, TreeTNOperationError::IndexOutOfBounds { .. }),
        "a coordinate count outside the index count must be out of bounds, got {error:?}"
    );

    // The count is the only coordinate mistake `evaluate_point` classifies: a
    // correctly sized list holding an out-of-range value still reaches the
    // one-hot construction and keeps the catch-all. This pins the current
    // boundary, so a change that starts classifying it must update this test.
    let (tree, site) = one_site_f64_tree();
    let error = tree.evaluate_point(&[site], &[2]).unwrap_err();
    assert!(
        matches!(error, TreeTNOperationError::Operation { .. }),
        "an out-of-range coordinate value is not classified at this boundary, got {error:?}"
    );
    assert!(
        source_chain(&error)
            .last()
            .is_some_and(|link: &String| link.contains("onehot")),
        "the catch-all must keep the one-hot diagnostic, got {:?}",
        source_chain(&error)
    );
}

#[test]
fn scalar_kind_mismatch_covers_stored_dtype_refusals() {
    let site = DynIndex::new_dyn(2);
    let tree = TreeTN::<_, usize>::from_tensors(
        vec![IdxTensor::from_dense(
            vec![site.clone()],
            vec![
                num_complex::Complex64::new(1.0, 2.0),
                num_complex::Complex64::new(3.0, 4.0),
            ],
        )
        .unwrap()],
        vec![0],
    )
    .unwrap();
    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &[site],
        CachedEvaluatorOptions {
            center: Some(0),
            ..Default::default()
        },
    )
    .unwrap();
    let points = ColMajorArrayRef::new(&[0usize, 1usize], &[1, 2]).unwrap();
    let error = evaluator
        .evaluate_batched_typed::<f64>(points, EvaluationHint::default())
        .unwrap_err();
    assert_eq!(
        source_chain(&error),
        vec!["cannot decode a c64 TreeTN value as f64"],
        "the mismatch payload must be the source link, not a None from transparent forwarding"
    );
    assert_eq!(error.to_string(), "cannot decode a c64 TreeTN value as f64");
    let TreeTNOperationError::ScalarKindMismatch(mismatch) = error else {
        panic!("a complex payload must not answer a real request, got {error:?}");
    };
    assert_eq!((mismatch.stored, mismatch.requested), ("c64", "f64"));
}

fn source_chain(error: &TreeTNOperationError) -> Vec<String> {
    let mut chain = Vec::new();
    let mut current = error.source();
    while let Some(link) = current {
        chain.push(link.to_string());
        current = link.source();
    }
    chain
}

#[test]
fn operation_stays_the_catch_all_and_renders_the_whole_chain() {
    // Any failure this boundary cannot classify keeps the catch-all and the
    // full source chain, which is what `From<anyhow::Error>` promises. The
    // cause is wrapped in a context on purpose: the alternate (`{source:#}`)
    // form is the only one that renders the root as well, so a formatter that
    // drops it loses the outermost classification *and* the root text.
    let error = TreeTNOperationError::from(anyhow::anyhow!("root cause").context("outer context"));
    assert!(
        matches!(error, TreeTNOperationError::Operation { .. }),
        "an unclassified failure must stay in the catch-all, got {error:?}"
    );
    assert_eq!(
        error.to_string(),
        "TreeTN operation failed: outer context: root cause",
        "the catch-all must render the context and the root, not only the context"
    );
    assert_eq!(source_chain(&error), vec!["outer context", "root cause"]);
}
