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
    let TreeTNOperationError::ScalarKindMismatch {
        payload: mismatch, ..
    } = error
    else {
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

#[test]
fn a_classified_error_used_as_context_does_not_replace_the_real_cause() {
    // `anyhow::downcast` also returns a `TreeTNOperationError` attached with
    // `.context(..)`. Taking it would drop the actual failure and rebuild the message around
    // an unrelated classification, so only the error itself may be classified.
    let error = TreeTNOperationError::from(
        anyhow::Error::new(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "missing input",
        ))
        .context(TreeTNOperationError::InvalidGraph {
            source: anyhow::anyhow!("graph request"),
        }),
    );
    let TreeTNOperationError::Operation { source } = &error else {
        panic!("a context-provided classification must not be taken, got {error:?}");
    };
    assert!(
        source.downcast_ref::<std::io::Error>().is_some(),
        "the real cause must stay typed and reachable, got {error:?}"
    );
}

#[test]
fn a_typed_note_keeps_its_message_without_nesting_the_catch_all() {
    // A layer attached with `.context(..)` is a note about the call, so its
    // message is what survives (as it already does for every other variant).
    // The failure's own payload stays as the source chain, and the note is not
    // wrapped around it as a second `Operation`, which is what made the chain
    // grow on every pass through the boundary.
    let wrapped = anyhow::Error::new(TreeTNOperationError::Operation {
        source: anyhow::anyhow!("root"),
    })
    .context(std::io::Error::new(
        std::io::ErrorKind::TimedOut,
        "retry later",
    ));
    let error = TreeTNOperationError::from(wrapped);
    assert!(
        matches!(error, TreeTNOperationError::Operation { .. }),
        "an unclassified failure stays in the catch-all, got {error:?}"
    );
    assert_eq!(source_chain(&error), vec!["retry later", "root"]);
}

#[test]
fn a_typed_context_does_not_replace_the_propagated_classification() {
    // Both the failure and the layer attached with `.context(..)` are classified.
    // The failure is the error object; a typed context is a note about the call,
    // so taking it as the variant would report the note as the failure.
    let wrapped = anyhow::Error::new(TreeTNOperationError::InvalidGraph {
        source: anyhow::anyhow!("real root"),
    })
    .context(TreeTNOperationError::InvalidConfiguration {
        source: anyhow::anyhow!("context note"),
    });
    let error = TreeTNOperationError::from(wrapped);
    assert!(
        matches!(error, TreeTNOperationError::InvalidGraph { .. }),
        "a typed context must not replace the propagated classification, got {error:?}"
    );
    let text = format!("{error:#}");
    assert!(
        text.contains("real root"),
        "the failure was dropped: {text}"
    );
    assert!(
        text.contains("context note"),
        "the context was dropped: {text}"
    );
}

#[test]
fn a_typed_context_above_the_failure_keeps_the_failure_chain() {
    // The note above the failure keeps `anyhow` from handing the failure back by
    // value, so the failure is only reachable by reference. Its own payload must
    // still be there afterwards: a typed source may not be flattened to text.
    let wrapped = anyhow::Error::new(TreeTNOperationError::InvalidGraph {
        source: anyhow::Error::new(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "missing input",
        )),
    })
    .context(TreeTNOperationError::InvalidConfiguration {
        source: anyhow::anyhow!("context note"),
    });
    let error = TreeTNOperationError::from(wrapped);
    assert!(
        matches!(error, TreeTNOperationError::InvalidGraph { .. }),
        "the failure keeps its variant, got {error:?}"
    );
    let TreeTNOperationError::InvalidGraph { source } = &error else {
        unreachable!("checked above")
    };
    assert!(
        source
            .chain()
            .any(|link| link.downcast_ref::<std::io::Error>().is_some()),
        "the failure's typed source must survive, got {error:?}"
    );
}

#[test]
fn a_classified_error_below_an_unclassified_wrapper_is_not_reclassified() {
    // The wrapper is what the boundary saw; the classified error below it is not
    // a note about the call, so promoting it would drop the wrapper's own
    // message and repeat the payload it already renders.
    #[derive(Debug)]
    struct Wrapper(TreeTNOperationError);

    impl std::fmt::Display for Wrapper {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "wrapper: {}", self.0)
        }
    }

    impl std::error::Error for Wrapper {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            Some(&self.0)
        }
    }

    let error = TreeTNOperationError::from(anyhow::Error::new(Wrapper(
        TreeTNOperationError::InvalidGraph {
            source: anyhow::anyhow!("real root"),
        },
    )));
    assert!(
        matches!(error, TreeTNOperationError::Operation { .. }),
        "an unclassified wrapper stays the catch-all, got {error:?}"
    );
    assert_eq!(
        source_chain(&error).first().map(String::as_str),
        Some("wrapper: invalid TreeTN graph: real root"),
        "the wrapper's own message is not repeated, got {:?}",
        source_chain(&error)
    );
}

#[test]
fn a_boundary_does_not_rewrap_the_chain_it_already_holds() {
    // `Operation` already carries the whole chain, so wrapping it again at each
    // boundary would nest one copy of the chain inside the next. One crossing may
    // add the note it was given, and nothing else.
    let mut error = TreeTNOperationError::Operation {
        source: anyhow::anyhow!("root"),
    };
    for hop in 0..3 {
        error =
            TreeTNOperationError::from(anyhow::Error::new(error).context(format!("note {hop}")));
    }
    assert_eq!(
        source_chain(&error),
        vec!["note 2", "note 1", "note 0", "root"],
        "each boundary may add its own note, not another copy of the chain"
    );
}

#[test]
fn wrappers_keep_the_classification() {
    // `rename_node` adds an `anyhow` context internally; the variant must survive it.
    let (mut tree, _site) = one_site_f64_tree();
    let error = tree.rename_node(&404usize, 405usize).unwrap_err();
    assert!(
        matches!(error, TreeTNOperationError::InvalidGraph { .. }),
        "rename_node must keep the unknown-node classification, got {error:?}"
    );

    // `canonicalize` goes through `setup_sweep_context`, which used to add a context too.
    let (tree, _site) = one_site_f64_tree();
    let error = tree
        .canonicalize(
            [404usize],
            tensor4all_treetn::CanonicalizationOptions::default(),
        )
        .unwrap_err();
    assert!(
        matches!(error, TreeTNOperationError::InvalidGraph { .. }),
        "canonicalize must keep the unknown-node classification, got {error:?}"
    );
}

#[test]
fn a_dtype_mismatch_keeps_the_anyhow_context() {
    // Same contract as `a_classified_error_keeps_the_anyhow_context`, for the
    // variant whose payload is not an `anyhow::Error`: the context layers are
    // rendered back onto it instead of being dropped.
    let mismatch = tensor4all_treetn::EvaluatedScalarKindMismatch {
        stored: "c64",
        requested: "f64",
    };
    let wrapped = anyhow::Error::new(TreeTNOperationError::from(mismatch)).context("request 42");
    let converted = TreeTNOperationError::from(wrapped);
    let TreeTNOperationError::ScalarKindMismatch { payload, .. } = &converted else {
        panic!("the dtype variant must survive the conversion, got {converted:?}");
    };
    assert_eq!((payload.stored, payload.requested), ("c64", "f64"));
    let text = format!("{converted:#}");
    assert!(
        text.contains("request 42"),
        "the outer context was dropped: {text}"
    );
    assert!(
        text.contains("cannot decode a c64 TreeTN value as f64"),
        "the diagnostic was dropped: {text}"
    );
}

#[test]
fn a_classified_error_keeps_the_anyhow_context() {
    let original = TreeTNOperationError::InvalidGraph {
        source: anyhow::anyhow!("missing node"),
    };
    let wrapped = anyhow::Error::new(original).context("request 42");
    let converted = TreeTNOperationError::from(wrapped);
    assert!(
        matches!(converted, TreeTNOperationError::InvalidGraph { .. }),
        "the variant must survive the conversion, got {converted:?}"
    );
    let text = format!("{converted:#}");
    assert!(
        text.contains("request 42"),
        "the outer context was dropped: {text}"
    );
    assert!(
        text.contains("missing node"),
        "the diagnostic was dropped: {text}"
    );
}
