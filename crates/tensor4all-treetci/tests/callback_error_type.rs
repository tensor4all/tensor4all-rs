//! A caller's own callback error type survives a tree-TCI run (#1023).
//!
//! The evaluator bound is `E: Into<anyhow::Error>`, not `anyhow::Result`, so the
//! caller returns its own error directly and recovers it from the returned
//! error's source chain without boxing it into `anyhow` itself.

use std::error::Error as _;

use tensor4all_treetci::{
    crossinterpolate2, DefaultProposer, GlobalIndexBatch, TreeTciError, TreeTciGraph,
    TreeTciOptions,
};

/// The caller's own error type: no `anyhow` in its definition.
#[derive(Debug, PartialEq, Eq)]
enum QueryError {
    OutOfRange { site: usize },
}

impl std::fmt::Display for QueryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutOfRange { site } => write!(f, "site {site} is out of range"),
        }
    }
}

impl std::error::Error for QueryError {}

#[test]
fn callback_returns_its_own_error_type_and_the_caller_recovers_it() {
    // The closure returns `Result<Vec<f64>, QueryError>`; nothing is boxed by
    // the caller.
    let evaluate = |_batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>, QueryError> {
        Err(QueryError::OutOfRange { site: 1 })
    };

    let error = crossinterpolate2::<f64, _, _, _, _>(
        evaluate,
        vec![2, 2],
        TreeTciGraph::new(2, &[tensor4all_treetci::TreeTciEdge::new(0, 1)]).unwrap(),
        vec![vec![0, 0]],
        TreeTciOptions::default(),
        None,
        &DefaultProposer,
        &mut (),
    )
    .unwrap_err();

    let TreeTciError::Operation { .. } = &error else {
        panic!("callback failures stay in the catch-all variant, got {error:?}");
    };
    let recovered = error
        .source()
        .expect("the callback error is the source of the run error")
        .downcast_ref::<QueryError>()
        .expect("the caller's own error type is preserved, not stringified");
    assert_eq!(recovered, &QueryError::OutOfRange { site: 1 });
    assert!(error.to_string().contains("site 1 is out of range"));
}
