//! #1521: `restrict` and `to_dense` honour the coverage contract.
//!
//! The stored patch sits on the **source of `restrict`**. `restrict` is driven
//! through both states of an unstored region, an uncomputed one (outside the
//! coverage) and a known zero (inside the coverage, no patch), and `to_dense`
//! then shows the difference by behaviour: a known zero materializes as `0.0`,
//! an uncomputed point is refused with `OutsideCoverage`.

use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_partitionedtreetn::{
    PartitionedTreeTN, PartitionedTreeTNError, Projector, SubDomainTreeTN,
};
use tensor4all_treetn::TreeTN;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn at(x: &DynIndex, value: usize) -> Result<Projector, Box<dyn std::error::Error>> {
    Ok(Projector::from_pairs([(x.clone(), value)])?)
}

/// Vector `[5, 6, 7]` over `x` with one stored patch at `x = 0`, and the given
/// coverage (`None` = defined everywhere).
fn source(
    x: &DynIndex,
    covered: Option<&[usize]>,
) -> Result<PartitionedTreeTN<usize>, Box<dyn std::error::Error>> {
    let tree = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(
            vec![x.clone()],
            vec![5.0_f64, 6.0, 7.0],
        )?],
        vec![0usize],
    )?;
    let partition = PartitionedTreeTN::from_subdomain(SubDomainTreeTN::new(tree, at(x, 0)?)?)?;
    match covered {
        None => Ok(partition),
        Some(values) => {
            let cov = values
                .iter()
                .map(|&v| at(x, v))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(partition.with_coverage(cov)?)
        }
    }
}

fn coverage_values(partition: &PartitionedTreeTN<usize>, x: &DynIndex) -> Option<Vec<usize>> {
    partition.coverage().map(|cov| {
        let mut v: Vec<usize> = cov.iter().filter_map(|p| p.get(x)).collect();
        v.sort_unstable();
        v
    })
}

fn all_points(x: &DynIndex) -> Result<Vec<Projector>, Box<dyn std::error::Error>> {
    (0..3).map(|v| at(x, v)).collect()
}

#[test]
fn restrict_refuses_a_selection_reaching_outside_the_coverage() -> TestResult {
    let x = DynIndex::new_dyn(3);
    // x = 2 lies outside the coverage: it is uncomputed, not zero.
    let src = source(&x, Some(&[0, 1]))?;
    match src.restrict(&all_points(&x)?) {
        Err(PartitionedTreeTNError::OutsideCoverage { projector }) => {
            assert_eq!(projector, at(&x, 2)?)
        }
        other => panic!("selection outside the coverage must be refused, got {other:?}"),
    }
    match src.restrict(&[Projector::new()]) {
        Err(PartitionedTreeTNError::OutsideCoverage { .. }) => {}
        other => panic!("whole-domain selection must be refused, got {other:?}"),
    }

    // A selection inside the coverage keeps the result partial: to_dense refuses it.
    let restricted = src.restrict(&[at(&x, 0)?, at(&x, 1)?])?;
    assert_eq!(coverage_values(&restricted, &x), Some(vec![0, 1]));
    assert_eq!(restricted.len(), 1);
    match restricted.to_dense::<f64>() {
        Err(PartitionedTreeTNError::OutsideCoverage { projector }) => {
            assert_eq!(projector, at(&x, 2)?)
        }
        other => panic!("uncomputed point must be refused, got {other:?}"),
    }
    Ok(())
}

#[test]
fn restrict_keeps_known_zero_and_to_dense_materializes_it() -> TestResult {
    let x = DynIndex::new_dyn(3);
    // x = 1, 2 are inside the coverage with no patch: exact zeros.
    let src = source(&x, Some(&[0, 1, 2]))?;
    let restricted = src.restrict(&all_points(&x)?)?;
    assert_eq!(coverage_values(&restricted, &x), Some(vec![0, 1, 2]));
    assert_eq!(
        restricted.to_dense::<f64>()?.to_vec::<f64>()?,
        vec![5.0, 0.0, 0.0]
    );

    // Selecting only the known zeros keeps them known; there is no patch.
    let zeros = src.restrict(&[at(&x, 1)?, at(&x, 2)?])?;
    assert!(zeros.is_empty());
    assert_eq!(coverage_values(&zeros, &x), Some(vec![1, 2]));
    // x = 0 is outside the selection, hence uncomputed, not zero.
    match zeros.to_dense::<f64>() {
        Err(PartitionedTreeTNError::OutsideCoverage { projector }) => {
            assert_eq!(projector, at(&x, 0)?)
        }
        other => panic!("outside-selection point must be refused, got {other:?}"),
    }
    Ok(())
}

#[test]
fn restrict_resets_coverage_to_none_only_for_full_selection_of_none() -> TestResult {
    let x = DynIndex::new_dyn(3);
    let src = source(&x, None)?;

    let partial = src.restrict(&[at(&x, 0)?, at(&x, 1)?])?;
    assert_eq!(coverage_values(&partial, &x), Some(vec![0, 1]));
    assert!(partial.to_dense::<f64>().is_err());

    let everything = src.restrict(&[Projector::new()])?;
    assert_eq!(everything.coverage(), None);
    assert_eq!(
        everything.to_dense::<f64>()?.to_vec::<f64>()?,
        vec![5.0, 0.0, 0.0]
    );

    Ok(())
}

#[test]
fn to_dense_refuses_partial_coverage_of_the_source() -> TestResult {
    let x = DynIndex::new_dyn(3);
    assert!(matches!(
        source(&x, Some(&[0, 1]))?.to_dense::<f64>(),
        Err(PartitionedTreeTNError::OutsideCoverage { .. })
    ));
    assert_eq!(
        source(&x, Some(&[0, 1, 2]))?
            .to_dense::<f64>()?
            .to_vec::<f64>()?,
        vec![5.0, 0.0, 0.0]
    );
    Ok(())
}
