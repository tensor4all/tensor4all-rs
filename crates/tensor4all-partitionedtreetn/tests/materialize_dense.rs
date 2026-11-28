//! Dense materialization of patches and partitions (#1544).
//!
//! The per-file coverage threshold caught that the `to_dense` paths added with
//! #1542 were only exercised through a single-patch, single-value scatter. These
//! tests drive the parts that were left uncovered: the multi-value scatter loop
//! with a non-zero base offset, both scatter axes of a two-dimensional patch, the
//! rank-0 results (all sites projected, and an empty partition), and the typed
//! dtype guard on both entry points.

use num_complex::Complex64;
use tensor4all_core::{DynIndex, IdxTensor, TensorIndex};
use tensor4all_partitionedtreetn::{
    PartitionedTreeTN, PartitionedTreeTNError, Projector, SubDomainTreeTN,
};
use tensor4all_treetn::TreeTN;

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// A single-node tree over `[x, y]`, column-major (x fastest), holding `values`.
fn plane(
    x: &DynIndex,
    y: &DynIndex,
    values: Vec<f64>,
) -> Result<TreeTN<IdxTensor, usize>, Box<dyn std::error::Error>> {
    Ok(TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![x.clone(), y.clone()], values)?],
        vec![0usize],
    )?)
}

fn fix(index: &DynIndex, value: usize) -> Result<Projector, Box<dyn std::error::Error>> {
    Ok(Projector::from_pairs([(index.clone(), value)])?)
}

#[test]
fn partition_scatter_fills_both_axes_from_two_patches() -> TestResult {
    // [x, y] = [[1, 2], [3, 4]] column-major: (x0,y0)=1, (x1,y0)=2,
    // (x0,y1)=3, (x1,y1)=4. One patch per y coordinate.
    let x = DynIndex::new_dyn(2);
    let y = DynIndex::new_dyn(2);
    let values = vec![1.0_f64, 2.0, 3.0, 4.0];

    let lower = SubDomainTreeTN::new(plane(&x, &y, values.clone())?, fix(&y, 0)?)?;
    let upper = SubDomainTreeTN::new(plane(&x, &y, values.clone())?, fix(&y, 1)?)?;
    let partition = PartitionedTreeTN::from_subdomains(vec![lower, upper])?;

    let dense = partition.to_dense::<f64>()?;
    assert_eq!(dense.external_indices(), vec![x, y]);
    assert_eq!(dense.to_vec::<f64>()?, vec![1.0, 2.0, 3.0, 4.0]);
    Ok(())
}

#[test]
fn subdomain_scatter_handles_a_leading_fixed_axis() -> TestResult {
    // Fixing x (the leading, fastest axis) puts the patch at a non-zero base
    // offset and leaves a single free y axis.
    let x = DynIndex::new_dyn(2);
    let y = DynIndex::new_dyn(2);
    let patch = SubDomainTreeTN::new(plane(&x, &y, vec![1.0_f64, 2.0, 3.0, 4.0])?, fix(&x, 1)?)?;

    let dense = patch.to_dense::<f64>()?;
    assert_eq!(dense.external_indices(), vec![y]);
    assert_eq!(dense.to_vec::<f64>()?, vec![2.0, 4.0]);
    Ok(())
}

#[test]
fn all_projected_patch_materializes_as_rank_zero() -> TestResult {
    let x = DynIndex::new_dyn(3);
    let tree = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(
            vec![x.clone()],
            vec![10.0_f64, 20.0, 30.0],
        )?],
        vec![0usize],
    )?;
    let patch = SubDomainTreeTN::new(tree, fix(&x, 1)?)?;

    let dense = patch.to_dense::<f64>()?;
    assert!(dense.external_indices().is_empty());
    assert!(dense.dims().is_empty());
    assert_eq!(dense.to_vec::<f64>()?, vec![20.0]);

    // Through the partition the dense domain is still the full site space, so
    // the one all-projected patch lands at its coordinate in a zero buffer.
    let partition = PartitionedTreeTN::from_subdomain(patch)?;
    let scattered = partition.to_dense::<f64>()?;
    assert_eq!(scattered.external_indices(), vec![x]);
    assert_eq!(scattered.to_vec::<f64>()?, vec![0.0, 20.0, 0.0]);
    Ok(())
}

#[test]
fn empty_partition_materializes_as_typed_rank_zero() -> TestResult {
    let empty = PartitionedTreeTN::<usize>::new();

    let as_f64 = empty.to_dense::<f64>()?;
    assert!(as_f64.is_f64());
    assert!(as_f64.external_indices().is_empty());
    assert_eq!(as_f64.to_vec::<f64>()?, vec![0.0]);

    let as_c64 = empty.to_dense::<Complex64>()?;
    assert!(as_c64.is_c64());
    assert!(as_c64.external_indices().is_empty());
    assert_eq!(
        as_c64.to_vec::<Complex64>()?,
        vec![Complex64::new(0.0, 0.0)]
    );
    Ok(())
}

#[test]
fn both_entry_points_reject_a_dtype_mismatch() -> TestResult {
    let x = DynIndex::new_dyn(2);
    let y = DynIndex::new_dyn(2);
    let patch = SubDomainTreeTN::new(plane(&x, &y, vec![1.0_f64, 2.0, 3.0, 4.0])?, fix(&y, 0)?)?;
    let partition = PartitionedTreeTN::from_subdomain(patch.clone())?;

    assert!(matches!(
        patch.to_dense::<Complex64>(),
        Err(PartitionedTreeTNError::DTypeMismatch { .. })
    ));
    assert!(matches!(
        partition.to_dense::<Complex64>(),
        Err(PartitionedTreeTNError::DTypeMismatch { .. })
    ));
    Ok(())
}

#[test]
fn complex_partition_scatters_with_a_fixed_trailing_axis() -> TestResult {
    let x = DynIndex::new_dyn(2);
    let y = DynIndex::new_dyn(2);
    let complex = |re: f64| Complex64::new(re, 10.0 * re);
    let values: Vec<Complex64> = [1.0, 2.0, 3.0, 4.0].into_iter().map(complex).collect();

    let left = SubDomainTreeTN::new(
        TreeTN::from_tensors(
            vec![IdxTensor::from_dense(
                vec![x.clone(), y.clone()],
                values.clone(),
            )?],
            vec![0usize],
        )?,
        fix(&y, 0)?,
    )?;
    let right = SubDomainTreeTN::new(
        TreeTN::from_tensors(
            vec![IdxTensor::from_dense(
                vec![x.clone(), y.clone()],
                values.clone(),
            )?],
            vec![0usize],
        )?,
        fix(&y, 1)?,
    )?;
    let partition = PartitionedTreeTN::from_subdomains(vec![left, right])?;

    let dense = partition.to_dense::<Complex64>()?;
    assert_eq!(dense.external_indices(), vec![x, y]);
    assert_eq!(dense.to_vec::<Complex64>()?, values);
    Ok(())
}
