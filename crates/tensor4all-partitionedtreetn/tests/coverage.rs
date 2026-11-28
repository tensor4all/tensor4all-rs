//! Coverage metadata tests for `PartitionedTreeTN`.

use num_complex::Complex64;
use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor};
use tensor4all_partitionedtreetn::{
    truncate_adaptive, BudgetAllocation, PartitionedTreeTN, PartitionedTreeTNError, Projector,
    SubDomainTreeTN,
};
use tensor4all_treetn::{contraction::ContractionOptions, TreeTN, TruncationOptions};

fn complex_matrix_rows(
    x: &DynIndex,
    k: &DynIndex,
) -> Result<(PartitionedTreeTN<usize>, PartitionedTreeTN<usize>), Box<dyn std::error::Error>> {
    // Column-major 2x2 complex matrix over (x, k):
    //   row x=0: [1+i, 2-i]  -> sum over k = 3 + 0i
    //   row x=1: [0, 0]      -> zero row
    let mat = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(
            vec![x.clone(), k.clone()],
            vec![
                Complex64::new(1.0, 1.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(0.0, 0.0),
            ],
        )?],
        vec![0usize],
    )?;
    let left = PartitionedTreeTN::from_subdomains(vec![SubDomainTreeTN::new(
        mat,
        Projector::from_pairs([(x.clone(), 0)])?,
    )?])?;
    let ones = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(
            vec![k.clone()],
            vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
        )?],
        vec![0usize],
    )?;
    let right = PartitionedTreeTN::from_subdomain(SubDomainTreeTN::from_treetn(ones)?)?;
    Ok((left, right))
}

#[test]
fn coverage_rejects_unknown_but_keeps_known_zero() -> Result<(), Box<dyn std::error::Error>> {
    let x = DynIndex::new_dyn(2);
    let k = DynIndex::new_dyn(2);
    let (left, right) = complex_matrix_rows(&x, &k)?;

    let p0 = Projector::from_pairs([(x.clone(), 0)])?;
    let p1 = Projector::from_pairs([(x.clone(), 1)])?;

    // 1. contract_selected with duplicate wanted projectors canonicalizes coverage.
    let partial = left.contract_selected(
        &right,
        &0,
        ContractionOptions::default(),
        &[p0.clone(), p0.clone()],
    )?;
    assert_eq!(partial.coverage(), Some(&[p0.clone()][..]));
    assert_eq!(partial.clone().coverage(), Some(&[p0.clone()][..]));

    // Reading the whole partition fails with OutsideCoverage naming p1, leaving partial untouched.
    let err = partial.to_treetn().unwrap_err();
    assert!(matches!(
        err,
        PartitionedTreeTNError::OutsideCoverage { ref projector } if projector == &p1
    ));
    assert!(matches!(
        partial.norm().unwrap_err(),
        PartitionedTreeTNError::OutsideCoverage { ref projector } if projector == &p1
    ));
    assert!(matches!(
        partial.norm_squared().unwrap_err(),
        PartitionedTreeTNError::OutsideCoverage { ref projector } if projector == &p1
    ));
    assert_eq!(partial.len(), 1);
    assert_eq!(partial.coverage(), Some(&[p0.clone()][..]));

    // 2. When both p0 and p1 are wanted, p1 has no stored patch (known zero) and the union
    //    covers the whole domain, so to_treetn and norm succeed and read p1 as exact zero.
    let full = left.contract_selected(
        &right,
        &0,
        ContractionOptions::default(),
        &[p1.clone(), p0.clone()],
    )?;
    assert_eq!(full.len(), 1);
    assert_eq!(full.coverage(), Some(&[p0.clone(), p1.clone()][..]));
    let dense = full.to_treetn()?.to_dense()?.to_vec::<Complex64>()?;
    assert_eq!(
        dense,
        vec![Complex64::new(3.0, 0.0), Complex64::new(0.0, 0.0)]
    );
    assert!((full.norm()? - 3.0).abs() < 1e-12);

    // 3. An empty partition with coverage = [p1] is known zero on p1 and unknown on p0.
    let known_zero_p1 = left.contract_selected(
        &right,
        &0,
        ContractionOptions::default(),
        std::slice::from_ref(&p1),
    )?;
    assert!(known_zero_p1.is_empty());
    assert_eq!(known_zero_p1.coverage(), Some(&[p1.clone()][..]));
    assert!(matches!(
        known_zero_p1.to_treetn().unwrap_err(),
        PartitionedTreeTNError::OutsideCoverage { ref projector } if projector == &p0
    ));

    Ok(())
}

#[test]
fn coverage_sum_intersects_and_preserves_inputs() -> Result<(), Box<dyn std::error::Error>> {
    let x = DynIndex::new_dyn(3);
    let tree = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(
            vec![x.clone()],
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
            ],
        )?],
        vec![0usize],
    )?;
    let p0 = Projector::from_pairs([(x.clone(), 0)])?;
    let p1 = Projector::from_pairs([(x.clone(), 1)])?;
    let p2 = Projector::from_pairs([(x.clone(), 2)])?;

    let left = PartitionedTreeTN::from_subdomains(vec![
        SubDomainTreeTN::new(tree.clone(), p0.clone())?,
        SubDomainTreeTN::new(tree.clone(), p1.clone())?,
    ])?
    .with_coverage(vec![p1.clone(), p0.clone()])?;

    let right = PartitionedTreeTN::from_subdomains(vec![
        SubDomainTreeTN::new(tree.clone(), p1.clone())?,
        SubDomainTreeTN::new(tree, p2.clone())?,
    ])?
    .with_coverage(vec![p2.clone(), p1.clone()])?;

    let sum = left.add(&right)?;
    assert_eq!(sum.coverage(), Some(&[p1.clone()][..]));
    assert_eq!(sum.len(), 1);
    assert!(sum.contains(&p1));

    let sum_trunc = left.add_truncated(&right, &0, TruncationOptions::default())?;
    assert_eq!(sum_trunc.coverage(), Some(&[p1.clone()][..]));
    assert_eq!(sum_trunc.len(), 1);

    // Inputs untouched.
    assert_eq!(left.coverage(), Some(&[p0, p1.clone()][..]));
    assert_eq!(right.coverage(), Some(&[p1, p2][..]));

    Ok(())
}

#[test]
fn coverage_contract_conservative_on_contracted_index() -> Result<(), Box<dyn std::error::Error>> {
    let x = DynIndex::new_dyn(2);
    let k = DynIndex::new_dyn(2);
    let (left, right) = complex_matrix_rows(&x, &k)?;

    let k0 = Projector::from_pairs([(k.clone(), 0)])?;
    let x0 = Projector::from_pairs([(x.clone(), 0)])?;
    let x1 = Projector::from_pairs([(x.clone(), 1)])?;

    // Constraining the contracted index `k` must fail conservatively and leave inputs untouched.
    let partial_right = PartitionedTreeTN::from_subdomain(SubDomainTreeTN::new(
        right.values().next().unwrap().data().clone(),
        k0.clone(),
    )?)?
    .with_coverage(vec![k0.clone()])?;

    let err = left
        .contract(&partial_right, &0, ContractionOptions::default())
        .unwrap_err();
    assert!(matches!(
        err,
        PartitionedTreeTNError::OutsideCoverage { ref projector } if projector == &k0
    ));
    assert_eq!(partial_right.coverage(), Some(&[k0.clone()][..]));

    // Constraining only the output index `x` succeeds and propagates coverage.
    let partial_left = left.clone().with_coverage(vec![x0.clone()])?;
    let contracted = partial_left.contract(&right, &0, ContractionOptions::default())?;
    assert_eq!(contracted.coverage(), Some(&[x0.clone()][..]));

    // contract_selected requesting an uncovered output projector fails with OutsideCoverage.
    let err = partial_left
        .contract_selected(
            &right,
            &0,
            ContractionOptions::default(),
            std::slice::from_ref(&x1),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        PartitionedTreeTNError::OutsideCoverage { ref projector } if projector == &x1
    ));
    assert_eq!(partial_left.coverage(), Some(&[x0][..]));

    Ok(())
}

#[test]
fn coverage_truncate_adaptive_does_not_shrink_coverage() -> Result<(), Box<dyn std::error::Error>> {
    let x = DynIndex::new_dyn(2);
    let tree_big = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(
            vec![x.clone()],
            vec![Complex64::new(10.0, 0.0), Complex64::new(0.0, 0.0)],
        )?],
        vec![0usize],
    )?;
    let tree_tiny = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(
            vec![x.clone()],
            vec![Complex64::new(0.0, 0.0), Complex64::new(1e-12, 0.0)],
        )?],
        vec![0usize],
    )?;
    let p0 = Projector::from_pairs([(x.clone(), 0)])?;
    let p1 = Projector::from_pairs([(x.clone(), 1)])?;

    let part = PartitionedTreeTN::from_subdomains(vec![
        SubDomainTreeTN::new(tree_big, p0.clone())?,
        SubDomainTreeTN::new(tree_tiny, p1.clone())?,
    ])?
    .with_coverage(vec![p0.clone(), p1.clone()])?;

    // Volume allocation drops the tiny patch p1, but coverage must remain [p0, p1].
    let trunc_vol = truncate_adaptive(&part, &0, 1e-6, BudgetAllocation::Volume, None)?;
    assert_eq!(trunc_vol.len(), 1);
    assert_eq!(trunc_vol.coverage(), Some(&[p0.clone(), p1.clone()][..]));

    let trunc_norm = truncate_adaptive(&part, &0, 1e-6, BudgetAllocation::Norm, None)?;
    assert_eq!(trunc_norm.len(), 2);
    assert_eq!(trunc_norm.coverage(), Some(&[p0, p1][..]));

    Ok(())
}

#[test]
fn coverage_none_preserves_existing_behavior() -> Result<(), Box<dyn std::error::Error>> {
    let x = DynIndex::new_dyn(2);
    let k = DynIndex::new_dyn(2);
    let (left, right) = complex_matrix_rows(&x, &k)?;

    assert_eq!(left.coverage(), None);
    assert_eq!(right.coverage(), None);

    let contracted = left.contract(&right, &0, ContractionOptions::default())?;
    assert_eq!(contracted.coverage(), None);
    assert_eq!(
        contracted.to_treetn()?.to_dense()?.to_vec::<Complex64>()?,
        vec![Complex64::new(3.0, 0.0), Complex64::new(0.0, 0.0)]
    );

    let sum = left.add(&left)?;
    assert_eq!(sum.coverage(), None);

    Ok(())
}

#[test]
fn coverage_add_clips_partially_covered_patch_and_preserves_values(
) -> Result<(), Box<dyn std::error::Error>> {
    let x = DynIndex::new_dyn(2);
    let y = DynIndex::new_dyn(2);
    // f(x, y): column-major over [x, y] -> (0,0)=1, (1,0)=2, (0,1)=3, (1,1)=4
    let f_tree = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(
            vec![x.clone(), y.clone()],
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
            ],
        )?],
        vec![0usize],
    )?;
    // g(x, y): constant 10
    let g_tree = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(
            vec![x.clone(), y.clone()],
            vec![Complex64::new(10.0, 0.0); 4],
        )?],
        vec![0usize],
    )?;

    let x0 = Projector::from_pairs([(x.clone(), 0)])?;
    let x0_y0 = Projector::from_pairs([(x.clone(), 0), (y.clone(), 0)])?;
    let x1_y0 = Projector::from_pairs([(x.clone(), 1), (y.clone(), 0)])?;

    // left has patch {x=0} with coverage None; right has patch {x=1, y=0} with coverage [{x=0,y=0}, {x=1,y=0}].
    let left = PartitionedTreeTN::from_subdomain(SubDomainTreeTN::new(f_tree, x0)?)?;
    let right = PartitionedTreeTN::from_subdomain(SubDomainTreeTN::new(g_tree, x1_y0.clone())?)?
        .with_coverage(vec![x1_y0.clone(), x0_y0.clone()])?;

    let sum = left.add(&right)?;
    assert_eq!(sum.coverage(), Some(&[x0_y0, x1_y0][..]));

    // Evaluate at covered points (0, 0) and (1, 0): must preserve f(0,0)=1 and g(1,0)=10.
    let pts = [0usize, 0, 1, 0];
    let vals = sum.evaluate_batched(
        &[x.clone(), y.clone()],
        ColMajorArrayRef::new(&pts, &[2, 2])?,
    )?;
    assert_eq!(vals[0].real(), 1.0);
    assert_eq!(vals[1].real(), 10.0);

    // Evaluating at an uncovered point (1, 1) must fail with OutsideCoverage.
    let bad_pt = [1usize, 1];
    let expected_offending = Projector::from_pairs([(x.clone(), 1), (y.clone(), 1)])?;
    let err = sum
        .evaluate_batched(&[x, y], ColMajorArrayRef::new(&bad_pt, &[2, 1])?)
        .unwrap_err();
    assert!(matches!(
        err,
        PartitionedTreeTNError::OutsideCoverage { ref projector } if projector == &expected_offending
    ));

    Ok(())
}
