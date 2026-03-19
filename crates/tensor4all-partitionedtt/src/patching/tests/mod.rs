
use super::*;
use crate::projector::Projector;
use tensor4all_core::index::Index;
use tensor4all_core::TensorDynLen;
use tensor4all_itensorlike::TensorTrain;

fn make_index(size: usize) -> DynIndex {
    Index::new_dyn(size)
}

fn make_tensor(indices: Vec<DynIndex>) -> TensorDynLen {
    let dims: Vec<usize> = indices.iter().map(|i| i.dim).collect();
    let size: usize = dims.iter().product();
    let data: Vec<f64> = (0..size).map(|i| (i + 1) as f64).collect();
    TensorDynLen::from_dense(indices, data).unwrap()
}

/// Create shared indices for testing
fn make_shared_indices() -> (Vec<DynIndex>, DynIndex) {
    let s0 = make_index(2);
    let l01 = make_index(3);
    let s1 = make_index(2);
    (vec![s0, s1], l01)
}

/// Create a TT using the provided indices
fn make_tt_with_indices(site_inds: &[DynIndex], link_ind: &DynIndex) -> TensorTrain {
    let t0 = make_tensor(vec![site_inds[0].clone(), link_ind.clone()]);
    let t1 = make_tensor(vec![link_ind.clone(), site_inds[1].clone()]);
    TensorTrain::new(vec![t0, t1]).unwrap()
}

#[test]
fn test_add_with_patching_simple() {
    let (site_inds, link_ind) = make_shared_indices();

    let tt1 = make_tt_with_indices(&site_inds, &link_ind);
    let subdomain1 = SubDomainTT::new(tt1, Projector::from_pairs([(site_inds[0].clone(), 0)]));

    let tt2 = make_tt_with_indices(&site_inds, &link_ind);
    let subdomain2 = SubDomainTT::new(tt2, Projector::from_pairs([(site_inds[0].clone(), 1)]));

    let options = PatchingOptions::default();
    let result = add_with_patching(vec![subdomain1, subdomain2], &options).unwrap();

    assert_eq!(result.len(), 2);
}

#[test]
fn test_add_with_patching_requires_splitting_fails() {
    let (site_inds, link_ind) = make_shared_indices();

    let tt1 = make_tt_with_indices(&site_inds, &link_ind);
    let subdomain1 = SubDomainTT::new(tt1, Projector::from_pairs([(site_inds[0].clone(), 0)]));

    // Options that require splitting (max_bond_dim smaller than actual)
    let options = PatchingOptions {
        rtol: 1e-12,
        max_bond_dim: 1,                         // Force splitting
        patch_order: vec![site_inds[0].clone()], // Non-empty patch order triggers check
    };

    let result = add_with_patching(vec![subdomain1], &options);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        PartitionedTTError::NotImplemented(_)
    ));
}

#[test]
fn test_truncate_adaptive_not_implemented() {
    let (site_inds, link_ind) = make_shared_indices();

    let tt1 = make_tt_with_indices(&site_inds, &link_ind);
    let subdomain1 = SubDomainTT::new(tt1, Projector::from_pairs([(site_inds[0].clone(), 0)]));

    let partitioned = PartitionedTT::from_subdomains(vec![subdomain1]).unwrap();

    let result = truncate_adaptive(&partitioned, 1e-12, 100);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        PartitionedTTError::NotImplemented(_)
    ));
}
