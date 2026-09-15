use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_partitionedtreetn::{
    reconstruction::{
        reconstruct, ReconstructionOptions, ReconstructionTarget, ReconstructionTolerance,
    },
    PartitionedTreeTN, SubDomainTreeTN, TreeTN,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ANCHOR: reconstruction
    let x = DynIndex::new_dyn(2);
    let y = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(2);
    // Column-major cores for T(x,y) = delta(x,y).
    let tree = TreeTN::from_tensors(
        vec![
            IdxTensor::from_dense(vec![x.clone(), bond.clone()], vec![1.0, 0.0, 0.0, 1.0])?,
            IdxTensor::from_dense(vec![bond, y], vec![1.0, 0.0, 0.0, 1.0])?,
        ],
        vec![0usize, 1],
    )?;
    let reference = tree.clone().to_dense()?; // Small reference example only.
    let partition = PartitionedTreeTN::from_subdomain(SubDomainTreeTN::from_treetn(tree)?)?;
    let target = ReconstructionTarget::from_partition(&partition)?;
    let output = reconstruct(
        &target,
        &0,
        ReconstructionTolerance {
            rtol: 1e-8,
            atol: 0.0,
        },
        &ReconstructionOptions {
            target_bond_dim: Some(1),
            split_indices: vec![x],
            ..Default::default()
        },
    )?;
    assert!((output.report().reference_norm - 2.0_f64.sqrt()).abs() < 1e-12);
    assert_eq!(output.report().region_count, 2);
    assert_eq!(output.report().max_bond_dim, 1);
    assert!(output.report().error_bound <= output.report().absolute_tolerance);
    // Conversion succeeds because each region has one retained term.
    let reconstructed = output.into_partition()?.to_treetn()?.to_dense()?;
    assert!(reconstructed.sub(&reference)?.maxabs()? < 1e-12);
    // ANCHOR_END: reconstruction
    Ok(())
}
