//! Level-coupled merge-refine scheduling of a one-bit subset transform.
//!
//! Run with `cargo run --release -p tensor4all-partitionedtreetn --example merge_refine`.

use std::collections::HashMap;

use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_partitionedtreetn::{
    reconstruction::{
        schedule_merge_refine, MergeRefineOptions, ReconstructionTarget, ReconstructionTolerance,
        SubsetOperatorOptions,
    },
    PartitionedTreeTN, Projector, SubDomainTreeTN, TreeTN,
};
use tensor4all_treetn::{IndexMapping, LinearOperator};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ANCHOR: merge_refine
    let site = DynIndex::new_dyn(2);
    let tree = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![site.clone()], vec![3.0, 4.0])?],
        vec![0usize],
    )?;
    // The schedule consumes the 2^d dyadic input leaves of the selected sites.
    let full = SubDomainTreeTN::from_treetn(tree)?;
    let leaves = (0..2)
        .map(
            |value| -> Result<SubDomainTreeTN, Box<dyn std::error::Error>> {
                let projector = Projector::from_pairs([(site.clone(), value)])?;
                full.project(&projector)?.ok_or_else(|| "zero leaf".into())
            },
        )
        .collect::<Result<Vec<_>, _>>()?;
    let preimage =
        ReconstructionTarget::from_partition(&PartitionedTreeTN::from_subdomains(leaves)?)?;

    // A normalized Hadamard transform written as a one-node MPO.
    let (internal_input, internal_output) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
    let scale = 1.0 / 2.0_f64.sqrt();
    let mpo = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(
            vec![internal_input.clone(), internal_output.clone()],
            vec![scale, scale, scale, -scale],
        )?],
        vec![0usize],
    )?;
    let mut input_mapping = HashMap::new();
    input_mapping.insert(
        0usize,
        IndexMapping {
            true_index: site.clone(),
            internal_index: internal_input,
        },
    );
    let mut output_mapping = HashMap::new();
    output_mapping.insert(
        0usize,
        IndexMapping {
            true_index: site.clone(),
            internal_index: internal_output,
        },
    );

    let result = schedule_merge_refine(
        &preimage,
        &0,
        &LinearOperator::new(mpo, input_mapping, output_mapping),
        std::slice::from_ref(&site),
        &SubsetOperatorOptions { unitary: true },
        ReconstructionTolerance {
            rtol: 1e-12,
            atol: 0.0,
        },
        &MergeRefineOptions::default(),
    )?;

    // One complete transform per input leaf, one level, one addition per child.
    let report = result.report();
    assert_eq!(report.applied_operator_count, 2);
    assert_eq!(report.additions, 2);
    assert_eq!(report.work_items_per_level, vec![2, 2]);
    assert!((report.reference_scale - 5.0).abs() < 1e-12);

    // The fully refined result is one patch per output coordinate.
    let partition = result.into_partition()?;
    let region = |value: usize| -> Result<f64, Box<dyn std::error::Error>> {
        let patch = partition
            .values()
            .find(|patch| patch.projector().get(&site) == Some(value))
            .ok_or("missing output region")?;
        Ok(patch.norm()?)
    };
    assert!((region(0)? - 7.0 / 2.0_f64.sqrt()).abs() < 1e-12);
    assert!((region(1)? - 1.0 / 2.0_f64.sqrt()).abs() < 1e-12);
    // ANCHOR_END: merge_refine
    Ok(())
}
