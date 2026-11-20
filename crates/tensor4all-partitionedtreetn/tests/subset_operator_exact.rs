//! The exact subset-operator application (#1279) against a dense reference.
//!
//! `ReconstructionTarget::from_subset_operator` applies the operator to every
//! patch with an exact zip-up centred at the balanced node. These tests check its
//! images against the dense product of the dense operator and the dense state,
//! on a chain whose balanced centre is a spectator node and on a branched tree
//! with a spectator branch and a node carrying two selected sites, for real and
//! complex data.
//!
//! The exact application factorizes every zip-up block by QR (#1285). The same
//! shapes also run through `schedule_merge_refine` with an untruncated SVD
//! zip-up as its measured application: the schedule then measures, per image, the
//! distance between the QR image and the SVD image, so a tight tolerance checks
//! that the two factorizations give the same images to roundoff.

use std::collections::HashMap;

use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tensor4all_core::{
    contract, AnyScalar, DynIndex, IdxTensor, IndexLike, SvdTruncationPolicy, TensorElement,
};
use tensor4all_partitionedtreetn::{
    reconstruction::*, PartitionedTreeTN, Projector, SubDomainTreeTN, TreeTN,
};
use tensor4all_treetn::{ApplyOptions, IndexMapping, LinearOperator};

/// A tree tensor network with the given sites per node and edges, filled with
/// seeded random entries; bond dimensions are 3.
pub(crate) fn random_tree<T: TensorElement>(
    sites: &[Vec<DynIndex>],
    edges: &[(usize, usize)],
    entry: fn(&mut ChaCha8Rng) -> T,
    random: &mut ChaCha8Rng,
) -> TreeTN<IdxTensor, usize> {
    let mut legs: Vec<Vec<DynIndex>> = sites.to_vec();
    for &(left, right) in edges {
        let bond = DynIndex::new_dyn(3);
        legs[left].push(bond.clone());
        legs[right].push(bond);
    }
    let tensors = legs
        .into_iter()
        .map(|indices| {
            let size = indices.iter().map(IndexLike::dim).product();
            let data = (0..size).map(|_| entry(random)).collect();
            IdxTensor::from_dense(indices, data).unwrap()
        })
        .collect();
    TreeTN::from_tensors(tensors, (0..sites.len()).collect()).unwrap()
}

/// A chain MPO with one random two-by-two node per selected site, bond 2.
///
/// Node `k` acts on `selection[k]`, so the caller orders `selection` along a
/// path of the state.
pub(crate) fn random_operator<T: TensorElement>(
    selection: &[DynIndex],
    entry: fn(&mut ChaCha8Rng) -> T,
    random: &mut ChaCha8Rng,
) -> LinearOperator<IdxTensor, usize> {
    let internal: Vec<(DynIndex, DynIndex)> = selection
        .iter()
        .map(|site| (DynIndex::new_dyn(site.dim()), DynIndex::new_dyn(site.dim())))
        .collect();
    let bonds: Vec<DynIndex> = (1..selection.len()).map(|_| DynIndex::new_dyn(2)).collect();
    let tensors = internal
        .iter()
        .enumerate()
        .map(|(node, (input, output))| {
            let mut indices = vec![input.clone(), output.clone()];
            indices.extend(node.checked_sub(1).map(|left| bonds[left].clone()));
            indices.extend(bonds.get(node).cloned());
            let size = indices.iter().map(IndexLike::dim).product();
            let data = (0..size).map(|_| entry(random)).collect();
            IdxTensor::from_dense(indices, data).unwrap()
        })
        .collect();
    let mpo = TreeTN::from_tensors(tensors, (0..selection.len()).collect()).unwrap();
    let mut inputs = HashMap::new();
    let mut outputs = HashMap::new();
    for (node, (input, output)) in internal.into_iter().enumerate() {
        inputs.insert(
            node,
            IndexMapping {
                true_index: DynIndex::new_dyn(input.dim()),
                internal_index: input,
            },
        );
        outputs.insert(
            node,
            IndexMapping {
                true_index: DynIndex::new_dyn(output.dim()),
                internal_index: output,
            },
        );
    }
    LinearOperator::new(mpo, inputs, outputs)
}

/// The dense `A x`: the dense operator contracted with the dense state over the
/// selected sites, the output written back onto the same site indices.
pub(crate) fn dense_reference(
    operator: &LinearOperator<IdxTensor, usize>,
    state: &TreeTN<IdxTensor, usize>,
    selection: &[DynIndex],
) -> IdxTensor {
    let mut dense_operator = operator.mpo().to_dense().unwrap();
    let mut outputs = Vec::with_capacity(selection.len());
    for (node, site) in selection.iter().enumerate() {
        let input = &operator.get_input_mapping(&node).unwrap().internal_index;
        let output = &operator.get_output_mapping(&node).unwrap().internal_index;
        let fresh = DynIndex::new_dyn(site.dim());
        dense_operator = dense_operator
            .replaceind(input, site)
            .unwrap()
            .replaceind(output, &fresh)
            .unwrap();
        outputs.push(fresh);
    }
    let dense_state = state.to_dense().unwrap();
    let mut product = contract(&[&dense_state, &dense_operator]).unwrap();
    for (fresh, site) in outputs.iter().zip(selection) {
        product = product.replaceind(fresh, site).unwrap();
    }
    product
}

/// The sum of the target's images, recovered exactly (zero tolerance).
fn dense_images(target: &ReconstructionTarget, template: &TreeTN<IdxTensor, usize>) -> IdxTensor {
    let output = reconstruct(
        target,
        &0,
        ReconstructionTolerance {
            rtol: 0.0,
            atol: 0.0,
        },
        &ReconstructionOptions {
            target_bond_dim: Some(1),
            max_regions: 256,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(output.report().error_bound, 0.0);
    let mut sum = template.scale(AnyScalar::new_real(0.0)).unwrap();
    for (_, terms) in output.regions() {
        for term in terms {
            sum = sum.add(term.data()).unwrap();
        }
    }
    sum.to_dense().unwrap()
}

/// Apply a random operator to `selection` of a random state split into two
/// patches on `split`, and compare the images with the dense reference.
fn check<T: TensorElement>(
    sites: &[Vec<DynIndex>],
    edges: &[(usize, usize)],
    selection: &[DynIndex],
    split: &DynIndex,
    entry: fn(&mut ChaCha8Rng) -> T,
) {
    let mut random = ChaCha8Rng::seed_from_u64(0x1279);
    let state = random_tree(sites, edges, entry, &mut random);
    let operator = random_operator(selection, entry, &mut random);
    let patch = |value| {
        SubDomainTreeTN::new(
            state.clone(),
            Projector::from_pairs([(split.clone(), value)]).unwrap(),
        )
        .unwrap()
    };
    let preimage = ReconstructionTarget::from_partition(
        &PartitionedTreeTN::from_subdomains(vec![patch(0), patch(1)]).unwrap(),
    )
    .unwrap();
    let target = ReconstructionTarget::from_subset_operator(
        &preimage,
        &0,
        &operator,
        selection,
        &SubsetOperatorOptions::default(),
    )
    .unwrap();

    let expected = dense_reference(&operator, &state, selection);
    let actual = dense_images(&target, &state);
    let scale = expected.maxabs().unwrap();
    let error = actual.sub(&expected).unwrap().maxabs().unwrap();
    assert!(scale > 0.0);
    assert!(
        error <= 1e-12 * scale,
        "max error {error:e} against max entry {scale:e}"
    );

    // QR images against untruncated SVD images (#1285). The smallest positive
    // threshold keeps every nonzero singular value. The schedule charges the
    // measured distance of every image pair to its bound and rejects a pair
    // above the allowance, so success under `rtol = 1e-12` with no rank goal
    // means each QR image equals its SVD image to roundoff. The schedule needs
    // every patch to fix a prefix of the selected bits, so this preimage splits
    // the same state on the first selected index.
    let prefix_patch = |value| {
        SubDomainTreeTN::new(
            state.clone(),
            Projector::from_pairs([(selection[0].clone(), value)]).unwrap(),
        )
        .unwrap()
    };
    let preimage = ReconstructionTarget::from_partition(
        &PartitionedTreeTN::from_subdomains(vec![prefix_patch(0), prefix_patch(1)]).unwrap(),
    )
    .unwrap();
    let tolerance = ReconstructionTolerance {
        rtol: 1e-12,
        atol: 0.0,
    };
    let compared = schedule_merge_refine(
        &preimage,
        &0,
        &operator,
        selection,
        &SubsetOperatorOptions::default(),
        tolerance,
        &MergeRefineOptions {
            application: OperatorApplication::Measured(ApplyOptions {
                svd_policy: Some(SvdTruncationPolicy::new(f64::MIN_POSITIVE)),
                max_bond_dim: None,
                ..ApplyOptions::zipup()
            }),
            ..Default::default()
        },
    )
    .unwrap();
    let report = compared.report();
    assert_eq!(report.applied_operator_count, 2);
    assert_eq!(report.compression_attempts, 0);
    assert!(
        report.error_bound <= report.absolute_tolerance,
        "QR and SVD images differ by {:e} against an allowance of {:e}",
        report.error_bound,
        report.absolute_tolerance
    );
}

pub(crate) fn real(random: &mut ChaCha8Rng) -> f64 {
    random.random_range(-1.0..1.0)
}

pub(crate) fn complex(random: &mut ChaCha8Rng) -> Complex64 {
    Complex64::new(
        random.random_range(-1.0..1.0),
        random.random_range(-1.0..1.0),
    )
}

/// Chain 0 - 1 - 2 - 3 - 4, one binary site per node, the operator on nodes 0
/// and 1: the balanced centre is node 2, a spectator, so the zip-up carries the
/// operator's remainder through spectator nodes. The split is on a spectator.
fn chain_case<T: TensorElement>(entry: fn(&mut ChaCha8Rng) -> T) {
    let sites: Vec<Vec<DynIndex>> = (0..5).map(|_| vec![DynIndex::new_dyn(2)]).collect();
    let edges = [(0, 1), (1, 2), (2, 3), (3, 4)];
    let selection = [sites[0][0].clone(), sites[1][0].clone()];
    check(&sites, &edges, &selection, &sites[3][0], entry);
}

/// Hub 0 with branches {1}, {2} and {3 - 4}. Node 1 carries two selected sites,
/// node 0 and node 3 one each; node 2 and node 4 are spectator leaves, so the
/// zip-up emits them unchanged. The operator chain runs 1a - 1b - 0 - 3 along
/// the tree, and the split is on a selected site.
fn branched_case<T: TensorElement>(entry: fn(&mut ChaCha8Rng) -> T) {
    let sites: Vec<Vec<DynIndex>> = [1, 2, 1, 1, 1]
        .iter()
        .map(|&count| (0..count).map(|_| DynIndex::new_dyn(2)).collect())
        .collect();
    let edges = [(0, 1), (0, 2), (0, 3), (3, 4)];
    let selection = [
        sites[1][0].clone(),
        sites[1][1].clone(),
        sites[0][0].clone(),
        sites[3][0].clone(),
    ];
    check(&sites, &edges, &selection, &sites[0][0], entry);
}

#[test]
fn chain_with_a_spectator_centre_matches_the_dense_reference_real() {
    chain_case(real);
}

#[test]
fn chain_with_a_spectator_centre_matches_the_dense_reference_complex() {
    chain_case(complex);
}

#[test]
fn branched_tree_with_a_multi_site_node_matches_the_dense_reference_real() {
    branched_case(real);
}

#[test]
fn branched_tree_with_a_multi_site_node_matches_the_dense_reference_complex() {
    branched_case(complex);
}
