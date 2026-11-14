//! The certified operator application of `schedule_merge_refine` (#1286).
//!
//! `OperatorApplication::Certified` applies the operator to every input leaf
//! with `apply_linear_operator_certified`, never forming the exact image, and
//! charges the leaves' proved bounds as the application error. These tests
//! compare the scheduled result with the dense product on a chain, for real and
//! complex data, and check the option validation and the error mapping.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensor4all_core::{AnyScalar, DynIndex, IdxTensor, TensorElement};
use tensor4all_partitionedtreetn::{
    reconstruction::*, PartitionedTreeTN, PartitionedTreeTNError, Projector, SubDomainTreeTN,
    TreeTN,
};

use crate::subset_operator_exact::{complex, dense_reference, random_operator, random_tree, real};

/// A six-node binary chain, a random state of bond 3 split into two prefix
/// patches on the first selected site, and a random bond-2 operator on the
/// first four sites.
struct Case {
    state: TreeTN<IdxTensor, usize>,
    preimage: ReconstructionTarget,
    operator: tensor4all_treetn::LinearOperator<IdxTensor, usize>,
    selection: Vec<DynIndex>,
}

fn case<T: TensorElement>(entry: fn(&mut ChaCha8Rng) -> T) -> Case {
    let mut random = ChaCha8Rng::seed_from_u64(0x1286);
    let sites: Vec<Vec<DynIndex>> = (0..6).map(|_| vec![DynIndex::new_dyn(2)]).collect();
    let edges: Vec<(usize, usize)> = (1..6).map(|node| (node - 1, node)).collect();
    let state = random_tree(&sites, &edges, entry, &mut random);
    let selection: Vec<DynIndex> = sites[..4].iter().map(|site| site[0].clone()).collect();
    let operator = random_operator(&selection, entry, &mut random);
    let patch = |value| {
        SubDomainTreeTN::new(
            state.clone(),
            Projector::from_pairs([(selection[0].clone(), value)]).unwrap(),
        )
        .unwrap()
    };
    let preimage = ReconstructionTarget::from_partition(
        &PartitionedTreeTN::from_subdomains(vec![patch(0), patch(1)]).unwrap(),
    )
    .unwrap();
    Case {
        state,
        preimage,
        operator,
        selection,
    }
}

fn schedule(
    case: &Case,
    rtol: f64,
    application: OperatorApplication<usize>,
) -> Result<MergeRefineResult<usize>, PartitionedTreeTNError> {
    schedule_merge_refine(
        &case.preimage,
        &0,
        &case.operator,
        &case.selection,
        &SubsetOperatorOptions::default(),
        ReconstructionTolerance { rtol, atol: 0.0 },
        &MergeRefineOptions {
            application,
            ..Default::default()
        },
    )
}

/// The sum of every retained term, as one dense tensor.
fn dense_result(
    result: &MergeRefineResult<usize>,
    template: &TreeTN<IdxTensor, usize>,
) -> IdxTensor {
    let mut sum = template.scale(AnyScalar::new_real(0.0)).unwrap();
    for (_, terms) in result.regions() {
        for term in terms {
            sum = sum.add(term.data()).unwrap();
        }
    }
    sum.to_dense().unwrap()
}

fn certified_matches_the_dense_reference<T: TensorElement>(entry: fn(&mut ChaCha8Rng) -> T) {
    let case = case(entry);
    let expected = dense_reference(&case.operator, &case.state, &case.selection);
    let scale = expected.norm().unwrap();
    let exact = schedule(&case, 0.0, OperatorApplication::Exact).unwrap();
    assert_eq!(exact.report().error_bound, 0.0);

    // A loose tolerance truncates; a tight one keeps the result at roundoff.
    for (rtol, share, truncates) in [(0.3, 0.5, true), (1e-10, 1.0, false)] {
        let result = schedule(&case, rtol, OperatorApplication::certified(share)).unwrap();
        let report = result.report();
        assert_eq!(report.applied_operator_count, 2);
        // No rank goal: the application error is the whole bound, and the
        // leaves' bounds stay within the application's share.
        assert!(
            report.error_bound <= share * report.absolute_tolerance,
            "bound {:e} above {share} of {:e}",
            report.error_bound,
            report.absolute_tolerance
        );
        let error = dense_result(&result, &case.state)
            .sub(&expected)
            .unwrap()
            .norm()
            .unwrap();
        assert!(
            error <= report.error_bound + 1e-12 * scale,
            "true error {error:e} above the certified bound {:e}",
            report.error_bound
        );
        if truncates {
            assert!(report.error_bound > 0.0);
            assert!(
                report.max_bond_dim < exact.report().max_bond_dim,
                "certified bond {} not below the exact {}",
                report.max_bond_dim,
                exact.report().max_bond_dim
            );
        }
    }
}

#[test]
fn certified_schedule_stays_within_its_bound_real() {
    certified_matches_the_dense_reference(real);
}

#[test]
fn certified_schedule_stays_within_its_bound_complex() {
    certified_matches_the_dense_reference(complex);
}

#[test]
fn a_share_outside_the_unit_interval_is_rejected_before_application() {
    let case = case(real);
    for share in [0.0, -0.5, 1.5, f64::NAN] {
        let error = schedule(&case, 1e-3, OperatorApplication::certified(share)).unwrap_err();
        assert!(
            matches!(error, PartitionedTreeTNError::InvalidOptions { reason, .. }
                if reason.contains("(0, 1]")),
            "share {share}: {error}"
        );
    }
}

#[test]
fn the_certified_memory_gate_is_a_resource_limit() {
    let case = case(real);
    let error = schedule(
        &case,
        1e-3,
        OperatorApplication::Certified {
            share: 0.5,
            memory_limit: 1,
        },
    )
    .unwrap_err();
    assert!(
        matches!(
            error,
            PartitionedTreeTNError::ResourceLimit { limit, value, .. }
                if limit == "memory_limit (bytes)" && value > 1
        ),
        "{error}"
    );
}

#[test]
fn a_certified_application_on_a_branched_state_is_rejected() {
    let mut random = ChaCha8Rng::seed_from_u64(0x1287);
    // Hub 0 with leaves 1, 2 and 3.
    let sites: Vec<Vec<DynIndex>> = (0..4).map(|_| vec![DynIndex::new_dyn(2)]).collect();
    let state = random_tree(&sites, &[(0, 1), (0, 2), (0, 3)], real, &mut random);
    let selection = vec![sites[1][0].clone(), sites[0][0].clone()];
    let operator = random_operator(&selection, real, &mut random);
    let patch = |value| {
        SubDomainTreeTN::new(
            state.clone(),
            Projector::from_pairs([(selection[0].clone(), value)]).unwrap(),
        )
        .unwrap()
    };
    let preimage = ReconstructionTarget::from_partition(
        &PartitionedTreeTN::from_subdomains(vec![patch(0), patch(1)]).unwrap(),
    )
    .unwrap();
    let error = schedule_merge_refine(
        &preimage,
        &0,
        &operator,
        &selection,
        &SubsetOperatorOptions::default(),
        ReconstructionTolerance {
            rtol: 1e-3,
            atol: 0.0,
        },
        &MergeRefineOptions {
            application: OperatorApplication::certified(0.5),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(
        matches!(error, PartitionedTreeTNError::InvalidOptions { reason, .. }
            if reason.contains("chain")),
        "{error}"
    );
}
