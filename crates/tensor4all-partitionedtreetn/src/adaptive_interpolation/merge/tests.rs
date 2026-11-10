//! Tests of sibling merging.
//!
//! Synthetic results pin the rule on a star with a degree-3 hub and one
//! binary site per node; runs of `patched_interpolate` on a star with two
//! binary sites per node check it against the function.

use std::collections::BTreeMap;

use tensor4all_core::{ColMajorArray, ColMajorArrayRef, DynIndex, IdxTensor};
use tensor4all_tensorbackend::{explicit::Session, CpuBackend};
use tensor4all_treetci::TreeTciInterpolator;
use tensor4all_treetn::interpolation::{EvalContext, InterpolationTermination, Parallelism};
use tensor4all_treetn::{NodeNameNetwork, TreeTN};

use super::*;
use crate::adaptive_interpolation::report::{
    GlobalL2Error, L2Measurement, L2ReferenceSource, MeasurementMethod, PatchRecord,
    ZeroPatchRecord,
};
use crate::adaptive_interpolation::verify::AuditedTerm;
use crate::adaptive_interpolation::{
    patched_interpolate, PatchedInterpolationOptions, VerificationOptions,
};
use crate::{ErrorNorm, ErrorTolerance, L2Reference};

// ---------------------------------------------------------------------------
// Synthetic results: one binary site per node of a star 0 - {1, 2, 3}.
// ---------------------------------------------------------------------------

/// The four sites, in node order; also the split order.
fn sites() -> Vec<DynIndex> {
    (0..4).map(|_| DynIndex::new_dyn(2)).collect()
}

/// The rank-one network `prod_i factors[i][s_i]` on the star.
fn product(sites: &[DynIndex], factors: [[f64; 2]; 4]) -> TreeTN<IdxTensor, usize> {
    let bonds: Vec<DynIndex> = (0..3).map(|_| DynIndex::new_dyn(1)).collect();
    let mut tensors = vec![IdxTensor::from_dense(
        vec![
            sites[0].clone(),
            bonds[0].clone(),
            bonds[1].clone(),
            bonds[2].clone(),
        ],
        factors[0].to_vec(),
    )
    .unwrap()];
    for leaf in 1..4 {
        tensors.push(
            IdxTensor::from_dense(
                vec![bonds[leaf - 1].clone(), sites[leaf].clone()],
                factors[leaf].to_vec(),
            )
            .unwrap(),
        );
    }
    TreeTN::from_tensors(tensors, vec![0usize, 1, 2, 3]).unwrap()
}

/// The dense tensor of `f` over `sites`, column-major.
fn dense(sites: &[DynIndex], f: impl Fn(&[usize]) -> f64) -> IdxTensor {
    let n = sites.len();
    let values = (0..1usize << n)
        .map(|i| {
            let point: Vec<usize> = (0..n).map(|k| (i >> k) & 1).collect();
            f(&point)
        })
        .collect();
    IdxTensor::from_dense(sites.to_vec(), values).unwrap()
}

const A: [[f64; 2]; 4] = [[1.0, 2.0], [1.0, -1.0], [0.5, 1.5], [2.0, 1.0]];
const B: [[f64; 2]; 4] = [[1.0, 1.0], [3.0, 1.0], [1.0, -2.0], [-1.0, 1.0]];

fn product_value(factors: [[f64; 2]; 4], point: &[usize]) -> f64 {
    (0..4).map(|i| factors[i][point[i]]).product()
}

/// One leaf of a synthetic result.
struct Spec {
    fixed: Vec<usize>,
    network: Option<TreeTN<IdxTensor, usize>>,
    rms: f64,
    status: PatchStatus,
}

fn leaf(fixed: &[usize], network: Option<TreeTN<IdxTensor, usize>>, rms: f64) -> Spec {
    Spec {
        fixed: fixed.to_vec(),
        network,
        rms,
        status: PatchStatus::WithinTolerance,
    }
}

fn measurement(rms: f64, patch_points: f64) -> L2Measurement {
    L2Measurement {
        method: MeasurementMethod::Exhaustive,
        points: patch_points as usize,
        patch_points,
        rms,
        mean_square_rel_std_error: 0.0,
        max_residual: rms,
        residual_rms: None,
        residual_saturated: false,
        sample_rms: None,
    }
}

/// A result with the given leaves, split order `sites`, cap, and `tau`; its
/// error is assembled from the leaves as the run would.
fn synthetic(
    sites: &[DynIndex],
    specs: Vec<Spec>,
    cap: usize,
    tau: f64,
) -> PatchedInterpolationResult<usize> {
    let domain_points = (1usize << sites.len()) as f64;
    let mut accepted = Vec::new();
    let mut zero_patches = Vec::new();
    let mut subdomains = Vec::new();
    for spec in specs {
        let projector = Projector::from_pairs(
            spec.fixed
                .iter()
                .enumerate()
                .map(|(k, &v)| (sites[k].clone(), v)),
        )
        .unwrap();
        let patch_points = domain_points / (1usize << spec.fixed.len()) as f64;
        let acceptance = Some(measurement(spec.rms, patch_points));
        match spec.network {
            Some(network) => {
                let subdomain = SubDomainTreeTN::new(network, projector.clone()).unwrap();
                let mut data = subdomain.data().clone();
                accepted.push(PatchRecord {
                    projector,
                    termination: InterpolationTermination::Converged,
                    engine_error_estimate: 0.0,
                    max_sample_magnitude: 1.0,
                    max_bond_dim: subdomain.max_bond_dim(),
                    retries_used: 0,
                    acceptance,
                    audit: None,
                    status: spec.status,
                    log_norm: data.log_norm().ok(),
                });
                subdomains.push(subdomain);
            }
            None => zero_patches.push(ZeroPatchRecord {
                projector,
                acceptance,
                audit: None,
            }),
        }
    }
    let placeholder = L2ErrorReport {
        domain_points,
        global: GlobalL2Error::AcceptanceOnly {
            acceptance_statistic_rms: 0.0,
        },
        certified_fraction: 0.0,
        approximation_rms: None,
    };
    let mut report = PatchedInterpolationReport {
        tolerance: ErrorTolerance {
            rtol: 0.0,
            atol: tau * domain_points.sqrt(),
        },
        patch_order: sites.to_vec(),
        max_bond_dim: cap,
        norm: NormReport::L2 {
            reference_rms: None,
            source: L2ReferenceSource::NotNeeded,
            tau,
            error: placeholder,
        },
        accepted,
        zero_patches,
        splits: 0,
        function_evaluations: 0,
        cache_hits: 0,
        measurement_evaluations: 0,
        audit_evaluations: 0,
        residual_evaluations: 0,
        residual_saturations: 0,
        verification_failures: 0,
        engine_retries: 0,
    };
    let partition = PartitionedTreeTN::from_subdomains(subdomains).unwrap();
    let leaves = initial_leaves(partition.clone(), &report).unwrap();
    let leaves: Vec<_> = leaves.into_values().collect();
    let error = merged_error(&leaves, domain_points, tau).unwrap();
    if let NormReport::L2 { error: slot, .. } = &mut report.norm {
        *slot = error;
    }
    PatchedInterpolationResult { partition, report }
}

fn run_error(result: &PatchedInterpolationResult<usize>) -> L2ErrorReport {
    result.report.norm.l2_error().cloned().unwrap()
}

fn assert_dense(partition: &PartitionedTreeTN<usize>, expected: &IdxTensor, tolerance: f64) {
    let actual = partition.to_treetn().unwrap().to_dense().unwrap();
    let error = actual.sub(expected).unwrap().maxabs().unwrap();
    assert!(error <= tolerance, "max-absolute residual {error:e}");
}

#[test]
fn a_rank_one_split_merges_back_into_one_patch() {
    let s = sites();
    let g = product(&s, A);
    let result = synthetic(
        &s,
        vec![leaf(&[0], Some(g.clone()), 0.0), leaf(&[1], Some(g), 0.0)],
        2,
        1e-8,
    );
    let merged = merge_siblings(result, &0, &SiblingMergeOptions::new()).unwrap();
    assert_eq!(merged.merge.merges.len(), 1);
    let record = &merged.merge.merges[0];
    assert!(record.region.is_empty());
    assert_eq!(record.members.len(), 2);
    assert_eq!(record.max_bond_dim, 1);
    assert_eq!(record.members_rms, 0.0);
    assert!(record.truncation_rms <= 1e-8);
    assert_eq!(merged.partition.len(), 1);
    assert!(merged.partition.get(&Projector::new()).is_some());
    assert_dense(
        &merged.partition,
        &dense(&s, |p| product_value(A, p)),
        1e-12,
    );
    let GlobalL2Error::Certified { rms_error, .. } = merged.merge.error.global else {
        panic!("a merge of exhaustive patches stays certified");
    };
    assert_eq!(rms_error, record.truncation_rms);
    assert_eq!(merged.merge.error.certified_fraction, 1.0);
}

#[test]
fn merges_run_bottom_up_through_merged_regions() {
    let s = sites();
    let g = product(&s, A);
    let specs = [[0, 0], [0, 1], [1, 0], [1, 1]]
        .iter()
        .map(|fixed| leaf(fixed, Some(g.clone()), 0.0))
        .collect();
    let merged = merge_siblings(synthetic(&s, specs, 2, 1e-8), &0, &Default::default()).unwrap();
    let regions: Vec<usize> = merged
        .merge
        .merges
        .iter()
        .map(|record| record.region.len())
        .collect();
    assert_eq!(regions, vec![1, 1, 0]);
    assert_eq!(merged.merge.merges[0].region.get(&s[0]), Some(0));
    assert_eq!(merged.merge.merges[1].region.get(&s[0]), Some(1));
    assert_eq!(merged.merge.merges[2].members.len(), 2);
    assert_eq!(merged.partition.len(), 1);
    assert_dense(
        &merged.partition,
        &dense(&s, |p| product_value(A, p)),
        1e-12,
    );
}

#[test]
fn a_merged_patch_at_the_cap_is_rejected() {
    let s = sites();
    let g = product(&s, A);
    let result = synthetic(
        &s,
        vec![leaf(&[0], Some(g.clone()), 0.0), leaf(&[1], Some(g), 0.0)],
        2,
        1e-8,
    );
    let before = run_error(&result);
    // The truncated sum has rank one, which equals the lowered cap.
    let options = SiblingMergeOptions::new().with_max_bond_dim(1);
    let merged = merge_siblings(result, &0, &options).unwrap();
    assert!(merged.merge.merges.is_empty());
    assert_eq!(merged.merge.rejected_by_cap, 1);
    assert_eq!(merged.merge.max_bond_dim, 1);
    assert_eq!(merged.partition.len(), 2);
    assert_eq!(merged.merge.error, before);
}

#[test]
fn without_slack_only_an_exact_sum_below_the_cap_merges() {
    let s = sites();
    let g = product(&s, A);
    let specs = || {
        vec![
            leaf(&[0], Some(g.clone()), 0.0),
            leaf(&[1], Some(g.clone()), 0.0),
        ]
    };
    // tau = 0 leaves no slack: the sum is kept untruncated, with rank two.
    let rejected = merge_siblings(synthetic(&s, specs(), 2, 0.0), &0, &Default::default()).unwrap();
    assert_eq!(rejected.merge.rejected_by_cap, 1);
    assert_eq!(rejected.partition.len(), 2);

    let accepted = merge_siblings(synthetic(&s, specs(), 3, 0.0), &0, &Default::default()).unwrap();
    let record = &accepted.merge.merges[0];
    assert_eq!(record.max_bond_dim, 2);
    assert_eq!(record.truncation_rms, 0.0);
    assert_dense(
        &accepted.partition,
        &dense(&s, |p| product_value(A, p)),
        1e-12,
    );
}

/// `A` on `s0 = 0` and `eps * B` on `s0 = 1`: their sum has rank two, and
/// dropping the small part costs `eps * ||B||_{s0 = 1}`.
fn unequal_halves(
    s: &[DynIndex],
    eps: f64,
    rms: f64,
    tau: f64,
) -> PatchedInterpolationResult<usize> {
    let mut small = B;
    small[0] = [eps * B[0][0], eps * B[0][1]];
    synthetic(
        s,
        vec![
            leaf(&[0], Some(product(s, A)), rms),
            leaf(&[1], Some(product(s, small)), rms),
        ],
        2,
        tau,
    )
}

#[test]
fn a_truncation_is_kept_only_within_the_slack() {
    let s = sites();
    let eps = 1e-3;
    let dropped = (dense(&s, |p| {
        if p[0] == 1 {
            eps * product_value(B, p)
        } else {
            0.0
        }
    }))
    .norm()
    .unwrap();
    let region_points = 16.0_f64;
    // Ample slack: the small half is truncated away.
    let tau = 100.0 * dropped / region_points.sqrt();
    let merged =
        merge_siblings(unequal_halves(&s, eps, 0.0, tau), &0, &Default::default()).unwrap();
    let record = &merged.merge.merges[0];
    assert_eq!(record.max_bond_dim, 1);
    let truncation = record.truncation_rms * region_points.sqrt();
    // The best rank-one approximation costs at most the small half.
    assert!(
        truncation > 0.0 && truncation <= dropped * (1.0 + 1e-9),
        "truncation {truncation:e}, dropped part {dropped:e}"
    );
    assert!(record.members_rms + record.truncation_rms <= tau);
    let exact = dense(&s, |p| {
        if p[0] == 0 {
            product_value(A, p)
        } else {
            eps * product_value(B, p)
        }
    });
    let actual = merged.partition.to_treetn().unwrap().to_dense().unwrap();
    let difference = actual.sub(&exact).unwrap().norm().unwrap();
    assert!((difference - truncation).abs() <= 1e-9 * dropped.max(difference));

    // The same truncation with members that use 99% of the allowance does not
    // fit: the exact sum is kept, and its rank reaches the cap.
    let tight = merge_siblings(
        unequal_halves(&s, eps, 0.99 * tau, tau),
        &0,
        &Default::default(),
    )
    .unwrap();
    assert!(tight.merge.merges.is_empty());
    assert_eq!(tight.merge.rejected_by_cap, 1);

    // So does a slack below the dropped part.
    let small = merge_siblings(
        unequal_halves(&s, eps, 0.0, 0.5 * dropped / region_points.sqrt()),
        &0,
        &Default::default(),
    )
    .unwrap();
    assert_eq!(small.merge.rejected_by_cap, 1);
}

#[test]
fn tolerance_not_met_blocks_the_group_and_its_ancestors() {
    let s = sites();
    let g = product(&s, A);
    let mut specs: Vec<Spec> = [[0, 0], [0, 1], [1, 0], [1, 1]]
        .iter()
        .map(|fixed| leaf(fixed, Some(g.clone()), 0.0))
        .collect();
    specs[1].rms = 2e-8;
    specs[1].status = PatchStatus::ToleranceNotMet;
    let merged = merge_siblings(synthetic(&s, specs, 2, 1e-8), &0, &Default::default()).unwrap();
    assert_eq!(merged.merge.rejected_by_status, 1);
    assert_eq!(merged.merge.merges.len(), 1);
    assert_eq!(merged.merge.merges[0].region.get(&s[0]), Some(1));
    assert_eq!(merged.partition.len(), 3);
    assert!(matches!(
        merged.merge.error.global,
        GlobalL2Error::ToleranceNotMet { .. }
    ));
}

#[test]
fn an_all_zero_group_becomes_a_zero_region() {
    let s = sites();
    let g = product(&s, A);
    let specs = vec![
        leaf(&[0], Some(g), 0.0),
        leaf(&[1, 0], None, 1e-9),
        leaf(&[1, 1], None, 0.0),
    ];
    let merged = merge_siblings(synthetic(&s, specs, 2, 1e-8), &0, &Default::default()).unwrap();
    let records = &merged.merge.merges;
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].region.get(&s[0]), Some(1));
    assert_eq!(records[0].max_bond_dim, 0);
    assert_eq!(records[0].truncation_rms, 0.0);
    assert!((records[0].members_rms - 1e-9 / 2.0_f64.sqrt()).abs() <= 1e-24);
    assert!(records[1].region.is_empty());
    assert_eq!(merged.partition.len(), 1);
    assert_dense(
        &merged.partition,
        &dense(&s, |p| if p[0] == 0 { product_value(A, p) } else { 0.0 }),
        1e-12,
    );
}

#[test]
fn merging_without_truncation_keeps_the_audited_estimate() {
    let term = |patch_points: f64, rms: f64, rel: f64| Term {
        patch_points,
        rms,
        certified: false,
        audited: Some(AuditedTerm {
            rms: 1.5 * rms,
            variance_rms: 2.0 * rms,
            mean_square_rel_std_error: rel,
        }),
        within_tolerance: true,
    };
    let members = [
        term(4.0, 1e-3, 0.2),
        term(4.0, 3e-3, 0.1),
        term(8.0, 2e-3, 0.3),
    ];
    let outside = term(16.0, 1e-3, 0.05);
    let separate = [members[0], members[1], members[2], outside];
    let merged = [Term::merged(&members, 0.0), outside];
    let (left, _) = global_error_of_terms(&separate, 32.0, 1.0, None, f64::EPSILON);
    let (right, _) = global_error_of_terms(&merged, 32.0, 1.0, None, f64::EPSILON);
    let (
        GlobalL2Error::Audited {
            rms_error_estimate: a,
            mean_square_rel_std_error: sa,
            ..
        },
        GlobalL2Error::Audited {
            rms_error_estimate: b,
            mean_square_rel_std_error: sb,
            ..
        },
    ) = (left, right)
    else {
        panic!("audited terms give an audited estimate");
    };
    assert!((a - b).abs() <= 1e-14 * a, "{a:e} vs {b:e}");
    assert!((sa - sb).abs() <= 1e-12 * sa, "{sa:e} vs {sb:e}");

    // A truncation adds to the estimate, not to its variance, and a member
    // without an audit leaves the region unaudited.
    let truncated = Term::merged(&members, 1e-4);
    let untruncated = Term::merged(&members, 0.0);
    assert_eq!(truncated.rms, untruncated.rms + 1e-4);
    let (t, u) = (truncated.audited.unwrap(), untruncated.audited.unwrap());
    assert_eq!(t.rms, u.rms + 1e-4);
    assert_eq!(t.variance_rms, u.variance_rms);
    let mut unaudited = members;
    unaudited[1].audited = None;
    assert!(Term::merged(&unaudited, 0.0).audited.is_none());
}

// ---------------------------------------------------------------------------
// Runs of patched_interpolate on a star with two binary sites per node.
// ---------------------------------------------------------------------------

/// `sin(40 x^2)` on 256 points, `x` read from the eight bits in split order.
fn chirp(point: &[usize]) -> f64 {
    let x = point.iter().fold(0.0, |a, &b| 2.0 * a + b as f64) / 256.0;
    (40.0 * x * x).sin()
}

/// Run on the star with two binary sites per node; the derived site order is
/// the node order, which is also the split order.
fn run_star(
    cap: usize,
    norm: ErrorNorm,
    rtol: f64,
) -> (PatchedInterpolationResult<usize>, Vec<DynIndex>) {
    let mut topology = NodeNameNetwork::new();
    for node in 0..4usize {
        topology.add_node(node).unwrap();
    }
    for leaf in 1..4usize {
        topology.add_edge(&0, &leaf).unwrap();
    }
    let mut node_sites = BTreeMap::new();
    let mut all = Vec::new();
    for node in 0..4usize {
        let sites: Vec<DynIndex> = (0..2).map(|_| DynIndex::new_dyn(2)).collect();
        all.extend(sites.iter().cloned());
        node_sites.insert(node, sites);
    }
    let options = PatchedInterpolationOptions::new(cap)
        .with_error_norm(norm)
        .with_tolerance(ErrorTolerance { rtol, atol: 0.0 })
        .with_verification(VerificationOptions::new().with_max_exhaustive_points(1 << 12))
        .with_patch_order(all.clone());
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let mut ctx = EvalContext::new(Session::new(backend.engine_mut()), Parallelism::Serial);
    let result = patched_interpolate(
        &TreeTciInterpolator::default(),
        topology,
        node_sites,
        ColMajorArray::new(vec![], vec![8, 0]).unwrap(),
        &mut ctx,
        |batch: ColMajorArrayRef<'_, usize>, _ctx: &mut EvalContext<'_>| {
            Ok(batch.data().chunks(8).map(chirp).collect())
        },
        &options,
    )
    .unwrap();
    (result, all)
}

fn chirp_norm() -> f64 {
    (0..256usize)
        .map(|i| {
            let point: Vec<usize> = (0..8).map(|k| (i >> (7 - k)) & 1).collect();
            chirp(&point).powi(2)
        })
        .sum::<f64>()
        .sqrt()
}

#[test]
fn a_merged_interpolation_keeps_its_l2_tolerance() {
    let reference = chirp_norm();
    let rtol = 1e-3;
    let (result, sites) = run_star(4, ErrorNorm::l2(L2Reference::Given(reference)), rtol);
    let patches = result.partition.len();
    let tau = match result.report.norm {
        NormReport::L2 { tau, .. } => tau,
        _ => unreachable!(),
    };
    let merged = merge_siblings(result, &0, &SiblingMergeOptions::new()).unwrap();
    assert!(!merged.merge.merges.is_empty());
    assert_eq!(merged.partition.len(), patches - merged.merge.merges.len());
    for record in &merged.merge.merges {
        assert!(record.max_bond_dim < 4);
        assert!(record.members_rms + record.truncation_rms <= tau * (1.0 + 1e-12));
    }
    let GlobalL2Error::Certified { rms_error, .. } = merged.merge.error.global else {
        panic!("exhaustive members give a certified merged result");
    };
    assert!(rms_error <= tau * (1.0 + 1e-12));
    // The dense error against f meets the run's allowance delta = rtol * ||f||.
    // Both the run and `dense` pass points in site order, which `chirp` reads.
    let exact = dense(&sites, chirp);
    let actual = merged.partition.to_treetn().unwrap().to_dense().unwrap();
    let error = actual.sub(&exact).unwrap().norm().unwrap();
    assert!(
        error <= rtol * reference * (1.0 + 1e-9),
        "L2 error {error:e} above delta {:e}",
        rtol * reference
    );
    // The certified value bounds the true RMS error (256 points), up to rounding.
    assert!(error / 16.0 <= rms_error + 1e-12 * reference);
}

#[test]
fn without_a_merge_the_error_is_the_runs_and_reassembles_bit_for_bit() {
    let reference = chirp_norm();
    let (result, _) = run_star(3, ErrorNorm::l2(L2Reference::Given(reference)), 1e-2);
    let before = run_error(&result);
    let tau = match result.report.norm {
        NormReport::L2 { tau, .. } => tau,
        _ => unreachable!(),
    };
    // Reassembling the run's own leaves reproduces its error exactly, so the
    // leaf order and terms are the run's.
    let leaves = initial_leaves(result.partition.clone(), &result.report).unwrap();
    let leaves: Vec<_> = leaves.into_values().collect();
    let reassembled = merged_error(&leaves, before.domain_points, tau).unwrap();
    assert_eq!(reassembled, before);

    let patches = result.partition.len();
    let merged = merge_siblings(result, &0, &SiblingMergeOptions::new()).unwrap();
    assert!(merged.merge.merges.is_empty());
    assert!(merged.merge.rejected_by_cap > 0);
    assert_eq!(merged.partition.len(), patches);
    assert_eq!(merged.merge.error, before);
}

#[test]
fn invalid_inputs_fail_before_any_merge() {
    let (sampled, _) = run_star(4, ErrorNorm::sampled_max(), 1e-3);
    let error = merge_siblings(sampled, &0, &SiblingMergeOptions::new()).unwrap_err();
    assert!(matches!(
        error,
        PatchedInterpolationError::InvalidInput { .. }
    ));

    let reference = chirp_norm();
    let run = || run_star(4, ErrorNorm::l2(L2Reference::Given(reference)), 1e-3).0;
    for cap in [0, 5] {
        let options = SiblingMergeOptions::new().with_max_bond_dim(cap);
        let error = merge_siblings(run(), &0, &options).unwrap_err();
        assert!(
            matches!(error, PatchedInterpolationError::InvalidInput { .. }),
            "{cap}"
        );
    }
    let error = merge_siblings(run(), &99, &SiblingMergeOptions::new()).unwrap_err();
    assert!(matches!(
        error,
        PatchedInterpolationError::Partition {
            source: PartitionedTreeTNError::InvalidCenter
        }
    ));
}
