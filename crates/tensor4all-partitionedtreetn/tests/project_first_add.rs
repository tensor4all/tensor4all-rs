//! Project-first `add_with_patching` (tensor4all/tensor4all-rs#1131).
//!
//! The pre-#1131 routine budget-truncated every patch before each split and
//! split that already lossy value, so the losses of successive split levels
//! added up. The cases below are the reproducer's worst ones: they fail unless
//! the split children come from the exact addends and the cutoff is spent once,
//! on the final leaves.
//!
//! The thresholds below are the errors measured at `56c25d4a` (the pre-fix
//! revision) with the issue's reproducer, relative squared error
//! `||F - sum_p R_p||^2 / ||F||^2`: 1.3056e-3 for `Volume` and 1.1479e-2 for
//! `Norm`. `docs/worklogs/2026-11-04-project-first-add.md` records the post-fix
//! numbers and the cost.

use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_partitionedtreetn::{
    add_with_patching, BudgetAllocation, PartitionedTreeTN, PatchSplitStrategy, PatchingOptions,
    SubDomainTreeTN,
};
use tensor4all_treetn::TreeTN;

type Fallible<T> = Result<T, Box<dyn std::error::Error>>;

/// Deterministic uniform generator, so the cases do not depend on a dev-dep.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
    }
}

/// Chain TreeTN over `sites` with bond dimension `r`; bond channel `k` is
/// scaled by `decay^k`.
fn chain(sites: &[DynIndex], r: usize, decay: f64, seed: u64) -> Fallible<SubDomainTreeTN<usize>> {
    let mut rng = Lcg(seed);
    let n = sites.len();
    let bonds: Vec<DynIndex> = (0..n - 1).map(|_| DynIndex::new_dyn(r)).collect();
    let mut tensors = Vec::new();
    for i in 0..n {
        let mut indices = Vec::new();
        if i > 0 {
            indices.push(bonds[i - 1].clone());
        }
        indices.push(sites[i].clone());
        if i + 1 < n {
            indices.push(bonds[i].clone());
        }
        let dims: Vec<usize> = indices.iter().map(|index| index.dim).collect();
        let total: usize = dims.iter().product();
        // Column-major: the first index is the fastest.
        let data: Vec<f64> = (0..total)
            .map(|flat| {
                let right = if i + 1 < n {
                    flat / (dims[0] * if i > 0 { dims[1] } else { 1 })
                } else {
                    0
                };
                rng.next() * decay.powi(right as i32)
            })
            .collect();
        tensors.push(IdxTensor::from_dense(indices, data)?);
    }
    let tree = TreeTN::from_tensors(tensors, (0..n).collect())?;
    Ok(SubDomainTreeTN::from_treetn(tree)?)
}

/// Relative squared error of a partition against the exact sum, from inner
/// products only.
fn relative_squared_error(
    exact: &SubDomainTreeTN<usize>,
    result: &PartitionedTreeTN<usize>,
) -> Fallible<f64> {
    let exact_squared = exact.norm_squared()?;
    let mut cross = 0.0;
    let mut result_squared = 0.0;
    for patch in result.values() {
        cross += exact.inner(patch)?.real();
        result_squared += patch.norm_squared()?;
    }
    Ok((exact_squared - 2.0 * cross + result_squared) / exact_squared)
}

/// The same error from a dense subtraction, as an independent check of the
/// inner-product expression above.
fn dense_relative_squared_error(
    exact: &SubDomainTreeTN<usize>,
    result: &PartitionedTreeTN<usize>,
) -> Fallible<f64> {
    let exact_data = exact.data().clone().to_dense()?.to_vec::<f64>()?;
    let result_data = result.to_treetn()?.to_dense()?.to_vec::<f64>()?;
    let difference: f64 = exact_data
        .iter()
        .zip(&result_data)
        .map(|(a, b)| (a - b) * (a - b))
        .sum();
    let norm: f64 = exact_data.iter().map(|value| value * value).sum();
    Ok(difference / norm)
}

/// Two random addends with the same key over the six-site chain, and their
/// exact sum.
fn exact_pair(
    r: usize,
    decay: f64,
) -> Fallible<(SubDomainTreeTN<usize>, Vec<SubDomainTreeTN<usize>>)> {
    let sites: Vec<DynIndex> = (0..6).map(|_| DynIndex::new_dyn(2)).collect();
    let addends = vec![chain(&sites, r, decay, 11)?, chain(&sites, r, decay, 23)?];
    let exact = addends[0].add(&addends[1])?;
    Ok((exact, addends))
}

fn options(allocation: BudgetAllocation, cutoff: f64, cap: usize) -> PatchingOptions {
    PatchingOptions {
        cutoff,
        allocation,
        max_bond_dim: Some(cap),
        patch_order: Vec::new(),
        split_strategy: PatchSplitStrategy::Sequential,
        ..PatchingOptions::default()
    }
}

#[test]
fn volume_worst_case_keeps_the_project_first_error() {
    let (exact, addends) = exact_pair(3, 0.15).unwrap();
    let result =
        add_with_patching(addends, &2, &options(BudgetAllocation::Volume, 1.0e-3, 1)).unwrap();

    let error = relative_squared_error(&exact, &result).unwrap();
    let dense = dense_relative_squared_error(&exact, &result).unwrap();
    assert!(
        (error - dense).abs() < 1.0e-12,
        "inner-product error {error:e} disagrees with the dense error {dense:e}"
    );
    assert!(result.values().all(|patch| patch.max_bond_dim() <= 1));
    // The pre-#1131 routine measured 1.3056e-3 on this case. Anything at or
    // above that level means the split children came from the truncated parent
    // again.
    assert!(error < 1.0e-4, "relative squared error {error:e}");
}

#[test]
fn norm_worst_case_keeps_the_project_first_error() {
    let (exact, addends) = exact_pair(3, 0.3).unwrap();
    let result =
        add_with_patching(addends, &2, &options(BudgetAllocation::Norm, 1.0e-2, 1)).unwrap();

    let error = relative_squared_error(&exact, &result).unwrap();
    let dense = dense_relative_squared_error(&exact, &result).unwrap();
    assert!(
        (error - dense).abs() < 1.0e-10,
        "inner-product error {error:e} disagrees with the dense error {dense:e}"
    );
    assert!(result.values().all(|patch| patch.max_bond_dim() <= 1));
    // Pre-#1131: 1.1479e-2.
    assert!(error < 1.0e-3, "relative squared error {error:e}");
}

/// Where the closing leaf pass has nothing to discard, a project-first result
/// is exact to round-off; the pre-#1131 routine loses weight at the split
/// levels there. The comparison is the leaf-wise project-first baseline: project
/// the exact sum onto the routine's own final leaves and truncate that partition
/// once with the same cutoff, allocation and cap.
#[test]
fn splits_are_exact_where_the_leaf_pass_discards_nothing() {
    let mut checked = 0;
    for (r, decay) in [(3usize, 0.3f64), (4, 0.5), (3, 0.15)] {
        let (exact, addends) = exact_pair(r, decay).unwrap();
        for allocation in [BudgetAllocation::Volume, BudgetAllocation::Norm] {
            for cutoff in [1.0e-4, 1.0e-3, 1.0e-2, 3.0e-2] {
                for cap in [1usize, 2, 3] {
                    let result =
                        add_with_patching(addends.clone(), &2, &options(allocation, cutoff, cap))
                            .unwrap();
                    let leaves = PartitionedTreeTN::from_subdomains(
                        result
                            .projectors()
                            .map(|projector| exact.project(projector))
                            .collect::<Result<Vec<_>, _>>()
                            .unwrap()
                            .into_iter()
                            .flatten()
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                    let baseline = tensor4all_partitionedtreetn::truncate_adaptive(
                        &leaves,
                        &2,
                        cutoff,
                        allocation,
                        Some(cap),
                    )
                    .unwrap();
                    if relative_squared_error(&exact, &baseline).unwrap() >= 1.0e-12 {
                        continue;
                    }
                    checked += 1;
                    let error = relative_squared_error(&exact, &result).unwrap();
                    assert!(
                        error < 1.0e-12,
                        "{allocation:?} cutoff {cutoff:e} cap {cap} r {r} decay {decay}: \
                         the leaf pass discards nothing but the error is {error:e}"
                    );
                }
            }
        }
    }
    assert!(
        checked > 0,
        "no case reached the discard-free leaf situation; the loop no longer covers it"
    );
}

/// A leaf that is already below its budget must not enlarge the other leaves'
/// Volume budgets: it stays in the result set until the closing pass, which
/// drops it and derives every threshold from the whole input weight.
#[test]
fn a_dropped_low_weight_patch_does_not_enlarge_the_others_budget() {
    // Squared spectra [93, 7] (norm^2 100) and [1, 0] (norm^2 1) over equal
    // volumes: ||F||^2 = 101, so each patch's budget is 0.1 * 101 / 2 = 5.05.
    // The light patch is dropped, and the heavy patch keeps its 7-weight mode
    // because 7 > 5.05. Deriving the budget from the survivor alone would give
    // 0.1 * 100 = 10 and discard that mode.
    let patches = super::patching::two_mode_chain([[93.0, 7.0], [1.0, 0.0]]).patches;
    let result = add_with_patching(
        patches.to_vec(),
        &0,
        &PatchingOptions {
            cutoff: 0.1,
            allocation: BudgetAllocation::Volume,
            max_bond_dim: None,
            ..PatchingOptions::default()
        },
    )
    .unwrap();

    assert_eq!(result.len(), 1, "the light patch is dropped");
    let kept: f64 = result
        .values()
        .map(|patch| patch.norm_squared().unwrap())
        .sum();
    // Only the dropped patch's weight 1 of 101 is missing; a budget of 10 would
    // also have discarded the heavy patch's weight-7 mode and left 93.
    assert!((kept - 100.0).abs() < 1.0e-9, "kept weight {kept}");
}
