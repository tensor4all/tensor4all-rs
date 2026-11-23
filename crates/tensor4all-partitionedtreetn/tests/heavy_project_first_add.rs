//! Full parameter grid of the discard-free project-first check in
//! `project_first_add.rs`.
//!
//! The ordinary test pins one bond dimension, decay, cutoff and cap for both
//! allocations. This file runs every combination of bond dimension and decay
//! `(3, 0.3)`, `(4, 0.5)`, `(3, 0.15)`, both allocations, cutoffs `1e-4`,
//! `1e-3`, `1e-2`, `3e-2` and caps 1, 2, 3, so a panic, an error or a lossy split
//! in any of them still fails a scheduled run. CONTRIBUTING.md, "CI time
//! budget", has the selection command and the budgets this file stays outside of.

use super::project_first_add::{exact_pair, options, relative_squared_error};
use tensor4all_partitionedtreetn::{add_with_patching, BudgetAllocation, PartitionedTreeTN};

/// Where the closing leaf pass has nothing to discard, a project-first result
/// is exact to round-off. The comparison is the leaf-wise project-first
/// baseline: project the exact sum onto the routine's own final leaves and
/// truncate that partition once with the same cutoff, allocation and cap.
/// Combinations whose baseline itself discards weight are skipped; at least one
/// combination must reach the discard-free case.
#[test]
#[ignore = "heavy: runs in the scheduled heavy-tests workflow (see CONTRIBUTING.md)"]
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
