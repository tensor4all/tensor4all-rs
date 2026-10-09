//! Public correctness and resource-contract regressions for issue #854.

use num_complex::{Complex32, Complex64};
use tensor4all_core::{DynIndex, IdxTensor, IndexLike, Scalar};
use tensor4all_treeaci::{
    hadamard_many, tree_elementwise, tree_elementwise_batched, TreeAciError, TreeAciOptions,
    TreeAciScalar, TreeAciTermination,
};
use tensor4all_treetn::TreeTN;

trait AuditScalar: TreeAciScalar {
    fn value(re: f64, im: f64) -> Self;
    fn tolerance() -> f64;
}
impl AuditScalar for f64 {
    fn value(re: f64, _: f64) -> Self {
        re
    }
    fn tolerance() -> f64 {
        1e-10
    }
}
impl AuditScalar for f32 {
    fn value(re: f64, _: f64) -> Self {
        re as f32
    }
    fn tolerance() -> f64 {
        2e-5
    }
}
impl AuditScalar for Complex64 {
    fn value(re: f64, im: f64) -> Self {
        Self::new(re, im)
    }
    fn tolerance() -> f64 {
        1e-10
    }
}
impl AuditScalar for Complex32 {
    fn value(re: f64, im: f64) -> Self {
        Self::new(re as f32, im as f32)
    }
    fn tolerance() -> f64 {
        2e-5
    }
}

fn tree<T: AuditScalar>(
    edges: &[(usize, usize)],
    sites: &[Vec<DynIndex>],
    rank: usize,
    variant: usize,
    reverse: bool,
) -> TreeTN<IdxTensor, usize> {
    let bonds: Vec<_> = edges.iter().map(|_| DynIndex::new_dyn(rank)).collect();
    let nodes: Vec<_> = if reverse {
        (0..sites.len()).rev().collect()
    } else {
        (0..sites.len()).collect()
    };
    let tensors = nodes
        .iter()
        .map(|&node| {
            let mut axes = sites[node].clone();
            for (e, &(a, b)) in edges.iter().enumerate() {
                if node == a || node == b {
                    axes.push(bonds[e].clone());
                }
            }
            if reverse {
                axes.reverse();
            }
            let length: usize = axes.iter().map(IndexLike::dim).product();
            let values = (0..length)
                .map(|flat| {
                    let mut q = flat;
                    let mut angle = 0.37 * (node + 1 + 2 * variant) as f64;
                    for axis in &axes {
                        let coordinate = q % axis.dim();
                        q /= axis.dim();
                        let weight = sites[node]
                            .iter()
                            .position(|s| s == axis)
                            .map(|p| 0.39 + 0.23 * p as f64)
                            .unwrap_or_else(|| {
                                0.61 + 0.17 * bonds.iter().position(|b| b == axis).unwrap() as f64
                            });
                        angle += weight * coordinate as f64;
                    }
                    let divisor = if node == 0 { 1.0 } else { rank as f64 };
                    T::value(
                        (1.0 + 0.27 * angle.sin()) / divisor,
                        0.07 * angle.cos() / divisor,
                    )
                })
                .collect::<Vec<_>>();
            IdxTensor::from_dense(axes, values).unwrap()
        })
        .collect();
    TreeTN::from_tensors(tensors, nodes).unwrap()
}

fn nonlinear<T: AuditScalar>(a: T, b: T) -> T {
    (a + T::value(0.31, -0.13) * b) / (T::value(2.7, 0.21) + a * b)
}

fn dense<T: AuditScalar>(tree: &TreeTN<IdxTensor, usize>, sites: &[DynIndex]) -> Vec<T> {
    tree.to_dense()
        .unwrap()
        .permute_indices(sites)
        .unwrap()
        .to_vec::<T>()
        .unwrap()
}

fn dense_suite<T: AuditScalar>() {
    type Geometry = (&'static str, Vec<(usize, usize)>, Vec<Vec<usize>>);
    let geometries: Vec<Geometry> = vec![
        ("scalar", vec![], vec![vec![]]),
        ("single-multiple-axes", vec![], vec![vec![2, 3]]),
        ("two-node", vec![(0, 1)], vec![vec![2], vec![3]]),
        (
            "path",
            vec![(0, 1), (1, 2), (2, 3), (3, 4)],
            vec![vec![2]; 5],
        ),
        (
            "star4",
            vec![(0, 1), (0, 2), (0, 3), (0, 4)],
            vec![vec![2]; 5],
        ),
        (
            "star5-site-free",
            vec![(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)],
            vec![vec![], vec![2], vec![2], vec![2], vec![2], vec![2]],
        ),
        (
            "comb",
            vec![(0, 1), (1, 2), (0, 3), (1, 4), (2, 5)],
            vec![vec![2]; 6],
        ),
        (
            "mixed-axes",
            vec![(0, 1), (1, 2), (1, 3)],
            vec![vec![2, 3], vec![], vec![1], vec![2, 2]],
        ),
        ("all-site-free", vec![(0, 1), (0, 2)], vec![vec![]; 3]),
    ];
    let mut failures = Vec::new();
    let mut cases = 0usize;
    for (name, edges, dims) in geometries {
        let physical: Vec<Vec<_>> = dims
            .iter()
            .map(|ds| ds.iter().map(|&d| DynIndex::new_dyn(d)).collect())
            .collect();
        let axes: Vec<_> = physical.iter().flatten().cloned().collect();
        for reverse in [false, true] {
            let inputs = [
                tree::<T>(&edges, &physical, 1, 0, reverse),
                tree::<T>(&edges, &physical, 2, 1, !reverse),
            ];
            let operands: Vec<_> = inputs.iter().map(|t| dense::<T>(t, &axes)).collect();
            for seed in [0, 7, 42] {
                let options = TreeAciOptions {
                    tolerance: T::tolerance(),
                    global_tolerance_margin: 1.0,
                    nsearch_global_pivots: 16,
                    max_nglobal_pivots: 16,
                    nsweeps_global_search: 30,
                    max_sweeps: 30,
                    rng_seed: seed,
                    root: reverse.then_some(0),
                    ..Default::default()
                };
                for product in [false, true] {
                    cases += 1;
                    let result = if product {
                        hadamard_many::<T, _>(&inputs, &options)
                    } else {
                        tree_elementwise::<T, _, _>(|v| nonlinear(v[0], v[1]), &inputs, &options)
                    };
                    match result {
                        Ok(result) => {
                            let values = dense::<T>(&result.tree, &axes);
                            let scale = operands[0]
                                .iter()
                                .zip(&operands[1])
                                .map(|(&a, &b)| if product { a * b } else { nonlinear(a, b) })
                                .map(Scalar::abs_val)
                                .fold(0.0_f64, f64::max);
                            let error = values
                                .iter()
                                .zip(&operands[0])
                                .zip(&operands[1])
                                .map(|((&got, &a), &b)| {
                                    let expected = if product { a * b } else { nonlinear(a, b) };
                                    Scalar::abs_val(got - expected)
                                })
                                .fold(0.0_f64, f64::max);
                            let finite = values.iter().all(|&v| Scalar::abs_val(v).is_finite());
                            let ok = finite && error <= 10.0 * T::tolerance() * scale.max(1.0);
                            eprintln!("DENSE_AUDIT type={} geometry={name} reverse={reverse} seed={seed} product={product} termination={:?} error={error:e} scale={scale:e} pass={}",
                                std::any::type_name::<T>(), result.termination, result.max_ranks.len());
                            if !ok {
                                failures.push(format!("{name}/{reverse}/{seed}/{product}: error={error:e}, finite={finite}"));
                            }
                        }
                        Err(e) => failures.push(format!("{name}/{reverse}/{seed}/{product}: {e}")),
                    }
                }
            }
        }
    }
    eprintln!(
        "DENSE_AUDIT_SUMMARY type={} cases={cases} failures={failures:?}",
        std::any::type_name::<T>()
    );
    assert!(failures.is_empty(), "{failures:?}");
}

#[test]
fn dense_oracle_f64() {
    dense_suite::<f64>();
}
#[test]
fn dense_oracle_f32() {
    dense_suite::<f32>();
}
#[test]
fn dense_oracle_c64() {
    dense_suite::<Complex64>();
}
#[test]
fn dense_oracle_c32() {
    dense_suite::<Complex32>();
}

#[test]
fn single_site_nonfinite_callback_must_error() {
    let sites = vec![vec![DynIndex::new_dyn(2)]];
    let input = tree::<f64>(&[], &sites, 1, 0, false);
    let mut failures = Vec::new();
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let result = tree_elementwise::<f64, _, _>(
            |_| bad,
            std::slice::from_ref(&input),
            &TreeAciOptions::default(),
        );
        if let Ok(result) = result {
            eprintln!(
                "SINGLE_SITE_NONFINITE bad={bad:?} termination={:?} values={:?}",
                result.termination,
                dense::<f64>(&result.tree, &sites[0])
            );
            failures.push(bad);
        }
    }
    assert!(
        failures.is_empty(),
        "non-finite callback outputs were accepted: {failures:?}"
    );
}

#[test]
fn single_site_hadamard_overflow_must_error() {
    let site = DynIndex::new_dyn(2);
    let input = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![site], vec![1e200_f64; 2]).unwrap()],
        vec![0usize],
    )
    .unwrap();
    let result = hadamard_many::<f64, _>(&[input.clone(), input], &TreeAciOptions::default());
    eprintln!("SINGLE_SITE_OVERFLOW result={result:?}");
    assert!(
        result.is_err(),
        "finite inputs overflowing in the operator must not converge"
    );
}

#[test]
fn complex32_relative_extreme_scales_must_work() {
    let sites = vec![vec![DynIndex::new_dyn(2)], vec![DynIndex::new_dyn(2)]];
    let input = tree::<Complex32>(&[(0, 1)], &sites, 1, 0, false);
    let mut failures = Vec::new();
    for scale in [1e-30_f32, 1e-10, 1.0, 1e10, 1e30] {
        let value = Complex32::new(scale, 0.5 * scale);
        let result = tree_elementwise::<Complex32, _, _>(
            |_| value,
            std::slice::from_ref(&input),
            &TreeAciOptions {
                tolerance: 1e-5,
                ..Default::default()
            },
        );
        match result {
            Ok(result) => {
                let axes: Vec<_> = sites.iter().flatten().cloned().collect();
                let actual = dense::<Complex32>(&result.tree, &axes);
                let error = actual
                    .iter()
                    .map(|&v| Scalar::abs_val(v - value) / Scalar::abs_val(value))
                    .fold(0.0_f64, f64::max);
                eprintln!(
                    "C32_SCALE scale={scale:e} termination={:?} error={error:e}",
                    result.termination
                );
                if !actual.iter().all(|&v| Scalar::abs_val(v).is_finite()) || error > 1e-4 {
                    failures.push(format!("{scale:e}: error {error:e}"));
                }
            }
            Err(e) => {
                eprintln!("C32_SCALE scale={scale:e} error={e}");
                failures.push(format!("{scale:e}: {e}"));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}

#[test]
fn complex64_extreme_scale_modes_and_real_controls() {
    let sites = vec![vec![DynIndex::new_dyn(2)], vec![DynIndex::new_dyn(2)]];
    let real_input = tree::<f64>(&[(0, 1)], &sites, 1, 0, false);
    let complex_input = tree::<Complex64>(&[(0, 1)], &sites, 1, 0, false);
    let axes: Vec<_> = sites.iter().flatten().cloned().collect();
    let mut failures = Vec::new();
    for scale in [1e-200_f64, 1e-100, 1.0, 1e100, 1e200] {
        for relative in [false, true] {
            let options = TreeAciOptions {
                scale_tolerance: relative,
                tolerance: if relative { 1e-12 } else { 1e-12 * scale },
                ..Default::default()
            };
            let real = tree_elementwise::<f64, _, _>(
                |_| scale,
                std::slice::from_ref(&real_input),
                &options,
            );
            match real {
                Ok(result) => {
                    let error = dense::<f64>(&result.tree, &axes)
                        .iter()
                        .map(|v| ((v - scale) / scale).abs())
                        .fold(0.0_f64, f64::max);
                    eprintln!("EXTREME_SCALE type=f64 scale={scale:e} relative={relative} error={error:e}");
                    if !error.is_finite() || error > 1e-10 {
                        failures.push(format!("f64 {scale:e}/{relative}: {error}"));
                    }
                }
                Err(e) => failures.push(format!("f64 {scale:e}/{relative}: {e}")),
            }
            let expected = Complex64::new(scale, 0.5 * scale);
            let complex = tree_elementwise::<Complex64, _, _>(
                |_| expected,
                std::slice::from_ref(&complex_input),
                &options,
            );
            match complex {
                Ok(result) => {
                    let error = dense::<Complex64>(&result.tree, &axes)
                        .iter()
                        .map(|v| {
                            Complex64::new(
                                (v.re - expected.re) / scale,
                                (v.im - expected.im) / scale,
                            )
                            .norm()
                        })
                        .fold(0.0_f64, f64::max);
                    eprintln!("EXTREME_SCALE type=c64 scale={scale:e} relative={relative} error={error:e}");
                    if !error.is_finite() || error > 1e-10 {
                        failures.push(format!("c64 {scale:e}/{relative}: {error}"));
                    }
                }
                Err(e) => {
                    eprintln!(
                        "EXTREME_SCALE type=c64 scale={scale:e} relative={relative} error={e}"
                    );
                    failures.push(format!("c64 {scale:e}/{relative}: {e}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}

#[test]
fn rank_cap_reports_inaccurate_output() {
    let sites = vec![vec![DynIndex::new_dyn(2)], vec![DynIndex::new_dyn(2)]];
    let inputs = [
        tree::<f64>(&[(0, 1)], &sites, 1, 0, false),
        tree::<f64>(&[(0, 1)], &sites, 2, 1, true),
    ];
    let result = tree_elementwise::<f64, _, _>(
        |v| nonlinear(v[0], v[1]),
        &inputs,
        &TreeAciOptions {
            max_bond_dim: Some(1),
            tolerance: 1e-12,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(result.termination, TreeAciTermination::RankLimited);
    assert_eq!(result.diagnostics.edge_ranks[0].2, 1);
}

fn sparse_delta_three_sites() -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
    let sites: Vec<_> = (0..3).map(|_| DynIndex::new_dyn(2)).collect();
    let bonds = [DynIndex::new_dyn(1), DynIndex::new_dyn(1)];
    let input = TreeTN::from_tensors(
        vec![
            IdxTensor::from_dense(vec![sites[0].clone(), bonds[0].clone()], vec![0.0_f64, 1.0])
                .unwrap(),
            IdxTensor::from_dense(
                vec![bonds[0].clone(), sites[1].clone(), bonds[1].clone()],
                vec![0.0_f64, 1.0],
            )
            .unwrap(),
            IdxTensor::from_dense(vec![bonds[1].clone(), sites[2].clone()], vec![0.0_f64, 1.0])
                .unwrap(),
        ],
        vec![0usize, 1, 2],
    )
    .unwrap();
    (input, sites)
}

#[test]
fn saturated_cuts_must_not_silently_bypass_enabled_guard() {
    let (input, sites) = sparse_delta_three_sites();
    let expected = dense::<f64>(&input, &sites);
    for cap in [None, Some(1)] {
        let options = TreeAciOptions {
            max_bond_dim: cap,
            nsearch_global_pivots: 128,
            max_nglobal_pivots: 16,
            global_tolerance_margin: 1.0,
            ..Default::default()
        };
        let result =
            tree_elementwise::<f64, _, _>(|v| v[0], std::slice::from_ref(&input), &options)
                .unwrap();
        let actual = dense::<f64>(&result.tree, &sites);
        let error = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        eprintln!("SATURATED_GUARD cap={cap:?} termination={:?} error={error:e} pivots={:?} evaluated_points={} values={actual:?}",
            result.termination, result.global_pivots_found, result.diagnostics.evaluated_points);
        assert!(
            error <= options.tolerance || result.termination != TreeAciTermination::Converged,
            "rank-one target returned an incorrect converged rank-one approximation"
        );
        if cap.is_some() {
            assert_eq!(result.termination, TreeAciTermination::RankLimited);
            assert!(result.global_pivots_found.iter().any(|&found| found > 0));
        } else {
            assert_eq!(result.termination, TreeAciTermination::Converged);
            assert!(error <= options.tolerance);
        }
    }

    let accurate = tree_elementwise::<f64, _, _>(
        |_| 1.0,
        std::slice::from_ref(&input),
        &TreeAciOptions {
            max_bond_dim: Some(1),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(accurate.termination, TreeAciTermination::Converged);
    assert_eq!(dense::<f64>(&accurate.tree, &sites), vec![1.0; 8]);
}

#[test]
fn nonfinite_seen_only_by_guard_must_error_publicly() {
    let (input, sites) = sparse_delta_three_sites();
    let options = TreeAciOptions {
        nsearch_global_pivots: 128,
        max_nglobal_pivots: 16,
        global_tolerance_margin: 1.0,
        ..Default::default()
    };
    let mut accepted = Vec::new();
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut observed = false;
        let result = tree_elementwise::<f64, _, _>(
            |v| {
                if v[0] > 0.5 {
                    observed = true;
                    bad
                } else {
                    1.0
                }
            },
            std::slice::from_ref(&input),
            &options,
        );
        if let Ok(ref result) = result {
            eprintln!("PUBLIC_GUARD_NONFINITE bad={bad:?} observed={observed} termination={:?} values={:?}",
                result.termination, dense::<f64>(&result.tree, &sites));
            accepted.push(bad);
        }
        assert!(
            observed,
            "the counterexample must actually evaluate the non-finite target"
        );
    }
    assert!(
        accepted.is_empty(),
        "evaluated non-finite targets were accepted: {accepted:?}"
    );
}

#[test]
fn large_sweep_limit_must_not_panic_before_an_early_converging_run() {
    let sites = vec![vec![DynIndex::new_dyn(2)], vec![DynIndex::new_dyn(2)]];
    let input = tree::<f64>(&[(0, 1)], &sites, 1, 0, false);
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        tree_elementwise::<f64, _, _>(
            |_| 1.0,
            &[input],
            &TreeAciOptions {
                max_sweeps: usize::MAX,
                ..Default::default()
            },
        )
    }));
    assert!(
        outcome.is_ok(),
        "an oversized stopping limit must return an error or converge without panicking"
    );
    assert_eq!(
        outcome.unwrap().unwrap().termination,
        TreeAciTermination::Converged
    );
}

#[test]
fn local_nonfinite_outputs_must_error() {
    let sites = vec![vec![DynIndex::new_dyn(2)], vec![DynIndex::new_dyn(2)]];
    let input = tree::<f64>(&[(0, 1)], &sites, 1, 0, false);
    let mut failures = Vec::new();
    for relative in [false, true] {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let result = tree_elementwise::<f64, _, _>(
                |_| bad,
                std::slice::from_ref(&input),
                &TreeAciOptions {
                    scale_tolerance: relative,
                    ..Default::default()
                },
            );
            eprintln!("LOCAL_NONFINITE relative={relative} bad={bad:?} result={result:?}");
            if result.is_ok() {
                failures.push((relative, bad));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "non-finite local targets were accepted: {failures:?}"
    );
}

#[test]
fn local_luci_payload_must_fit_working_budget() {
    let n = 32usize;
    let sites = [DynIndex::new_dyn(n), DynIndex::new_dyn(n)];
    let make = |row: bool| {
        let bond = DynIndex::new_dyn(1);
        let coordinates: Vec<_> = (0..n).map(|p| p as f64).collect();
        TreeTN::from_tensors(
            vec![
                IdxTensor::from_dense(
                    vec![sites[0].clone(), bond.clone()],
                    if row {
                        coordinates.clone()
                    } else {
                        vec![1.0; n]
                    },
                )
                .unwrap(),
                IdxTensor::from_dense(
                    vec![bond, sites[1].clone()],
                    if row { vec![1.0; n] } else { coordinates },
                )
                .unwrap(),
            ],
            vec![0usize, 1],
        )
        .unwrap()
    };
    // The original preflight charged input/output arrays and rank-one frames,
    // but not the simultaneously retained full-rank rrLU L/U matrices.
    let current_charge = (3 * n * n + 4 * n) * std::mem::size_of::<f64>();
    let lu_scalar_lower_bound = 5 * n * n * std::mem::size_of::<f64>();
    assert!(lu_scalar_lower_bound > current_charge);
    let inputs = [make(true), make(false)];
    let mut called = false;
    let result = tree_elementwise::<f64, _, _>(
        |v| {
            called = true;
            if v[0] == v[1] {
                1.0
            } else {
                0.0
            }
        },
        &inputs,
        &TreeAciOptions {
            max_working_bytes: current_charge,
            enable_global_guard: false,
            max_core_elements: Some(n * n),
            max_local_matrix_elements: Some(n * n),
            ..Default::default()
        },
    );
    assert!(
        matches!(
            result,
            Err(TreeAciError::ResourceLimit {
                resource: "working bytes",
                ..
            })
        ),
        "a budget smaller than the factorization's live scalar payload was admitted"
    );
    assert!(!called, "local admission must precede callback evaluation");
    let generous = tree_elementwise::<f64, _, _>(
        |v| if v[0] == v[1] { 1.0 } else { 0.0 },
        &inputs,
        &TreeAciOptions {
            max_working_bytes: 1024 * 1024,
            enable_global_guard: false,
            max_core_elements: Some(n * n),
            max_local_matrix_elements: Some(n * n),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(generous.termination, TreeAciTermination::Converged);
    let values = dense::<f64>(&generous.tree, &sites);
    assert_eq!(values.len(), n * n);
    for (p, &v) in values.iter().enumerate() {
        assert_eq!(v, if p % n == p / n { 1.0 } else { 0.0 });
    }
}

#[test]
fn candidate_metadata_must_fit_working_budget() {
    let edges = vec![(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)];
    let sites: Vec<Vec<_>> = (0..6)
        .map(|node| vec![DynIndex::new_dyn(if node == 0 { 32 } else { 1 })])
        .collect();
    let input = tree::<f64>(&edges, &sites, 1, 0, false);
    let budget = 1152usize;
    // At each hub edge, 32 candidates retain four incoming (edge,sample)
    // pairs apiece before the working check. Their payload alone exceeds the
    // configured budget, excluding ComponentSample/Vec headers and matrices.
    let incoming_pair_payload = 32 * 4 * std::mem::size_of::<(usize, usize)>();
    assert!(incoming_pair_payload > budget);
    let mut called = false;
    let result = tree_elementwise::<f64, _, _>(
        |_| {
            called = true;
            1.0
        },
        std::slice::from_ref(&input),
        &TreeAciOptions {
            max_working_bytes: budget,
            enable_global_guard: false,
            ..Default::default()
        },
    );
    assert!(
        matches!(
            result,
            Err(TreeAciError::ResourceLimit {
                resource: "working bytes",
                ..
            })
        ),
        "candidate metadata larger than the working budget was allocated and admitted"
    );
    assert!(
        !called,
        "candidate admission must precede callback evaluation"
    );
    let generous = tree_elementwise::<f64, _, _>(
        |_| 1.0,
        std::slice::from_ref(&input),
        &TreeAciOptions {
            max_working_bytes: 1024 * 1024,
            enable_global_guard: false,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(generous.termination, TreeAciTermination::Converged);
    let axes: Vec<_> = sites.iter().flatten().cloned().collect();
    assert_eq!(dense::<f64>(&generous.tree, &axes), vec![1.0; 32]);
}

#[test]
fn callback_error_and_invalid_options_are_propagated() {
    for edges in [vec![], vec![(0, 1)]] {
        let sites = vec![vec![DynIndex::new_dyn(2)]; edges.len() + 1];
        // Distinct site identities are required even when dimensions agree.
        let sites: Vec<Vec<_>> = sites.iter().map(|_| vec![DynIndex::new_dyn(2)]).collect();
        let input = tree::<f64>(&edges, &sites, 1, 0, false);
        let result = tree_elementwise_batched::<f64, _, _>(
            |_, _| {
                Err(TreeAciError::Callback {
                    message: "audit marker".into(),
                })
            },
            std::slice::from_ref(&input),
            &TreeAciOptions::default(),
        );
        assert!(
            matches!(result, Err(TreeAciError::Callback { message }) if message == "audit marker")
        );
        for tolerance in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let options = TreeAciOptions {
                tolerance,
                ..Default::default()
            };
            let mut called = false;
            let result = tree_elementwise::<f64, _, _>(
                |_| {
                    called = true;
                    1.0
                },
                std::slice::from_ref(&input),
                &options,
            );
            assert!(matches!(result, Err(TreeAciError::InvalidOption { .. })));
            assert!(!called);
        }
    }
}

#[cfg(feature = "diagnostics")]
#[test]
fn aggregate_guard_message_budget_must_be_respected() {
    use tensor4all_treeaci::branch_diagnostics as diag;
    let edges = vec![(0, 1), (0, 2), (0, 3), (0, 4)];
    let sites: Vec<Vec<_>> = (0..5).map(|_| vec![DynIndex::new_dyn(2)]).collect();
    let inputs = [
        tree::<f64>(&edges, &sites, 2, 0, false),
        tree::<f64>(&edges, &sites, 2, 1, true),
    ];
    let mut retained_with_headroom = false;
    for ninputs in [1, 2] {
        for budget in [0, 96, 512, 4096] {
            diag::reset();
            let result = tree_elementwise::<f64, _, _>(
                |_| 1.0,
                &inputs[..ninputs],
                &TreeAciOptions {
                    message_cache_max_bytes: budget,
                    nsearch_global_pivots: 16,
                    max_sweeps: 4,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(result.termination, TreeAciTermination::Converged);
            let rows = diag::snapshot();
            let mut peaks = [0usize; 3];
            for row in rows {
                for (i, name) in ["input:0:", "input:1:", "output:"].iter().enumerate() {
                    if row.node.starts_with(name) {
                        peaks[i] = peaks[i].max(row.query_cache.message_payload_bytes);
                    }
                }
            }
            eprintln!("GUARD_MESSAGE_BUDGET inputs={ninputs} configured={budget} per_evaluator_peaks={peaks:?}");
            let aggregate: usize = peaks.iter().sum();
            assert!(
                aggregate <= budget,
                "aggregate message payload exceeded the declared budget: {peaks:?}"
            );
            retained_with_headroom |= budget >= 512 && aggregate > 0;
        }
    }
    assert!(
        retained_with_headroom,
        "adequate budgets must permit message retention"
    );
}
