//! Bounded accuracy/rank diagnosis; not a timing benchmark or a tuned acceptance run.
#[path = "../src/diagnostics.rs"]
mod diagnostics;
#[path = "../../../crates/tensor4all-treersi/examples/support/mod.rs"]
mod support;
use serde_json::json;
use tensor4all_core::{
    DynIndex, FactorizeOptions, IdxTensor, SvdTruncationPolicy, TensorFactorizationLike,
};
use tensor4all_treeaci::TreeAciOptions;
use tensor4all_treersi::TreeRsiOptions;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sites = (0..15).map(|_| DynIndex::new_dyn(2)).collect::<Vec<_>>();
    let edges = (1..15).map(|i| ((i - 1) / 2, i)).collect::<Vec<_>>();
    let inputs = [11, 29].map(|seed| {
        support::random_tree(&sites, &edges, 2, seed, |x, _| {
            (0.8 + 0.2 * x) / 2.0_f64.sqrt()
        })
    });
    // Materialize each input once. Every subsequent error covers the full grid.
    let a = inputs[0]
        .to_dense()?
        .permute_indices(&sites)?
        .to_vec::<f64>()?;
    let b = inputs[1]
        .to_dense()?
        .permute_indices(&sites)?
        .to_vec::<f64>()?;
    let expected = a.iter().zip(b).map(|(x, y)| x * y).collect::<Vec<_>>();
    let norm_squared = expected.iter().map(|y| y * y).sum::<f64>();
    // Cutting (0, 1) separates these seven physical sites from the rest.
    // Eckart--Young gives a lower bound for every tree with rank <= cap on
    // this edge, independently of either interpolation algorithm.
    let reference = IdxTensor::from_dense(sites.clone(), expected.clone())?;
    let left = [1, 3, 4, 7, 8, 9, 10].map(|i| sites[i].clone());
    let factorization = reference.factorize(
        &left,
        &FactorizeOptions::svd().with_svd_policy(SvdTruncationPolicy::new(0.0)),
    )?;
    let singular = factorization
        .singular_values
        .ok_or("SVD omitted singular values")?;
    let svd_norm_squared = singular.iter().map(|s| s * s).sum::<f64>();
    if (svd_norm_squared / norm_squared - 1.0).abs() > 1e-12 {
        return Err("cut SVD does not preserve the reference Frobenius norm".into());
    }
    let rank_bound = |rank: usize| {
        (singular.iter().skip(rank).map(|s| s * s).sum::<f64>() / norm_squared).sqrt()
    };
    let rank_two_lower_bound = rank_bound(2);
    println!(
        "{}",
        json!({"type":"protocol", "case":"binary-n15-chi2", "root":14,
        "cap_min_sweeps":[[2,2],[4,2],[8,2],[16,2],[4,4]], "seeds":[1,2,3], "local_tolerance":1e-12,
        "metric":"full-grid relative L2", "points":32768,
        "cut":[0,1], "relative_singular_values":singular.iter().take(4).map(|s| s / norm_squared.sqrt()).collect::<Vec<_>>(),
        "rank_two_relative_l2_lower_bound":rank_two_lower_bound,
        "rank_four_relative_l2_lower_bound":rank_bound(4),
        "rsi_sketch_dim":7, "purpose":"rank sensitivity; no timing or retuning of the fixed benchmark"})
    );
    for (cap, min_sweeps) in [(2, 2), (4, 2), (8, 2), (16, 2), (4, 4)] {
        for seed in [1, 2, 3] {
            for algorithm in ["treeaci", "rsi"] {
                if algorithm == "rsi" && min_sweeps != 2 {
                    continue;
                }
                let (tree, mut diagnostics) = if algorithm == "treeaci" {
                    let out = tensor4all_treeaci::hadamard_many::<f64, _>(
                        &inputs,
                        &TreeAciOptions {
                            max_bond_dim: Some(cap),
                            min_sweeps,
                            rng_seed: seed,
                            root: Some(14),
                            tolerance: 1e-12,
                            scale_tolerance: true,
                            ..Default::default()
                        },
                    )?;
                    (
                        out.tree,
                        json!({"termination":format!("{:?}",out.termination),
                        "edge_ranks":out.diagnostics.edge_ranks,"sweep_ranks":out.max_ranks,
                        "local_errors":out.max_errors,"global_pivots":out.global_pivots_found}),
                    )
                } else {
                    // Keep k fixed at the original cap-4 setting to isolate the rank cap.
                    let out = tensor4all_treersi::hadamard_many::<f64, _>(
                        &inputs,
                        &TreeRsiOptions {
                            max_bond_dim: Some(cap),
                            sketch_dim: Some(7),
                            seed,
                            root: Some(14),
                            rel_tol: 1e-12,
                            ..Default::default()
                        },
                    )?;
                    (
                        out.tree,
                        json!({"sketch_dim":out.diagnostics.sketch_dim,
                        "edge_ranks":out.diagnostics.edges.iter().map(|e| (e.child,e.parent,e.rank)).collect::<Vec<_>>()}),
                    )
                };
                diagnostics["edge_ranks"] = json!(diagnostics::edge_ranks(&tree)?);
                let actual = tree.to_dense()?.permute_indices(&sites)?.to_vec::<f64>()?;
                let error = (actual
                    .iter()
                    .zip(&expected)
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f64>()
                    / norm_squared)
                    .sqrt();
                if cap == 2 && error + 1e-12 < rank_two_lower_bound {
                    return Err("observed error contradicts the cut rank lower bound".into());
                }
                if !error.is_finite() {
                    return Err("nonfinite full-grid error".into());
                }
                println!(
                    "{}",
                    json!({"type":"observation","cap":cap,"min_sweeps":min_sweeps,"seed":seed,"algorithm":algorithm,
                    "error":error,"diagnostics":diagnostics})
                );
            }
        }
    }
    Ok(())
}
