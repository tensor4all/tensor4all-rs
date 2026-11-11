//! Counterexamples separating the stopping criteria from exact reconstruction.

use std::collections::HashSet;

use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_treeaci::{tree_elementwise_batched, TreeAciOptions, TreeAciTermination};
use tensor4all_treetn::TreeTN;

/// Rank-two input encoding a binary assignment as its exact integer value.
/// Flat physical coordinates are column-major: site zero is the low bit.
fn encoded_binary_chain(n: usize) -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
    let sites: Vec<_> = (0..n).map(|_| DynIndex::new_dyn(2)).collect();
    let bonds: Vec<_> = (1..n).map(|_| DynIndex::new_dyn(2)).collect();
    let cores = (0..n)
        .map(|site| {
            let left_rank = if site == 0 { 1 } else { 2 };
            let right_rank = if site + 1 == n { 1 } else { 2 };
            let mut values = vec![0.0; left_rank * 2 * right_rank];
            for bit in 0..2 {
                let weight = ((1usize << site) * bit) as f64;
                if site == 0 {
                    values[bit] = 1.0;
                    values[bit + 2] = weight;
                } else if site + 1 == n {
                    values[2 * bit] = weight;
                    values[1 + 2 * bit] = 1.0;
                } else {
                    values[2 * bit] = 1.0;
                    values[2 * bit + 4] = weight;
                    values[1 + 2 * bit + 4] = 1.0;
                }
            }
            let mut indices = Vec::new();
            if site > 0 {
                indices.push(bonds[site - 1].clone());
            }
            indices.push(sites[site].clone());
            if site + 1 < n {
                indices.push(bonds[site].clone());
            }
            IdxTensor::from_dense(indices, values).unwrap()
        })
        .collect();
    (
        TreeTN::from_tensors(cores, (0..n).collect()).unwrap(),
        sites,
    )
}

#[test]
fn positive_tolerance_can_converge_with_nonzero_error_on_a_fully_sampled_matrix() {
    let (input, sites) = encoded_binary_chain(2);
    let small = 1e-5;
    let expected = IdxTensor::from_dense(sites, vec![1.0, 0.0, 0.0, small]).unwrap();
    for (tolerance, expected_error, cap, termination) in [
        (1e-4, small, None, TreeAciTermination::Converged),
        (1e-8, 0.0, None, TreeAciTermination::Converged),
        (1e-8, small, Some(1), TreeAciTermination::RankLimited),
    ] {
        let options = TreeAciOptions {
            tolerance,
            max_bond_dim: cap,
            rng_seed: 2026,
            ..Default::default()
        };
        let result = tree_elementwise_batched(
            |batch: tensor4all_treeaci::TreeElementwiseBatch<'_, f64>, out: &mut [f64]| {
                for (point, value) in out.iter_mut().enumerate() {
                    *value = match batch.get(0, point)? as usize {
                        0 => 1.0,
                        3 => small,
                        _ => 0.0,
                    };
                }
                Ok(())
            },
            std::slice::from_ref(&input),
            &options,
        )
        .unwrap();
        let error = result
            .tree
            .to_dense()
            .unwrap()
            .sub(&expected)
            .unwrap()
            .maxabs()
            .unwrap();
        assert_eq!(result.termination, termination);
        assert!((error - expected_error).abs() < 1e-15, "error={error:e}");
    }
}

#[test]
fn default_guard_can_miss_an_unsampled_spike_and_report_converged() {
    // A finite adaptive query transcript cannot distinguish a constant from a
    // function with a spike at an unqueried point. Choose the witness *after*
    // recording the constant transcript, then replay the same options/RNG.
    let n = 12;
    let (input, sites) = encoded_binary_chain(n);
    assert_eq!(
        input
            .to_dense()
            .unwrap()
            .permute_indices(&sites)
            .unwrap()
            .to_vec::<f64>()
            .unwrap(),
        (0..1 << n).map(|value| value as f64).collect::<Vec<_>>()
    );
    let options = TreeAciOptions {
        rng_seed: 2026,
        ..Default::default()
    };
    let mut transcripts = [Vec::new(), Vec::new()];
    let mut hidden = None;
    let mut result = None;
    for (run, transcript) in transcripts.iter_mut().enumerate() {
        let spike = if run == 0 { None } else { hidden };
        let current = tree_elementwise_batched(
            |batch: tensor4all_treeaci::TreeElementwiseBatch<'_, f64>, out: &mut [f64]| {
                for (point, value) in out.iter_mut().enumerate() {
                    let encoded = batch.get(0, point)?;
                    assert_eq!(encoded.fract(), 0.0);
                    let encoded = encoded as usize;
                    transcript.push(encoded);
                    *value = if Some(encoded) == spike { 1025.0 } else { 1.0 };
                }
                Ok(())
            },
            std::slice::from_ref(&input),
            &options,
        )
        .unwrap();
        assert_eq!(current.termination, TreeAciTermination::Converged);
        assert!(current.global_pivots_found.iter().all(|&found| found == 0));
        if run == 0 {
            let visited: HashSet<_> = transcript.iter().copied().collect();
            hidden = (0..1 << n).find(|point| !visited.contains(point));
            assert!(
                hidden.is_some(),
                "the control unexpectedly sampled the entire grid"
            );
        }
        result = Some(current);
    }
    assert_eq!(transcripts[0], transcripts[1]);
    let mut exact = vec![1.0; 1 << n];
    exact[hidden.unwrap()] = 1025.0;
    let exact = IdxTensor::from_dense(sites, exact).unwrap();
    let error = result
        .unwrap()
        .tree
        .to_dense()
        .unwrap()
        .sub(&exact)
        .unwrap()
        .maxabs()
        .unwrap();
    assert_eq!(error, 1024.0);
    eprintln!(
        "EXACTNESS {}",
        serde_json::json!({"grid_points":1 << n,"queried_distinct":transcripts[0].iter().collect::<HashSet<_>>().len(),"hidden_point":hidden,"maximum_error":error})
    );
}
