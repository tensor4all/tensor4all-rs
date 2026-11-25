//! GMRES on a four-site branched tree with an explicit compression callback.
use std::sync::Arc;

use tensor4all_core::krylov::{gmres_with_truncation, GmresOptions};
use tensor4all_core::{AnyScalar, DynIndex, IdxTensor, SvdTruncationPolicy, TensorContractionLike};
use tensor4all_tensorbackend::{CpuBackend, CpuExecutionContext, ExecutionContext};
use tensor4all_treetn::{TreeTN, TruncationOptions};

fn main() -> anyhow::Result<()> {
    let backend = CpuBackend::with_threads(1)?;
    let context = ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(backend)));
    let sites = (0..4).map(|_| DynIndex::new_dyn(2)).collect::<Vec<_>>();
    let bonds = (0..3).map(|_| DynIndex::new_dyn(2)).collect::<Vec<_>>();

    // Center 0 has three neighbors. The right-hand side is 2|0000> + 6|1111>.
    let mut center_data = vec![0.0_f64; 16];
    center_data[0] = 2.0;
    center_data[15] = 6.0;
    let mut center_indices = vec![sites[0].clone()];
    center_indices.extend(bonds.iter().cloned());
    let mut tensors = vec![IdxTensor::from_dense_in(
        &context,
        center_indices,
        center_data,
    )?];
    for (bond, site) in bonds.into_iter().zip(&sites[1..]) {
        tensors.push(IdxTensor::from_dense_in(
            &context,
            vec![bond, site.clone()],
            vec![1.0_f64, 0.0, 0.0, 1.0],
        )?);
    }
    let rhs = TreeTN::<_, usize>::from_tensors(tensors, vec![0, 1, 2, 3])?;
    let initial = rhs.scale(AnyScalar::new_real(0.0))?;

    // A multiplies the center site's |0> and |1> components by 2 and 3.
    let operator = IdxTensor::from_dense_in(
        &context,
        vec![sites[0].prime(), sites[0].clone()],
        vec![2.0_f64, 0.0, 0.0, 3.0],
    )?;
    let apply = |tree: &TreeTN<IdxTensor, usize>| -> anyhow::Result<_> {
        let node = tree
            .node_index(&0)
            .ok_or_else(|| anyhow::anyhow!("missing center"))?;
        let tensor = tree
            .tensor(node)
            .ok_or_else(|| anyhow::anyhow!("missing center tensor"))?;
        let transformed = IdxTensor::contract(&[tensor, &operator])?;
        let mut result = tree.clone();
        result.replace_tensor(node, transformed)?;
        // The operator's output leg is primed. Explicitly restore the vector's
        // physical index identities before returning it to GMRES.
        Ok(result.reindex_site_space_like(tree)?)
    };
    let truncation = TruncationOptions::default()
        .with_max_bond_dim(2)
        .with_svd_policy(SvdTruncationPolicy::new(1e-12));
    let options = GmresOptions {
        max_iter: 4,
        max_restarts: 2,
        check_true_residual: true,
        ..Default::default()
    };
    let result = gmres_with_truncation(apply, &rhs, &initial, &options, |tree| {
        Ok(tree.truncate_mut_in([0], truncation.clone(), &context)?)
    })?;

    // A bounded, explicit dense reference: x = |0000> + 2|1111>.
    let mut expected_data = vec![0.0_f64; 16];
    expected_data[0] = 1.0;
    expected_data[15] = 2.0;
    let expected = IdxTensor::from_dense_in(&context, sites, expected_data)?;
    let error = result.solution.to_dense()?.sub(&expected)?.maxabs()?;
    assert!(
        result.converged,
        "relative residual: {}",
        result.residual_norm
    );
    assert!(error < 1e-9, "maximum absolute solution error: {error}");
    Ok(())
}
