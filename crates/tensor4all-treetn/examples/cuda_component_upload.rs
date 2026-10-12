//! Per-component upload timings for the TreeTN CUDA route (#553 / #859 F5).
//!
//! Uploads a chain one node at a time and prints each component's time, their sum, and the whole-network upload as
//! the comparison, so the transfer share is attributable per component instead of appearing as one figure. It also
//! runs the per-node loop twice, because the first call of a node carries initialisation that later calls do not.
//!
//! It does **not** measure a wait or an overlap: this route has no stream or pending-transfer surface, so every
//! transfer is synchronous. Read the compute figures from `cuda_tree_contraction` (synchronised compute latency)
//! and this example's upload figures separately.
//!
//! ```text
//! cargo run --release -p tensor4all-treetn --example cuda_component_upload \
//!   --features tenferro-cuda,tenferro-cpu-faer -- --chain-length 6 --bond-dim 4
//! ```

use std::error::Error;
use std::time::Instant;

use tensor4all_core::{CudaExecutionContext, DynIndex, IdxTensor, IndexLike};
use tensor4all_treetn::TreeTN;

fn dense(indices: Vec<DynIndex>, seed: usize) -> Result<IdxTensor, Box<dyn Error>> {
    let elements = indices
        .iter()
        .map(IndexLike::dim)
        .try_fold(1usize, |size, dim| size.checked_mul(dim))
        .ok_or("tensor element count overflow")?;
    let values = (0..elements)
        .map(|offset| ((seed + offset) % 19) as f64 / 19.0 + 0.25)
        .collect();
    Ok(IdxTensor::from_dense(indices, values)?)
}

fn chain(length: usize, bond_dim: usize) -> Result<TreeTN<IdxTensor, usize>, Box<dyn Error>> {
    let sites: Vec<_> = (0..length).map(|_| DynIndex::new_dyn(2)).collect();
    let bonds: Vec<_> = (0..length.saturating_sub(1))
        .map(|_| DynIndex::new_dyn(bond_dim))
        .collect();
    let mut tensors = Vec::with_capacity(length);
    for node in 0..length {
        let mut indices = Vec::new();
        if let Some(left) = node.checked_sub(1).and_then(|index| bonds.get(index)) {
            indices.push(left.clone());
        }
        indices.push(sites.get(node).cloned().ok_or("site index missing")?);
        if let Some(right) = bonds.get(node) {
            indices.push(right.clone());
        }
        tensors.push(dense(indices, node * 31 + 1)?);
    }
    Ok(TreeTN::from_tensors(tensors, (0..length).collect())?)
}

fn main() -> Result<(), Box<dyn Error>> {
    let length: usize = std::env::var("CUDA_COMPONENT_CHAIN_LENGTH")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(6);
    let bond_dim: usize = std::env::var("CUDA_COMPONENT_BOND_DIM")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(4);

    let host = chain(length, bond_dim)?;
    let context = CudaExecutionContext::new()?;

    // The order matters and is therefore explicit: the first figure of each shape carries whatever initialisation
    // the route needs on first use (context, engine, caches), the second one does not. Reporting only the first
    // would attribute initialisation to transfer; reporting only the second would hide it.
    let mut whole_ms = Vec::new();
    let mut passes: Vec<(usize, Vec<f64>, f64)> = Vec::new();

    for round in 0..2 {
        // Per-component upload: one call per node.
        let mut partial = host.clone();
        let mut component_ms = Vec::with_capacity(length);
        let started = Instant::now();
        for node in 0..length {
            let step = Instant::now();
            partial = partial.upload_nodes_cuda(&[node], &context)?;
            component_ms.push(step.elapsed().as_secs_f64() * 1e3);
        }
        let pass_ms = started.elapsed().as_secs_f64() * 1e3;
        for node in 0..length {
            let index = partial.node_index(&node).ok_or("node missing")?;
            partial
                .tensor(index)
                .ok_or("tensor missing")?
                .validate_cuda_residency(&context)?;
        }
        passes.push((round, component_ms, pass_ms));

        // Whole-network upload, the comparison figure.
        let started = Instant::now();
        let whole = host.upload_cuda(&context)?;
        whole_ms.push(started.elapsed().as_secs_f64() * 1e3);
        for node in 0..length {
            let index = whole.node_index(&node).ok_or("node missing")?;
            whole
                .tensor(index)
                .ok_or("tensor missing")?
                .validate_cuda_residency(&context)?;
        }
    }

    println!(
        "cuda_component_upload: device={:?} chain_length={length} bond_dim={bond_dim} \
         whole_upload_ms=[{:.3},{:.3}] (first, second: the first carries first-use initialisation)",
        context.device_name(),
        whole_ms[0],
        whole_ms[1]
    );
    for (pass, component_ms, pass_ms) in &passes {
        let sum: f64 = component_ms.iter().sum();
        println!(
            "cuda_component_upload: pass={pass} per_node_ms=[{}] sum={sum:.3} wall={pass_ms:.3}",
            component_ms
                .iter()
                .map(|value| format!("{value:.3}"))
                .collect::<Vec<_>>()
                .join(",")
        );
    }
    println!(
        "cuda_component_upload: note=each per-node call also clones the network and updates one node, and includes \
         native duplication, session entry and eager wrapping; wait/overlap are not measurable on this route"
    );

    Ok(())
}
