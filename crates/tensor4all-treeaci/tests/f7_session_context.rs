//! Caller-owned-context entry for tree-valued elementwise ACI (issue #859 F7).
//!
//! `tree_elementwise_batched_with_rng_in` runs the local updates' session-bound
//! numeric work (the per-edge LUCI factorizations and the candidate-matrix
//! products) on the supplied caller-owned CPU context instead of the
//! process-global context. This test pins two properties:
//! - parity: the context route produces bit-identical results to the plain
//!   route (same algorithm, same RNG stream; only the numeric execution
//!   context differs), and
//! - measured overhead: the caller-owned context is within noise of the plain
//!   route on one thread; see `benches/f7_context_overhead.rs` for the
//!   release-mode comparison.
//!
//! The frame contractions the crate performs itself still use their existing
//! route; the inventory (docs/design/859-f7-consumer-inventory.md) records
//! that exclusion.

use crate::g0;

use num_complex::Complex64;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensor4all_tensorbackend::CpuBackend;
use tensor4all_tensorbackend::CpuExecutionContext;
use tensor4all_treeaci::{
    tree_elementwise_batched_with_rng, tree_elementwise_batched_with_rng_in, TreeAciOptions,
    TreeElementwiseBatch,
};

fn constant_operator(
    _batch: TreeElementwiseBatch<'_, Complex64>,
    out: &mut [Complex64],
) -> Result<(), tensor4all_treeaci::TreeAciError> {
    out.fill(Complex64::new(1.0, 0.0));
    Ok(())
}

#[test]
fn context_route_matches_plain_route_bit_for_bit() {
    const R: usize = 4;
    let (_sites, inputs) = g0::fixture(R, "cttn").unwrap();
    let options = TreeAciOptions {
        rng_seed: 7,
        ..Default::default()
    };

    let mut plain_rng = ChaCha8Rng::seed_from_u64(0);
    let mut context_rng = ChaCha8Rng::seed_from_u64(0);
    let plain =
        tree_elementwise_batched_with_rng(constant_operator, &inputs, &options, &mut plain_rng)
            .unwrap();
    let context = CpuExecutionContext::from_backend(CpuBackend::with_threads(1).unwrap());
    let contextual = tree_elementwise_batched_with_rng_in(
        constant_operator,
        &inputs,
        &options,
        &mut context_rng,
        &context,
    )
    .unwrap();

    let plain_dense = plain
        .tree
        .to_dense()
        .unwrap()
        .to_vec::<Complex64>()
        .unwrap();
    let contextual_dense = contextual
        .tree
        .to_dense()
        .unwrap()
        .to_vec::<Complex64>()
        .unwrap();
    assert_eq!(plain_dense.len(), contextual_dense.len());
    for (a, b) in plain_dense.iter().zip(&contextual_dense) {
        assert_eq!(a, b, "context route diverged from the plain route");
    }
}
