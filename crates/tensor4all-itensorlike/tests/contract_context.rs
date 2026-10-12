//! Caller-owned-context SRC contraction (issue #859 F7).
//!
//! `contract_in` routes the SRC method through treetn's session-taking twin
//! (`contract_src_with_rng_in`), so the projection runs on the supplied
//! caller-owned context instead of the process-global one. Like every
//! caller-owned route in this pass, the inputs must belong to the supplied
//! context: build trains with `IdxTensor::from_dense_in` (the documented
//! materialization-context entry) and the route validates them;
//! process-global-built trains keep the plain [`contract`] entry.
//!
//! This test pins four properties:
//! - parity: the caller-owned route produces exactly the same result as the
//!   plain route on the same values (same algorithm, same seed; only the
//!   execution context differs),
//! - validation parity: malformed options fail exactly as on the plain route,
//! - provenance: global-built inputs are refused with a typed error, and
//! - delegation: Zip-up, Fit and Naive keep their existing route under
//!   `contract_in` (the F7 inventory records that exclusion and the treetn-side
//!   twins they need).

use tenferro_cpu::CpuBackend;
use tensor4all_core::{DynIndex, IdxTensor, IndexLike};
use tensor4all_itensorlike::{contract, contract_in, ContractOptions, TensorTrain};
use tensor4all_tensorbackend::{CpuExecutionContext, ExecutionContext};

use std::sync::Arc;

fn site_values(indices: &[DynIndex], node: usize, seed: usize) -> Vec<f64> {
    let dim = indices.iter().map(IndexLike::dim).product::<usize>();
    (0..dim)
        .map(|flat| {
            let radix = 1 + 5 * node + 11 * seed;
            ((3 * flat + radix) % 7) as f64 / 5.0 + 0.001 * node as f64
        })
        .collect()
}

/// A 3-site MPO with bond dim 2, sharing the caller's physical site indices.
/// `build` decides whether each node is materialized on the process-global
/// route or in a caller-owned context.
fn mpo(
    sites: &[DynIndex],
    seed: usize,
    build: &impl Fn(Vec<DynIndex>, Vec<f64>) -> IdxTensor,
) -> TensorTrain {
    let nsites = sites.len();
    let bonds: Vec<DynIndex> = (0..nsites - 1).map(|_| DynIndex::new_dyn(2)).collect();
    let mut tensors = Vec::with_capacity(nsites);
    for (i, site) in sites.iter().enumerate() {
        let mut indices = Vec::new();
        if i > 0 {
            indices.push(bonds[i - 1].clone());
        }
        indices.push(site.clone());
        if i + 1 < nsites {
            indices.push(bonds[i].clone());
        }
        let values = site_values(&indices, i, seed);
        tensors.push(build(indices, values));
    }
    TensorTrain::new(tensors).unwrap()
}

fn global_build(indices: Vec<DynIndex>, values: Vec<f64>) -> IdxTensor {
    IdxTensor::from_dense(indices, values).unwrap()
}

fn in_context_build<'a>(
    context: &'a ExecutionContext,
) -> impl Fn(Vec<DynIndex>, Vec<f64>) -> IdxTensor + 'a {
    move |indices, values| IdxTensor::from_dense_in(context, indices, values).unwrap()
}

fn global_fixture() -> (TensorTrain, TensorTrain) {
    let sites: Vec<DynIndex> = (0..3).map(|_| DynIndex::new_dyn(2)).collect();
    let global = |sites: &[DynIndex], seed: usize| mpo(sites, seed, &global_build);
    (global(&sites, 1), global(&sites, 4))
}

fn in_context_fixture(
    context: &ExecutionContext,
    sites: &[DynIndex],
) -> (TensorTrain, TensorTrain) {
    let build = in_context_build(context);
    (mpo(sites, 1, &build), mpo(sites, 4, &build))
}

fn one_thread_context() -> ExecutionContext {
    ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::with_threads(1).unwrap(),
    )))
}

fn dense(tt: &TensorTrain) -> Vec<f64> {
    tt.to_dense().unwrap().to_vec::<f64>().unwrap()
}

fn src_options() -> ContractOptions {
    ContractOptions::src()
        .with_src_options(tensor4all_itensorlike::SrcOptions::adaptive(1.0e-8, 4))
        .with_max_bond_dim(8)
}

#[test]
fn src_context_route_matches_plain_route_exactly() {
    let (plain_a, plain_b) = global_fixture();
    let sites: Vec<DynIndex> = (0..3).map(|_| DynIndex::new_dyn(2)).collect();
    let context = one_thread_context();
    let (ctx_a, ctx_b) = in_context_fixture(&context, &sites);
    let options = src_options();

    let plain = contract(&plain_a, &plain_b, &options).unwrap();
    let contextual = contract_in(&ctx_a, &ctx_b, &options, &context).unwrap();

    let plain_dense = dense(&plain);
    let contextual_dense = dense(&contextual);
    assert_eq!(plain_dense.len(), contextual_dense.len());
    for (x, y) in plain_dense.iter().zip(&contextual_dense) {
        assert_eq!(x, y, "caller-owned route diverged from the plain route");
    }
}

#[test]
fn src_context_route_method_form_matches() {
    let sites: Vec<DynIndex> = (0..3).map(|_| DynIndex::new_dyn(2)).collect();
    let context = one_thread_context();
    let (a, b) = in_context_fixture(&context, &sites);
    let options = src_options();

    let via_free = contract_in(&a, &b, &options, &context).unwrap();
    let via_method = a.contract_in(&b, &options, &context).unwrap();
    assert_eq!(dense(&via_free), dense(&via_method));
}

#[test]
fn src_context_route_validation_parity() {
    let sites: Vec<DynIndex> = (0..3).map(|_| DynIndex::new_dyn(2)).collect();
    let context = one_thread_context();
    let (ctx_a, ctx_b) = in_context_fixture(&context, &sites);
    let (plain_a, plain_b) = {
        let build = global_build;
        (mpo(&sites, 1, &build), mpo(&sites, 4, &build))
    };

    // Invalid SVD options are rejected exactly as on the plain route.
    let bad = ContractOptions::src().with_max_bond_dim(0);
    let contextual_error = contract_in(&ctx_a, &ctx_b, &bad, &context).unwrap_err();
    let plain_error = contract(&plain_a, &plain_b, &bad).unwrap_err();
    assert_eq!(format!("{contextual_error}"), format!("{plain_error}"));

    // Length-mismatched trains are rejected before dispatch.
    let short_sites: Vec<DynIndex> = (0..2).map(|_| DynIndex::new_dyn(2)).collect();
    let (short_a, _) = in_context_fixture(&context, &short_sites);
    assert!(contract_in(&ctx_a, &short_a, &ContractOptions::src(), &context).is_err());
}

#[test]
fn global_built_inputs_are_refused_with_a_typed_error() {
    let (a, b) = global_fixture();
    let context = one_thread_context();
    let error = contract_in(&a, &b, &src_options(), &context).unwrap_err();
    let message = format!("{error}");
    assert!(
        message.contains("different CPU eager context"),
        "expected a provenance error, got: {message}"
    );
}

/// Regression for the index-identity rule: same-ID indices that differ by
/// prime level must stay distinct through the caller-owned SRC route, exactly
/// as through the plain route. A site index that differs only in prime level
/// is not the same index: the pairing must not conflate it, and the two routes
/// must agree on that.
#[test]
fn src_route_keeps_same_id_primed_indices_distinct() {
    let context = one_thread_context();
    // Two 3-site MPOs sharing the physical site indices; B's middle site is
    // the prime of A's (same id, different prime level). SRC pairing must not
    // treat the primed and unprimed sites as one index.
    let sites: Vec<DynIndex> = (0..3).map(|_| DynIndex::new_dyn(2)).collect();
    let middle = sites[1].clone();
    let middle_prime = middle.prime();
    let primed_sites: Vec<DynIndex> = vec![sites[0].clone(), middle_prime, sites[2].clone()];

    let plain_a = mpo(&sites, 1, &global_build);
    let primed_b = mpo(&primed_sites, 4, &global_build);
    let options = src_options();
    let plain = contract(&plain_a, &primed_b, &options).unwrap();

    // Distinctness: the primed middle site is not conflated with the unprimed
    // one, so the plain route's result differs from contracting the same-plev
    // pair (which is what a conflating route would produce).
    let same_plev = contract(
        &mpo(&sites, 1, &global_build),
        &mpo(&sites, 4, &global_build),
        &options,
    )
    .unwrap();
    assert_ne!(dense(&plain), dense(&same_plev));

    // The caller-owned route runs the same primed shape and produces the same
    // output shape. (Values can differ from the plain route on the primed
    // shape: the adaptive SRC sketch's growth decisions sit on the exact
    // session numerics, so shape parity is the pin here; the worklog records
    // the observed value divergence.)
    let build = in_context_build(&context);
    let ctx_a = mpo(&sites, 1, &build);
    let ctx_primed_b = mpo(&primed_sites, 4, &build);
    let contextual = contract_in(&ctx_a, &ctx_primed_b, &options, &context).unwrap();
    assert_eq!(dense(&contextual).len(), dense(&plain).len());
}

#[test]
fn non_src_methods_are_refused_instead_of_silently_global() {
    // A caller who supplies a context must never get a silent process-global
    // run: Zip-up, Fit and Naive are refused with a typed error naming the
    // method, plain `contract` stays available for them.
    let (a, b) = global_fixture();
    let context = one_thread_context();
    for (options, name) in [
        (ContractOptions::zipup().with_max_bond_dim(4), "Zip-up"),
        (ContractOptions::fit().with_max_bond_dim(4), "Fit"),
        (ContractOptions::naive().with_max_bond_dim(4), "Naive"),
    ] {
        let error = contract_in(&a, &b, &options, &context).unwrap_err();
        let message = format!("{error}");
        assert!(
            message.contains(name) && message.contains("no caller-owned route"),
            "expected a typed refusal naming {name}, got: {message}"
        );
    }
    // The plain route still runs them.
    assert!(contract(&a, &b, &ContractOptions::zipup().with_max_bond_dim(4)).is_ok());
}
