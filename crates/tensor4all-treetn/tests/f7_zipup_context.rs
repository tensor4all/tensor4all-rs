//! Caller-owned-context zip-up contraction (issue #859 F7).
//!
//! `TreeTN::contract_zipup_with_in` runs the per-site contractions
//! (`contract_pair_on_session`) and factorizations (`factorize_in`) of the
//! zip-up schedule in the supplied caller-owned `ExecutionContext` instead of
//! the process-global one. This test pins:
//! - parity: the caller-owned route agrees with the plain route to tight
//!   tolerance on the shared-site (general-path) and distinct-site (chain
//!   path) layouts, with and without an SVD truncation policy (the session
//!   route factorizes directly instead of the `factorize_auto` Gram shortcut,
//!   so tolerance, not bitwise equality, is the contract);
//! - refusal: `LU`/`CI` forms have no context-scoped factorization path and
//!   are refused with a typed error rather than silently run globally.
//!
//! Input provenance follows `TreeTN::validate_context`: host-only CPU tensors
//! (`from_dense`) validate against any CPU context, so the route accepts them;
//! resident (CUDA) or foreign-eager inputs are refused. On a CPU-only build
//! that refusal is not reachable from a test, so it is covered by the
//! `validate_context` tests instead.

use std::sync::Arc;

use tenferro_cpu::CpuBackend;
use tensor4all_core::{DynIndex, ExecutionContext, IdxTensor, IndexLike, SvdTruncationPolicy};
use tensor4all_tensorbackend::CpuExecutionContext;
use tensor4all_treetn::{CanonicalForm, TreeTN};

fn cpu_context() -> ExecutionContext {
    ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::with_threads(1).unwrap(),
    )))
}

/// A `n_sites` site chain with bond dim `chi`. When `distinct_sites` is set,
/// each operand carries its own site indices per node (an MPO-style layout
/// with per-node output spaces, which takes the zip-up chain path); otherwise
/// both operands share one physical index per site (the general path).
fn chain(
    n_sites: usize,
    chi: usize,
    shared_sites: Option<&[DynIndex]>,
    build: &impl Fn(Vec<DynIndex>, Vec<f64>) -> IdxTensor,
) -> TreeTN<IdxTensor, usize> {
    let bonds: Vec<DynIndex> = (0..n_sites - 1).map(|_| DynIndex::new_dyn(chi)).collect();
    let tensors = (0..n_sites)
        .map(|node| {
            let site = shared_sites
                .map(|sites| sites[node].clone())
                .unwrap_or_else(|| DynIndex::new_dyn(2));
            let mut indices = Vec::new();
            if node > 0 {
                indices.push(bonds[node - 1].clone());
            }
            indices.push(site);
            if node + 1 < n_sites {
                indices.push(bonds[node].clone());
            }
            let dim = indices.iter().map(IndexLike::dim).product::<usize>();
            let values = (0..dim)
                .map(|flat| {
                    let radix = 1 + 3 * node;
                    ((2 * flat + radix) % 5) as f64 / 4.0
                })
                .collect();
            build(indices, values)
        })
        .collect();
    TreeTN::from_tensors(tensors, (0..n_sites).collect()).unwrap()
}

fn global_build(indices: Vec<DynIndex>, values: Vec<f64>) -> IdxTensor {
    IdxTensor::from_dense(indices, values).unwrap()
}

fn in_context_build<'a>(
    context: &'a ExecutionContext,
) -> impl Fn(Vec<DynIndex>, Vec<f64>) -> IdxTensor + 'a {
    move |indices, values| IdxTensor::from_dense_in(context, indices, values).unwrap()
}

fn dense_values(tree: &TreeTN<IdxTensor, usize>) -> IdxTensor {
    tree.contract_to_tensor().unwrap()
}

/// Materialize each result once and compare the dense payloads element-wise.
///
/// A whole-result `sub`/`maxabs` comparison is not available here: zip-up
/// renames internal indices with fresh IDs per run (`sim_internal_inds`), so
/// the two results carry equal shapes but different index IDs and `sub`
/// refuses the index-set mismatch. Both routes run the same schedule over the
/// same index order, so the dense payloads are positionally aligned.
fn assert_close(left: &IdxTensor, right: &IdxTensor) {
    assert_eq!(left.dims(), right.dims());
    let (left, right) = (
        left.to_vec::<f64>().unwrap(),
        right.to_vec::<f64>().unwrap(),
    );
    assert_eq!(left.len(), right.len());
    for (x, y) in left.iter().zip(&right) {
        let scale = x.abs().max(y.abs()).max(1.0);
        assert!(
            (x - y).abs() <= 1e-11 * scale,
            "zip-up context route diverged: |{x} - {y}|"
        );
    }
}

#[test]
fn zipup_general_path_matches_plain_route_by_dense_size_and_values() {
    for with_policy in [false, true] {
        let context = cpu_context();
        let sites: Vec<DynIndex> = (0..2).map(|_| DynIndex::new_dyn(2)).collect();
        let (global_a, global_b) = (
            chain(2, 2, Some(&sites), &global_build),
            chain(2, 2, Some(&sites), &global_build),
        );
        let build = in_context_build(&context);
        let (ctx_a, ctx_b) = (
            chain(2, 2, Some(&sites), &build),
            chain(2, 2, Some(&sites), &build),
        );

        let policy = with_policy.then(|| SvdTruncationPolicy::new(1e-10));
        let plain = global_a
            .contract_zipup_with(&global_b, &0, CanonicalForm::Unitary, policy, Some(4))
            .unwrap();
        let contextual = ctx_a
            .contract_zipup_with_in(
                &ctx_b,
                &0,
                CanonicalForm::Unitary,
                policy,
                Some(4),
                &context,
            )
            .unwrap();
        assert_close(&dense_values(&plain), &dense_values(&contextual));
    }
}

#[test]
fn zipup_chain_path_matches_plain_route() {
    // Distinct per-node site spaces take the chain path.
    let context = cpu_context();
    let build = in_context_build(&context);
    let (ctx_a, ctx_b) = (chain(3, 2, None, &build), chain(3, 2, None, &build));
    let (global_a, global_b) = (
        chain(3, 2, None, &global_build),
        chain(3, 2, None, &global_build),
    );

    let plain = global_a
        .contract_zipup_with(&global_b, &1, CanonicalForm::Unitary, None, Some(4))
        .unwrap();
    let contextual = ctx_a
        .contract_zipup_with_in(&ctx_b, &1, CanonicalForm::Unitary, None, Some(4), &context)
        .unwrap();
    assert_close(&dense_values(&plain), &dense_values(&contextual));
}

#[test]
fn branch_path_matches_plain_route() {
    // A star (root 0 with leaves 1 and 2) sharing one physical site per node:
    // the general (non-chain) path.
    let context = cpu_context();
    let build = in_context_build(&context);
    let star = |seed: usize, build: &dyn Fn(Vec<DynIndex>, Vec<f64>) -> IdxTensor| {
        let b1 = DynIndex::new_dyn(2);
        let b2 = DynIndex::new_dyn(2);
        let s0 = DynIndex::new_dyn(2);
        let s1 = DynIndex::new_dyn(2);
        let s2 = DynIndex::new_dyn(2);
        let node = |indices: Vec<DynIndex>, radix: usize| {
            let dim = indices.iter().map(IndexLike::dim).product::<usize>();
            let values = (0..dim)
                .map(|flat| 0.1 * ((flat + radix + seed) % 7) as f64 + 0.5)
                .collect();
            build(indices, values)
        };
        vec![
            node(vec![b1.clone(), b2.clone(), s0], 1),
            node(vec![b1, s1], 2),
            node(vec![b2, s2], 3),
        ]
    };
    let ga = TreeTN::<IdxTensor, usize>::from_tensors(
        star(
            1,
            &global_build as &dyn Fn(Vec<DynIndex>, Vec<f64>) -> IdxTensor,
        ),
        vec![0, 1, 2],
    )
    .unwrap();
    let gb = TreeTN::<IdxTensor, usize>::from_tensors(
        star(
            4,
            &global_build as &dyn Fn(Vec<DynIndex>, Vec<f64>) -> IdxTensor,
        ),
        vec![0, 1, 2],
    )
    .unwrap();
    let ca = TreeTN::<IdxTensor, usize>::from_tensors(
        star(1, &build as &dyn Fn(Vec<DynIndex>, Vec<f64>) -> IdxTensor),
        vec![0, 1, 2],
    )
    .unwrap();
    let cb = TreeTN::<IdxTensor, usize>::from_tensors(
        star(4, &build as &dyn Fn(Vec<DynIndex>, Vec<f64>) -> IdxTensor),
        vec![0, 1, 2],
    )
    .unwrap();

    let plain = ga
        .contract_zipup_with(&gb, &0, CanonicalForm::Unitary, None, Some(4))
        .unwrap();
    let contextual = ca
        .contract_zipup_with_in(&cb, &0, CanonicalForm::Unitary, None, Some(4), &context)
        .unwrap();
    assert_close(&dense_values(&plain), &dense_values(&contextual));
}

/// Regression for the index-identity rule: same-ID indices that differ by
/// prime level stay distinct through the context-scoped pairing and
/// factorization. If the route conflated them, the primed result would equal
/// the same-prime-level result.
#[test]
fn zipup_context_route_keeps_same_id_primed_indices_distinct() {
    let context = cpu_context();
    let sites: Vec<DynIndex> = (0..3).map(|_| DynIndex::new_dyn(2)).collect();
    let primed_sites: Vec<DynIndex> =
        vec![sites[0].clone(), sites[1].clone().prime(), sites[2].clone()];
    let build = in_context_build(&context);
    let center = 1usize;

    let same_plev = chain(3, 2, Some(&sites), &global_build)
        .contract_zipup_with(
            &chain(3, 2, Some(&sites), &global_build),
            &center,
            CanonicalForm::Unitary,
            None,
            Some(4),
        )
        .unwrap();
    let primed_plain = chain(3, 2, Some(&sites), &global_build)
        .contract_zipup_with(
            &chain(3, 2, Some(&primed_sites), &global_build),
            &center,
            CanonicalForm::Unitary,
            None,
            Some(4),
        )
        .unwrap();

    // Distinctness: the primed middle site is not conflated with the unprimed
    // one, so the result differs from the same-prime-level contraction.
    assert_ne!(
        dense_values(&same_plev).to_vec::<f64>().unwrap(),
        dense_values(&primed_plain).to_vec::<f64>().unwrap()
    );

    // The context route runs the same primed shape and keeps the same result
    // shape (values may differ from the plain route only at fp level).
    let primed_contextual = chain(3, 2, Some(&sites), &build)
        .contract_zipup_with_in(
            &chain(3, 2, Some(&primed_sites), &build),
            &center,
            CanonicalForm::Unitary,
            None,
            Some(4),
            &context,
        )
        .unwrap();
    assert_eq!(
        dense_values(&primed_contextual).dims(),
        dense_values(&primed_plain).dims()
    );
}

#[test]
fn non_unitary_forms_are_refused_instead_of_silently_global() {
    let context = cpu_context();
    let sites: Vec<DynIndex> = (0..2).map(|_| DynIndex::new_dyn(2)).collect();
    let build = in_context_build(&context);
    let (a, b) = (
        chain(2, 2, Some(&sites), &build),
        chain(2, 2, Some(&sites), &build),
    );
    for form in [CanonicalForm::LU, CanonicalForm::CI] {
        let error = a
            .contract_zipup_with_in(&b, &0, form, None, Some(4), &context)
            .unwrap_err();
        let message = format!("{error}");
        assert!(
            message.contains("no context-scoped"),
            "expected a typed refusal for {form:?}, got: {message}"
        );
    }
}
