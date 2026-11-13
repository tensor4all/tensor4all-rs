//! Exact zip-up by QR (#1285).
//!
//! A zero SVD threshold without a cap asks for an exact zip-up. These tests
//! check that such a request factorizes every block by QR, and that its result
//! equals the untruncated SVD schedule and the dense product to roundoff, with
//! the same bond dimensions and a valid canonical center. They cover the chain
//! schedule and the general-tree schedule, real and complex data, and the
//! process-global and caller-owned-context leaf routes.

use std::cell::RefCell;
use std::sync::Arc;

use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tensor4all_core::{
    DynIndex, FactorizeAlg, FactorizeOptions, FactorizeResult, IdxTensor, IndexLike,
    SvdTruncationPolicy, TensorElement,
};
use tensor4all_tensorbackend::{CpuBackend, CpuExecutionContext, ExecutionContext};

use crate::treetn::contraction::{
    GlobalLeaves, IdentityNodes, LeafOps, SessionLeaves, ZipupTopologyMode,
};
use crate::{CanonicalForm, TreeTN};

/// Delegates to `base` and records the algorithm of every factorization.
struct RecordingLeaves<'a> {
    base: &'a dyn LeafOps<IdxTensor>,
    algorithms: RefCell<Vec<FactorizeAlg>>,
}

impl<'a> RecordingLeaves<'a> {
    fn new(base: &'a dyn LeafOps<IdxTensor>) -> Self {
        Self {
            base,
            algorithms: RefCell::new(Vec::new()),
        }
    }

    fn take(&self) -> Vec<FactorizeAlg> {
        std::mem::take(&mut self.algorithms.borrow_mut())
    }
}

impl LeafOps<IdxTensor> for RecordingLeaves<'_> {
    fn contract_pair(&self, lhs: &IdxTensor, rhs: &IdxTensor) -> anyhow::Result<IdxTensor> {
        self.base.contract_pair(lhs, rhs)
    }

    fn contract_list(&self, tensors: &[IdxTensor]) -> anyhow::Result<IdxTensor> {
        self.base.contract_list(tensors)
    }

    fn contract_refs(&self, tensors: &[&IdxTensor]) -> anyhow::Result<IdxTensor> {
        self.base.contract_refs(tensors)
    }

    fn outer_product(&self, lhs: &IdxTensor, rhs: &IdxTensor) -> anyhow::Result<IdxTensor> {
        self.base.outer_product(lhs, rhs)
    }

    fn factorize(
        &self,
        tensor: &IdxTensor,
        left: &[DynIndex],
        options: &FactorizeOptions,
    ) -> anyhow::Result<FactorizeResult<IdxTensor>> {
        self.algorithms.borrow_mut().push(options.alg);
        self.base.factorize(tensor, left, options)
    }

    fn factorize_auto(
        &self,
        tensor: &IdxTensor,
        left: &[DynIndex],
        options: &FactorizeOptions,
    ) -> anyhow::Result<FactorizeResult<IdxTensor>> {
        self.algorithms.borrow_mut().push(options.alg);
        self.base.factorize_auto(tensor, left, options)
    }
}

/// An operator-like network (input and output site per node) and a state
/// (the operator's input site per node) on `edges`, bond dimension 3, with
/// seeded random entries.
fn operator_and_state<T: TensorElement>(
    edges: &[(usize, usize)],
    entry: fn(&mut ChaCha8Rng) -> T,
) -> (TreeTN<IdxTensor, usize>, TreeTN<IdxTensor, usize>) {
    let nodes = edges.len() + 1;
    let mut random = ChaCha8Rng::seed_from_u64(1285);
    let input: Vec<DynIndex> = (0..nodes).map(|_| DynIndex::new_dyn(2)).collect();
    let output: Vec<DynIndex> = (0..nodes).map(|_| DynIndex::new_dyn(2)).collect();
    let mut build = |sites: &dyn Fn(usize) -> Vec<DynIndex>| {
        let mut legs: Vec<Vec<DynIndex>> = (0..nodes).map(sites).collect();
        for &(left, right) in edges {
            let bond = DynIndex::new_dyn(3);
            legs[left].push(bond.clone());
            legs[right].push(bond);
        }
        let tensors = legs
            .into_iter()
            .map(|indices| {
                let size = indices.iter().map(IndexLike::dim).product();
                let data = (0..size).map(|_| entry(&mut random)).collect();
                IdxTensor::from_dense(indices, data).unwrap()
            })
            .collect();
        TreeTN::from_tensors(tensors, (0..nodes).collect()).unwrap()
    };
    let operator = build(&|node| vec![input[node].clone(), output[node].clone()]);
    let state = build(&|node| vec![input[node].clone()]);
    (operator, state)
}

fn real(random: &mut ChaCha8Rng) -> f64 {
    random.random_range(-1.0..1.0)
}

fn complex(random: &mut ChaCha8Rng) -> Complex64 {
    Complex64::new(
        random.random_range(-1.0..1.0),
        random.random_range(-1.0..1.0),
    )
}

fn context() -> ExecutionContext {
    ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::with_threads(1).unwrap(),
    )))
}

/// The bond dimension of every edge of `edges` in `network`.
fn bond_dims(network: &TreeTN<IdxTensor, usize>, edges: &[(usize, usize)]) -> Vec<usize> {
    edges
        .iter()
        .map(|(left, right)| {
            let edge = network.edge_between(left, right).unwrap();
            network.bond_index(edge).unwrap().dim()
        })
        .collect()
}

/// The dense tensor of `network`, read back to host storage.
fn dense(network: &TreeTN<IdxTensor, usize>, leaves: &dyn LeafOps<IdxTensor>) -> IdxTensor {
    let dense = network.contract_to_tensor_with_leaves(leaves).unwrap();
    IdxTensor::from_storage(dense.indices().to_vec(), dense.to_storage().unwrap()).unwrap()
}

fn host(tensor: &IdxTensor) -> IdxTensor {
    IdxTensor::from_storage(tensor.indices().to_vec(), tensor.to_storage().unwrap()).unwrap()
}

/// `chain` selects the chain schedule, which also records the edge
/// orientations; the general-tree schedule sets only the canonical region.
fn check<T: TensorElement>(
    edges: &[(usize, usize)],
    centers: &[usize],
    chain: bool,
    entry: fn(&mut ChaCha8Rng) -> T,
) {
    let context = context();
    for scoped in [false, true] {
        let (mut operator, mut state) = operator_and_state(edges, entry);
        let expected = host(&operator.contract_naive(&state).unwrap());
        if scoped {
            operator.adopt_into_in(&context).unwrap();
            state.adopt_into_in(&context).unwrap();
        }
        let session = SessionLeaves(context.clone());
        let base: &dyn LeafOps<IdxTensor> = if scoped { &session } else { &GlobalLeaves };
        let leaves = RecordingLeaves::new(base);
        let run = |center: usize, policy: SvdTruncationPolicy| {
            operator
                .contract_zipup_impl(
                    &state,
                    &center,
                    CanonicalForm::Unitary,
                    Some(policy),
                    None,
                    ZipupTopologyMode::PruneScalarSubtrees,
                    &IdentityNodes::new(),
                    &leaves,
                )
                .unwrap()
        };
        let scale = expected.maxabs().unwrap();
        for &center in centers {
            let exact = run(center, SvdTruncationPolicy::new(0.0));
            let exact_algorithms = leaves.take();
            assert!(
                !exact_algorithms.is_empty()
                    && exact_algorithms.iter().all(|alg| *alg == FactorizeAlg::QR),
                "an exact zip-up factorizes by QR only, got {exact_algorithms:?}"
            );

            // The smallest positive threshold keeps every nonzero singular
            // value, so this is the untruncated SVD schedule.
            let svd = run(center, SvdTruncationPolicy::new(f64::MIN_POSITIVE));
            let svd_algorithms = leaves.take();
            assert!(
                !svd_algorithms.is_empty()
                    && svd_algorithms.iter().all(|alg| *alg == FactorizeAlg::SVD),
                "a positive threshold keeps the SVD schedule, got {svd_algorithms:?}"
            );

            assert_eq!(bond_dims(&exact, edges), bond_dims(&svd, edges));

            let exact_dense = dense(&exact, base);
            let against_dense = exact_dense.sub(&expected).unwrap().maxabs().unwrap();
            let against_svd = exact_dense
                .sub(&dense(&svd, base))
                .unwrap()
                .maxabs()
                .unwrap();
            assert!(
                against_dense <= 1e-12 * scale && against_svd <= 1e-12 * scale,
                "center {center}: QR deviates by {against_dense:e} from the dense product \
                 and by {against_svd:e} from the SVD schedule (scale {scale:e})"
            );

            // Every tensor but the center is an isometry towards it, so the
            // center carries the whole norm.
            if chain {
                exact.validate_ortho_consistency().unwrap();
            }
            assert_eq!(exact.canonical_region(), &std::iter::once(center).collect());
            let center_tensor = host(exact.tensor(exact.node_index(&center).unwrap()).unwrap());
            let center_norm = center_tensor.norm().unwrap();
            let norm = expected.norm().unwrap();
            assert!(
                (center_norm - norm).abs() <= 1e-12 * norm,
                "center {center}: center norm {center_norm:e} differs from the norm {norm:e}"
            );
        }
    }
}

const CHAIN: [(usize, usize); 5] = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)];
/// Nodes 1 and 3 have degree three.
const BRANCHED: [(usize, usize); 5] = [(0, 1), (1, 2), (1, 3), (3, 4), (3, 5)];

#[test]
fn exact_chain_zipup_factorizes_by_qr_and_matches_untruncated_svd_real() {
    check(&CHAIN, &[5, 0, 2], true, real);
}

#[test]
fn exact_chain_zipup_factorizes_by_qr_and_matches_untruncated_svd_complex() {
    check(&CHAIN, &[5, 0, 2], true, complex);
}

#[test]
fn exact_tree_zipup_factorizes_by_qr_and_matches_untruncated_svd_real() {
    check(&BRANCHED, &[1, 3, 5], false, real);
}

#[test]
fn exact_tree_zipup_factorizes_by_qr_and_matches_untruncated_svd_complex() {
    check(&BRANCHED, &[1, 3, 5], false, complex);
}
