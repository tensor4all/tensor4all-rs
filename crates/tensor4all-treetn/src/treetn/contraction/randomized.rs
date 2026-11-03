//! Continuous-RNG factorization strategy for existing Fit/Zipup schedules.
use std::cell::RefCell;
use std::hash::Hash;

use anyhow::Result;
use rand::Rng;
use tensor4all_core::{
    FactorizeAlg, FactorizeOptions, FactorizeResult, IndexLike, RangeOptions, TensorLike,
};
use tensor4all_tensorbackend::ExecutionContext;

use super::{
    ContractionMethod, ContractionOptions, GlobalLeaves, LeafOps, SessionLeaves, ZipupTopologyMode,
};
use crate::{CanonicalForm, TreeTN, TreeTNOperationError};

struct RandomizedLeaves<'a, T: TensorLike, R: Rng + ?Sized> {
    base: Box<dyn LeafOps<T> + 'a>,
    rng: RefCell<&'a mut R>,
    options: RangeOptions,
}

pub(crate) fn leaves<'a, T: TensorLike + 'a, R: Rng + ?Sized + 'a>(
    base: Box<dyn LeafOps<T> + 'a>,
    options: RangeOptions,
    rng: &'a mut R,
) -> Box<dyn LeafOps<T> + 'a> {
    Box::new(RandomizedLeaves {
        base,
        rng: RefCell::new(rng),
        options,
    })
}

impl<T: TensorLike, R: Rng + ?Sized> LeafOps<T> for RandomizedLeaves<'_, T, R> {
    fn contract_pair(&self, a: &T, b: &T) -> Result<T> {
        self.base.contract_pair(a, b)
    }
    fn contract_list(&self, tensors: &[T]) -> Result<T> {
        self.base.contract_list(tensors)
    }
    fn contract_refs(&self, tensors: &[&T]) -> Result<T> {
        self.base.contract_refs(tensors)
    }
    fn outer_product(&self, a: &T, b: &T) -> Result<T> {
        self.base.outer_product(a, b)
    }

    fn factorize(
        &self,
        t: &T,
        left: &[T::Index],
        options: &FactorizeOptions,
    ) -> Result<FactorizeResult<T>> {
        let mut options = options.clone();
        options.alg = FactorizeAlg::RSVD;
        options.randomized = self.options;
        let mut rng = self
            .rng
            .try_borrow_mut()
            .map_err(|_| anyhow::anyhow!("randomized factorization RNG is already borrowed"))?;
        self.base.factorize_with_rng(t, left, &options, &mut *rng)
    }

    fn factorize_with_rng(
        &self,
        t: &T,
        left: &[T::Index],
        options: &FactorizeOptions,
        _rng: &mut dyn rand::RngCore,
    ) -> Result<FactorizeResult<T>> {
        self.factorize(t, left, options)
    }

    fn factorize_auto(
        &self,
        t: &T,
        left: &[T::Index],
        options: &FactorizeOptions,
    ) -> Result<FactorizeResult<T>> {
        // The deterministic Gram shortcut does not implement RSVD.
        self.factorize(t, left, options)
    }
}

fn dispatch<'a, T, V>(
    a: &TreeTN<T, V>,
    b: &TreeTN<T, V>,
    center: &V,
    options: ContractionOptions,
    leaves: Box<dyn LeafOps<T> + 'a>,
) -> Result<TreeTN<T, V>, TreeTNOperationError>
where
    T: TensorLike + 'a,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    match options.method {
        ContractionMethod::Fit => super::super::fit::contract_fit_with_leaves(
            a,
            b,
            center,
            super::fit_options_from(&options),
            leaves,
        ),
        ContractionMethod::Zipup => a
            .contract_zipup_impl(
                b,
                center,
                CanonicalForm::Unitary,
                options.svd_policy,
                options.max_bond_dim,
                ZipupTopologyMode::PruneScalarSubtrees,
                leaves.as_ref(),
            )
            .map_err(TreeTNOperationError::from),
        _ => Err(anyhow::anyhow!("caller-RNG RSVD contraction requires Fit or Zipup").into()),
    }
}

/// Contract using fixed-rank RSVD and a caller-owned RNG.
///
/// Select `FactorizeAlg::RSVD` with `Fit` or `Zipup` and a positive
/// `max_bond_dim`. The same RNG is consumed directly across initialization and
/// all sweeps; `rsvd_seed` is ignored. No threshold policy is accepted.
///
/// # Errors
/// Returns [`TreeTNOperationError`] for invalid options/topology, unsupported
/// tensor factorization, or an underlying contraction/decomposition failure.
///
/// # Examples
/// ```
/// use rand::SeedableRng;
/// use rand_chacha::ChaCha8Rng;
/// use tensor4all_core::{DynIndex, FactorizeAlg, IdxTensor};
/// use tensor4all_treetn::{TreeTN, contraction::{contract_with_rng, ContractionOptions}};
/// let i = DynIndex::new_dyn(2);
/// let a = TreeTN::from_tensors(vec![IdxTensor::from_dense(vec![i.clone()], vec![1., 2.])?], vec![0])?;
/// let b = TreeTN::from_tensors(vec![IdxTensor::from_dense(vec![i], vec![3., 4.])?], vec![0])?;
/// let mut rng = ChaCha8Rng::seed_from_u64(11);
/// let options = ContractionOptions::zipup().with_max_bond_dim(1).with_factorize_alg(FactorizeAlg::RSVD);
/// let result = contract_with_rng(&a, &b, &0, options, &mut rng)?;
/// assert!((result.to_dense()?.to_vec::<f64>()?[0] - 11.).abs() < 1e-12);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn contract_with_rng<T, V, R>(
    a: &TreeTN<T, V>,
    b: &TreeTN<T, V>,
    center: &V,
    options: ContractionOptions,
    rng: &mut R,
) -> Result<TreeTN<T, V>, TreeTNOperationError>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    R: Rng + ?Sized,
{
    super::validate_contract_options(&options)?;
    if options.factorize_alg != FactorizeAlg::RSVD {
        return Err(anyhow::anyhow!("contract_with_rng requires FactorizeAlg::RSVD; use contract for deterministic factorization").into());
    }
    let leaves = leaves(Box::new(GlobalLeaves), options.randomized, rng);
    dispatch(a, b, center, options, leaves)
}

/// Run caller-RNG RSVD contraction in a caller-owned execution context.
///
/// Same fixed-rank and RNG contract as [`contract_with_rng`]. Both inputs are
/// validated against `context` before consuming randomness; all leaves use it.
///
/// # Errors
/// Returns [`TreeTNOperationError`] for invalid options, foreign contexts,
/// unsupported topology/storage, or a backend failure.
///
/// # Examples
/// ```
/// use std::sync::Arc;
/// use rand::SeedableRng;
/// use rand_chacha::ChaCha8Rng;
/// use tensor4all_core::{DynIndex, ExecutionContext, FactorizeAlg, IdxTensor};
/// use tensor4all_tensorbackend::{CpuBackend, CpuExecutionContext};
/// use tensor4all_treetn::{TreeTN, contraction::{contract_with_rng_in, ContractionOptions}};
/// let context = ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(CpuBackend::with_threads(1)?)));
/// let i = DynIndex::new_dyn(2);
/// let a = TreeTN::from_tensors(vec![IdxTensor::from_dense_in(&context, vec![i.clone()], vec![1., 2.])?], vec![0])?;
/// let b = TreeTN::from_tensors(vec![IdxTensor::from_dense_in(&context, vec![i], vec![3., 4.])?], vec![0])?;
/// let mut rng = ChaCha8Rng::seed_from_u64(11);
/// let options = ContractionOptions::zipup().with_max_bond_dim(1).with_factorize_alg(FactorizeAlg::RSVD);
/// let result = contract_with_rng_in(&a, &b, &0, options, &mut rng, &context)?;
/// assert!((result.to_dense()?.to_vec::<f64>()?[0] - 11.).abs() < 1e-12);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn contract_with_rng_in<V, R>(
    a: &TreeTN<tensor4all_core::IdxTensor, V>,
    b: &TreeTN<tensor4all_core::IdxTensor, V>,
    center: &V,
    options: ContractionOptions,
    rng: &mut R,
    context: &ExecutionContext,
) -> Result<TreeTN<tensor4all_core::IdxTensor, V>, TreeTNOperationError>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    R: Rng + ?Sized,
{
    super::validate_contract_options(&options)?;
    if options.factorize_alg != FactorizeAlg::RSVD {
        return Err(anyhow::anyhow!("contract_with_rng_in requires FactorizeAlg::RSVD").into());
    }
    validate_untracked_context_inputs(a, b)?;
    a.validate_context(context)?;
    b.validate_context(context)?;
    let leaves = leaves(
        Box::new(SessionLeaves(context.clone())),
        options.randomized,
        rng,
    );
    dispatch(a, b, center, options, leaves)
}

pub(crate) fn validate_untracked_context_inputs<V>(
    a: &TreeTN<tensor4all_core::IdxTensor, V>,
    b: &TreeTN<tensor4all_core::IdxTensor, V>,
) -> Result<(), TreeTNOperationError>
where
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    for tree in [a, b] {
        for node in tree.node_indices() {
            if tree
                .tensor(node)
                .is_some_and(tensor4all_core::IdxTensor::tracks_grad)
            {
                return Err(anyhow::Error::new(tensor4all_core::FactorizeError::UnsupportedStorage(
                    "context-scoped TreeTN RSVD does not support tracked inputs; use core factorize_with_rng_in for differentiable local tensors")).into());
            }
        }
    }
    Ok(())
}
