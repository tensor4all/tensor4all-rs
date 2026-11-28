//! Truncation methods for TreeTN.
//!
//! This module provides methods for truncating tree tensor networks.
//!
//! The truncation algorithm uses a two-site sweep (nsite=2) based on Euler tour traversal:
//! 1. First, canonicalize the network towards the specified center
//! 2. Generate a sweep plan that visits each edge twice (forward and backward)
//! 3. For each step: extract two adjacent nodes, perform SVD-based truncation, replace
//! 4. The canonical center moves along the sweep path
//!
//! This ensures all bonds are optimally truncated in both directions.

use crate::error::TreeTNOperationError;
use std::collections::HashSet;
use std::hash::Hash;

use anyhow::{Context, Result};

use tensor4all_core::{validate_svd_truncation_options, SvdTruncationPolicy};
use tensor4all_core::{IndexLike, TensorLike};
use tensor4all_tensorbackend::ExecutionContext;

use super::localupdate::{apply_local_update_sweep, LocalUpdateSweepPlan, TruncateUpdater};
use super::TreeTN;
use crate::options::{CanonicalizationOptions, TruncationOptions};
use crate::CanonicalForm;

impl<T, V> TreeTN<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// Truncate the network towards the specified center using options.
    ///
    /// This is the recommended unified API for truncation. It accepts:
    /// - Center nodes specified by their node names (V)
    /// - [`TruncationOptions`] to control the SVD policy and max_bond_dim
    ///
    /// # Algorithm
    /// 1. Canonicalize the network towards the center (required for truncation)
    /// 2. Generate a two-site sweep plan using Euler tour traversal
    /// 3. Apply SVD-based truncation at each step, visiting each edge twice
    ///
    /// # Errors
    ///
    /// Returns [`TreeTNOperationError`] when `max_bond_dim` is zero or the SVD policy is invalid,
    /// when more than one center node is given, or when the
    /// pre-canonicalization, sweep plan, or truncation sweep fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetn::{TreeTN, TruncationOptions};
    /// use tensor4all_core::{DynIndex, IdxTensor, TensorLike};
    ///
    /// // Build a 2-node chain
    /// let s0 = DynIndex::new_dyn(2);
    /// let bond = DynIndex::new_dyn(3);
    /// let s1 = DynIndex::new_dyn(2);
    ///
    /// let t0 = IdxTensor::from_dense(
    ///     vec![s0.clone(), bond.clone()],
    ///     vec![1.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0],
    /// ).unwrap();
    /// let t1 = IdxTensor::from_dense(
    ///     vec![bond.clone(), s1.clone()],
    ///     vec![1.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0],
    /// ).unwrap();
    ///
    /// let tn = TreeTN::<_, String>::from_tensors(
    ///     vec![t0, t1],
    ///     vec!["A".to_string(), "B".to_string()],
    /// ).unwrap();
    ///
    /// // Truncate with max rank 2 towards node "A"
    /// let tn = tn.truncate(
    ///     ["A".to_string()],
    ///     TruncationOptions::default().with_max_bond_dim(2),
    /// ).unwrap();
    ///
    /// // Bond dimension is now at most 2
    /// assert_eq!(tn.node_count(), 2);
    /// ```
    pub fn truncate(
        mut self,
        canonical_region: impl IntoIterator<Item = V>,
        options: TruncationOptions,
    ) -> std::result::Result<Self, TreeTNOperationError>
    where
        V: Ord,
        <T::Index as IndexLike>::Id: Ord,
    {
        self.truncate_impl(
            canonical_region,
            options.svd_policy(),
            options.truncation.max_bond_dim,
            "truncate",
        )?;
        Ok(self)
    }

    /// Truncate the network in-place towards the specified center using options.
    ///
    /// This is the `&mut self` version of [`Self::truncate`].
    /// # Errors
    ///
    /// Returns [`TreeTNOperationError`] under the same conditions as [`Self::truncate`].
    ///
    pub fn truncate_mut(
        &mut self,
        canonical_region: impl IntoIterator<Item = V>,
        options: TruncationOptions,
    ) -> std::result::Result<(), TreeTNOperationError>
    where
        V: Ord,
        <T::Index as IndexLike>::Id: Ord,
    {
        self.truncate_impl(
            canonical_region,
            options.svd_policy(),
            options.truncation.max_bond_dim,
            "truncate_mut",
        )
        .map_err(TreeTNOperationError::from)
    }

    /// Truncate the network towards the specified center, in a caller-owned
    /// execution context.
    ///
    /// The same algorithm and result as [`Self::truncate`], except that the
    /// pre-canonicalization and the two-site SVD sweep run in `context`, so the
    /// caller chooses the thread width of the numerics and no process-wide
    /// default context is entered.
    ///
    /// # Tensors that do not belong to `context` yet
    ///
    /// The tree is adopted into `context` first, through
    /// [`Self::adopt_into_in`], so a tree that has no context of its own works
    /// here without a caller-side rebuild: a TCI materialisation (which
    /// publishes runtime-free compact host storage), a tree loaded from disk and
    /// a tree built by the context-free constructors (both bound to the
    /// process-global implicit runtime). Adoption keeps the index objects and
    /// the ranks.
    ///
    /// A tensor whose eager runtime is already bound to a *different*
    /// execution context is an error: re-binding it means reading it back and
    /// re-publishing it, i.e. a copy, so it is refused instead of performed
    /// silently. The same holds for structured storage, which adoption would
    /// have to expand densely, and for a tensor that tracks gradients. Build
    /// such a tensor with
    /// [`IdxTensor::from_dense_in`](tensor4all_core::IdxTensor::from_dense_in) in
    /// `context` instead.
    ///
    /// # Arguments
    /// * `canonical_region` - The single center node to truncate towards
    /// * `options` - [`TruncationOptions`] (SVD policy and maximum bond dimension)
    /// * `context` - The caller-owned execution context the numerics run in
    ///
    /// # Errors
    ///
    /// Returns [`TreeTNOperationError`] under the same conditions as
    /// [`Self::truncate`]: invalid options, a center of more than one node, or a
    /// failed canonicalization or factorization step in `context`. It also
    /// fails when a tensor cannot be adopted into `context` (see above).
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_tensorbackend::{CpuBackend, CpuExecutionContext, ExecutionContext};
    /// use tensor4all_treetn::{TreeTN, TruncationOptions};
    ///
    /// let context = ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(
    ///     CpuBackend::with_threads(2)?,
    /// )));
    /// let (s0, s1, bond) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2), DynIndex::new_dyn(3));
    /// let t0 = IdxTensor::from_dense_in(
    ///     &context,
    ///     vec![s0, bond.clone()],
    ///     vec![1.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0],
    /// )?;
    /// let t1 = IdxTensor::from_dense_in(
    ///     &context,
    ///     vec![bond, s1],
    ///     vec![1.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0],
    /// )?;
    /// let tn = TreeTN::<_, usize>::from_tensors(vec![t0, t1], vec![0, 1])?;
    ///
    /// let tn = tn.truncate_in([0], TruncationOptions::default().with_max_bond_dim(2), &context)?;
    /// assert!(tn.link_dims().iter().all(|&d| d <= 2));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn truncate_in(
        mut self,
        canonical_region: impl IntoIterator<Item = V>,
        options: TruncationOptions,
        context: &ExecutionContext,
    ) -> std::result::Result<Self, TreeTNOperationError>
    where
        V: Ord,
        <T::Index as IndexLike>::Id: Ord,
    {
        self.truncate_mut_in(canonical_region, options, context)?;
        Ok(self)
    }

    /// Truncate the network in place towards the specified center, in a
    /// caller-owned execution context.
    ///
    /// This is the `&mut self` version of [`Self::truncate_in`].
    ///
    /// # Errors
    ///
    /// Returns [`TreeTNOperationError`] under the same conditions as
    /// [`Self::truncate_in`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_tensorbackend::{CpuBackend, CpuExecutionContext, ExecutionContext};
    /// use tensor4all_treetn::{TreeTN, TruncationOptions};
    ///
    /// let context = ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(
    ///     CpuBackend::with_threads(1)?,
    /// )));
    /// let (s0, s1, bond) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2), DynIndex::new_dyn(3));
    /// let t0 = IdxTensor::from_dense_in(
    ///     &context,
    ///     vec![s0, bond.clone()],
    ///     vec![1.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0],
    /// )?;
    /// let t1 = IdxTensor::from_dense_in(
    ///     &context,
    ///     vec![bond, s1],
    ///     vec![1.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0],
    /// )?;
    /// let mut tn = TreeTN::<_, usize>::from_tensors(vec![t0, t1], vec![0, 1])?;
    ///
    /// tn.truncate_mut_in([1], TruncationOptions::default().with_max_bond_dim(1), &context)?;
    /// assert_eq!(tn.link_dims(), vec![1]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn truncate_mut_in(
        &mut self,
        canonical_region: impl IntoIterator<Item = V>,
        options: TruncationOptions,
        context: &ExecutionContext,
    ) -> std::result::Result<(), TreeTNOperationError>
    where
        V: Ord,
        <T::Index as IndexLike>::Id: Ord,
    {
        self.truncate_impl_in(
            canonical_region,
            options.svd_policy(),
            options.truncation.max_bond_dim,
            "truncate_mut_in",
            context,
        )
        .map_err(TreeTNOperationError::from)
    }

    /// Internal implementation for truncation.
    ///
    /// Uses LocalUpdateSweepPlan with TruncateUpdater for full two-site sweeps.
    pub(crate) fn truncate_impl(
        &mut self,
        canonical_region: impl IntoIterator<Item = V>,
        svd_policy: Option<SvdTruncationPolicy>,
        max_bond_dim: Option<usize>,
        context_name: &str,
    ) -> Result<()>
    where
        V: Ord,
        <T::Index as IndexLike>::Id: Ord,
    {
        self.truncate_impl_scoped(
            canonical_region,
            svd_policy,
            max_bond_dim,
            context_name,
            None,
        )
    }

    /// Context-scoped truncation.
    ///
    /// Pre-canonicalization and the two-site SVD sweep run through the scoped
    /// factorization paths, so all intermediates stay in `context`.
    pub(crate) fn truncate_impl_in(
        &mut self,
        canonical_region: impl IntoIterator<Item = V>,
        svd_policy: Option<SvdTruncationPolicy>,
        max_bond_dim: Option<usize>,
        context_name: &str,
        context: &tensor4all_tensorbackend::ExecutionContext,
    ) -> Result<()>
    where
        V: Ord,
        <T::Index as IndexLike>::Id: Ord,
    {
        self.truncate_impl_scoped(
            canonical_region,
            svd_policy,
            max_bond_dim,
            context_name,
            Some(context),
        )
    }

    fn truncate_impl_scoped(
        &mut self,
        canonical_region: impl IntoIterator<Item = V>,
        svd_policy: Option<SvdTruncationPolicy>,
        max_bond_dim: Option<usize>,
        context_name: &str,
        context: Option<&tensor4all_tensorbackend::ExecutionContext>,
    ) -> Result<()>
    where
        V: Ord,
        <T::Index as IndexLike>::Id: Ord,
    {
        self.truncate_impl_with(
            canonical_region,
            svd_policy,
            max_bond_dim,
            context_name,
            |tree, center_node, form, label| {
                if let Some(execution) = context {
                    // A tree loaded from disk or materialised by TCI carries no
                    // eager runtime of its own, so the sweeps below would combine
                    // factors from two runtimes. Adopt the whole tree first.
                    if let Err(error) = tree.adopt_into_in(execution) {
                        let error = anyhow::Error::new(error);
                        let cause = error.root_cause().to_string();
                        return Err(error
                            .context(format!("{context_name}: context adoption failed: {cause}")));
                    }
                    tree.canonicalize_impl_in([center_node], form, label, execution)
                } else {
                    tree.canonicalize_impl([center_node], form, label)
                }
            },
            |tree, plan| {
                let mut updater = match context {
                    Some(execution) => {
                        TruncateUpdater::new_in(max_bond_dim, svd_policy, execution.clone())
                    }
                    None => TruncateUpdater::new(max_bond_dim, svd_policy),
                };
                apply_local_update_sweep(tree, plan, &mut updater).map_err(anyhow::Error::from)
            },
        )
    }

    fn truncate_impl_with<PreCanon, RunSweep>(
        &mut self,
        canonical_region: impl IntoIterator<Item = V>,
        svd_policy: Option<SvdTruncationPolicy>,
        max_bond_dim: Option<usize>,
        context_name: &str,
        pre_canonicalize: PreCanon,
        run_sweep: RunSweep,
    ) -> Result<()>
    where
        V: Ord,
        <T::Index as IndexLike>::Id: Ord,
        PreCanon: FnOnce(&mut Self, V, CanonicalForm, &str) -> Result<()>,
        RunSweep: FnOnce(&mut Self, &LocalUpdateSweepPlan<V>) -> Result<()>,
    {
        // Validate before the empty-center no-op shortcut so a NaN policy or
        // `max_bond_dim == 0` is rejected on single-node and empty sweeps too.
        validate_svd_truncation_options(max_bond_dim, svd_policy)
            .with_context(|| format!("{context_name}: invalid truncation options"))?;

        let center_nodes: HashSet<V> = canonical_region.into_iter().collect();
        if center_nodes.is_empty() {
            return Ok(());
        }
        if center_nodes.len() != 1 {
            return Err(anyhow::anyhow!(
                "truncate currently requires a single-node center, got {} nodes",
                center_nodes.len()
            )
            .context(format!("{context_name}: multi-node center not supported")));
        }

        let center_node = center_nodes
            .iter()
            .next()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("truncate requires a center node"))
            .context(format!("{context_name}: missing center node"))?;

        let canonicalize_options =
            CanonicalizationOptions::default().with_form(CanonicalForm::Unitary);
        pre_canonicalize(
            self,
            center_node.clone(),
            canonicalize_options.form,
            &format!("{context_name}: pre-canonicalize"),
        )?;

        let plan = LocalUpdateSweepPlan::from_treetn(self, &center_node, 2)
            .ok_or_else(|| {
                anyhow::anyhow!("Failed to create sweep plan from center {:?}", center_node)
            })
            .context(format!("{context_name}: sweep plan creation failed"))?;

        if plan.is_empty() {
            return Ok(());
        }

        run_sweep(self, &plan).context(format!("{context_name}: truncation sweep failed"))?;

        self.canonical_form = Some(CanonicalForm::Unitary);
        Ok(())
    }
}

impl<V> TreeTN<tensor4all_core::IdxTensor, V>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    /// Truncate this TreeTN towards `canonical_region` on a caller-owned
    /// explicit `Session`.
    ///
    /// Pre-canonicalization, the conditioning QR, the two-site SVD, and every
    /// internal contraction run through `session`, so no stage enters the
    /// process-global execution backend.
    ///
    /// # Rounding
    ///
    /// The concrete-session SVD multiplies `S * V^H` via dense contraction on
    /// `session`, whereas the legacy eager `truncate` path scales `V^H` by a
    /// compact diagonal tensor. Results are bit-identical across repeated
    /// session runs, and agree with eager `truncate` up to ~1 ULP rounding.
    ///
    /// # Errors
    ///
    /// Returns [`TreeTNOperationError`] when truncation options are invalid,
    /// when `canonical_region` has more than one node, or when canonicalization
    /// or a two-site factorization on `session` fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_tensorbackend::{explicit::Session, CpuBackend};
    /// use tensor4all_treetn::{TreeTN, TruncationOptions};
    ///
    /// let s0 = DynIndex::new_dyn(2);
    /// let s1 = DynIndex::new_dyn(2);
    /// let bond = DynIndex::new_dyn(3);
    /// let t0 = IdxTensor::from_dense(vec![s0, bond.clone()], vec![1.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0])?;
    /// let t1 = IdxTensor::from_dense(vec![bond, s1], vec![1.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0])?;
    /// let tn = TreeTN::<_, usize>::from_tensors(vec![t0, t1], vec![0, 1])?;
    ///
    /// let mut backend = CpuBackend::with_threads(1)?;
    /// let mut session = Session::new(backend.engine_mut());
    /// let truncated = tn.truncate_on_session(
    ///     [1],
    ///     TruncationOptions::default().with_max_bond_dim(1),
    ///     &mut session,
    /// )?;
    /// assert_eq!(truncated.link_dims(), vec![1]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn truncate_on_session(
        mut self,
        canonical_region: impl IntoIterator<Item = V>,
        options: TruncationOptions,
        session: &mut tensor4all_tensorbackend::explicit::Session<'_>,
    ) -> std::result::Result<Self, TreeTNOperationError> {
        self.truncate_impl_on_session(
            canonical_region,
            options.svd_policy(),
            options.truncation.max_bond_dim,
            "truncate_on_session",
            session,
        )
        .map_err(TreeTNOperationError::from)?;
        Ok(self)
    }

    /// Truncate this TreeTN in place towards `canonical_region` on a
    /// caller-owned explicit `Session`.
    ///
    /// The `&mut self` twin of [`Self::truncate_on_session`]. The network is
    /// left unchanged if an error is returned.
    ///
    /// # Errors
    ///
    /// Returns [`TreeTNOperationError`] under the same conditions as
    /// [`Self::truncate_on_session`].
    pub fn truncate_mut_on_session(
        &mut self,
        canonical_region: impl IntoIterator<Item = V>,
        options: TruncationOptions,
        session: &mut tensor4all_tensorbackend::explicit::Session<'_>,
    ) -> std::result::Result<(), TreeTNOperationError> {
        let original = self.clone();
        let taken = std::mem::take(self);
        match taken.truncate_on_session(canonical_region, options, session) {
            Ok(result) => {
                *self = result;
                Ok(())
            }
            Err(error) => {
                *self = original;
                Err(error)
            }
        }
    }

    fn truncate_impl_on_session(
        &mut self,
        canonical_region: impl IntoIterator<Item = V>,
        svd_policy: Option<SvdTruncationPolicy>,
        max_bond_dim: Option<usize>,
        context_name: &str,
        session: &mut tensor4all_tensorbackend::explicit::Session<'_>,
    ) -> Result<()> {
        let session = std::cell::RefCell::new(session);
        self.truncate_impl_with(
            canonical_region,
            svd_policy,
            max_bond_dim,
            context_name,
            |tree, center_node, form, label| {
                tree.canonicalize_impl_on_session(
                    [center_node],
                    form,
                    label,
                    &mut session.borrow_mut(),
                )
            },
            |tree, plan| {
                let mut session_ref = session.borrow_mut();
                let mut updater = super::localupdate::SessionTruncateUpdater {
                    max_bond_dim,
                    svd_policy,
                    session: &mut session_ref,
                };
                apply_local_update_sweep(tree, plan, &mut updater).map_err(anyhow::Error::from)
            },
        )
    }
}
