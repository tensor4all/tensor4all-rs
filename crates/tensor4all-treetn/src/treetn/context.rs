//! Binding a whole tree tensor network to a caller-owned execution context.

use crate::error::TreeTNOperationError;
use std::hash::Hash;

use anyhow::Context;

use super::TreeTN;
use tensor4all_core::TensorLike;

impl<T, V> TreeTN<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// Re-publish every node tensor in `context`.
    ///
    /// Use this to bind a network that was loaded from disk or materialised by
    /// TCI to a caller-owned context, so context-scoped operations accept it.
    /// Every tensor is re-published through
    /// [`TensorConstructionLike::adopt_into`](tensor4all_core::TensorConstructionLike::adopt_into)
    /// with the same index objects, so the topology, the site space and every
    /// bond rank are unchanged. The context-scoped truncation entry
    /// ([`TreeTN::truncate_in`](crate::TreeTN::truncate_in)) calls this itself,
    /// so a caller-side rebuild is not needed there.
    ///
    /// # Errors
    ///
    /// Returns a [`TreeTNOperationError`] when a tensor cannot be re-published:
    /// a payload that is neither dense `f64` nor dense `Complex64`, a context
    /// that refuses the publication, or a tensor whose eager runtime already
    /// belongs to a *different* execution context. Such a tensor cannot be
    /// re-bound without a readback and re-publish, i.e. a copy, so it is an
    /// error rather than a silent copy; build it in `context` instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tensor4all_core::IdxTensor;
    /// use tensor4all_tensorbackend::{CpuBackend, CpuExecutionContext, ExecutionContext};
    /// use tensor4all_treetn::{TreeTN, TruncationOptions};
    ///
    /// let context = ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(
    ///     CpuBackend::new(),
    /// )));
    /// let (s0, bond, s1) = (
    ///     tensor4all_core::DynIndex::new_dyn(2),
    ///     tensor4all_core::DynIndex::new_dyn(2),
    ///     tensor4all_core::DynIndex::new_dyn(2),
    /// );
    /// // Runtime-free host storage, as a TCI materialisation publishes it; a
    /// // disk load binds the implicit runtime and is adopted the same way.
    /// let mut tree = TreeTN::<_, usize>::from_tensors(
    ///     vec![
    ///         IdxTensor::from_dense(vec![s0, bond.clone()], vec![1.0, 0.0, 0.0, 1.0])?,
    ///         IdxTensor::from_dense(vec![bond, s1], vec![1.0, 0.0, 0.0, 1.0])?,
    ///     ],
    ///     vec![0, 1],
    /// )?;
    ///
    /// tree.adopt_into_in(&context)?;
    /// let shortened = tree.truncate_in(
    ///     [0],
    ///     TruncationOptions::default().with_max_bond_dim(1),
    ///     &context,
    /// )?;
    /// assert_eq!(shortened.link_dims(), vec![1]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn adopt_into_in(
        &mut self,
        context: &tensor4all_tensorbackend::ExecutionContext,
    ) -> Result<(), TreeTNOperationError> {
        for node in self.node_indices() {
            let Some(slot) = self.tensor_mut(node) else {
                continue;
            };
            let adopted = slot.adopt_into(context).with_context(|| {
                format!("adopt_into_in: failed to adopt the tensor at node {node:?}")
            })?;
            *slot = adopted;
        }
        Ok(())
    }
}
