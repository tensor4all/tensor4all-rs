//! Forward-mode AD helpers for tensor4all tensors.

use anyhow::{anyhow, ensure, Result};

use crate::TensorDynLen;

/// Scoped forward-mode builder mirroring `tenferro::forward_ad::DualLevel`.
pub struct DualLevel<'a> {
    inner: &'a tenferro::forward_ad::DualLevel,
}

impl<'a> DualLevel<'a> {
    /// Creates a dual tensor from a primal tensor and its tangent seed.
    pub fn make_dual(&self, primal: &TensorDynLen, tangent: &TensorDynLen) -> Result<TensorDynLen> {
        ensure!(
            primal.indices() == tangent.indices(),
            "forward_ad::make_dual requires matching indices, got {:?} vs {:?}",
            primal.indices(),
            tangent.indices()
        );
        let native = self
            .inner
            .make_dual(primal.as_native(), tangent.as_native())
            .map_err(|e| anyhow!("forward_ad::make_dual failed: {e}"))?;
        TensorDynLen::from_native(primal.indices().to_vec(), native)
    }

    /// Unpacks a dual tensor into its detached primal value and optional tangent.
    pub fn unpack_dual(
        &self,
        value: &TensorDynLen,
    ) -> Result<(TensorDynLen, Option<TensorDynLen>)> {
        let (primal, tangent) = self
            .inner
            .unpack_dual(value.as_native())
            .map_err(|e| anyhow!("forward_ad::unpack_dual failed: {e}"))?;
        let primal = TensorDynLen::from_native(value.indices().to_vec(), primal)?;
        let tangent = tangent
            .map(|native| TensorDynLen::from_native(value.indices().to_vec(), native))
            .transpose()?;
        Ok((primal, tangent))
    }
}

/// Runs a scoped forward-mode computation.
pub fn dual_level<R>(f: impl for<'a> FnOnce(&DualLevel<'a>) -> Result<R>) -> Result<R> {
    tenferro::forward_ad::dual_level(|inner| {
        let wrapper = DualLevel { inner };
        f(&wrapper).map_err(|e| tenferro::Error::InvalidAdTensor {
            message: format!("tensor4all forward_ad wrapper failed: {e}"),
        })
    })
    .map_err(|e| anyhow!("forward_ad::dual_level failed: {e}"))
}
