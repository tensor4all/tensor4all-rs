//! The per-patch engine seam of the patch driver.
//!
//! The driver runs one engine per patch. [`PatchEngine`] is what it needs
//! from that engine: whether the engine can run on a region at all, the run
//! itself, and the black-box engine of the residual estimator. A black-box
//! [`TreeInterpolator`] ([`BlackBox`], the engine of `patched_interpolate`)
//! runs on every region and samples the function; an element-wise engine
//! (`patched_elementwise`) runs only on regions that lie inside one patch of
//! every input, and receives the inputs restricted to the region.

use std::fmt::Debug;
use std::hash::Hash;

use tensor4all_core::ColMajorArrayRef;
use tensor4all_treetn::interpolation::{
    EvalContext, InterpolationError, InterpolationOutcome, InterpolationProblem, TreeInterpolator,
};

use super::layout::SiteLayout;
use super::PatchedInterpolationError;
use crate::Projector;

/// What the patch driver needs from the engine of one run.
pub(super) trait PatchEngine<T, V>
where
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    /// The black-box engine of the residual-TT estimator.
    type Residual: TreeInterpolator<T>;

    /// The engine that interpolates a measured residual.
    fn residual_engine(&self) -> &Self::Residual;

    /// Whether the engine can run on the region `projector`. A region it
    /// cannot run on is split before any evaluation.
    ///
    /// # Errors
    ///
    /// Returns an error when the region cannot be classified.
    fn runs_on(&self, projector: &Projector) -> Result<bool, PatchedInterpolationError>;

    /// Run the engine on one patch: `problem` holds the active sites, `fixed`
    /// the fixed coordinate of every site of `layout`, and `evaluate` the
    /// patch's sampler of the target function.
    ///
    /// # Errors
    ///
    /// Returns the engine's error.
    fn run<F>(
        &self,
        problem: &InterpolationProblem<V>,
        layout: &SiteLayout<V>,
        fixed: &[Option<usize>],
        projector: &Projector,
        ctx: &mut EvalContext<'_>,
        evaluate: F,
    ) -> Result<InterpolationOutcome<V>, InterpolationError>
    where
        F: FnMut(ColMajorArrayRef<'_, usize>, &mut EvalContext<'_>) -> anyhow::Result<Vec<T>>;
}

/// A black-box engine: it samples the function and runs on every region.
pub(super) struct BlackBox<'a, E>(pub(super) &'a E);

impl<T, V, E> PatchEngine<T, V> for BlackBox<'_, E>
where
    V: Clone + Hash + Eq + Ord + Debug + Send + Sync,
    E: TreeInterpolator<T>,
{
    type Residual = E;

    fn residual_engine(&self) -> &E {
        self.0
    }

    fn runs_on(&self, _projector: &Projector) -> Result<bool, PatchedInterpolationError> {
        Ok(true)
    }

    fn run<F>(
        &self,
        problem: &InterpolationProblem<V>,
        _layout: &SiteLayout<V>,
        _fixed: &[Option<usize>],
        _projector: &Projector,
        ctx: &mut EvalContext<'_>,
        evaluate: F,
    ) -> Result<InterpolationOutcome<V>, InterpolationError>
    where
        F: FnMut(ColMajorArrayRef<'_, usize>, &mut EvalContext<'_>) -> anyhow::Result<Vec<T>>,
    {
        self.0.interpolate(problem, ctx, evaluate)
    }
}
