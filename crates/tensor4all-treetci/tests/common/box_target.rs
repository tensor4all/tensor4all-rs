//! Simon's box from issue #1433, shared by its ordinary and heavy regressions.
//!
//! Three binary chains of `r` sites meet at a centre of local dimension 1
//! (site `3 r`). Bit `i` (weight `2^i`) of axis `j` sits at site `3 i + j`, so
//! the least significant bit is at the leaf. The target is a product over the
//! three axes of `t(v)`: 0.5 on the box faces `lo` and `hi`, 1 strictly
//! between them, and 0 outside. Its support is the cube `[lo, hi]^3`.

use anyhow::Result;
use tensor4all_core::IdxTensor;
use tensor4all_treetci::{
    crossinterpolate2, DefaultProposer, GlobalIndexBatch, StopRule, TreeTciEdge, TreeTciGraph,
    TreeTciOptions, TreeTciTermination,
};

/// Tolerance of every box run, relative to the largest sampled value (1).
pub const TOLERANCE: f64 = 1e-13;

/// One box: `r` bits per axis, faces at `lo` and `hi`, started at
/// `(start, start, start)`.
#[derive(Clone, Copy, Debug)]
pub struct BoxTarget {
    pub r: usize,
    pub lo: usize,
    pub hi: usize,
    pub start: usize,
}

/// The box of width 32 that the global search misses on `main` (#1433).
pub const BOX_R7: BoxTarget = BoxTarget {
    r: 7,
    lo: 64,
    hi: 95,
    start: 80,
};

/// The width-16 control box, which `main` already interpolates exactly.
pub const BOX_R6: BoxTarget = BoxTarget {
    r: 6,
    lo: 32,
    hi: 47,
    start: 40,
};

impl BoxTarget {
    fn centre(&self) -> usize {
        3 * self.r
    }

    fn t(&self, v: usize) -> f64 {
        if v == self.lo || v == self.hi {
            0.5
        } else if v > self.lo && v < self.hi {
            1.0
        } else {
            0.0
        }
    }

    /// The target at a full-site multi-index (the centre coordinate is 0).
    pub fn value(&self, point: &[usize]) -> f64 {
        (0..3)
            .map(|axis| {
                let v = (0..self.r).map(|bit| point[3 * bit + axis] << bit).sum();
                self.t(v)
            })
            .product()
    }

    fn graph(&self) -> Result<TreeTciGraph> {
        let (r, centre) = (self.r, self.centre());
        let mut edges: Vec<TreeTciEdge> = (0..3)
            .flat_map(|axis| {
                (0..r - 1).map(move |bit| TreeTciEdge::new(3 * bit + axis, 3 * (bit + 1) + axis))
            })
            .collect();
        edges.extend((0..3).map(|axis| TreeTciEdge::new(3 * (r - 1) + axis, centre)));
        Ok(TreeTciGraph::new(centre + 1, &edges)?)
    }

    /// Run every seed (tolerance [`TOLERANCE`], `max_iter = 40`, normalized,
    /// global pivots on, [`DefaultProposer`]) and assert that each one stops
    /// `Converged` with a dense error of at most [`TOLERANCE`].
    ///
    /// The dense target is built once and relabelled for each run's indices.
    ///
    /// # Errors
    ///
    /// Returns, wrapped in `anyhow`, the error of the step that failed:
    /// - `TreeTciError::InvalidGraph` from `TreeTciGraph::new` if the edge
    ///   list is not a tree (an edited `r` or edge construction);
    /// - `TreeTciError::InvalidConfiguration` or `TreeTciError::Operation`
    ///   from `crossinterpolate2` (invalid options, start out of range,
    ///   candidate or LU failure);
    /// - a "no site index" error when a materialized site has no site index;
    /// - an `IdxTensor` construction, relabelling, contraction or subtraction
    ///   error from the dense comparison.
    ///
    /// The target never fails. An inexact or non-converged run panics through
    /// the assertion rather than returning an error.
    pub fn assert_exact(&self, seeds: &[u64], stop_rule: StopRule) -> Result<()> {
        let centre = self.centre();
        let start: Vec<usize> = (0..=centre)
            .map(|site| {
                if site == centre {
                    0
                } else {
                    (self.start >> (site / 3)) & 1
                }
            })
            .collect();
        let mut values = Some(
            (0..1usize << centre)
                .map(|flat| {
                    let mut point = vec![0; centre + 1];
                    for (site, bit) in point[..centre].iter_mut().enumerate() {
                        *bit = (flat >> site) & 1;
                    }
                    self.value(&point)
                })
                .collect::<Vec<_>>(),
        );
        let mut reference: Option<IdxTensor> = None;
        for &seed in seeds {
            let mut dims = vec![2; centre];
            dims.push(1);
            let evaluate = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
                Ok(batch
                    .data()
                    .chunks_exact(batch.n_sites())
                    .map(|point| self.value(point))
                    .collect())
            };
            let options = TreeTciOptions {
                tolerance: TOLERANCE,
                stop_rule,
                max_iter: 40,
                seed: Some(seed),
                enable_global_pivots: true,
                scale_tolerance: true,
                ..Default::default()
            };
            let run = crossinterpolate2(
                evaluate,
                dims,
                self.graph()?,
                vec![start.clone()],
                options,
                Some(centre),
                &DefaultProposer,
                &mut (),
            )?;
            let indices = (0..=centre)
                .map(|site| {
                    run.treetn
                        .site_space(&site)
                        .and_then(|set| set.iter().next().cloned())
                        .ok_or_else(|| anyhow::anyhow!("no site index at {site}"))
                })
                .collect::<Result<Vec<_>>>()?;
            if reference.is_none() {
                let values = values
                    .take()
                    .ok_or_else(|| anyhow::anyhow!("dense values already consumed"))?;
                reference = Some(IdxTensor::from_dense(indices.clone(), values)?);
            }
            let stored = reference
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("no dense reference"))?;
            let target = stored.replace_indices(stored.indices(), &indices)?;
            let error = run.treetn.to_dense()?.sub(&target)?.maxabs()?;
            assert!(
                run.termination == TreeTciTermination::Converged && error <= TOLERANCE,
                "box r = {} seed {seed}, {stop_rule:?}: {:?} after {} iterations, dense error {error:e}",
                self.r,
                run.termination,
                run.errors.len(),
            );
        }
        Ok(())
    }
}
