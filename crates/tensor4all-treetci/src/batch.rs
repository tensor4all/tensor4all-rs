use crate::error::Result as TreeTciResult;

pub(crate) fn checked_batch_len(n_sites: usize, n_points: usize) -> TreeTciResult<usize> {
    n_sites
        .checked_mul(n_points)
        .ok_or_else(|| anyhow::anyhow!("global index batch shape product overflowed usize"))
        .map_err(Into::into)
}

/// Maximum number of points per evaluator call when TreeTCI evaluates a large
/// point set it assembles itself (edge candidate matrices, materialized site
/// tensors).
///
/// Bounds the reused index buffer to `n_sites * EVALUATION_CHUNK_POINTS`
/// entries (16 MiB at 32 sites) instead of one entry per point and site,
/// while keeping each call large enough to amortize per-call overhead of the
/// user's batch evaluator.
pub(crate) const EVALUATION_CHUNK_POINTS: usize = 65_536;

/// Evaluate `n_points` global points in calls of at most `chunk_points`
/// points, reusing one index buffer.
///
/// `fill_point` is called once per point, in point order, with that point's
/// `n_sites` slots of the buffer. The slots still hold an earlier point, so
/// `fill_point` must assign every site. `what` names the points in the error
/// raised when the evaluator returns the wrong number of values; that error
/// gives the failing call's point range and the total point count.
///
/// Returns the values in point order, exactly as one call over all points
/// would.
pub(crate) fn evaluate_points_chunked<T, F, G>(
    n_sites: usize,
    n_points: usize,
    chunk_points: usize,
    mut fill_point: G,
    evaluate: &F,
    what: &str,
) -> anyhow::Result<Vec<T>>
where
    F: Fn(GlobalIndexBatch<'_>) -> anyhow::Result<Vec<T>>,
    G: FnMut(&mut [usize]),
{
    anyhow::ensure!(chunk_points > 0, "evaluation chunk size must be positive");
    anyhow::ensure!(
        n_sites > 0 && n_points > 0,
        "at least one point with one site is required"
    );
    let buffer_points = chunk_points.min(n_points);
    let mut buffer = vec![0usize; checked_batch_len(n_sites, buffer_points)?];
    let mut values = Vec::new();
    let mut done = 0;
    while done < n_points {
        let count = buffer_points.min(n_points - done);
        let data = &mut buffer[..count * n_sites];
        for point in data.chunks_exact_mut(n_sites) {
            fill_point(point);
        }
        let chunk = evaluate(GlobalIndexBatch::new(data, n_sites, count)?)?;
        anyhow::ensure!(
            chunk.len() == count,
            "batch evaluator returned {} values for {} {what} (points {}..{} of {})",
            chunk.len(),
            count,
            done,
            done + count,
            n_points
        );
        if count == n_points {
            // Single call: hand over the evaluator's buffer without a copy.
            return Ok(chunk);
        }
        if values.is_empty() {
            values.reserve_exact(n_points);
        }
        values.extend(chunk);
        done += count;
    }
    Ok(values)
}

/// Borrowed view of a global site-order batch.
///
/// The data is stored in column-major layout with shape `(n_sites, n_points)`.
/// Each column is one multi-index point, with `data[site + n_sites * point]`
/// giving the local index at `site` for `point`.
///
/// This type is the main interface for the batch evaluator closure passed to
/// [`crossinterpolate2`](crate::crossinterpolate2),
/// [`optimize_default`](crate::optimize_default),
/// [`optimize_with_proposer`](crate::optimize_with_proposer) and
/// [`to_treetn`](crate::to_treetn).
///
/// # Batch sizes
///
/// TreeTCI splits the point sets it assembles for edge candidate matrices and
/// materialized site tensors into batches of at most 65,536 points, so one
/// matrix or tensor may take several calls. Initial-pivot evaluation and the
/// global pivot search pass their point sets in a single batch each. An
/// evaluator must not rely on how points are grouped into calls.
///
/// # Examples
///
/// ```
/// use tensor4all_treetci::GlobalIndexBatch;
///
/// // 2 sites, 3 points: column-major layout
/// // point 0: (0, 1), point 1: (1, 0), point 2: (0, 0)
/// let data = vec![0, 1, 1, 0, 0, 0];
/// let batch = GlobalIndexBatch::new(&data, 2, 3).unwrap();
///
/// assert_eq!(batch.n_sites(), 2);
/// assert_eq!(batch.n_points(), 3);
/// assert_eq!(batch.get(0, 0), Some(0)); // site 0, point 0
/// assert_eq!(batch.get(1, 0), Some(1)); // site 1, point 0
/// assert_eq!(batch.get(0, 1), Some(1)); // site 0, point 1
/// assert_eq!(batch.get(0, 5), None);    // out of bounds
/// ```
#[derive(Clone, Copy, Debug)]
pub struct GlobalIndexBatch<'a> {
    data: &'a [usize],
    n_sites: usize,
    n_points: usize,
}

impl<'a> GlobalIndexBatch<'a> {
    /// Create a borrowed batch view with column-major `(n_sites, n_points)` storage.
    ///
    /// Returns an error if `data.len() != n_sites * n_points`.
    ///
    /// # Errors
    ///
    /// Returns an error when the construction or conversion fails (a shape or
    /// /// index mismatch, or a backend failure).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetci::GlobalIndexBatch;
    ///
    /// let data = vec![0, 1, 2, 3];
    /// let batch = GlobalIndexBatch::new(&data, 2, 2).unwrap();
    /// assert_eq!(batch.n_sites(), 2);
    /// assert_eq!(batch.n_points(), 2);
    ///
    /// // Wrong length produces an error
    /// assert!(GlobalIndexBatch::new(&data, 3, 2).is_err());
    /// ```
    pub fn new(data: &'a [usize], n_sites: usize, n_points: usize) -> TreeTciResult<Self> {
        let expected = checked_batch_len(n_sites, n_points)?;
        if data.len() != expected {
            return Err(anyhow::anyhow!(
                "global index batch has length {}, expected {}",
                data.len(),
                expected
            )
            .into());
        };
        Ok(Self {
            data,
            n_sites,
            n_points,
        })
    }

    /// Borrow the raw column-major backing storage.
    pub fn data(&self) -> &'a [usize] {
        self.data
    }

    /// Number of sites per point.
    pub fn n_sites(&self) -> usize {
        self.n_sites
    }

    /// Number of points in the batch.
    pub fn n_points(&self) -> usize {
        self.n_points
    }

    /// Get one value from `(site, point)` coordinates.
    ///
    /// Returns `None` if either index is out of bounds.
    pub fn get(&self, site: usize, point: usize) -> Option<usize> {
        (site < self.n_sites && point < self.n_points)
            .then(|| self.data[site + self.n_sites * point])
    }
}

/// Owned column-major batch buffer for global site-order evaluation.
///
/// Same layout as [`GlobalIndexBatch`] but owns its data. Useful for
/// constructing batches programmatically.
///
/// # Examples
///
/// ```
/// use tensor4all_treetci::OwnedGlobalIndexBatch;
///
/// let batch = OwnedGlobalIndexBatch::new(vec![0, 1, 1, 0], 2, 2).unwrap();
/// let view = batch.as_view();
/// assert_eq!(view.get(0, 0), Some(0));
/// assert_eq!(view.get(1, 0), Some(1));
///
/// let raw = batch.into_vec();
/// assert_eq!(raw, vec![0, 1, 1, 0]);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OwnedGlobalIndexBatch {
    data: Vec<usize>,
    n_sites: usize,
    n_points: usize,
}

impl OwnedGlobalIndexBatch {
    /// Create an owned batch buffer with column-major `(n_sites, n_points)` storage.
    ///
    /// Returns an error if `data.len() != n_sites * n_points`.
    ///
    /// # Errors
    ///
    /// Returns an error when the construction or conversion fails (a shape or
    /// /// index mismatch, or a backend failure).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetci::OwnedGlobalIndexBatch;
    ///
    /// let batch = OwnedGlobalIndexBatch::new(vec![10, 20, 30, 40], 2, 2).unwrap();
    /// assert_eq!(batch.as_view().n_sites(), 2);
    /// assert_eq!(batch.as_view().n_points(), 2);
    ///
    /// // Wrong length is an error
    /// assert!(OwnedGlobalIndexBatch::new(vec![1, 2, 3], 2, 2).is_err());
    /// ```
    pub fn new(data: Vec<usize>, n_sites: usize, n_points: usize) -> TreeTciResult<Self> {
        GlobalIndexBatch::new(&data, n_sites, n_points)?;
        Ok(Self {
            data,
            n_sites,
            n_points,
        })
    }

    /// Borrow this batch as a view.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetci::OwnedGlobalIndexBatch;
    ///
    /// let batch = OwnedGlobalIndexBatch::new(vec![0, 1, 2, 3], 2, 2).unwrap();
    /// let view = batch.as_view();
    /// assert_eq!(view.get(0, 0), Some(0));
    /// assert_eq!(view.get(1, 1), Some(3));
    /// ```
    pub fn as_view(&self) -> GlobalIndexBatch<'_> {
        GlobalIndexBatch {
            data: &self.data,
            n_sites: self.n_sites,
            n_points: self.n_points,
        }
    }

    /// Consume the batch and return the raw backing storage.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetci::OwnedGlobalIndexBatch;
    ///
    /// let batch = OwnedGlobalIndexBatch::new(vec![5, 6, 7, 8], 2, 2).unwrap();
    /// let raw = batch.into_vec();
    /// assert_eq!(raw, vec![5, 6, 7, 8]);
    /// ```
    pub fn into_vec(self) -> Vec<usize> {
        self.data
    }
}

#[cfg(test)]
mod tests {
    use super::{checked_batch_len, evaluate_points_chunked, GlobalIndexBatch};
    use std::cell::RefCell;

    /// Encode each point as a number so values pin both content and order.
    fn encode(batch: GlobalIndexBatch<'_>) -> anyhow::Result<Vec<u64>> {
        Ok(batch
            .data()
            .chunks(batch.n_sites())
            .map(|point| point.iter().fold(0u64, |acc, &v| acc * 100 + v as u64))
            .collect())
    }

    fn run_chunked(n_points: usize, chunk_points: usize) -> (Vec<u64>, Vec<usize>) {
        let calls = RefCell::new(Vec::new());
        let evaluate = |batch: GlobalIndexBatch<'_>| {
            calls.borrow_mut().push(batch.n_points());
            encode(batch)
        };
        let mut next = 0usize;
        let values = evaluate_points_chunked(
            3,
            n_points,
            chunk_points,
            |point| {
                point.copy_from_slice(&[next % 7, next / 7, 42]);
                next += 1;
            },
            &evaluate,
            "test points",
        )
        .unwrap();
        (values, calls.into_inner())
    }

    #[test]
    fn evaluate_points_chunked_matches_single_call_for_every_chunk_size() {
        let (reference, calls) = run_chunked(23, usize::MAX);
        assert_eq!(calls, vec![23]);
        let expected: Vec<u64> = (0..23u64)
            .map(|p| (p % 7) * 10_000 + (p / 7) * 100 + 42)
            .collect();
        assert_eq!(reference, expected);

        // A chunk size that does not divide the point count, one that does,
        // one point per call, and an exact single chunk.
        for (chunk, sizes) in [
            (5, vec![5, 5, 5, 5, 3]),
            (1, vec![1; 23]),
            (23, vec![23]),
            (24, vec![23]),
        ] {
            let (values, calls) = run_chunked(23, chunk);
            assert_eq!(values, reference, "chunk {chunk}");
            assert_eq!(calls, sizes, "chunk {chunk}");
        }
    }

    #[test]
    fn evaluate_points_chunked_rejects_invalid_input() {
        let evaluate = |batch: GlobalIndexBatch<'_>| encode(batch);
        assert!(evaluate_points_chunked(2, 3, 0, |_| {}, &evaluate, "p").is_err());
        assert!(evaluate_points_chunked(0, 3, 2, |_| {}, &evaluate, "p").is_err());
        assert!(evaluate_points_chunked(2, 0, 2, |_| {}, &evaluate, "p").is_err());

        // A wrong value count in a later chunk is reported with its size.
        let short = |batch: GlobalIndexBatch<'_>| -> anyhow::Result<Vec<u64>> {
            let n = batch.n_points();
            Ok(vec![0; if n == 3 { n } else { n - 1 }])
        };
        let error = evaluate_points_chunked(1, 5, 3, |point| point[0] = 0, &short, "test points")
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "batch evaluator returned 1 values for 2 test points (points 3..5 of 5)"
        );
    }

    #[test]
    fn checked_batch_len_accepts_valid_shape() {
        assert_eq!(checked_batch_len(2, 3).unwrap(), 6);
    }

    #[test]
    fn checked_batch_len_rejects_overflow() {
        assert!(checked_batch_len(usize::MAX, 2).is_err());
    }
}
