//! Run-owned samples for checking the current approximation against known values.

use crate::GlobalIndexBatch;
use anyhow::Result;
use std::collections::HashMap;
use tensor4all_core::index_key::{FlatIndexer, IndexKey};

#[derive(Debug)]
struct Sample<T> {
    key: IndexKey,
    point: Vec<usize>,
    value: T,
}

/// A bounded recent-sample pool with one protected residual witness.
///
/// Integer keys deduplicate samples; full points are payload needed for replay.
/// The byte budget accounts for payload and estimates container overhead. Spare
/// container capacity is excluded, so reported bytes are not process RSS.
#[derive(Debug)]
pub(crate) struct SampleHistory<T> {
    indexer: FlatIndexer,
    positions: HashMap<IndexKey, usize>,
    samples: Vec<Sample<T>>,
    bytes_per_sample: usize,
    limit: usize,
    next_evict: usize,
    witness: Option<usize>,
    best_error: f64,
    limit_reached: bool,
    evictions: usize,
    recording: bool,
}

impl<T: Clone> SampleHistory<T> {
    pub(crate) fn new(dims: &[usize], limit: usize) -> Result<Self> {
        let indexer = FlatIndexer::try_new(dims)?;
        let key = indexer.encode(&vec![0; dims.len()])?;
        // INVARIANT: key heap storage depends only on the indexer's packed
        // width. Full-point payload and conservative per-entry table overhead
        // bound the retained sample count without allocating the whole budget.
        let bytes_per_sample = dims
            .len()
            .checked_mul(std::mem::size_of::<usize>())
            .and_then(|n| n.checked_add(std::mem::size_of::<Sample<T>>()))
            .and_then(|n| n.checked_add(2 * std::mem::size_of::<(IndexKey, usize)>()))
            .and_then(|n| n.checked_add(key.owned_heap_bytes()))
            .and_then(|n| n.checked_add(key.owned_heap_bytes()))
            .ok_or_else(|| anyhow::anyhow!("validation sample size overflowed usize"))?;
        Ok(Self {
            indexer,
            positions: HashMap::new(),
            samples: Vec::new(),
            bytes_per_sample,
            limit,
            next_evict: 0,
            witness: None,
            best_error: 0.0,
            limit_reached: false,
            evictions: 0,
            recording: true,
        })
    }

    pub(crate) fn record(&mut self, batch: GlobalIndexBatch<'_>, values: &[T]) -> Result<()> {
        if !self.recording {
            return Ok(());
        }
        anyhow::ensure!(
            batch.n_sites() == self.indexer.len(),
            "validation sample site count mismatch"
        );
        anyhow::ensure!(
            batch.n_points() == values.len(),
            "validation sample value count mismatch"
        );
        for (point, value) in batch.data().chunks_exact(batch.n_sites()).zip(values) {
            self.insert(point, value, false)?;
        }
        Ok(())
    }

    fn insert(
        &mut self,
        point: &[usize],
        value: &T,
        replace_witness: bool,
    ) -> Result<Option<usize>> {
        let key = self.indexer.encode(point)?;
        if let Some(&position) = self.positions.get(&key) {
            self.samples[position].value.clone_from(value);
            return Ok(Some(position));
        }
        if self.samples.len() < self.limit / self.bytes_per_sample {
            let position = self.samples.len();
            // INVARIANT: the compact key is kept beside its single point/value
            // payload so eviction removes the index entry in constant time.
            // Both packed-key copies are included in the sample accounting.
            self.positions.insert(key.clone(), position);
            self.samples.push(Sample {
                key,
                point: point.to_vec(),
                value: value.clone(),
            });
            return Ok(Some(position));
        }
        self.limit_reached = true;
        if self.samples.is_empty() {
            return Ok(None);
        }
        let mut position = self.next_evict;
        if !replace_witness && self.witness == Some(position) {
            position = (position + 1) % self.samples.len();
            if self.witness == Some(position) {
                return Ok(None);
            }
        }
        let sample = &mut self.samples[position];
        self.positions.remove(&sample.key);
        self.positions.insert(key.clone(), position);
        sample.key = key;
        sample.point.clear();
        sample.point.extend_from_slice(point);
        sample.value.clone_from(value);
        self.next_evict = (position + 1) % self.samples.len();
        self.evictions = self.evictions.saturating_add(1);
        Ok(Some(position))
    }

    /// Start a new approximation's residual pass, keeping the old witness
    /// protected until the retained pool has been replayed.
    pub(crate) fn begin_validation(&mut self) {
        self.best_error = 0.0;
    }

    /// Protect the largest residual measured on this approximation. Ordinary
    /// oracle traffic cannot evict it; a stronger measured point may replace it.
    pub(crate) fn observe_residuals(
        &mut self,
        batch: GlobalIndexBatch<'_>,
        values: &[T],
        errors: &[f64],
    ) -> Result<()> {
        if !self.recording {
            return Ok(());
        }
        anyhow::ensure!(
            batch.n_sites() == self.indexer.len()
                && batch.n_points() == values.len()
                && values.len() == errors.len(),
            "validation residual batch shape mismatch"
        );
        anyhow::ensure!(
            errors
                .iter()
                .all(|error| error.is_finite() && *error >= 0.0),
            "validation residuals must be finite and nonnegative"
        );
        for ((point, value), &error) in batch
            .data()
            .chunks_exact(batch.n_sites())
            .zip(values)
            .zip(errors)
        {
            if error > self.best_error {
                let position = self.insert(point, value, true)?;
                self.best_error = error;
                if let Some(position) = position {
                    self.witness = Some(position);
                }
            }
        }
        Ok(())
    }

    pub(crate) fn contains(&self, point: &[usize]) -> Result<bool> {
        Ok(self.positions.contains_key(&self.indexer.encode(point)?))
    }

    pub(crate) fn finish_recording(&mut self) {
        self.recording = false;
    }

    pub(crate) fn len(&self) -> usize {
        self.samples.len()
    }
    pub(crate) fn retained_bytes(&self) -> usize {
        // INVARIANT: insert bounds len by limit / bytes_per_sample, so this
        // product is at most the configured usize limit.
        self.samples.len() * self.bytes_per_sample
    }
    pub(crate) fn limit_reached(&self) -> bool {
        self.limit_reached
    }
    pub(crate) fn evictions(&self) -> usize {
        self.evictions
    }

    /// Copy one bounded batch into reusable readout scratch, in slot order.
    pub(crate) fn copy_batch(
        &self,
        start: usize,
        count: usize,
        points: &mut Vec<usize>,
        values: &mut Vec<T>,
    ) -> usize {
        points.clear();
        values.clear();
        let end = start.saturating_add(count).min(self.samples.len());
        let Some(samples) = self.samples.get(start..end) else {
            return 0;
        };
        for sample in samples {
            points.extend_from_slice(&sample.point);
            values.push(sample.value.clone());
        }
        samples.len()
    }
}

#[cfg(test)]
mod tests;
