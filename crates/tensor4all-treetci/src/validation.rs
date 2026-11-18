//! Run-owned samples for checking the current approximation against known values.

use crate::GlobalIndexBatch;
use anyhow::Result;
use std::collections::HashMap;
use tensor4all_core::index_key::{FlatIndexer, IndexKey};

#[derive(Debug)]
struct Sample<T> {
    point: Vec<usize>,
    value: T,
}

/// A bounded, insertion-ordered validation history, independent of oracle memoization.
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
    exhausted: bool,
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
            .ok_or_else(|| anyhow::anyhow!("validation sample size overflowed usize"))?;
        Ok(Self {
            indexer,
            positions: HashMap::new(),
            samples: Vec::new(),
            bytes_per_sample,
            limit,
            exhausted: false,
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
            let key = self.indexer.encode(point)?;
            if let Some(&position) = self.positions.get(&key) {
                self.samples[position].value = value.clone();
                continue;
            }
            if self.samples.len() >= self.limit / self.bytes_per_sample {
                self.exhausted = true;
                continue;
            }
            // INVARIANT: one payload copy per distinct retained point is
            // necessary for batched replay; integer keys deduplicate it and
            // the configured budget bounds its lifetime and retained count.
            self.positions.insert(key, self.samples.len());
            self.samples.push(Sample {
                point: point.to_vec(),
                value: value.clone(),
            });
        }
        Ok(())
    }

    pub(crate) fn finish_recording(&mut self) {
        self.recording = false;
    }

    pub(crate) fn len(&self) -> usize {
        self.samples.len()
    }
    pub(crate) fn retained_bytes(&self) -> usize {
        // INVARIANT: record bounds len by limit / bytes_per_sample, so this
        // product is at most the configured usize limit.
        self.samples.len() * self.bytes_per_sample
    }
    pub(crate) fn exhausted(&self) -> bool {
        self.exhausted
    }

    /// Copy one bounded batch into reusable readout scratch, in observation order.
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
