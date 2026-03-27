use anyhow::{ensure, Result};

/// Borrowed view of a global site-order batch.
#[derive(Clone, Copy, Debug)]
pub struct GlobalIndexBatch<'a> {
    data: &'a [usize],
    n_sites: usize,
    n_points: usize,
}

impl<'a> GlobalIndexBatch<'a> {
    /// Create a borrowed batch view with column-major `(n_sites, n_points)` storage.
    pub fn new(data: &'a [usize], n_sites: usize, n_points: usize) -> Result<Self> {
        ensure!(
            data.len() == n_sites * n_points,
            "global index batch has length {}, expected {}",
            data.len(),
            n_sites * n_points
        );
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
    pub fn get(&self, site: usize, point: usize) -> Option<usize> {
        (site < self.n_sites && point < self.n_points)
            .then(|| self.data[site + self.n_sites * point])
    }
}

/// Owned column-major batch buffer for global site-order evaluation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OwnedGlobalIndexBatch {
    data: Vec<usize>,
    n_sites: usize,
    n_points: usize,
}

impl OwnedGlobalIndexBatch {
    /// Create an owned batch buffer with column-major `(n_sites, n_points)` storage.
    pub fn new(data: Vec<usize>, n_sites: usize, n_points: usize) -> Result<Self> {
        GlobalIndexBatch::new(&data, n_sites, n_points)?;
        Ok(Self {
            data,
            n_sites,
            n_points,
        })
    }

    /// Borrow this batch as a view.
    pub fn as_view(&self) -> GlobalIndexBatch<'_> {
        GlobalIndexBatch {
            data: &self.data,
            n_sites: self.n_sites,
            n_points: self.n_points,
        }
    }

    /// Consume the batch and return the raw backing storage.
    pub fn into_vec(self) -> Vec<usize> {
        self.data
    }
}
