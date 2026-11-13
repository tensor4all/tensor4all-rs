//! Restrict an enlarged candidate grid to the unchanged selected child basis.
use crate::{Result, TreeAciError, TreeAciScalar};

pub(crate) struct CandidateBasisProjection {
    pub(crate) source_len: usize,
    pub(crate) target_len: usize,
    run_len: usize,
    outer_dims: Vec<usize>,
    outer_strides: Vec<usize>,
    rewind: Vec<usize>,
}
impl CandidateBasisProjection {
    pub(crate) fn new(source: &[usize], basis: &[usize]) -> Result<Self> {
        if source.len() != basis.len()
            || source
                .iter()
                .zip(basis)
                .any(|(&source, &basis)| basis > source)
        {
            return Err(TreeAciError::InternalInvariant {
                message: "selected child basis must be a prefix of the candidate grid",
            });
        }
        let product = |dims: &[usize]| {
            dims.iter().try_fold(1usize, |n, &d| {
                n.checked_mul(d).ok_or(TreeAciError::SizeOverflow {
                    context: "candidate basis projection elements",
                })
            })
        };
        let source_len = product(source)?;
        let target_len = product(basis)?;
        let split = source
            .iter()
            .zip(basis)
            .position(|(a, b)| a != b)
            .map_or(source.len(), |axis| axis + 1);
        let run_len = product(&basis[..split])?;
        let mut stride = product(&source[..split])?;
        let mut outer_strides = Vec::with_capacity(source.len() - split);
        let mut rewind = Vec::with_capacity(source.len() - split);
        for (&basis_dim, &source_dim) in basis[split..].iter().zip(&source[split..]) {
            outer_strides.push(stride);
            rewind.push(basis_dim.saturating_sub(1).checked_mul(stride).ok_or(
                TreeAciError::SizeOverflow {
                    context: "candidate basis projection stride",
                },
            )?);
            stride = stride
                .checked_mul(source_dim)
                .ok_or(TreeAciError::SizeOverflow {
                    context: "candidate basis projection stride",
                })?;
        }
        Ok(Self {
            source_len,
            target_len,
            run_len,
            outer_dims: basis[split..].to_vec(),
            outer_strides,
            rewind,
        })
    }
    pub(crate) fn is_identity(&self) -> bool {
        self.source_len == self.target_len
    }
    pub(crate) fn project<T: TreeAciScalar>(&self, values: Vec<T>) -> Result<Vec<T>> {
        if values.len() != self.source_len {
            return Err(TreeAciError::InternalInvariant {
                message: "candidate factor length disagrees with its grid",
            });
        }
        if self.is_identity() {
            return Ok(values);
        }
        if self.target_len == 0 {
            return Ok(Vec::new());
        }
        let mut result = Vec::with_capacity(self.target_len);
        let mut cursor = vec![0usize; self.outer_dims.len()];
        let mut source = 0usize;
        while result.len() < self.target_len {
            // INVARIANT: the constructor validates a prefix hyperrectangle
            // inside the candidate grid. Each contiguous run is copied once;
            // every destination element is written, so zero-fill is unnecessary.
            result.extend_from_slice(&values[source..source + self.run_len]);
            for (axis, coordinate) in cursor.iter_mut().enumerate() {
                *coordinate += 1;
                if *coordinate < self.outer_dims[axis] {
                    source += self.outer_strides[axis];
                    break;
                }
                *coordinate = 0;
                source -= self.rewind[axis];
            }
        }
        Ok(result)
    }
}
#[cfg(test)]
mod tests;
