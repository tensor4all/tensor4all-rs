//! Core types for tensor train operations

use mdarray::DTensor;

/// Local index type (index within a single tensor site)
pub type LocalIndex = usize;

/// Multi-index type (indices across all sites)
pub type MultiIndex = Vec<LocalIndex>;

/// A 3D tensor represented using mdarray
/// Shape is (left_dim, site_dim, right_dim)
pub type Tensor3<T> = DTensor<T, 3>;

/// Helper functions for Tensor3 operations
pub trait Tensor3Ops<T: Clone + Default> {
    /// Get the left (bond) dimension
    fn left_dim(&self) -> usize;

    /// Get the site (physical) dimension
    fn site_dim(&self) -> usize;

    /// Get the right (bond) dimension
    fn right_dim(&self) -> usize;

    /// Get element at (left, site, right)
    fn get3(&self, l: usize, s: usize, r: usize) -> &T;

    /// Get mutable element at (left, site, right)
    fn get3_mut(&mut self, l: usize, s: usize, r: usize) -> &mut T;

    /// Set element at (left, site, right)
    fn set3(&mut self, l: usize, s: usize, r: usize, value: T);

    /// Get a slice for fixed site index: returns (left_dim, right_dim) matrix as flat Vec
    fn slice_site(&self, s: usize) -> Vec<T>;

    /// Reshape this tensor to a matrix (left_dim * site_dim, right_dim)
    fn as_left_matrix(&self) -> (Vec<T>, usize, usize);

    /// Reshape this tensor to a matrix (left_dim, site_dim * right_dim)
    fn as_right_matrix(&self) -> (Vec<T>, usize, usize);
}

impl<T: Clone + Default> Tensor3Ops<T> for Tensor3<T> {
    fn left_dim(&self) -> usize {
        self.dim(0)
    }

    fn site_dim(&self) -> usize {
        self.dim(1)
    }

    fn right_dim(&self) -> usize {
        self.dim(2)
    }

    fn get3(&self, l: usize, s: usize, r: usize) -> &T {
        &self[[l, s, r]]
    }

    fn get3_mut(&mut self, l: usize, s: usize, r: usize) -> &mut T {
        &mut self[[l, s, r]]
    }

    fn set3(&mut self, l: usize, s: usize, r: usize, value: T) {
        self[[l, s, r]] = value;
    }

    fn slice_site(&self, s: usize) -> Vec<T> {
        let left_dim = self.left_dim();
        let right_dim = self.right_dim();
        let mut result = Vec::with_capacity(left_dim * right_dim);
        for l in 0..left_dim {
            for r in 0..right_dim {
                result.push(self[[l, s, r]].clone());
            }
        }
        result
    }

    fn as_left_matrix(&self) -> (Vec<T>, usize, usize) {
        let left_dim = self.left_dim();
        let site_dim = self.site_dim();
        let right_dim = self.right_dim();
        let rows = left_dim * site_dim;
        let cols = right_dim;
        let mut result = Vec::with_capacity(rows * cols);
        for l in 0..left_dim {
            for s in 0..site_dim {
                for r in 0..right_dim {
                    result.push(self[[l, s, r]].clone());
                }
            }
        }
        (result, rows, cols)
    }

    fn as_right_matrix(&self) -> (Vec<T>, usize, usize) {
        let left_dim = self.left_dim();
        let site_dim = self.site_dim();
        let right_dim = self.right_dim();
        let rows = left_dim;
        let cols = site_dim * right_dim;
        let mut result = Vec::with_capacity(rows * cols);
        for l in 0..left_dim {
            for s in 0..site_dim {
                for r in 0..right_dim {
                    result.push(self[[l, s, r]].clone());
                }
            }
        }
        (result, rows, cols)
    }
}

/// Create a zero-filled Tensor3
pub fn tensor3_zeros<T: Clone + Default>(
    left_dim: usize,
    site_dim: usize,
    right_dim: usize,
) -> Tensor3<T> {
    Tensor3::from_elem([left_dim, site_dim, right_dim], T::default())
}

/// Create a Tensor3 from flat data (row-major order)
pub fn tensor3_from_data<T: Clone>(
    data: Vec<T>,
    left_dim: usize,
    site_dim: usize,
    right_dim: usize,
) -> Tensor3<T> {
    assert_eq!(data.len(), left_dim * site_dim * right_dim);
    Tensor3::from_fn([left_dim, site_dim, right_dim], |idx| {
        let l = idx[0];
        let s = idx[1];
        let r = idx[2];
        data[(l * site_dim + s) * right_dim + r].clone()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor3_zeros() {
        let t: Tensor3<f64> = tensor3_zeros(2, 3, 4);
        assert_eq!(t.left_dim(), 2);
        assert_eq!(t.site_dim(), 3);
        assert_eq!(t.right_dim(), 4);

        for l in 0..2 {
            for s in 0..3 {
                for r in 0..4 {
                    assert_eq!(*t.get3(l, s, r), 0.0);
                }
            }
        }
    }

    #[test]
    fn test_tensor3_from_data() {
        let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
        let t = tensor3_from_data(data, 2, 3, 4);

        assert_eq!(t.left_dim(), 2);
        assert_eq!(t.site_dim(), 3);
        assert_eq!(t.right_dim(), 4);

        assert_eq!(*t.get3(0, 0, 0), 0.0);
        assert_eq!(*t.get3(0, 0, 1), 1.0);
        assert_eq!(*t.get3(0, 0, 3), 3.0);
        assert_eq!(*t.get3(0, 1, 0), 4.0);
        assert_eq!(*t.get3(1, 0, 0), 12.0);
        assert_eq!(*t.get3(1, 2, 3), 23.0);
    }

    #[test]
    fn test_get3_set3_get3_mut() {
        let mut t: Tensor3<f64> = tensor3_zeros(2, 3, 4);

        t.set3(1, 2, 3, 42.0);
        assert_eq!(*t.get3(1, 2, 3), 42.0);
        assert_eq!(*t.get3(0, 0, 0), 0.0);

        *t.get3_mut(0, 1, 2) = 7.5;
        assert_eq!(*t.get3(0, 1, 2), 7.5);
    }

    #[test]
    fn test_slice_site() {
        let mut t: Tensor3<f64> = tensor3_zeros(2, 3, 4);
        for l in 0..2 {
            for r in 0..4 {
                t.set3(l, 1, r, (l * 4 + r) as f64);
            }
        }

        let slice = t.slice_site(1);
        assert_eq!(slice.len(), 8); // 2 * 4
        assert_eq!(slice[0], 0.0); // l=0, r=0
        assert_eq!(slice[1], 1.0); // l=0, r=1
        assert_eq!(slice[2], 2.0); // l=0, r=2
        assert_eq!(slice[3], 3.0); // l=0, r=3
        assert_eq!(slice[4], 4.0); // l=1, r=0
        assert_eq!(slice[5], 5.0); // l=1, r=1

        let slice_zero = t.slice_site(0);
        assert!(slice_zero.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_as_left_matrix() {
        let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
        let t = tensor3_from_data(data, 2, 3, 4);

        let (mat, rows, cols) = t.as_left_matrix();
        assert_eq!(rows, 6); // 2 * 3
        assert_eq!(cols, 4);
        assert_eq!(mat.len(), 24);

        // The data should be laid out as (l, s, r) -> row = l*site_dim + s, col = r
        // First row (l=0, s=0): elements 0,1,2,3
        assert_eq!(mat[0], 0.0);
        assert_eq!(mat[1], 1.0);
        assert_eq!(mat[2], 2.0);
        assert_eq!(mat[3], 3.0);
        // Second row (l=0, s=1): elements 4,5,6,7
        assert_eq!(mat[4], 4.0);
    }

    #[test]
    fn test_as_right_matrix() {
        let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
        let t = tensor3_from_data(data, 2, 3, 4);

        let (mat, rows, cols) = t.as_right_matrix();
        assert_eq!(rows, 2); // left_dim
        assert_eq!(cols, 12); // 3 * 4
        assert_eq!(mat.len(), 24);

        // First row (l=0): elements 0..12
        assert_eq!(mat[0], 0.0);
        assert_eq!(mat[11], 11.0);
        // Second row (l=1): elements 12..24
        assert_eq!(mat[12], 12.0);
        assert_eq!(mat[23], 23.0);
    }
}
