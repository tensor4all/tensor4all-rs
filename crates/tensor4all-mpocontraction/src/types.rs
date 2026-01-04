//! Core types for MPO (Matrix Product Operator) operations
//!
//! MPO site tensors are 4D tensors with shape (left_bond, site_dim_1, site_dim_2, right_bond).
//! - `left_bond`: Bond dimension connecting to the left neighbor
//! - `site_dim_1`: Physical index (e.g., row index for operators)
//! - `site_dim_2`: Physical index (e.g., column index, often contracted)
//! - `right_bond`: Bond dimension connecting to the right neighbor

use mdarray::DTensor;

/// Local index type (index within a single tensor site)
pub type LocalIndex = usize;

/// Multi-index type (indices across all sites)
pub type MultiIndex = Vec<LocalIndex>;

/// A 4D tensor represented using mdarray
/// Shape is (left_dim, site_dim_1, site_dim_2, right_dim)
pub type Tensor4<T> = DTensor<T, 4>;

/// Helper functions for Tensor4 operations
pub trait Tensor4Ops<T: Clone + Default> {
    /// Get the left (bond) dimension
    fn left_dim(&self) -> usize;

    /// Get the first site (physical) dimension
    fn site_dim_1(&self) -> usize;

    /// Get the second site (physical) dimension
    fn site_dim_2(&self) -> usize;

    /// Get the right (bond) dimension
    fn right_dim(&self) -> usize;

    /// Get element at (left, site1, site2, right)
    fn get4(&self, l: usize, s1: usize, s2: usize, r: usize) -> &T;

    /// Get mutable element at (left, site1, site2, right)
    fn get4_mut(&mut self, l: usize, s1: usize, s2: usize, r: usize) -> &mut T;

    /// Set element at (left, site1, site2, right)
    fn set4(&mut self, l: usize, s1: usize, s2: usize, r: usize, value: T);

    /// Get a slice for fixed site indices: returns (left_dim, right_dim) matrix as flat Vec
    fn slice_site(&self, s1: usize, s2: usize) -> Vec<T>;

    /// Reshape this tensor to a matrix (left_dim * site_dim_1 * site_dim_2, right_dim)
    fn as_left_matrix(&self) -> (Vec<T>, usize, usize);

    /// Reshape this tensor to a matrix (left_dim, site_dim_1 * site_dim_2 * right_dim)
    fn as_right_matrix(&self) -> (Vec<T>, usize, usize);

    /// Reshape this tensor to a matrix (left_dim * site_dim_1, site_dim_2 * right_dim)
    fn as_center_matrix(&self) -> (Vec<T>, usize, usize);
}

impl<T: Clone + Default> Tensor4Ops<T> for Tensor4<T> {
    fn left_dim(&self) -> usize {
        self.dim(0)
    }

    fn site_dim_1(&self) -> usize {
        self.dim(1)
    }

    fn site_dim_2(&self) -> usize {
        self.dim(2)
    }

    fn right_dim(&self) -> usize {
        self.dim(3)
    }

    fn get4(&self, l: usize, s1: usize, s2: usize, r: usize) -> &T {
        &self[[l, s1, s2, r]]
    }

    fn get4_mut(&mut self, l: usize, s1: usize, s2: usize, r: usize) -> &mut T {
        &mut self[[l, s1, s2, r]]
    }

    fn set4(&mut self, l: usize, s1: usize, s2: usize, r: usize, value: T) {
        self[[l, s1, s2, r]] = value;
    }

    fn slice_site(&self, s1: usize, s2: usize) -> Vec<T> {
        let left_dim = self.left_dim();
        let right_dim = self.right_dim();
        let mut result = Vec::with_capacity(left_dim * right_dim);
        for l in 0..left_dim {
            for r in 0..right_dim {
                result.push(self[[l, s1, s2, r]].clone());
            }
        }
        result
    }

    fn as_left_matrix(&self) -> (Vec<T>, usize, usize) {
        let left_dim = self.left_dim();
        let site_dim_1 = self.site_dim_1();
        let site_dim_2 = self.site_dim_2();
        let right_dim = self.right_dim();
        let rows = left_dim * site_dim_1 * site_dim_2;
        let cols = right_dim;
        let mut result = Vec::with_capacity(rows * cols);
        for l in 0..left_dim {
            for s1 in 0..site_dim_1 {
                for s2 in 0..site_dim_2 {
                    for r in 0..right_dim {
                        result.push(self[[l, s1, s2, r]].clone());
                    }
                }
            }
        }
        (result, rows, cols)
    }

    fn as_right_matrix(&self) -> (Vec<T>, usize, usize) {
        let left_dim = self.left_dim();
        let site_dim_1 = self.site_dim_1();
        let site_dim_2 = self.site_dim_2();
        let right_dim = self.right_dim();
        let rows = left_dim;
        let cols = site_dim_1 * site_dim_2 * right_dim;
        let mut result = Vec::with_capacity(rows * cols);
        for l in 0..left_dim {
            for s1 in 0..site_dim_1 {
                for s2 in 0..site_dim_2 {
                    for r in 0..right_dim {
                        result.push(self[[l, s1, s2, r]].clone());
                    }
                }
            }
        }
        (result, rows, cols)
    }

    fn as_center_matrix(&self) -> (Vec<T>, usize, usize) {
        let left_dim = self.left_dim();
        let site_dim_1 = self.site_dim_1();
        let site_dim_2 = self.site_dim_2();
        let right_dim = self.right_dim();
        let rows = left_dim * site_dim_1;
        let cols = site_dim_2 * right_dim;
        let mut result = Vec::with_capacity(rows * cols);
        for l in 0..left_dim {
            for s1 in 0..site_dim_1 {
                for s2 in 0..site_dim_2 {
                    for r in 0..right_dim {
                        result.push(self[[l, s1, s2, r]].clone());
                    }
                }
            }
        }
        (result, rows, cols)
    }
}

/// Create a zero-filled Tensor4
pub fn tensor4_zeros<T: Clone + Default>(
    left_dim: usize,
    site_dim_1: usize,
    site_dim_2: usize,
    right_dim: usize,
) -> Tensor4<T> {
    Tensor4::from_elem([left_dim, site_dim_1, site_dim_2, right_dim], T::default())
}

/// Create a Tensor4 from flat data (row-major order)
pub fn tensor4_from_data<T: Clone>(
    data: Vec<T>,
    left_dim: usize,
    site_dim_1: usize,
    site_dim_2: usize,
    right_dim: usize,
) -> Tensor4<T> {
    assert_eq!(data.len(), left_dim * site_dim_1 * site_dim_2 * right_dim);
    Tensor4::from_fn([left_dim, site_dim_1, site_dim_2, right_dim], |idx| {
        let l = idx[0];
        let s1 = idx[1];
        let s2 = idx[2];
        let r = idx[3];
        data[((l * site_dim_1 + s1) * site_dim_2 + s2) * right_dim + r].clone()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor4_zeros() {
        let t: Tensor4<f64> = tensor4_zeros(2, 3, 4, 5);
        assert_eq!(t.left_dim(), 2);
        assert_eq!(t.site_dim_1(), 3);
        assert_eq!(t.site_dim_2(), 4);
        assert_eq!(t.right_dim(), 5);

        for l in 0..2 {
            for s1 in 0..3 {
                for s2 in 0..4 {
                    for r in 0..5 {
                        assert_eq!(*t.get4(l, s1, s2, r), 0.0);
                    }
                }
            }
        }
    }

    #[test]
    fn test_tensor4_get_set() {
        let mut t: Tensor4<f64> = tensor4_zeros(2, 2, 2, 2);
        t.set4(0, 1, 0, 1, 3.14);
        assert_eq!(*t.get4(0, 1, 0, 1), 3.14);
        assert_eq!(*t.get4(0, 0, 0, 0), 0.0);
    }

    #[test]
    fn test_tensor4_from_data() {
        let data: Vec<f64> = (0..24).map(|x| x as f64).collect();
        let t = tensor4_from_data(data, 2, 3, 2, 2);

        assert_eq!(t.left_dim(), 2);
        assert_eq!(t.site_dim_1(), 3);
        assert_eq!(t.site_dim_2(), 2);
        assert_eq!(t.right_dim(), 2);

        // Check some values
        assert_eq!(*t.get4(0, 0, 0, 0), 0.0);
        assert_eq!(*t.get4(0, 0, 0, 1), 1.0);
        assert_eq!(*t.get4(0, 0, 1, 0), 2.0);
        assert_eq!(*t.get4(1, 2, 1, 1), 23.0);
    }

    #[test]
    fn test_slice_site() {
        let mut t: Tensor4<f64> = tensor4_zeros(2, 2, 2, 3);
        for l in 0..2 {
            for r in 0..3 {
                t.set4(l, 1, 0, r, (l * 3 + r) as f64);
            }
        }

        let slice = t.slice_site(1, 0);
        assert_eq!(slice.len(), 6); // 2 * 3
        assert_eq!(slice[0], 0.0); // l=0, r=0
        assert_eq!(slice[1], 1.0); // l=0, r=1
        assert_eq!(slice[2], 2.0); // l=0, r=2
        assert_eq!(slice[3], 3.0); // l=1, r=0
    }

    #[test]
    fn test_as_left_matrix() {
        let t: Tensor4<f64> = tensor4_from_data(
            (0..24).map(|x| x as f64).collect(),
            2,
            3,
            2,
            2,
        );

        let (mat, rows, cols) = t.as_left_matrix();
        assert_eq!(rows, 12); // 2 * 3 * 2
        assert_eq!(cols, 2);
        assert_eq!(mat.len(), 24);
    }

    #[test]
    fn test_as_right_matrix() {
        let t: Tensor4<f64> = tensor4_from_data(
            (0..24).map(|x| x as f64).collect(),
            2,
            3,
            2,
            2,
        );

        let (mat, rows, cols) = t.as_right_matrix();
        assert_eq!(rows, 2);
        assert_eq!(cols, 12); // 3 * 2 * 2
        assert_eq!(mat.len(), 24);
    }

    #[test]
    fn test_as_center_matrix() {
        let t: Tensor4<f64> = tensor4_from_data(
            (0..24).map(|x| x as f64).collect(),
            2,
            3,
            2,
            2,
        );

        let (mat, rows, cols) = t.as_center_matrix();
        assert_eq!(rows, 6); // 2 * 3
        assert_eq!(cols, 4); // 2 * 2
        assert_eq!(mat.len(), 24);
    }
}
