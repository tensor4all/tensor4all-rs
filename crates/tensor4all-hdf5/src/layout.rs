//! Row-major / column-major data layout conversion.
//!
//! tensor4all-rs stores data in row-major (C) order, while ITensors.jl uses
//! column-major (Fortran) order. This module provides generic, error-returning
//! conversion functions.

use anyhow::{ensure, Result};
use ndarray::ArrayD;

/// Convert row-major flat data to column-major flat data.
///
/// Row-major (C order): last axis varies fastest.
/// Column-major (Fortran order): first axis varies fastest.
pub(crate) fn row_major_to_col_major<T: Clone>(data: &[T], dims: &[usize]) -> Result<Vec<T>> {
    if dims.is_empty() || data.len() <= 1 {
        return Ok(data.to_vec());
    }
    let expected: usize = dims.iter().product();
    ensure!(
        data.len() == expected,
        "Shape mismatch in row_major_to_col_major: data length {} vs shape {:?} (expected {})",
        data.len(),
        dims,
        expected
    );
    let arr = ArrayD::from_shape_vec(dims.to_vec(), data.to_vec())?;
    let arr_f = arr.reversed_axes();
    Ok(arr_f.iter().cloned().collect())
}

/// Convert column-major flat data to row-major flat data.
pub(crate) fn col_major_to_row_major<T: Clone>(data: &[T], dims: &[usize]) -> Result<Vec<T>> {
    if dims.is_empty() || data.len() <= 1 {
        return Ok(data.to_vec());
    }
    let expected: usize = dims.iter().product();
    ensure!(
        data.len() == expected,
        "Shape mismatch in col_major_to_row_major: data length {} vs shape {:?} (expected {})",
        data.len(),
        dims,
        expected
    );
    let reversed_dims: Vec<usize> = dims.iter().rev().copied().collect();
    let arr = ArrayD::from_shape_vec(reversed_dims, data.to_vec())?;
    let arr_c = arr.reversed_axes();
    Ok(arr_c.iter().cloned().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    #[test]
    fn test_row_col_major_roundtrip_f64() {
        let dims = vec![2, 3];
        let row_major = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let col_major = row_major_to_col_major(&row_major, &dims).unwrap();
        assert_eq!(col_major, vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);

        let back = col_major_to_row_major(&col_major, &dims).unwrap();
        assert_eq!(back, row_major);
    }

    #[test]
    fn test_row_col_major_3d() {
        let dims = vec![2, 3, 4];
        let n: usize = dims.iter().product();
        let row_major: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let col_major = row_major_to_col_major(&row_major, &dims).unwrap();
        let back = col_major_to_row_major(&col_major, &dims).unwrap();
        assert_eq!(back, row_major);
    }

    #[test]
    fn test_scalar_tensor() {
        let data = vec![42.0];
        let dims: Vec<usize> = vec![];
        let col = row_major_to_col_major(&data, &dims).unwrap();
        assert_eq!(col, vec![42.0]);
    }

    #[test]
    fn test_complex_roundtrip() {
        let dims = vec![2, 3];
        let row_major: Vec<Complex64> = (0..6)
            .map(|i| Complex64::new(i as f64, -(i as f64)))
            .collect();
        let col_major = row_major_to_col_major(&row_major, &dims).unwrap();
        let back = col_major_to_row_major(&col_major, &dims).unwrap();
        assert_eq!(back, row_major);
    }

    #[test]
    fn test_shape_mismatch_returns_error() {
        let data = vec![1.0, 2.0, 3.0];
        let dims = vec![2, 3]; // expects 6 elements
        let result = row_major_to_col_major(&data, &dims);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Shape mismatch"));
    }
}
