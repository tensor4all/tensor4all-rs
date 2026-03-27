//! Utility for optimizing the first pivot for tensor cross interpolation.
//!
//! Port of Julia's `optfirstpivot` from TensorCrossInterpolation.jl.

use tensor4all_simplett::TTScalar;
use tensor4all_tcicore::{MultiIndex, Scalar};

/// Find a good initial pivot by greedy local search.
///
/// Sweeps through dimensions, selecting the index that maximizes `|f(pivot)|`.
/// Repeats until no improvement is found or `max_sweep` is reached.
///
/// # Arguments
/// * `f` - Function to interpolate
/// * `local_dims` - Local dimensions for each site
/// * `first_pivot` - Starting point for the search
/// * `max_sweep` - Maximum number of sweeps (default: 1000)
///
/// # Returns
/// The optimized pivot (multi-index with the largest `|f|` found)
pub fn opt_first_pivot<T, F>(
    f: &F,
    local_dims: &[usize],
    first_pivot: &MultiIndex,
    max_sweep: usize,
) -> MultiIndex
where
    T: Scalar + TTScalar,
    F: Fn(&MultiIndex) -> T,
{
    let n = local_dims.len();
    let mut pivot = first_pivot.to_vec();
    let mut val_f = f64::sqrt(Scalar::abs_sq(f(&pivot)));

    for _ in 0..max_sweep {
        let prev_val = val_f;
        for i in 0..n {
            for d in 0..local_dims[i] {
                let bak = pivot[i];
                pivot[i] = d;
                let new_val = f64::sqrt(Scalar::abs_sq(f(&pivot)));
                if new_val > val_f {
                    val_f = new_val;
                } else {
                    pivot[i] = bak;
                }
            }
        }
        if prev_val == val_f {
            break;
        }
    }

    pivot
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opt_first_pivot_finds_nonzero() {
        // Function that is zero at origin but nonzero elsewhere
        let f = |idx: &MultiIndex| (idx[0] as f64 + idx[1] as f64 + 1.0).powi(2);
        let local_dims = vec![4, 4];
        let first_pivot = vec![0, 0]; // f(0,0) = 1.0

        let pivot = opt_first_pivot::<f64, _>(&f, &local_dims, &first_pivot, 1000);
        // Should find the maximum: f(3,3) = 49.0
        assert_eq!(pivot, vec![3, 3]);
    }

    #[test]
    fn test_opt_first_pivot_already_optimal() {
        let f = |idx: &MultiIndex| idx[0] as f64 * idx[1] as f64;
        let local_dims = vec![4, 4];
        let first_pivot = vec![3, 3]; // Already at maximum

        let pivot = opt_first_pivot::<f64, _>(&f, &local_dims, &first_pivot, 1000);
        assert_eq!(pivot, vec![3, 3]);
    }

    #[test]
    fn test_opt_first_pivot_complex() {
        use num_complex::Complex64;
        let f = |idx: &MultiIndex| Complex64::new(idx[0] as f64 * 2.0, idx[1] as f64 * 3.0);
        let local_dims = vec![4, 4];
        let first_pivot = vec![0, 0];

        let pivot = opt_first_pivot::<Complex64, _>(&f, &local_dims, &first_pivot, 1000);
        // Maximum |f| at (3, 3) = |6 + 9i| = sqrt(117)
        assert_eq!(pivot, vec![3, 3]);
    }
}
