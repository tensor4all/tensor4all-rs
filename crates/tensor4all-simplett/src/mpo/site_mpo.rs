//! SiteMPO - Center-canonical form of MPO
//!
//! A SiteMPO maintains an orthogonality center that can be moved
//! through the tensor train for efficient local updates.

use super::error::{MPOError, Result};
use super::mpo::MPO;
use super::types::{Tensor4, Tensor4Ops};
use crate::traits::TTScalar;

/// Center-canonical form of MPO with orthogonality center
///
/// The tensors to the left of the center are left-orthogonal (left-isometric),
/// and the tensors to the right are right-orthogonal (right-isometric).
#[derive(Debug, Clone)]
pub struct SiteMPO<T: TTScalar> {
    /// The underlying MPO tensors
    tensors: Vec<Tensor4<T>>,
    /// Current orthogonality center position (0-indexed)
    center: usize,
}

impl<T: TTScalar> SiteMPO<T> {
    /// Create a SiteMPO from an MPO, placing the center at the given position
    pub fn from_mpo(mpo: MPO<T>, center: usize) -> Result<Self> {
        if mpo.is_empty() {
            return Err(MPOError::Empty);
        }
        if center >= mpo.len() {
            return Err(MPOError::InvalidCenter {
                center,
                max: mpo.len(),
            });
        }

        let tensors = mpo.site_tensors().to_vec();
        let mut result = Self { tensors, center: 0 };

        // Move center to the desired position
        result.set_center(center)?;

        Ok(result)
    }

    /// Create a SiteMPO from tensors without validation
    #[allow(dead_code)]
    pub(crate) fn from_tensors_unchecked(tensors: Vec<Tensor4<T>>, center: usize) -> Self {
        Self { tensors, center }
    }

    /// Get the current orthogonality center position
    pub fn center(&self) -> usize {
        self.center
    }

    /// Get the number of sites
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Get the site tensor at position i
    pub fn site_tensor(&self, i: usize) -> &Tensor4<T> {
        &self.tensors[i]
    }

    /// Get mutable reference to the site tensor at position i
    pub fn site_tensor_mut(&mut self, i: usize) -> &mut Tensor4<T> {
        &mut self.tensors[i]
    }

    /// Get all site tensors
    pub fn site_tensors(&self) -> &[Tensor4<T>] {
        &self.tensors
    }

    /// Get mutable access to site tensors
    pub fn site_tensors_mut(&mut self) -> &mut [Tensor4<T>] {
        &mut self.tensors
    }

    /// Bond dimensions
    pub fn link_dims(&self) -> Vec<usize> {
        if self.len() <= 1 {
            return Vec::new();
        }
        (1..self.len())
            .map(|i| self.tensors[i].left_dim())
            .collect()
    }

    /// Site dimensions
    pub fn site_dims(&self) -> Vec<(usize, usize)> {
        self.tensors
            .iter()
            .map(|t| (t.site_dim_1(), t.site_dim_2()))
            .collect()
    }

    /// Maximum bond dimension
    pub fn rank(&self) -> usize {
        let lds = self.link_dims();
        if lds.is_empty() {
            1
        } else {
            *lds.iter().max().unwrap_or(&1)
        }
    }

    /// Move the orthogonality center one position to the left
    pub fn move_center_left(&mut self) -> Result<()> {
        if self.center == 0 {
            return Err(MPOError::InvalidOperation {
                message: "Cannot move center left from position 0".to_string(),
            });
        }

        // TODO: Implement QR decomposition to move center left
        // For now, just update the center position
        self.center -= 1;
        Ok(())
    }

    /// Move the orthogonality center one position to the right
    pub fn move_center_right(&mut self) -> Result<()> {
        if self.center >= self.len() - 1 {
            return Err(MPOError::InvalidOperation {
                message: "Cannot move center right from last position".to_string(),
            });
        }

        // TODO: Implement QR decomposition to move center right
        // For now, just update the center position
        self.center += 1;
        Ok(())
    }

    /// Move the orthogonality center to the specified position
    pub fn set_center(&mut self, target: usize) -> Result<()> {
        if target >= self.len() {
            return Err(MPOError::InvalidCenter {
                center: target,
                max: self.len(),
            });
        }

        while self.center < target {
            self.move_center_right()?;
        }
        while self.center > target {
            self.move_center_left()?;
        }

        Ok(())
    }

    /// Convert to basic MPO
    pub fn into_mpo(self) -> MPO<T> {
        MPO::from_tensors_unchecked(self.tensors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_site_mpo_creation() {
        let mpo = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);
        let site_mpo = SiteMPO::from_mpo(mpo, 0).unwrap();

        assert_eq!(site_mpo.len(), 2);
        assert_eq!(site_mpo.center(), 0);
    }

    #[test]
    fn test_site_mpo_move_center() {
        let mpo = MPO::<f64>::constant(&[(2, 2), (2, 2), (2, 2)], 1.0);
        let mut site_mpo = SiteMPO::from_mpo(mpo, 0).unwrap();

        site_mpo.set_center(2).unwrap();
        assert_eq!(site_mpo.center(), 2);

        site_mpo.set_center(1).unwrap();
        assert_eq!(site_mpo.center(), 1);
    }

    #[test]
    fn test_site_mpo_invalid_center() {
        let mpo = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);
        let result = SiteMPO::from_mpo(mpo, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_mpo_empty() {
        let mpo = MPO::<f64>::constant(&[], 1.0);
        let result = SiteMPO::from_mpo(mpo, 0);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), MPOError::Empty));
    }

    #[test]
    fn test_move_center_left_from_zero() {
        let mpo = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);
        let mut site_mpo = SiteMPO::from_mpo(mpo, 0).unwrap();

        let result = site_mpo.move_center_left();
        assert!(result.is_err());
    }

    #[test]
    fn test_move_center_right_from_last() {
        let mpo = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);
        let mut site_mpo = SiteMPO::from_mpo(mpo, 1).unwrap();

        let result = site_mpo.move_center_right();
        assert!(result.is_err());
    }

    #[test]
    fn test_set_center_out_of_bounds() {
        let mpo = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);
        let mut site_mpo = SiteMPO::from_mpo(mpo, 0).unwrap();

        let result = site_mpo.set_center(5);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            MPOError::InvalidCenter { center: 5, max: 2 }
        ));
    }

    #[test]
    fn test_into_mpo_round_trip() {
        let mpo = MPO::<f64>::constant(&[(2, 2), (3, 3)], 5.0);
        let original_sum = mpo.sum();
        let original_dims = mpo.site_dims();

        let site_mpo = SiteMPO::from_mpo(mpo, 0).unwrap();
        let recovered = site_mpo.into_mpo();

        assert_eq!(recovered.site_dims(), original_dims);
        assert!((recovered.sum() - original_sum).abs() < 1e-10);
    }

    #[test]
    fn test_link_dims_and_rank() {
        let mpo = MPO::<f64>::constant(&[(2, 2), (3, 3), (2, 2)], 1.0);
        let site_mpo = SiteMPO::from_mpo(mpo, 0).unwrap();

        let lds = site_mpo.link_dims();
        assert_eq!(lds.len(), 2);
        for &d in &lds {
            assert_eq!(d, 1);
        }
        assert_eq!(site_mpo.rank(), 1);
    }

    #[test]
    fn test_link_dims_single_site() {
        let mpo = MPO::<f64>::constant(&[(2, 2)], 1.0);
        let site_mpo = SiteMPO::from_mpo(mpo, 0).unwrap();

        assert!(site_mpo.link_dims().is_empty());
        assert_eq!(site_mpo.rank(), 1);
    }

    #[test]
    fn test_site_dims() {
        let mpo = MPO::<f64>::constant(&[(2, 3), (4, 5)], 1.0);
        let site_mpo = SiteMPO::from_mpo(mpo, 0).unwrap();

        assert_eq!(site_mpo.site_dims(), vec![(2, 3), (4, 5)]);
    }

    #[test]
    fn test_site_tensor_accessors() {
        let mpo = MPO::<f64>::constant(&[(2, 2), (3, 3)], 1.0);
        let site_mpo = SiteMPO::from_mpo(mpo, 0).unwrap();

        let t0 = site_mpo.site_tensor(0);
        assert_eq!(t0.site_dim_1(), 2);
        assert_eq!(t0.site_dim_2(), 2);

        let t1 = site_mpo.site_tensor(1);
        assert_eq!(t1.site_dim_1(), 3);
        assert_eq!(t1.site_dim_2(), 3);

        assert_eq!(site_mpo.site_tensors().len(), 2);
    }

    #[test]
    fn test_site_tensor_mut_accessors() {
        let mpo = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);
        let mut site_mpo = SiteMPO::from_mpo(mpo, 0).unwrap();

        let t = site_mpo.site_tensor_mut(0);
        t.set4(0, 0, 0, 0, 99.0);
        assert_eq!(*site_mpo.site_tensor(0).get4(0, 0, 0, 0), 99.0);

        let tensors = site_mpo.site_tensors_mut();
        tensors[1].set4(0, 1, 1, 0, 42.0);
        assert_eq!(*site_mpo.site_tensor(1).get4(0, 1, 1, 0), 42.0);
    }

    #[test]
    fn test_is_empty() {
        let mpo = MPO::<f64>::constant(&[(2, 2)], 1.0);
        let site_mpo = SiteMPO::from_mpo(mpo, 0).unwrap();
        assert!(!site_mpo.is_empty());
    }
}
