//! TensorTrain implementation

use crate::error::{Result, TensorTrainError};
use crate::traits::{AbstractTensorTrain, TTScalar};
use crate::types::{tensor3_zeros, Tensor3, Tensor3Ops};

/// Tensor Train (Matrix Product State) representation
///
/// A tensor train represents a high-dimensional tensor as a product of
/// lower-dimensional tensors:
///
/// T\[i1, i2, ..., iL\] = A1\[i1\] * A2\[i2\] * ... * AL\[iL\]
///
/// where each Ak\[ik\] is a matrix of shape (rk-1, rk).
///
/// # Examples
///
/// ```
/// use tensor4all_simplett::{TensorTrain, AbstractTensorTrain};
///
/// // Create a constant tensor train: T[i,j,k] = 3.0 for all i,j,k
/// let tt = TensorTrain::<f64>::constant(&[2, 3, 4], 3.0);
///
/// assert_eq!(tt.len(), 3);
/// assert_eq!(tt.site_dims(), vec![2, 3, 4]);
///
/// // Evaluate at a specific index
/// let val = tt.evaluate(&[0, 1, 2]).unwrap();
/// assert!((val - 3.0).abs() < 1e-12);
///
/// // Sum over all indices: 3.0 * 2 * 3 * 4 = 72.0
/// let s = tt.sum();
/// assert!((s - 72.0).abs() < 1e-10);
/// ```
#[derive(Debug, Clone)]
pub struct TensorTrain<T: TTScalar> {
    /// The tensors that make up the tensor train
    /// Each tensor has shape (left_bond, site_dim, right_bond)
    tensors: Vec<Tensor3<T>>,
}

impl<T: TTScalar> TensorTrain<T> {
    /// Create a new tensor train from a list of 3D tensors
    ///
    /// Each tensor should have shape (left_bond, site_dim, right_bond)
    /// where the right_bond of tensor i equals the left_bond of tensor i+1.
    pub fn new(tensors: Vec<Tensor3<T>>) -> Result<Self> {
        // Validate dimensions
        for i in 0..tensors.len().saturating_sub(1) {
            if tensors[i].right_dim() != tensors[i + 1].left_dim() {
                return Err(TensorTrainError::DimensionMismatch { site: i });
            }
        }

        // First tensor should have left_dim = 1
        if !tensors.is_empty() && tensors[0].left_dim() != 1 {
            return Err(TensorTrainError::InvalidOperation {
                message: "First tensor must have left dimension 1".to_string(),
            });
        }

        // Last tensor should have right_dim = 1
        if !tensors.is_empty() && tensors.last().unwrap().right_dim() != 1 {
            return Err(TensorTrainError::InvalidOperation {
                message: "Last tensor must have right dimension 1".to_string(),
            });
        }

        Ok(Self { tensors })
    }

    /// Create a tensor train from tensors without dimension validation
    /// (for internal use when dimensions are known to be correct)
    pub(crate) fn from_tensors_unchecked(tensors: Vec<Tensor3<T>>) -> Self {
        Self { tensors }
    }

    /// Create a tensor train representing the zero function
    pub fn zeros(site_dims: &[usize]) -> Self {
        let tensors: Vec<Tensor3<T>> = site_dims.iter().map(|&d| tensor3_zeros(1, d, 1)).collect();
        Self { tensors }
    }

    /// Create a tensor train representing a constant function
    pub fn constant(site_dims: &[usize], value: T) -> Self {
        if site_dims.is_empty() {
            return Self {
                tensors: Vec::new(),
            };
        }

        let n = site_dims.len();
        let mut tensors = Vec::with_capacity(n);

        // First tensor: all ones
        let mut first = tensor3_zeros(1, site_dims[0], 1);
        for s in 0..site_dims[0] {
            first.set3(0, s, 0, T::one());
        }
        tensors.push(first);

        // Middle tensors: all ones (only if n > 2)
        if n > 2 {
            for &d in &site_dims[1..n - 1] {
                let mut tensor = tensor3_zeros(1, d, 1);
                for s in 0..d {
                    tensor.set3(0, s, 0, T::one());
                }
                tensors.push(tensor);
            }
        }

        // Last tensor: multiply by value
        if n > 1 {
            let mut last = tensor3_zeros(1, site_dims[n - 1], 1);
            for s in 0..site_dims[n - 1] {
                last.set3(0, s, 0, value);
            }
            tensors.push(last);
        } else {
            // Single site: multiply the first (and only) tensor by value
            for s in 0..site_dims[0] {
                let current = *tensors[0].get3(0, s, 0);
                tensors[0].set3(0, s, 0, current * value);
            }
        }

        Self { tensors }
    }

    /// Get mutable access to the site tensors
    pub fn site_tensors_mut(&mut self) -> &mut [Tensor3<T>] {
        &mut self.tensors
    }

    /// Multiply the tensor train by a scalar
    pub fn scale(&mut self, factor: T) {
        if !self.tensors.is_empty() {
            let last = self.tensors.len() - 1;
            let tensor = &mut self.tensors[last];
            for l in 0..tensor.left_dim() {
                for s in 0..tensor.site_dim() {
                    for r in 0..tensor.right_dim() {
                        let val = *tensor.get3(l, s, r);
                        tensor.set3(l, s, r, val * factor);
                    }
                }
            }
        }
    }

    /// Create a scaled copy of the tensor train
    pub fn scaled(&self, factor: T) -> Self {
        let mut result = self.clone();
        result.scale(factor);
        result
    }

    /// Reverse the tensor train (swap left and right)
    pub fn reverse(&self) -> Self {
        let mut new_tensors = Vec::with_capacity(self.tensors.len());
        for tensor in self.tensors.iter().rev() {
            // Swap left and right dimensions
            let mut new_tensor =
                tensor3_zeros(tensor.right_dim(), tensor.site_dim(), tensor.left_dim());
            for l in 0..tensor.left_dim() {
                for s in 0..tensor.site_dim() {
                    for r in 0..tensor.right_dim() {
                        new_tensor.set3(r, s, l, *tensor.get3(l, s, r));
                    }
                }
            }
            new_tensors.push(new_tensor);
        }
        Self {
            tensors: new_tensors,
        }
    }
}

impl<T: TTScalar> TensorTrain<T> {
    /// Convert the tensor train to a full tensor
    ///
    /// Returns a flat vector containing all tensor elements in column-major order,
    /// along with the shape (site dimensions).
    ///
    /// Warning: This can be very large for high-dimensional tensors!
    pub fn fulltensor(&self) -> (Vec<T>, Vec<usize>) {
        if self.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let site_dims: Vec<usize> = self.site_dims();
        let total_size: usize = site_dims.iter().product();

        if total_size == 0 {
            return (Vec::new(), site_dims);
        }

        // Build full tensor by iterating over all indices
        let mut result = Vec::with_capacity(total_size);
        let mut indices = vec![0usize; site_dims.len()];

        loop {
            // Evaluate at current indices
            if let Ok(val) = self.evaluate(&indices) {
                result.push(val);
            } else {
                result.push(T::zero());
            }

            // Increment indices in column-major order, leftmost index fastest.
            let mut carry = true;
            for i in 0..site_dims.len() {
                if carry {
                    indices[i] += 1;
                    if indices[i] >= site_dims[i] {
                        indices[i] = 0;
                    } else {
                        carry = false;
                    }
                }
            }

            if carry {
                break; // All indices wrapped around
            }
        }

        (result, site_dims)
    }
}

impl<T: TTScalar> TensorTrain<T> {
    /// Sum over selected dimensions, returning a new TensorTrain with remaining dimensions.
    ///
    /// `dims` is a slice of 0-indexed site positions to sum over.
    /// If all dimensions are summed, returns a 1-site TensorTrain wrapping the scalar result.
    ///
    /// Port of Julia's `_sum(tt; dims)` from `abstracttensortrain.jl`.
    pub fn partial_sum(&self, dims: &[usize]) -> Result<TensorTrain<T>> {
        use tensor4all_tcicore::matrix::{mat_mul, ncols, nrows, zeros as mat_zeros};

        let n = self.len();
        if n == 0 {
            return Ok(Self {
                tensors: Vec::new(),
            });
        }

        // Validate dims
        for &d in dims {
            if d >= n {
                return Err(TensorTrainError::InvalidOperation {
                    message: format!("Dimension {} out of range (0..{})", d, n),
                });
            }
        }

        let mut result_tensors: Vec<Tensor3<T>> = Vec::new();

        // Tprod: accumulator matrix, starts as 1x1 identity
        let mut tprod = mat_zeros(1, 1);
        tprod[[0, 0]] = T::one();

        for site in 0..n {
            let t = self.site_tensor(site);
            let left_dim = t.left_dim();
            let site_dim = t.site_dim();
            let right_dim = t.right_dim();

            if dims.contains(&site) {
                // Sum over site index: result is (left_dim, right_dim) matrix
                // sum(T, dims=2)[:, 1, :] in Julia
                let mut site_sum = mat_zeros(left_dim, right_dim);
                for l in 0..left_dim {
                    for r in 0..right_dim {
                        let mut acc = T::zero();
                        for s in 0..site_dim {
                            acc = acc + *t.get3(l, s, r);
                        }
                        site_sum[[l, r]] = acc;
                    }
                }
                // Tprod = Tprod * site_sum
                tprod = mat_mul(&tprod, &site_sum);
            } else {
                // Keep this dimension: multiply Tprod into the site tensor
                // Tprod (tprod_rows, left_dim) * T reshaped to (left_dim, site_dim * right_dim)
                let tprod_rows = nrows(&tprod);
                let mut t_reshaped = mat_zeros(left_dim, site_dim * right_dim);
                for l in 0..left_dim {
                    for s in 0..site_dim {
                        for r in 0..right_dim {
                            t_reshaped[[l, s * right_dim + r]] = *t.get3(l, s, r);
                        }
                    }
                }
                let product = mat_mul(&tprod, &t_reshaped);

                // Reshape product (tprod_rows, site_dim * right_dim)
                // into tensor (tprod_rows, site_dim, right_dim)
                let mut new_tensor = tensor3_zeros(tprod_rows, site_dim, right_dim);
                for l in 0..tprod_rows {
                    for s in 0..site_dim {
                        for r in 0..right_dim {
                            new_tensor.set3(l, s, r, product[[l, s * right_dim + r]]);
                        }
                    }
                }
                result_tensors.push(new_tensor);

                // Reset Tprod to identity of size right_dim
                tprod = mat_zeros(right_dim, right_dim);
                for i in 0..right_dim {
                    tprod[[i, i]] = T::one();
                }
            }
        }

        if result_tensors.is_empty() {
            // All dims summed → return 1-site TT wrapping scalar
            // tprod should be 1×1
            let scalar = tprod[[0, 0]];
            let mut t = tensor3_zeros(1, 1, 1);
            t.set3(0, 0, 0, scalar);
            return TensorTrain::new(vec![t]);
        }

        // Contract final Tprod into last result tensor
        let last = result_tensors.last().unwrap();
        let last_left = last.left_dim();
        let last_site = last.site_dim();
        let last_right = last.right_dim();
        let tprod_cols = ncols(&tprod);

        // Reshape last tensor to (last_left * last_site, last_right)
        let mut last_mat = mat_zeros(last_left * last_site, last_right);
        for l in 0..last_left {
            for s in 0..last_site {
                for r in 0..last_right {
                    last_mat[[l * last_site + s, r]] = *last.get3(l, s, r);
                }
            }
        }

        // Multiply: last_mat * Tprod → (last_left * last_site, tprod_cols)
        let contracted = mat_mul(&last_mat, &tprod);

        // Reshape back to tensor (last_left, last_site, tprod_cols)
        let mut new_last = tensor3_zeros(last_left, last_site, tprod_cols);
        for l in 0..last_left {
            for s in 0..last_site {
                for r in 0..tprod_cols {
                    new_last.set3(l, s, r, contracted[[l * last_site + s, r]]);
                }
            }
        }
        *result_tensors.last_mut().unwrap() = new_last;

        TensorTrain::new(result_tensors)
    }
}

impl<T: TTScalar> AbstractTensorTrain<T> for TensorTrain<T> {
    fn len(&self) -> usize {
        self.tensors.len()
    }

    fn site_tensor(&self, i: usize) -> &Tensor3<T> {
        &self.tensors[i]
    }

    fn site_tensors(&self) -> &[Tensor3<T>] {
        &self.tensors
    }
}

#[cfg(test)]
mod tests;
