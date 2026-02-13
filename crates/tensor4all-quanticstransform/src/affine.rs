//! Affine transformation operator: y = A*x + b
//!
//! This implements general affine transformations with rational coefficients.
//! The transformation computes y = A*x + b where A is an M×N rational matrix
//! and b is an M-dimensional rational vector.
//!
//! Based on the algorithm from Quantics.jl/src/affine.jl

use std::collections::HashMap;

use anyhow::Result;
use mdarray::DTensor;
use num_complex::Complex64;
use num_integer::Integer;
use num_rational::Rational64;
use num_traits::One;
use sprs::CsMat;
use tensor4all_simplett::{types::tensor3_zeros, AbstractTensorTrain, Tensor3Ops, TensorTrain};

use crate::common::{
    tensortrain_to_linear_operator_asymmetric, BoundaryCondition, QuanticsOperator,
};

/// Affine transformation parameters.
///
/// Represents the transformation y = A*x + b where:
/// - A is an M×N matrix (stored row-major)
/// - b is an M-dimensional vector
/// - x is an N-dimensional input
/// - y is an M-dimensional output
#[derive(Clone, Debug)]
pub struct AffineParams {
    /// Transformation matrix A (M×N), stored row-major
    pub a: Vec<Rational64>,
    /// Translation vector b (M elements)
    pub b: Vec<Rational64>,
    /// Number of output dimensions (M)
    pub m: usize,
    /// Number of input dimensions (N)
    pub n: usize,
}

impl AffineParams {
    /// Create new affine parameters.
    ///
    /// # Arguments
    /// * `a` - M×N matrix in row-major order
    /// * `b` - M-dimensional translation vector
    /// * `m` - Number of output dimensions
    /// * `n` - Number of input dimensions
    pub fn new(a: Vec<Rational64>, b: Vec<Rational64>, m: usize, n: usize) -> Result<Self> {
        if a.len() != m * n {
            return Err(anyhow::anyhow!(
                "Matrix A has {} elements but expected {}×{}={}",
                a.len(),
                m,
                n,
                m * n
            ));
        }
        if b.len() != m {
            return Err(anyhow::anyhow!(
                "Vector b has {} elements but expected {}",
                b.len(),
                m
            ));
        }
        Ok(Self { a, b, m, n })
    }

    /// Create affine parameters from integer matrix and vector.
    pub fn from_integers(a: Vec<i64>, b: Vec<i64>, m: usize, n: usize) -> Result<Self> {
        let a_rat: Vec<Rational64> = a.into_iter().map(Rational64::from_integer).collect();
        let b_rat: Vec<Rational64> = b.into_iter().map(Rational64::from_integer).collect();
        Self::new(a_rat, b_rat, m, n)
    }

    /// Get element A[i, j] (0-indexed)
    #[allow(dead_code)]
    fn get_a(&self, i: usize, j: usize) -> Rational64 {
        self.a[i * self.n + j]
    }

    /// Convert to integer representation by scaling with LCM of denominators.
    /// Returns (A_int, b_int, scale) where A_int = scale * A and b_int = scale * b.
    fn to_integer_scaled(&self) -> (Vec<i64>, Vec<i64>, i64) {
        // Find LCM of all denominators
        let mut denom_lcm = 1i64;
        for r in &self.a {
            denom_lcm = denom_lcm.lcm(r.denom());
        }
        for r in &self.b {
            denom_lcm = denom_lcm.lcm(r.denom());
        }

        // Scale to integers
        let a_int: Vec<i64> = self
            .a
            .iter()
            .map(|r| (r * denom_lcm).to_integer())
            .collect();
        let b_int: Vec<i64> = self
            .b
            .iter()
            .map(|r| (r * denom_lcm).to_integer())
            .collect();

        (a_int, b_int, denom_lcm)
    }
}

/// Remap site indices of the affine MPO from internal encoding to the convention
/// expected by `tensortrain_to_linear_operator_asymmetric`.
///
/// Internal encoding: `site_idx = y_bits | (x_bits << m)` (y-minor, x-major)
/// Expected encoding: `s = s_out * in_dim + s_in = y_bits * 2^n + x_bits` (x-minor, y-major)
fn remap_affine_site_indices(
    mpo: &TensorTrain<Complex64>,
    m: usize,
    n: usize,
    site_dim: usize,
) -> Result<TensorTrain<Complex64>> {
    let input_dim = 1 << n;

    // Build permutation table: perm[old_idx] = remapped index
    let perm: Vec<usize> = (0..site_dim)
        .map(|old_idx| {
            let y_bits = old_idx & ((1 << m) - 1);
            let x_bits = old_idx >> m;
            y_bits * input_dim + x_bits
        })
        .collect();

    let r = mpo.len();
    let mut new_tensors = Vec::with_capacity(r);

    for i in 0..r {
        let tensor = mpo.site_tensor(i);
        let left_dim = tensor.left_dim();
        let right_dim = tensor.right_dim();

        let mut t = tensor3_zeros(left_dim, site_dim, right_dim);
        for l in 0..left_dim {
            for (old_s, &new_s) in perm.iter().enumerate() {
                for rr in 0..right_dim {
                    let val = *tensor.get3(l, old_s, rr);
                    if val != Complex64::new(0.0, 0.0) {
                        t.set3(l, new_s, rr, val);
                    }
                }
            }
        }
        new_tensors.push(t);
    }

    TensorTrain::new(new_tensors)
        .map_err(|e| anyhow::anyhow!("Failed to create remapped MPO: {}", e))
}

/// Create an affine transformation operator.
///
/// This operator transforms a quantics tensor train representing a function
/// f(x_1, ..., x_N) to g(y_1, ..., y_M) where y = A*x + b.
///
/// # Arguments
/// * `r` - Number of bits per variable
/// * `params` - Affine transformation parameters
/// * `bc` - Boundary conditions for each output variable
///
/// # Returns
/// LinearOperator representing the affine transformation
///
/// # Example
/// ```ignore
/// use tensor4all_quantics_transform::{affine_operator, AffineParams, BoundaryCondition};
/// use num_rational::Rational64;
///
/// // Transform g(x, y) -> g(x + y, x - y) (rotation by 45°, scaled)
/// let a = vec![
///     Rational64::from_integer(1), Rational64::from_integer(1),  // row 0: x + y
///     Rational64::from_integer(1), Rational64::from_integer(-1), // row 1: x - y
/// ];
/// let b = vec![Rational64::from_integer(0), Rational64::from_integer(0)];
/// let params = AffineParams::new(a, b, 2, 2).unwrap();
/// let bc = vec![BoundaryCondition::Periodic; 2];
/// let op = affine_operator(8, &params, &bc).unwrap();
/// ```
pub fn affine_operator(
    r: usize,
    params: &AffineParams,
    bc: &[BoundaryCondition],
) -> Result<QuanticsOperator> {
    if r == 0 {
        return Err(anyhow::anyhow!("Number of bits must be positive"));
    }
    if bc.len() != params.m {
        return Err(anyhow::anyhow!(
            "Boundary conditions length {} doesn't match output dimensions {}",
            bc.len(),
            params.m
        ));
    }

    let mpo = affine_transform_mpo(r, params, bc)?;
    // Site dimensions: M output variables, N input variables
    // Input dimension per site: 2^N (N input bits)
    // Output dimension per site: 2^M (M output bits)
    let m = params.m;
    let n = params.n;
    let input_dim = 1 << n;
    let output_dim = 1 << m;

    // The internal affine MPO uses site encoding: site_idx = y_bits | (x_bits << m)
    // (y-minor, x-major). But tensortrain_to_linear_operator_asymmetric expects
    // s = s_out * in_dim + s_in = y_bits * 2^N + x_bits (x-minor, y-major).
    // We need to remap the site indices.
    let site_dim = input_dim * output_dim;
    let remapped_mpo = remap_affine_site_indices(&mpo, m, n, site_dim)?;

    let input_dims = vec![input_dim; r];
    let output_dims = vec![output_dim; r];
    tensortrain_to_linear_operator_asymmetric(&remapped_mpo, &input_dims, &output_dims)
}

/// Compute the full affine transformation matrix directly (for verification).
///
/// This computes the transformation matrix by directly evaluating y = A*x + b
/// for all possible input values. The result is a sparse boolean matrix.
///
/// # Arguments
/// * `r` - Number of bits per variable
/// * `params` - Affine transformation parameters
/// * `bc` - Boundary conditions for each output variable
///
/// # Returns
/// Sparse matrix of size 2^(R*M) × 2^(R*N) where entry (y_flat, x_flat) = 1
/// if the transformation maps x to y.
///
/// # Note
/// This is only practical for small R due to exponential size.
/// Use for testing/verification only.
pub fn affine_transform_matrix(
    r: usize,
    params: &AffineParams,
    bc: &[BoundaryCondition],
) -> Result<CsMat<f64>> {
    if r == 0 {
        return Err(anyhow::anyhow!("Number of bits must be positive"));
    }
    if bc.len() != params.m {
        return Err(anyhow::anyhow!(
            "Boundary conditions length {} doesn't match output dimensions {}",
            bc.len(),
            params.m
        ));
    }

    let (a_int, b_int, scale) = params.to_integer_scaled();
    let m = params.m;
    let n = params.n;

    let bc_periodic: Vec<bool> = bc
        .iter()
        .map(|b| matches!(b, BoundaryCondition::Periodic))
        .collect();

    let input_size = 1usize << (r * n); // 2^(R*N)
    let output_size = 1usize << (r * m); // 2^(R*M)
    let modulus = 1i64 << r; // 2^R

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();

    let mask = modulus - 1; // 2^R - 1

    // Iterate over all (x, y) pairs, matching Julia's approach.
    // For periodic BC with scale > 1, multiple y values can satisfy
    // scale * y ≡ A*x + b (mod 2^R), so we must check all pairs.
    for x_flat in 0..input_size {
        // Decode x_flat to N-dimensional x vector
        // x_flat = x[0] + x[1]*2^R + x[2]*2^(2R) + ...
        let x: Vec<i64> = (0..n)
            .map(|var| ((x_flat >> (var * r)) & ((1 << r) - 1)) as i64)
            .collect();

        // Compute v = A*x + b (unscaled)
        let mut v: Vec<i64> = vec![0; m];
        for i in 0..m {
            v[i] = b_int[i];
            for j in 0..n {
                v[i] += a_int[i * n + j] * x[j];
            }
        }

        for y_flat in 0..output_size {
            // Decode y_flat to M-dimensional y vector
            let y: Vec<i64> = (0..m)
                .map(|var| ((y_flat >> (var * r)) & ((1 << r) - 1)) as i64)
                .collect();

            // Compute scale * y
            let sy: Vec<i64> = y.iter().map(|&yi| scale * yi).collect();

            // Check equiv(v, s*y, R, boundary) per component
            let equiv = v.iter().zip(sy.iter()).enumerate().all(|(i, (&vi, &syi))| {
                if bc_periodic[i] {
                    // Periodic: v ≡ s*y (mod 2^R)
                    (vi - syi) & mask == 0
                } else {
                    // Open: v == s*y (exact)
                    vi == syi
                }
            });

            if equiv {
                rows.push(y_flat);
                cols.push(x_flat);
                vals.push(1.0);
            }
        }
    }

    // Build sparse matrix in CSR format
    let triplet = sprs::TriMat::from_triplets((output_size, input_size), rows, cols, vals);
    Ok(triplet.to_csr())
}

/// Create the affine transformation MPO as a TensorTrain.
fn affine_transform_mpo(
    r: usize,
    params: &AffineParams,
    bc: &[BoundaryCondition],
) -> Result<TensorTrain<Complex64>> {
    let (a_int, b_int, scale) = params.to_integer_scaled();
    let m = params.m;
    let n = params.n;

    // Convert boundary conditions to weights
    let bc_periodic: Vec<bool> = bc
        .iter()
        .map(|b| matches!(b, BoundaryCondition::Periodic))
        .collect();

    // Compute core tensors
    let tensors = affine_transform_tensors(r, &a_int, &b_int, scale, m, n, &bc_periodic)?;

    TensorTrain::new(tensors)
        .map_err(|e| anyhow::anyhow!("Failed to create affine transform MPO: {}", e))
}

/// Create unfused affine transformation tensors.
///
/// Returns a vector of R tensors, where each tensor has shape:
/// `[left_bond, 2, 2, ..., 2, right_bond]` with M+N physical indices of dimension 2.
///
/// The physical index order matches Quantics.jl:
/// `(y[1], y[2], ..., y[M], x[1], x[2], ..., x[N])`
/// where y are output variables and x are input variables.
///
/// # Arguments
/// * `r` - Number of bits per variable (number of sites)
/// * `params` - Affine transformation parameters
/// * `bc` - Boundary conditions for each output variable
///
/// # Returns
/// Vector of R tensors with unfused physical indices.
///
/// # Example
/// ```ignore
/// use tensor4all_quantics_transform::{affine_transform_tensors_unfused, AffineParams, BoundaryCondition};
///
/// let params = AffineParams::from_integers(vec![1, 0, 1, 1], vec![0, 0], 2, 2).unwrap();
/// let bc = vec![BoundaryCondition::Periodic; 2];
/// let tensors = affine_transform_tensors_unfused(4, &params, &bc).unwrap();
/// // Each tensor has shape [left, 2, 2, 2, 2, right] for M=2, N=2
/// ```
pub fn affine_transform_tensors_unfused(
    r: usize,
    params: &AffineParams,
    bc: &[BoundaryCondition],
) -> Result<Vec<DTensor<Complex64, 3>>> {
    if r == 0 {
        return Err(anyhow::anyhow!("Number of bits must be positive"));
    }
    if bc.len() != params.m {
        return Err(anyhow::anyhow!(
            "Boundary conditions length {} doesn't match output dimensions {}",
            bc.len(),
            params.m
        ));
    }

    let (a_int, b_int, scale) = params.to_integer_scaled();
    let m = params.m;
    let n = params.n;

    // Convert boundary conditions to weights
    let bc_periodic: Vec<bool> = bc
        .iter()
        .map(|b| matches!(b, BoundaryCondition::Periodic))
        .collect();

    // Compute fused tensors first
    let fused_tensors = affine_transform_tensors(r, &a_int, &b_int, scale, m, n, &bc_periodic)?;

    // Convert fused tensors to unfused format
    // Fused: [left, fused_site, right] where fused_site = 2^(M+N)
    // Unfused: [left, 2, 2, ..., 2, right] with M+N dimensions of size 2
    //
    // Fused index encoding: site_idx = y_bits | (x_bits << M)
    // where y_bits = y[0] + 2*y[1] + ... + 2^(M-1)*y[M-1]
    // and   x_bits = x[0] + 2*x[1] + ... + 2^(N-1)*x[N-1]
    //
    // Quantics.jl order: (y[0], y[1], ..., y[M-1], x[0], x[1], ..., x[N-1])
    // For DTensor with row-major (C order), rightmost index varies fastest.
    // We want the order to match Quantics.jl, so:
    // unfused[left, y0, y1, ..., yM-1, x0, x1, ..., xN-1, right]

    let mut unfused_tensors = Vec::with_capacity(r);
    let site_dim = 1 << (m + n);

    for tensor in fused_tensors.iter() {
        let left_dim = tensor.left_dim();
        let right_dim = tensor.right_dim();

        // Create unfused tensor
        // Shape: [left_dim, 2^(M+N), right_dim] but we keep it as 3D for now
        // The reshape to (M+N+2)-dimensional tensor will be done by the caller if needed
        // For now, we provide a 3D tensor where the middle dimension is the fused site
        // and document how to unfuse it.
        //
        // Actually, let's return it properly unfused using a flat storage with
        // the correct index order for reshape.
        //
        // Total size: left_dim * 2^(M+N) * right_dim
        // Shape for unfused: [left_dim, 2, 2, ..., 2, right_dim]
        //
        // Index mapping from fused to unfused:
        // fused site_idx -> (y0, y1, ..., yM-1, x0, x1, ..., xN-1)
        // site_idx = y0 + 2*y1 + ... + 2^(M-1)*yM-1 + 2^M * (x0 + 2*x1 + ...)

        // For Quantics.jl compatibility, we need to reorder the data.
        // Julia uses column-major (Fortran order), Rust/C uses row-major.
        // In Julia: tensor[link_in, link_out, y1, y2, ..., yM, x1, x2, ..., xN]
        // ITensor reshapes this appropriately.
        //
        // For Rust with row-major:
        // We'll create a tensor where the physical indices are in the order
        // (y0, y1, ..., yM-1, x0, x1, ..., xN-1) matching Quantics.jl

        let mut unfused_data = vec![Complex64::new(0.0, 0.0); left_dim * site_dim * right_dim];

        for l in 0..left_dim {
            for fused_idx in 0..site_dim {
                for rr in 0..right_dim {
                    let val = tensor.get3(l, fused_idx, rr);
                    if val.norm() > 0.0 {
                        // The fused index encodes: site_idx = y_bits | (x_bits << M)
                        // This matches Quantics.jl's ordering (y variables first, then x)
                        // so we can use fused_idx directly.
                        //
                        // The caller can reshape [left, site_dim, right] to
                        // [left, 2, 2, ..., 2, right] with M+N dimensions of size 2,
                        // where indices are in order (y[0], y[1], ..., y[M-1], x[0], ..., x[N-1])
                        let flat_idx = l * site_dim * right_dim + fused_idx * right_dim + rr;
                        unfused_data[flat_idx] = *val;
                    }
                }
            }
        }

        // Create DTensor with shape [left_dim, site_dim, right_dim]
        // The caller can reshape this to [left_dim, 2, 2, ..., 2, right_dim]
        // with the understanding that the indices are ordered as (y0, y1, ..., x0, x1, ...)
        let unfused_tensor =
            DTensor::<Complex64, 3>::from_fn([left_dim, site_dim, right_dim], |idx| {
                let l = idx[0];
                let s = idx[1];
                let r = idx[2];
                unfused_data[l * site_dim * right_dim + s * right_dim + r]
            });

        unfused_tensors.push(unfused_tensor);
    }

    Ok(unfused_tensors)
}

/// Information about the unfused tensor structure.
///
/// This helper provides metadata for reshaping the unfused tensors.
#[derive(Clone, Debug)]
pub struct UnfusedTensorInfo {
    /// Number of output variables (M)
    pub m: usize,
    /// Number of input variables (N)
    pub n: usize,
    /// Total physical dimensions per site (M + N)
    pub num_physical_dims: usize,
    /// Dimension of each physical index (always 2)
    pub physical_dim: usize,
}

impl UnfusedTensorInfo {
    /// Create info for the given affine parameters.
    pub fn new(params: &AffineParams) -> Self {
        Self {
            m: params.m,
            n: params.n,
            num_physical_dims: params.m + params.n,
            physical_dim: 2,
        }
    }

    /// Get the shape for a fully unfused tensor at a given site.
    ///
    /// Returns `[left_bond, 2, 2, ..., 2, right_bond]` where there are M+N 2s.
    pub fn unfused_shape(&self, left_bond: usize, right_bond: usize) -> Vec<usize> {
        let mut shape = Vec::with_capacity(2 + self.num_physical_dims);
        shape.push(left_bond);
        shape.extend(std::iter::repeat_n(2, self.num_physical_dims));
        shape.push(right_bond);
        shape
    }

    /// Decode a fused site index to individual variable bits.
    ///
    /// Returns `(y_bits, x_bits)` where:
    /// - `y_bits[i]` is the bit for output variable i
    /// - `x_bits[j]` is the bit for input variable j
    pub fn decode_fused_index(&self, fused_idx: usize) -> (Vec<usize>, Vec<usize>) {
        let y_combined = fused_idx & ((1 << self.m) - 1);
        let x_combined = fused_idx >> self.m;

        let y_bits: Vec<usize> = (0..self.m).map(|i| (y_combined >> i) & 1).collect();
        let x_bits: Vec<usize> = (0..self.n).map(|j| (x_combined >> j) & 1).collect();

        (y_bits, x_bits)
    }

    /// Encode individual variable bits to a fused site index.
    ///
    /// # Arguments
    /// * `y_bits` - Bits for output variables (length M)
    /// * `x_bits` - Bits for input variables (length N)
    pub fn encode_fused_index(&self, y_bits: &[usize], x_bits: &[usize]) -> usize {
        let y_combined: usize = y_bits.iter().enumerate().map(|(i, &b)| b << i).sum();
        let x_combined: usize = x_bits.iter().enumerate().map(|(j, &b)| b << j).sum();
        y_combined | (x_combined << self.m)
    }
}

/// Compute the core tensors for the affine transformation.
///
/// This implements the algorithm from Quantics.jl that handles:
/// - Carry propagation for multi-bit arithmetic
/// - Scaling factor s from rational to integer conversion
///
/// Uses big-endian convention: site 0 = MSB, site R-1 = LSB.
///
/// Carry propagation direction (matching shift.rs):
/// - Arithmetic carry flows LSB → MSB (physical fact)
/// - In big-endian: site R-1 → site 0 (right → left)
/// - Tensor structure: t[left, site, right] where left=carry_out (going left), right=carry_in (from right)
/// - Site 0 (MSB): BC applied on left, receives carry from right → shape (1, site_dim, num_carries)
/// - Site R-1 (LSB): initial carry=0, sends carry to left → shape (num_carries, site_dim, 1)
/// - Middle sites: shape (num_carries, site_dim, num_carries)
fn affine_transform_tensors(
    r: usize,
    a_int: &[i64],
    b_int: &[i64],
    scale: i64,
    m: usize,
    n: usize,
    bc_periodic: &[bool],
) -> Result<Vec<tensor4all_simplett::Tensor3<Complex64>>> {
    let site_dim = 1 << (m + n); // 2^(M+N) for fused representation

    // Track sign separately and work with absolute value
    // so that right-shifting always terminates (Julia PR #45 approach)
    let bsign: Vec<i64> = b_int.iter().map(|&b| if b >= 0 { 1 } else { -1 }).collect();
    let mut b_work: Vec<i64> = b_int.iter().map(|&b| b.abs()).collect();

    // Process from LSB (site R-1) to MSB (site 0)
    let mut carries: Vec<Vec<i64>> = vec![vec![0i64; m]];
    let mut core_data_list: Vec<AffineCoreData> = Vec::with_capacity(r);

    for _site in (0..r).rev() {
        // Extract current bit: (b_work & 1) * bsign
        let b_curr: Vec<i64> = b_work
            .iter()
            .zip(bsign.iter())
            .map(|(&b, &s)| (b & 1) * s)
            .collect();

        let core_data = affine_transform_core(a_int, &b_curr, scale, m, n, &carries, true)?;
        carries = core_data.carries_out.clone();
        core_data_list.push(core_data);

        // Shift right
        b_work.iter_mut().for_each(|b| *b >>= 1);
    }

    // core_data_list is now in order: [site R-1, site R-2, ..., site 0]

    // Extension loop: handle remaining bits of b for Open BC
    // When abs(b) >= 2^R, high bits of b contribute to carries that affect validity.
    // Extension tensors have site_dim=1 (activebit=false: only x=0, y=0).
    // We fold them into the MSB tensor as a "cap matrix" (Julia approach).
    let cap_matrix: Option<Vec<f64>> = if !bc_periodic.iter().all(|&p| p)
        && b_work.iter().any(|&b| b > 0)
    {
        let mut ext_data_list: Vec<AffineCoreData> = Vec::new();
        while b_work.iter().any(|&b| b > 0) {
            let b_curr: Vec<i64> = b_work
                .iter()
                .zip(bsign.iter())
                .map(|(&b, &s)| (b & 1) * s)
                .collect();

            let core_data = affine_transform_core(a_int, &b_curr, scale, m, n, &carries, false)?;
            carries = core_data.carries_out.clone();
            ext_data_list.push(core_data);

            b_work.iter_mut().for_each(|b| *b >>= 1);
        }

        // Build cap matrix by contracting extension tensors with BC weights.
        // Extension tensors have site_dim=1, so they are carry transition matrices:
        //   ext_matrix[cout_idx, cin_idx] = core_data.tensor[[cout_idx, cin_idx, 0]]
        //
        // Process: outermost (last computed) gets BC weights applied,
        // then multiply inward toward the main tensor chain.

        // Start with BC weights on the final carries
        let bc_weights: Vec<f64> = carries
            .iter()
            .map(|c| {
                if c.iter().all(|&ci| ci == 0) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();

        // Contract extension tensors from outermost to innermost
        // ext_data_list is [innermost, ..., outermost] (order of computation)
        // We process from outermost to innermost
        let mut current_weights = bc_weights;
        for ext_data in ext_data_list.iter().rev() {
            let (_, num_cin, _) = *ext_data.tensor.shape();
            let mut new_weights = vec![0.0; num_cin];
            for (cin_idx, nw) in new_weights.iter_mut().enumerate() {
                for (cout_idx, &w) in current_weights.iter().enumerate() {
                    if w != 0.0 && ext_data.tensor[[cout_idx, cin_idx, 0]] {
                        *nw += w;
                    }
                }
            }
            current_weights = new_weights;
        }

        // current_weights now maps: MSB carry_out index -> effective BC weight
        Some(current_weights)
    } else {
        None
    };

    // Build tensors in the same order, then reverse to get [site 0, site 1, ..., site R-1]
    let mut tensors = Vec::with_capacity(r);

    // Helper: compute BC weight for a carry-out index
    let compute_bc_weight = |cout_idx: usize, core_data: &AffineCoreData| -> Complex64 {
        if bc_periodic.iter().all(|&p| p) {
            Complex64::one()
        } else if let Some(ref cap) = cap_matrix {
            // Extension loop was used: weight comes from cap matrix
            Complex64::new(cap[cout_idx], 0.0)
        } else {
            // No extension: weight is 1 if carry is zero, 0 otherwise
            let carry = &core_data.carries_out[cout_idx];
            if carry.iter().all(|&c| c == 0) {
                Complex64::one()
            } else {
                Complex64::new(0.0, 0.0)
            }
        }
    };

    for (idx, core_data) in core_data_list.iter().enumerate() {
        // idx=0 corresponds to site R-1 (LSB), idx=R-1 corresponds to site 0 (MSB)
        let actual_site = r - 1 - idx;
        let num_carry_out = core_data.carries_out.len();
        let (_, num_carry_in, _) = *core_data.tensor.shape();

        // Tensor shape follows shift.rs pattern:
        // t[left, site, right] where left=carry_out (going left), right=carry_in (from right)
        //
        // - Site 0 (MSB): left_dim=1 (BC applied), right_dim=num_carry (receives from right)
        // - Site R-1 (LSB): left_dim=num_carry (sends to left), right_dim=1 (initial carry=0)
        // - Middle: left_dim=num_carry, right_dim=num_carry
        let is_msb = actual_site == 0;
        let is_lsb = actual_site == r - 1;

        let left_dim = if is_msb { 1 } else { num_carry_out };
        let right_dim = if is_lsb { 1 } else { num_carry_in };

        let mut t: tensor4all_simplett::Tensor3<Complex64> =
            tensor3_zeros(left_dim, site_dim, right_dim);

        if is_lsb && is_msb {
            // R==1: single site case
            for cout_idx in 0..num_carry_out {
                let bc_weight = compute_bc_weight(cout_idx, core_data);

                for site_idx in 0..site_dim {
                    if core_data.tensor[[cout_idx, 0, site_idx]] {
                        let old = t.get3(0, site_idx, 0);
                        t.set3(0, site_idx, 0, *old + bc_weight);
                    }
                }
            }
        } else if is_lsb {
            // LSB (site R-1): initial carry_in=0, send carry_out to left
            // Shape (num_carry_out, site_dim, 1)
            // core_data.tensor[carry_out_idx, carry_in_idx, site_idx]
            // Only carry_in_idx=0 matters (initial carry is the first entry: zero vector)
            for cout_idx in 0..num_carry_out {
                for site_idx in 0..site_dim {
                    if core_data.tensor[[cout_idx, 0, site_idx]] {
                        t.set3(cout_idx, site_idx, 0, Complex64::one());
                    }
                }
            }
        } else if is_msb {
            // MSB (site 0): apply BC on carry_out, receive carry from right
            for cout_idx in 0..num_carry_out {
                let bc_weight = compute_bc_weight(cout_idx, core_data);

                for cin_idx in 0..num_carry_in {
                    for site_idx in 0..site_dim {
                        if core_data.tensor[[cout_idx, cin_idx, site_idx]] {
                            let old = t.get3(0, site_idx, cin_idx);
                            t.set3(0, site_idx, cin_idx, *old + bc_weight);
                        }
                    }
                }
            }
        } else {
            // Middle tensors: receive carry from right, send carry to left
            // Shape (num_carry_out, site_dim, num_carry_in)
            for cout_idx in 0..num_carry_out {
                for cin_idx in 0..num_carry_in {
                    for site_idx in 0..site_dim {
                        if core_data.tensor[[cout_idx, cin_idx, site_idx]] {
                            t.set3(cout_idx, site_idx, cin_idx, Complex64::one());
                        }
                    }
                }
            }
        }

        tensors.push(t);
    }

    // tensors is in order [site R-1, ..., site 0], reverse to get [site 0, ..., site R-1]
    tensors.reverse();

    Ok(tensors)
}

/// Core tensor data for affine transformation.
///
/// Shape: (num_carry_out, num_carry_in, site_dim)
/// where site_dim = 2^(M+N)
struct AffineCoreData {
    /// Possible outgoing carry vectors
    carries_out: Vec<Vec<i64>>,
    /// Tensor data: tensor[carry_out_idx, carry_in_idx, site_idx]
    tensor: DTensor<bool, 3>,
}

/// Compute a single core tensor for the affine transformation.
///
/// The core tensor encodes: 2 * carry_out = A * x + b_curr - scale * y + carry_in
///
/// Returns AffineCoreData containing:
/// - carries_out: list of possible outgoing carry vectors
/// - tensor: shape (num_carry_out, num_carry_in, site_dim)
fn affine_transform_core(
    a_int: &[i64],
    b_curr: &[i64],
    scale: i64,
    m: usize,
    n: usize,
    carries_in: &[Vec<i64>],
    activebit: bool,
) -> Result<AffineCoreData> {
    let mut carry_out_map: HashMap<Vec<i64>, DTensor<bool, 2>> = HashMap::new();
    let x_range = if activebit { 1 << n } else { 1 };
    let y_range = if activebit { 1 << m } else { 1 };
    let site_dim = x_range * y_range;
    let num_carry_in = carries_in.len();

    // Iterate over all input carries
    for (c_idx, carry_in) in carries_in.iter().enumerate() {
        // Iterate over all possible x values (N bits)
        for x_bits in 0..x_range {
            let x: Vec<i64> = (0..n).map(|j| ((x_bits >> j) & 1) as i64).collect();

            // Compute z = A*x + b + carry_in
            let mut z: Vec<i64> = vec![0; m];
            for i in 0..m {
                z[i] = carry_in[i] + b_curr[i];
                for j in 0..n {
                    z[i] += a_int[i * n + j] * x[j];
                }
            }

            if scale % 2 == 1 {
                // Scale is odd: unique y that satisfies condition
                let y: Vec<i64> = z.iter().map(|&zi| zi & 1).collect();

                // When bits are inactive, y must be zero (Julia PR #45 fix)
                if !activebit && y.iter().any(|&yi| yi != 0) {
                    continue;
                }

                let y_bits: usize = y
                    .iter()
                    .enumerate()
                    .map(|(i, &yi)| (yi as usize) << i)
                    .sum();

                // Compute carry_out = (z - scale * y) / 2
                let carry_out: Vec<i64> = z
                    .iter()
                    .zip(y.iter())
                    .map(|(&zi, &yi)| (zi - scale * yi) >> 1)
                    .collect();

                // Site index: y bits in lower positions, x bits in upper positions
                let site_idx = y_bits | (x_bits << m);

                let entry = carry_out_map.entry(carry_out).or_insert_with(|| {
                    DTensor::<bool, 2>::from_elem([num_carry_in, site_dim], false)
                });
                entry[[c_idx, site_idx]] = true;
            } else {
                // Scale is even: z must be even for valid y
                if z.iter().any(|&zi| zi % 2 != 0) {
                    continue;
                }

                // y can be any value
                for y_bits in 0..y_range {
                    let y: Vec<i64> = (0..m).map(|i| ((y_bits >> i) & 1) as i64).collect();

                    // Compute carry_out = (z - scale * y) / 2
                    let carry_out: Vec<i64> = z
                        .iter()
                        .zip(y.iter())
                        .map(|(&zi, &yi)| (zi - scale * yi) >> 1)
                        .collect();

                    let site_idx = y_bits | (x_bits << m);

                    let entry = carry_out_map.entry(carry_out).or_insert_with(|| {
                        DTensor::<bool, 2>::from_elem([num_carry_in, site_dim], false)
                    });
                    entry[[c_idx, site_idx]] = true;
                }
            }
        }
    }

    // Convert to sorted vectors for deterministic ordering
    let mut carries_out: Vec<Vec<i64>> = carry_out_map.keys().cloned().collect();
    carries_out.sort();

    let num_carry_out = carries_out.len();

    // Build 3D tensor: (num_carry_out, num_carry_in, site_dim)
    let mut tensor = DTensor::<bool, 3>::from_elem([num_carry_out, num_carry_in, site_dim], false);
    for (cout_idx, carry) in carries_out.iter().enumerate() {
        let data_2d = &carry_out_map[carry];
        for cin_idx in 0..num_carry_in {
            for site_idx in 0..site_dim {
                tensor[[cout_idx, cin_idx, site_idx]] = data_2d[[cin_idx, site_idx]];
            }
        }
    }

    Ok(AffineCoreData {
        carries_out,
        tensor,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_affine_params_new() {
        let a = vec![
            Rational64::from_integer(1),
            Rational64::from_integer(0),
            Rational64::from_integer(0),
            Rational64::from_integer(1),
        ];
        let b = vec![Rational64::from_integer(0), Rational64::from_integer(0)];
        let params = AffineParams::new(a, b, 2, 2);
        assert!(params.is_ok());
    }

    #[test]
    fn test_affine_params_from_integers() {
        let a = vec![1, 0, 0, 1];
        let b = vec![0, 0];
        let params = AffineParams::from_integers(a, b, 2, 2);
        assert!(params.is_ok());
    }

    #[test]
    fn test_affine_params_to_integer_scaled() {
        // Test with rational coefficients
        let a = vec![
            Rational64::new(1, 2), // 1/2
            Rational64::new(1, 3), // 1/3
        ];
        let b = vec![Rational64::new(1, 6)]; // 1/6
        let params = AffineParams::new(a, b, 1, 2).unwrap();

        let (a_int, b_int, scale) = params.to_integer_scaled();

        // LCM of denominators (2, 3, 6) = 6
        assert_eq!(scale, 6);
        assert_eq!(a_int, vec![3, 2]); // [1/2 * 6, 1/3 * 6]
        assert_eq!(b_int, vec![1]); // [1/6 * 6]
    }

    #[test]
    fn test_affine_transform_identity() {
        // Identity transformation: y = x
        let a = vec![1i64];
        let b = vec![0i64];
        let params = AffineParams::from_integers(a, b, 1, 1).unwrap();
        let bc = vec![BoundaryCondition::Periodic];

        let result = affine_operator(4, &params, &bc);
        assert!(result.is_ok());
    }

    #[test]
    fn test_affine_operator_creation() {
        // Simple 2D transformation
        let a = vec![1i64, 1, 1, -1]; // [[1, 1], [1, -1]]
        let b = vec![0i64, 0];
        let params = AffineParams::from_integers(a, b, 2, 2).unwrap();
        let bc = vec![BoundaryCondition::Periodic; 2];

        let result = affine_operator(4, &params, &bc);
        assert!(result.is_ok());
    }

    #[test]
    fn test_affine_error_zero_bits() {
        let a = vec![1i64];
        let b = vec![0i64];
        let params = AffineParams::from_integers(a, b, 1, 1).unwrap();
        let bc = vec![BoundaryCondition::Periodic];

        let result = affine_operator(0, &params, &bc);
        assert!(result.is_err());
    }

    #[test]
    fn test_affine_error_bc_mismatch() {
        let a = vec![1i64, 0, 0, 1];
        let b = vec![0i64, 0];
        let params = AffineParams::from_integers(a, b, 2, 2).unwrap();
        let bc = vec![BoundaryCondition::Periodic]; // Only 1 BC but M=2

        let result = affine_operator(4, &params, &bc);
        assert!(result.is_err());
    }

    #[test]
    fn test_affine_params_dimension_error() {
        // a.len() != m * n
        let a = vec![Rational64::from_integer(1), Rational64::from_integer(0)]; // 2 elements
        let b = vec![Rational64::from_integer(0)];
        let params = AffineParams::new(a, b, 2, 2); // expects 4 elements
        assert!(params.is_err());

        // b.len() != m
        let a = vec![
            Rational64::from_integer(1),
            Rational64::from_integer(0),
            Rational64::from_integer(0),
            Rational64::from_integer(1),
        ];
        let b = vec![Rational64::from_integer(0)]; // 1 element but m=2
        let params = AffineParams::new(a, b, 2, 2);
        assert!(params.is_err());
    }

    #[test]
    fn test_affine_with_rational_coefficients() {
        // y = (1/2)*x
        let a = vec![Rational64::new(1, 2)];
        let b = vec![Rational64::from_integer(0)];
        let params = AffineParams::new(a, b, 1, 1).unwrap();
        let bc = vec![BoundaryCondition::Periodic];

        let result = affine_operator(4, &params, &bc);
        assert!(result.is_ok());
    }

    #[test]
    fn test_affine_shift_only() {
        // y = x + 3
        let a = vec![1i64];
        let b = vec![3i64];
        let params = AffineParams::from_integers(a, b, 1, 1).unwrap();
        let bc = vec![BoundaryCondition::Periodic];

        let result = affine_operator(4, &params, &bc);
        assert!(result.is_ok());
    }

    #[test]
    fn test_affine_scale_by_two() {
        // y = 2*x
        let a = vec![2i64];
        let b = vec![0i64];
        let params = AffineParams::from_integers(a, b, 1, 1).unwrap();
        let bc = vec![BoundaryCondition::Periodic];

        let result = affine_operator(4, &params, &bc);
        assert!(result.is_ok());
    }

    #[test]
    fn test_affine_asymmetric_dimensions() {
        // M=1, N=2: y = x1 + x2 (sum of two inputs to one output)
        let a = vec![1i64, 1]; // 1×2 matrix
        let b = vec![0i64];
        let params = AffineParams::from_integers(a, b, 1, 2).unwrap();
        let bc = vec![BoundaryCondition::Periodic];

        let result = affine_operator(4, &params, &bc);
        assert!(result.is_ok());
    }

    #[test]
    fn test_affine_open_boundary() {
        // Identity with open boundary
        let a = vec![1i64];
        let b = vec![0i64];
        let params = AffineParams::from_integers(a, b, 1, 1).unwrap();
        let bc = vec![BoundaryCondition::Open];

        let result = affine_operator(4, &params, &bc);
        assert!(result.is_ok());
    }

    #[test]
    fn test_affine_negation() {
        // y = -x
        let a = vec![-1i64];
        let b = vec![0i64];
        let params = AffineParams::from_integers(a, b, 1, 1).unwrap();
        let bc = vec![BoundaryCondition::Periodic];

        let result = affine_operator(4, &params, &bc);
        assert!(result.is_ok());
    }

    #[test]
    fn test_affine_mpo_structure() {
        // Verify MPO tensor structure for identity transform
        let a = vec![1i64];
        let b = vec![0i64];
        let params = AffineParams::from_integers(a, b, 1, 1).unwrap();
        let bc = vec![BoundaryCondition::Periodic];

        let op = affine_operator(4, &params, &bc).unwrap();
        // Check that the operator was created successfully
        // (More detailed structure tests would require accessing internal TreeTN)
        let _ = op;
    }

    #[test]
    fn test_affine_larger_bits() {
        // Test with more bits
        let a = vec![1i64];
        let b = vec![0i64];
        let params = AffineParams::from_integers(a, b, 1, 1).unwrap();
        let bc = vec![BoundaryCondition::Periodic];

        let result = affine_operator(8, &params, &bc);
        assert!(result.is_ok());

        let result = affine_operator(16, &params, &bc);
        assert!(result.is_ok());
    }

    // ========== Matrix verification tests ==========

    #[test]
    fn test_affine_matrix_identity() {
        // Identity transformation: y = x
        let r = 3;
        let params = AffineParams::from_integers(vec![1], vec![0], 1, 1).unwrap();
        let bc = vec![BoundaryCondition::Periodic];

        let matrix = affine_transform_matrix(r, &params, &bc).unwrap();

        // Should be identity matrix of size 2^R × 2^R
        let size = 1 << r;
        assert_eq!(matrix.rows(), size);
        assert_eq!(matrix.cols(), size);
        assert_eq!(matrix.nnz(), size); // Identity has exactly N non-zeros

        // Check that it's identity
        for i in 0..size {
            assert_eq!(*matrix.get(i, i).unwrap_or(&0.0), 1.0);
        }
    }

    #[test]
    fn test_affine_matrix_shift() {
        // Shift transformation: y = x + 3 (mod 2^R)
        let r = 3;
        let params = AffineParams::from_integers(vec![1], vec![3], 1, 1).unwrap();
        let bc = vec![BoundaryCondition::Periodic];

        let matrix = affine_transform_matrix(r, &params, &bc).unwrap();

        let size = 1 << r; // 8
        assert_eq!(matrix.nnz(), size); // Permutation has exactly N non-zeros

        // Check specific mappings: y = (x + 3) mod 8
        // x=0 -> y=3, x=1 -> y=4, ..., x=5 -> y=0, x=6 -> y=1, x=7 -> y=2
        for x in 0..size {
            let y = (x + 3) % size;
            assert_eq!(*matrix.get(y, x).unwrap_or(&0.0), 1.0);
        }
    }

    #[test]
    fn test_affine_matrix_scale_by_two() {
        // Scale: y = 2*x (mod 2^R)
        let r = 3;
        let params = AffineParams::from_integers(vec![2], vec![0], 1, 1).unwrap();
        let bc = vec![BoundaryCondition::Periodic];

        let matrix = affine_transform_matrix(r, &params, &bc).unwrap();

        let size = 1 << r; // 8
        assert_eq!(matrix.nnz(), size);

        // Check: y = 2*x mod 8
        for x in 0..size {
            let y = (2 * x) % size;
            assert_eq!(*matrix.get(y, x).unwrap_or(&0.0), 1.0);
        }
    }

    #[test]
    fn test_affine_matrix_sum_2d() {
        // y = x1 + x2 (M=1, N=2)
        let r = 2;
        let params = AffineParams::from_integers(vec![1, 1], vec![0], 1, 2).unwrap();
        let bc = vec![BoundaryCondition::Periodic];

        let matrix = affine_transform_matrix(r, &params, &bc).unwrap();

        let input_size = 1 << (r * 2); // 2^(2*2) = 16
        let output_size = 1 << r; // 2^2 = 4

        assert_eq!(matrix.rows(), output_size);
        assert_eq!(matrix.cols(), input_size);

        // Check specific cases
        // x_flat = x1 + x2 * 2^R
        // x1=1, x2=2: x_flat = 1 + 2*4 = 9, y = (1+2) mod 4 = 3
        assert_eq!(*matrix.get(3, 9).unwrap_or(&0.0), 1.0);
        // x1=3, x2=3: x_flat = 3 + 3*4 = 15, y = (3+3) mod 4 = 2
        assert_eq!(*matrix.get(2, 15).unwrap_or(&0.0), 1.0);
    }

    #[test]
    fn test_affine_matrix_2d_identity() {
        // 2D identity: y = [x1, x2] (M=2, N=2)
        let r = 2;
        let params = AffineParams::from_integers(vec![1, 0, 0, 1], vec![0, 0], 2, 2).unwrap();
        let bc = vec![BoundaryCondition::Periodic; 2];

        let matrix = affine_transform_matrix(r, &params, &bc).unwrap();

        let size = 1 << (r * 2); // 2^(2*2) = 16
        assert_eq!(matrix.rows(), size);
        assert_eq!(matrix.cols(), size);
        assert_eq!(matrix.nnz(), size); // Identity

        // Check it's identity
        for i in 0..size {
            assert_eq!(*matrix.get(i, i).unwrap_or(&0.0), 1.0);
        }
    }

    #[test]
    fn test_affine_matrix_2d_swap() {
        // Swap: y1 = x2, y2 = x1 (M=2, N=2)
        let r = 2;
        let params = AffineParams::from_integers(vec![0, 1, 1, 0], vec![0, 0], 2, 2).unwrap();
        let bc = vec![BoundaryCondition::Periodic; 2];

        let matrix = affine_transform_matrix(r, &params, &bc).unwrap();

        let size = 1 << (r * 2); // 16
        assert_eq!(matrix.nnz(), size); // Permutation

        // x_flat = x1 + x2 * 2^R, y_flat = y1 + y2 * 2^R
        // Swap: y1 = x2, y2 = x1, so y_flat = x2 + x1 * 2^R
        let modulus = 1 << r;
        for x1 in 0..modulus {
            for x2 in 0..modulus {
                let x_flat = x1 + x2 * modulus;
                let y_flat = x2 + x1 * modulus; // swapped
                assert_eq!(*matrix.get(y_flat, x_flat).unwrap_or(&0.0), 1.0);
            }
        }
    }

    #[test]
    fn test_affine_matrix_half_scale() {
        // y = x/2, scale=2, R=3, Periodic BC
        // Condition: 2*y ≡ x (mod 2^R=8), so each even x has 2 solutions
        let r = 3;
        let a = vec![Rational64::new(1, 2)];
        let b = vec![Rational64::from_integer(0)];
        let params = AffineParams::new(a, b, 1, 1).unwrap();
        let bc = vec![BoundaryCondition::Periodic];

        let matrix = affine_transform_matrix(r, &params, &bc).unwrap();

        // x=0: y∈{0,4}, x=2: y∈{1,5}, x=4: y∈{2,6}, x=6: y∈{3,7}
        assert_eq!(matrix.nnz(), 8);

        assert_eq!(*matrix.get(0, 0).unwrap_or(&0.0), 1.0);
        assert_eq!(*matrix.get(4, 0).unwrap_or(&0.0), 1.0);
        assert_eq!(*matrix.get(1, 2).unwrap_or(&0.0), 1.0);
        assert_eq!(*matrix.get(5, 2).unwrap_or(&0.0), 1.0);
        assert_eq!(*matrix.get(2, 4).unwrap_or(&0.0), 1.0);
        assert_eq!(*matrix.get(6, 4).unwrap_or(&0.0), 1.0);
        assert_eq!(*matrix.get(3, 6).unwrap_or(&0.0), 1.0);
        assert_eq!(*matrix.get(7, 6).unwrap_or(&0.0), 1.0);

        assert_affine_mpo_matches_matrix(r, &params, &bc);
    }

    // ========== MPO vs Matrix comparison tests (from Quantics.jl) ==========

    use tensor4all_simplett::{AbstractTensorTrain, Tensor3Ops};

    /// Convert MPO (TensorTrain) to dense matrix for comparison.
    ///
    /// The MPO has R sites. Each site has physical dimension 2^(M+N).
    /// The site index encodes: site_idx = y_bits | (x_bits << M)
    /// where y_bits and x_bits are the bits of y and x variables at that site.
    ///
    /// Flat index convention (matching affine_transform_matrix):
    /// - x_flat = x[0] + x[1]*2^R + x[2]*2^(2R) + ...
    /// - y_flat = y[0] + y[1]*2^R + y[2]*2^(2R) + ...
    ///
    /// Big-endian convention: site 0 = MSB, site R-1 = LSB.
    #[allow(clippy::needless_range_loop)]
    fn mpo_to_dense_matrix(
        mpo: &TensorTrain<Complex64>,
        m: usize,
        n: usize,
        r: usize,
    ) -> Vec<Vec<Complex64>> {
        let output_size = 1 << (m * r);
        let input_size = 1 << (n * r);
        let mut matrix = vec![vec![Complex64::new(0.0, 0.0); input_size]; output_size];

        let tensors = mpo.site_tensors();

        // For each input/output combination, compute the matrix element
        for y_flat in 0..output_size {
            for x_flat in 0..input_size {
                // Contract the MPO for this (y_flat, x_flat) pair
                // Start with a row vector of size 1 (left boundary)
                let mut left_vec = vec![Complex64::one()];

                for site in 0..r {
                    // Big-endian: site 0 = MSB, so bit_pos = R-1-site
                    let bit_pos = r - 1 - site;

                    // Extract bits for this site from each variable
                    // y_flat = y[0] + y[1]*2^R + ... where each y[i] is R bits
                    // For variable i, extract bit at position bit_pos
                    let mut y_bits = 0usize;
                    for var in 0..m {
                        // y[var] occupies bits [var*R, (var+1)*R) in y_flat
                        let y_var = (y_flat >> (var * r)) & ((1 << r) - 1);
                        let bit = (y_var >> bit_pos) & 1;
                        y_bits |= bit << var;
                    }

                    let mut x_bits = 0usize;
                    for var in 0..n {
                        let x_var = (x_flat >> (var * r)) & ((1 << r) - 1);
                        let bit = (x_var >> bit_pos) & 1;
                        x_bits |= bit << var;
                    }

                    // Site index: y_bits in lower M bits, x_bits in upper N bits
                    let site_idx = y_bits | (x_bits << m);

                    let tensor = &tensors[site];
                    let left_dim = tensor.left_dim();
                    let right_dim = tensor.right_dim();

                    // Contract: new_vec[r] = sum_l left_vec[l] * tensor[l, site_idx, r]
                    let mut new_vec = vec![Complex64::new(0.0, 0.0); right_dim];
                    for l in 0..left_dim.min(left_vec.len()) {
                        for rr in 0..right_dim {
                            new_vec[rr] += left_vec[l] * tensor.get3(l, site_idx, rr);
                        }
                    }
                    left_vec = new_vec;
                }

                // After all sites, left_vec should have size 1 (right boundary)
                matrix[y_flat][x_flat] = if left_vec.is_empty() {
                    Complex64::new(0.0, 0.0)
                } else {
                    left_vec[0]
                };
            }
        }
        matrix
    }

    /// Assert that the MPO representation matches the direct sparse matrix computation
    /// for all elements. This is the primary correctness check: two independent algorithms
    /// (carry-based MPO vs direct enumeration) must agree.
    #[allow(clippy::needless_range_loop)]
    fn assert_affine_mpo_matches_matrix(r: usize, params: &AffineParams, bc: &[BoundaryCondition]) {
        let m = params.m;
        let n = params.n;

        let matrix = affine_transform_matrix(r, params, bc).unwrap();
        let mpo = affine_transform_mpo(r, params, bc).unwrap();
        let mpo_matrix = mpo_to_dense_matrix(&mpo, m, n, r);

        let output_size = 1 << (m * r);
        let input_size = 1 << (n * r);

        for y in 0..output_size {
            for x in 0..input_size {
                let sparse_val = *matrix.get(y, x).unwrap_or(&0.0);
                let mpo_val = mpo_matrix[y][x].re;
                assert!(
                    (sparse_val - mpo_val).abs() < 1e-10,
                    "MPO vs matrix mismatch at ({}, {}): sparse={}, mpo={} \
                     [r={}, m={}, n={}, bc={:?}]",
                    y,
                    x,
                    sparse_val,
                    mpo_val,
                    r,
                    m,
                    n,
                    bc
                );
            }
        }
    }

    /// Assert that affine_transform_matrix produces correct results by independently
    /// computing y = A*x + b using Rational64 arithmetic (no integer scaling).
    /// Equivalent to Julia's test_affine_transform_matrix_multi_variables.
    #[allow(clippy::needless_range_loop)]
    fn assert_affine_matrix_correctness(r: usize, params: &AffineParams, bc: &[BoundaryCondition]) {
        let m = params.m;
        let n = params.n;
        let modulus = 1i64 << r;

        let matrix = affine_transform_matrix(r, params, bc).unwrap();

        let input_size = 1usize << (r * n);
        let output_size = 1usize << (r * m);

        // Build expected matrix independently using Rational64
        for x_flat in 0..input_size {
            // Decode x_flat to N-dimensional vector
            let x_vals: Vec<i64> = (0..n)
                .map(|var| ((x_flat >> (var * r)) & ((1 << r) - 1)) as i64)
                .collect();

            // Compute y = A*x + b using Rational64 (independent of to_integer_scaled)
            let y_rational: Vec<Rational64> = (0..m)
                .map(|i| {
                    let mut val = params.b[i];
                    for j in 0..n {
                        val += params.a[i * n + j] * Rational64::from_integer(x_vals[j]);
                    }
                    val
                })
                .collect();

            // Check if all y values are integers
            if y_rational.iter().any(|y| !y.is_integer()) {
                // No valid output for this input - all entries in this column must be 0
                for y_flat in 0..output_size {
                    let val = *matrix.get(y_flat, x_flat).unwrap_or(&0.0);
                    assert!(
                        val.abs() < 1e-10,
                        "Expected zero at ({}, {}) for non-integer y, got {} [r={}, bc={:?}]",
                        y_flat,
                        x_flat,
                        val,
                        r,
                        bc
                    );
                }
                continue;
            }

            let y_int: Vec<i64> = y_rational.iter().map(|y| y.to_integer()).collect();

            // Apply boundary conditions
            let bc_periodic: Vec<bool> = bc
                .iter()
                .map(|b| matches!(b, BoundaryCondition::Periodic))
                .collect();

            let y_bounded: Vec<i64> = y_int
                .iter()
                .enumerate()
                .map(|(i, &yi)| {
                    if bc_periodic[i] {
                        ((yi % modulus) + modulus) % modulus
                    } else {
                        yi
                    }
                })
                .collect();

            let valid = y_bounded
                .iter()
                .enumerate()
                .all(|(i, &yi)| bc_periodic[i] || (yi >= 0 && yi < modulus));

            if valid {
                let y_flat: usize = y_bounded
                    .iter()
                    .enumerate()
                    .map(|(var, &yi)| (yi as usize) << (var * r))
                    .sum();

                // This (y_flat, x_flat) should be 1
                let val = *matrix.get(y_flat, x_flat).unwrap_or(&0.0);
                assert!(
                    (val - 1.0).abs() < 1e-10,
                    "Expected 1 at ({}, {}) but got {} [r={}, x={:?}, y={:?}, bc={:?}]",
                    y_flat,
                    x_flat,
                    val,
                    r,
                    x_vals,
                    y_bounded,
                    bc
                );
            }
        }
    }

    // MPO vs matrix comparison tests

    #[test]
    fn test_affine_mpo_vs_matrix_1d_identity() {
        let params = AffineParams::from_integers(vec![1], vec![0], 1, 1).unwrap();
        let bc = vec![BoundaryCondition::Periodic];
        assert_affine_mpo_matches_matrix(3, &params, &bc);
    }

    #[test]
    fn test_affine_mpo_vs_matrix_1d_shift() {
        let params = AffineParams::from_integers(vec![1], vec![3], 1, 1).unwrap();
        let bc = vec![BoundaryCondition::Periodic];
        assert_affine_mpo_matches_matrix(3, &params, &bc);
    }

    #[test]
    fn test_affine_mpo_vs_matrix_simple() {
        let params = AffineParams::from_integers(vec![1, 0, 1, 1], vec![0, 0], 2, 2).unwrap();
        let bc = vec![BoundaryCondition::Periodic; 2];
        assert_affine_mpo_matches_matrix(3, &params, &bc);
    }

    #[test]
    fn test_affine_matrix_3x3_hard() {
        // From Quantics.jl compare_hard test
        // A = [1 0 1; 1 2 -1; 0 1 1], b = [11; 23; -15]
        let r = 3;
        let a = vec![1i64, 0, 1, 1, 2, -1, 0, 1, 1];
        let b = vec![11i64, 23, -15];
        let params = AffineParams::from_integers(a, b, 3, 3).unwrap();
        let bc = vec![BoundaryCondition::Periodic; 3];
        assert_affine_mpo_matches_matrix(r, &params, &bc);
    }

    #[test]
    fn test_affine_matrix_rectangular() {
        // From Quantics.jl compare_rect test
        // A = [1 0 1; 1 2 0] (2x3), b = [11; -3]
        let r = 4;
        let a = vec![1i64, 0, 1, 1, 2, 0];
        let b = vec![11i64, -3];
        let params = AffineParams::from_integers(a, b, 2, 3).unwrap();
        let bc = vec![BoundaryCondition::Periodic; 2];
        assert_affine_mpo_matches_matrix(r, &params, &bc);
    }

    #[test]
    fn test_affine_matrix_denom_odd() {
        // From Quantics.jl compare_denom_odd test
        // A = [1/3], b = [0]
        for r in [1, 3, 6] {
            for bc in [BoundaryCondition::Periodic, BoundaryCondition::Open] {
                let a = vec![Rational64::new(1, 3)];
                let b = vec![Rational64::from_integer(0)];
                let params = AffineParams::new(a, b, 1, 1).unwrap();
                let bcs = vec![bc];
                assert_affine_mpo_matches_matrix(r, &params, &bcs);
            }
        }
    }

    #[test]
    fn test_affine_matrix_light_cone() {
        // From Quantics.jl compare_light_cone test
        // Light cone transformation: A = 1/2 * [[1, 1], [1, -1]], b = [2, 3]
        for r in [3, 4] {
            for bc in [BoundaryCondition::Periodic, BoundaryCondition::Open] {
                let a = vec![
                    Rational64::new(1, 2),
                    Rational64::new(1, 2),
                    Rational64::new(1, 2),
                    Rational64::new(-1, 2),
                ];
                let b = vec![Rational64::from_integer(2), Rational64::from_integer(3)];
                let params = AffineParams::new(a, b, 2, 2).unwrap();
                let bcs = vec![bc; 2];
                assert_affine_matrix_correctness(r, &params, &bcs);
                assert_affine_mpo_matches_matrix(r, &params, &bcs);
            }
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn test_affine_matrix_unitarity_full() {
        // From Quantics.jl full test - verify T'*T == I for orthogonal transforms
        // A = [[1, 0], [1, 1]], b = [0, 0]
        let r = 4;
        let a = vec![1i64, 0, 1, 1]; // [[1, 0], [1, 1]]
        let b = vec![0i64, 0];
        let params = AffineParams::from_integers(a, b, 2, 2).unwrap();
        let bc = vec![BoundaryCondition::Periodic; 2];

        let t = affine_transform_matrix(r, &params, &bc).unwrap();

        // Compute T' * T
        let size = 1 << (2 * r);
        let mut prod = vec![vec![0.0; size]; size];
        for i in 0..size {
            for j in 0..size {
                let mut sum = 0.0;
                for k in 0..size {
                    let t_ki = *t.get(k, i).unwrap_or(&0.0);
                    let t_kj = *t.get(k, j).unwrap_or(&0.0);
                    sum += t_ki * t_kj;
                }
                prod[i][j] = sum;
            }
        }

        // Check T' * T == I
        for i in 0..size {
            for j in 0..size {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (prod[i][j] - expected).abs() < 1e-10,
                    "T'*T not identity at ({}, {}): got {}",
                    i,
                    j,
                    prod[i][j]
                );
            }
        }
    }

    #[test]
    fn test_affine_mpo_vs_matrix_r1() {
        let bc = vec![BoundaryCondition::Periodic];
        // Identity R=1
        let params = AffineParams::from_integers(vec![1], vec![0], 1, 1).unwrap();
        assert_affine_mpo_matches_matrix(1, &params, &bc);
        // Shift R=1 (y = x + 1 mod 2)
        let params = AffineParams::from_integers(vec![1], vec![1], 1, 1).unwrap();
        assert_affine_mpo_matches_matrix(1, &params, &bc);
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn test_affine_matrix_unitarity_with_shift() {
        // From Quantics.jl full test with shift - verify T*T' == I
        // A = [[1, 0], [1, 1]], b = [4, 1]
        let r = 4;
        let a = vec![1i64, 0, 1, 1];
        let b = vec![4i64, 1];
        let params = AffineParams::from_integers(a, b, 2, 2).unwrap();
        let bc = vec![BoundaryCondition::Periodic; 2];

        let t = affine_transform_matrix(r, &params, &bc).unwrap();

        // Compute T * T'
        let size = 1 << (2 * r);
        let mut prod = vec![vec![0.0; size]; size];
        for i in 0..size {
            for j in 0..size {
                let mut sum = 0.0;
                for k in 0..size {
                    let t_ik = *t.get(i, k).unwrap_or(&0.0);
                    let t_jk = *t.get(j, k).unwrap_or(&0.0);
                    sum += t_ik * t_jk;
                }
                prod[i][j] = sum;
            }
        }

        // Check T * T' == I
        for i in 0..size {
            for j in 0..size {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (prod[i][j] - expected).abs() < 1e-10,
                    "T*T' not identity at ({}, {}): got {}",
                    i,
                    j,
                    prod[i][j]
                );
            }
        }
    }

    // ========== Unfused API tests ==========

    #[test]
    fn test_affine_unfused_basic() {
        // Test unfused API basic functionality
        let r = 3;
        let params = AffineParams::from_integers(vec![1], vec![0], 1, 1).unwrap();
        let bc = vec![BoundaryCondition::Periodic];

        let unfused = affine_transform_tensors_unfused(r, &params, &bc).unwrap();

        assert_eq!(unfused.len(), r);

        // For M=1, N=1, site_dim = 2^2 = 4
        let site_dim = 4;
        // For identity transform y=x, carry is always 0, so bond dimension is 1
        assert_eq!(*unfused[0].shape(), (1, site_dim, 1)); // First tensor: (1, 4, 1)
        assert_eq!(*unfused[r - 1].shape(), (1, site_dim, 1)); // Last tensor: (1, 4, 1)
    }

    #[test]
    fn test_affine_unfused_2d() {
        // Test unfused API with 2D transformation
        let r = 2;
        let params = AffineParams::from_integers(vec![1, 0, 0, 1], vec![0, 0], 2, 2).unwrap();
        let bc = vec![BoundaryCondition::Periodic; 2];

        let unfused = affine_transform_tensors_unfused(r, &params, &bc).unwrap();

        assert_eq!(unfused.len(), r);

        // For M=2, N=2, site_dim = 2^4 = 16
        let site_dim = 16;
        for tensor in &unfused {
            assert_eq!(tensor.shape().1, site_dim);
        }
    }

    #[test]
    fn test_unfused_tensor_info() {
        let params = AffineParams::from_integers(vec![1, 0, 0, 1], vec![0, 0], 2, 2).unwrap();
        let info = UnfusedTensorInfo::new(&params);

        assert_eq!(info.m, 2);
        assert_eq!(info.n, 2);
        assert_eq!(info.num_physical_dims, 4);
        assert_eq!(info.physical_dim, 2);

        // Test shape
        let shape = info.unfused_shape(3, 5);
        assert_eq!(shape, vec![3, 2, 2, 2, 2, 5]);

        // Test index encoding/decoding
        // site_idx = y_bits | (x_bits << m)
        // y_bits = y0 + 2*y1, x_bits = x0 + 2*x1
        // Example: y0=1, y1=0, x0=0, x1=1 -> y_bits=1, x_bits=2 -> site_idx = 1 + 4*2 = 9
        let (y_bits, x_bits) = info.decode_fused_index(9);
        assert_eq!(y_bits, vec![1, 0]);
        assert_eq!(x_bits, vec![0, 1]);

        let encoded = info.encode_fused_index(&[1, 0], &[0, 1]);
        assert_eq!(encoded, 9);
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn test_unfused_vs_fused_equivalence() {
        // Verify that unfused tensors give the same matrix as fused
        let r = 2;
        let params = AffineParams::from_integers(vec![1, 0, 1, 1], vec![0, 0], 2, 2).unwrap();
        let bc = vec![BoundaryCondition::Periodic; 2];

        let matrix = affine_transform_matrix(r, &params, &bc).unwrap();
        let unfused = affine_transform_tensors_unfused(r, &params, &bc).unwrap();

        // Contract unfused tensors to matrix
        let info = UnfusedTensorInfo::new(&params);
        let m = info.m;
        let n = info.n;
        let output_size = 1 << (m * r);
        let input_size = 1 << (n * r);

        let mut unfused_matrix = vec![vec![Complex64::new(0.0, 0.0); input_size]; output_size];

        for y_flat in 0..output_size {
            for x_flat in 0..input_size {
                let mut left_vec = vec![Complex64::one()];

                for site in 0..r {
                    let bit_pos = r - 1 - site;

                    let mut y_bits = 0usize;
                    for var in 0..m {
                        let y_var = (y_flat >> (var * r)) & ((1 << r) - 1);
                        let bit = (y_var >> bit_pos) & 1;
                        y_bits |= bit << var;
                    }

                    let mut x_bits = 0usize;
                    for var in 0..n {
                        let x_var = (x_flat >> (var * r)) & ((1 << r) - 1);
                        let bit = (x_var >> bit_pos) & 1;
                        x_bits |= bit << var;
                    }

                    let site_idx = y_bits | (x_bits << m);
                    let tensor = &unfused[site];
                    let (left_dim, _, right_dim) = *tensor.shape();

                    let mut new_vec = vec![Complex64::new(0.0, 0.0); right_dim];
                    for l in 0..left_dim.min(left_vec.len()) {
                        for rr in 0..right_dim {
                            new_vec[rr] += left_vec[l] * tensor[[l, site_idx, rr]];
                        }
                    }
                    left_vec = new_vec;
                }

                unfused_matrix[y_flat][x_flat] = if left_vec.is_empty() {
                    Complex64::new(0.0, 0.0)
                } else {
                    left_vec[0]
                };
            }
        }

        // Compare
        let size = 1 << (2 * r);
        for y in 0..size {
            for x in 0..size {
                let sparse_val = *matrix.get(y, x).unwrap_or(&0.0);
                let unfused_val = unfused_matrix[y][x].re;
                assert!(
                    (sparse_val - unfused_val).abs() < 1e-10,
                    "Unfused vs fused mismatch at ({}, {}): sparse={}, unfused={}",
                    y,
                    x,
                    sparse_val,
                    unfused_val
                );
            }
        }
    }

    #[test]
    fn test_affine_parametric_full() {
        // From Quantics.jl "full R=$R, boundary=$boundary, M=$M, N=$N" test
        struct TestCase {
            a: Vec<i64>,
            b: Vec<i64>,
            m: usize,
            n: usize,
        }

        let cases = vec![
            TestCase {
                a: vec![1],
                b: vec![1],
                m: 1,
                n: 1,
            },
            TestCase {
                a: vec![1, 0],
                b: vec![0],
                m: 1,
                n: 2,
            },
            TestCase {
                a: vec![2, -1],
                b: vec![1],
                m: 1,
                n: 2,
            },
            TestCase {
                a: vec![1, 0],
                b: vec![0, 0],
                m: 2,
                n: 1,
            },
            TestCase {
                a: vec![2, -1],
                b: vec![1, -1],
                m: 2,
                n: 1,
            },
            TestCase {
                a: vec![1, 0, 1, 1],
                b: vec![0, 1],
                m: 2,
                n: 2,
            },
            TestCase {
                a: vec![2, 0, 4, 1],
                b: vec![100, -1],
                m: 2,
                n: 2,
            },
        ];

        for r in [1, 2] {
            for bc_type in [BoundaryCondition::Open, BoundaryCondition::Periodic] {
                for case in &cases {
                    let params =
                        AffineParams::from_integers(case.a.clone(), case.b.clone(), case.m, case.n)
                            .unwrap();
                    let bc = vec![bc_type; case.m];
                    assert_affine_matrix_correctness(r, &params, &bc);
                    assert_affine_mpo_matches_matrix(r, &params, &bc);
                }
            }
        }
    }

    #[test]
    fn test_affine_denom_even() {
        // From Quantics.jl compare_denom_even test
        let a = vec![Rational64::new(1, 2)];
        for b_val in [3i64, 5, -3, -5] {
            let b = vec![Rational64::from_integer(b_val)];
            let params = AffineParams::new(a.clone(), b, 1, 1).unwrap();
            let bc = vec![BoundaryCondition::Periodic];
            for r in [3, 5] {
                assert_affine_mpo_matches_matrix(r, &params, &bc);
            }
        }
    }

    #[test]
    fn test_affine_extension_loop() {
        // Test abs(b) >= 2^R with Open BC (requires extension loop)

        // b=[-32, 32] with R=5, identity matrix: abs(32)=2^5=2^R triggers extension
        let r = 5;
        let params = AffineParams::from_integers(vec![1, 0, 0, 1], vec![-32, 32], 2, 2).unwrap();
        let bc = vec![BoundaryCondition::Open; 2];
        assert_affine_mpo_matches_matrix(r, &params, &bc);
        assert_affine_matrix_correctness(r, &params, &bc);

        // abs(b) clearly exceeds 2^R: 2^4=16, abs(b)=32 > 16
        let r = 4;
        assert_affine_mpo_matches_matrix(r, &params, &bc);
        assert_affine_matrix_correctness(r, &params, &bc);

        // 1D case: y = x + 64 with R=6, Open BC
        let r = 6;
        let params = AffineParams::from_integers(vec![1], vec![64], 1, 1).unwrap();
        let bc = vec![BoundaryCondition::Open];
        assert_affine_mpo_matches_matrix(r, &params, &bc);
        assert_affine_matrix_correctness(r, &params, &bc);
    }
}
