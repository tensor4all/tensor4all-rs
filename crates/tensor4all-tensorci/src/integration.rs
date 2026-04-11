//! Numerical integration using TCI and Gauss-Kronrod quadrature.
//!
//! [`integrate`] approximates a multi-dimensional integral over a hypercube
//! by building a tensor-train representation of the integrand on
//! Gauss-Kronrod nodes and then summing the tensor train.
//!
//! Supported quadrature orders: 15 and 31 (standard Gauss-Kronrod rules).

use crate::error::{Result, TCIError};
use crate::tensorci2::{crossinterpolate2, TCI2Options};
use tensor4all_simplett::{AbstractTensorTrain, TTScalar};
use tensor4all_tcicore::{
    DenseFaerLuKernel, LazyBlockRookKernel, MatrixLuciScalar, MultiIndex, PivotKernel, Scalar,
};

/// Gauss-Kronrod 15-point rule: nodes on [-1, 1]
const GK15_NODES: [f64; 15] = [
    -0.991_455_371_120_812_6,
    -0.949_107_912_342_758_5,
    -0.864_864_423_359_769_1,
    -0.741_531_185_599_394_5,
    -0.586_087_235_467_691_1,
    -0.405_845_151_377_397_2,
    -0.207_784_955_007_898_48,
    0.000000000000000000000000000000000,
    0.207_784_955_007_898_48,
    0.405_845_151_377_397_2,
    0.586_087_235_467_691_1,
    0.741_531_185_599_394_5,
    0.864_864_423_359_769_1,
    0.949_107_912_342_758_5,
    0.991_455_371_120_812_6,
];

/// Gauss-Kronrod 15-point rule: weights on [-1, 1]
const GK15_WEIGHTS: [f64; 15] = [
    0.022_935_322_010_529_224,
    0.063_092_092_629_978_56,
    0.104_790_010_322_250_19,
    0.140_653_259_715_525_92,
    0.169_004_726_639_267_9,
    0.190_350_578_064_785_42,
    0.204_432_940_075_298_89,
    0.209_482_141_084_727_82,
    0.204_432_940_075_298_89,
    0.190_350_578_064_785_42,
    0.169_004_726_639_267_9,
    0.140_653_259_715_525_92,
    0.104_790_010_322_250_19,
    0.063_092_092_629_978_56,
    0.022_935_322_010_529_224,
];

/// Gauss-Kronrod 31-point rule: nodes on [-1, 1]
/// Source: QUADPACK (Piessens et al., 1983)
const GK31_NODES: [f64; 31] = [
    -0.998_002_298_693_397_1,
    -0.987_992_518_020_485_4,
    -0.967_739_075_679_139_1,
    -0.937_273_392_400_706,
    -0.897_264_532_344_081_9,
    -0.848_206_583_410_427_2,
    -0.790_418_501_442_466,
    -0.724_417_731_360_170_1,
    -0.650_996_741_297_416_9,
    -0.570_972_172_608_539,
    -0.485_081_863_640_239_7,
    -0.394_151_347_077_563_4,
    -0.299_180_007_153_168_8,
    -0.201_194_093_997_434_5,
    -0.101_142_066_918_717_5,
    0.0,
    0.101_142_066_918_717_5,
    0.201_194_093_997_434_5,
    0.299_180_007_153_168_8,
    0.394_151_347_077_563_4,
    0.485_081_863_640_239_7,
    0.570_972_172_608_539,
    0.650_996_741_297_416_9,
    0.724_417_731_360_170_1,
    0.790_418_501_442_466,
    0.848_206_583_410_427_2,
    0.897_264_532_344_081_9,
    0.937_273_392_400_706,
    0.967_739_075_679_139_1,
    0.987_992_518_020_485_4,
    0.998_002_298_693_397_1,
];

/// Gauss-Kronrod 31-point rule: weights on [-1, 1]
const GK31_WEIGHTS: [f64; 31] = [
    0.005_377_479_872_923_349,
    0.015_007_947_329_316_122,
    0.025_460_847_326_715_32,
    0.035_346_360_791_375_85,
    0.044_589_751_324_764_88,
    0.053_481_524_690_928_09,
    0.062_009_567_800_670_64,
    0.069_854_121_318_728_26,
    0.076_849_680_757_720_38,
    0.083_080_502_823_133_17,
    0.088_564_443_056_211_77,
    0.093_126_598_170_825_32,
    0.096_642_726_983_623_68,
    0.099_173_598_721_791_96,
    0.100_769_845_523_875_6,
    0.101_330_007_014_791_55,
    0.100_769_845_523_875_6,
    0.099_173_598_721_791_96,
    0.096_642_726_983_623_68,
    0.093_126_598_170_825_32,
    0.088_564_443_056_211_77,
    0.083_080_502_823_133_17,
    0.076_849_680_757_720_38,
    0.069_854_121_318_728_26,
    0.062_009_567_800_670_64,
    0.053_481_524_690_928_09,
    0.044_589_751_324_764_88,
    0.035_346_360_791_375_85,
    0.025_460_847_326_715_32,
    0.015_007_947_329_316_122,
    0.005_377_479_872_923_349,
];

/// Get Gauss-Kronrod nodes and weights for order `gk_order`.
///
/// Supported orders: 15, 31.
fn gk_nodes_weights(gk_order: usize) -> Result<(&'static [f64], &'static [f64])> {
    match gk_order {
        15 => Ok((&GK15_NODES, &GK15_WEIGHTS)),
        31 => Ok((&GK31_NODES, &GK31_WEIGHTS)),
        _ => Err(TCIError::InvalidOperation {
            message: format!(
                "GK order {} not supported. Supported orders: 15, 31.",
                gk_order
            ),
        }),
    }
}

/// Integrate a function over a hypercube `[a, b]` using TCI and
/// Gauss-Kronrod quadrature.
///
/// The integrand is sampled at Gauss-Kronrod nodes in each dimension,
/// and a tensor-train approximation is built via [`crossinterpolate2`].
/// The integral is then computed as a weighted sum of the tensor train.
///
/// # Arguments
///
/// * `f` -- function to integrate, takes a coordinate slice `&[f64]`
/// * `a` -- lower bounds for each dimension
/// * `b` -- upper bounds for each dimension
/// * `gk_order` -- Gauss-Kronrod quadrature order (15 or 31)
/// * `tci_options` -- options for the TCI approximation
///
/// # Returns
///
/// The approximate integral value.
///
/// # Errors
///
/// Returns an error if `a` and `b` have different lengths, or if
/// `gk_order` is not 15 or 31.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorci::integration::integrate;
/// use tensor4all_tensorci::TCI2Options;
///
/// // Integrate (x^2 + y^2) over [0,1]^2 = 2/3
/// let f = |x: &[f64]| x.iter().map(|&xi| xi * xi).sum::<f64>();
/// let a = vec![0.0, 0.0];
/// let b = vec![1.0, 1.0];
/// let options = TCI2Options { tolerance: 1e-10, ..TCI2Options::default() };
///
/// let result: f64 = integrate(&f, &a, &b, 15, options).unwrap();
/// assert!((result - 2.0 / 3.0).abs() < 1e-8);
/// ```
pub fn integrate<T, F>(
    f: &F,
    a: &[f64],
    b: &[f64],
    gk_order: usize,
    tci_options: TCI2Options,
) -> Result<T>
where
    T: Scalar + TTScalar + Default + MatrixLuciScalar,
    DenseFaerLuKernel: PivotKernel<T>,
    LazyBlockRookKernel: PivotKernel<T>,
    F: Fn(&[f64]) -> T,
{
    if a.len() != b.len() {
        return Err(TCIError::DimensionMismatch {
            message: format!(
                "Lower bounds ({}) and upper bounds ({}) must have same length",
                a.len(),
                b.len()
            ),
        });
    }

    let ndims = a.len();
    let (nodes_ref, weights_ref) = gk_nodes_weights(gk_order)?;
    let n_nodes = nodes_ref.len();

    // Transform nodes from [-1, 1] to [a_i, b_i] and weights
    let mut nodes: Vec<Vec<f64>> = Vec::with_capacity(ndims);
    let mut weights: Vec<Vec<f64>> = Vec::with_capacity(ndims);
    for d in 0..ndims {
        let mut dim_nodes = Vec::with_capacity(n_nodes);
        let mut dim_weights = Vec::with_capacity(n_nodes);
        for k in 0..n_nodes {
            // x = (b - a) * (node + 1) / 2 + a
            dim_nodes.push((b[d] - a[d]) * (nodes_ref[k] + 1.0) / 2.0 + a[d]);
            // w = (b - a) * weight / 2
            dim_weights.push((b[d] - a[d]) * weights_ref[k] / 2.0);
        }
        nodes.push(dim_nodes);
        weights.push(dim_weights);
    }

    let normalization = (gk_order as f64).powi(ndims as i32);

    // Create wrapper function: F(indices) = w_prod * f(x) * normalization
    let wrapper = |idx: &MultiIndex| -> T {
        let x: Vec<f64> = (0..ndims).map(|d| nodes[d][idx[d]]).collect();
        let w: f64 = (0..ndims).map(|d| weights[d][idx[d]]).product();
        let fval = f(&x);
        // fval * (w * normalization)
        let scale = <T as Scalar>::from_f64(w * normalization);
        fval * scale
    };

    let local_dims = vec![n_nodes; ndims];

    let (tci, _ranks, _errors) = crossinterpolate2::<T, _, fn(&[MultiIndex]) -> Vec<T>>(
        wrapper,
        None,
        local_dims,
        vec![],
        tci_options,
    )?;

    let tt = tci.to_tensor_train()?;
    let sum = tt.sum();

    // Return sum / normalization
    let inv_norm = <T as Scalar>::from_f64(1.0 / normalization);
    Ok(sum * inv_norm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integrate_constant() {
        // Integral of 1.0 over [0, 1]^2 = 1.0
        let f = |_x: &[f64]| 1.0f64;
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 1.0];
        let options = TCI2Options {
            tolerance: 1e-10,
            ..TCI2Options::default()
        };

        let result: f64 = integrate(&f, &a, &b, 15, options).unwrap();
        assert!((result - 1.0).abs() < 1e-8, "Expected 1.0, got {}", result);
    }

    #[test]
    fn test_integrate_polynomial() {
        // Integral of x*y over [0, 1]^2 = 1/4
        let f = |x: &[f64]| x[0] * x[1];
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 1.0];
        let options = TCI2Options {
            tolerance: 1e-10,
            ..TCI2Options::default()
        };

        let result: f64 = integrate(&f, &a, &b, 15, options).unwrap();
        assert!(
            (result - 0.25).abs() < 1e-8,
            "Expected 0.25, got {}",
            result
        );
    }

    #[test]
    fn test_integrate_5d_polynomial() {
        // Integral of product of polynomials over [0, 1]^5
        // p(x) = 1 + x + x^2, integral = 1 + 1/2 + 1/3 = 11/6
        // 5D integral = (11/6)^5
        let polynomial = |x: f64| 1.0 + x + x * x;
        let f = |coords: &[f64]| coords.iter().map(|&x| polynomial(x)).product::<f64>();

        let n = 5;
        let a = vec![0.0; n];
        let b = vec![1.0; n];
        let options = TCI2Options {
            tolerance: 1e-10,
            ..TCI2Options::default()
        };

        let result: f64 = integrate(&f, &a, &b, 15, options).unwrap();
        let expected = (11.0 / 6.0_f64).powi(5);
        assert!(
            (result - expected).abs() < 1e-6,
            "Expected {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn test_integrate_with_gk31() {
        // Same polynomial test with GK31 — should be more accurate
        let polynomial = |x: f64| 1.0 + x + x * x;
        let f = |coords: &[f64]| coords.iter().map(|&x| polynomial(x)).product::<f64>();

        let n = 3;
        let a = vec![0.0; n];
        let b = vec![1.0; n];
        let options = TCI2Options {
            tolerance: 1e-12,
            ..TCI2Options::default()
        };

        let result: f64 = integrate(&f, &a, &b, 31, options).unwrap();
        let expected = (11.0 / 6.0_f64).powi(3);
        assert!(
            (result - expected).abs() < 1e-10,
            "GK31: Expected {}, got {}, diff={}",
            expected,
            result,
            (result - expected).abs()
        );
    }

    #[test]
    fn test_gk_order_unsupported() {
        let f = |_x: &[f64]| 1.0f64;
        let a = vec![0.0];
        let b = vec![1.0];
        let options = TCI2Options::default();
        let result: std::result::Result<f64, _> = integrate(&f, &a, &b, 7, options);
        assert!(result.is_err());
    }

    /// Port of Julia test_integration.jl: "Integrate real polynomials" with arbitrary bounds
    #[test]
    fn test_integrate_arbitrary_bounds() {
        // p(x) = 1 + 0.5*x + 0.3*x^2
        // integral of p from a to b = [x + 0.25*x^2 + 0.1*x^3]_a^b
        let polynomial = |x: f64| 1.0 + 0.5 * x + 0.3 * x * x;
        let poly_integral = |x: f64| x + 0.25 * x * x + 0.1 * x * x * x;
        let f = |coords: &[f64]| coords.iter().map(|&x| polynomial(x)).product::<f64>();

        let n = 3;
        let a = vec![0.2, 0.5, 0.1];
        let b = vec![0.8, 0.9, 0.7];
        let expected: f64 = (0..n)
            .map(|i| poly_integral(b[i]) - poly_integral(a[i]))
            .product();

        let options = TCI2Options {
            tolerance: 1e-10,
            ..TCI2Options::default()
        };
        let result: f64 = integrate(&f, &a, &b, 15, options).unwrap();
        assert!(
            (result - expected).abs() < 1e-8,
            "Arbitrary bounds: expected {expected}, got {result}"
        );
    }

    /// Port of Julia test_integration.jl: "Integrate complex polynomials"
    #[test]
    fn test_integrate_complex_polynomial() {
        use num_complex::Complex64;

        let polynomial = |x: f64| Complex64::new(1.0 + x, 0.5 * x * x);
        let poly_integral = |x: f64| Complex64::new(x + 0.5 * x * x, 0.5 * x * x * x / 3.0);
        let f = |coords: &[f64]| coords.iter().map(|&x| polynomial(x)).product::<Complex64>();

        let n = 3;
        let a = vec![0.0; n];
        let b = vec![1.0; n];
        let expected: Complex64 = (0..n)
            .map(|_| poly_integral(1.0) - poly_integral(0.0))
            .product();

        let options = TCI2Options {
            tolerance: 1e-10,
            ..TCI2Options::default()
        };
        let result: Complex64 = integrate(&f, &a, &b, 15, options).unwrap();
        assert!(
            (result - expected).norm() < 1e-8,
            "Complex: expected {expected}, got {result}"
        );
    }
}
