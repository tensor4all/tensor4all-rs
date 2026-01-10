# tensor4all-quantics-transform

## src/affine.rs

### `pub fn new(a: Vec < Rational64 >, b: Vec < Rational64 >, m: usize, n: usize) -> Result < Self >` (impl AffineParams)

Create new affine parameters.

### `pub fn from_integers(a: Vec < i64 >, b: Vec < i64 >, m: usize, n: usize) -> Result < Self >` (impl AffineParams)

Create affine parameters from integer matrix and vector.

### ` fn get_a(&self, i: usize, j: usize) -> Rational64` (impl AffineParams)

Get element A[i, j] (0-indexed)

### ` fn to_integer_scaled(&self) -> (Vec < i64 > , Vec < i64 > , i64)` (impl AffineParams)

Convert to integer representation by scaling with LCM of denominators. Returns (A_int, b_int, scale) where A_int = scale * A and b_int = scale * b.

### `pub fn affine_operator(r: usize, params: & AffineParams, bc: & [BoundaryCondition]) -> Result < QuanticsOperator >`

Create an affine transformation operator. This operator transforms a quantics tensor train representing a function f(x_1, ..., x_N) to g(y_1, ..., y_M) where y = A*x + b.

### ` fn affine_transform_mpo(r: usize, params: & AffineParams, bc: & [BoundaryCondition]) -> Result < TensorTrain < Complex64 > >`

Create the affine transformation MPO as a TensorTrain.

### ` fn affine_transform_tensors(r: usize, a_int: & [i64], b_int: & [i64], scale: i64, m: usize, n: usize, bc_periodic: & [bool]) -> Result < Vec < tensor4all_simpletensortrain :: Tensor3 < Complex64 > > >`

Compute the core tensors for the affine transformation. This implements the algorithm from Quantics.jl that handles: - Carry propagation for multi-bit arithmetic

### ` fn affine_transform_core(a_int: & [i64], b_curr: & [i64], scale: i64, m: usize, n: usize, carries_in: & [Vec < i64 >], _is_lsb: bool) -> Result < (Vec < Vec < i64 > > , Vec < Vec < Vec < bool > > >) >`

Compute a single core tensor for the affine transformation. The core tensor encodes: 2 * carry_out = A * x + b_curr - scale * y + carry_in Returns (new_carries, data) where:

### ` fn test_affine_params_new()`

### ` fn test_affine_params_from_integers()`

### ` fn test_affine_params_to_integer_scaled()`

### ` fn test_affine_transform_identity()`

### ` fn test_affine_operator_creation()`

### ` fn test_affine_error_zero_bits()`

### ` fn test_affine_error_bc_mismatch()`

### ` fn test_affine_params_dimension_error()`

### ` fn test_affine_with_rational_coefficients()`

### ` fn test_affine_shift_only()`

### ` fn test_affine_scale_by_two()`

### ` fn test_affine_asymmetric_dimensions()`

### ` fn test_affine_open_boundary()`

### ` fn test_affine_negation()`

### ` fn test_affine_mpo_structure()`

### ` fn test_affine_larger_bits()`

## src/binaryop.rs

### `pub fn new(a: i8, b: i8) -> Result < Self >` (impl BinaryCoeffs)

Create new binary coefficients. Returns error if |a| > 1, |b| > 1, or (a, b) == (-1, -1).

### `pub fn select_x() -> Self` (impl BinaryCoeffs)

Create identity transformation for first variable: (a, b) = (1, 0)

### `pub fn select_y() -> Self` (impl BinaryCoeffs)

Create identity transformation for second variable: (a, b) = (0, 1)

### `pub fn sum() -> Self` (impl BinaryCoeffs)

Create sum transformation: (a, b) = (1, 1)

### `pub fn difference() -> Self` (impl BinaryCoeffs)

Create difference transformation: (a, b) = (1, -1)

### `pub fn binaryop_operator(r: usize, coeffs1: BinaryCoeffs, coeffs2: BinaryCoeffs, bc: [BoundaryCondition ; 2]) -> Result < QuanticsOperator >`

Create a binary operation operator for two-variable quantics representation. This operator transforms a function g(x, y) to f(x, y) = g(a1*x + b1*y, a2*x + b2*y) where the variables x and y are in interleaved quantics representation:

### ` fn binaryop_mpo(r: usize, coeffs1: BinaryCoeffs, _coeffs2: BinaryCoeffs, bc: [BoundaryCondition ; 2]) -> Result < TensorTrain < Complex64 > >`

Create the binary operation MPO as a TensorTrain. The MPO operates on interleaved sites [x_1, y_1, x_2, y_2, ..., x_R, y_R] and computes two output variables:

### ` fn binaryop_tensor_single(a: i8, b: i8, cin_on: bool, cout_on: bool, bc: i8) -> Vec < Vec < Vec < Vec < Vec < Complex64 > > > > >`

Create a single binaryop tensor for one site. This is a direct port of _binaryop_tensor from Quantics.jl. Computes: a*x + b*y + carry_in = 2*carry_out + out

### `pub fn binaryop_single_mpo(r: usize, a: i8, b: i8, bc: BoundaryCondition) -> Result < TensorTrain < Complex64 > >`

Create a binaryop MPO for a simpler case: single output variable. This computes f(x, y) = g(a*x + b*y, y) where x and y are interleaved.

### `pub fn binaryop_single_operator(r: usize, a: i8, b: i8, bc: BoundaryCondition) -> Result < QuanticsOperator >`

Create a binary operation operator for a single transformation. This transforms f(x, y) where the first variable is transformed by a*x + b*y and the second variable y remains unchanged.

### ` fn test_binary_coeffs_valid()`

### ` fn test_binary_coeffs_invalid()`

### ` fn test_binaryop_tensor_single()`

### ` fn test_binaryop_tensor_difference()`

### ` fn test_binaryop_single_mpo_structure()`

### ` fn test_binaryop_single_operator_creation()`

### ` fn test_binaryop_single_identity()`

### ` fn test_binaryop_error_cases()`

## src/common.rs

### `pub fn tensortrain_to_linear_operator(tt: & TensorTrain < Complex64 >, site_dims: & [usize]) -> Result < QuanticsOperator >`

Convert a TensorTrain (MPO form) to a LinearOperator. The TensorTrain is assumed to be an MPO with site dimension 4 (2x2 for input/output). Each site tensor has shape (left_bond, site_dim=4, right_bond) where site_dim

### `pub fn tensortrain_to_linear_operator_asymmetric(tt: & TensorTrain < Complex64 >, input_dims: & [usize], output_dims: & [usize]) -> Result < QuanticsOperator >`

Convert a TensorTrain (MPO form) to a LinearOperator with asymmetric dimensions. This variant supports different input and output dimensions, useful for multi-variable transformations like affine transforms.

### `pub fn identity_mpo(r: usize) -> Result < TensorTrain < Complex64 > >`

Create an identity MPO for r sites with dimension 2.

### `pub fn scalar_mpo(r: usize, value: Complex64) -> Result < TensorTrain < Complex64 > >`

Create a scalar MPO (constant times identity).

### ` fn test_boundary_condition_default()`

### ` fn test_carry_direction_default()`

### ` fn test_identity_mpo()`

## src/cumsum.rs

### `pub fn cumsum_operator(r: usize) -> Result < QuanticsOperator >`

Create a cumulative sum operator: y_i = Σ_{j < i} x_j This MPO implements a strict upper triangular matrix filled with ones. For a function g defined on {0, 1, ..., 2^R - 1}, it computes:

### ` fn cumsum_mpo(r: usize) -> Result < TensorTrain < Complex64 > >`

Create the cumulative sum MPO as a TensorTrain. The cumulative sum is implemented as an upper triangular matrix. The MPO tracks whether a comparison has been made:

### ` fn upper_triangle_tensor() -> [[[[Complex64 ; 2] ; 2] ; 2] ; 2]`

Create the single-site tensor for upper triangular matrix. Returns a 4D tensor [cin][cout][y_bit][x_bit] where: - cin: input state (0 = no comparison yet, 1 = comparison made)

### ` fn test_upper_triangle_tensor()`

### ` fn test_cumsum_mpo_structure()`

### ` fn test_cumsum_operator_creation()`

### ` fn test_cumsum_error_zero_sites()`

## src/flip.rs

### `pub fn flip_operator(r: usize, bc: BoundaryCondition) -> Result < QuanticsOperator >`

Create a flip operator: f(x) = g(2^R - x) This MPO transforms a function g(x) to f(x) = g(2^R - x) for x = 0, 1, ..., 2^R - 1.

### ` fn flip_mpo(r: usize, bc: BoundaryCondition) -> Result < TensorTrain < Complex64 > >`

Create the flip MPO as a TensorTrain. The flip operation computes -x using two's complement arithmetic: -x = ~x + 1 (bitwise NOT plus one)

### ` fn single_tensor_flip() -> [[[[Complex64 ; 2] ; 2] ; 2] ; 2]`

Create the single-site tensor for flip operation. Returns a 4D tensor [cin][cout][s_out][s_in] where: - cin: input carry state (0 = carry -1, 1 = carry 0)

### ` fn test_single_tensor_flip()`

### ` fn test_flip_mpo_structure()`

### ` fn test_flip_operator_creation()`

### ` fn test_flip_error_single_site()`

### ` fn test_flip_error_zero_sites()`

## src/fourier.rs

### ` fn default() -> Self` (impl FourierOptions)

### `pub fn forward() -> Self` (impl FourierOptions)

Create options for forward Fourier transform.

### `pub fn inverse() -> Self` (impl FourierOptions)

Create options for inverse Fourier transform.

### `pub fn new(r: usize, options: FourierOptions) -> Result < Self >` (impl FTCore)

Create a new FTCore for r bits.

### `pub fn forward(&self) -> Result < QuanticsOperator >` (impl FTCore)

Get the forward Fourier transform operator.

### `pub fn backward(&self) -> Result < QuanticsOperator >` (impl FTCore)

Get the backward (inverse) Fourier transform operator.

### `pub fn r(&self) -> usize` (impl FTCore)

Get the number of bits.

### `pub fn quantics_fourier_operator(r: usize, options: FourierOptions) -> Result < QuanticsOperator >`

Create a Quantics Fourier Transform operator. This implements the Chen & Lindsey construction of the DFT as a matrix product operator. The resulting operator transforms a quantics tensor train representing a function

### ` fn quantics_fourier_mpo(r: usize, options: & FourierOptions) -> Result < TensorTrain < Complex64 > >`

Create the QFT MPO as a TensorTrain using Chen & Lindsey construction.

### ` fn chebyshev_grid(k: usize) -> (Vec < f64 > , Vec < f64 >)`

Get Chebyshev grid points and barycentric weights. Returns (grid, bary_weights) where: - grid[j] = 0.5 * (1 - cos(π*j/K)) for j = 0, ..., K

### ` fn lagrange_polynomial(grid: & [f64], bary_weights: & [f64], alpha: usize, x: f64) -> f64`

Evaluate Lagrange polynomial P_alpha(x).

### ` fn build_dft_core_tensor(grid: & [f64], bary_weights: & [f64], sign: f64) -> Vec < Vec < Vec < Vec < Complex64 > > > >`

Build the DFT core tensor A[alpha, tau, sigma, beta]. A[alpha, tau, sigma, beta] = P_alpha(x) * exp(2πi * sign * x * tau) where x = (sigma + grid[beta]) / 2

### ` fn test_chebyshev_grid()`

### ` fn test_lagrange_polynomial_at_grid_points()`

### ` fn test_fourier_mpo_structure()`

### ` fn test_fourier_operator_creation()`

### ` fn test_ftcore_creation()`

### ` fn test_fourier_inverse_sign()`

### ` fn test_fourier_error_zero_sites()`

## src/phase_rotation.rs

### `pub fn phase_rotation_operator(r: usize, theta: f64) -> Result < QuanticsOperator >`

Create a phase rotation operator: f(x) = exp(i*θ*x) * g(x) This MPO multiplies a function g(x) by the phase factor exp(i*θ*x). In quantics representation, x = Σ_n x_n * 2^(R-n), so:

### ` fn phase_rotation_mpo(r: usize, theta: f64) -> Result < TensorTrain < Complex64 > >`

Create the phase rotation MPO as a TensorTrain. Each site tensor is diagonal with entries: - For x_n = 0: 1

### ` fn test_phase_rotation_mpo_structure()`

### ` fn test_phase_rotation_zero_theta()`

### ` fn test_phase_rotation_pi()`

### ` fn test_phase_rotation_operator_creation()`

### ` fn test_phase_rotation_error_zero_sites()`

### ` fn test_phase_rotation_periodicity()`

## src/shift.rs

### `pub fn shift_operator(r: usize, offset: i64, bc: BoundaryCondition) -> Result < QuanticsOperator >`

Create a shift operator: f(x) = g(x + offset) mod 2^R This MPO transforms a function g(x) to f(x) = g(x + offset) for x = 0, 1, ..., 2^R - 1.

### ` fn shift_mpo(r: usize, offset: i64, bc: BoundaryCondition) -> Result < TensorTrain < Complex64 > >`

Create the shift MPO as a TensorTrain. The shift operation computes x + offset using binary addition with carry propagation. The offset is decomposed into binary: offset = Σ_n offset_n * 2^(R-n)

### ` fn test_shift_mpo_structure()`

### ` fn test_shift_zero()`

### ` fn test_shift_operator_creation()`

### ` fn test_shift_negative()`

### ` fn test_shift_single_site()`

### ` fn test_shift_error_zero_sites()`

