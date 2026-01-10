# matrixci

## src/matrixaca.rs

### `pub fn new(nr: usize, nc: usize) -> Self` (impl MatrixACA < T >)

Create an empty MatrixACA for a matrix of given size

### `pub fn from_matrix_with_pivot(a: & Matrix < T >, first_pivot: (usize , usize)) -> Self` (impl MatrixACA < T >)

Create a MatrixACA from a matrix with an initial pivot

### `pub fn npivots(&self) -> usize` (impl MatrixACA < T >)

Number of pivots

### `pub fn u(&self) -> & Matrix < T >` (impl MatrixACA < T >)

Get reference to U matrix

### `pub fn v(&self) -> & Matrix < T >` (impl MatrixACA < T >)

Get reference to V matrix

### `pub fn alpha(&self) -> & [T]` (impl MatrixACA < T >)

Get reference to alpha values

### ` fn compute_uk(&self, a: & Matrix < T >) -> Vec < T >` (impl MatrixACA < T >)

Compute u_k(x) for all x (the k-th column of U after adding a new pivot column)

### ` fn compute_vk(&self, a: & Matrix < T >) -> Vec < T >` (impl MatrixACA < T >)

Compute v_k(y) for all y (the k-th row of V after adding a new pivot row)

### `pub fn add_pivot_col(&mut self, a: & Matrix < T >, col_index: usize) -> Result < () >` (impl MatrixACA < T >)

Add a pivot column

### `pub fn add_pivot_row(&mut self, a: & Matrix < T >, row_index: usize) -> Result < () >` (impl MatrixACA < T >)

Add a pivot row

### `pub fn add_pivot(&mut self, a: & Matrix < T >, pivot: (usize , usize)) -> Result < () >` (impl MatrixACA < T >)

Add a pivot at the given position

### `pub fn add_best_pivot(&mut self, a: & Matrix < T >) -> Result < (usize , usize) >` (impl MatrixACA < T >)

Add a pivot that maximizes the error using ACA heuristic

### `pub fn set_cols(&mut self, new_pivot_rows: & Matrix < T >, permutation: & [usize])` (impl MatrixACA < T >)

Set columns with new pivot rows and permutation

### `pub fn set_rows(&mut self, new_pivot_cols: & Matrix < T >, permutation: & [usize])` (impl MatrixACA < T >)

Set rows with new pivot columns and permutation

### ` fn nrows(&self) -> usize` (impl MatrixACA < T >)

### ` fn ncols(&self) -> usize` (impl MatrixACA < T >)

### ` fn rank(&self) -> usize` (impl MatrixACA < T >)

### ` fn row_indices(&self) -> & [usize]` (impl MatrixACA < T >)

### ` fn col_indices(&self) -> & [usize]` (impl MatrixACA < T >)

### ` fn evaluate(&self, i: usize, j: usize) -> T` (impl MatrixACA < T >)

### ` fn submatrix(&self, rows: & [usize], cols: & [usize]) -> Matrix < T >` (impl MatrixACA < T >)

### ` fn test_matrixaca_new()`

### ` fn test_matrixaca_from_matrix()`

### ` fn test_matrixaca_add_pivot()`

## src/matrixci.rs

### `pub fn new(nr: usize, nc: usize) -> Self` (impl MatrixCI < T >)

Create an empty MatrixCI for a matrix of given size

### `pub fn from_parts(row_indices: Vec < usize >, col_indices: Vec < usize >, pivot_cols: Matrix < T >, pivot_rows: Matrix < T >) -> Self` (impl MatrixCI < T >)

Create a MatrixCI from existing parts

### `pub fn from_matrix_with_pivot(a: & Matrix < T >, first_pivot: (usize , usize)) -> Self` (impl MatrixCI < T >)

Create a MatrixCI from a matrix with an initial pivot

### `pub fn iset(&self) -> & [usize]` (impl MatrixCI < T >)

Get I set (row indices)

### `pub fn jset(&self) -> & [usize]` (impl MatrixCI < T >)

Get J set (column indices)

### `pub fn pivot_matrix(&self) -> Matrix < T >` (impl MatrixCI < T >)

Get the pivot matrix P = A[I, J]

### `pub fn left_matrix(&self) -> Matrix < T >` (impl MatrixCI < T >)

Get the left matrix L = A[:, J] * P^{-1}

### `pub fn right_matrix(&self) -> Matrix < T >` (impl MatrixCI < T >)

Get the right matrix R = P^{-1} * A[I, :]

### `pub fn first_pivot_value(&self) -> T` (impl MatrixCI < T >)

Get the value of the first pivot

### `pub fn add_pivot_row(&mut self, a: & Matrix < T >, row_index: usize) -> Result < () >` (impl MatrixCI < T >)

Add a pivot row

### `pub fn add_pivot_col(&mut self, a: & Matrix < T >, col_index: usize) -> Result < () >` (impl MatrixCI < T >)

Add a pivot column

### `pub fn add_pivot(&mut self, a: & Matrix < T >, pivot: (usize , usize)) -> Result < () >` (impl MatrixCI < T >)

Add a pivot at the given position

### `pub fn add_best_pivot(&mut self, a: & Matrix < T >) -> Result < (usize , usize) >` (impl MatrixCI < T >)

Add a pivot that maximizes the error

### ` fn nrows(&self) -> usize` (impl MatrixCI < T >)

### ` fn ncols(&self) -> usize` (impl MatrixCI < T >)

### ` fn rank(&self) -> usize` (impl MatrixCI < T >)

### ` fn row_indices(&self) -> & [usize]` (impl MatrixCI < T >)

### ` fn col_indices(&self) -> & [usize]` (impl MatrixCI < T >)

### ` fn evaluate(&self, i: usize, j: usize) -> T` (impl MatrixCI < T >)

### ` fn submatrix(&self, rows: & [usize], cols: & [usize]) -> Matrix < T >` (impl MatrixCI < T >)

### ` fn default() -> Self` (impl CrossInterpolateOptions)

### `pub fn crossinterpolate(a: & Matrix < T >, options: Option < CrossInterpolateOptions >) -> MatrixCI < T >`

Perform cross interpolation of a matrix

### ` fn test_matrixci_new()`

### ` fn test_matrixci_from_matrix()`

### ` fn test_matrixci_add_pivot()`

### ` fn test_crossinterpolate()`

## src/matrixlu.rs

### `pub fn new(nr: usize, nc: usize, left_orthogonal: bool) -> Self` (impl RrLU < T >)

Create an empty rrLU for a matrix of given size

### `pub fn nrows(&self) -> usize` (impl RrLU < T >)

Number of rows

### `pub fn ncols(&self) -> usize` (impl RrLU < T >)

Number of columns

### `pub fn npivots(&self) -> usize` (impl RrLU < T >)

Number of pivots

### `pub fn row_permutation(&self) -> & [usize]` (impl RrLU < T >)

Row permutation

### `pub fn col_permutation(&self) -> & [usize]` (impl RrLU < T >)

Column permutation

### `pub fn row_indices(&self) -> Vec < usize >` (impl RrLU < T >)

Get row indices (selected pivots)

### `pub fn col_indices(&self) -> Vec < usize >` (impl RrLU < T >)

Get column indices (selected pivots)

### `pub fn left(&self, permute: bool) -> Matrix < T >` (impl RrLU < T >)

Get left matrix (optionally permuted)

### `pub fn right(&self, permute: bool) -> Matrix < T >` (impl RrLU < T >)

Get right matrix (optionally permuted)

### `pub fn diag(&self) -> Vec < T >` (impl RrLU < T >)

Get diagonal elements

### `pub fn pivot_errors(&self) -> Vec < f64 >` (impl RrLU < T >)

Get pivot errors

### `pub fn last_pivot_error(&self) -> f64` (impl RrLU < T >)

Get last pivot error

### `pub fn transpose(&self) -> RrLU < T >` (impl RrLU < T >)

Transpose the decomposition

### `pub fn is_left_orthogonal(&self) -> bool` (impl RrLU < T >)

Check if left-orthogonal (L has 1s on diagonal)

### ` fn default() -> Self` (impl RrLUOptions)

### `pub fn rrlu_inplace(a: & mut Matrix < T >, options: Option < RrLUOptions >) -> RrLU < T >`

Perform in-place rank-revealing LU decomposition

### `pub fn rrlu(a: & Matrix < T >, options: Option < RrLUOptions >) -> RrLU < T >`

Perform rank-revealing LU decomposition (non-destructive)

### `pub fn cols_to_l_matrix(c: & mut Matrix < T >, p: & Matrix < T >, _left_orthogonal: bool)`

Convert L matrix to solve L * X = B given pivot matrix P

### `pub fn rows_to_u_matrix(r: & mut Matrix < T >, p: & Matrix < T >, _left_orthogonal: bool)`

Convert R matrix to solve X * U = B given pivot matrix P

### `pub fn solve_lu(l: & Matrix < T >, u: & Matrix < T >, b: & Matrix < T >) -> Result < Matrix < T > >`

Solve LU * x = b

### ` fn test_rrlu_identity()`

### ` fn test_rrlu_rank_deficient()`

### ` fn test_rrlu_full_rank()`

### ` fn test_rrlu_reconstruct()`

## src/matrixluci.rs

### `pub fn from_matrix(a: & Matrix < T >, options: Option < RrLUOptions >) -> Self` (impl MatrixLUCI < T >)

Create a MatrixLUCI from a matrix

### `pub fn from_rrlu(lu: RrLU < T >) -> Self` (impl MatrixLUCI < T >)

Create from an existing rrLU decomposition

### `pub fn lu(&self) -> & RrLU < T >` (impl MatrixLUCI < T >)

Get reference to underlying rrLU

### `pub fn col_matrix(&self) -> Matrix < T >` (impl MatrixLUCI < T >)

Get column matrix: L * U[:, :npivots]

### `pub fn row_matrix(&self) -> Matrix < T >` (impl MatrixLUCI < T >)

Get row matrix: L[:npivots, :] * U

### `pub fn cols_times_pivot_inv(&self) -> Matrix < T >` (impl MatrixLUCI < T >)

Get cols times pivot inverse

### `pub fn pivot_inv_times_rows(&self) -> Matrix < T >` (impl MatrixLUCI < T >)

Get pivot inverse times rows

### `pub fn left(&self) -> Matrix < T >` (impl MatrixLUCI < T >)

Get left matrix for CI representation

### `pub fn right(&self) -> Matrix < T >` (impl MatrixLUCI < T >)

Get right matrix for CI representation

### `pub fn pivot_errors(&self) -> Vec < f64 >` (impl MatrixLUCI < T >)

Get pivot errors

### `pub fn last_pivot_error(&self) -> f64` (impl MatrixLUCI < T >)

Get last pivot error

### ` fn nrows(&self) -> usize` (impl MatrixLUCI < T >)

### ` fn ncols(&self) -> usize` (impl MatrixLUCI < T >)

### ` fn rank(&self) -> usize` (impl MatrixLUCI < T >)

### ` fn row_indices(&self) -> & [usize]` (impl MatrixLUCI < T >)

### ` fn col_indices(&self) -> & [usize]` (impl MatrixLUCI < T >)

### ` fn evaluate(&self, i: usize, j: usize) -> T` (impl MatrixLUCI < T >)

### ` fn submatrix(&self, rows: & [usize], cols: & [usize]) -> Matrix < T >` (impl MatrixLUCI < T >)

### ` fn test_matrixluci_from_matrix()`

### ` fn test_matrixluci_reconstruct()`

### ` fn test_matrixluci_rank_deficient()`

## src/traits.rs

### `pub fn nrows(&self) -> usize` (trait AbstractMatrixCI)

Number of rows in the approximated matrix

### `pub fn ncols(&self) -> usize` (trait AbstractMatrixCI)

Number of columns in the approximated matrix

### `pub fn rank(&self) -> usize` (trait AbstractMatrixCI)

Current rank of the approximation (number of pivots)

### `pub fn row_indices(&self) -> & [usize]` (trait AbstractMatrixCI)

Row indices selected as pivots (I set)

### `pub fn col_indices(&self) -> & [usize]` (trait AbstractMatrixCI)

Column indices selected as pivots (J set)

### `pub fn is_empty(&self) -> bool` (trait AbstractMatrixCI default)

Check if the approximation is empty (no pivots)

### `pub fn evaluate(&self, i: usize, j: usize) -> T` (trait AbstractMatrixCI)

Evaluate the approximation at position (i, j)

### `pub fn submatrix(&self, rows: & [usize], cols: & [usize]) -> Matrix < T >` (trait AbstractMatrixCI)

Get a submatrix of the approximation

### `pub fn row(&self, i: usize) -> Vec < T >` (trait AbstractMatrixCI default)

Get a row of the approximation

### `pub fn col(&self, j: usize) -> Vec < T >` (trait AbstractMatrixCI default)

Get a column of the approximation

### `pub fn to_matrix(&self) -> Matrix < T >` (trait AbstractMatrixCI default)

Get the full approximated matrix

### `pub fn available_rows(&self) -> Vec < usize >` (trait AbstractMatrixCI default)

Get available row indices (rows without pivots)

### `pub fn available_cols(&self) -> Vec < usize >` (trait AbstractMatrixCI default)

Get available column indices (columns without pivots)

### `pub fn local_error(&self, a: & Matrix < T >, rows: & [usize], cols: & [usize]) -> Matrix < T >` (trait AbstractMatrixCI default)

Compute local error |A - CI| for given indices

### `pub fn find_new_pivot(&self, a: & Matrix < T >) -> Result < ((usize , usize) , T) >` (trait AbstractMatrixCI default)

Find a new pivot that maximizes the local error

### `pub fn find_new_pivot_in(&self, a: & Matrix < T >, rows: & [usize], cols: & [usize]) -> Result < ((usize , usize) , T) >` (trait AbstractMatrixCI default)

Find a new pivot in the given row/column subsets

## src/util.rs

### `pub fn from_elem(nrows: usize, ncols: usize, elem: T) -> Self` (impl Matrix < T >)

Create a new matrix from dimensions and initial value

### `pub fn nrows(&self) -> usize` (impl Matrix < T >)

Number of rows

### `pub fn ncols(&self) -> usize` (impl Matrix < T >)

Number of columns

### `pub fn zeros(nrows: usize, ncols: usize) -> Self` (impl Matrix < T >)

Create a zeros matrix

### ` fn index(&self, idx: [usize ; 2]) -> & Self :: Output` (impl Matrix < T >)

### ` fn index_mut(&mut self, idx: [usize ; 2]) -> & mut Self :: Output` (impl Matrix < T >)

### `pub fn zeros(nrows: usize, ncols: usize) -> Matrix < T >`

Create a zeros matrix with given dimensions

### `pub fn eye(n: usize) -> Matrix < T >`

Create an identity matrix

### `pub fn from_vec2d(data: Vec < Vec < T > >) -> Matrix < T >`

Create a matrix from a 2D vector (row-major)

### `pub fn nrows(m: & Matrix < T >) -> usize`

Get number of rows

### `pub fn ncols(m: & Matrix < T >) -> usize`

Get number of columns

### `pub fn get_row(m: & Matrix < T >, i: usize) -> Vec < T >`

Get a row as a vector

### `pub fn get_col(m: & Matrix < T >, j: usize) -> Vec < T >`

Get a column as a vector

### `pub fn submatrix(m: & Matrix < T >, rows: & [usize], cols: & [usize]) -> Matrix < T >`

Get a submatrix by selecting specific rows and columns

### `pub fn append_col(m: & Matrix < T >, col: & [T]) -> Matrix < T >`

Append a column to the right of a matrix

### `pub fn append_row(m: & Matrix < T >, row: & [T]) -> Matrix < T >`

Append a row to the bottom of a matrix

### `pub fn swap_rows(m: & Matrix < T >, a: usize, b: usize) -> Matrix < T >`

Swap two rows in a matrix (in-place style, returns new matrix)

### `pub fn swap_cols(m: & Matrix < T >, a: usize, b: usize) -> Matrix < T >`

Swap two columns in a matrix (in-place style, returns new matrix)

### `pub fn transpose(m: & Matrix < T >) -> Matrix < T >`

Transpose the matrix

### `pub fn abs(self) -> Self` (trait Scalar)

Absolute value

### `pub fn abs_sq(self) -> f64` (trait Scalar)

Square of absolute value (for complex numbers, |z|^2)

### `pub fn is_nan(self) -> bool` (trait Scalar)

Check if value is NaN

### `pub fn epsilon() -> f64` (trait Scalar default)

Small epsilon value for numerical comparisons

### ` fn abs(self) -> Self` (impl f64)

### ` fn abs_sq(self) -> f64` (impl f64)

### ` fn is_nan(self) -> bool` (impl f64)

### ` fn abs(self) -> Self` (impl f32)

### ` fn abs_sq(self) -> f64` (impl f32)

### ` fn is_nan(self) -> bool` (impl f32)

### ` fn abs(self) -> Self` (impl num_complex :: Complex64)

### ` fn abs_sq(self) -> f64` (impl num_complex :: Complex64)

### ` fn is_nan(self) -> bool` (impl num_complex :: Complex64)

### ` fn abs(self) -> Self` (impl num_complex :: Complex32)

### ` fn abs_sq(self) -> f64` (impl num_complex :: Complex32)

### ` fn is_nan(self) -> bool` (impl num_complex :: Complex32)

### `pub fn a_times_b_inv(a: & Matrix < T >, b: & Matrix < T >) -> Matrix < T >`

Calculates A * B^{-1} using Gaussian elimination for numerical stability.

### `pub fn a_inv_times_b(a: & Matrix < T >, b: & Matrix < T >) -> Matrix < T >`

Calculates A^{-1} * B using Gaussian elimination for numerical stability.

### ` fn solve_linear_system(a: & Matrix < T >, b: & Matrix < T >) -> Matrix < T >`

Solve linear system AX = B using Gaussian elimination with partial pivoting

### `pub fn submatrix_argmax(a: & Matrix < T >, rows: & [usize], cols: & [usize]) -> (usize , usize , T)`

Find the position of maximum absolute value in a submatrix

### `pub fn random_subset(set: & [T], n: usize, rng: & mut R) -> Vec < T >`

Select a random subset of elements from a set

### `pub fn set_diff(set: & [usize], exclude: & [usize]) -> Vec < usize >`

Set difference: elements in `set` that are not in `exclude`

### `pub fn dot(a: & [T], b: & [T]) -> T`

Dot product of two vectors

### `pub fn mat_mul(a: & Matrix < T >, b: & Matrix < T >) -> Matrix < T >`

Matrix multiplication: A * B

### ` fn test_matrix_basic()`

### ` fn test_matrix_transpose()`

### ` fn test_submatrix_argmax()`

### ` fn test_set_diff()`

### ` fn test_mat_mul()`

