use crate::scalar::AciScalar;
use crate::{initial_guess, AciError, AciOptions, Result};
use tensor4all_simplett::{
    tensor3_from_data, AbstractTensorTrain, Tensor3, Tensor3Ops, TensorTrain,
};
use tensor4all_tcicore::{matrix_luci_factors_from_matrix, RrLUOptions};
use tensor4all_tensorbackend::{mat_mul, Matrix};

pub(crate) struct ElementwiseProblem<T: AciScalar> {
    pub(crate) inputs: Vec<TensorTrain<T>>,
    pub(crate) solution: TensorTrain<T>,
    pub(crate) left_frames: Vec<Vec<Option<Matrix<T>>>>,
    pub(crate) right_frames: Vec<Vec<Option<Matrix<T>>>>,
    pub(crate) pivot_errors: Vec<f64>,
}

impl<T: AciScalar> ElementwiseProblem<T> {
    pub(crate) fn new(inputs: Vec<TensorTrain<T>>, options: AciOptions<T>) -> Result<Self> {
        let solution = initial_guess(&inputs, &options)?;
        let n = solution.len();
        let n_inputs = inputs.len();
        let mut left_frames = vec![vec![None; n + 1]; n_inputs];
        let mut right_frames = vec![vec![None; n + 1]; n_inputs];

        for input in 0..n_inputs {
            left_frames[input][0] = Some(unit_frame());
            right_frames[input][n] = Some(unit_frame());
        }

        let mut problem = Self {
            inputs,
            solution,
            left_frames,
            right_frames,
            pivot_errors: vec![0.0; n.saturating_sub(1)],
        };
        problem.initialize_right_frames()?;
        Ok(problem)
    }

    pub(crate) fn len(&self) -> usize {
        self.solution.len()
    }

    pub(crate) fn n_inputs(&self) -> usize {
        self.inputs.len()
    }

    pub(crate) fn left_frame_shape(&self, input: usize, site: usize) -> Option<(usize, usize)> {
        self.left_frames
            .get(input)
            .and_then(|frames| frames.get(site))
            .and_then(|frame| frame.as_ref())
            .map(|frame| (frame.nrows(), frame.ncols()))
    }

    pub(crate) fn right_frame_shape(&self, input: usize, site: usize) -> Option<(usize, usize)> {
        self.right_frames
            .get(input)
            .and_then(|frames| frames.get(site))
            .and_then(|frame| frame.as_ref())
            .map(|frame| (frame.nrows(), frame.ncols()))
    }

    pub(crate) fn left_frame_value(
        &self,
        input: usize,
        site: usize,
        row: usize,
        col: usize,
    ) -> Option<T> {
        frame_value(&self.left_frames, input, site, row, col)
    }

    pub(crate) fn right_frame_value(
        &self,
        input: usize,
        site: usize,
        row: usize,
        col: usize,
    ) -> Option<T> {
        frame_value(&self.right_frames, input, site, row, col)
    }

    pub(crate) fn local_input_shape(&self, input: usize, bond: usize) -> Result<(usize, usize)> {
        let context = self.local_input_context(input, bond)?;
        Ok((context.nrows, context.ncols))
    }

    pub(crate) fn local_input_value(
        &self,
        input: usize,
        bond: usize,
        row: usize,
        col: usize,
    ) -> Result<T> {
        let context = self.local_input_context(input, bond)?;
        if row >= context.nrows {
            return Err(AciError::InvalidInitialGuess {
                message: format!(
                    "local row index {row} out of bounds for bond {bond} with {} rows",
                    context.nrows
                ),
            });
        }
        if col >= context.ncols {
            return Err(AciError::InvalidInitialGuess {
                message: format!(
                    "local column index {col} out of bounds for bond {bond} with {} columns",
                    context.ncols
                ),
            });
        }

        let r_left = context.left_frame.nrows();
        let d_right = context.right_core.site_dim();
        let left_pivot = row % r_left;
        let site_left = row / r_left;
        let site_right = col % d_right;
        let right_pivot = col / d_right;

        let mut sum = T::zero();
        for a in 0..context.left_core.left_dim() {
            for m in 0..context.left_core.right_dim() {
                for b in 0..context.right_core.right_dim() {
                    sum = sum
                        + context.left_frame[[left_pivot, a]]
                            * *context.left_core.get3(a, site_left, m)
                            * *context.right_core.get3(m, site_right, b)
                            * context.right_frame[[b, right_pivot]];
                }
            }
        }
        Ok(sum)
    }

    pub(crate) fn update_left_frame(
        &mut self,
        input: usize,
        site: usize,
        row_indices: &[usize],
    ) -> Result<()> {
        let n = self.len();
        validate_input_site(input, site, self.n_inputs(), n)?;
        let source = self.left_frames[input][site]
            .as_ref()
            .ok_or_else(|| missing_frame("left", input, site))?
            .clone();
        let core = self.inputs[input].site_tensor(site);

        if source.ncols() != core.left_dim() {
            return Err(AciError::InvalidInitialGuess {
                message: format!(
                    "left frame/input bond mismatch at input {input}, site {site}: \
                     frame has {} columns, core left bond is {}",
                    source.ncols(),
                    core.left_dim()
                ),
            });
        }

        let full_rows = checked_frame_mul(source.nrows(), core.site_dim(), "left frame row count")?;
        validate_selection("row", row_indices, full_rows)?;

        let mut selected = Matrix::zeros(row_indices.len(), core.right_dim());
        for (selected_row, &full_row) in row_indices.iter().enumerate() {
            let source_row = full_row % source.nrows();
            let physical = full_row / source.nrows();
            for right in 0..core.right_dim() {
                let mut sum = T::zero();
                for left in 0..core.left_dim() {
                    sum = sum + source[[source_row, left]] * *core.get3(left, physical, right);
                }
                selected[[selected_row, right]] = sum;
            }
        }

        self.left_frames[input][site + 1] = Some(selected);
        Ok(())
    }

    pub(crate) fn update_right_frame(
        &mut self,
        input: usize,
        site: usize,
        col_indices: &[usize],
    ) -> Result<()> {
        let n = self.len();
        validate_input_site(input, site, self.n_inputs(), n)?;
        let source_site = site + 1;
        let source = self.right_frames[input][source_site]
            .as_ref()
            .ok_or_else(|| missing_frame("right", input, source_site))?
            .clone();
        let core = self.inputs[input].site_tensor(site);

        if core.right_dim() != source.nrows() {
            return Err(AciError::InvalidInitialGuess {
                message: format!(
                    "right frame/input bond mismatch at input {input}, site {site}: \
                     core right bond is {}, frame has {} rows",
                    core.right_dim(),
                    source.nrows()
                ),
            });
        }

        let full_cols =
            checked_frame_mul(core.site_dim(), source.ncols(), "right frame column count")?;
        validate_selection("column", col_indices, full_cols)?;

        let mut selected = Matrix::zeros(core.left_dim(), col_indices.len());
        for left in 0..core.left_dim() {
            for (selected_col, &full_col) in col_indices.iter().enumerate() {
                let physical = full_col % core.site_dim();
                let source_col = full_col / core.site_dim();
                let mut sum = T::zero();
                for right in 0..core.right_dim() {
                    sum = sum + *core.get3(left, physical, right) * source[[right, source_col]];
                }
                selected[[left, selected_col]] = sum;
            }
        }

        self.right_frames[input][site] = Some(selected);
        Ok(())
    }

    pub(crate) fn update_left_frames(&mut self, site: usize, row_indices: &[usize]) -> Result<()> {
        for input in 0..self.n_inputs() {
            self.update_left_frame(input, site, row_indices)?;
        }
        Ok(())
    }

    pub(crate) fn update_right_frames(&mut self, site: usize, col_indices: &[usize]) -> Result<()> {
        for input in 0..self.n_inputs() {
            self.update_right_frame(input, site, col_indices)?;
        }
        Ok(())
    }

    fn initialize_right_frames(&mut self) -> Result<()> {
        let n = self.solution.len();
        let mut solution_cores = self.solution.site_tensors().to_vec();

        for site in (1..n).rev() {
            let current = &solution_cores[site];
            let site_dim = current.site_dim();
            let right_dim = current.right_dim();
            let matrix = right_matrix_julia_order(current);
            let factors = matrix_luci_factors_from_matrix(
                &matrix,
                Some(RrLUOptions {
                    left_orthogonal: false,
                    rel_tol: 0.0,
                    abs_tol: 0.0,
                    max_rank: usize::MAX,
                }),
            )?;

            validate_selection(
                "column",
                &factors.col_indices,
                checked_frame_mul(site_dim, right_dim, "solution right matrix column count")?,
            )?;
            solution_cores[site] =
                right_factor_to_tensor3(&factors.right, factors.rank, site_dim, right_dim)?;

            let previous = &solution_cores[site - 1];
            let previous_left_dim = previous.left_dim();
            let previous_site_dim = previous.site_dim();
            let previous_matrix = left_matrix_julia_order(previous);
            let product = matmul_checked(&previous_matrix, &factors.left, site - 1)?;
            solution_cores[site - 1] =
                matrix_to_tensor3(&product, previous_left_dim, previous_site_dim, factors.rank)?;

            self.update_right_frames(site, &factors.col_indices)?;
        }

        self.solution = TensorTrain::new(solution_cores)?;
        Ok(())
    }

    fn local_input_context(&self, input: usize, bond: usize) -> Result<LocalInputContext<'_, T>> {
        let n = self.len();
        if input >= self.n_inputs() {
            return Err(AciError::InvalidInitialGuess {
                message: format!(
                    "input index {input} out of bounds for {} inputs",
                    self.n_inputs()
                ),
            });
        }
        if bond >= n.saturating_sub(1) {
            return Err(AciError::InvalidInitialGuess {
                message: format!("bond index {bond} out of bounds for tensor train length {n}"),
            });
        }

        let right_frame_site = bond + 2;
        let left_frame = self
            .left_frames
            .get(input)
            .and_then(|frames| frames.get(bond))
            .and_then(|frame| frame.as_ref())
            .ok_or_else(|| missing_frame("left", input, bond))?;
        let right_frame = self
            .right_frames
            .get(input)
            .and_then(|frames| frames.get(right_frame_site))
            .and_then(|frame| frame.as_ref())
            .ok_or_else(|| missing_frame("right", input, right_frame_site))?;
        let left_core = self.inputs[input].site_tensor(bond);
        let right_core = self.inputs[input].site_tensor(bond + 1);

        if left_frame.nrows() == 0
            || left_core.site_dim() == 0
            || right_core.site_dim() == 0
            || right_frame.ncols() == 0
        {
            return Err(AciError::InvalidInitialGuess {
                message: format!("local block at input {input}, bond {bond} has a zero dimension"),
            });
        }
        if left_frame.ncols() != left_core.left_dim() {
            return Err(AciError::InvalidInitialGuess {
                message: format!(
                    "left frame/input bond mismatch at input {input}, bond {bond}: \
                     frame has {} columns, left core left bond is {}",
                    left_frame.ncols(),
                    left_core.left_dim()
                ),
            });
        }
        if left_core.right_dim() != right_core.left_dim() {
            return Err(AciError::InvalidInitialGuess {
                message: format!(
                    "adjacent input core bond mismatch at input {input}, bond {bond}: \
                     left core right bond is {}, right core left bond is {}",
                    left_core.right_dim(),
                    right_core.left_dim()
                ),
            });
        }
        if right_core.right_dim() != right_frame.nrows() {
            return Err(AciError::InvalidInitialGuess {
                message: format!(
                    "right frame/input bond mismatch at input {input}, bond {bond}: \
                     right core right bond is {}, frame has {} rows",
                    right_core.right_dim(),
                    right_frame.nrows()
                ),
            });
        }

        let nrows = checked_frame_mul(
            left_frame.nrows(),
            left_core.site_dim(),
            "local block row count",
        )?;
        let ncols = checked_frame_mul(
            right_core.site_dim(),
            right_frame.ncols(),
            "local block column count",
        )?;

        Ok(LocalInputContext {
            left_frame,
            right_frame,
            left_core,
            right_core,
            nrows,
            ncols,
        })
    }
}

struct LocalInputContext<'a, T: AciScalar> {
    left_frame: &'a Matrix<T>,
    right_frame: &'a Matrix<T>,
    left_core: &'a Tensor3<T>,
    right_core: &'a Tensor3<T>,
    nrows: usize,
    ncols: usize,
}

fn unit_frame<T: AciScalar>() -> Matrix<T> {
    Matrix::from_col_major_vec(1, 1, vec![T::one()])
}

fn frame_value<T: AciScalar>(
    frames: &[Vec<Option<Matrix<T>>>],
    input: usize,
    site: usize,
    row: usize,
    col: usize,
) -> Option<T> {
    let frame = frames.get(input)?.get(site)?.as_ref()?;
    if row >= frame.nrows() || col >= frame.ncols() {
        return None;
    }
    Some(frame[[row, col]])
}

fn validate_input_site(input: usize, site: usize, n_inputs: usize, n_sites: usize) -> Result<()> {
    if input >= n_inputs {
        return Err(AciError::InvalidInitialGuess {
            message: format!("input index {input} out of bounds for {n_inputs} inputs"),
        });
    }
    if site >= n_sites {
        return Err(AciError::InvalidInitialGuess {
            message: format!("site index {site} out of bounds for tensor train length {n_sites}"),
        });
    }
    Ok(())
}

fn validate_selection(kind: &'static str, indices: &[usize], len: usize) -> Result<()> {
    for &index in indices {
        if index >= len {
            return Err(AciError::InvalidInitialGuess {
                message: format!("{kind} index {index} out of bounds for length {len}"),
            });
        }
    }
    Ok(())
}

fn checked_frame_mul(lhs: usize, rhs: usize, description: &str) -> Result<usize> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| AciError::InvalidInitialGuess {
            message: format!("{description} overflows usize"),
        })
}

fn missing_frame(kind: &'static str, input: usize, site: usize) -> AciError {
    AciError::InvalidInitialGuess {
        message: format!("{kind} frame for input {input}, site {site} is not initialized"),
    }
}

fn right_matrix_julia_order<T: AciScalar>(core: &Tensor3<T>) -> Matrix<T> {
    let left_dim = core.left_dim();
    let site_dim = core.site_dim();
    let right_dim = core.right_dim();
    let mut data = Vec::with_capacity(left_dim * site_dim * right_dim);
    for right in 0..right_dim {
        for site in 0..site_dim {
            for left in 0..left_dim {
                data.push(*core.get3(left, site, right));
            }
        }
    }
    Matrix::from_col_major_vec(left_dim, site_dim * right_dim, data)
}

fn left_matrix_julia_order<T: AciScalar>(core: &Tensor3<T>) -> Matrix<T> {
    let left_dim = core.left_dim();
    let site_dim = core.site_dim();
    let right_dim = core.right_dim();
    let mut data = Vec::with_capacity(left_dim * site_dim * right_dim);
    for right in 0..right_dim {
        for site in 0..site_dim {
            for left in 0..left_dim {
                data.push(*core.get3(left, site, right));
            }
        }
    }
    Matrix::from_col_major_vec(left_dim * site_dim, right_dim, data)
}

fn matrix_to_tensor3<T: AciScalar>(
    matrix: &Matrix<T>,
    left_dim: usize,
    site_dim: usize,
    right_dim: usize,
) -> Result<Tensor3<T>> {
    if matrix.nrows() != left_dim * site_dim || matrix.ncols() != right_dim {
        return Err(AciError::InvalidInitialGuess {
            message: format!(
                "cannot reshape matrix of shape ({}, {}) to tensor core ({left_dim}, \
                 {site_dim}, {right_dim})",
                matrix.nrows(),
                matrix.ncols()
            ),
        });
    }
    Ok(tensor3_from_data(
        matrix.as_col_major_slice().to_vec(),
        left_dim,
        site_dim,
        right_dim,
    )?)
}

fn right_factor_to_tensor3<T: AciScalar>(
    matrix: &Matrix<T>,
    left_dim: usize,
    site_dim: usize,
    right_dim: usize,
) -> Result<Tensor3<T>> {
    if matrix.nrows() != left_dim || matrix.ncols() != site_dim * right_dim {
        return Err(AciError::InvalidInitialGuess {
            message: format!(
                "cannot reshape right factor of shape ({}, {}) to tensor core ({left_dim}, \
                 {site_dim}, {right_dim})",
                matrix.nrows(),
                matrix.ncols()
            ),
        });
    }
    Ok(tensor3_from_data(
        matrix.as_col_major_slice().to_vec(),
        left_dim,
        site_dim,
        right_dim,
    )?)
}

fn matmul_checked<T: AciScalar>(
    left: &Matrix<T>,
    right: &Matrix<T>,
    site: usize,
) -> Result<Matrix<T>> {
    if left.ncols() != right.nrows() {
        return Err(AciError::InvalidInitialGuess {
            message: format!(
                "solution core update at site {site} has incompatible matrix shapes: \
                 left ({}, {}), right ({}, {})",
                left.nrows(),
                left.ncols(),
                right.nrows(),
                right.ncols()
            ),
        });
    }

    mat_mul(left, right).map_err(|err| AciError::InvalidInitialGuess {
        message: format!("solution core update at site {site} failed: {err}"),
    })
}
