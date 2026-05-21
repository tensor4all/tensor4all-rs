use std::cell::RefCell;
use std::collections::HashMap;

use crate::scalar::AciScalar;
use crate::{AciError, ElementwiseBatch, ElementwiseProblem, Result};

type LocalOperator<'a, T> =
    dyn for<'batch> Fn(ElementwiseBatch<'batch, T>, &mut [T]) -> Result<()> + 'a;

pub(crate) struct LocalBlockEvaluator<'a, T: AciScalar> {
    problem: &'a ElementwiseProblem<T>,
    operator: &'a LocalOperator<'a, T>,
    bond: usize,
    nrows: usize,
    ncols: usize,
    cache: RefCell<HashMap<usize, T>>,
    first_error: RefCell<Option<AciError>>,
    max_output_abs: RefCell<f64>,
}

impl<'a, T: AciScalar> LocalBlockEvaluator<'a, T> {
    pub(crate) fn new<F>(
        problem: &'a ElementwiseProblem<T>,
        bond: usize,
        operator: &'a F,
    ) -> Result<Self>
    where
        F: for<'batch> Fn(ElementwiseBatch<'batch, T>, &mut [T]) -> Result<()> + 'a,
    {
        if problem.n_inputs() == 0 {
            return Err(AciError::EmptyInputs);
        }

        let (nrows, ncols) = problem.local_input_shape(0, bond)?;
        for input in 1..problem.n_inputs() {
            let (input_nrows, input_ncols) = problem.local_input_shape(input, bond)?;
            if input_nrows != nrows || input_ncols != ncols {
                return Err(AciError::InvalidInitialGuess {
                    message: format!(
                        "local block shape mismatch at input {input}, bond {bond}: \
                         expected ({nrows}, {ncols}), got ({input_nrows}, {input_ncols})"
                    ),
                });
            }
        }

        Ok(Self {
            problem,
            operator,
            bond,
            nrows,
            ncols,
            cache: RefCell::new(HashMap::new()),
            first_error: RefCell::new(None),
            max_output_abs: RefCell::new(0.0),
        })
    }

    pub(crate) fn nrows(&self) -> usize {
        self.nrows
    }

    pub(crate) fn ncols(&self) -> usize {
        self.ncols
    }

    pub(crate) fn fill_local_block(
        &self,
        rows: &[usize],
        cols: &[usize],
        out: &mut [T],
    ) -> Result<()> {
        let n_points = checked_local_mul(rows.len(), cols.len(), "local block point count")?;
        if out.len() != n_points {
            return Err(AciError::LengthMismatch {
                expected: n_points,
                got: out.len(),
            });
        }
        validate_indices("row", rows, self.nrows)?;
        validate_indices("column", cols, self.ncols)?;
        if n_points == 0 {
            return Ok(());
        }

        let mut point_keys = Vec::with_capacity(n_points);
        let mut missing_by_key = HashMap::new();
        let mut missing_rows = Vec::new();
        let mut missing_cols = Vec::new();

        {
            let cache = self.cache.borrow();
            for (col_position, &col) in cols.iter().enumerate() {
                for (row_position, &row) in rows.iter().enumerate() {
                    let point = row_position + rows.len() * col_position;
                    let key = self.entry_key(row, col)?;
                    point_keys.push(key);
                    if let Some(&value) = cache.get(&key) {
                        out[point] = value;
                    } else if let std::collections::hash_map::Entry::Vacant(entry) =
                        missing_by_key.entry(key)
                    {
                        entry.insert(missing_rows.len());
                        missing_rows.push(row);
                        missing_cols.push(col);
                    }
                }
            }
        }

        if missing_rows.is_empty() {
            return Ok(());
        }

        let n_inputs = self.problem.n_inputs();
        let n_missing = missing_rows.len();
        let input_value_count =
            checked_local_mul(n_inputs, n_missing, "local batch input value count")?;
        let mut input_values = vec![T::zero(); input_value_count];
        for (point, (&row, &col)) in missing_rows.iter().zip(&missing_cols).enumerate() {
            for input in 0..n_inputs {
                input_values[input + n_inputs * point] =
                    self.problem.local_input_value(input, self.bond, row, col)?;
            }
        }

        let batch = ElementwiseBatch::new(&input_values, n_inputs, n_missing)?;
        let mut missing_output = vec![T::zero(); n_missing];
        (self.operator)(batch, &mut missing_output)?;
        self.record_output_scale(&missing_output);

        {
            let mut cache = self.cache.borrow_mut();
            for (&key, &missing_index) in &missing_by_key {
                cache.insert(key, missing_output[missing_index]);
            }
        }

        for (point, &key) in point_keys.iter().enumerate() {
            if let Some(&missing_index) = missing_by_key.get(&key) {
                out[point] = missing_output[missing_index];
            }
        }
        Ok(())
    }

    pub(crate) fn fill_local_block_or_zero(&self, rows: &[usize], cols: &[usize], out: &mut [T]) {
        if let Err(err) = self.fill_local_block(rows, cols, out) {
            self.record_error(err);
            out.fill(T::zero());
        }
    }

    pub(crate) fn take_error(&self) -> Option<AciError> {
        self.first_error.borrow_mut().take()
    }

    pub(crate) fn clear_cache(&self) {
        self.cache.borrow_mut().clear();
    }

    pub(crate) fn max_output_abs(&self) -> f64 {
        *self.max_output_abs.borrow()
    }

    fn record_error(&self, err: AciError) {
        let mut first_error = self.first_error.borrow_mut();
        if first_error.is_none() {
            *first_error = Some(err);
        }
    }

    fn entry_key(&self, row: usize, col: usize) -> Result<usize> {
        let col_offset = checked_local_mul(self.nrows, col, "local block cache key")?;
        col_offset
            .checked_add(row)
            .ok_or_else(|| AciError::InvalidInitialGuess {
                message: "local block cache key overflows usize".to_string(),
            })
    }

    fn record_output_scale(&self, values: &[T]) {
        let mut max_output_abs = self.max_output_abs.borrow_mut();
        for &value in values {
            *max_output_abs =
                max_output_abs.max(tensor4all_tcicore::MatrixLuciScalar::abs_val(value));
        }
    }
}

fn validate_indices(kind: &'static str, indices: &[usize], len: usize) -> Result<()> {
    for &index in indices {
        if index >= len {
            return Err(AciError::InvalidInitialGuess {
                message: format!("local {kind} index {index} out of bounds for length {len}"),
            });
        }
    }
    Ok(())
}

fn checked_local_mul(lhs: usize, rhs: usize, description: &str) -> Result<usize> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| AciError::InvalidInitialGuess {
            message: format!("{description} overflows usize"),
        })
}
