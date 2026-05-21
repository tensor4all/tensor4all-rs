use crate::{AciError, Result};
use tensor4all_simplett::{AbstractTensorTrain, TTScalar, TensorTrain};

pub(crate) fn validate_options<T: TTScalar>(options: &crate::AciOptions<T>) -> Result<()> {
    if options.max_iters == 0 {
        return Err(AciError::InvalidOptions {
            message: "max_iters must be at least 1".to_string(),
        });
    }

    if options.max_bond_dim == 0 {
        return Err(AciError::InvalidOptions {
            message: "max_bond_dim must be at least 1".to_string(),
        });
    }

    if options.min_iters > options.max_iters {
        return Err(AciError::InvalidOptions {
            message: format!(
                "min_iters ({}) must be less than or equal to max_iters ({})",
                options.min_iters, options.max_iters
            ),
        });
    }

    if !options.tolerance.is_finite() || options.tolerance < 0.0 {
        return Err(AciError::InvalidOptions {
            message: "tolerance must be finite and non-negative".to_string(),
        });
    }

    Ok(())
}

pub(crate) fn validate_inputs<T: TTScalar>(inputs: &[TensorTrain<T>]) -> Result<Vec<usize>> {
    let Some(first) = inputs.first() else {
        return Err(AciError::EmptyInputs);
    };

    let site_dims = first.site_dims();
    let expected_len = site_dims.len();
    if expected_len == 0 {
        return Err(AciError::InvalidOptions {
            message: "input tensor trains must have at least one site".to_string(),
        });
    }

    for input in &inputs[1..] {
        if input.len() != expected_len {
            return Err(AciError::LengthMismatch {
                expected: expected_len,
                got: input.len(),
            });
        }

        for (site, (expected, got)) in site_dims.iter().zip(input.site_dims()).enumerate() {
            if *expected != got {
                return Err(AciError::SiteDimMismatch {
                    site,
                    expected: *expected,
                    got,
                });
            }
        }
    }

    Ok(site_dims)
}
