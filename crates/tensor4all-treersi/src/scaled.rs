//! A common binary scale for each dense block; no scale is discarded.

use crate::{
    dense::Dense,
    scalar::{loses_component, magnitude, product_underflows, scale},
    Result, TreeRsiError, TreeRsiScalar,
};

#[derive(Clone, Debug)]
pub(crate) struct Scaled<T> {
    pub(crate) array: Dense<T>,
    pub(crate) exponent: i64,
}

impl<T: TreeRsiScalar> Scaled<T> {
    pub(crate) fn new(array: Dense<T>, exponent: i64) -> Result<Self> {
        let mut value = Self { array, exponent };
        value.normalize()?;
        Ok(value)
    }

    fn normalize(&mut self) -> Result<()> {
        let mut maximum = 0.0_f64;
        for &x in &self.array.data {
            let a = magnitude(x);
            if !a.is_finite() {
                return Err(TreeRsiError::NonFiniteValue {
                    node: "dense block".into(),
                    stage: "normalization",
                });
            }
            maximum = maximum.max(a);
        }
        if maximum == 0.0 {
            self.exponent = 0;
            return Ok(());
        }
        let exponent = maximum.log2().floor() as i64;
        for x in &mut self.array.data {
            let y = scale(*x, -exponent);
            // `loses_component` only rejects underflow to zero; a component
            // scaled into the subnormal range may still round to a different
            // nonzero value. The block-scale contract is exact only while
            // every scaled component stays normal.
            if loses_component(*x, y) {
                return Err(TreeRsiError::DynamicRange {
                    stage: "block normalization",
                });
            }
            *x = y;
        }
        self.exponent = add_exponents(self.exponent, exponent)?;
        Ok(())
    }

    pub(crate) fn into_values(self) -> Result<Vec<T>> {
        self.array
            .data
            .into_iter()
            .map(|x| {
                let y = scale(x, self.exponent);
                if !magnitude(y).is_finite() || loses_component(x, y) {
                    Err(TreeRsiError::DynamicRange {
                        stage: "output conversion",
                    })
                } else {
                    Ok(y)
                }
            })
            .collect()
    }

    pub(crate) fn multiply(mut self, other: &Self) -> Result<Self> {
        if self.array.dims != other.array.dims {
            return Err(TreeRsiError::InternalInvariant {
                message: "product block shapes differ",
            });
        }
        for (x, &y) in self.array.data.iter_mut().zip(&other.array.data) {
            let z = *x * y;
            if product_underflows(*x, y) {
                return Err(TreeRsiError::DynamicRange {
                    stage: "local product",
                });
            }
            *x = z;
        }
        self.exponent = add_exponents(self.exponent, other.exponent)?;
        self.normalize()?;
        Ok(self)
    }
}

pub(crate) fn add_exponents(a: i64, b: i64) -> Result<i64> {
    a.checked_add(b).ok_or(TreeRsiError::SizeOverflow {
        context: "binary scale exponent",
    })
}

#[cfg(test)]
mod tests;
