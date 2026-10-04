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
mod tests {
    use super::*;
    fn block(data: Vec<f64>, exponent: i64) -> Result<Scaled<f64>> {
        Scaled::new(
            Dense {
                dims: vec![data.len()],
                data,
            },
            exponent,
        )
    }
    #[test]
    fn scale_roundtrips_extremes_and_preserves_column_ratios() {
        for x in [
            f64::from_bits(1),
            f64::MIN_POSITIVE,
            1e-200,
            1.0,
            1e200,
            f64::MAX,
        ] {
            assert_eq!(
                block(vec![x, -x], 0).unwrap().into_values().unwrap(),
                vec![x, -x]
            );
        }
        let x = block(vec![2.0, 8.0], 500).unwrap();
        assert_eq!(x.array.data[1] / x.array.data[0], 4.0);
        let y = block(vec![8.0, 2.0], -500).unwrap();
        assert_eq!(
            x.multiply(&y).unwrap().into_values().unwrap(),
            vec![16.0, 16.0]
        );
        assert_eq!(block(vec![0.0], i64::MAX).unwrap().exponent, 0);
    }
    #[test]
    fn lost_range_overflow_and_shape_mismatch_are_errors() {
        assert!(matches!(
            block(vec![f64::from_bits(1), f64::MAX], 0),
            Err(TreeRsiError::DynamicRange { .. })
        ));
        assert!(matches!(
            add_exponents(i64::MAX, 1),
            Err(TreeRsiError::SizeOverflow { .. })
        ));
        let a = block(vec![1e-200, 1.0], 0).unwrap();
        assert!(matches!(
            a.clone().multiply(&a),
            Err(TreeRsiError::DynamicRange { .. })
        ));
        assert!(matches!(
            a.multiply(&block(vec![1.0], 0).unwrap()),
            Err(TreeRsiError::InternalInvariant { .. })
        ));
    }
    #[test]
    fn complex_scaling_reports_loss_of_either_component() {
        macro_rules! check {
            ($complex:ty, $real:ty) => {
                for value in [
                    <$complex>::new(<$real>::from_bits(1), <$real>::MAX),
                    <$complex>::new(<$real>::MAX, <$real>::from_bits(1)),
                ] {
                    assert!(matches!(
                        Scaled::new(
                            Dense {
                                dims: vec![1],
                                data: vec![value]
                            },
                            0
                        ),
                        Err(TreeRsiError::DynamicRange { .. })
                    ));
                }
                let value = Scaled {
                    array: Dense {
                        dims: vec![1],
                        data: vec![<$complex>::new(<$real>::from_bits(1), 1.0)],
                    },
                    exponent: -1,
                };
                assert!(matches!(
                    value.into_values(),
                    Err(TreeRsiError::DynamicRange { .. })
                ));
            };
        }
        check!(num_complex::Complex32, f32);
        check!(num_complex::Complex64, f64);
    }
}
