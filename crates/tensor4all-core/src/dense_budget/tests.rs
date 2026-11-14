use super::{DenseBudgetError, DenseElementBudget, DEFAULT_MAX_DENSE_ELEMENTS};

#[test]
fn equality_passes_and_one_over_fails() {
    assert_eq!(DenseElementBudget::new(4).check_dims("x", &[2, 2]), Ok(4));
    assert_eq!(
        DenseElementBudget::new(3).check_dims("x", &[2, 2]),
        Err(DenseBudgetError::Exceeded {
            what: "x",
            elements: 4,
            max_elements: 3,
        })
    );
}

#[test]
fn zero_extent_is_empty_even_under_overflowing_dims() {
    assert_eq!(
        DenseElementBudget::new(0).check_dims("x", &[usize::MAX, 2, 0]),
        Ok(0)
    );
    assert_eq!(
        DenseElementBudget::new(0).check_total("x", std::iter::empty::<&[usize]>()),
        Ok(0)
    );
}

#[test]
fn overflow_is_an_error() {
    assert_eq!(
        DenseElementBudget::unbounded().check_dims("x", &[usize::MAX, 2]),
        Err(DenseBudgetError::Overflow { what: "x" })
    );
    assert_eq!(
        DenseElementBudget::unbounded().check_total("x", [[usize::MAX].as_slice(), [2].as_slice()]),
        Err(DenseBudgetError::Overflow { what: "x" })
    );
    assert_eq!(DenseElementBudget::unbounded().check_dims("x", &[]), Ok(1));
    assert_eq!(
        DenseElementBudget::unbounded().check_dims("x", &[usize::MAX]),
        Ok(usize::MAX)
    );
    assert_eq!(
        DenseElementBudget::default().max_elements(),
        DEFAULT_MAX_DENSE_ELEMENTS
    );
}
