//! Shared element-count budget for dense materialization in production paths.
//!
//! Production algorithms that must build a dense local object (an eigenproblem
//! matrix, an embedded operator site, ...) check its size here before
//! allocating. The unit is elements, not bytes: bytes = elements ×
//! `size_of::<T>()`.

use thiserror::Error;

/// Default dense-element budget: `1 << 24` elements, i.e. 256 MiB per dense
/// `Complex64` object.
///
/// The measured TreeTN GSE peak at `q_dim = 4096` (`q_dim² = 1 << 24`) is
/// 1.9 GiB and 44 s on one machine, so this default admits that case and
/// rejects `q_dim = 4097`. Callers with a tighter memory bound pass their own
/// [`DenseElementBudget`] instead.
pub const DEFAULT_MAX_DENSE_ELEMENTS: usize = 1 << 24;

/// Element-count budget for one dense materialization.
///
/// A budget is checked against the element count of a dense object (or the
/// total of several) before the allocation. Counts equal to the limit pass;
/// only counts above it fail.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DenseElementBudget, DEFAULT_MAX_DENSE_ELEMENTS};
///
/// assert_eq!(
///     DenseElementBudget::default().max_elements(),
///     DEFAULT_MAX_DENSE_ELEMENTS
/// );
/// assert_eq!(DenseElementBudget::unbounded().max_elements(), usize::MAX);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DenseElementBudget {
    max_elements: usize,
}

impl DenseElementBudget {
    /// Create a budget that admits at most `max_elements` elements.
    ///
    /// `new(0)` admits only empty objects (an extent of `0` or no objects at
    /// all).
    pub const fn new(max_elements: usize) -> Self {
        Self { max_elements }
    }

    /// Create a budget that admits every object whose element count fits in
    /// `usize`.
    pub const fn unbounded() -> Self {
        Self::new(usize::MAX)
    }

    /// The maximum element count this budget admits.
    pub const fn max_elements(self) -> usize {
        self.max_elements
    }

    /// Element count of the dense object with `dims`, checked against the
    /// budget.
    ///
    /// `dims` is the extent of each axis; an empty slice is a scalar (1
    /// element) and any extent of `0` is an empty object (0 elements).
    ///
    /// # Errors
    ///
    /// Returns [`DenseBudgetError::Overflow`] when the element count does not
    /// fit in `usize`, and [`DenseBudgetError::Exceeded`] when it is above
    /// [`Self::max_elements`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DenseBudgetError, DenseElementBudget};
    ///
    /// let budget = DenseElementBudget::new(4);
    /// assert_eq!(budget.check_dims("density", &[2, 2]).unwrap(), 4);
    /// assert_eq!(budget.check_dims("scalar", &[]).unwrap(), 1);
    /// assert_eq!(budget.check_dims("empty", &[0, 9]).unwrap(), 0);
    /// assert_eq!(
    ///     budget.check_dims("density", &[2, 3]),
    ///     Err(DenseBudgetError::Exceeded {
    ///         what: "density",
    ///         elements: 6,
    ///         max_elements: 4,
    ///     })
    /// );
    /// ```
    pub fn check_dims(self, what: &'static str, dims: &[usize]) -> Result<usize, DenseBudgetError> {
        self.check_total(what, std::iter::once(dims))
    }

    /// Total element count of several dense objects, checked against the
    /// budget once.
    ///
    /// Use this when one operation allocates a list of dense objects whose sum,
    /// not any single object, is the cost.
    ///
    /// # Errors
    ///
    /// Returns [`DenseBudgetError::Overflow`] when the total does not fit in
    /// `usize`, and [`DenseBudgetError::Exceeded`] when it is above
    /// [`Self::max_elements`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::DenseElementBudget;
    ///
    /// let budget = DenseElementBudget::new(8);
    /// let sites = [[1usize, 4, 1].as_slice(), [1, 4, 1].as_slice()];
    /// assert_eq!(budget.check_total("mpo", sites).unwrap(), 8);
    /// assert_eq!(
    ///     budget.check_total("mpo", std::iter::once(sites[0])).unwrap(),
    ///     4
    /// );
    /// assert!(DenseElementBudget::new(7).check_total("mpo", sites).is_err());
    /// ```
    pub fn check_total<'a>(
        self,
        what: &'static str,
        objects: impl IntoIterator<Item = &'a [usize]>,
    ) -> Result<usize, DenseBudgetError> {
        let mut elements = 0usize;
        for dims in objects {
            let count = element_count(what, dims)?;
            elements = elements
                .checked_add(count)
                .ok_or(DenseBudgetError::Overflow { what })?;
        }
        if elements > self.max_elements {
            return Err(DenseBudgetError::Exceeded {
                what,
                elements,
                max_elements: self.max_elements,
            });
        }
        Ok(elements)
    }
}

impl Default for DenseElementBudget {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_DENSE_ELEMENTS)
    }
}

/// Element count of one dense object, checked against `usize`.
///
/// A zero extent wins over the product so that an empty object never reports
/// an overflow.
fn element_count(what: &'static str, dims: &[usize]) -> Result<usize, DenseBudgetError> {
    if dims.contains(&0) {
        return Ok(0);
    }
    dims.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or(DenseBudgetError::Overflow { what })
    })
}

/// Error returned by a [`DenseElementBudget`] check.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum DenseBudgetError {
    /// The element count of `what` does not fit in `usize`.
    #[error("{what}: dense element count overflows usize")]
    Overflow {
        /// The dense object or group that was checked.
        what: &'static str,
    },
    /// The element count of `what` is above the configured budget.
    #[error(
        "{what}: dense materialization of {elements} elements exceeds the budget of {max_elements} elements"
    )]
    Exceeded {
        /// The dense object or group that was checked.
        what: &'static str,
        /// The element count that was requested.
        elements: usize,
        /// The element count the budget admits.
        max_elements: usize,
    },
}

#[cfg(test)]
mod tests;
