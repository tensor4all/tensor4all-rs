//! Options and traits for TreeTN operations.
//!
//! Provides:
//! - [`CanonicalizationOptions`]: Options for canonicalization
//! - [`TruncationOptions`]: Options for truncation
//! - [`SplitOptions`]: Options for split operations

use crate::algorithm::CanonicalForm;

/// Options for canonicalization operations.
///
/// # Builder Pattern
///
/// ```ignore
/// let options = CanonicalizationOptions::default()
///     .with_form(CanonicalForm::LU)
///     .force();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct CanonicalizationOptions {
    /// Canonical form to use (QR, LU, or CI)
    pub form: CanonicalForm,
    /// If true, always performs full canonicalization.
    /// If false, checks current state and may skip or optimize.
    pub force: bool,
}

impl Default for CanonicalizationOptions {
    fn default() -> Self {
        Self {
            form: CanonicalForm::Unitary,
            force: false,
        }
    }
}

impl CanonicalizationOptions {
    /// Create options with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create options that force full canonicalization.
    pub fn forced() -> Self {
        Self {
            form: CanonicalForm::Unitary,
            force: true,
        }
    }

    /// Set the canonical form.
    pub fn with_form(mut self, form: CanonicalForm) -> Self {
        self.form = form;
        self
    }

    /// Set force mode (always perform full canonicalization).
    pub fn force(mut self) -> Self {
        self.force = true;
        self
    }

    /// Disable force mode (check current state before canonicalizing).
    pub fn smart(mut self) -> Self {
        self.force = false;
        self
    }
}

/// Options for truncation operations.
///
/// # Builder Pattern
///
/// ```ignore
/// let options = TruncationOptions::default()
///     .with_max_rank(50)
///     .with_rtol(1e-10);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct TruncationOptions {
    /// Canonical form / algorithm to use (SVD, LU, or CI)
    pub form: CanonicalForm,
    /// Relative tolerance for truncation (keep singular values > rtol * max_sv)
    pub rtol: Option<f64>,
    /// Maximum bond dimension
    pub max_rank: Option<usize>,
}

impl Default for TruncationOptions {
    fn default() -> Self {
        Self {
            form: CanonicalForm::Unitary,
            rtol: None,
            max_rank: None,
        }
    }
}

impl TruncationOptions {
    /// Create options with default settings (no truncation limits).
    pub fn new() -> Self {
        Self::default()
    }

    /// Create options with a maximum rank.
    pub fn with_max_rank(mut self, rank: usize) -> Self {
        self.max_rank = Some(rank);
        self
    }

    /// Create options with a relative tolerance.
    pub fn with_rtol(mut self, rtol: f64) -> Self {
        self.rtol = Some(rtol);
        self
    }

    /// Set the canonical form / algorithm.
    pub fn with_form(mut self, form: CanonicalForm) -> Self {
        self.form = form;
        self
    }
}

/// Options for split operations.
///
/// # Builder Pattern
///
/// ```ignore
/// let options = SplitOptions::default()
///     .with_max_rank(50)
///     .with_rtol(1e-10)
///     .with_final_sweep(true);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SplitOptions {
    /// Canonical form / algorithm to use (SVD, QR, etc.)
    pub form: CanonicalForm,
    /// Relative tolerance for truncation (keep singular values > rtol * max_sv)
    pub rtol: Option<f64>,
    /// Maximum bond dimension
    pub max_rank: Option<usize>,
    /// Whether to perform a final sweep for global bond dimension optimization
    pub final_sweep: bool,
}

impl Default for SplitOptions {
    fn default() -> Self {
        Self {
            form: CanonicalForm::Unitary,
            rtol: None,
            max_rank: None,
            final_sweep: false,
        }
    }
}

impl SplitOptions {
    /// Create options with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create options with a maximum rank.
    pub fn with_max_rank(mut self, rank: usize) -> Self {
        self.max_rank = Some(rank);
        self
    }

    /// Create options with a relative tolerance.
    pub fn with_rtol(mut self, rtol: f64) -> Self {
        self.rtol = Some(rtol);
        self
    }

    /// Set the canonical form / algorithm.
    pub fn with_form(mut self, form: CanonicalForm) -> Self {
        self.form = form;
        self
    }

    /// Enable or disable final sweep for global optimization.
    pub fn with_final_sweep(mut self, final_sweep: bool) -> Self {
        self.final_sweep = final_sweep;
        self
    }
}
