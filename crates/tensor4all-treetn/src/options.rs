//! Options and traits for TreeTN operations.
//!
//! Provides:
//! - [`CanonicalizationOptions`]: Options for canonicalization
//! - [`TruncationOptions`]: Options for truncation
//! - [`SplitOptions`]: Options for split operations

use crate::algorithm::CanonicalForm;
use tensor4all_core::truncation::{HasTruncationParams, TruncationParams};

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
    /// Truncation parameters (rtol, max_rank).
    pub truncation: TruncationParams,
}

impl Default for TruncationOptions {
    fn default() -> Self {
        Self {
            form: CanonicalForm::Unitary,
            truncation: TruncationParams::default(),
        }
    }
}

impl HasTruncationParams for TruncationOptions {
    fn truncation_params(&self) -> &TruncationParams {
        &self.truncation
    }

    fn truncation_params_mut(&mut self) -> &mut TruncationParams {
        &mut self.truncation
    }
}

impl TruncationOptions {
    /// Create options with default settings (no truncation limits).
    pub fn new() -> Self {
        Self::default()
    }

    /// Create options with a maximum rank.
    pub fn with_max_rank(mut self, rank: usize) -> Self {
        self.truncation.max_rank = Some(rank);
        self
    }

    /// Create options with a relative tolerance.
    pub fn with_rtol(mut self, rtol: f64) -> Self {
        self.truncation.rtol = Some(rtol);
        self
    }

    /// Set the canonical form / algorithm.
    pub fn with_form(mut self, form: CanonicalForm) -> Self {
        self.form = form;
        self
    }

    /// Get rtol (for backwards compatibility).
    pub fn rtol(&self) -> Option<f64> {
        self.truncation.rtol
    }

    /// Get max_rank (for backwards compatibility).
    pub fn max_rank(&self) -> Option<usize> {
        self.truncation.max_rank
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
    /// Truncation parameters (rtol, max_rank).
    pub truncation: TruncationParams,
    /// Whether to perform a final sweep for global bond dimension optimization
    pub final_sweep: bool,
}

impl Default for SplitOptions {
    fn default() -> Self {
        Self {
            form: CanonicalForm::Unitary,
            truncation: TruncationParams::default(),
            final_sweep: false,
        }
    }
}

impl HasTruncationParams for SplitOptions {
    fn truncation_params(&self) -> &TruncationParams {
        &self.truncation
    }

    fn truncation_params_mut(&mut self) -> &mut TruncationParams {
        &mut self.truncation
    }
}

impl SplitOptions {
    /// Create options with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create options with a maximum rank.
    pub fn with_max_rank(mut self, rank: usize) -> Self {
        self.truncation.max_rank = Some(rank);
        self
    }

    /// Create options with a relative tolerance.
    pub fn with_rtol(mut self, rtol: f64) -> Self {
        self.truncation.rtol = Some(rtol);
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

    /// Get rtol (for backwards compatibility).
    pub fn rtol(&self) -> Option<f64> {
        self.truncation.rtol
    }

    /// Get max_rank (for backwards compatibility).
    pub fn max_rank(&self) -> Option<usize> {
        self.truncation.max_rank
    }
}
