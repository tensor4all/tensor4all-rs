//! Index integer trait for multi-index elements.
//!
//! Use smaller types (e.g., `u8`) for quantics TCI where `local_dim = 2`.

/// Trait for index element types.
///
/// All indices within a single `CachedFunction` must use the same integer type.
pub trait IndexInt: Copy + Send + Sync + 'static {
    /// Convert this index element to `usize`.
    fn to_usize(self) -> usize;
}

impl IndexInt for u8 {
    fn to_usize(self) -> usize {
        self as usize
    }
}

impl IndexInt for u16 {
    fn to_usize(self) -> usize {
        self as usize
    }
}

impl IndexInt for u32 {
    fn to_usize(self) -> usize {
        self as usize
    }
}

impl IndexInt for usize {
    fn to_usize(self) -> usize {
        self
    }
}
