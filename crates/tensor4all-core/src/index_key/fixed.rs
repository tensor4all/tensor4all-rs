//! Fixed-width bit-packed key arms.
//!
//! Each helper packs an already-validated multi-index into the bit fields its
//! indexer reserved. The caller has checked that every value is below its
//! local dimension, so a value never carries out of its field, and the
//! indexer guarantees that every field lies below the arm's width (a
//! singleton field encodes zero at offset zero).
//!
//! Arms wider than 64 bits pack into plain `u64` digits on the stack and
//! convert to the `u128`/`bnum` type once, at the end. Shifting a `bnum` value
//! per component, as this module did before #1224, touches every digit of the
//! number for every component; on the `TTCache` pure-hit path that shift was
//! 84% of the call at 150 bits on an x86 host. A [`DigitLayout`] computed once
//! per indexer tells the packer which components start in which digit, so each
//! digit is filled by the same branch-free shift-or loop as the `u64` arm, and
//! the rare field that straddles two digits gets its high part afterwards.
//!
//! Every helper is `#[inline]`, and that is load-bearing rather than
//! decorative: they are called from `FlatIndexer::encode`, and without the
//! attribute whether they inline depends on codegen-unit partitioning (#642
//! cost the old `bnum` arms a factor of about three through that alone).
//!
//! Shift amounts are masked to 63. Under the invariant above the mask never
//! changes a non-zero value's shift (a singleton's zero value is unaffected by
//! any shift); it keeps the loops free of panicking shifts and per-component
//! checked conversions, which is what lets them vectorize.

/// The high part of a field that starts in one digit and ends in the next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Straddler {
    /// Position of the component in the multi-index.
    component: usize,
    /// Digit receiving the high part (the one after the field's start).
    digit: usize,
    /// Right shift that moves the high part to bit zero: `64 - start_bit`.
    shift: u32,
}

/// Which components start in which 64-bit digit of a packed key.
///
/// Components whose field starts in digit `d` are `starts[d]..starts[d + 1]`.
/// A singleton component can fall in any range: its value is always zero, so
/// it contributes nothing wherever it is packed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct DigitLayout {
    starts: Vec<usize>,
    straddlers: Vec<Straddler>,
}

impl DigitLayout {
    /// Layout of `width_bits` worth of fields at `offsets` with `field_bits`.
    ///
    /// `offsets` are the indexer's offsets, increasing over non-singleton
    /// fields; singleton fields are skipped, whatever their offset.
    pub(super) fn new(offsets: &[u64], field_bits: &[u32], width_bits: u64) -> Self {
        let digits = usize::try_from(width_bits.div_ceil(64)).unwrap_or(usize::MAX);
        let mut starts = Vec::with_capacity(digits.saturating_add(1));
        starts.push(0);
        let mut straddlers = Vec::new();
        for (component, (&offset, &bits)) in offsets.iter().zip(field_bits).enumerate() {
            if bits == 0 {
                continue;
            }
            let digit = (offset / 64) as usize;
            while starts.len() <= digit {
                starts.push(component);
            }
            let start_bit = (offset % 64) as u32;
            if start_bit + bits > 64 {
                straddlers.push(Straddler {
                    component,
                    digit: digit + 1,
                    shift: 64 - start_bit,
                });
            }
        }
        while starts.len() <= digits {
            starts.push(offsets.len());
        }
        Self { starts, straddlers }
    }
}

/// Packs `idx` at `offsets` into a 64-bit key.
#[inline]
pub(super) fn pack_u64(idx: &[usize], offsets: &[u64]) -> u64 {
    let mut key = 0u64;
    for (&value, &offset) in idx.iter().zip(offsets) {
        key |= (value as u64) << (offset & 63);
    }
    key
}

/// Packs `idx` at `offsets` into `N` little-endian 64-bit digits.
///
/// `layout` must have been built for the same offsets with a width of at most
/// `64 * N` bits.
#[inline]
pub(super) fn pack_digits<const N: usize>(
    idx: &[usize],
    offsets: &[u64],
    layout: &DigitLayout,
) -> [u64; N] {
    let mut digits = [0u64; N];
    for (digit, bounds) in digits.iter_mut().zip(layout.starts.windows(2)) {
        let range = bounds[0]..bounds[1];
        *digit = pack_u64(&idx[range.clone()], &offsets[range]);
    }
    for straddler in &layout.straddlers {
        digits[straddler.digit] |= (idx[straddler.component] as u64) >> straddler.shift;
    }
    digits
}
