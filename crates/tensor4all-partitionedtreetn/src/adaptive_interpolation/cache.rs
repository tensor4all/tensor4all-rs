//! Per-patch evaluation cache with variable-length packed keys.
//!
//! A key packs the active coordinates of one point into `u64` words. Each
//! coordinate of a site with dimension `d` takes `bits(d - 1)` bits (none for
//! `d = 1`) and never straddles two words, so any domain size is supported
//! without a width limit.
//!
//! Keys of at most one or two words are stored inline as `u64` or `u128`;
//! wider keys are boxed slices. Lookups encode into a reused buffer and
//! borrow it, so a cache hit allocates nothing, and every map hashes with
//! the unseeded [`WordHasher`] instead of SipHash.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hash, Hasher};

use tensor4all_core::{ColMajorArrayRef, CommonScalar, TensorElement};

/// The `width` low bits set; `width` is in `1..=64`.
fn low_bits(width: u32) -> u64 {
    u64::MAX >> (u64::BITS - width)
}

/// A coordinate outside its site dimension.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct OutOfRange {
    pub(super) slot: usize,
    pub(super) value: usize,
    pub(super) dim: usize,
}

/// Packing of the active coordinates of a patch into key words.
#[derive(Clone, Debug)]
pub(super) struct KeyLayout {
    dims: Vec<usize>,
    /// `(word, shift, width)` of every active slot.
    slots: Vec<(usize, u32, u32)>,
    n_words: usize,
}

impl KeyLayout {
    /// Lay out the active coordinates with the given dimensions (all positive).
    pub(super) fn new(dims: Vec<usize>) -> Self {
        let mut slots = Vec::with_capacity(dims.len());
        let mut word = 0usize;
        let mut used = 0u32;
        for &dim in &dims {
            let width = usize::BITS - dim.saturating_sub(1).leading_zeros();
            if used + width > u64::BITS {
                word += 1;
                used = 0;
            }
            slots.push((word, used, width));
            used += width;
        }
        let n_words = if used == 0 { word } else { word + 1 };
        Self {
            dims,
            slots,
            n_words,
        }
    }

    /// Dimensions of the active slots.
    #[cfg(test)]
    pub(super) fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Number of key words.
    pub(super) fn n_words(&self) -> usize {
        self.n_words
    }

    /// Pack one point given in active-slot order into `key`, which has
    /// [`Self::n_words`] words.
    pub(super) fn encode_into(&self, coords: impl Iterator<Item = usize>, key: &mut [u64]) {
        key.fill(0);
        for (&(word, shift, width), value) in self.slots.iter().zip(coords) {
            if width > 0 {
                key[word] |= (value as u64) << shift;
            }
        }
    }

    /// Pack one point given in active-slot order.
    pub(super) fn encode(&self, coords: impl Iterator<Item = usize>) -> Box<[u64]> {
        let mut key = vec![0u64; self.n_words].into_boxed_slice();
        self.encode_into(coords, &mut key);
        key
    }

    /// Check one point (one coordinate per active slot) against the slot
    /// dimensions and pack it into `key`, which has [`Self::n_words`] words.
    /// On error `key` holds a partial encoding.
    pub(super) fn encode_checked(
        &self,
        coords: &[usize],
        key: &mut [u64],
    ) -> Result<(), OutOfRange> {
        key.fill(0);
        for (slot, ((&(word, shift, width), &dim), &value)) in
            self.slots.iter().zip(&self.dims).zip(coords).enumerate()
        {
            if value >= dim {
                return Err(OutOfRange { slot, value, dim });
            }
            if width > 0 {
                key[word] |= (value as u64) << shift;
            }
        }
        Ok(())
    }

    /// Unpack one key into active-slot coordinates.
    pub(super) fn decode(&self, key: &[u64]) -> Vec<usize> {
        self.slots
            .iter()
            .map(|&(word, shift, width)| {
                if width == 0 {
                    0
                } else {
                    ((key[word] >> shift) & low_bits(width)) as usize
                }
            })
            .collect()
    }

    /// The layout without active slot `slot`, keeping every other slot at its
    /// word and shift, and the mask of the bits to keep in every key word.
    ///
    /// A key of this layout is a parent key with the bits of `slot` cleared,
    /// so splitting a cache moves the key words instead of re-encoding them.
    fn without_slot(&self, slot: usize) -> (Self, Vec<u64>) {
        let mut keep = vec![u64::MAX; self.n_words];
        let (word, shift, width) = self.slots[slot];
        if width > 0 {
            keep[word] &= !(low_bits(width) << shift);
        }
        let mut dims = self.dims.clone();
        dims.remove(slot);
        let mut slots = self.slots.clone();
        slots.remove(slot);
        let layout = Self {
            dims,
            slots,
            n_words: self.n_words,
        };
        (layout, keep)
    }

    /// The coordinate of active slot `slot`, read from `word`, the key word
    /// that holds the slot.
    fn coordinate(&self, slot: usize, word: u64) -> usize {
        let (_, shift, width) = self.slots[slot];
        if width == 0 {
            0
        } else {
            ((word >> shift) & low_bits(width)) as usize
        }
    }
}

/// An unseeded, non-cryptographic hasher for packed key words.
///
/// Each word is folded in by a rotate, xor, and multiply, and `finish`
/// applies the MurmurHash3 64-bit finalizer so every output bit depends on
/// every input bit (the map takes its bucket index from the low bits). The
/// keys are coordinates chosen by the driver and the engine, not by an
/// adversary, so the HashDoS protection of SipHash is not needed; being
/// unseeded, the hasher also keeps map layouts identical across runs, although
/// nothing observes the iteration order of a cache.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct WordHasher {
    state: u64,
}

impl Hasher for WordHasher {
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            let mut word = [0u8; 8];
            word.copy_from_slice(chunk);
            self.write_u64(u64::from_le_bytes(word));
        }
        let rest = chunks.remainder();
        if !rest.is_empty() {
            let mut word = [0u8; 8];
            word[..rest.len()].copy_from_slice(rest);
            self.write_u64(u64::from_le_bytes(word));
        }
    }

    fn write_u64(&mut self, word: u64) {
        self.state = (self.state.rotate_left(26) ^ word).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }

    fn write_u128(&mut self, value: u128) {
        self.write_u64(value as u64);
        self.write_u64((value >> 64) as u64);
    }

    fn write_usize(&mut self, value: usize) {
        self.write_u64(value as u64);
    }

    fn finish(&self) -> u64 {
        let mut hash = self.state;
        hash ^= hash >> 33;
        hash = hash.wrapping_mul(0xff51_afd7_ed55_8ccd);
        hash ^= hash >> 33;
        hash = hash.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        hash ^ (hash >> 33)
    }
}

/// A map keyed by packed key words.
pub(super) type WordMap<K, V> = HashMap<K, V, BuildHasherDefault<WordHasher>>;

/// Debug check of the invariant that ties an [`Entries`] variant to
/// [`KeyLayout::n_words`]: a key type only receives word counts it holds
/// (at most one for `u64`, two for `u128`, three or more for a boxed key).
/// The counts come from the cache's own layout, never from caller input.
fn debug_check_word_count(key: &str, words: &[u64], holds: fn(usize) -> bool) {
    debug_assert!(
        holds(words.len()),
        "a {key} key cannot hold {} words",
        words.len()
    );
}

/// A packed key: inline `u64` (at most one word), inline `u128` (two
/// words), or a boxed slice (three or more words).
pub(super) trait PackedKey: Hash + Eq + Sized {
    /// The owned key of the given words.
    fn from_words(words: &[u64]) -> Self;

    /// Look up the key with the given words without allocating.
    fn find<'m, V>(map: &'m WordMap<Self, V>, words: &[u64]) -> Option<&'m V>;

    /// Key word `index`.
    fn word(&self, index: usize) -> u64;

    /// The key with only the bits set in `mask` kept.
    fn masked(self, mask: &[u64]) -> Self;

    /// Wrap a map of this key type.
    fn wrap<V>(map: WordMap<Self, V>) -> Entries<V>;
}

impl PackedKey for u64 {
    fn from_words(words: &[u64]) -> Self {
        debug_check_word_count("u64", words, |count| count <= 1);
        words.first().copied().unwrap_or_default()
    }

    fn find<'m, V>(map: &'m WordMap<Self, V>, words: &[u64]) -> Option<&'m V> {
        map.get(&Self::from_words(words))
    }

    fn word(&self, _index: usize) -> u64 {
        *self
    }

    fn masked(self, mask: &[u64]) -> Self {
        self & Self::from_words(mask)
    }

    fn wrap<V>(map: WordMap<Self, V>) -> Entries<V> {
        Entries::Narrow(map)
    }
}

impl PackedKey for u128 {
    fn from_words(words: &[u64]) -> Self {
        debug_check_word_count("u128", words, |count| count == 2);
        let low = words.first().copied().unwrap_or_default();
        let high = words.get(1).copied().unwrap_or_default();
        u128::from(low) | (u128::from(high) << 64)
    }

    fn find<'m, V>(map: &'m WordMap<Self, V>, words: &[u64]) -> Option<&'m V> {
        map.get(&Self::from_words(words))
    }

    fn word(&self, index: usize) -> u64 {
        (*self >> (64 * index.min(1))) as u64
    }

    fn masked(self, mask: &[u64]) -> Self {
        self & Self::from_words(mask)
    }

    fn wrap<V>(map: WordMap<Self, V>) -> Entries<V> {
        Entries::Double(map)
    }
}

impl PackedKey for Box<[u64]> {
    fn from_words(words: &[u64]) -> Self {
        debug_check_word_count("boxed", words, |count| count >= 3);
        words.into()
    }

    fn find<'m, V>(map: &'m WordMap<Self, V>, words: &[u64]) -> Option<&'m V> {
        map.get(words)
    }

    fn word(&self, index: usize) -> u64 {
        self.get(index).copied().unwrap_or_default()
    }

    fn masked(mut self, mask: &[u64]) -> Self {
        for (word, &keep) in self.iter_mut().zip(mask) {
            *word &= keep;
        }
        self
    }

    fn wrap<V>(map: WordMap<Self, V>) -> Entries<V> {
        Entries::Wide(map)
    }
}

/// The entries of a cache, keyed by the narrowest key type that holds
/// [`KeyLayout::n_words`] words.
#[derive(Clone, Debug)]
pub(super) enum Entries<T> {
    /// At most one key word.
    Narrow(WordMap<u64, T>),
    /// Two key words.
    Double(WordMap<u128, T>),
    /// Three or more key words.
    Wide(WordMap<Box<[u64]>, T>),
}

/// Run `$body` with `$map` bound to the map of whichever key type `$entries`
/// holds.
macro_rules! with_map {
    ($entries:expr, $map:ident => $body:expr) => {
        match $entries {
            Entries::Narrow($map) => $body,
            Entries::Double($map) => $body,
            Entries::Wide($map) => $body,
        }
    };
}

impl<T> Entries<T> {
    /// Empty entries for keys of `n_words` words, with room for `capacity`
    /// entries.
    fn with_capacity(n_words: usize, capacity: usize) -> Self {
        let hasher = BuildHasherDefault::default();
        match n_words {
            0 | 1 => Self::Narrow(WordMap::with_capacity_and_hasher(capacity, hasher)),
            2 => Self::Double(WordMap::with_capacity_and_hasher(capacity, hasher)),
            _ => Self::Wide(WordMap::with_capacity_and_hasher(capacity, hasher)),
        }
    }

    fn len(&self) -> usize {
        with_map!(self, map => map.len())
    }

    fn insert_words(&mut self, words: &[u64], value: T) {
        with_map!(self, map => {
            map.insert(PackedKey::from_words(words), value);
        })
    }

    /// Call `visit` with the words and value of every entry, in map order.
    fn for_each_words(&self, mut visit: impl FnMut(&[u64], &T)) {
        match self {
            Self::Narrow(map) => map.iter().for_each(|(key, value)| visit(&[*key], value)),
            Self::Double(map) => map
                .iter()
                .for_each(|(key, value)| visit(&[key.word(0), key.word(1)], value)),
            Self::Wide(map) => map.iter().for_each(|(key, value)| visit(key, value)),
        }
    }

    /// Call `visit` with the words and value of every entry, consuming them.
    fn drain_words(self, mut visit: impl FnMut(&[u64], T)) {
        match self {
            Self::Narrow(map) => map
                .into_iter()
                .for_each(|(key, value)| visit(&[key], value)),
            Self::Double(map) => map
                .into_iter()
                .for_each(|(key, value)| visit(&[key.word(0), key.word(1)], value)),
            Self::Wide(map) => map.into_iter().for_each(|(key, value)| visit(&key, value)),
        }
    }
}

/// Split `map` among `n_children` children by the coordinate of active slot
/// `slot` of `layout`, moving every key with the bits of `slot` cleared.
fn split_masked<K: PackedKey, T>(
    map: WordMap<K, T>,
    layout: &KeyLayout,
    slot: usize,
    keep: &[u64],
    n_children: usize,
) -> Vec<Entries<T>> {
    let (word, _, _) = layout.slots[slot];
    let capacity = map.len() / n_children.max(1);
    let mut maps: Vec<WordMap<K, T>> = (0..n_children)
        .map(|_| WordMap::with_capacity_and_hasher(capacity, BuildHasherDefault::default()))
        .collect();
    for (key, value) in map {
        let child = layout.coordinate(slot, key.word(word));
        // INVARIANT: every cached coordinate passed `encode_checked`, so
        // `child < n_children`.
        if let Some(child_map) = maps.get_mut(child) {
            child_map.insert(key.masked(keep), value);
        }
    }
    maps.into_iter().map(K::wrap).collect()
}

/// Evaluation cache of one patch, keyed by its active coordinates.
///
/// `tensor4all_core::CachedFunction` does not fit here: it wraps an
/// infallible point function (the driver's evaluator is fallible and batched),
/// its keys stop at 1024 bits, and its entries can only be cleared at once,
/// whereas this cache must be split among the children of a split patch.
/// Keys of one or two words are stored inline and lookups borrow a reused
/// encode buffer, so no lookup allocates (see the "Owned Vector Cache Keys"
/// rule in `PERFORMANCE_TIPS.md`); only a new entry with a key of three or
/// more words allocates its boxed key.
#[derive(Clone, Debug)]
pub(super) struct PatchCache<T> {
    layout: KeyLayout,
    entries: Entries<T>,
}

impl<T: Copy> PatchCache<T> {
    pub(super) fn new(active_dims: Vec<usize>) -> Self {
        Self::with_layout(KeyLayout::new(active_dims), 0)
    }

    fn with_layout(layout: KeyLayout, capacity: usize) -> Self {
        let entries = Entries::with_capacity(layout.n_words, capacity);
        Self { layout, entries }
    }

    pub(super) fn layout(&self) -> &KeyLayout {
        &self.layout
    }

    #[cfg(test)]
    pub(super) fn get(&self, key: &[u64]) -> Option<T> {
        with_map!(&self.entries, map => PackedKey::find(map, key).copied())
    }

    #[cfg(test)]
    pub(super) fn insert(&mut self, key: Box<[u64]>, value: T) {
        self.entries.insert_words(&key, value);
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }

    /// The entries, for tests of the key type.
    #[cfg(test)]
    pub(super) fn entries(&self) -> &Entries<T> {
        &self.entries
    }

    /// The active coordinates of the at most `limit` entries with the largest
    /// positive `magnitude`, largest first. Equal magnitudes are ordered by
    /// their coordinates, so the result does not depend on the map order or
    /// the key packing. Entries of magnitude zero are skipped.
    pub(super) fn largest_points(
        &self,
        limit: usize,
        magnitude: impl Fn(T) -> f64,
    ) -> Vec<Vec<usize>> {
        if limit == 0 {
            return Vec::new();
        }
        let mut best: Vec<(f64, Vec<usize>)> = Vec::with_capacity(limit.saturating_add(1));
        let before = |a: &(f64, Vec<usize>), b: &(f64, Vec<usize>)| {
            b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)).is_lt()
        };
        self.entries.for_each_words(|words, &value| {
            let size = magnitude(value);
            if size.is_nan() || size <= 0.0 {
                return;
            }
            if best.len() == limit && best.last().is_some_and(|last| size < last.0) {
                return;
            }
            let entry = (size, self.layout.decode(words));
            let at = best.partition_point(|kept| before(kept, &entry));
            if at < limit {
                best.insert(at, entry);
                best.truncate(limit);
            }
        });
        best.into_iter().map(|(_, point)| point).collect()
    }

    /// Split the cache among the children of a split at active slot `slot`,
    /// in one pass: child `c` receives every entry whose coordinate at `slot`
    /// is `c`, keyed without that coordinate.
    ///
    /// The children keep the parent's packing with the bits of `slot`
    /// cleared, so every entry moves with its key words and nothing is
    /// re-encoded. Only when removing the coordinate lets a compact packing
    /// use fewer words are the keys re-encoded into that packing, so a wide
    /// root cache reaches the inline key types after enough splits.
    pub(super) fn split(self, slot: usize) -> Vec<Self> {
        let mut child_dims = self.layout.dims.clone();
        let n_children = child_dims.remove(slot);
        let compact = KeyLayout::new(child_dims);
        if compact.n_words < self.layout.n_words {
            let capacity = self.entries.len() / n_children.max(1);
            let mut children: Vec<Self> = (0..n_children)
                .map(|_| Self::with_layout(compact.clone(), capacity))
                .collect();
            let mut child_key = vec![0u64; compact.n_words];
            let layout = self.layout;
            self.entries.drain_words(|words, value| {
                let mut coords = layout.decode(words);
                let child = coords.remove(slot);
                compact.encode_into(coords.into_iter(), &mut child_key);
                // INVARIANT: every cached coordinate passed `encode_checked`,
                // so `child < n_children`.
                if let Some(child) = children.get_mut(child) {
                    child.entries.insert_words(&child_key, value);
                }
            });
            return children;
        }
        let (child_layout, keep) = self.layout.without_slot(slot);
        let layout = &self.layout;
        let entries =
            with_map!(self.entries, map => split_masked(map, layout, slot, &keep, n_children));
        entries
            .into_iter()
            .map(|entries| Self {
                layout: child_layout.clone(),
                entries,
            })
            .collect()
    }
}

/// Evaluation counters of a run.
#[derive(Default)]
pub(super) struct Counters {
    pub(super) evaluations: Cell<usize>,
    pub(super) cache_hits: Cell<usize>,
}

impl Counters {
    pub(super) fn add(counter: &Cell<usize>, amount: usize) {
        counter.set(counter.get().saturating_add(amount));
    }
}

/// The value of one requested point: cached, or the index of a new point.
enum Slot<T> {
    Known(T),
    Missing(usize),
}

/// Samples one patch through its cache.
pub(super) struct PatchSampler<'a, T, F> {
    pub(super) evaluate: &'a F,
    pub(super) fixed: &'a [Option<usize>],
    pub(super) n_active: usize,
    pub(super) counters: &'a Counters,
    pub(super) cache: RefCell<PatchCache<T>>,
}

impl<T, F> PatchSampler<'_, T, F>
where
    T: CommonScalar + TensorElement,
    F: Fn(ColMajorArrayRef<'_, usize>) -> anyhow::Result<Vec<T>>,
{
    /// Values at a column-major `[n_active, n_points]` batch of active
    /// coordinates. Only points absent from the cache reach the evaluator,
    /// each once, completed with the fixed coordinates. The values are checked
    /// for count and finiteness before they are cached; nothing is cached when
    /// the evaluator fails.
    pub(super) fn sample(&self, batch: ColMajorArrayRef<'_, usize>) -> anyhow::Result<Vec<T>> {
        let n_active = self.n_active;
        let shape = batch.shape();
        anyhow::ensure!(
            shape.len() == 2 && shape[0] == n_active,
            "batch shape {shape:?} does not match the {n_active} active sites of the patch"
        );
        let n_points = shape[1];
        // The evaluator is the caller's function and cannot reach this
        // sampler, so the cache stays borrowed for the whole call.
        let mut cache = self.cache.borrow_mut();
        let PatchCache { layout, entries } = &mut *cache;
        with_map!(entries, map => self.sample_with(layout, map, batch.data(), n_points))
    }

    /// [`Self::sample`] on the map of one key type.
    fn sample_with<K: PackedKey>(
        &self,
        layout: &KeyLayout,
        map: &mut WordMap<K, T>,
        data: &[usize],
        n_points: usize,
    ) -> anyhow::Result<Vec<T>> {
        let n_active = self.n_active;
        let mut words = vec![0u64; layout.n_words()];
        let mut slots = Vec::with_capacity(n_points);
        // New points of this batch, each with its index among them.
        let mut pending: WordMap<K, usize> = WordMap::default();
        let mut missing_points: Vec<usize> = Vec::new();
        let mut hits = 0usize;
        for point in 0..n_points {
            let local = &data[point * n_active..(point + 1) * n_active];
            if let Err(OutOfRange { slot, value, dim }) = layout.encode_checked(local, &mut words) {
                anyhow::bail!(
                    "point {point} has coordinate {value} at active site {slot}, out of range for \
                     dimension {dim}"
                );
            }
            if let Some(&value) = K::find(map, &words) {
                hits += 1;
                slots.push(Slot::Known(value));
                continue;
            }
            if let Some(&index) = K::find(&pending, &words) {
                hits += 1;
                slots.push(Slot::Missing(index));
                continue;
            }
            let index = pending.len();
            pending.insert(K::from_words(&words), index);
            slots.push(Slot::Missing(index));
            let mut active = local.iter().copied();
            missing_points.extend(
                self.fixed
                    .iter()
                    .map(|fixed| fixed.or_else(|| active.next()).unwrap_or_default()),
            );
        }
        Counters::add(&self.counters.cache_hits, hits);

        let mut fresh = Vec::new();
        if !pending.is_empty() {
            let n_sites = self.fixed.len();
            let n_missing = pending.len();
            let full_shape = [n_sites, n_missing];
            fresh = (self.evaluate)(ColMajorArrayRef::new(&missing_points, &full_shape)?)?;
            anyhow::ensure!(
                fresh.len() == n_missing,
                "the evaluator returned {} values for {n_missing} points",
                fresh.len()
            );
            if let Some((index, defect)) = fresh
                .iter()
                .enumerate()
                .find_map(|(index, &value)| value_defect(value).map(|defect| (index, defect)))
            {
                let point = &missing_points[index * n_sites..(index + 1) * n_sites];
                match defect {
                    ValueDefect::NonFinite => anyhow::bail!(
                        "the evaluator returned a non-finite value at the point {point:?}"
                    ),
                    ValueDefect::MagnitudeOverflow => anyhow::bail!(
                        "the evaluator returned a value whose magnitude overflows at the point \
                         {point:?}"
                    ),
                }
            }
            Counters::add(&self.counters.evaluations, n_missing);
            // The new keys move into the cache; the order of insertion does
            // not affect the cached mapping.
            map.reserve(n_missing);
            for (key, index) in pending {
                map.insert(key, fresh[index]);
            }
        }

        Ok(slots
            .into_iter()
            .map(|slot| match slot {
                Slot::Known(value) => value,
                Slot::Missing(index) => fresh[index],
            })
            .collect())
    }
}

/// Why a sampled value cannot be used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ValueDefect {
    /// A component (real or imaginary part) is infinite or NaN.
    NonFinite,
    /// Every component is finite but the magnitude `abs_val` overflows, as for
    /// a complex value near the largest finite float in both parts.
    MagnitudeOverflow,
}

/// The defect of a sampled value, or `None` when its components and its
/// magnitude are finite. Every magnitude the driver uses (the pinned
/// reference scale, zero screening, the maximum sample) is then finite.
pub(super) fn value_defect<T: CommonScalar>(value: T) -> Option<ValueDefect> {
    if !is_finite(value) {
        Some(ValueDefect::NonFinite)
    } else if !value.abs_val().is_finite() {
        Some(ValueDefect::MagnitudeOverflow)
    } else {
        None
    }
}

/// Whether every component (real and imaginary part) of `value` is finite.
///
/// Multiplying by zero gives exactly zero for finite components and NaN when
/// a component is infinite or NaN. Unlike `abs_val().is_finite()`, this
/// accepts a finite complex value whose magnitude overflows.
pub(super) fn is_finite<T: CommonScalar>(value: T) -> bool {
    (value * T::from_f64(0.0)).abs_val() == 0.0
}
