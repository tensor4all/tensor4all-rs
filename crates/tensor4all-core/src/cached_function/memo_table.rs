//! The bounded table behind the caches: two generations of one hash map, sized
//! by an upper bound on the bytes their tables occupy.

use rustc_hash::FxHashMap;
use std::hash::Hash;

/// Control bytes past the last bucket, counted at hashbrown's widest group (SSE2,
/// 16): exact on x86_64, 8 bytes more than a NEON target's 8-byte group, so the
/// accounting is an upper bound there.
const GROUP_WIDTH: usize = 16;

/// What [`MemoTable::insert`] did with an entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Inserted {
    /// False when the limit is below the accounted size (an upper bound) of the
    /// smallest table that holds one entry.
    pub(crate) stored: bool,
    /// Older entries dropped to make room.
    pub(crate) evicted: usize,
}

/// Two generations of a hash map under one byte limit.
///
/// New keys go to `current`. When `current` is full and its next growth would
/// take both tables over the limit, `previous` is dropped and `current` becomes
/// `previous`: the oldest generation is evicted, and the most recent inserts
/// always stay. Entries are never removed from a live table, so no tombstone can
/// force a growth. Lookups check both generations.
///
/// Bytes are an upper bound on the tables themselves, from their capacities:
/// buckets × (`size_of::<(K, V)>()` + 1 control byte) + one 16-byte group of
/// trailing control bytes (exact with SSE2 groups, 8 bytes over on NEON targets).
/// During a growth the outgoing table is alive next to the new one: both tables
/// fit the limit afterwards and the outgoing one is half the new one plus half a
/// group, so the transient peak is at most 1.5 × limit + `GROUP_WIDTH` / 2.
#[derive(Debug, Clone)]
pub(crate) struct MemoTable<K, V> {
    current: FxHashMap<K, V>,
    previous: FxHashMap<K, V>,
}

impl<K, V> Default for MemoTable<K, V> {
    fn default() -> Self {
        Self {
            current: FxHashMap::default(),
            previous: FxHashMap::default(),
        }
    }
}

/// Buckets of a hashbrown table of usable capacity `capacity` (the inverse of its
/// `bucket_mask_to_capacity`: 7/8 load from 8 buckets on, one free bucket below).
pub(crate) fn buckets_of_capacity(capacity: usize) -> usize {
    match capacity {
        0 => 0,
        1..=7 => (capacity + 1).next_power_of_two(),
        _ => capacity / 7 * 8,
    }
}

/// Bytes of a table with `buckets` buckets of `(K, V)` entries.
pub(crate) fn table_bytes<K, V>(buckets: usize) -> usize {
    if buckets == 0 {
        return 0;
    }
    buckets
        .saturating_mul(std::mem::size_of::<(K, V)>() + 1)
        .saturating_add(GROUP_WIDTH)
}

fn map_bytes<K, V>(map: &FxHashMap<K, V>) -> usize {
    table_bytes::<K, V>(buckets_of_capacity(map.capacity()))
}

/// Bytes of `map` after the growth its next new key triggers.
fn grown_bytes<K, V>(map: &FxHashMap<K, V>) -> usize {
    let buckets = buckets_of_capacity(map.capacity());
    table_bytes::<K, V>(if buckets == 0 { 4 } else { buckets * 2 })
}

impl<K: Hash + Eq, V> MemoTable<K, V> {
    pub(crate) fn get(&self, key: &K) -> Option<&V> {
        self.current.get(key).or_else(|| self.previous.get(key))
    }

    pub(crate) fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Entries in both generations (disjoint: a key lives in one of them).
    pub(crate) fn len(&self) -> usize {
        self.current.len() + self.previous.len()
    }

    /// Bytes of both tables.
    pub(crate) fn bytes(&self) -> usize {
        map_bytes(&self.current).saturating_add(map_bytes(&self.previous))
    }

    pub(crate) fn clear(&mut self) {
        self.current = FxHashMap::default();
        self.previous = FxHashMap::default();
    }

    /// Bytes of both tables once one more new key is stored.
    fn bytes_after_new_key(&self) -> usize {
        // INVARIANT: entries are never removed from a live map, so a new key
        // grows `current` exactly when it is at capacity.
        let current = if self.current.len() == self.current.capacity() {
            grown_bytes(&self.current)
        } else {
            map_bytes(&self.current)
        };
        map_bytes(&self.previous).saturating_add(current)
    }

    /// Store `value` under `key` so that both tables stay within `limit` bytes:
    /// first by making the current generation the previous one (dropping the
    /// older), then by dropping that one too; refused only when the limit is below
    /// the smallest table. A lowered limit thus takes effect at the next new key.
    pub(crate) fn insert(&mut self, key: K, value: V, limit: usize) -> Inserted {
        if let Some(slot) = self.current.get_mut(&key) {
            *slot = value;
            return Inserted {
                stored: true,
                evicted: 0,
            };
        }
        if let Some(slot) = self.previous.get_mut(&key) {
            *slot = value;
            return Inserted {
                stored: true,
                evicted: 0,
            };
        }
        let mut evicted = 0;
        if self.bytes_after_new_key() > limit {
            evicted += self.previous.len();
            self.previous = std::mem::take(&mut self.current);
        }
        if self.bytes_after_new_key() > limit {
            evicted += self.previous.len();
            self.previous = FxHashMap::default();
        }
        let stored = self.bytes_after_new_key() <= limit;
        if stored {
            self.current.insert(key, value);
        }
        Inserted { stored, evicted }
    }

    /// Capacities of both tables, for tests of the accounting.
    #[cfg(test)]
    pub(crate) fn capacities(&self) -> (usize, usize) {
        (self.current.capacity(), self.previous.capacity())
    }

    /// Entries of the previous generation, for tests of the rotation.
    #[cfg(test)]
    pub(crate) fn previous_len(&self) -> usize {
        self.previous.len()
    }
}

#[cfg(test)]
mod tests;
