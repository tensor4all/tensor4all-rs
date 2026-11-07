use super::*;

/// The accounted bytes never fall below the buckets the tables really hold:
/// capacity() × (entry + 1) per generation, at every entry count.
#[test]
fn accounted_bytes_cover_the_capacities_at_every_entry_count() {
    let entry = std::mem::size_of::<(u64, f64)>();
    let mut table = MemoTable::<u64, f64>::default();
    for key in 0..5_000u64 {
        assert!(table.insert(key, key as f64, usize::MAX).stored);
        let (current, previous) = table.capacities();
        assert!(table.bytes() >= (current + previous) * (entry + 1), "{key}");
        assert_eq!(table.len(), key as usize + 1);
    }
}

/// `buckets_of_capacity` inverts hashbrown's bucket-to-capacity rule, so the
/// accounting follows the real growth steps of `std::collections::HashMap`.
#[test]
fn bucket_count_follows_the_real_growth_steps() {
    let mut map = FxHashMap::<u64, f64>::default();
    let mut seen = vec![buckets_of_capacity(map.capacity())];
    for key in 0..10_000u64 {
        map.insert(key, 0.0);
        let buckets = buckets_of_capacity(map.capacity());
        assert!(buckets.is_power_of_two(), "{} buckets", buckets);
        assert!(
            buckets > map.len(),
            "{} buckets for {} entries",
            buckets,
            map.len()
        );
        if seen.last() != Some(&buckets) {
            seen.push(buckets);
        }
    }
    // 0, 4, 8, 16, …: every growth doubles, as `grown_bytes` assumes.
    assert_eq!(seen[..3], [0, 4, 8]);
    assert!(seen[1..].windows(2).all(|w| w[1] == 2 * w[0]), "{seen:?}");
}

/// A full table stays within its limit, still answers its most recent keys, and
/// accounts for every key as kept or evicted.
#[test]
fn a_full_table_evicts_the_oldest_generation_and_keeps_recent_keys() {
    for limit in [100, 1_000, 10_000, 100_000, 1 << 20] {
        let mut table = MemoTable::<u64, f64>::default();
        let mut evicted = 0;
        for key in 0..50_000u64 {
            let inserted = table.insert(key, key as f64, limit);
            assert!(inserted.stored, "limit {limit} key {key}");
            evicted += inserted.evicted;
            assert!(
                table.bytes() <= limit,
                "limit {limit} key {key}: {} bytes",
                table.bytes()
            );
            assert_eq!(
                table.get(&key),
                Some(&(key as f64)),
                "the key just inserted"
            );
        }
        // Eviction is oldest first: the table holds exactly the latest keys.
        let n = table.len() as u64;
        assert!(
            (50_000 - n..50_000).all(|key| table.contains_key(&key)),
            "limit {limit}"
        );
        assert!(evicted > 0, "limit {limit}");
        assert_eq!(evicted + table.len(), 50_000, "limit {limit}");
    }
}

/// Below the smallest table nothing can be stored; the insertion is refused.
#[test]
fn a_limit_below_one_table_refuses() {
    let mut table = MemoTable::<u64, f64>::default();
    let smallest = table_bytes::<u64, f64>(4);
    assert_eq!(
        table.insert(1, 1.0, smallest - 1),
        Inserted {
            stored: false,
            evicted: 0
        }
    );
    assert_eq!(table.len(), 0);
    assert!(table.insert(1, 1.0, smallest).stored);
}

/// Overwriting a key in either generation never evicts or grows.
#[test]
fn overwriting_never_evicts() {
    let mut table = MemoTable::<u64, f64>::default();
    for key in 0..1_000u64 {
        table.insert(key, 0.0, 2_000);
    }
    let (len, bytes) = (table.len(), table.bytes());
    let kept: Vec<u64> = (0..1_000).filter(|key| table.contains_key(key)).collect();
    for &key in &kept {
        assert_eq!(
            table.insert(key, 1.0, 2_000),
            Inserted {
                stored: true,
                evicted: 0
            }
        );
    }
    assert_eq!((table.len(), table.bytes()), (len, bytes));
    assert!(kept.iter().all(|key| table.get(key) == Some(&1.0)));
}
