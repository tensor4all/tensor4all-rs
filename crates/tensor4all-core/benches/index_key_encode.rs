//! Cost of encoding one multi-index with `FlatIndexer`, isolated (#1224).
//!
//! Next to it is the `Vec<usize>` key that `TTCache` builds today. The two are
//! compared because #1146 moves `TTCache` onto `FlatIndexer`, and on a pure
//! cache hit the encode is most of the call: 1623.8 of 1930.4 ns at 150 bits
//! before #1224. The `n300` hit cells of `simplett`'s `cache_overhead` encode
//! 150 binary components.
//!
//! Widths straddle every arm boundary of the packed key: 64/65 (u64 -> u128),
//! 128/129 (-> 256-bit), 256/257 (-> 512-bit) and 512/513 (-> limbs).

use std::hash::BuildHasher;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rustc_hash::FxBuildHasher;
use tensor4all_core::index_key::FlatIndexer;

/// Component counts for binary local dimensions (one bit each).
const BINARY_COMPONENTS: [usize; 13] =
    [16, 50, 64, 65, 100, 128, 129, 150, 200, 256, 257, 300, 513];

/// A deterministic, non-constant multi-index over binary dimensions.
fn binary_index(len: usize) -> Vec<usize> {
    (0..len).map(|i| (i * 7 + i / 3) % 2).collect()
}

fn bench_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_key_encode/encode");
    for len in BINARY_COMPONENTS {
        let indexer = FlatIndexer::try_new(&vec![2usize; len]).unwrap();
        let idx = binary_index(len);
        group.bench_with_input(BenchmarkId::new("binary", len), &idx, |b, idx| {
            b.iter(|| indexer.encode(black_box(idx)).unwrap())
        });
    }
    group.finish();
}

fn bench_encode_hash(c: &mut Criterion) {
    let state = FxBuildHasher;
    let mut group = c.benchmark_group("index_key_encode/encode_hash");
    for len in BINARY_COMPONENTS {
        let indexer = FlatIndexer::try_new(&vec![2usize; len]).unwrap();
        let idx = binary_index(len);
        group.bench_with_input(BenchmarkId::new("binary", len), &idx, |b, idx| {
            b.iter(|| {
                let key = indexer.encode(black_box(idx)).unwrap();
                state.hash_one(&key)
            })
        });
    }
    group.finish();
}

/// The pre-#1146 key: an owned copy of the index, hashed in full.
fn bench_vec_key_hash(c: &mut Criterion) {
    let state = FxBuildHasher;
    let mut group = c.benchmark_group("index_key_encode/vec_key_hash");
    for len in BINARY_COMPONENTS {
        let idx = binary_index(len);
        group.bench_with_input(BenchmarkId::new("binary", len), &idx, |b, idx| {
            b.iter(|| {
                let key: Vec<usize> = black_box(idx).to_vec();
                state.hash_one(&key)
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_encode, bench_encode_hash, bench_vec_key_hash);
criterion_main!(benches);
