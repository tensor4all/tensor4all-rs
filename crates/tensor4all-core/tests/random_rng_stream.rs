//! Caller-owned RNG streams remain usable through the object-safe RngCore API.

use num_complex::Complex64;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use std::fmt::Debug;
use tensor4all_core::{tensor::RandomScalar, DynIndex, IdxTensor};

struct CountingRng {
    inner: ChaCha8Rng,
    calls: usize,
}

impl RngCore for CountingRng {
    fn next_u32(&mut self) -> u32 {
        self.calls += 1;
        self.inner.next_u32()
    }

    fn next_u64(&mut self) -> u64 {
        self.calls += 1;
        self.inner.next_u64()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.calls += 1;
        self.inner.fill_bytes(dest);
    }
}

fn assert_stream<T: RandomScalar + PartialEq + Debug>(sample: fn(&mut ChaCha8Rng) -> T) {
    for seed in [0, 42] {
        for dims in [vec![], vec![2, 3]] {
            let mut rng = CountingRng {
                inner: ChaCha8Rng::seed_from_u64(seed),
                calls: 0,
            };
            let mut reference = rng.inner.clone();
            let erased: &mut dyn RngCore = &mut rng;

            assert_eq!(T::random_value(erased), sample(&mut reference));
            let expected: Vec<T> = (0..dims.iter().product::<usize>())
                .map(|_| sample(&mut reference))
                .collect();
            let indices = dims.into_iter().map(DynIndex::new_dyn).collect();
            let tensor = IdxTensor::random::<T, _>(erased, indices).unwrap();
            assert_eq!(tensor.to_vec::<T>().unwrap(), expected);
            assert!(rng.calls > 0, "the caller's RNG must supply the samples");
            assert_eq!(rng.inner.get_word_pos(), reference.get_word_pos());
            assert_eq!(rng.next_u64(), reference.next_u64());
        }
    }
}

#[test]
fn erased_real_rng_uses_the_callers_standard_normal_stream() {
    assert_stream::<f64>(|rng| StandardNormal.sample(rng));
}

#[test]
fn erased_complex_rng_uses_independent_real_and_imaginary_draws() {
    assert_stream::<Complex64>(|rng| {
        Complex64::new(StandardNormal.sample(rng), StandardNormal.sample(rng))
    });
}
