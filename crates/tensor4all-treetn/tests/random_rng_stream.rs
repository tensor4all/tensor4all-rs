//! Random TreeTNs consume the caller's erased RNG in node/column-major order.

use num_complex::Complex64;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::HashSet;
use std::fmt::Debug;
use tensor4all_core::{tensor::RandomScalar, DynIndex};
use tensor4all_treetn::{random_treetn, LinkSpace, SiteIndexNetwork};

fn assert_stream<T: RandomScalar + PartialEq + Debug>() {
    let mut network = SiteIndexNetwork::<usize, DynIndex>::new();
    network
        .add_node(0, HashSet::from([DynIndex::new_dyn(2)]))
        .unwrap();
    network
        .add_node(1, HashSet::from([DynIndex::new_dyn(3)]))
        .unwrap();
    network.add_edge(&0, &1).unwrap();

    for seed in [0, 42] {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut reference = rng.clone();
        let erased: &mut dyn RngCore = &mut rng;
        let tree = random_treetn::<T, _, _>(erased, &network, LinkSpace::uniform(2)).unwrap();
        for name in network.node_names() {
            let tensor = tree.tensor(tree.node_index(name).unwrap()).unwrap();
            let expected_len = if *name == 0 { 4 } else { 6 };
            let expected: Vec<T> = (0..expected_len)
                .map(|_| T::random_value(&mut reference))
                .collect();
            assert_eq!(tensor.to_vec::<T>().unwrap(), expected);
        }
        assert_eq!(rng.get_word_pos(), reference.get_word_pos());
        assert_eq!(rng.next_u64(), reference.next_u64());
    }
}

#[test]
fn erased_real_rng_is_consumed_directly_in_node_order() {
    assert_stream::<f64>();
}

#[test]
fn erased_complex_rng_is_consumed_directly_in_node_order() {
    assert_stream::<Complex64>();
}
