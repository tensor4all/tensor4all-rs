//! Regression test for tensor4all-rs issue #795: rebuild stability of
//! [`TreeTNCachedEvaluator`].
//!
//! The cached evaluator contracts an N-ary batch of the canonical center with
//! its cached environments through `contract_with_options`, whose contraction
//! path is selected inside the backend. Issue #795 recorded that this path is
//! covered neither by the #791 determinism suite nor by any other test, so a
//! path order that depends on index identity would silently produce
//! non-reproducible evaluation values.
//!
//! Every rebuild below creates freshly identified indices and the same data,
//! so any dependence on index-identity hashing changes the summation order and
//! shows up as a different bit pattern.
//!
//! LIMITATION (measured while investigating #795): this test pins *in-process*
//! rebuild stability only. Repeated runs of the same binary return two
//! different bit patterns for this network when the evaluation center is a leaf
//! (a 4-operand rooted message contraction) or when the center is chosen by the
//! greedy search, while the uncached `TreeTN::evaluate` path stays
//! bit-identical and the contraction spec passed to the backend is identical in
//! every process. The variation is therefore the N-ary contraction path
//! selection, which is not process-reproducible: `ContractionTree::optimize`
//! returns different equal-cost paths per process. That is tracked in
//! tensor4all/tenferro-rs#1963, so cross-process bitwise reproducibility is not
//! guaranteed by this evaluator until that lands.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor};
use tensor4all_treetn::{CachedEvaluatorOptions, TreeTN, TreeTNCachedEvaluator};

/// Number of rebuilds per check. Each rebuild draws fresh index identities.
const REBUILDS: usize = 64;

/// Site dimensions, in `site_indices` order. The hub carries three sites and
/// the five leaves carry the rest, so one N-ary center contraction has six
/// operands with distinct local and bond dimensions.
const SITE_DIMS: [usize; 8] = [2, 3, 2, 2, 2, 3, 2, 3];
/// Bond dimensions of the hub's five edges, all different.
const BOND_DIMS: [usize; 5] = [2, 3, 2, 4, 3];
/// Points evaluated after every rebuild, one column per point.
const POINTS: [[usize; 8]; 9] = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [1, 2, 1, 1, 1, 2, 1, 2],
    [1, 1, 1, 0, 1, 1, 0, 2],
    [0, 2, 1, 1, 0, 2, 1, 0],
    [1, 0, 0, 1, 1, 0, 1, 1],
    [0, 1, 1, 1, 1, 1, 0, 2],
    [1, 2, 0, 0, 0, 2, 1, 0],
    [0, 0, 1, 0, 1, 0, 1, 1],
    [1, 2, 1, 1, 0, 1, 0, 2],
];

/// A six-node star: hub `0` with three sites, leaves `1..=5` with one each.
///
/// ```text
///   1 - 0 - 2
///   3 - | - 4
///       5
/// ```
fn star_tree() -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
    let sites = SITE_DIMS
        .iter()
        .map(|&dim| DynIndex::new_dyn(dim))
        .collect::<Vec<_>>();
    let bonds = BOND_DIMS
        .iter()
        .map(|&dim| DynIndex::new_dyn(dim))
        .collect::<Vec<_>>();

    let mut rng = ChaCha8Rng::seed_from_u64(20_261_003);
    let mut data = |len: usize| (0..len).map(|_| rng.random::<f64>()).collect::<Vec<f64>>();

    let mut hub_indices = vec![sites[0].clone(), sites[1].clone(), sites[2].clone()];
    hub_indices.extend(bonds.iter().cloned());
    let hub_size = SITE_DIMS[0] * SITE_DIMS[1] * SITE_DIMS[2] * BOND_DIMS.iter().product::<usize>();
    let hub = IdxTensor::from_dense(hub_indices, data(hub_size)).unwrap();

    let mut tensors = vec![hub];
    for (leaf, (&bond_dim, &site_dim)) in BOND_DIMS.iter().zip(&SITE_DIMS[3..]).enumerate() {
        tensors.push(
            IdxTensor::from_dense(
                vec![bonds[leaf].clone(), sites[3 + leaf].clone()],
                data(bond_dim * site_dim),
            )
            .unwrap(),
        );
    }
    let tree =
        TreeTN::<IdxTensor, usize>::from_tensors(tensors, (0..6).collect::<Vec<usize>>()).unwrap();

    (tree, sites)
}

/// Evaluates `POINTS` through a freshly built cached evaluator.
fn evaluate_points() -> Vec<f64> {
    let (tree, sites) = star_tree();
    let flat = POINTS
        .iter()
        .flat_map(|point| point.iter().copied())
        .collect::<Vec<usize>>();
    let shape = [SITE_DIMS.len(), POINTS.len()];
    let values = ColMajorArrayRef::new(&flat, &shape).unwrap();

    let mut evaluator =
        TreeTNCachedEvaluator::new(&tree, &sites, CachedEvaluatorOptions::<usize>::default())
            .unwrap();
    evaluator
        .evaluate_batched(values)
        .unwrap()
        .into_iter()
        .map(|value| value.real())
        .collect()
}

#[test]
fn cached_evaluator_is_bitwise_stable_across_rebuilds() {
    let reference = evaluate_points();
    assert!(
        reference.iter().all(|value| value.is_finite()),
        "cached-evaluator values must be finite: {reference:?}"
    );
    assert!(
        reference.windows(2).any(|pair| pair[0] != pair[1]),
        "the probe points must not all evaluate to the same value: {reference:?}"
    );

    let reference_bits = reference
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();
    for rebuild in 1..REBUILDS {
        let current = evaluate_points()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        assert_eq!(
            current, reference_bits,
            "rebuild {rebuild} changed the cached-evaluator result"
        );
    }
}
