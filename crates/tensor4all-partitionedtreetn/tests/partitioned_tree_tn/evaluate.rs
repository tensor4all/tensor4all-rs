//! `PartitionedTreeTN::evaluate_batched` on a star with a degree-3 hub.

use super::*;
use tensor4all_core::{AnyScalar, ColMajorArrayRef, TensorElement};

/// The sites of the star: hub sites `h0`, `h1`, then leaf sites `l1`, `l2`,
/// `l3`; `l2` has dimension three, every other site two.
struct Star {
    tree: TreeTN<IdxTensor, usize>,
    h0: DynIndex,
    h1: DynIndex,
    l1: DynIndex,
    l2: DynIndex,
    l3: DynIndex,
}

/// A star `0 - {1, 2, 3}` with bonds of dimensions 2, 3, 2 and pseudo-random
/// entries built by `scalar(re, im)`.
fn star<T: TensorElement>(scalar: fn(f64, f64) -> T) -> Star {
    let (h0, h1) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
    let leaves = [
        DynIndex::new_dyn(2),
        DynIndex::new_dyn(3),
        DynIndex::new_dyn(2),
    ];
    let bonds = [
        DynIndex::new_dyn(2),
        DynIndex::new_dyn(3),
        DynIndex::new_dyn(2),
    ];
    let tensor = |legs: Vec<DynIndex>, seed: u64| {
        let values: Vec<T> = grid_points(&legs)
            .iter()
            .map(|coord| scalar(seeded(seed, coord), seeded(seed + 100, coord)))
            .collect();
        IdxTensor::from_dense(legs, values).unwrap()
    };
    let mut tensors = vec![tensor(
        vec![
            h0.clone(),
            h1.clone(),
            bonds[0].clone(),
            bonds[1].clone(),
            bonds[2].clone(),
        ],
        1,
    )];
    for leaf in 0..3 {
        tensors.push(tensor(
            vec![bonds[leaf].clone(), leaves[leaf].clone()],
            2 + leaf as u64,
        ));
    }
    let tree = TreeTN::from_tensors(tensors, vec![0usize, 1, 2, 3]).unwrap();
    let [l1, l2, l3] = leaves;
    Star {
        tree,
        h0,
        h1,
        l1,
        l2,
        l3,
    }
}

impl Star {
    /// An irregular layout whose projectors fix one, two, and three sites.
    fn projectors(&self) -> Vec<Projector> {
        vec![
            projector(&[(&self.h0, 0)]),
            projector(&[(&self.h0, 1), (&self.l1, 0)]),
            projector(&[(&self.h0, 1), (&self.l1, 1), (&self.h1, 0)]),
            projector(&[(&self.h0, 1), (&self.l1, 1), (&self.h1, 1)]),
        ]
    }

    /// The site order of the rows: a permutation of the derived order.
    fn order(&self) -> Vec<DynIndex> {
        vec![
            self.l2.clone(),
            self.h1.clone(),
            self.l3.clone(),
            self.h0.clone(),
            self.l1.clone(),
        ]
    }
}

fn evaluate(
    partition: &PartitionedTreeTN<usize>,
    order: &[DynIndex],
    points: &[Vec<usize>],
) -> Result<Vec<AnyScalar>, PartitionedTreeTNError> {
    let data: Vec<usize> = points.iter().flatten().copied().collect();
    let shape = [order.len(), points.len()];
    partition.evaluate_batched(order, ColMajorArrayRef::new(&data, &shape).unwrap())
}

fn assert_values(actual: &[AnyScalar], expected: &[AnyScalar]) {
    assert_eq!(actual.len(), expected.len());
    for (point, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.is_complex(), e.is_complex(), "dtype at point {point}");
        let difference = (a.real() - e.real()).hypot(a.imag() - e.imag());
        assert!(difference <= 1e-12, "point {point}: {difference:e}");
    }
}

fn check_star<T: TensorElement>(scalar: fn(f64, f64) -> T, complex: bool) {
    let star = star(scalar);
    let order = star.order();
    let points = grid_points(&order);
    let data: Vec<usize> = points.iter().flatten().copied().collect();
    let shape = [order.len(), points.len()];
    let expected = star
        .tree
        .evaluate(&order, ColMajorArrayRef::new(&data, &shape).unwrap())
        .unwrap();

    // A complete layout reproduces the unpatched network everywhere.
    let projectors = star.projectors();
    let full = layout(&star.tree, &projectors);
    let actual = evaluate(&full, &order, &points).unwrap();
    assert_eq!(actual.len(), 48);
    assert!(actual.iter().all(|value| value.is_complex() == complex));
    assert_values(&actual, &expected);

    // Without the third patch its points evaluate to a zero of the same dtype.
    let partial = layout(
        &star.tree,
        &[
            projectors[0].clone(),
            projectors[1].clone(),
            projectors[3].clone(),
        ],
    );
    let actual = evaluate(&partial, &order, &points).unwrap();
    let missing = &projectors[2];
    let zero = if complex {
        AnyScalar::new_complex(0.0, 0.0)
    } else {
        AnyScalar::new_real(0.0)
    };
    let expected: Vec<AnyScalar> = points
        .iter()
        .zip(&expected)
        .map(|(point, value)| {
            let inside = missing.iter().all(|(index, &fixed)| {
                let row = order.iter().position(|site| site == index).unwrap();
                point[row] == fixed
            });
            if inside {
                zero.clone()
            } else {
                value.clone()
            }
        })
        .collect();
    assert_eq!(
        expected.iter().filter(|value| value.is_zero()).count(),
        6,
        "the missing patch covers 48 / 8 points"
    );
    assert_values(&actual, &expected);
}

#[test]
fn evaluate_batched_matches_the_unpatched_network_and_zero_outside_the_patches() {
    check_star(|re, _| re, false);
    check_star(Complex64::new, true);
}

#[test]
fn evaluate_batched_of_an_empty_partition_is_zero_after_the_input_checks() {
    let empty = PartitionedTreeTN::<usize>::new();
    let site = DynIndex::new_dyn(2);
    let values = evaluate(&empty, std::slice::from_ref(&site), &[vec![1], vec![0]]).unwrap();
    assert_eq!(values.len(), 2);
    assert!(values
        .iter()
        .all(|value| value.is_zero() && value.is_real()));
    let data = [0usize, 1];
    let shape = [1usize, 2];
    let error = empty
        .evaluate_batched(&[], ColMajorArrayRef::new(&data, &shape).unwrap())
        .unwrap_err();
    assert!(matches!(
        error,
        PartitionedTreeTNError::InvalidOptions { .. }
    ));
    // The values are still checked against the indices: bounds and repeats.
    let out_of_bounds = evaluate(&empty, std::slice::from_ref(&site), &[vec![2]]);
    assert!(matches!(
        out_of_bounds,
        Err(PartitionedTreeTNError::InvalidOptions { .. })
    ));
    let repeated = evaluate(&empty, &[site.clone(), site.clone()], &[vec![0, 0]]);
    assert!(matches!(
        repeated,
        Err(PartitionedTreeTNError::InvalidOptions { .. })
    ));
}

#[test]
fn evaluate_batched_rejects_invalid_indices_and_values() {
    let star = star(|re, _| re);
    let projectors = star.projectors();
    // The covered patches leave {h0 = 1, l1 = 1, h1 = 0} uncovered.
    let partition = layout(
        &star.tree,
        &[
            projectors[0].clone(),
            projectors[1].clone(),
            projectors[3].clone(),
        ],
    );
    let order = star.order();
    let invalid = |result: Result<Vec<AnyScalar>, PartitionedTreeTNError>| {
        matches!(result, Err(PartitionedTreeTNError::InvalidOptions { .. }))
    };
    let mismatch = |result: Result<Vec<AnyScalar>, PartitionedTreeTNError>| {
        matches!(result, Err(PartitionedTreeTNError::SiteIndexMismatch))
    };

    // Shape: a row per index, and two dimensions.
    let data = [0usize; 5];
    let shape = [4usize, 1];
    assert!(invalid(partition.evaluate_batched(
        &order,
        ColMajorArrayRef::new(&data[..4], &shape).unwrap()
    )));
    let shape = [5usize, 1, 1];
    assert!(invalid(partition.evaluate_batched(
        &order,
        ColMajorArrayRef::new(&data, &shape).unwrap()
    )));
    // A repeated index.
    let mut repeated = order.clone();
    repeated[4] = repeated[0].clone();
    assert!(invalid(evaluate(&partition, &repeated, &[vec![0; 5]])));
    // Out of bounds, also at a point that no patch covers (l2 has dimension 3).
    assert!(invalid(evaluate(
        &partition,
        &order,
        &[vec![3, 0, 0, 0, 0]]
    )));
    assert!(invalid(evaluate(
        &partition,
        &order,
        &[vec![0, 0, 0, 1, 2]]
    )));
    // Another site space: a foreign index, a missing index, a dimension alias.
    let mut foreign = order.clone();
    foreign[2] = DynIndex::new_dyn(2);
    assert!(mismatch(evaluate(&partition, &foreign, &[vec![0; 5]])));
    assert!(mismatch(evaluate(&partition, &order[..4], &[vec![0; 4]])));
    let mut alias = order.clone();
    alias[0].dim = 2;
    assert!(mismatch(evaluate(&partition, &alias, &[vec![0; 5]])));
}
