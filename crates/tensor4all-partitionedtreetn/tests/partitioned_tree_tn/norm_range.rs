use super::*;
use tensor4all_core::TensorElement;

fn check<T: TensorElement>(values: Vec<T>, expected: f64) {
    let site = DynIndex::new_dyn(2);
    let source = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![site.clone()], values).unwrap()],
        vec![0usize],
    )
    .unwrap();
    let partition = PartitionedTreeTN::from_subdomains(
        (0..2)
            .map(|value| {
                SubDomainTreeTN::new(
                    source.clone(),
                    Projector::from_pairs([(site.clone(), value)]).unwrap(),
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap();
    let actual = partition.norm().unwrap();
    assert!(
        (actual / expected - 1.0).abs() < 1e-12,
        "{actual:e} != {expected:e}"
    );
}

#[test]
fn representable_partition_norms_survive_overflowing_and_underflowing_squares() {
    for scale in [1e200, 1e-200] {
        check(vec![3.0 * scale, 4.0 * scale], 5.0 * scale);
        check(
            vec![
                Complex64::new(3.0 * scale, 4.0 * scale),
                Complex64::new(0.0, 12.0 * scale),
            ],
            13.0 * scale,
        );
    }
}

#[test]
fn norm_and_norm_squared_accumulate_in_canonical_patch_order() {
    // Irregular magnitudes make the rounded folds depend on their order;
    // `PartitionedTreeTN` stores patches in a `HashMap`, so each rebuilt
    // partition would iterate them in a different order without the sort.
    const N: usize = 16;
    let values: Vec<f64> = (0..N)
        .map(|k| (0.37 + k as f64).sqrt() * (1.0 + 0.1 * k as f64).ln_1p())
        .collect();
    let expected = values.iter().map(|v| v * v).sum::<f64>().sqrt();
    let site = DynIndex::new_dyn(N);
    let source = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![site.clone()], values).unwrap()],
        vec![0usize],
    )
    .unwrap();
    let mut patches: Vec<_> = (0..N)
        .map(|v| {
            SubDomainTreeTN::new(
                source.clone(),
                Projector::from_pairs([(site.clone(), v)]).unwrap(),
            )
            .unwrap()
        })
        .collect();
    let bits = |partition: &PartitionedTreeTN| {
        (
            partition.norm().unwrap().to_bits(),
            partition.norm_squared().unwrap().to_bits(),
        )
    };
    let reference = PartitionedTreeTN::from_subdomains(patches.clone()).unwrap();
    assert!((reference.norm().unwrap() / expected - 1.0).abs() < 1e-14);
    assert!((reference.norm_squared().unwrap() / (expected * expected) - 1.0).abs() < 1e-14);
    let reference = bits(&reference);
    for shift in 1..N {
        patches.rotate_left(1);
        let partition = PartitionedTreeTN::from_subdomains(patches.clone()).unwrap();
        assert_eq!(bits(&partition), reference, "shift {shift}");
    }
}
