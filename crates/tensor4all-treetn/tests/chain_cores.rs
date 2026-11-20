use num_complex::Complex64;
use tensor4all_core::{ColMajorArray, DynIndex, IdxTensor, IndexLike, TagSet};
use tensor4all_tensorbackend::{explicit::Session, CpuBackend};
use tensor4all_treetn::{CanonicalForm, TreeTN, TreeTNOperationError};

fn core<T>(shape: [usize; 3], data: Vec<T>) -> ColMajorArray<T> {
    ColMajorArray::new(data, shape.to_vec()).unwrap()
}

#[test]
fn chain_cores_roundtrip_in_a_live_session() {
    let cores = vec![
        core([1, 3, 2], (1..=6).map(|x| x as f32).collect()),
        core([2, 2, 3], (1..=12).map(|x| -(x as f32)).collect()),
        core([3, 2, 1], (1..=6).map(|x| x as f32 + 0.5).collect()),
    ];

    let mut backend = CpuBackend::with_threads(1).unwrap();
    let _session = Session::new(backend.engine_mut());
    let (tree, sites) = TreeTN::from_chain_cores(cores.clone()).unwrap();
    assert_eq!(tree.chain_cores::<f32>(&sites).unwrap(), cores);
    assert!(tree.chain_cores::<f64>(&sites).is_err());
    assert!(tree.canonical_region().is_empty());
}

#[test]
fn readout_permutations_preserve_full_index_identity() {
    let first = DynIndex::new_dyn(2);
    let second = first.prime();
    let bond = DynIndex::new_with_tags(*first.id(), 2, TagSet::from_str("Link").unwrap());
    let tree = TreeTN::from_tensors(
        vec![
            IdxTensor::from_dense_compact(
                vec![bond.clone(), first.clone()],
                vec![1.0, 3.0, 2.0, 4.0],
            )
            .unwrap(),
            IdxTensor::from_dense_compact(vec![second.clone(), bond], vec![5.0, 7.0, 6.0, 8.0])
                .unwrap(),
        ],
        vec![0, 1],
    )
    .unwrap();

    assert_eq!(
        tree.chain_cores::<f64>(&[first.clone(), second.clone()])
            .unwrap(),
        vec![
            core([1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]),
            core([2, 2, 1], vec![5.0, 6.0, 7.0, 8.0]),
        ]
    );
    let wrong_dim = DynIndex::new(*first.id(), 3);
    assert!(tree
        .chain_cores::<f64>(&[wrong_dim, second.clone()])
        .is_err());
    assert!(tree.chain_cores::<f64>(&[second, first]).is_err());
}

#[test]
fn rejects_invalid_chain_boundaries_and_topology() {
    let invalid_cores = [
        vec![ColMajorArray::new(vec![1.0], vec![1, 1]).unwrap()],
        vec![core([1, 0, 1], vec![])],
        vec![core([2, 1, 1], vec![1.0, 2.0])],
        vec![core([1, 1, 2], vec![1.0, 2.0])],
        vec![core([1, 1, 2], vec![1.0, 2.0]), core([1, 1, 1], vec![3.0])],
    ];
    for cores in invalid_cores {
        assert!(matches!(
            TreeTN::from_chain_cores(cores),
            Err(TreeTNOperationError::InvalidConfiguration { .. })
        ));
    }

    let (tree, sites) = TreeTN::from_chain_cores(vec![
        core([1, 1, 1], vec![1.0]),
        core([1, 1, 1], vec![2.0]),
        core([1, 1, 1], vec![3.0]),
    ])
    .unwrap();
    assert!(tree.chain_cores::<f64>(&sites[..2]).is_err());

    let mut non_chain = tree;
    non_chain.rename_node(&1, 3).unwrap();
    non_chain.rename_node(&2, 1).unwrap();
    non_chain.rename_node(&3, 2).unwrap();
    assert!(matches!(
        non_chain.chain_cores::<f64>(&[sites[0].clone(), sites[2].clone(), sites[1].clone()]),
        Err(TreeTNOperationError::InvalidGraph { .. })
    ));
}

#[test]
fn readout_refuses_structured_cores() {
    let (site, other) = (DynIndex::new_dyn(2), DynIndex::new_dyn(2));
    let bond = DynIndex::new_dyn(2);
    let left = IdxTensor::from_diag(vec![site.clone(), bond.clone()], vec![1.0, 2.0]).unwrap();
    let right =
        IdxTensor::from_dense_compact(vec![bond, other.clone()], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let chain = TreeTN::from_tensors(vec![left, right], vec![0, 1]).unwrap();
    assert!(matches!(
        chain.chain_cores::<f64>(&[site, other]),
        Err(TreeTNOperationError::InvalidConfiguration { .. })
    ));
}

#[test]
fn compression_applies_relative_singular_value_thresholds_and_caps() {
    let identity = core([1, 3, 3], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    let diagonal = core([3, 3, 1], vec![4.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
    let (tree, sites) = TreeTN::from_chain_cores(vec![identity, diagonal]).unwrap();
    let mut backend = CpuBackend::with_threads(1).unwrap();

    for (tolerance, cap, rank, expected_middle) in [
        (0.0, None, 3, 1.0),
        (0.0, Some(3), 3, 1.0),
        (0.1, None, 2, 1.0),
        (0.25, None, 2, 1.0),
        (0.3, None, 1, 0.0),
        (0.1, Some(1), 1, 0.0),
    ] {
        let compressed = {
            let mut session = Session::new(backend.engine_mut());
            tree.clone()
                .compress_chain_on_session(tolerance, cap, &mut session)
                .unwrap()
        };
        assert_eq!(
            compressed.chain_cores::<f64>(&sites).unwrap()[0].shape(),
            &[1, 3, rank]
        );
        assert_eq!(compressed.canonical_form(), Some(CanonicalForm::Unitary));
        assert_eq!(compressed.canonical_region().len(), 1);
        assert!(compressed.canonical_region().contains(&0));
        let expected = IdxTensor::from_dense(
            sites.clone(),
            vec![4.0, 0.0, 0.0, 0.0, expected_middle, 0.0, 0.0, 0.0, 0.0],
        )
        .unwrap();
        let actual = compressed.contract_to_tensor().unwrap();
        let residual = expected.sub(&actual).unwrap().maxabs().unwrap();
        assert!(residual < 1e-12, "maxabs residual {residual:e}");
    }
}

#[test]
fn compression_preserves_a_complex_chain_without_truncation() {
    let values = |len| {
        (0..len)
            .map(|i| Complex64::new(f64::from(i + 1), f64::from(i % 3 - 1)))
            .collect()
    };
    let (tree, sites) = TreeTN::from_chain_cores(vec![
        core([1, 3, 2], values(6)),
        core([2, 2, 3], values(12)),
        core([3, 2, 1], values(6)),
    ])
    .unwrap();
    let expected = tree.contract_to_tensor().unwrap();
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let compressed = {
        let mut session = Session::new(backend.engine_mut());
        let compressed = tree
            .compress_chain_on_session(0.0, None, &mut session)
            .unwrap();
        compressed.chain_cores::<Complex64>(&sites).unwrap();
        compressed
    };
    let residual = expected
        .sub(&compressed.contract_to_tensor().unwrap())
        .unwrap()
        .maxabs()
        .unwrap();
    assert!(
        residual / expected.maxabs().unwrap() < 1e-12,
        "maxabs residual {residual:e}"
    );
}

#[test]
fn compression_validates_options_even_for_a_single_core() {
    let cores = vec![core([1, 2, 1], vec![1.0, 2.0])];
    let (tree, sites) = TreeTN::from_chain_cores(cores.clone()).unwrap();
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let mut session = Session::new(backend.engine_mut());

    for (tolerance, cap) in [(f64::NAN, None), (-1.0, None), (0.0, Some(0))] {
        assert!(matches!(
            tree.clone()
                .compress_chain_on_session(tolerance, cap, &mut session),
            Err(TreeTNOperationError::InvalidConfiguration { .. })
        ));
    }
    let unchanged = tree
        .compress_chain_on_session(0.1, Some(1), &mut session)
        .unwrap();
    assert_eq!(unchanged.chain_cores::<f64>(&sites).unwrap(), cores);
    assert!(!unchanged.is_canonicalized());

    let (empty, empty_sites) = TreeTN::from_chain_cores::<f64>(Vec::new()).unwrap();
    let empty = empty
        .compress_chain_on_session(0.1, Some(1), &mut session)
        .unwrap();
    assert!(empty.chain_cores::<f64>(&empty_sites).unwrap().is_empty());
}
