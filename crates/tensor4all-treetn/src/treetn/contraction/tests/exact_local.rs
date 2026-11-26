//! Exact cleanup shares zip-up's topology plan without factorization.

use super::*;

#[test]
fn exact_local_checks_product_bond_dimensions_before_allocation() {
    use super::super::exact_local::checked_product_bond_dimension;
    assert_eq!(
        checked_product_bond_dimension(usize::MAX, 1).unwrap(),
        usize::MAX
    );
    assert!(matches!(
        checked_product_bond_dimension(usize::MAX, 2),
        Err(TreeTNOperationError::InvalidConfiguration { .. })
    ));
}

fn assert_exact_topology(
    a: &TreeTN<IdxTensor, String>,
    b: &TreeTN<IdxTensor, String>,
    center: &str,
    expected_nodes: &[&str],
) {
    let center = center.to_string();
    let reference = a
        .to_dense()
        .unwrap()
        .contract_pair(&b.to_dense().unwrap())
        .unwrap();
    for mode in [
        ContractionTopology::PruneScalarSubtrees,
        ContractionTopology::PreserveInputTopology,
    ] {
        let actual = contract_local_exact(a, b, &center, mode).unwrap();
        let expected: HashSet<_> = if mode == ContractionTopology::PruneScalarSubtrees {
            expected_nodes.iter().map(|node| node.to_string()).collect()
        } else {
            a.node_names().into_iter().collect()
        };
        assert_eq!(
            actual.node_names().into_iter().collect::<HashSet<_>>(),
            expected
        );
        let error = actual
            .to_dense()
            .unwrap()
            .sub(&reference)
            .unwrap()
            .maxabs()
            .unwrap()
            / reference.maxabs().unwrap().max(1.0);
        assert!(
            error < 1e-12,
            "exact local relative max-absolute error {error:e}"
        );
        assert!(actual.canonical_region().is_empty());
        assert_eq!(actual.canonical_form(), None);
        actual.validate_ortho_consistency().unwrap();
        for (left, right) in actual.site_index_network().edges() {
            let dim = |tree: &TreeTN<IdxTensor, String>| {
                tree.bond_index(tree.edge_between(&left, &right).unwrap())
                    .unwrap()
                    .dim()
            };
            assert_eq!(dim(&actual), dim(a) * dim(b));
        }
        let compressed = if mode == ContractionTopology::PruneScalarSubtrees {
            a.contract_zipup(b, &center, Some(SvdTruncationPolicy::new(0.0)), None)
                .unwrap()
        } else {
            a.contract_zipup_preserving_topology_with(
                b,
                &center,
                CanonicalForm::Unitary,
                Some(SvdTruncationPolicy::new(0.0)),
                None,
            )
            .unwrap()
        };
        assert_eq!(
            compressed.node_names().into_iter().collect::<HashSet<_>>(),
            expected
        );
    }
}

#[test]
fn exact_local_topology_covers_chain_terminal_blocks_bridges_and_scalar_centers() {
    for (outputs, center, expected) in [
        (vec![false, true, true], "S2", vec!["S1", "S2"]),
        (vec![false, true, true], "S0", vec!["S1", "S2"]),
        (
            vec![true, false, false, false],
            "S3",
            vec!["S0", "S1", "S2"],
        ),
        (vec![true, false, false, false], "S0", vec!["S0"]),
        (vec![true, false, true], "S1", vec!["S0", "S1", "S2"]),
        (vec![true, false], "S1", vec!["S0"]),
        (vec![false, false, false], "S0", vec!["S0"]),
        (vec![false, false, false], "S1", vec!["S1"]),
        (vec![false, false, false], "S2", vec!["S2"]),
        (vec![false], "S0", vec!["S0"]),
    ] {
        let (a, b) = make_chain_pair_with_outputs(&outputs);
        assert_exact_topology(&a, &b, center, &expected);
    }
}

#[test]
fn exact_local_prunes_a_branched_scalar_subtree_but_keeps_scalar_bridges() {
    let shared: Vec<_> = (0..6).map(|_| DynIndex::new_dyn(2)).collect();
    let output = DynIndex::new_dyn(2);
    let build = |offset: f64| {
        let bonds: Vec<_> = (0..5).map(|_| DynIndex::new_dyn(2)).collect();
        let mut indices: Vec<Vec<_>> = shared.iter().map(|index| vec![index.clone()]).collect();
        // 0--1--2 and 0--3--4. Output at 2; 1 is a scalar bridge, 3/4
        // form a scalar-only subtree. Center 0 survives without local output.
        for (edge, (a, b)) in [(0, 1), (1, 2), (0, 3), (3, 4), (0, 5)]
            .into_iter()
            .enumerate()
        {
            indices[a].push(bonds[edge].clone());
            indices[b].push(bonds[edge].clone());
        }
        indices[2].push(output.clone());
        let tensors = indices
            .into_iter()
            .map(|indices| {
                let len = indices.iter().map(IndexLike::dim).product();
                IdxTensor::from_dense(
                    indices,
                    (0..len)
                        .map(|i| (i as f64 + offset).sin())
                        .collect::<Vec<_>>(),
                )
                .unwrap()
            })
            .collect();
        TreeTN::from_tensors(tensors, (0..6).map(|i| format!("N{i}")).collect()).unwrap()
    };
    let a = build(0.2);
    let mut b = build(0.4);
    // Keep a true output at node 2 instead of contracting it with its twin.
    b = b.replaceind(&output, &output.prime()).unwrap();
    assert_exact_topology(&a, &b, "N0", &["N0", "N1", "N2"]);
}

#[test]
fn exact_local_validates_tree_center_and_full_site_assignment_before_contraction() {
    let (a, b) = make_chain_pair_with_outputs(&[true, true]);
    let mode = ContractionTopology::PruneScalarSubtrees;
    let empty = TreeTN::<IdxTensor, String>::new();
    for error in [
        contract_local_exact(&empty, &empty, &"S0".to_string(), mode).unwrap_err(),
        contract_local_exact(&a, &b, &"absent".to_string(), mode).unwrap_err(),
        contract_local_exact(&a, &empty, &"S0".to_string(), mode).unwrap_err(),
    ] {
        assert!(matches!(error, TreeTNOperationError::InvalidGraph { .. }));
    }
    let x = DynIndex::new_dyn(2);
    let y = x.prime();
    let make = |sites: [DynIndex; 2]| {
        let bond = DynIndex::new_dyn(1);
        TreeTN::from_tensors(
            vec![
                IdxTensor::from_dense(vec![sites[0].clone(), bond.clone()], vec![1.0, 2.0])
                    .unwrap(),
                IdxTensor::from_dense(vec![bond, sites[1].clone()], vec![3.0, 4.0]).unwrap(),
            ],
            vec!["A".to_string(), "B".to_string()],
        )
        .unwrap()
    };
    let left = make([x.clone(), y.clone()]);
    let right = make([y, x]);
    assert!(matches!(
        contract_local_exact(&left, &right, &"A".to_string(), mode),
        Err(TreeTNOperationError::InvalidGraph { .. })
    ));
}

/// Metadata-only directed index: IdxTensor's numerical backend is undirected.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct DirectedIndex {
    base: DynIndex,
    state: tensor4all_core::ConjState,
}

impl IndexLike for DirectedIndex {
    type Id = <DynIndex as IndexLike>::Id;
    fn id(&self) -> &Self::Id {
        self.base.id()
    }
    fn dim(&self) -> usize {
        self.base.dim()
    }
    fn plev(&self) -> i64 {
        self.base.plev()
    }
    fn conj_state(&self) -> tensor4all_core::ConjState {
        self.state
    }
    fn conj(&self) -> Self {
        use tensor4all_core::ConjState::*;
        Self {
            base: self.base.clone(),
            state: match self.state {
                Ket => Bra,
                Bra => Ket,
                Undirected => Undirected,
            },
        }
    }
    fn sim(&self) -> Self {
        Self {
            base: self.base.sim(),
            state: self.state,
        }
    }
    fn create_dummy_link_pair() -> (Self, Self) {
        let (a, b) = DynIndex::create_dummy_link_pair();
        (
            Self {
                base: a,
                state: tensor4all_core::ConjState::Undirected,
            },
            Self {
                base: b,
                state: tensor4all_core::ConjState::Undirected,
            },
        )
    }
    fn new_link(dim: usize) -> anyhow::Result<Self> {
        Ok(Self {
            base: DynIndex::new_link(dim)?,
            state: tensor4all_core::ConjState::Undirected,
        })
    }
    fn product_link(indices: &[Self]) -> anyhow::Result<Self> {
        let bases: Vec<_> = indices.iter().map(|index| index.base.clone()).collect();
        Ok(Self {
            base: DynIndex::product_link(&bases)?,
            state: tensor4all_core::ConjState::Undirected,
        })
    }
}

#[test]
fn exact_local_directed_metadata_preserves_legacy_chain_routing() {
    use super::super::exact_local::{distinct_site_spaces, output_nodes};
    use tensor4all_core::ConjState;
    let ket = DirectedIndex {
        base: DynIndex::new_dyn(2),
        state: ConjState::Ket,
    };
    let bra = ket.conj();
    let ket_sites = HashSet::from([ket.clone()]);
    let bra_sites = HashSet::from([bra.clone()]);
    assert!(
        dense_contract_output_indices(std::slice::from_ref(&ket), std::slice::from_ref(&bra))
            .is_empty()
    );
    // Legacy zip-up compares full sets when choosing its numerical schedule,
    // so directed bra/ket operands still take the chain route in this case.
    assert!(distinct_site_spaces(Some(&ket_sites), Some(&bra_sites)));
    let (a, _) = make_chain_pair_with_outputs(&[false, false, false]);
    let center = "S1".to_string();
    let chain = if distinct_site_spaces(Some(&ket_sites), Some(&bra_sites)) {
        a.chain_order(&center)
    } else {
        None
    };
    let retained = output_nodes(
        &a,
        &center,
        ContractionTopology::PruneScalarSubtrees,
        chain.as_deref(),
        HashSet::new(),
    )
    .unwrap();
    assert_eq!(retained, HashSet::from(["S2".to_string()]));
    let preserve = output_nodes(
        &a,
        &center,
        ContractionTopology::PreserveInputTopology,
        chain.as_deref(),
        HashSet::new(),
    )
    .unwrap();
    assert_eq!(
        preserve,
        HashSet::from(["S0".to_string(), "S1".to_string(), "S2".to_string()])
    );
    let mut primed_bra = bra;
    primed_bra.base = primed_bra.base.prime();
    assert_eq!(
        dense_contract_output_indices(&[ket], &[primed_bra]).len(),
        2
    );
    assert!(!distinct_site_spaces(Some(&ket_sites), Some(&ket_sites)));
    assert!(!distinct_site_spaces(None, Some(&ket_sites)));
    assert!(distinct_site_spaces(Some(&ket_sites), None));
}
