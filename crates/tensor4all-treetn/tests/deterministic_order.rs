//! Regression tests for tensor4all-rs issue #791: index layouts and
//! floating-point results must not depend on `HashMap`/`HashSet` iteration
//! order.
//!
//! Every hash map or hash set gets a fresh random seed, so rebuilding the same
//! network many times in one process exposes any order taken from hash
//! iteration. Each test rebuilds identical inputs (the same index objects and
//! the same data) and asserts a fixed positional layout and bitwise-identical
//! values.

use std::collections::{BTreeMap, HashSet};

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensor4all_core::{DynIndex, IdxTensor, IndexLike, TensorIndex};
use tensor4all_treetn::{
    random_treetn, CanonicalizationOptions, LinkSpace, SiteIndexNetwork, TreeTN, TruncationOptions,
};

/// Number of rebuilds per check. Each rebuild draws fresh hasher seeds.
const REBUILDS: usize = 64;

type Tn = TreeTN<IdxTensor, usize>;

/// A tree topology together with the order in which nodes are supplied to
/// `from_tensors`.
struct Topology {
    name: &'static str,
    /// Node names in the order the tensors are passed to `from_tensors`.
    input_order: Vec<usize>,
    edges: Vec<(usize, usize)>,
    /// Number of site indices per node, indexed by node name.
    sites_per_node: Vec<usize>,
    /// A node used as canonical center.
    center: usize,
}

/// Star: hub 0 (two site indices) with leaves 1, 2, 3; hub supplied second.
fn star() -> Topology {
    Topology {
        name: "star",
        input_order: vec![1, 0, 3, 2],
        edges: vec![(0, 1), (0, 2), (0, 3)],
        sites_per_node: vec![2, 1, 1, 1],
        center: 0,
    }
}

/// Tree with two degree-3 nodes (1 and 3); node 3 carries two site indices.
///
/// ```text
///   0       4
///    \     /
///     1 - 3
///    /     \
///   2       5
/// ```
fn two_hub_tree() -> Topology {
    Topology {
        name: "two_hub_tree",
        input_order: vec![4, 1, 5, 0, 3, 2],
        edges: vec![(0, 1), (1, 2), (1, 3), (3, 4), (3, 5)],
        sites_per_node: vec![1, 1, 1, 2, 1, 1],
        center: 3,
    }
}

/// Chain 0 - 1 - 2 - 3 - 4 supplied out of path order; node 2 has two sites.
fn scrambled_chain() -> Topology {
    Topology {
        name: "scrambled_chain",
        input_order: vec![2, 0, 4, 1, 3],
        edges: vec![(0, 1), (1, 2), (2, 3), (3, 4)],
        sites_per_node: vec![1, 1, 2, 1, 1],
        center: 2,
    }
}

fn topologies() -> Vec<Topology> {
    vec![star(), two_hub_tree(), scrambled_chain()]
}

/// Site indices shared by every network built on `topology`, indexed by node.
fn make_sites(topology: &Topology) -> Vec<Vec<DynIndex>> {
    topology
        .sites_per_node
        .iter()
        .map(|&count| (0..count).map(|_| DynIndex::new_dyn(2)).collect())
        .collect()
}

/// Input tensors (in `input_order`) of one network on `topology`.
///
/// Every call creates fresh bond indices, so several networks built from the
/// same `sites` can be added. Site and bond legs are interleaved so that the
/// tensor's leg order differs from both "sites first" and any sorted order.
fn make_tensors(
    topology: &Topology,
    sites: &[Vec<DynIndex>],
    rng: &mut ChaCha8Rng,
) -> Vec<IdxTensor> {
    let bonds: Vec<DynIndex> = (0..topology.edges.len())
        .map(|edge| DynIndex::new_dyn(2 + edge % 2))
        .collect();
    topology
        .input_order
        .iter()
        .map(|&node| {
            let node_bonds: Vec<DynIndex> = topology
                .edges
                .iter()
                .zip(&bonds)
                .filter(|((a, b), _)| *a == node || *b == node)
                .map(|(_, bond)| bond.clone())
                .collect();
            let mut legs = Vec::new();
            let mut site_iter = sites[node].iter().rev();
            let mut bond_iter = node_bonds.iter();
            loop {
                let bond = bond_iter.next();
                let site = site_iter.next();
                if bond.is_none() && site.is_none() {
                    break;
                }
                legs.extend(bond.cloned());
                legs.extend(site.cloned());
            }
            IdxTensor::random::<f64, _>(rng, legs).unwrap()
        })
        .collect()
}

fn build(topology: &Topology, tensors: &[IdxTensor]) -> Tn {
    TreeTN::from_tensors(tensors.to_vec(), topology.input_order.clone()).unwrap()
}

/// The site legs of the node tensor `tensor`, in the tensor's own leg order.
fn site_legs_of(tensor: &IdxTensor, sites: &[Vec<DynIndex>]) -> Vec<DynIndex> {
    let all_sites: HashSet<&DynIndex> = sites.iter().flatten().collect();
    tensor
        .indices()
        .iter()
        .filter(|index| all_sites.contains(index))
        .cloned()
        .collect()
}

/// A leg of a node tensor, described without index IDs so that layouts of
/// separately built networks (with freshly generated bond IDs) can be compared.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Leg {
    Site(DynIndex),
    Bond { neighbor: usize, dim: usize },
}

/// Positional leg layout of every node tensor, in `node_names()` order.
fn layout(tn: &Tn) -> Vec<(usize, Vec<Leg>)> {
    tn.node_names()
        .into_iter()
        .map(|node| {
            let tensor = tn.tensor(tn.node_index(&node).unwrap()).unwrap();
            let site_space = tn.site_space(&node).unwrap();
            let legs = tensor
                .indices()
                .iter()
                .map(|index| {
                    if site_space.contains(index) {
                        return Leg::Site(index.clone());
                    }
                    let neighbor = tn
                        .site_index_network()
                        .neighbors(&node)
                        .find(|neighbor| {
                            let edge = tn.edge_between(&node, neighbor).unwrap();
                            tn.bond_index(edge).unwrap() == index
                        })
                        .unwrap();
                    Leg::Bond {
                        neighbor,
                        dim: index.dim(),
                    }
                })
                .collect();
            (node, legs)
        })
        .collect()
}

fn bits(tensor: &IdxTensor) -> Vec<u64> {
    tensor
        .to_vec::<f64>()
        .unwrap()
        .into_iter()
        .map(f64::to_bits)
        .collect()
}

/// Everything that must be reproducible about a network: node order, leg
/// layout, neighbor order, node-tensor bits and dense-contraction bits.
#[derive(Debug, PartialEq, Eq)]
struct Fingerprint {
    layout: Vec<(usize, Vec<Leg>)>,
    neighbors: Vec<(usize, Vec<usize>)>,
    node_bits: Vec<Vec<u64>>,
    external_indices: Vec<DynIndex>,
    dense_indices: Vec<DynIndex>,
    dense_bits: Vec<u64>,
}

fn fingerprint(tn: &Tn) -> Fingerprint {
    let dense = tn.to_dense().unwrap();
    Fingerprint {
        layout: layout(tn),
        neighbors: tn
            .node_names()
            .into_iter()
            .map(|node| (node, tn.site_index_network().neighbors(&node).collect()))
            .collect(),
        node_bits: tn
            .node_names()
            .into_iter()
            .map(|node| bits(tn.tensor(tn.node_index(&node).unwrap()).unwrap()))
            .collect(),
        external_indices: tn.external_indices(),
        dense_indices: dense.indices().to_vec(),
        dense_bits: bits(&dense),
    }
}

/// Assert that `make` produces the same fingerprint on every rebuild and return it.
fn assert_reproducible(what: &str, mut make: impl FnMut() -> Tn) -> Fingerprint {
    let reference = fingerprint(&make());
    for rebuild in 1..REBUILDS {
        let current = fingerprint(&make());
        assert_eq!(
            current, reference,
            "{what}: rebuild {rebuild} differs from the first build"
        );
    }
    reference
}

/// Expected `external_indices()` of a network built from `tensors`: nodes in
/// input order, each node's site legs in its tensor's leg order.
fn expected_external(tensors: &[IdxTensor], sites: &[Vec<DynIndex>]) -> Vec<DynIndex> {
    tensors
        .iter()
        .flat_map(|tensor| site_legs_of(tensor, sites))
        .collect()
}

/// Expected `to_dense()` index order: node names sorted, then each node's
/// site legs in its tensor's leg order.
fn expected_dense(
    topology: &Topology,
    tensors: &[IdxTensor],
    sites: &[Vec<DynIndex>],
) -> Vec<DynIndex> {
    let by_name: BTreeMap<usize, &IdxTensor> =
        topology.input_order.iter().copied().zip(tensors).collect();
    by_name
        .values()
        .flat_map(|tensor| site_legs_of(tensor, sites))
        .collect()
}

/// Expected layout of `a + b` (and of chained sums): nodes in `a`'s input
/// order, `a`'s site legs in `a`'s leg order, then bonds by neighbor name.
fn expected_sum_layout(
    topology: &Topology,
    a_tensors: &[IdxTensor],
    sites: &[Vec<DynIndex>],
    bond_dims: impl Fn(usize, usize) -> usize,
) -> Vec<(usize, Vec<Leg>)> {
    topology
        .input_order
        .iter()
        .zip(a_tensors)
        .map(|(&node, tensor)| {
            let mut legs: Vec<Leg> = site_legs_of(tensor, sites)
                .into_iter()
                .map(Leg::Site)
                .collect();
            let mut neighbors: Vec<usize> = topology
                .edges
                .iter()
                .filter_map(|&(a, b)| match (a == node, b == node) {
                    (true, _) => Some(b),
                    (_, true) => Some(a),
                    _ => None,
                })
                .collect();
            neighbors.sort();
            legs.extend(neighbors.into_iter().map(|neighbor| Leg::Bond {
                neighbor,
                dim: bond_dims(node, neighbor),
            }));
            (node, legs)
        })
        .collect()
}

/// Bond dimension of edge `(u, v)` in one network from `make_tensors`.
fn edge_dim(topology: &Topology, u: usize, v: usize) -> usize {
    let edge = topology
        .edges
        .iter()
        .position(|&(a, b)| (a, b) == (u, v) || (a, b) == (v, u))
        .unwrap();
    2 + edge % 2
}

#[test]
fn from_tensors_layout_and_dense_are_reproducible() {
    let mut rng = ChaCha8Rng::seed_from_u64(791);
    for topology in topologies() {
        let sites = make_sites(&topology);
        let tensors = make_tensors(&topology, &sites, &mut rng);

        let reference = assert_reproducible(topology.name, || build(&topology, &tensors));

        // The layout is not only stable but meaningful: it follows the input.
        assert_eq!(
            reference.external_indices,
            expected_external(&tensors, &sites),
            "{}: external_indices must list nodes in input order and each node's \
             site legs in its tensor's leg order",
            topology.name
        );
        assert_eq!(
            reference.dense_indices,
            expected_dense(&topology, &tensors, &sites),
            "{}: to_dense must order nodes by name and site legs by tensor leg order",
            topology.name
        );
        let tn = build(&topology, &tensors);
        assert_eq!(tn.node_names(), topology.input_order, "{}", topology.name);
        assert_eq!(
            tn.all_site_indices().unwrap().0,
            reference.external_indices,
            "{}: all_site_indices must match external_indices",
            topology.name
        );
        for (node, tensor) in topology.input_order.iter().zip(&tensors) {
            assert_eq!(
                tn.node_site_indices(node).unwrap(),
                site_legs_of(tensor, &sites),
                "{}: node {node}",
                topology.name
            );
        }
        // Node tensors are stored unchanged, in input order.
        let input_bits: Vec<Vec<u64>> = tensors.iter().map(bits).collect();
        assert_eq!(reference.node_bits, input_bits, "{}", topology.name);
    }
}

#[test]
fn add_and_chained_add_have_a_fixed_topology_only_bond_layout() {
    let mut rng = ChaCha8Rng::seed_from_u64(7910);
    for topology in topologies() {
        let sites = make_sites(&topology);
        let a = make_tensors(&topology, &sites, &mut rng);
        let b = make_tensors(&topology, &sites, &mut rng);
        let c = make_tensors(&topology, &sites, &mut rng);

        let sum = assert_reproducible(&format!("{} a+b", topology.name), || {
            build(&topology, &a).add(&build(&topology, &b)).unwrap()
        });
        assert_eq!(
            sum.layout,
            expected_sum_layout(&topology, &a, &sites, |u, v| 2 * edge_dim(&topology, u, v)),
            "{}: a + b node tensors must be [a's site legs, bonds by neighbor name]",
            topology.name
        );

        let chained = assert_reproducible(&format!("{} (a+b)+c", topology.name), || {
            build(&topology, &a)
                .add(&build(&topology, &b))
                .unwrap()
                .add(&build(&topology, &c))
                .unwrap()
        });
        assert_eq!(
            chained.layout,
            expected_sum_layout(&topology, &a, &sites, |u, v| 3 * edge_dim(&topology, u, v)),
            "{}: chained sums must keep the same layout",
            topology.name
        );
        assert_eq!(chained.external_indices, expected_external(&a, &sites));
        assert_eq!(chained.dense_indices, expected_dense(&topology, &a, &sites));
    }
}

#[test]
fn canonicalize_and_truncate_are_bitwise_reproducible() {
    let mut rng = ChaCha8Rng::seed_from_u64(79101);
    for topology in topologies() {
        let sites = make_sites(&topology);
        let a = make_tensors(&topology, &sites, &mut rng);
        let b = make_tensors(&topology, &sites, &mut rng);
        let sum = || build(&topology, &a).add(&build(&topology, &b)).unwrap();

        assert_reproducible(&format!("{} canonicalize", topology.name), || {
            sum()
                .canonicalize([topology.center], CanonicalizationOptions::default())
                .unwrap()
        });
        let truncated = assert_reproducible(&format!("{} truncate", topology.name), || {
            sum()
                .truncate(
                    [topology.center],
                    TruncationOptions::default().with_max_bond_dim(2),
                )
                .unwrap()
        });
        // Truncation really happened (the sums have bond dimension >= 4).
        let tn = sum()
            .truncate(
                [topology.center],
                TruncationOptions::default().with_max_bond_dim(2),
            )
            .unwrap();
        assert!(tn.link_dims().iter().all(|&dim| dim <= 2));
        assert_eq!(
            truncated.dense_indices,
            expected_dense(&topology, &a, &sites)
        );
    }
}

#[test]
fn canonicalization_edge_order_breaks_distance_ties_deterministically() {
    let mut rng = ChaCha8Rng::seed_from_u64(79102);
    let topology = star();
    let sites = make_sites(&topology);
    let tensors = make_tensors(&topology, &sites, &mut rng);

    let edges_for = |tn: &Tn| {
        tn.site_index_network()
            .edges_to_canonicalize_to_region_by_names(&HashSet::from([topology.center]))
            .unwrap()
    };
    let reference = edges_for(&build(&topology, &tensors));
    // All leaves are at distance one from the hub; ties follow NodeIndex order,
    // which is the input order of the leaves.
    assert_eq!(reference, vec![(1, 0), (3, 0), (2, 0)]);
    for _ in 1..REBUILDS {
        assert_eq!(edges_for(&build(&topology, &tensors)), reference);
    }
}

#[test]
fn contract_zipup_result_is_reproducible() {
    let mut rng = ChaCha8Rng::seed_from_u64(79103);
    for topology in [star(), two_hub_tree()] {
        let sites = make_sites(&topology);
        let state = make_tensors(&topology, &sites, &mut rng);
        // An operator-like network: each node carries the state's site legs
        // (contracted away) and one output site leg.
        let outputs: Vec<Vec<DynIndex>> = topology
            .sites_per_node
            .iter()
            .map(|_| vec![DynIndex::new_dyn(2)])
            .collect();
        let op_sites: Vec<Vec<DynIndex>> = sites
            .iter()
            .zip(&outputs)
            .map(|(s, o)| s.iter().chain(o).cloned().collect())
            .collect();
        let operator = make_tensors(&topology, &op_sites, &mut rng);

        let result = assert_reproducible(&format!("{} contract_zipup", topology.name), || {
            build(&topology, &state)
                .contract_zipup(&build(&topology, &operator), &topology.center, None, None)
                .unwrap()
        });
        assert_eq!(
            result
                .layout
                .iter()
                .map(|(node, _)| *node)
                .collect::<Vec<_>>(),
            topology.input_order,
            "{}: the result keeps the first operand's node order",
            topology.name
        );
    }
}

#[test]
fn random_treetn_is_reproducible_for_a_fixed_seed() {
    let topology = two_hub_tree();
    let sites = make_sites(&topology);
    // Rebuild the site network every time: its site spaces are hash sets whose
    // iteration order changes with each new set.
    let network = || {
        let mut network = SiteIndexNetwork::<usize, DynIndex>::new();
        for &node in &topology.input_order {
            network
                .add_node(node, sites[node].iter().cloned().collect::<HashSet<_>>())
                .unwrap();
        }
        for &(a, b) in &topology.edges {
            network.add_edge(&a, &b).unwrap();
        }
        network
    };

    let make = || {
        let mut rng = ChaCha8Rng::seed_from_u64(79104);
        random_treetn::<f64, _, _>(&mut rng, &network(), LinkSpace::uniform(2)).unwrap()
    };
    let reference = assert_reproducible("random_treetn", make);
    assert_eq!(
        reference
            .layout
            .iter()
            .map(|(node, _)| *node)
            .collect::<Vec<_>>(),
        topology.input_order
    );
}
