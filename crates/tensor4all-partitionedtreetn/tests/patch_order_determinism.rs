//! Bitwise determinism of the patch and contribution order in
//! `PartitionedTreeTN` sums (#1084).
//!
//! `contract` adds the contributions of one output patch in the order of the
//! projected `DynIndex` ids, and those ids come from a thread-local RNG
//! (`generate_id`), so the same numerical data rounded differently in different
//! processes. The tests here pin the property the issue asks for: the same
//! numbers give bitwise identical primal values whatever the id order, the
//! patch construction order, or the process.
//!
//! The reproducer is the one from the issue: a two-node chain `A` over
//! `{k0, u0}` and `{k1, u1}` split into the four patches projected on
//! `(k0, k1)`, contracted against a two-node chain `B` over `{k0, v0}` and
//! `{k1, v1}` with one unprojected patch. All four pairs land on the same
//! output projector, so one output patch is the sum of four contributions
//! truncated once.

use std::process::Command;

use tensor4all_core::{DynId, DynIndex, IdxTensor, IndexLike, TagSet};
use tensor4all_partitionedtreetn::{PartitionedTreeTN, Projector, SubDomainTreeTN};
use tensor4all_treetn::{contraction::ContractionOptions, TreeTN};

/// Bond dimension of both chains in the reproducer.
const BOND: usize = 4;
/// Cases with explicit site ids: both id orders, varied insertion order.
const EXPLICIT_CASES: usize = 100;
/// Cases rebuilt on fresh random ids, as an ordinary program would build them.
const FRESH_ID_CASES: usize = 8;
/// Child-process marker for `contract_is_bitwise_identical_across_processes`.
const CHILD_ENV: &str = "T4A_PATCH_ORDER_CHILD";
/// Prefix of each digest line the child process prints.
const DIGEST_TAG: &str = "PATCH_ORDER_DIGEST";

/// Deterministic values in `[-1, 1)`, so the tests need no RNG crate.
struct Lcg(u64);

impl Lcg {
    fn next_value(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }

    /// A small non-negative integer, for permutations.
    fn below(&mut self, bound: usize) -> usize {
        (self.next_value().abs() * bound as f64) as usize % bound
    }
}

/// The six site indices of one build, in a fixed logical order.
struct Sites {
    k0: DynIndex,
    k1: DynIndex,
    u0: DynIndex,
    u1: DynIndex,
    v0: DynIndex,
    v1: DynIndex,
}

impl Sites {
    /// Fresh random ids (`None`), or explicit ids with a chosen order of `k0`
    /// and `k1` (`Some(true)`: `id(k0) < id(k1)`), which is the order the
    /// projection list is walked in.
    fn new(k0_first: Option<bool>) -> Self {
        let make = |fixed: u64| match k0_first {
            None => DynIndex::new_dyn(2),
            Some(_) => DynIndex::new_with_tags(DynId(fixed), 2, TagSet::new()),
        };
        let (id_k0, id_k1) = if k0_first == Some(false) {
            (2, 1)
        } else {
            (1, 2)
        };
        Sites {
            k0: make(id_k0),
            k1: make(id_k1),
            u0: make(3),
            u1: make(4),
            v0: make(5),
            v1: make(6),
        }
    }

    fn k0_first(&self) -> bool {
        self.k0.id().value() < self.k1.id().value()
    }
}

/// A two-node chain: node 0 carries `left`, node 1 carries `right`.
fn chain(left: [&DynIndex; 2], right: [&DynIndex; 2], rng: &mut Lcg) -> TreeTN<IdxTensor, usize> {
    let bond = DynIndex::new_dyn(BOND);
    let mut node = |mut indices: Vec<DynIndex>| {
        indices.push(bond.clone());
        let len: usize = indices.iter().map(|index| index.dim()).product();
        let values: Vec<f64> = (0..len).map(|_| rng.next_value()).collect();
        IdxTensor::from_dense(indices, values).expect("node tensor")
    };
    let n0 = node(left.map(Clone::clone).to_vec());
    let n1 = node(right.map(Clone::clone).to_vec());
    TreeTN::from_tensors(vec![n0, n1], vec![0usize, 1]).expect("chain")
}

/// Contract the reproducer once and return the output values in the canonical
/// site order `[u0, v0, u1, v1]` as raw bits, so rounding-level differences
/// show up and the incidental axis order does not.
fn contract_bits(sites: &Sites, patch_order: &[usize; 4]) -> Vec<u64> {
    let mut rng = Lcg(0x5eed);
    let a_full = chain([&sites.k0, &sites.u0], [&sites.k1, &sites.u1], &mut rng);
    let b_full = chain([&sites.k0, &sites.v0], [&sites.k1, &sites.v1], &mut rng);

    // The four `(k0, k1)` projections, created in the permuted order.
    let mut patches = Vec::with_capacity(4);
    for &slot in patch_order {
        let (x, y) = (slot / 2, slot % 2);
        let projector = Projector::from_pairs([(sites.k0.clone(), x), (sites.k1.clone(), y)])
            .expect("projector");
        patches.push(SubDomainTreeTN::new(a_full.clone(), projector).expect("patch"));
    }
    let a = PartitionedTreeTN::from_subdomains(patches).expect("A partition");
    let b =
        PartitionedTreeTN::from_subdomain(SubDomainTreeTN::from_treetn(b_full).expect("B patch"))
            .expect("B partition");

    let contracted = a
        .contract(&b, &0usize, ContractionOptions::default())
        .expect("contract");
    assert_eq!(contracted.len(), 1, "one output patch");
    let dense = contracted
        .to_treetn()
        .expect("to_treetn")
        .to_dense()
        .expect("dense");
    let canonical = dense
        .permute_indices(&[
            sites.u0.clone(),
            sites.v0.clone(),
            sites.u1.clone(),
            sites.v1.clone(),
        ])
        .expect("canonical site order");
    let bits: Vec<u64> = canonical
        .to_vec::<f64>()
        .expect("values")
        .iter()
        .map(|value| value.to_bits())
        .collect();
    assert_eq!(bits.len(), 16, "four sites of dimension two");
    bits
}

/// A shuffled `[0, 1, 2, 3]`, i.e. one patch insertion order.
fn shuffled_order(rng: &mut Lcg) -> [usize; 4] {
    let mut order = [0, 1, 2, 3];
    for position in (1..order.len()).rev() {
        let swap = rng.below(position + 1);
        order.swap(position, swap);
    }
    order
}

/// Hex encoding of the contracted bits, for cross-process comparison.
fn encode(bits: &[u64]) -> String {
    bits.iter()
        .map(|bits| format!("{bits:016x}"))
        .collect::<Vec<_>>()
        .join(",")
}

/// What the two-process test compares: one digest per fresh-id build.
fn fresh_id_digests() -> Vec<String> {
    (0..FRESH_ID_CASES)
        .map(|_| encode(&contract_bits(&Sites::new(None), &[0, 1, 2, 3])))
        .collect()
}

#[test]
fn contract_is_bitwise_identical_across_index_id_orders() {
    let mut reference: Option<Vec<u64>> = None;
    let mut cases = Lcg(0xc0ffee);

    // Explicit ids, so both id orders are covered deterministically rather than
    // by the luck of the thread-local RNG.
    let mut id_orders = [false, false];
    for case in 0..EXPLICIT_CASES {
        let k0_first = cases.below(2) == 0;
        let patch_order = shuffled_order(&mut cases);
        let sites = Sites::new(Some(k0_first));
        assert_eq!(sites.k0_first(), k0_first, "case {case}: explicit id order");
        id_orders[usize::from(k0_first)] = true;
        let bits = contract_bits(&sites, &patch_order);
        let baseline: &Vec<u64> = reference.get_or_insert_with(|| bits.clone());
        assert_eq!(
            &bits, baseline,
            "case {case}: id(k0) < id(k1) = {k0_first}, patch insertion order {patch_order:?}; \
             the same numbers must give bitwise identical output"
        );
    }
    assert_eq!(
        id_orders,
        [true, true],
        "both id orders must be covered: they are what the rounding difference followed"
    );

    // Fresh random ids, as an ordinary program allocates them.
    let expected = reference.expect("explicit cases ran");
    for case in 0..FRESH_ID_CASES {
        let sites = Sites::new(None);
        let bits = contract_bits(&sites, &[0, 1, 2, 3]);
        assert_eq!(
            &bits,
            &expected,
            "fresh-id case {case}: id(k0) < id(k1) = {}",
            sites.k0_first()
        );
    }
}

/// The child side of `contract_is_bitwise_identical_across_processes`: prints
/// one digest per fresh-id build. Without [`CHILD_ENV`] this test does nothing.
#[test]
fn print_patch_order_digest() {
    if std::env::var_os(CHILD_ENV).is_none() {
        return;
    }
    for digest in fresh_id_digests() {
        println!("{DIGEST_TAG} {digest}");
    }
}

/// The same program in two processes on the same machine and build gives the
/// same bits.
///
/// The ids are drawn from the process's thread-local RNG, so this is the only
/// check that the roundoff does not follow process-level state at all.
#[test]
fn contract_is_bitwise_identical_across_processes() {
    if std::env::var_os(CHILD_ENV).is_some() {
        // Running as the child of this test.
        return;
    }
    let local = fresh_id_digests();
    // The child filter is a substring of the child test name, so this works
    // whether the file is its own target or a module of `main`.
    let output = Command::new(std::env::current_exe().expect("the test binary"))
        .args(["print_patch_order_digest", "--nocapture"])
        .env(CHILD_ENV, "1")
        .output()
        .expect("the child test process starts");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "the child process failed:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    // The harness writes its own `test <name> ... ` announcement to the same stdout the
    // child's `--nocapture` output goes to, and the two can share a line, so the tag is
    // located rather than required at the start of one.
    let child: Vec<String> = stdout
        .lines()
        .filter_map(|line| {
            line.find(DIGEST_TAG)
                .map(|at| line[at + DIGEST_TAG.len()..].trim().to_owned())
        })
        .collect();
    assert_eq!(
        child.len(),
        local.len(),
        "the child printed {} digests, expected {}:\nstdout:\n{stdout}\nstderr:\n{stderr}",
        child.len(),
        local.len()
    );
    for (build, (local, child)) in local.iter().zip(&child).enumerate() {
        assert_eq!(
            local, child,
            "build {build}: two processes on the same build must give the same bits"
        );
    }
}

/// Summing patches must not densify a structured payload.
///
/// `TreeTN::add` keeps the diagonal payload of its operands. `contract` sums
/// its contribution groups through the same `TreeTN::add` path.
#[test]
fn to_treetn_keeps_a_diagonal_payload_structured() {
    let site = DynIndex::new_dyn(3);
    let aux = DynIndex::new_dyn(3);
    let masked = |values: [f64; 3], coordinate: usize| {
        let tensor = IdxTensor::from_diag(vec![site.clone(), aux.clone()], values.to_vec())
            .expect("diagonal tensor");
        let tree = TreeTN::from_tensors(vec![tensor], vec![0usize]).expect("tree");
        let projector = Projector::from_pairs([(site.clone(), coordinate)]).expect("projector");
        SubDomainTreeTN::new(tree, projector).expect("patch")
    };

    let partition = PartitionedTreeTN::from_subdomains(vec![
        masked([2.0, 3.0, 5.0], 0),
        masked([1.0, 1.0, 1.0], 1),
    ])
    .expect("partition");
    let summed = partition.to_treetn().expect("to_treetn");
    let tensor = summed
        .tensor(summed.node_index(&0usize).expect("node"))
        .expect("tensor");
    assert!(
        tensor.is_diag(),
        "the diagonal payload must survive the patch sum"
    );
    // Masking keeps coordinate 0 of the first patch and coordinate 1 of the
    // second, so the sum is diag(2, 1, 0) and its squared norm is 5.
    let values = summed
        .to_dense()
        .expect("dense")
        .to_vec::<f64>()
        .expect("values");
    let norm_squared: f64 = values.iter().map(|value| value * value).sum();
    assert!(
        (norm_squared - 5.0).abs() < 1.0e-12,
        "unexpected summed norm {norm_squared}"
    );
}
