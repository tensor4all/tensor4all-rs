//! Output selection in partition contraction (`contract_selected`,
//! `contract_adaptive_selected`).
//!
//! The fixture is a block product `C = A * B` over branch-like indices of
//! dimension 3: `A` has one patch per block `(row, mid)`, `B` one per block
//! `(mid, col)`, and contracting `mid` gives 27 compatible pairs and 9 output
//! blocks `(row, col)`. Each block is a two-node chain whose qubit sites
//! multiply as matrix-product operators.

use super::*;
use crate::patching::{contract_adaptive, contract_adaptive_selected, BudgetAllocation};
use crate::PatchingOptions;
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_treetn::TreeTN;

const BRANCHES: usize = 3;
const QUBITS: usize = 2;
const BOND: usize = 2;

/// Deterministic values in [-1, 1), so the fixture needs no RNG crate.
struct Lcg(u64);

impl Lcg {
    fn next_value(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

struct BlockProduct {
    a: PartitionedTreeTN,
    b: PartitionedTreeTN,
    row: DynIndex,
    mid: DynIndex,
    col: DynIndex,
}

impl BlockProduct {
    /// Build the fixture; every `A` patch in row `small_row` is scaled by
    /// `scale`, so the output blocks of that row have a much smaller norm.
    fn new(small_row: Option<usize>, scale: f64) -> Self {
        let mut rng = Lcg(0x5eed_1048);
        let row = DynIndex::new_dyn(BRANCHES);
        let mid = DynIndex::new_dyn(BRANCHES);
        let col = DynIndex::new_dyn(BRANCHES);
        let out_a: Vec<_> = (0..QUBITS).map(|_| DynIndex::new_dyn(2)).collect();
        let shared: Vec<_> = (0..QUBITS).map(|_| DynIndex::new_dyn(2)).collect();
        let out_b: Vec<_> = (0..QUBITS).map(|_| DynIndex::new_dyn(2)).collect();

        let mut a_sites: Vec<Vec<DynIndex>> = (0..QUBITS)
            .map(|k| vec![out_a[k].clone(), shared[k].clone()])
            .collect();
        a_sites[0].extend([row.clone(), mid.clone()]);
        let mut b_sites: Vec<Vec<DynIndex>> = (0..QUBITS)
            .map(|k| vec![shared[k].clone(), out_b[k].clone()])
            .collect();
        b_sites[0].extend([mid.clone(), col.clone()]);

        let a = block_patches(&a_sites, (&row, &mid), &mut rng, |r, _| {
            if small_row == Some(r) {
                scale
            } else {
                1.0
            }
        });
        let b = block_patches(&b_sites, (&mid, &col), &mut rng, |_, _| 1.0);
        Self {
            a,
            b,
            row,
            mid,
            col,
        }
    }

    fn key(&self, (r, s): (usize, usize)) -> Projector {
        Projector::from_pairs([(self.row.clone(), r), (self.col.clone(), s)]).unwrap()
    }

    fn keys(&self, blocks: &[(usize, usize)]) -> Vec<Projector> {
        blocks.iter().map(|&block| self.key(block)).collect()
    }

    /// Caller-side reference for one block: contract the `A` patches of row
    /// `r` with the `B` patches of column `s`.
    fn per_block(&self, (r, s): (usize, usize)) -> SubDomainTreeTN {
        let a_row = select(&self.a, &self.row, r);
        let b_col = select(&self.b, &self.col, s);
        let out = a_row
            .contract(&b_col, &0, ContractionOptions::default())
            .unwrap();
        out.get(&self.key((r, s))).unwrap().clone()
    }
}

/// One chain per block `(i, j)` of `coordinates`, with independent random
/// values times `scale(i, j)`, masked to that block.
fn block_patches(
    sites: &[Vec<DynIndex>],
    (first, second): (&DynIndex, &DynIndex),
    rng: &mut Lcg,
    scale: impl Fn(usize, usize) -> f64,
) -> PartitionedTreeTN {
    let mut patches = Vec::new();
    for i in 0..BRANCHES {
        for j in 0..BRANCHES {
            let data = chain(sites, rng, scale(i, j));
            let projector =
                Projector::from_pairs([(first.clone(), i), (second.clone(), j)]).unwrap();
            patches.push(SubDomainTreeTN::new(data, projector).unwrap());
        }
    }
    PartitionedTreeTN::from_subdomains(patches).unwrap()
}

fn chain(sites: &[Vec<DynIndex>], rng: &mut Lcg, scale: f64) -> TreeTN<IdxTensor, usize> {
    let n = sites.len();
    let bonds: Vec<_> = (1..n).map(|_| DynIndex::new_dyn(BOND)).collect();
    let tensors = sites
        .iter()
        .enumerate()
        .map(|(k, site_indices)| {
            let mut indices = site_indices.clone();
            if k > 0 {
                indices.push(bonds[k - 1].clone());
            }
            if k + 1 < n {
                indices.push(bonds[k].clone());
            }
            let len: usize = indices.iter().map(|index| index.dim).product();
            let values: Vec<f64> = (0..len).map(|_| scale * rng.next_value()).collect();
            IdxTensor::from_dense(indices, values).unwrap()
        })
        .collect();
    TreeTN::from_tensors(tensors, (0..n).collect()).unwrap()
}

fn select(patches: &PartitionedTreeTN, index: &DynIndex, coordinate: usize) -> PartitionedTreeTN {
    let kept = patches
        .values()
        .filter(|patch| patch.projector().get(index) == Some(coordinate))
        .cloned()
        .collect();
    PartitionedTreeTN::from_subdomains(kept).unwrap()
}

fn dense_difference(left: &SubDomainTreeTN, right: &SubDomainTreeTN) -> f64 {
    left.data()
        .to_dense()
        .unwrap()
        .sub(&right.data().to_dense().unwrap())
        .unwrap()
        .maxabs()
        .unwrap()
}

fn dense_maxabs(patch: &SubDomainTreeTN) -> f64 {
    patch.data().to_dense().unwrap().maxabs().unwrap()
}

fn sorted(mut projectors: Vec<Projector>) -> Vec<Projector> {
    projectors.sort_by(|a, b| a.canonical_cmp(b));
    projectors.dedup();
    projectors
}

/// Run `operation` and return its result with the number of top-level pairs
/// it contracted.
fn count_pairs<T>(operation: impl FnOnce() -> T) -> (T, usize) {
    CONTRACTED_PAIRS.with(|count| count.set(0));
    let result = operation();
    (result, CONTRACTED_PAIRS.with(|count| count.get()))
}

fn all_blocks() -> Vec<(usize, usize)> {
    (0..BRANCHES)
        .flat_map(|r| (0..BRANCHES).map(move |s| (r, s)))
        .collect()
}

#[test]
fn contract_selected_contracts_only_pairs_of_wanted_outputs() {
    let product = BlockProduct::new(None, 1.0);
    let options = ContractionOptions::default();

    let (full, full_pairs) = count_pairs(|| product.a.contract(&product.b, &0, options.clone()));
    assert_eq!(full_pairs, 27);
    assert_eq!(full.unwrap().len(), 9);

    let all = all_blocks();
    let cases: [(&[(usize, usize)], usize); 4] = [
        (&all, 27),
        (&[(1, 1)], 3),
        (&[(0, 0), (1, 1)], 6),
        (&[(0, 0), (1, 1), (2, 0)], 9),
    ];
    for (blocks, expected_pairs) in cases {
        let wanted = product.keys(blocks);
        let (selected, pairs) = count_pairs(|| {
            product
                .a
                .contract_selected(&product.b, &0, options.clone(), &wanted)
        });
        let selected = selected.unwrap();
        assert_eq!(pairs, expected_pairs, "pairs for {blocks:?}");
        assert_eq!(
            sorted(selected.projectors().cloned().collect()),
            sorted(wanted),
            "output keys for {blocks:?}"
        );
    }
}

#[test]
fn contract_selected_blocks_equal_full_contract_and_per_block_reference() {
    let product = BlockProduct::new(None, 1.0);
    let options = ContractionOptions::default();
    let full = product.a.contract(&product.b, &0, options.clone()).unwrap();

    // A non-rectangular selection: rows {0, 1, 2} x columns {0, 1} would
    // also form (0, 1), (1, 0), (2, 1).
    let blocks = [(0, 0), (1, 1), (2, 0)];
    let selected = product
        .a
        .contract_selected(&product.b, &0, options, &product.keys(&blocks))
        .unwrap();
    for block in blocks {
        let key = product.key(block);
        let patch = selected.get(&key).unwrap();
        let full_patch = full.get(&key).unwrap();
        let scale = dense_maxabs(full_patch);
        assert!(scale > 0.1, "block {block:?} has scale {scale}");
        assert_eq!(patch.all_indices(), full_patch.all_indices());
        // Same pairs, summed in the same order as the full call.
        assert_eq!(dense_difference(patch, full_patch), 0.0, "block {block:?}");
        let reference = dense_difference(patch, &product.per_block(block));
        assert!(
            reference <= 1.0e-12 * scale,
            "block {block:?}: max |selected - per-block reference| = {reference:e}"
        );
    }
}

#[test]
fn predicted_output_projector_equals_the_contracted_pair_projector() {
    // `DynIndex` is undirected, so `is_contractable` reduces to equality and
    // no index pair is contractable without being equal.
    let product = BlockProduct::new(None, 1.0);
    let options = ContractionOptions::default();
    let output_sites = contraction_output_site_indices(
        product.a.values().next().unwrap().data(),
        product.b.values().next().unwrap().data(),
    )
    .unwrap();
    assert!(!output_sites.contains(&product.mid));
    assert!(output_sites.contains(&product.row));
    assert!(output_sites.contains(&product.col));

    let pairs = product
        .a
        .contract_pairs(&product.b, &0, &options, None)
        .unwrap();
    assert_eq!(pairs.len(), 27);
    for (left, right, contribution) in pairs {
        let predicted = left
            .projector()
            .intersection(right.projector())
            .unwrap()
            .filter_indices(&output_sites);
        let r = left.projector().get(&product.row).unwrap();
        let s = right.projector().get(&product.col).unwrap();
        assert_eq!(contribution.projector(), &predicted);
        assert_eq!(contribution.projector(), &product.key((r, s)));
    }
}

#[test]
fn contract_selected_without_matching_pairs_returns_empty_after_operand_checks() {
    let product = BlockProduct::new(None, 1.0);
    let options = ContractionOptions::default();

    let (empty, pairs) = count_pairs(|| {
        product
            .a
            .contract_selected(&product.b, &0, options.clone(), &[])
    });
    assert!(empty.unwrap().is_empty());
    assert_eq!(pairs, 0);

    // A key on the contracted index matches no output projector.
    let contracted_key = Projector::from_pairs([(product.mid.clone(), 0)]).unwrap();
    let (none, pairs) = count_pairs(|| {
        product
            .a
            .contract_selected(&product.b, &0, options.clone(), &[contracted_key])
    });
    assert!(none.unwrap().is_empty());
    assert_eq!(pairs, 0);

    // Duplicate keys select the block once.
    let key = product.key((2, 2));
    let (duplicate, pairs) = count_pairs(|| {
        product
            .a
            .contract_selected(&product.b, &0, options.clone(), &[key.clone(), key])
    });
    assert_eq!(duplicate.unwrap().len(), 1);
    assert_eq!(pairs, 3);

    // The operand checks still run when nothing is selected.
    assert!(matches!(
        product
            .a
            .contract_selected(&product.b, &7, options.clone(), &[]),
        Err(PartitionedTreeTNError::InvalidCenter)
    ));
    assert!(matches!(
        PartitionedTreeTN::new().contract_selected(&product.b, &0, options, &[]),
        Err(PartitionedTreeTNError::Empty)
    ));
}

#[test]
fn contract_adaptive_selected_skips_unwanted_pairs_and_matches_the_full_call() {
    let product = BlockProduct::new(None, 1.0);
    let contraction = ContractionOptions::default();
    let patching = PatchingOptions {
        cutoff: 0.0,
        ..PatchingOptions::default()
    };

    let (full, full_pairs) = count_pairs(|| {
        contract_adaptive(&product.a, &product.b, &0, &contraction, &patching).unwrap()
    });
    assert_eq!(full_pairs, 27);
    let (all, all_pairs) = count_pairs(|| {
        contract_adaptive_selected(
            &product.a,
            &product.b,
            &0,
            &contraction,
            &patching,
            &product.keys(&all_blocks()),
        )
        .unwrap()
    });
    assert_eq!(all_pairs, 27);
    assert_eq!(all.len(), full.len());
    for (key, patch) in &full {
        assert_eq!(dense_difference(all.get(key).unwrap(), patch), 0.0);
    }

    let (one, one_pairs) = count_pairs(|| {
        contract_adaptive_selected(
            &product.a,
            &product.b,
            &0,
            &contraction,
            &patching,
            &[product.key((1, 1))],
        )
        .unwrap()
    });
    assert_eq!(one_pairs, 3);
    assert_eq!(one.len(), 1);
    let key = product.key((1, 1));
    // With cutoff 0 the selection does not change the block.
    assert_eq!(
        dense_difference(one.get(&key).unwrap(), full.get(&key).unwrap()),
        0.0
    );
}

#[test]
fn contract_adaptive_selected_budgets_over_the_selected_outputs() {
    // Row 2 of A is scaled by 1e-4, so blocks (2, s) carry about 1e-8 of the
    // squared norm of the other blocks. Under the volume allocation a patch is
    // dropped when ||F_p||^2 <= cutoff * ||F||^2 * V_p / V, with ||F||^2 and V
    // the totals of the selected outputs.
    let product = BlockProduct::new(Some(2), 1.0e-4);
    let contraction = ContractionOptions::default();
    let patching = PatchingOptions {
        cutoff: 1.0e-6,
        allocation: BudgetAllocation::Volume,
        ..PatchingOptions::default()
    };
    let small = product.key((2, 2));
    let large = product.key((0, 0));
    let run = |blocks: &[(usize, usize)]| {
        contract_adaptive_selected(
            &product.a,
            &product.b,
            &0,
            &contraction,
            &patching,
            &product.keys(blocks),
        )
        .unwrap()
    };

    // All outputs: the budget of (2, 2) is about 1e-6 * 6 / 9 of a large
    // block's squared norm, far above its own, so it is dropped.
    let full = contract_adaptive(&product.a, &product.b, &0, &contraction, &patching).unwrap();
    assert!(!full.contains(&small));
    assert!(full.contains(&large));

    // (2, 2) alone: the totals are its own, the budget is 1e-6 of its squared
    // norm, and it is retained.
    let alone = run(&[(2, 2)]);
    assert_eq!(alone.len(), 1);
    let kept = alone.get(&small).unwrap();
    let exact = product.per_block((2, 2));
    let scale = dense_maxabs(&exact);
    assert!(dense_difference(kept, &exact) <= 1.0e-2 * scale);

    // (2, 2) with (0, 0): the totals include the large block again, so (2, 2)
    // is dropped although the per-block loop would keep it.
    let with_large = run(&[(2, 2), (0, 0)]);
    assert!(!with_large.contains(&small));
    assert!(with_large.contains(&large));
}
