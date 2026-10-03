//! Regression tests for `TruncatedDefaultProposer` at branching vertices
//! (issue #804): the proposer must converge to the tolerance both when the
//! junction vertex is site-free (local dimension 1) and when it carries a
//! site, and the result must match a dense reference.
//!
//! Global pivots are disabled so the local proposer has to find every pivot
//! itself: with them, the global search alone restores full rank on a tree
//! this small. At this size each half of the #804 fix is needed: with the
//! old `local_dim * rank` budget the site-free junction bonds stay at their
//! starting rank 1, and without keeping the previous pivots both junction
//! variants stop at a dense residual of about 1e-8 to 1e-7.

use anyhow::Result;
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_treetci::{
    optimize_with_proposer, to_treetn, GlobalIndexBatch, TreeTCI2, TreeTciEdge, TreeTciGraph,
    TreeTciOptions, TruncatedDefaultProposer,
};
use tensor4all_treetn::TreeTN;

/// Quantics bits per variable.
const BITS: usize = 6;
const TOLERANCE: f64 = 1e-10;
const MAX_ITER: usize = 20;

/// One quantics bit of one variable, most significant bit first.
#[derive(Clone, Copy)]
struct Bit {
    var: usize,
    level: usize,
}

/// A three-arm tree around vertex 0.
///
/// `junction_bit` is the bit carried by the junction vertex (`None` for a
/// site-free junction of local dimension 1). Every arm is a chain of the
/// remaining bits of one variable, starting from its most significant bit.
struct JunctionTree {
    bits: Vec<Option<Bit>>,
    graph: TreeTciGraph,
}

impl JunctionTree {
    fn new(junction_bit: Option<Bit>) -> Self {
        let mut bits = vec![junction_bit];
        let mut edges = Vec::new();
        for var in 0..3 {
            let mut previous = 0;
            for level in 0..BITS {
                if junction_bit.is_some_and(|bit| bit.var == var && bit.level == level) {
                    continue;
                }
                let vertex = bits.len();
                bits.push(Some(Bit { var, level }));
                edges.push(TreeTciEdge::new(previous, vertex));
                previous = vertex;
            }
        }
        let graph = TreeTciGraph::new(bits.len(), &edges).unwrap();
        Self { bits, graph }
    }

    fn local_dims(&self) -> Vec<usize> {
        self.bits
            .iter()
            .map(|bit| if bit.is_some() { 2 } else { 1 })
            .collect()
    }

    /// Lorentzian of a 3D tight-binding dispersion on a `2^BITS` grid per axis.
    fn value(&self, point: &[usize]) -> f64 {
        let mut coords = [0.0f64; 3];
        for (bit, &value) in self.bits.iter().zip(point) {
            if let Some(bit) = bit {
                coords[bit.var] += value as f64 * 0.5f64.powi(bit.level as i32 + 1);
            }
        }
        let tau = std::f64::consts::TAU;
        let energy: f64 = -2.0 * coords.iter().map(|x| (tau * x).cos()).sum::<f64>();
        let (mu, eta) = (0.5, 0.1);
        eta / ((energy - mu).powi(2) + eta * eta)
    }

    fn evaluate(&self, batch: GlobalIndexBatch<'_>) -> Result<Vec<f64>> {
        let mut point = vec![0usize; batch.n_sites()];
        let mut values = Vec::with_capacity(batch.n_points());
        for p in 0..batch.n_points() {
            for (site, slot) in point.iter_mut().enumerate() {
                *slot = batch.get(site, p).unwrap();
            }
            values.push(self.value(&point));
        }
        Ok(values)
    }

    /// Dense reference over the network's own site indices, in site order.
    fn dense_reference(&self, tn: &TreeTN<IdxTensor, usize>) -> IdxTensor {
        let local_dims = self.local_dims();
        let indices: Vec<DynIndex> = (0..local_dims.len())
            .map(|site| tn.site_space(&site).unwrap().iter().next().unwrap().clone())
            .collect();
        let total: usize = local_dims.iter().product();
        let mut point = vec![0usize; local_dims.len()];
        let mut values = Vec::with_capacity(total);
        for _ in 0..total {
            values.push(self.value(&point));
            // Column-major: site 0 varies fastest.
            for (slot, &dim) in point.iter_mut().zip(&local_dims) {
                *slot += 1;
                if *slot < dim {
                    break;
                }
                *slot = 0;
            }
        }
        IdxTensor::from_dense(indices, values).unwrap()
    }
}

fn assert_truncated_proposer_converges(tree: &JunctionTree) {
    let local_dims = tree.local_dims();
    let n_sites = local_dims.len();
    let evaluate = |batch: GlobalIndexBatch<'_>| tree.evaluate(batch);
    let mut state = TreeTCI2::<f64>::new(local_dims, tree.graph.clone()).unwrap();
    state.add_global_pivots(&[vec![0; n_sites]]).unwrap();
    state.max_sample_value = tree.value(&vec![0; n_sites]).abs();
    let junction_edges = state.graph.adjacent_edges(0, &[]);
    assert_eq!(junction_edges.len(), 3);
    let junction_rank = |state: &TreeTCI2<f64>, edge: TreeTciEdge| {
        let (key, _) = state.graph.subregion_vertices(edge).unwrap();
        state.ijset[&key].ncols().unwrap()
    };
    for &edge in &junction_edges {
        assert_eq!(junction_rank(&state, edge), 1);
    }

    let options = TreeTciOptions {
        tolerance: TOLERANCE,
        max_iter: MAX_ITER,
        enable_global_pivots: false,
        ..Default::default()
    };
    let (ranks, errors) = optimize_with_proposer(
        &mut state,
        evaluate,
        &options,
        &TruncatedDefaultProposer::seeded(3),
    )
    .unwrap();

    assert!(
        ranks.len() < MAX_ITER,
        "did not converge within {MAX_ITER} iterations: ranks {ranks:?}, errors {errors:?}"
    );
    let last_error = errors.last().copied().unwrap();
    assert!(
        last_error < TOLERANCE,
        "normalized bond error {last_error:e} is not below {TOLERANCE:e}"
    );
    for &edge in &junction_edges {
        let rank = junction_rank(&state, edge);
        assert!(rank > 1, "junction bond {edge:?} stayed at rank {rank}");
    }

    let tn = to_treetn(&state, evaluate, None).unwrap();
    let reference = tree.dense_reference(&tn);
    let residual = tn
        .to_dense()
        .unwrap()
        .sub(&reference)
        .unwrap()
        .maxabs()
        .unwrap();
    let scale = reference.maxabs().unwrap();
    assert!(
        residual <= TOLERANCE * scale,
        "max abs residual {residual:e} exceeds {TOLERANCE:e} * {scale:e} (ranks {ranks:?})"
    );
}

#[test]
fn truncated_proposer_converges_on_site_free_junction() {
    let tree = JunctionTree::new(None);
    assert_eq!(tree.local_dims()[0], 1);
    assert_eq!(tree.graph.adjacent_edges(0, &[]).len(), 3);
    assert_truncated_proposer_converges(&tree);
}

#[test]
fn truncated_proposer_converges_on_junction_with_site() {
    let tree = JunctionTree::new(Some(Bit { var: 0, level: 0 }));
    assert_eq!(tree.local_dims()[0], 2);
    assert_eq!(tree.graph.adjacent_edges(0, &[]).len(), 3);
    assert_truncated_proposer_converges(&tree);
}
