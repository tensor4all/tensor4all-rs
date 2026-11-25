//! Deterministic random tree fixtures and invariant checks shared by unit
//! tests and diagnostics.

use tensor4all_core::{DynIndex, IdxTensor, IndexLike};
use tensor4all_treetn::TreeTN;

/// Explicit splitmix64 stream, so the fixture does not depend on `rand`.
pub(crate) struct SplitMix(pub(crate) u64);

impl SplitMix {
    /// Uniform value in `[-1, 1)`.
    pub(crate) fn next_signed(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        2.0 * ((z >> 11) as f64 / (1u64 << 53) as f64) - 1.0
    }

    /// Uniform real and imaginary parts in `[-1, 1)` (real part only for real `T`).
    pub(crate) fn next_scalar<T: crate::TreeAciScalar>(&mut self) -> T {
        let (re, im) = (self.next_signed(), self.next_signed());
        T::from_evaluated_scalar(tensor4all_core::AnyScalar::new_complex(re, im))
            .or_else(|_| T::from_evaluated_scalar(tensor4all_core::AnyScalar::new_real(re)))
            .unwrap()
    }
}

/// Deterministic tree whose bond component `k` is weighted by `decay^k` at
/// each edge's second node, so cross-interpolation pivots decay geometrically.
pub(crate) fn random_decaying_tree<T: crate::TreeAciScalar>(
    edges: &[(usize, usize)],
    physical: &[DynIndex],
    bond: usize,
    decay: f64,
    rng: &mut SplitMix,
) -> TreeTN<IdxTensor, usize> {
    let bonds = edges
        .iter()
        .map(|_| DynIndex::new_dyn(bond))
        .collect::<Vec<_>>();
    let tensors = physical
        .iter()
        .enumerate()
        .map(|(node, site)| {
            let mut indices = vec![site.clone()];
            for (edge, &(left, right)) in edges.iter().enumerate() {
                if left == node || right == node {
                    indices.push(bonds[edge].clone());
                }
            }
            let weighted_axes = edges
                .iter()
                .filter(|&&(left, right)| left == node || right == node)
                .map(|&(_, right)| right == node)
                .collect::<Vec<_>>();
            let len = indices.iter().map(IndexLike::dim).product();
            let values = (0..len)
                .map(|flat| {
                    // Column-major: physical first, then the incident bonds.
                    let mut rest = flat / site.dim();
                    let mut weight = 1.0;
                    for &weighted in &weighted_axes {
                        if weighted {
                            weight *= decay.powi((rest % bond) as i32);
                        }
                        rest /= bond;
                    }
                    rng.next_scalar::<T>() * <T as tensor4all_core::Scalar>::from_f64(weight)
                })
                .collect();
            IdxTensor::from_dense(indices, values).unwrap()
        })
        .collect();
    TreeTN::from_tensors(tensors, (0..physical.len()).collect()).unwrap()
}

/// Edges of a seven-node tree whose nodes 1 and 3 have degree 3.
pub(crate) const BRANCHED_EDGES: [(usize, usize); 6] =
    [(0, 1), (1, 2), (1, 3), (3, 4), (3, 5), (5, 6)];

/// Two deterministic rank-3 inputs on [`BRANCHED_EDGES`] with binary sites.
///
/// Their elementwise product ([`product_of_two`]) needs bond rank 9 on most
/// cuts, so a rank cap below that leaves an inexact output.
pub(crate) fn branched_product_inputs() -> (Vec<DynIndex>, Vec<TreeTN<IdxTensor, usize>>) {
    let sites = (0..7).map(|_| DynIndex::new_dyn(2)).collect::<Vec<_>>();
    let mut rng = SplitMix(1451);
    let inputs = (0..2)
        .map(|_| random_decaying_tree::<f64>(&BRANCHED_EDGES, &sites, 3, 0.5, &mut rng))
        .collect();
    (sites, inputs)
}

/// Batched elementwise product of the first two inputs.
pub(crate) fn product_of_two(
    batch: crate::TreeElementwiseBatch<'_, f64>,
    output: &mut [f64],
) -> crate::Result<()> {
    for (point, value) in output.iter_mut().enumerate() {
        *value = batch.get(0, point)? * batch.get(1, point)?;
    }
    Ok(())
}

/// Largest `|M - I|` of the pivot-basis invariant on a state's output.
///
/// For every directed edge `c -> p` toward the output's canonical centre,
/// `M[c -> p]` is the `c`-side subtree of the output evaluated at the first
/// `r` retained candidates `candidates.ids[c -> p]`, where `r` is the bond
/// dimension: row `k` is the subtree's vector over the bond at the `k`-th
/// candidate's configuration. The invariant (#1449, #1450) is `M = I` on
/// every such edge. Test-only: it evaluates every retained candidate, which a
/// production run must not pay for.
pub(crate) fn pivot_basis_deviation<T: crate::TreeAciScalar, V: crate::TreeAciNode>(
    state: &crate::state::TreeAciState<'_, T, V>,
) -> f64 {
    let region = state.output.canonical_region();
    assert_eq!(region.len(), 1, "the output has no single canonical centre");
    let centre = state.problem.node_positions[region.iter().next().unwrap()];
    let mut deviation: f64 = 0.0;
    for edge in 0..state.problem.directed_edges.len() {
        if side_contains(&state.problem, edge, centre) {
            continue;
        }
        let rank = bond_dim(state, edge);
        let ids = &state.candidates.ids[edge];
        assert!(
            ids.len() >= rank,
            "fewer retained candidates than bond axes"
        );
        for (k, &id) in ids[..rank].iter().enumerate() {
            let mut coordinates = vec![None; state.problem.node_order.len()];
            component_coordinates(state, edge, id, &mut coordinates);
            let row = subtree_vector(state, edge, &coordinates);
            for (j, value) in row.into_iter().enumerate() {
                let target = if j == k { T::one() } else { T::zero() };
                deviation = deviation.max(tensor4all_core::Scalar::abs_val(value - target));
            }
        }
    }
    deviation
}

/// Whether `node` lies on the `from` side of `edge`.
fn side_contains<V: crate::TreeAciNode>(
    problem: &crate::problem::PreparedTreeProblem<V>,
    edge: crate::problem::DirectedEdgeId,
    node: usize,
) -> bool {
    let directed = &problem.directed_edges[edge];
    problem.node_positions[&directed.from] == node
        || directed
            .incoming_to_from
            .iter()
            .any(|&incoming| side_contains(problem, incoming, node))
}

fn bond_dim<T: crate::TreeAciScalar, V: crate::TreeAciNode>(
    state: &crate::state::TreeAciState<'_, T, V>,
    edge: crate::problem::DirectedEdgeId,
) -> usize {
    let directed = &state.problem.directed_edges[edge];
    let graph_edge = state
        .output
        .edge_between(&directed.from, &directed.to)
        .unwrap();
    state.output.bond_index(graph_edge).unwrap().dim()
}

/// Local coordinate of every node on the `from` side of `edge` in sample `id`.
fn component_coordinates<T: crate::TreeAciScalar, V: crate::TreeAciNode>(
    state: &crate::state::TreeAciState<'_, T, V>,
    edge: crate::problem::DirectedEdgeId,
    id: crate::samples::SampleId,
    coordinates: &mut [Option<usize>],
) {
    let directed = &state.problem.directed_edges[edge];
    let record = state.sample_arena.record(edge, id).unwrap();
    coordinates[state.problem.node_positions[&directed.from]] = Some(record.local_coordinate);
    for &(incoming, sample) in &record.incoming {
        component_coordinates(state, incoming, sample, coordinates);
    }
}

/// The `from`-side subtree of `edge`, evaluated at `coordinates`, as a vector
/// over the bond of `edge`.
fn subtree_vector<T: crate::TreeAciScalar, V: crate::TreeAciNode>(
    state: &crate::state::TreeAciState<'_, T, V>,
    edge: crate::problem::DirectedEdgeId,
    coordinates: &[Option<usize>],
) -> Vec<T> {
    let problem = &state.problem;
    let output = &state.output;
    let directed = &problem.directed_edges[edge];
    let position = problem.node_positions[&directed.from];
    let plan = &problem.physical[position];
    let local = coordinates[position].unwrap();
    let bond_of = |from: &V, to: &V| {
        output
            .bond_index(output.edge_between(from, to).unwrap())
            .unwrap()
            .clone()
    };
    let outgoing = bond_of(&directed.from, &directed.to);
    let children = directed
        .incoming_to_from
        .iter()
        .map(|&incoming| {
            let child = &problem.directed_edges[incoming];
            (
                bond_of(&child.from, &child.to),
                subtree_vector(state, incoming, coordinates),
            )
        })
        .collect::<Vec<_>>();
    enum Axis {
        Site(usize),
        Child(usize),
        Outgoing,
    }
    let core = output
        .tensor(output.node_index(&directed.from).unwrap())
        .unwrap();
    let axes = core
        .indices()
        .iter()
        .map(|index| {
            if let Some(k) = plan.indices.iter().position(|site| site == index) {
                Axis::Site((local / plan.strides[k]) % plan.dims[k])
            } else if let Some(k) = children.iter().position(|(bond, _)| bond == index) {
                Axis::Child(k)
            } else {
                assert!(*index == outgoing, "unexpected core index");
                Axis::Outgoing
            }
        })
        .collect::<Vec<_>>();
    let dims = core.indices().iter().map(|i| i.dim()).collect::<Vec<_>>();
    let mut result = vec![T::zero(); outgoing.dim()];
    // Column-major traversal of the core.
    let mut multi = vec![0usize; dims.len()];
    for value in core.to_vec::<T>().unwrap() {
        let mut factor = value;
        let mut slot = 0;
        let mut on_site = true;
        for (axis, &m) in axes.iter().zip(&multi) {
            match axis {
                Axis::Site(c) => on_site &= m == *c,
                Axis::Child(k) => factor = factor * children[*k].1[m],
                Axis::Outgoing => slot = m,
            }
        }
        if on_site {
            result[slot] = result[slot] + factor;
        }
        for (m, &d) in multi.iter_mut().zip(&dims) {
            *m += 1;
            if *m < d {
                break;
            }
            *m = 0;
        }
    }
    result
}
