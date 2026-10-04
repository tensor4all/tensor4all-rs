//! Options and explicit probes for tree RSI.

use std::collections::HashMap;

use tensor4all_tensorbackend::Matrix;

/// Explicit product-state probes, one `d_v x k` matrix per input and node.
///
/// Probes replace the random standard-normal draws (seeded by
/// [`TreeRsiOptions::seed`] or taken from a caller-owned RNG). `per_input[a][v]` is the column-major matrix
/// `Omega_v^(a)(s_v, l)` whose rows enumerate the combined physical index of
/// node `v` (first physical index fastest, in the order of input 0's tensor)
/// and whose `k` columns are the probe labels. Only nodes that are actually
/// sketched need an entry; missing required entries are reported before random draws or contractions. Use explicit probes to replay another implementation's
/// random draws or to share probes between runs.
///
/// Related types: [`TreeRsiOptions`] holds an optional value of this type.
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
/// use tensor4all_tensorbackend::Matrix;
/// use tensor4all_treersi::TreeRsiProbes;
///
/// let mut node_probes = HashMap::new();
/// node_probes.insert(0usize, Matrix::from_col_major_vec(2, 1, vec![1.0, -1.0]));
/// let probes = TreeRsiProbes::new(vec![node_probes]);
/// assert_eq!(probes.n_inputs(), 1);
/// assert_eq!(probes.probe(0, &0).unwrap().nrows(), 2);
/// ```
#[derive(Clone, Debug)]
pub struct TreeRsiProbes<V> {
    per_input: Vec<HashMap<V, Matrix<f64>>>,
}

impl<V: Eq + std::hash::Hash> TreeRsiProbes<V> {
    /// Wraps per-input, per-node probe matrices.
    ///
    /// # Arguments
    ///
    /// * `per_input` - One map per input, in input order, from node name to its
    ///   `d_v x k` probe matrix.
    ///
    /// # Returns
    ///
    /// The probe set; shapes are validated when a run uses it.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    /// use tensor4all_treersi::TreeRsiProbes;
    ///
    /// let probes = TreeRsiProbes::<usize>::new(vec![HashMap::new(), HashMap::new()]);
    /// assert_eq!(probes.n_inputs(), 2);
    /// ```
    pub fn new(per_input: Vec<HashMap<V, Matrix<f64>>>) -> Self {
        Self { per_input }
    }

    /// Returns the number of inputs covered by this probe set.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treersi::TreeRsiProbes;
    ///
    /// assert_eq!(TreeRsiProbes::<usize>::new(Vec::new()).n_inputs(), 0);
    /// ```
    pub fn n_inputs(&self) -> usize {
        self.per_input.len()
    }

    /// Returns the probe of `input` at `node`, if present.
    ///
    /// # Arguments
    ///
    /// * `input` - Zero-based input position.
    /// * `node` - Node name.
    ///
    /// # Returns
    ///
    /// `None` when the input or node has no probe entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    /// use tensor4all_treersi::TreeRsiProbes;
    ///
    /// let probes = TreeRsiProbes::<usize>::new(vec![HashMap::new()]);
    /// assert!(probes.probe(0, &3).is_none());
    /// assert!(probes.probe(1, &3).is_none());
    /// ```
    pub fn probe(&self, input: usize, node: &V) -> Option<&Matrix<f64>> {
        self.per_input.get(input)?.get(node)
    }
}

/// Controls a tree RSI run.
///
/// The output rank on each edge is bounded by [`Self::max_bond_dim`] and by the
/// rank revealed by the local interpolative decomposition, which stops when a
/// pivot falls below `rel_tol` times the largest pivot. This is a local sketch
/// criterion, not a global product error bound.
/// [`Self::sketch_dim`] is the probe width `k`; a local matrix has `d_p * k`
/// columns, so `k` must be at least `max_bond_dim / d_p` for the cap to be
/// reachable (paper Eq. 7). When in doubt set only `max_bond_dim` and keep the
/// defaults: `k = ceil(max_bond_dim / d) + oversampling`.
///
/// Wherever the physical complement of a node has at most `k` assignments,
/// the run uses those exact assignments instead of random probes, so a large
/// `k` makes small problems deterministic and exact up to truncation.
///
/// Related types: [`crate::hadamard_many_in`] consumes these options.
///
/// # Examples
///
/// ```
/// use tensor4all_treersi::TreeRsiOptions;
///
/// let options = TreeRsiOptions::<usize> {
///     max_bond_dim: Some(16),
///     seed: 7,
///     ..TreeRsiOptions::default()
/// };
/// assert_eq!(options.oversampling, 5);
/// assert!(options.sketch_dim.is_none());
/// ```
#[derive(Clone, Debug)]
pub struct TreeRsiOptions<V> {
    /// Maximum output bond dimension per edge. `None` leaves the rank to the
    /// tolerances and to the local matrix size; then [`Self::sketch_dim`] must
    /// be set. Default: `None`. Must be positive when set.
    pub max_bond_dim: Option<usize>,
    /// Probe width `k`. `None` derives `ceil(max_bond_dim / d) + oversampling`
    /// with `d` the smallest physical dimension of any parent node (at least 1).
    /// Must be positive when set. Default: `None`.
    pub sketch_dim: Option<usize>,
    /// Extra probe columns `p` used only when [`Self::sketch_dim`] is derived.
    /// Default: `5`.
    pub oversampling: usize,
    /// Relative pivot tolerance of the local rank-revealing LU: pivots below
    /// `rel_tol` times the largest pivot are dropped. Default: `1e-14`.
    pub rel_tol: f64,
    /// Root node. The run interpolates leaves first and evaluates the root
    /// exactly. `None` uses the last node in sorted node order, which for a
    /// chain named `0..n` reproduces the paper's left-to-right order.
    pub root: Option<V>,
    /// Seed of the standard-normal probe draws of the convenience entry
    /// points, which use `ChaCha8Rng::seed_from_u64(seed)`. Ignored by the
    /// `*_with_rng_in` entry points (they consume the caller's RNG) and when
    /// [`Self::probes`] is set. Default: `0`.
    pub seed: u64,
    /// Explicit probes replacing the seeded draws. Default: `None`.
    pub probes: Option<TreeRsiProbes<V>>,
    /// Maximum elements in each algorithm-owned dense block, including
    /// contraction intermediates. Backend decomposition scratch is additional.
    /// Default: `1 << 26` (512 MiB for one f64 block).
    pub max_local_elements: usize,
}

impl<V> Default for TreeRsiOptions<V> {
    fn default() -> Self {
        Self {
            max_bond_dim: None,
            sketch_dim: None,
            oversampling: 5,
            rel_tol: 1e-14,
            root: None,
            seed: 0,
            probes: None,
            max_local_elements: 1 << 26,
        }
    }
}
