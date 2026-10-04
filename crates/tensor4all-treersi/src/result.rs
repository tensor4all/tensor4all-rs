//! Diagnostics explicitly describe local computations, never certified error.
use tensor4all_core::IdxTensor;
use tensor4all_treetn::TreeTN;

/// Local row-interpolation diagnostics for one child-parent edge.
/// These describe a sketch (or exact local matrix), not global accuracy.
///
/// # Examples
/// ```
/// use tensor4all_treersi::TreeRsiEdgeReport;
/// let edge=TreeRsiEdgeReport { child:0usize,parent:1,rank:1,rows:2,columns:2,
///     exact_columns:true,rank_limit_reached:false,relative_pivot:0.0 };
/// assert_eq!((edge.child,edge.parent),(0,1));
/// ```
#[derive(Clone, Debug)]
pub struct TreeRsiEdgeReport<V> {
    /// Child node in the rooted traversal.
    pub child: V,
    /// Parent node in the rooted traversal.
    pub parent: V,
    /// Output bond dimension (zero matrices use rank one).
    pub rank: usize,
    /// Candidate row count.
    pub rows: usize,
    /// Local sketch or exact-complement column count.
    pub columns: usize,
    /// Whether the local columns enumerate the exact complement.
    pub exact_columns: bool,
    /// Whether rank reached the configured/local dimension limit. This alone
    /// does not mean that a nonzero residual was truncated.
    pub rank_limit_reached: bool,
    /// Maximum entry magnitude of the remaining LU Schur complement, divided
    /// by the largest local component. This describes the local sketched
    /// matrix, not physical tensor error; never use it as global acceptance.
    pub relative_pivot: f64,
}

/// Root, sketch width and per-edge local diagnostics of one product call.
/// Related type: [`TreeRsiResult`] holds this record beside the approximation.
///
/// # Examples
/// ```
/// use tensor4all_treersi::TreeRsiDiagnostics;
/// let d=TreeRsiDiagnostics { root:0usize,sketch_dim:4,edges:Vec::new(),
///     sketch_messages_per_input:0,exact_messages_per_input:0 };
/// assert_eq!(d.max_rank(),1);
/// ```
#[derive(Clone, Debug)]
pub struct TreeRsiDiagnostics<V> {
    /// Root used for final exact row evaluation.
    pub root: V,
    /// Number of product-state probe columns.
    pub sketch_dim: usize,
    /// Non-root edges in postorder.
    pub edges: Vec<TreeRsiEdgeReport<V>>,
    /// Number of directed sketch messages constructed per input.
    pub sketch_messages_per_input: usize,
    /// Number of cached exact component messages constructed per input.
    pub exact_messages_per_input: usize,
}
impl<V> TreeRsiDiagnostics<V> {
    /// Returns the largest output bond dimension, or one for a single-node tree.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_treersi::TreeRsiDiagnostics;
    /// let d=TreeRsiDiagnostics { root:0usize,sketch_dim:1,edges:Vec::new(),
    ///     sketch_messages_per_input:0,exact_messages_per_input:0 };
    /// assert_eq!(d.max_rank(),1);
    /// ```
    pub fn max_rank(&self) -> usize {
        self.edges.iter().map(|e| e.rank).max().unwrap_or(1)
    }
}

/// A product approximation and local diagnostics; success is not an accuracy
/// certificate. Validate independently before accepting downstream results.
///
/// # Examples
/// ```
/// use tensor4all_core::{DynIndex,IdxTensor};
/// use tensor4all_treetn::TreeTN;
/// use tensor4all_treersi::{hadamard_many,TreeRsiOptions};
/// let t=TreeTN::from_tensors(vec![IdxTensor::from_dense(vec![DynIndex::new_dyn(1)],
///     vec![3.0_f64])?],vec![0usize])?;
/// let out=hadamard_many::<f64,_>(&[t.clone(),t],&TreeRsiOptions { max_bond_dim:Some(1),..Default::default() })?;
/// assert_eq!(out.tree.to_dense()?.to_vec::<f64>()?,vec![9.0]);
/// # Ok::<(),Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug)]
pub struct TreeRsiResult<V: crate::TreeRsiNode> {
    /// Approximation with the input physical-index semantics.
    pub tree: TreeTN<IdxTensor, V>,
    /// Local diagnostics; see [`TreeRsiEdgeReport::relative_pivot`].
    pub diagnostics: TreeRsiDiagnostics<V>,
}
