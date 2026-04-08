//! Partial contraction for TreeTN.
//!
//! This module provides [`PartialContractionSpec`] and [`partial_contract`] for
//! contracting two TreeTNs while specifying which site index pairs should be
//! contracted (summed over) and which should be multiplied (made to share the
//! same index in the output).
//!
//! The standard [`contract`] function contracts **all** shared site indices.
//! `partial_contract` gives fine-grained control by allowing the caller to
//! specify explicit index pairings.

use std::fmt::Debug;
use std::hash::Hash;

use anyhow::Result;

use super::contraction::{contract, ContractionOptions};
use super::TreeTN;
use tensor4all_core::index_like::IndexLike;
use tensor4all_core::{TensorIndex, TensorLike};

/// Specification for partial contraction of two TreeTNs.
///
/// Each pair `(idx_a, idx_b)` describes a pairing between an index from
/// network A and an index from network B.
///
/// - **`contract_pairs`**: The index `idx_b` in B is replaced with `idx_a`,
///   making them share the same ID so that [`contract`] sums over them.
/// - **`multiply_pairs`**: These index pairs are left as-is (not replaced).
///   Both `idx_a` and `idx_b` will appear as separate external indices in the
///   output, effectively performing an outer product along those axes.
///
/// # Examples
///
/// ```ignore
/// use tensor4all_treetn::treetn::partial_contraction::PartialContractionSpec;
/// use tensor4all_core::DynIndex;
///
/// let idx_a = DynIndex::new_dyn(2);
/// let idx_b = DynIndex::new_dyn(2);
/// let idx_c = DynIndex::new_dyn(3);
/// let idx_d = DynIndex::new_dyn(3);
///
/// let spec = PartialContractionSpec {
///     contract_pairs: vec![(idx_a.clone(), idx_b.clone())],
///     multiply_pairs: vec![(idx_c.clone(), idx_d.clone())],
/// };
///
/// assert_eq!(spec.contract_pairs.len(), 1);
/// assert_eq!(spec.multiply_pairs.len(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct PartialContractionSpec<I: IndexLike> {
    /// Index pairs to be contracted (summed over).
    ///
    /// For each `(idx_a, idx_b)`, `idx_b` in network B is replaced with
    /// `idx_a` so that the contraction engine sums over the shared index.
    pub contract_pairs: Vec<(I, I)>,

    /// Index pairs to be multiplied (outer product).
    ///
    /// These pairs are left unmodified: both `idx_a` (from network A) and
    /// `idx_b` (from network B) will appear as separate indices in the output.
    pub multiply_pairs: Vec<(I, I)>,
}

/// Contract two TreeTNs with explicit control over which index pairs are
/// contracted vs. multiplied.
///
/// This function clones `b`, replaces indices according to the
/// [`PartialContractionSpec`], and then delegates to [`contract`].
///
/// # Arguments
///
/// * `a` - First TreeTN (indices are not modified).
/// * `b` - Second TreeTN (a clone is created with indices replaced per `spec`).
/// * `spec` - Specifies which index pairs to contract and which to multiply.
/// * `center` - The canonical center node for the result.
/// * `options` - Contraction algorithm options (method, truncation, etc.).
///
/// # Errors
///
/// Returns an error if:
/// - Any index in `spec` is not found in the corresponding TreeTN.
/// - Index dimension mismatches occur during replacement.
/// - The underlying [`contract`] fails.
///
/// # Examples
///
/// ```ignore
/// use tensor4all_treetn::treetn::partial_contraction::{partial_contract, PartialContractionSpec};
/// use tensor4all_treetn::treetn::contraction::ContractionOptions;
/// use tensor4all_core::{DynIndex, TensorDynLen, TensorLike};
/// use tensor4all_treetn::TreeTN;
///
/// // Build two single-node TreeTNs with distinct site indices
/// let s_a = DynIndex::new_dyn(2);
/// let s_b = DynIndex::new_dyn(2);
/// let t_a = TensorDynLen::from_dense(vec![s_a.clone()], vec![1.0, 2.0]).unwrap();
/// let t_b = TensorDynLen::from_dense(vec![s_b.clone()], vec![3.0, 4.0]).unwrap();
///
/// let tn_a = TreeTN::<TensorDynLen, String>::from_tensors(
///     vec![t_a], vec!["A".to_string()],
/// ).unwrap();
/// let tn_b = TreeTN::<TensorDynLen, String>::from_tensors(
///     vec![t_b], vec!["A".to_string()],
/// ).unwrap();
///
/// // Contract s_a with s_b (inner product)
/// let spec = PartialContractionSpec {
///     contract_pairs: vec![(s_a.clone(), s_b.clone())],
///     multiply_pairs: vec![],
/// };
///
/// let result = partial_contract(&tn_a, &tn_b, &spec, &"A".to_string(), ContractionOptions::default()).unwrap();
/// let dense = result.contract_to_tensor().unwrap();
/// // 1*3 + 2*4 = 11
/// let val = dense.sum().real();
/// assert!((val - 11.0).abs() < 1e-10);
/// ```
pub fn partial_contract<T, V>(
    a: &TreeTN<T, V>,
    b: &TreeTN<T, V>,
    spec: &PartialContractionSpec<T::Index>,
    center: &V,
    options: ContractionOptions,
) -> Result<TreeTN<T, V>>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + Debug + Ord,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + Debug + Send + Sync,
{
    let mut b_modified = b.clone();

    // Replace contract_pairs indices: make b's index match a's so they
    // share the same ID and will be summed over during contraction.
    for (idx_a, idx_b) in &spec.contract_pairs {
        b_modified = b_modified.replaceind(idx_b, idx_a)?;
    }

    // multiply_pairs are intentionally NOT replaced: both idx_a (from a) and
    // idx_b (from b) remain as separate indices in the output.

    contract(a, &b_modified, center, options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor4all_core::{DynIndex, TensorDynLen};

    /// Helper to create a single-node TreeTN with one site index.
    fn make_single_node_treetn(
        name: &str,
        site_idx: &DynIndex,
        data: Vec<f64>,
    ) -> TreeTN<TensorDynLen, String> {
        let t = TensorDynLen::from_dense(vec![site_idx.clone()], data).unwrap();
        TreeTN::<TensorDynLen, String>::from_tensors(vec![t], vec![name.to_string()]).unwrap()
    }

    /// Helper to create a two-node TreeTN (A -- bond -- B) with one site index per node.
    fn make_two_node_treetn() -> (TreeTN<TensorDynLen, String>, DynIndex, DynIndex, DynIndex) {
        let s0 = DynIndex::new_dyn(2);
        let bond = DynIndex::new_dyn(3);
        let s1 = DynIndex::new_dyn(2);

        let t0 = TensorDynLen::from_dense(
            vec![s0.clone(), bond.clone()],
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        )
        .unwrap();
        let t1 = TensorDynLen::from_dense(
            vec![bond.clone(), s1.clone()],
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        )
        .unwrap();

        let tn = TreeTN::<TensorDynLen, String>::from_tensors(
            vec![t0, t1],
            vec!["A".to_string(), "B".to_string()],
        )
        .unwrap();

        (tn, s0, bond, s1)
    }

    #[test]
    fn test_partial_contraction_spec_creation() {
        let idx_a = DynIndex::new_dyn(2);
        let idx_b = DynIndex::new_dyn(2);
        let idx_c = DynIndex::new_dyn(3);
        let idx_d = DynIndex::new_dyn(3);

        let spec = PartialContractionSpec {
            contract_pairs: vec![(idx_a.clone(), idx_b.clone())],
            multiply_pairs: vec![(idx_c.clone(), idx_d.clone())],
        };

        assert_eq!(spec.contract_pairs.len(), 1);
        assert_eq!(spec.multiply_pairs.len(), 1);

        // Verify clone works
        let spec2 = spec.clone();
        assert_eq!(spec2.contract_pairs.len(), 1);
        assert_eq!(spec2.multiply_pairs.len(), 1);
    }

    #[test]
    fn test_partial_contraction_spec_empty() {
        let spec: PartialContractionSpec<DynIndex> = PartialContractionSpec {
            contract_pairs: vec![],
            multiply_pairs: vec![],
        };

        assert!(spec.contract_pairs.is_empty());
        assert!(spec.multiply_pairs.is_empty());
    }

    #[test]
    fn test_partial_contract_single_node_inner_product() {
        // Two single-node TreeTNs, contract their site indices (inner product)
        let s_a = DynIndex::new_dyn(3);
        let s_b = DynIndex::new_dyn(3);

        let tn_a = make_single_node_treetn("A", &s_a, vec![1.0, 2.0, 3.0]);
        let tn_b = make_single_node_treetn("A", &s_b, vec![1.0, 1.0, 1.0]);

        let spec = PartialContractionSpec {
            contract_pairs: vec![(s_a.clone(), s_b.clone())],
            multiply_pairs: vec![],
        };

        let result = partial_contract(
            &tn_a,
            &tn_b,
            &spec,
            &"A".to_string(),
            ContractionOptions::default(),
        )
        .unwrap();

        // Inner product: 1*1 + 2*1 + 3*1 = 6
        let dense = result.contract_to_tensor().unwrap();
        let val = dense.sum().real();
        assert!((val - 6.0).abs() < 1e-10, "Expected 6.0, got {}", val);
    }

    #[test]
    fn test_partial_contract_two_node_contract_all() {
        // Two 2-node TreeTNs, contract all site indices
        let (tn_a, s0_a, _bond_a, s1_a) = make_two_node_treetn();

        // Create second TreeTN with fresh site indices but same structure
        let s0_b = DynIndex::new_dyn(2);
        let bond_b = DynIndex::new_dyn(3);
        let s1_b = DynIndex::new_dyn(2);

        let t0_b = TensorDynLen::from_dense(
            vec![s0_b.clone(), bond_b.clone()],
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        let t1_b = TensorDynLen::from_dense(
            vec![bond_b.clone(), s1_b.clone()],
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        )
        .unwrap();

        let tn_b = TreeTN::<TensorDynLen, String>::from_tensors(
            vec![t0_b, t1_b],
            vec!["A".to_string(), "B".to_string()],
        )
        .unwrap();

        let spec = PartialContractionSpec {
            contract_pairs: vec![(s0_a.clone(), s0_b.clone()), (s1_a.clone(), s1_b.clone())],
            multiply_pairs: vec![],
        };

        let result = partial_contract(
            &tn_a,
            &tn_b,
            &spec,
            &"A".to_string(),
            ContractionOptions::default(),
        )
        .unwrap();

        // The result should be a scalar (all site indices contracted)
        let dense = result.contract_to_tensor().unwrap();
        let ext = dense.external_indices();
        assert!(
            ext.is_empty(),
            "Expected scalar result, got {} external indices",
            ext.len()
        );
    }

    #[test]
    fn test_partial_contract_multiply_pairs() {
        // Two single-node TreeTNs with 2D tensors.
        // contract_pairs contracts one index pair; multiply_pairs keeps both indices
        // in the output (outer product along those axes).
        let s_a1 = DynIndex::new_dyn(2);
        let s_a2 = DynIndex::new_dyn(3);
        let s_b1 = DynIndex::new_dyn(2);
        let s_b2 = DynIndex::new_dyn(3);

        // tn_a: tensor with indices (s_a1, s_a2), data: 2x3 = 6 elements
        let t_a = TensorDynLen::from_dense(
            vec![s_a1.clone(), s_a2.clone()],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap();

        // tn_b: tensor with indices (s_b1, s_b2), data: 2x3 = 6 elements
        let t_b = TensorDynLen::from_dense(
            vec![s_b1.clone(), s_b2.clone()],
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )
        .unwrap();

        let tn_a =
            TreeTN::<TensorDynLen, String>::from_tensors(vec![t_a], vec!["A".to_string()]).unwrap();
        let tn_b =
            TreeTN::<TensorDynLen, String>::from_tensors(vec![t_b], vec!["A".to_string()]).unwrap();

        // Contract s_a1 with s_b1, keep s_a2 and s_b2 as separate output indices
        let spec = PartialContractionSpec {
            contract_pairs: vec![(s_a1.clone(), s_b1.clone())],
            multiply_pairs: vec![(s_a2.clone(), s_b2.clone())],
        };

        let result = partial_contract(
            &tn_a,
            &tn_b,
            &spec,
            &"A".to_string(),
            ContractionOptions::default(),
        )
        .unwrap();

        // After contraction:
        //   contract s_a1/s_b1 (dim=2): summed over
        //   multiply s_a2/s_b2: both remain as separate output indices
        // Result should have 2 external indices (s_a2 dim=3, s_b2 dim=3)
        let dense = result.contract_to_tensor().unwrap();
        let ext = dense.external_indices();
        assert_eq!(
            ext.len(),
            2,
            "Expected 2 external indices, got {}",
            ext.len()
        );

        // Result[j,k] = sum_i A[i,j] * B[i,k]
        // A = [[1,2,3],[4,5,6]], B = [[1,1,1],[1,1,1]]
        // Result[j,k] = A[0,j]*B[0,k] + A[1,j]*B[1,k]
        //             = (1+4)*1, (2+5)*1, (3+6)*1 for all k
        //             = [[5,5,5],[7,7,7],[9,9,9]]
        // Total sum = 3*(5+7+9) = 63
        let total_sum = dense.sum().real();
        assert!(
            (total_sum - 63.0).abs() < 1e-10,
            "Expected total sum 63.0, got {}",
            total_sum
        );
    }

    #[test]
    fn test_partial_contract_dimension_mismatch_error() {
        // Indices with mismatched dimensions should produce an error
        let s_a = DynIndex::new_dyn(2);
        let s_b = DynIndex::new_dyn(3); // different dim

        let tn_a = make_single_node_treetn("A", &s_a, vec![1.0, 2.0]);
        let tn_b = make_single_node_treetn("A", &s_b, vec![1.0, 2.0, 3.0]);

        let spec = PartialContractionSpec {
            contract_pairs: vec![(s_a.clone(), s_b.clone())],
            multiply_pairs: vec![],
        };

        let result = partial_contract(
            &tn_a,
            &tn_b,
            &spec,
            &"A".to_string(),
            ContractionOptions::default(),
        );

        assert!(result.is_err(), "Expected error for dimension mismatch");
    }

    #[test]
    fn test_partial_contract_index_not_found_error() {
        // Specifying an index not in the TreeTN should produce an error
        let s_a = DynIndex::new_dyn(2);
        let s_b = DynIndex::new_dyn(2);
        let unknown = DynIndex::new_dyn(2);

        let tn_a = make_single_node_treetn("A", &s_a, vec![1.0, 2.0]);
        let tn_b = make_single_node_treetn("A", &s_b, vec![3.0, 4.0]);

        // unknown is not in tn_b
        let spec = PartialContractionSpec {
            contract_pairs: vec![(s_a.clone(), unknown.clone())],
            multiply_pairs: vec![],
        };

        let result = partial_contract(
            &tn_a,
            &tn_b,
            &spec,
            &"A".to_string(),
            ContractionOptions::default(),
        );

        assert!(result.is_err(), "Expected error for index not found");
    }
}
