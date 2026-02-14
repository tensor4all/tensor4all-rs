//! Operator trait implementation for TreeTN.
//!
//! This module implements the Operator trait for TreeTN, allowing TreeTNs
//! to be used with the operator composition infrastructure.

use std::collections::HashSet;
use std::hash::Hash;

use tensor4all_core::{IndexLike, TensorLike};

use crate::operator::Operator;
use crate::site_index_network::SiteIndexNetwork;
use crate::treetn::TreeTN;

impl<T, V> Operator<T, V> for TreeTN<T, V>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    fn site_indices(&self) -> HashSet<T::Index> {
        // Collect all site indices from all nodes
        let mut result = HashSet::new();
        for node_name in self.site_index_network.node_names() {
            if let Some(site_space) = self.site_index_network.site_space(node_name) {
                result.extend(site_space.iter().cloned());
            }
        }
        result
    }

    fn site_index_network(&self) -> &SiteIndexNetwork<V, T::Index> {
        &self.site_index_network
    }

    fn node_names(&self) -> HashSet<V> {
        self.site_index_network
            .node_names()
            .into_iter()
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use tensor4all_core::{DynIndex, IndexLike, TensorDynLen};

    use crate::operator::Operator;
    use crate::treetn::TreeTN;

    /// Helper to create a simple 2-node TreeTN: A -- bond -- B
    fn make_two_node_treetn() -> (
        TreeTN<TensorDynLen, String>,
        DynIndex, // s0
        DynIndex, // bond
        DynIndex, // s1
    ) {
        let s0 = DynIndex::new_dyn(2);
        let bond = DynIndex::new_dyn(3);
        let s1 = DynIndex::new_dyn(4);

        let t0 = TensorDynLen::from_dense_f64(vec![s0.clone(), bond.clone()], vec![1.0; 6]);
        let t1 = TensorDynLen::from_dense_f64(vec![bond.clone(), s1.clone()], vec![1.0; 12]);

        let tn = TreeTN::<TensorDynLen, String>::from_tensors(
            vec![t0, t1],
            vec!["A".to_string(), "B".to_string()],
        )
        .unwrap();

        (tn, s0, bond, s1)
    }

    #[test]
    fn test_site_indices() {
        let (tn, s0, bond, s1) = make_two_node_treetn();
        let site_indices = tn.site_indices();

        // Site indices should include s0 and s1
        let site_ids: HashSet<_> = site_indices.iter().map(|i| i.id().clone()).collect();
        assert!(site_ids.contains(s0.id()));
        assert!(site_ids.contains(s1.id()));
        // Bond should NOT be a site index
        assert!(!site_ids.contains(bond.id()));
        assert_eq!(site_indices.len(), 2);
    }

    #[test]
    fn test_site_index_network() {
        let (tn, _s0, _bond, _s1) = make_two_node_treetn();
        let sin = Operator::site_index_network(&tn);

        // Should have same node count as the TreeTN
        assert_eq!(sin.node_names().len(), 2);
    }

    #[test]
    fn test_node_names() {
        let (tn, _s0, _bond, _s1) = make_two_node_treetn();
        let names: HashSet<String> = Operator::node_names(&tn);

        assert_eq!(names.len(), 2);
        assert!(names.contains("A"));
        assert!(names.contains("B"));
    }

    #[test]
    fn test_site_indices_single_node() {
        let s0 = DynIndex::new_dyn(2);
        let s1 = DynIndex::new_dyn(3);
        let t = TensorDynLen::from_dense_f64(vec![s0.clone(), s1.clone()], vec![0.0; 6]);
        let tn =
            TreeTN::<TensorDynLen, String>::from_tensors(vec![t], vec!["A".to_string()]).unwrap();

        let site_indices = tn.site_indices();
        assert_eq!(site_indices.len(), 2);

        let names: HashSet<String> = Operator::node_names(&tn);
        assert_eq!(names.len(), 1);
        assert!(names.contains("A"));
    }

    #[test]
    fn test_site_indices_three_node_chain() {
        let s0 = DynIndex::new_dyn(2);
        let bond01 = DynIndex::new_dyn(3);
        let s1 = DynIndex::new_dyn(2);
        let bond12 = DynIndex::new_dyn(3);
        let s2 = DynIndex::new_dyn(2);

        let t0 = TensorDynLen::from_dense_f64(vec![s0.clone(), bond01.clone()], vec![1.0; 6]);
        let t1 = TensorDynLen::from_dense_f64(
            vec![bond01.clone(), s1.clone(), bond12.clone()],
            vec![1.0; 18],
        );
        let t2 = TensorDynLen::from_dense_f64(vec![bond12.clone(), s2.clone()], vec![1.0; 6]);

        let tn = TreeTN::<TensorDynLen, String>::from_tensors(
            vec![t0, t1, t2],
            vec!["A".to_string(), "B".to_string(), "C".to_string()],
        )
        .unwrap();

        let site_indices = tn.site_indices();
        assert_eq!(site_indices.len(), 3);

        let names: HashSet<String> = Operator::node_names(&tn);
        assert_eq!(names.len(), 3);
        assert!(names.contains("A"));
        assert!(names.contains("B"));
        assert!(names.contains("C"));
    }
}
