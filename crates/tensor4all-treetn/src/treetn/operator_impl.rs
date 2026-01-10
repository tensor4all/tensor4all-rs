//! Operator trait implementation for TreeTN.

use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;

use tensor4all_core::index::{Index, Symmetry};

use super::TreeTN;
use crate::operator::Operator;
use crate::SiteIndexNetwork;

impl<I, V> Operator<I, V> for TreeTN<I, V>
where
    Id: Clone + Hash + Eq + Debug + Send + Sync,
    Symm: Clone + Symmetry + Debug + Send + Sync,
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    fn site_indices(&self) -> HashSet<I> {
        // Collect all site indices from all nodes
        let mut all_indices = HashSet::new();
        for name in self.node_names() {
            if let Some(site_space) = self.site_space(&name) {
                all_indices.extend(site_space.iter().cloned());
            }
        }
        all_indices
    }

    fn site_index_network(&self) -> &SiteIndexNetwork<V, I> {
        &self.site_index_network
    }

    fn node_names(&self) -> HashSet<V> {
        self.node_names().into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::{random_treetn_f64, LinkSpace};
    use tensor4all_core::index::{DynId, Index, NoSymmSpace};

    type DynIndex = Index<DynId, NoSymmSpace>;

    fn make_index(dim: usize) -> DynIndex {
        Index::new_dyn(dim)
    }

    fn create_chain_site_network(n: usize) -> SiteIndexNetwork<String, Index<DynId, NoSymmSpace>> {
        let mut net = SiteIndexNetwork::new();
        for i in 0..n {
            let name = format!("N{}", i);
            let site_idx = make_index(2);
            net.add_node(name, [site_idx].into_iter().collect::<HashSet<_>>())
                .unwrap();
        }
        for i in 0..(n - 1) {
            net.add_edge(&format!("N{}", i), &format!("N{}", i + 1))
                .unwrap();
        }
        net
    }

    #[test]
    fn test_treetn_operator_trait_site_indices() {
        let site_network = create_chain_site_network(3);
        let link_space = LinkSpace::uniform(4);
        let mut rng = rand::thread_rng();

        let treetn = random_treetn_f64(&mut rng, &site_network, link_space);

        // Test site_indices method
        let site_indices = Operator::site_indices(&treetn);
        assert_eq!(site_indices.len(), 3);

        // Verify dimensions (all should be dim 2)
        for idx in &site_indices {
            assert_eq!(idx.symm.total_dim(), 2);
        }
    }

    #[test]
    fn test_treetn_operator_trait_node_names() {
        let site_network = create_chain_site_network(3);
        let link_space = LinkSpace::uniform(4);
        let mut rng = rand::thread_rng();

        let treetn = random_treetn_f64(&mut rng, &site_network, link_space);

        // Test node_names method
        let names = Operator::node_names(&treetn);
        assert_eq!(names.len(), 3);
        assert!(names.contains(&"N0".to_string()));
        assert!(names.contains(&"N1".to_string()));
        assert!(names.contains(&"N2".to_string()));
    }

    #[test]
    fn test_treetn_operator_trait_site_index_network() {
        let site_network = create_chain_site_network(3);
        let link_space = LinkSpace::uniform(4);
        let mut rng = rand::thread_rng();

        let treetn = random_treetn_f64(&mut rng, &site_network, link_space);

        // Test site_index_network method
        let network = Operator::site_index_network(&treetn);
        assert_eq!(network.node_count(), 3);
        assert_eq!(network.edge_count(), 2);
    }
}
