//! Inspect the returned tree independently of algorithm-reported ranks.
use tensor4all_core::{IdxTensor, IndexLike};
use tensor4all_treetn::TreeTN;

type EdgeRanks = Vec<(usize, usize, usize)>;

pub(crate) fn edge_ranks(
    tree: &TreeTN<IdxTensor, usize>,
) -> Result<EdgeRanks, Box<dyn std::error::Error>> {
    let mut ranks = Vec::new();
    for (a, b) in tree.site_index_network().edges() {
        let edge = tree.edge_between(&a, &b).ok_or("missing output edge")?;
        let bond = tree.bond_index(edge).ok_or("missing output bond")?;
        ranks.push((a.min(b), a.max(b), bond.dim()));
    }
    ranks.sort_unstable();
    Ok(ranks)
}
