//! Rooted tree validation and bounded complement planning. The exact-tail
//! policy generalizes RSI, arXiv:2602.17974v1 III.A.4, to tree components.

use std::collections::HashMap;

use tensor4all_core::{DynIndex, IdxTensor, IndexLike};
use tensor4all_treetn::TreeTN;

use crate::{Result, TreeRsiError, TreeRsiNode, TreeRsiOptions};

/// Physical indices of one node, in input 0's tensor order.
#[derive(Clone, Debug)]
pub(crate) struct PhysPlan {
    pub(crate) indices: Vec<DynIndex>,
    pub(crate) dim: usize,
}

/// A validated, rooted traversal plan shared by every input.
#[derive(Clone, Debug)]
pub(crate) struct Plan<V> {
    pub(crate) nodes: Vec<V>,
    pub(crate) root: usize,
    pub(crate) parent: Vec<Option<usize>>,
    pub(crate) children: Vec<Vec<usize>>,
    /// Non-root nodes, children before parents.
    pub(crate) postorder: Vec<usize>,
    /// Every node, parents before children.
    pub(crate) preorder: Vec<usize>,
    pub(crate) phys: Vec<PhysPlan>,
    pub(crate) k: usize,
    /// Non-root node uses exact complement columns (`D_v <= k`).
    pub(crate) exact: Vec<bool>,
    pub(crate) exact_up: Vec<bool>,
    pub(crate) exact_down: Vec<bool>,
    /// Sketched environment `H_v` is required.
    pub(crate) needs_sketched_env: Vec<bool>,
    /// Downward message `mu_(parent -> v)` is required.
    pub(crate) needs_down: Vec<bool>,
    /// Upward message `mu_(v -> parent)` is required.
    pub(crate) needs_up: Vec<bool>,
    /// Probe `Omega_v` is read by some required message.
    pub(crate) needs_probe: Vec<bool>,
}

impl<V> Plan<V> {
    /// Number of required messages per input.
    pub(crate) fn message_count(&self) -> usize {
        self.needs_up.iter().filter(|&&needed| needed).count()
            + self.needs_down.iter().filter(|&&needed| needed).count()
    }
}

/// Checked product that saturates at `cap`, used for complement sizes.
fn capped_mul(a: usize, b: usize, cap: usize) -> usize {
    a.checked_mul(b).map_or(cap, |product| product.min(cap))
}

pub(crate) fn checked_product(
    values: impl IntoIterator<Item = usize>,
    context: &'static str,
) -> Result<usize> {
    values.into_iter().try_fold(1usize, |product, value| {
        product
            .checked_mul(value)
            .ok_or(TreeRsiError::SizeOverflow { context })
    })
}

pub(crate) fn enforce_limit(resource: &'static str, requested: usize, limit: usize) -> Result<()> {
    if requested > limit {
        return Err(TreeRsiError::ResourceLimit {
            resource,
            requested,
            limit,
        });
    }
    Ok(())
}

fn validate_options<V>(options: &TreeRsiOptions<V>) -> Result<()> {
    if options.max_bond_dim == Some(0) {
        return Err(TreeRsiError::InvalidOption {
            option: "max_bond_dim",
            message: "a configured bond limit must be positive",
        });
    }
    if options.sketch_dim == Some(0) {
        return Err(TreeRsiError::InvalidOption {
            option: "sketch_dim",
            message: "the probe width must be positive",
        });
    }
    if options.max_bond_dim.is_none() && options.sketch_dim.is_none() {
        return Err(TreeRsiError::InvalidOption {
            option: "sketch_dim",
            message: "set sketch_dim when max_bond_dim is unbounded",
        });
    }
    if !options.rel_tol.is_finite() || options.rel_tol < 0.0 {
        return Err(TreeRsiError::InvalidOption {
            option: "rel_tol",
            message: "the relative tolerance must be finite and nonnegative",
        });
    }
    if options.max_local_elements == 0 {
        return Err(TreeRsiError::InvalidOption {
            option: "max_local_elements",
            message: "the element limit must be positive",
        });
    }
    Ok(())
}

/// Validates inputs and options and builds the shared traversal plan.
pub(crate) fn build_plan<V: TreeRsiNode>(
    inputs: &[TreeTN<IdxTensor, V>],
    options: &TreeRsiOptions<V>,
) -> Result<Plan<V>> {
    validate_options(options)?;
    let reference = inputs.first().ok_or(TreeRsiError::NoInputs)?;
    for (input, tree) in inputs.iter().enumerate() {
        if tree.node_count() == 0 {
            return Err(TreeRsiError::EmptyTree { input });
        }
        tree.validate_tree()?;
        if input > 0 && !reference.same_topology(tree) {
            return Err(TreeRsiError::TopologyMismatch { input });
        }
    }

    let mut nodes = reference.node_names();
    nodes.sort();
    let position = nodes
        .iter()
        .cloned()
        .enumerate()
        .map(|(position, node)| (node, position))
        .collect::<HashMap<_, _>>();

    let mut phys = Vec::with_capacity(nodes.len());
    for node in &nodes {
        let space = reference
            .site_space(node)
            .ok_or(TreeRsiError::InternalInvariant {
                message: "a validated node has no physical-index space",
            })?;
        for (input, tree) in inputs.iter().enumerate().skip(1) {
            if tree.site_space(node) != Some(space) {
                return Err(TreeRsiError::PhysicalIndexMismatch {
                    input,
                    node: format!("{node:?}"),
                });
            }
        }
        let tensor = node_tensor(reference, node)?;
        let indices = tensor
            .indices()
            .iter()
            .filter(|index| space.contains(*index))
            .cloned()
            .collect::<Vec<_>>();
        if indices.len() != space.len() {
            return Err(TreeRsiError::InternalInvariant {
                message: "physical-index metadata disagrees with tensor axes",
            });
        }
        let dim = checked_product(
            indices.iter().map(IndexLike::dim),
            "local physical dimension",
        )?;
        if dim == 0 {
            return Err(TreeRsiError::PhysicalIndexMismatch {
                input: 0,
                node: format!("{node:?}"),
            });
        }
        phys.push(PhysPlan { indices, dim });
    }

    // default root is the last sorted node (paper order on a path).
    let root = match &options.root {
        Some(root) => *position.get(root).ok_or(TreeRsiError::InvalidOption {
            option: "root",
            message: "the requested root is not a node of the input tree",
        })?,
        None => nodes.len() - 1,
    };

    let n = nodes.len();
    let mut adjacency = vec![Vec::new(); n];
    for (a, b) in reference.site_index_network().edges() {
        let (a, b) = (position[&a], position[&b]);
        adjacency[a].push(b);
        adjacency[b].push(a);
    }
    let mut parent = vec![None; n];
    let mut children = vec![Vec::new(); n];
    let mut preorder = Vec::with_capacity(n);
    let mut visited = vec![false; n];
    let mut stack = vec![root];
    visited[root] = true;
    while let Some(node) = stack.pop() {
        preorder.push(node);
        let mut next = adjacency[node]
            .iter()
            .copied()
            .filter(|&neighbor| !visited[neighbor])
            .collect::<Vec<_>>();
        next.sort_unstable();
        for &child in &next {
            visited[child] = true;
            parent[child] = Some(node);
        }
        children[node] = next.clone();
        // Push in reverse so the smallest child is visited first.
        stack.extend(next.into_iter().rev());
    }
    if preorder.len() != n {
        return Err(TreeRsiError::InternalInvariant {
            message: "validated tree is not connected",
        });
    }
    let postorder = preorder
        .iter()
        .rev()
        .copied()
        .filter(|&node| node != root)
        .collect::<Vec<_>>();

    // k = ceil(chi_max / d) + p (paper Eq. 7).
    let min_parent_dim = (0..n)
        .filter(|&node| !children[node].is_empty())
        .map(|node| phys[node].dim)
        .min()
        .unwrap_or(1)
        .max(1);
    let k = match (options.sketch_dim, options.max_bond_dim) {
        (Some(k), _) => k,
        (None, Some(max_bond_dim)) => max_bond_dim
            .div_ceil(min_parent_dim)
            .checked_add(options.oversampling)
            .ok_or(TreeRsiError::SizeOverflow {
                context: "derived sketch dimension",
            })?,
        (None, None) => {
            return Err(TreeRsiError::InternalInvariant {
                message: "options validation admitted no sketch dimension",
            })
        }
    };

    // physical-assignment counts, saturated just above `k`
    // (invariant: only the comparison `D_v <= k` matters; author
    // src/multiply_rsi.py:94-98).
    let cap = k.checked_add(1).ok_or(TreeRsiError::SizeOverflow {
        context: "complement size sentinel",
    })?;
    let mut subtree = vec![1usize; n];
    for &node in preorder.iter().rev() {
        let mut size = phys[node].dim.min(cap);
        for &child in &children[node] {
            size = capped_mul(size, subtree[child], cap);
        }
        subtree[node] = size;
    }
    // outside[v]: assignments of every node outside subtree(v).
    let mut outside = vec![1usize; n];
    let mut exact = vec![false; n];
    for &node in &preorder {
        // Prefix/suffix products avoid a quadratic sibling scan at a star.
        let kids = &children[node];
        let mut prefix = Vec::with_capacity(kids.len() + 1);
        prefix.push(outside[node]);
        for &child in kids {
            prefix.push(capped_mul(
                *prefix.last().unwrap_or(&1),
                subtree[child],
                cap,
            ));
        }
        let mut suffix = 1;
        for i in (0..kids.len()).rev() {
            let child = kids[i];
            let complement = capped_mul(prefix[i], suffix, cap);
            exact[child] = complement <= k;
            outside[child] = capped_mul(complement, phys[node].dim, cap);
            suffix = capped_mul(suffix, subtree[child], cap);
        }
    }
    let exact_up = subtree.iter().map(|&size| size <= k).collect();
    let exact_down = outside.iter().map(|&size| size <= k).collect();

    // requirement closure: a sketched environment needs the
    // messages of every other neighbor of the parent; a downward message is
    // derived from the parent-side environment.
    // `uses_parent_side[v]`: H_v (sketched v) or mu_(p -> v) is built, and both
    // contract A_p with the messages of every neighbor of p except v.
    let mut needs_sketched_env = vec![false; n];
    let mut needs_down = vec![false; n];
    let mut uses_parent_side = vec![false; n];
    for &node in &postorder {
        needs_down[node] = children[node].iter().any(|&child| uses_parent_side[child]);
        needs_sketched_env[node] = !exact[node];
        uses_parent_side[node] = needs_sketched_env[node] || needs_down[node];
    }
    let mut needs_up = vec![false; n];
    for siblings in &children {
        let needed = siblings
            .iter()
            .filter(|&&child| uses_parent_side[child])
            .count();
        for &child in siblings {
            needs_up[child] = needed > usize::from(uses_parent_side[child]);
        }
    }
    for &node in &preorder {
        if let Some(p) = parent[node]
            && needs_up[p]
        {
            needs_up[node] = true;
        }
    }
    let mut needs_probe = vec![false; n];
    for node in 0..n {
        if needs_up[node] {
            needs_probe[node] = true;
        }
        if needs_down[node]
            && let Some(p) = parent[node]
        {
            needs_probe[p] = true;
        }
    }

    Ok(Plan {
        nodes,
        root,
        parent,
        children,
        postorder,
        preorder,
        phys,
        k,
        exact,
        exact_up,
        exact_down,
        needs_sketched_env,
        needs_down,
        needs_up,
        needs_probe,
    })
}

pub(crate) fn node_tensor<'a, V: TreeRsiNode>(
    tree: &'a TreeTN<IdxTensor, V>,
    node: &V,
) -> Result<&'a IdxTensor> {
    let index = tree
        .node_index(node)
        .ok_or(TreeRsiError::InternalInvariant {
            message: "a validated node has no graph index",
        })?;
    tree.tensor(index).ok_or(TreeRsiError::InternalInvariant {
        message: "a validated node has no tensor",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn size_arithmetic_saturates_or_reports_overflow_before_allocation() {
        assert_eq!(capped_mul(usize::MAX, 2, 7), 7);
        assert_eq!(capped_mul(2, 3, 7), 6);
        assert_eq!(capped_mul(0, usize::MAX, 7), 0);
        assert!(matches!(
            checked_product([usize::MAX, 2], "test"),
            Err(TreeRsiError::SizeOverflow { .. })
        ));
        assert_eq!(checked_product([], "test").unwrap(), 1);
        assert!(enforce_limit("test", 7, 7).is_ok());
        assert!(matches!(
            enforce_limit("test", 8, 7),
            Err(TreeRsiError::ResourceLimit { .. })
        ));
    }
}
