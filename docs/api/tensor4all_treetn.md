# tensor4all-treetn

## src/dyn_treetn.rs

### `pub fn new() -> Self` (impl DynTreeTN < V >)

Create a new empty DynTreeTN.

### `pub fn add_node(&mut self, node_name: V, tensor_like: T) -> Result < NodeIndex >` (impl DynTreeTN < V >)

Add a TensorLike object as a node in the network. The object is boxed and stored as a trait object. Its external indices become the site/physical indices for this node.

### `pub fn add_boxed_node(&mut self, node_name: V, boxed: BoxedTensorLike) -> Result < NodeIndex >` (impl DynTreeTN < V >)

Add a pre-boxed TensorLike object as a node. This is useful when you already have a `BoxedTensorLike` or when working with trait objects directly.

### `pub fn node(&self, node: NodeIndex) -> Option < & dyn TensorLike < Id = DynId , Symm = NoSymmSpace , Tags = TagSet > >` (impl DynTreeTN < V >)

Get a reference to a node's TensorLike object by NodeIndex.

### `pub fn node_by_name(&self, name: & V) -> Option < & dyn TensorLike < Id = DynId , Symm = NoSymmSpace , Tags = TagSet > >` (impl DynTreeTN < V >)

Get a reference to a node's TensorLike object by node name.

### `pub fn connect(&mut self, node_a: NodeIndex, index_a: & DynIndex, node_b: NodeIndex, index_b: & DynIndex) -> Result < EdgeIndex >` (impl DynTreeTN < V >)

Connect two nodes with a bond. The indices must exist in the respective nodes' external indices and have matching dimensions.

### `pub fn bond_index(&self, edge: EdgeIndex) -> Option < & Index < DynId , NoSymmSpace > >` (impl DynTreeTN < V >)

Get a reference to a bond index by EdgeIndex.

### `pub fn bond_index_mut(&mut self, edge: EdgeIndex) -> Option < & mut Index < DynId , NoSymmSpace > >` (impl DynTreeTN < V >)

Get a mutable reference to a bond index by EdgeIndex.

### `pub fn node_count(&self) -> usize` (impl DynTreeTN < V >)

Get the number of nodes in the network.

### `pub fn edge_count(&self) -> usize` (impl DynTreeTN < V >)

Get the number of edges in the network.

### `pub fn node_names(&self) -> Vec < V >` (impl DynTreeTN < V >)

Get all node names in the network.

### `pub fn node_indices(&self) -> Vec < NodeIndex >` (impl DynTreeTN < V >)

Get all node indices in the network.

### `pub fn node_index(&self, name: & V) -> Option < NodeIndex >` (impl DynTreeTN < V >)

Get the NodeIndex for a node name.

### `pub fn node_name(&self, node: NodeIndex) -> Option < & V >` (impl DynTreeTN < V >)

Get the node name for a NodeIndex.

### `pub fn site_index_network(&self) -> & SiteIndexNetwork < V , DynId , NoSymmSpace , TagSet >` (impl DynTreeTN < V >)

Get a reference to the site index network.

### `pub fn canonical_center(&self) -> & HashSet < V >` (impl DynTreeTN < V >)

Get a reference to the orthogonalization region.

### `pub fn contract_to_tensor(&self) -> Result < TensorDynLen < DynId , NoSymmSpace > >` (impl DynTreeTN < V >)

Contract the entire network to a single tensor. Each node's `to_tensor()` is called to convert it to a `TensorDynLen`, then all tensors are contracted along their connected indices.

### ` fn default() -> Self` (impl DynTreeTN < V >)

### ` fn clone(&self) -> Self` (impl DynTreeTN < V >)

### ` fn fmt(&self, f: & mut std :: fmt :: Formatter < '_ >) -> std :: fmt :: Result` (impl DynTreeTN < V >)

### ` fn make_tensor(indices: Vec < DynIndex >) -> TensorDynLen < DynId , NoSymmSpace >`

### ` fn test_dyn_treetn_new()`

### ` fn test_dyn_treetn_add_node()`

### ` fn test_dyn_treetn_heterogeneous_nodes()`

### ` fn test_dyn_treetn_connect()`

### ` fn test_dyn_treetn_contract_single_node()`

### ` fn test_dyn_treetn_contract_two_nodes()`

### ` fn test_dyn_treetn_clone()`

### ` fn test_dyn_treetn_mix_tensor_and_treetn()`

### ` fn test_dyn_treetn_nested_networks()`

## src/named_graph.rs

### `pub fn new() -> Self` (impl NamedGraph < NodeName , NodeData , EdgeData , Ty >)

Create a new empty NamedGraph.

### `pub fn with_capacity(nodes: usize, edges: usize) -> Self` (impl NamedGraph < NodeName , NodeData , EdgeData , Ty >)

Create a new NamedGraph with initial capacity.

### `pub fn add_node(&mut self, node_name: NodeName, data: NodeData) -> Result < NodeIndex , String >` (impl NamedGraph < NodeName , NodeData , EdgeData , Ty >)

Add a node with the given name and data. Returns an error if the node already exists.

### `pub fn has_node(&self, node_name: & NodeName) -> bool` (impl NamedGraph < NodeName , NodeData , EdgeData , Ty >)

Check if a node exists.

### `pub fn node_index(&self, node_name: & NodeName) -> Option < NodeIndex >` (impl NamedGraph < NodeName , NodeData , EdgeData , Ty >)

Get the NodeIndex for a node name.

### `pub fn node_name(&self, node: NodeIndex) -> Option < & NodeName >` (impl NamedGraph < NodeName , NodeData , EdgeData , Ty >)

Get the node name for a NodeIndex.

### `pub fn node_data(&self, node_name: & NodeName) -> Option < & NodeData >` (impl NamedGraph < NodeName , NodeData , EdgeData , Ty >)

Get a reference to the data of a node (by node name).

### `pub fn node_data_mut(&mut self, node_name: & NodeName) -> Option < & mut NodeData >` (impl NamedGraph < NodeName , NodeData , EdgeData , Ty >)

Get a mutable reference to the data of a node (by node name).

### `pub fn node_weight(&self, node: NodeIndex) -> Option < & NodeData >` (impl NamedGraph < NodeName , NodeData , EdgeData , Ty >)

Get a reference to the data of a node (by NodeIndex).

### `pub fn node_weight_mut(&mut self, node: NodeIndex) -> Option < & mut NodeData >` (impl NamedGraph < NodeName , NodeData , EdgeData , Ty >)

Get a mutable reference to the data of a node (by NodeIndex).

### `pub fn add_edge(&mut self, n1: & NodeName, n2: & NodeName, weight: EdgeData) -> Result < EdgeIndex , String >` (impl NamedGraph < NodeName , NodeData , EdgeData , Ty >)

Add an edge between two nodes. Returns an error if either node doesn't exist.

### `pub fn edge_weight(&self, n1: & NodeName, n2: & NodeName) -> Option < & EdgeData >` (impl NamedGraph < NodeName , NodeData , EdgeData , Ty >)

Get the weight of an edge between two nodes.

### `pub fn edge_weight_mut(&mut self, n1: & NodeName, n2: & NodeName) -> Option < & mut EdgeData >` (impl NamedGraph < NodeName , NodeData , EdgeData , Ty >)

Get a mutable reference to the weight of an edge between two nodes.

### `pub fn neighbors(&self, node_name: & NodeName) -> Vec < & NodeName >` (impl NamedGraph < NodeName , NodeData , EdgeData , Ty >)

Get all neighbors of a node.

### `pub fn node_names(&self) -> Vec < & NodeName >` (impl NamedGraph < NodeName , NodeData , EdgeData , Ty >)

Get all node names.

### `pub fn node_count(&self) -> usize` (impl NamedGraph < NodeName , NodeData , EdgeData , Ty >)

Get the number of nodes.

### `pub fn edge_count(&self) -> usize` (impl NamedGraph < NodeName , NodeData , EdgeData , Ty >)

Get the number of edges.

### `pub fn remove_node(&mut self, node_name: & NodeName) -> Option < NodeData >` (impl NamedGraph < NodeName , NodeData , EdgeData , Ty >)

Remove a node and all its edges. Returns the node data if the node existed.

### `pub fn remove_edge(&mut self, n1: & NodeName, n2: & NodeName) -> Option < EdgeData >` (impl NamedGraph < NodeName , NodeData , EdgeData , Ty >)

Remove an edge between two nodes. Returns the edge weight if the edge existed.

### `pub fn contains_node(&self, node: NodeIndex) -> bool` (impl NamedGraph < NodeName , NodeData , EdgeData , Ty >)

Check if a node exists in the internal graph.

### `pub fn graph(&self) -> & StableGraph < NodeData , EdgeData , Ty >` (impl NamedGraph < NodeName , NodeData , EdgeData , Ty >)

Get a reference to the internal graph. This allows direct access to petgraph algorithms that work with NodeIndex.

### `pub fn graph_mut(&mut self) -> & mut StableGraph < NodeData , EdgeData , Ty >` (impl NamedGraph < NodeName , NodeData , EdgeData , Ty >)

Get a mutable reference to the internal graph. **Warning**: Directly modifying the internal graph can break the node-name-to-index mapping. Use the provided methods instead.

### `pub fn euler_tour_edges(&self, root: & NodeName) -> Option < Vec < (NodeIndex , NodeIndex) > >` (impl NamedGraph < NodeName , NodeData , EdgeData , Ty >)

Perform an Euler tour traversal starting from the given root node. The Euler tour visits each edge exactly twice (once in each direction), forming a closed walk that covers all edges. This is useful for sweep

### `pub fn euler_tour_edges_by_index(&self, root: NodeIndex) -> Vec < (NodeIndex , NodeIndex) >` (impl NamedGraph < NodeName , NodeData , EdgeData , Ty >)

Perform an Euler tour traversal starting from the given root NodeIndex. See [`euler_tour_edges`] for details.

### `pub fn euler_tour_vertices(&self, root: & NodeName) -> Option < Vec < NodeIndex > >` (impl NamedGraph < NodeName , NodeData , EdgeData , Ty >)

Perform an Euler tour traversal and return the vertex sequence. This returns the sequence of vertices visited, including repeated visits. Each vertex appears multiple times based on its degree in the tree.

### `pub fn euler_tour_vertices_by_index(&self, root: NodeIndex) -> Vec < NodeIndex >` (impl NamedGraph < NodeName , NodeData , EdgeData , Ty >)

Perform an Euler tour traversal and return the vertex sequence by NodeIndex. See [`euler_tour_vertices`] for details.

### ` fn default() -> Self` (impl NamedGraph < NodeName , NodeData , EdgeData , Ty >)

### ` fn clone(&self) -> Self` (impl NamedGraph < NodeName , NodeData , EdgeData , Ty >)

### ` fn fmt(&self, f: & mut std :: fmt :: Formatter < '_ >) -> std :: fmt :: Result` (impl NamedGraph < NodeName , NodeData , EdgeData , Ty >)

### ` fn test_named_graph_basic()`

### ` fn test_named_graph_tuple_nodes()`

### ` fn test_named_graph_remove()`

### ` fn test_euler_tour_chain()`

### ` fn test_euler_tour_single_node()`

## src/node_name_network.rs

### `pub fn empty() -> Self` (impl CanonicalizeEdges)

Create an empty edge sequence (no-op canonicalization).

### `pub fn from_edges(edges: Vec < (NodeIndex , NodeIndex) >) -> Self` (impl CanonicalizeEdges)

Create from a list of edges. Note: For path-based canonicalization, edges should be connected (each edge's `to` equals next edge's `from`). For full canonicalization, edges may not be connected

### `pub fn is_empty(&self) -> bool` (impl CanonicalizeEdges)

Check if empty (already at target, no work needed).

### `pub fn len(&self) -> usize` (impl CanonicalizeEdges)

Number of edges to process.

### `pub fn iter(&self) -> impl Iterator < Item = & (NodeIndex , NodeIndex) >` (impl CanonicalizeEdges)

Iterate over edges in order.

### `pub fn target(&self) -> Option < NodeIndex >` (impl CanonicalizeEdges)

Get the final target node (orthogonality center). Returns `None` if empty.

### `pub fn start(&self) -> Option < NodeIndex >` (impl CanonicalizeEdges)

Get the starting node (first node to be factorized). Returns `None` if empty.

### ` fn into_iter(self) -> Self :: IntoIter` (impl CanonicalizeEdges)

### ` fn into_iter(self) -> Self :: IntoIter` (impl & 'a CanonicalizeEdges)

### `pub fn new() -> Self` (impl NodeNameNetwork < NodeName >)

Create a new empty NodeNameNetwork.

### `pub fn with_capacity(nodes: usize, edges: usize) -> Self` (impl NodeNameNetwork < NodeName >)

Create a new NodeNameNetwork with initial capacity.

### `pub fn add_node(&mut self, node_name: NodeName) -> Result < NodeIndex , String >` (impl NodeNameNetwork < NodeName >)

Add a node to the network. Returns an error if the node already exists.

### `pub fn has_node(&self, node_name: & NodeName) -> bool` (impl NodeNameNetwork < NodeName >)

Check if a node exists.

### `pub fn add_edge(&mut self, n1: & NodeName, n2: & NodeName) -> Result < EdgeIndex , String >` (impl NodeNameNetwork < NodeName >)

Add an edge between two nodes. Returns an error if either node doesn't exist.

### `pub fn node_index(&self, node_name: & NodeName) -> Option < NodeIndex >` (impl NodeNameNetwork < NodeName >)

Get the NodeIndex for a node name.

### `pub fn node_name(&self, node: NodeIndex) -> Option < & NodeName >` (impl NodeNameNetwork < NodeName >)

Get the node name for a NodeIndex.

### `pub fn node_names(&self) -> Vec < & NodeName >` (impl NodeNameNetwork < NodeName >)

Get all node names.

### `pub fn node_count(&self) -> usize` (impl NodeNameNetwork < NodeName >)

Get the number of nodes.

### `pub fn edge_count(&self) -> usize` (impl NodeNameNetwork < NodeName >)

Get the number of edges.

### `pub fn graph(&self) -> & StableGraph < () , () , Undirected >` (impl NodeNameNetwork < NodeName >)

Get a reference to the internal graph.

### `pub fn graph_mut(&mut self) -> & mut StableGraph < () , () , Undirected >` (impl NodeNameNetwork < NodeName >)

Get a mutable reference to the internal graph. **Warning**: Directly modifying the internal graph can break the node-name-to-index mapping.

### `pub fn post_order_dfs(&self, root: & NodeName) -> Option < Vec < NodeName > >` (impl NodeNameNetwork < NodeName >)

Perform a post-order DFS traversal starting from the given root node. Returns node names in post-order (children before parents, leaves first).

### `pub fn post_order_dfs_by_index(&self, root: NodeIndex) -> Vec < NodeIndex >` (impl NodeNameNetwork < NodeName >)

Perform a post-order DFS traversal starting from the given root NodeIndex. Returns NodeIndex in post-order (children before parents, leaves first).

### `pub fn euler_tour_edges(&self, root: & NodeName) -> Option < Vec < (NodeIndex , NodeIndex) > >` (impl NodeNameNetwork < NodeName >)

Perform an Euler tour traversal starting from the given root node. Delegates to [`NamedGraph::euler_tour_edges`].

### `pub fn euler_tour_edges_by_index(&self, root: NodeIndex) -> Vec < (NodeIndex , NodeIndex) >` (impl NodeNameNetwork < NodeName >)

Perform an Euler tour traversal starting from the given root NodeIndex. Delegates to [`NamedGraph::euler_tour_edges_by_index`].

### `pub fn euler_tour_vertices(&self, root: & NodeName) -> Option < Vec < NodeIndex > >` (impl NodeNameNetwork < NodeName >)

Perform an Euler tour traversal and return the vertex sequence. Delegates to [`NamedGraph::euler_tour_vertices`].

### `pub fn euler_tour_vertices_by_index(&self, root: NodeIndex) -> Vec < NodeIndex >` (impl NodeNameNetwork < NodeName >)

Perform an Euler tour traversal and return the vertex sequence by NodeIndex. Delegates to [`NamedGraph::euler_tour_vertices_by_index`].

### `pub fn path_between(&self, from: NodeIndex, to: NodeIndex) -> Option < Vec < NodeIndex > >` (impl NodeNameNetwork < NodeName >)

Find the shortest path between two nodes using A* algorithm. Since this is an unweighted graph, we use unit edge weights.

### `pub fn is_connected_subset(&self, nodes: & HashSet < NodeIndex >) -> bool` (impl NodeNameNetwork < NodeName >)

Check if a subset of nodes forms a connected subgraph. Uses DFS to verify that all nodes in the subset are reachable from each other within the induced subgraph.

### ` fn nodes_to_edges(nodes: & [NodeIndex]) -> CanonicalizeEdges` (impl NodeNameNetwork < NodeName >)

Convert a node sequence to an edge sequence.

### `pub fn edges_to_canonicalize(&self, current_region: Option < & HashSet < NodeIndex > >, target: NodeIndex) -> CanonicalizeEdges` (impl NodeNameNetwork < NodeName >)

Compute edges to canonicalize from current state to target.

### `pub fn edges_to_canonicalize_by_names(&self, target: & NodeName) -> Option < Vec < (NodeName , NodeName) > >` (impl NodeNameNetwork < NodeName >)

Compute edges to canonicalize from leaves to target, returning node names. This is similar to `edges_to_canonicalize(None, target)` but returns `(from_name, to_name)` pairs instead of `(NodeIndex, NodeIndex)`.

### ` fn compute_parent_edges(&self, nodes: & [NodeIndex], root: NodeIndex) -> CanonicalizeEdges` (impl NodeNameNetwork < NodeName >)

Compute parent edges for each node in the given order.

### `pub fn edges_to_canonicalize_to_region(&self, target_region: & HashSet < NodeIndex >) -> CanonicalizeEdges` (impl NodeNameNetwork < NodeName >)

Compute edges to canonicalize from leaves towards a connected region (multiple centers). Given a set of target nodes forming a connected region, this function returns all edges (src, dst) where:

### `pub fn edges_to_canonicalize_to_region_by_names(&self, target_region: & HashSet < NodeName >) -> Option < Vec < (NodeName , NodeName) > >` (impl NodeNameNetwork < NodeName >)

Compute edges to canonicalize towards a region, returning node names. This is similar to `edges_to_canonicalize_to_region` but takes and returns node names instead of NodeIndex.

### `pub fn same_topology(&self, other: & Self) -> bool` (impl NodeNameNetwork < NodeName >)

Check if two networks have the same topology (same nodes and edges).

### ` fn default() -> Self` (impl NodeNameNetwork < NodeName >)

### ` fn test_node_name_network_basic()`

### ` fn test_post_order_dfs_chain()`

### ` fn test_path_between()`

### ` fn test_is_connected_subset()`

### ` fn test_same_topology()`

### ` fn test_euler_tour_chain()`

### ` fn test_euler_tour_y_shape()`

### ` fn test_euler_tour_single_node()`

### ` fn test_euler_tour_star()`

## src/operator/compose.rs

### `pub fn are_exclusive_operators(target: & SiteIndexNetwork < V , Id , Symm >, operators: & [& O]) -> bool`

Check if a set of operators are exclusive (non-overlapping) on the target network. Operators are exclusive if: 1. **Vertex-disjoint**: No two operators share a node

### ` fn check_path_exclusive(target: & SiteIndexNetwork < V , Id , Symm >, set_a: & HashSet < V >, set_b: & HashSet < V >, all_sets: & [HashSet < V >]) -> bool`

Check if paths between two operator regions don't cross other operators.

### `pub fn compose_exclusive_linear_operators(target: & SiteIndexNetwork < V , Id , Symm >, operators: & [& LinearOperator < Id , Symm , V >], gap_site_indices: & HashMap < V , Vec < (Index < Id , Symm > , Index < Id , Symm >) > >) -> Result < LinearOperator < Id , Symm , V > >`

Compose exclusive LinearOperators into a single LinearOperator. This function takes multiple non-overlapping operators and combines them into a single operator that acts on the full target space. Gap positions (nodes not

### `pub fn compose_exclusive_operators(_target: & SiteIndexNetwork < V , Id , Symm >, _operators: & [& O], _gap_site_indices: & HashMap < V , Vec < (Index < Id , Symm > , Index < Id , Symm >) > >) -> Result < LinearOperator < Id , Symm , V > >`

Compose exclusive operators into a single operator (convenience wrapper). This is a generic version that accepts any type implementing the Operator trait. For actual composition, use [`compose_exclusive_linear_operators`] with LinearOperator inputs.

### ` fn make_index(dim: usize) -> DynIndex`

### ` fn create_chain_site_network(n: usize) -> SiteIndexNetwork < String , DynId , NoSymmSpace >`

### ` fn create_linear_operator_from_treetn(mpo: TreeTN < DynId , NoSymmSpace , String >, input_indices: & [(String , DynIndex , DynIndex)], output_indices: & [(String , DynIndex , DynIndex)]) -> LinearOperator < DynId , NoSymmSpace , String >`

Create a simple LinearOperator from a TreeTN with explicit index mappings.

### ` fn test_are_exclusive_disjoint()`

### ` fn test_are_exclusive_overlapping()`

### ` fn test_are_exclusive_single_node_operators()`

### ` fn test_compose_exclusive_linear_operators_basic()`

### ` fn test_compose_exclusive_linear_operators_single_operators()`

### ` fn test_compose_exclusive_linear_operators_no_gap()`

### ` fn test_compose_exclusive_linear_operators_overlap_error()`

### ` fn test_compose_gap_identity_tensor_is_diagonal()`

## src/operator/identity.rs

### `pub fn build_identity_operator_tensor(site_indices: & [Index < Id , Symm >], output_site_indices: & [Index < Id , Symm >]) -> Result < TensorDynLen < Id , Symm > >`

Build an identity operator tensor for a gap node. For a node with site indices `{s1, s2, ...}` and bond indices `{l1, l2, ...}`, this creates an identity tensor where:

### `pub fn build_identity_operator_tensor_c64(site_indices: & [Index < Id , Symm >], output_site_indices: & [Index < Id , Symm >]) -> Result < TensorDynLen < Id , Symm > >`

Build an identity operator tensor with complex data type. Same as [`build_identity_operator_tensor`] but returns a complex tensor.

### ` fn make_index(dim: usize) -> DynIndex`

### ` fn get_f64_data(tensor: & TensorDynLen < DynId , NoSymmSpace >) -> & [f64]`

### ` fn get_c64_data(tensor: & TensorDynLen < DynId , NoSymmSpace >) -> & [Complex64]`

### ` fn test_identity_single_site()`

### ` fn test_identity_two_sites()`

### ` fn test_identity_dimension_mismatch()`

### ` fn test_identity_empty()`

### ` fn test_identity_c64()`

## src/operator/mod.rs

### `pub fn site_indices(&self) -> HashSet < Index < Id , Symm > >` (trait Operator)

Get all site indices this operator acts on. Returns the union of site indices across all nodes.

### `pub fn site_index_network(&self) -> & SiteIndexNetwork < V , Id , Symm >` (trait Operator)

Get the site index network describing this operator's structure. The site index network contains: - Topology: which nodes connect to which

### `pub fn node_names(&self) -> HashSet < V >` (trait Operator default)

Get the set of node names this operator covers. Default implementation extracts node names from the site index network.

## src/options.rs

### ` fn default() -> Self` (impl CanonicalizationOptions)

### `pub fn new() -> Self` (impl CanonicalizationOptions)

Create options with default settings.

### `pub fn forced() -> Self` (impl CanonicalizationOptions)

Create options that force full canonicalization.

### `pub fn with_form(mut self, form: CanonicalForm) -> Self` (impl CanonicalizationOptions)

Set the canonical form.

### `pub fn force(mut self) -> Self` (impl CanonicalizationOptions)

Set force mode (always perform full canonicalization).

### `pub fn smart(mut self) -> Self` (impl CanonicalizationOptions)

Disable force mode (check current state before canonicalizing).

### ` fn default() -> Self` (impl TruncationOptions)

### `pub fn new() -> Self` (impl TruncationOptions)

Create options with default settings (no truncation limits).

### `pub fn with_max_rank(mut self, rank: usize) -> Self` (impl TruncationOptions)

Create options with a maximum rank.

### `pub fn with_rtol(mut self, rtol: f64) -> Self` (impl TruncationOptions)

Create options with a relative tolerance.

### `pub fn with_form(mut self, form: CanonicalForm) -> Self` (impl TruncationOptions)

Set the canonical form / algorithm.

### ` fn default() -> Self` (impl SplitOptions)

### `pub fn new() -> Self` (impl SplitOptions)

Create options with default settings.

### `pub fn with_max_rank(mut self, rank: usize) -> Self` (impl SplitOptions)

Create options with a maximum rank.

### `pub fn with_rtol(mut self, rtol: f64) -> Self` (impl SplitOptions)

Create options with a relative tolerance.

### `pub fn with_form(mut self, form: CanonicalForm) -> Self` (impl SplitOptions)

Set the canonical form / algorithm.

### `pub fn with_final_sweep(mut self, final_sweep: bool) -> Self` (impl SplitOptions)

Enable or disable final sweep for global optimization.

## src/random.rs

### `pub fn uniform(dim: usize) -> Self` (impl LinkSpace < V >)

Create a uniform link space where all bonds have the same dimension.

### `pub fn per_edge(dims: HashMap < (V , V) , usize >) -> Self` (impl LinkSpace < V >)

Create a per-edge link space from a map of edge dimensions.

### `pub fn get(&self, a: & V, b: & V) -> Option < usize >` (impl LinkSpace < V >)

Get the dimension for an edge between two nodes. For `PerEdge`, the key is normalized to `(min(a, b), max(a, b))`.

### `pub fn random_treetn_f64(rng: & mut R, site_network: & SiteIndexNetwork < V , DynId , NoSymmSpace >, link_space: LinkSpace < V >) -> TreeTN < DynId , NoSymmSpace , V >`

Create a random f64 TreeTN from a site index network. Generates random tensors at each node with: - Site indices from the `site_network`

### `pub fn random_treetn_c64(rng: & mut R, site_network: & SiteIndexNetwork < V , DynId , NoSymmSpace >, link_space: LinkSpace < V >) -> TreeTN < DynId , NoSymmSpace , V >`

Create a random Complex64 TreeTN from a site index network. Similar to [`random_treetn_f64`], but generates complex-valued tensors where both real and imaginary parts are drawn from standard normal distribution.

### ` fn random_treetn_impl(rng: & mut R, site_network: & SiteIndexNetwork < V , DynId , NoSymmSpace >, link_space: LinkSpace < V >, is_complex: bool) -> TreeTN < DynId , NoSymmSpace , V >`

Internal implementation for creating random TreeTN.

### ` fn test_random_treetn_f64_two_nodes()`

### ` fn test_random_treetn_c64_chain()`

### ` fn test_link_space_per_edge()`

## src/site_index_network.rs

### `pub fn new() -> Self` (impl SiteIndexNetwork < NodeName , Id , Symm , Tags >)

Create a new empty SiteIndexNetwork.

### `pub fn with_capacity(nodes: usize, edges: usize) -> Self` (impl SiteIndexNetwork < NodeName , Id , Symm , Tags >)

Create a new SiteIndexNetwork with initial capacity.

### `pub fn add_node(&mut self, node_name: NodeName, site_space: impl Into < HashSet < Index < Id , Symm , Tags > > >) -> Result < NodeIndex , String >` (impl SiteIndexNetwork < NodeName , Id , Symm , Tags >)

Add a node with site space (physical indices).

### `pub fn has_node(&self, node_name: & NodeName) -> bool` (impl SiteIndexNetwork < NodeName , Id , Symm , Tags >)

Check if a node exists.

### `pub fn site_space(&self, node_name: & NodeName) -> Option < & HashSet < Index < Id , Symm , Tags > > >` (impl SiteIndexNetwork < NodeName , Id , Symm , Tags >)

Get the site space (physical indices) for a node.

### `pub fn site_space_mut(&mut self, node_name: & NodeName) -> Option < & mut HashSet < Index < Id , Symm , Tags > > >` (impl SiteIndexNetwork < NodeName , Id , Symm , Tags >)

Get a mutable reference to the site space for a node.

### `pub fn site_space_by_index(&self, node: NodeIndex) -> Option < & HashSet < Index < Id , Symm , Tags > > >` (impl SiteIndexNetwork < NodeName , Id , Symm , Tags >)

Get the site space by NodeIndex.

### `pub fn add_edge(&mut self, n1: & NodeName, n2: & NodeName) -> Result < EdgeIndex , String >` (impl SiteIndexNetwork < NodeName , Id , Symm , Tags >)

Add an edge between two nodes. Returns an error if either node doesn't exist.

### `pub fn node_index(&self, node_name: & NodeName) -> Option < NodeIndex >` (impl SiteIndexNetwork < NodeName , Id , Symm , Tags >)

Get the NodeIndex for a node name.

### `pub fn node_name(&self, node: NodeIndex) -> Option < & NodeName >` (impl SiteIndexNetwork < NodeName , Id , Symm , Tags >)

Get the node name for a NodeIndex.

### `pub fn node_names(&self) -> Vec < & NodeName >` (impl SiteIndexNetwork < NodeName , Id , Symm , Tags >)

Get all node names.

### `pub fn node_count(&self) -> usize` (impl SiteIndexNetwork < NodeName , Id , Symm , Tags >)

Get the number of nodes.

### `pub fn edge_count(&self) -> usize` (impl SiteIndexNetwork < NodeName , Id , Symm , Tags >)

Get the number of edges.

### `pub fn topology(&self) -> & NodeNameNetwork < NodeName >` (impl SiteIndexNetwork < NodeName , Id , Symm , Tags >)

Get a reference to the underlying topology (NodeNameNetwork).

### `pub fn edges(&self) -> impl Iterator < Item = (NodeName , NodeName) > + '_` (impl SiteIndexNetwork < NodeName , Id , Symm , Tags >)

Get all edges as pairs of node names. Returns an iterator of `(NodeName, NodeName)` pairs.

### `pub fn neighbors(&self, node_name: & NodeName) -> impl Iterator < Item = NodeName > + '_` (impl SiteIndexNetwork < NodeName , Id , Symm , Tags >)

Get all neighbors of a node. Returns an iterator of neighbor node names.

### `pub fn graph(&self) -> & StableGraph < () , () , Undirected >` (impl SiteIndexNetwork < NodeName , Id , Symm , Tags >)

Get a reference to the internal graph.

### `pub fn graph_mut(&mut self) -> & mut StableGraph < () , () , Undirected >` (impl SiteIndexNetwork < NodeName , Id , Symm , Tags >)

Get a mutable reference to the internal graph. **Warning**: Directly modifying the internal graph can break consistency.

### `pub fn share_equivalent_site_index_network(&self, other: & Self) -> bool` (impl SiteIndexNetwork < NodeName , Id , Symm , Tags >)

Check if two SiteIndexNetworks share equivalent site index structure. Two networks are equivalent if: - Same topology (nodes and edges)

### `pub fn post_order_dfs(&self, root: & NodeName) -> Option < Vec < NodeName > >` (impl SiteIndexNetwork < NodeName , Id , Symm , Tags >)

Perform a post-order DFS traversal starting from the given root node.

### `pub fn post_order_dfs_by_index(&self, root: NodeIndex) -> Vec < NodeIndex >` (impl SiteIndexNetwork < NodeName , Id , Symm , Tags >)

Perform a post-order DFS traversal starting from the given root NodeIndex.

### `pub fn path_between(&self, from: NodeIndex, to: NodeIndex) -> Option < Vec < NodeIndex > >` (impl SiteIndexNetwork < NodeName , Id , Symm , Tags >)

Find the shortest path between two nodes.

### `pub fn is_connected_subset(&self, nodes: & HashSet < NodeIndex >) -> bool` (impl SiteIndexNetwork < NodeName , Id , Symm , Tags >)

Check if a subset of nodes forms a connected subgraph.

### `pub fn edges_to_canonicalize(&self, current_region: Option < & HashSet < NodeIndex > >, target: NodeIndex) -> CanonicalizeEdges` (impl SiteIndexNetwork < NodeName , Id , Symm , Tags >)

Compute edges to canonicalize from current state to target.

### `pub fn edges_to_canonicalize_by_names(&self, target: & NodeName) -> Option < Vec < (NodeName , NodeName) > >` (impl SiteIndexNetwork < NodeName , Id , Symm , Tags >)

Compute edges to canonicalize from leaves to target, returning node names. This is similar to `edges_to_canonicalize(None, target)` but returns `(from_name, to_name)` pairs instead of `(NodeIndex, NodeIndex)`.

### `pub fn edges_to_canonicalize_to_region(&self, target_region: & HashSet < NodeIndex >) -> CanonicalizeEdges` (impl SiteIndexNetwork < NodeName , Id , Symm , Tags >)

Compute edges to canonicalize from leaves towards a connected region (multiple centers). See [`NodeNameNetwork::edges_to_canonicalize_to_region`] for details.

### `pub fn edges_to_canonicalize_to_region_by_names(&self, target_region: & HashSet < NodeName >) -> Option < Vec < (NodeName , NodeName) > >` (impl SiteIndexNetwork < NodeName , Id , Symm , Tags >)

Compute edges to canonicalize towards a region, returning node names. See [`NodeNameNetwork::edges_to_canonicalize_to_region_by_names`] for details.

### `pub fn apply_operator_topology(&self, operator: & Self) -> Result < Self , String >` (impl SiteIndexNetwork < NodeName , Id , Symm , Tags >)

Check if an operator can act on this state (as a ket). Returns `Ok(result_network)` if the operator can act on self, where `result_network` is the SiteIndexNetwork of the output state.

### `pub fn compatible_site_dimensions(&self, other: & Self) -> bool` (impl SiteIndexNetwork < NodeName , Id , Symm , Tags >)

Check if this network has compatible site dimensions with another. Two networks have compatible site dimensions if: - Same topology (nodes and edges)

### ` fn default() -> Self` (impl SiteIndexNetwork < NodeName , Id , Symm , Tags >)

### ` fn test_site_index_network_basic()`

### ` fn test_post_order_dfs_chain()`

### ` fn test_path_between_chain()`

### ` fn test_edges_to_canonicalize_full()`

### ` fn test_is_connected_subset()`

### ` fn test_share_equivalent_site_index_network()`

### ` fn test_apply_operator_topology()`

### ` fn test_compatible_site_dimensions()`

## src/treetn/addition.rs

### `pub fn compute_merged_bond_indices(&self, other: & Self) -> Result < HashMap < (V , V) , MergedBondInfo < Id , Symm > > >` (impl TreeTN < Id , Symm , V >)

Compute merged bond indices for direct-sum addition. For each edge in the network, compute the merged bond information containing dimensions from both networks and a new merged index.

### `pub fn direct_sum_tensors(tensor_a: & TensorDynLen < Id , Symm >, tensor_b: & TensorDynLen < Id , Symm >, site_indices: & HashSet < Id >, bond_info_by_neighbor: & HashMap < V , & MergedBondInfo < Id , Symm > >, neighbor_names_a: & HashMap < Id , V >, neighbor_names_b: & HashMap < Id , V >) -> Result < TensorDynLen < Id , Symm > >`

Compute the direct sum of two tensors with merged bond indices. This function creates a new tensor that embeds tensor_a in the "A block" and tensor_b in the "B block" of each bond dimension.

### ` fn embed_block_f64(dest: & mut [f64], dest_dims: & [usize], src: & [f64], src_dims: & [usize], bond_positions: & [(usize , usize , usize)], is_a_block: bool) -> Result < () >`

Embed a source tensor block into a larger destination tensor for f64 data.

### ` fn embed_block_c64(dest: & mut [Complex64], dest_dims: & [usize], src: & [Complex64], src_dims: & [usize], bond_positions: & [(usize , usize , usize)], is_a_block: bool) -> Result < () >`

Embed a source tensor block into a larger destination tensor for Complex64 data.

### ` fn permute_data_f64(data: & [f64], dims: & [usize], perm: & [usize]) -> Vec < f64 >`

Permute tensor data according to axis permutation (f64 version).

### ` fn permute_data_c64(data: & [Complex64], dims: & [usize], perm: & [usize]) -> Vec < Complex64 >`

Permute tensor data according to axis permutation (Complex64 version).

## src/treetn/canonicalize.rs

### `pub fn canonicalize(mut self, canonical_center: impl IntoIterator < Item = V >, options: CanonicalizationOptions) -> Result < Self >` (impl TreeTN < Id , Symm , V >)

Canonicalize the network towards the specified center using options. This is the recommended unified API for canonicalization. It accepts: - Center nodes specified by their node names (V)

### `pub fn canonicalize_mut(&mut self, canonical_center: impl IntoIterator < Item = V >, options: CanonicalizationOptions) -> Result < () >` (impl TreeTN < Id , Symm , V >)

Canonicalize the network in-place towards the specified center using options. This is the `&mut self` version of [`canonicalize`].

### `pub(crate) fn canonicalize_impl(&mut self, canonical_center: impl IntoIterator < Item = V >, form: CanonicalForm, context_name: & str) -> Result < () >` (impl TreeTN < Id , Symm , V >)

Internal implementation for canonicalization. This is the core canonicalization logic that public methods delegate to.

## src/treetn/contraction.rs

### `pub fn sim_internal_inds(&self) -> Self` (impl TreeTN < Id , Symm , V >)

Create a copy with all internal (link/bond) indices replaced by fresh IDs. External (site/physical) indices remain unchanged. This is useful when contracting two TreeTNs that might have overlapping internal index IDs.

### `pub fn add(self, other: Self) -> Result < Self >` (impl TreeTN < Id , Symm , V >)

Add two TreeTN together using direct-sum (block) construction. This method constructs a new TTN whose bond indices are the **direct sums** of the original bond indices, so that the resulting network represents the

### `pub fn contract_to_tensor(&self) -> Result < TensorDynLen < Id , Symm > >` (impl TreeTN < Id , Symm , V >)

Contract the TreeTN to a single dense tensor. This method contracts all tensors in the network into a single tensor containing all physical indices. The contraction is performed using

### `pub fn contract_zipup(&self, other: & Self, center: & V, rtol: Option < f64 >, max_rank: Option < usize >) -> Result < Self >` (impl TreeTN < Id , Symm , V >)

Contract two TreeTNs with the same topology using the zip-up algorithm. The zip-up algorithm traverses from leaves towards the center, contracting corresponding nodes from both networks and optionally truncating at each step.

### `pub fn contract_zipup_with(&self, other: & Self, center: & V, form: CanonicalForm, rtol: Option < f64 >, max_rank: Option < usize >) -> Result < Self >` (impl TreeTN < Id , Symm , V >)

Contract two TreeTNs with the same topology using the zip-up algorithm with a specified form. See [`contract_zipup`](Self::contract_zipup) for details.

### `pub fn contract_naive(&self, other: & Self) -> Result < TensorDynLen < Id , Symm > >` (impl TreeTN < Id , Symm , V >)

Contract two TreeTNs using naive full contraction. This is a reference implementation that: 1. Replaces internal indices with fresh IDs (sim_internal_inds)

### `pub fn validate_ortho_consistency(&self) -> Result < () >` (impl TreeTN < Id , Symm , V >)

Validate that `canonical_center` and edge `ortho_towards` are consistent. Rules: - If `canonical_center` is empty (not canonicalized), all indices must have `ortho_towards == None`.

### ` fn default() -> Self` (impl ContractionOptions)

### `pub fn new(method: ContractionMethod) -> Self` (impl ContractionOptions)

Create options with specified method.

### `pub fn zipup() -> Self` (impl ContractionOptions)

Create options for zipup contraction.

### `pub fn fit() -> Self` (impl ContractionOptions)

Create options for fit contraction.

### `pub fn with_max_rank(mut self, max_rank: usize) -> Self` (impl ContractionOptions)

Set maximum bond dimension.

### `pub fn with_rtol(mut self, rtol: f64) -> Self` (impl ContractionOptions)

Set relative tolerance.

### `pub fn with_nsweeps(mut self, nsweeps: usize) -> Self` (impl ContractionOptions)

Set number of sweeps for Fit method.

### `pub fn with_convergence_tol(mut self, tol: f64) -> Self` (impl ContractionOptions)

Set convergence tolerance for Fit method.

### `pub fn with_factorize_alg(mut self, alg: FactorizeAlg) -> Self` (impl ContractionOptions)

Set factorization algorithm for Fit method.

### `pub fn contract(tn_a: & TreeTN < Id , Symm , V >, tn_b: & TreeTN < Id , Symm , V >, center: & V, options: ContractionOptions) -> Result < TreeTN < Id , Symm , V > >`

Contract two TreeTNs using the specified method. This is the main entry point for TreeTN contraction. It dispatches to the appropriate algorithm based on the options.

### `pub fn contract_naive_to_treetn(tn_a: & TreeTN < Id , Symm , V >, tn_b: & TreeTN < Id , Symm , V >, _center: & V, _max_rank: Option < usize >, _rtol: Option < f64 >) -> Result < TreeTN < Id , Symm , V > >`

Contract two TreeTNs using naive contraction, then decompose back to TreeTN. This method: 1. Contracts both networks to full tensors

## src/treetn/decompose.rs

### `pub fn new(nodes: HashMap < V , Vec < usize > >, edges: Vec < (V , V) >) -> Self` (impl TreeTopology < V >)

Create a new tree topology with the given nodes and edges.

### `pub fn validate(&self) -> Result < () >` (impl TreeTopology < V >)

Validate that this topology describes a tree.

### `pub fn factorize_tensor_to_treetn(tensor: & TensorDynLen < Id , Symm >, topology: & TreeTopology < V >) -> Result < TreeTN < Id , Symm , V > >`

Decompose a dense tensor into a TreeTN using QR-based factorization. This function takes a dense tensor and a tree topology specification, then recursively decomposes the tensor using QR factorization to create a TreeTN.

### `pub fn factorize_tensor_to_treetn_with(tensor: & TensorDynLen < Id , Symm >, topology: & TreeTopology < V >, alg: FactorizeAlg) -> Result < TreeTN < Id , Symm , V > >`

Factorize a dense tensor into a TreeTN using a specified factorization algorithm. This function takes a dense tensor and a tree topology specification, then recursively decomposes the tensor using the specified algorithm to create a TreeTN.

## src/treetn/fit.rs

### `pub fn new() -> Self` (impl FitEnvironment < Id , Symm , V >)

Create an empty environment cache.

### `pub fn get(&self, from: & V, to: & V) -> Option < & TensorDynLen < Id , Symm > >` (impl FitEnvironment < Id , Symm , V >)

Get the environment tensor for edge (from, to) if it exists.

### `pub(crate) fn insert(&mut self, from: V, to: V, env: TensorDynLen < Id , Symm >)` (impl FitEnvironment < Id , Symm , V >)

Insert an environment tensor for edge (from, to). This is mainly for testing; normally use `get_or_compute` for lazy evaluation.

### `pub fn contains(&self, from: & V, to: & V) -> bool` (impl FitEnvironment < Id , Symm , V >)

Check if environment exists for edge (from, to).

### `pub fn len(&self) -> usize` (impl FitEnvironment < Id , Symm , V >)

Get the number of cached environments.

### `pub fn is_empty(&self) -> bool` (impl FitEnvironment < Id , Symm , V >)

Check if the cache is empty.

### `pub fn clear(&mut self)` (impl FitEnvironment < Id , Symm , V >)

Clear all cached environments.

### `pub fn get_or_compute(&mut self, from: & V, to: & V, tn_a: & TreeTN < Id , Symm , V >, tn_b: & TreeTN < Id , Symm , V >, tn_c: & TreeTN < Id , Symm , V >) -> Result < TensorDynLen < Id , Symm > >` (impl FitEnvironment < Id , Symm , V >)

Get or compute the environment tensor for edge (from, to). If the environment is cached, returns it directly. Otherwise, recursively computes it from child environments (towards leaves)

### `pub fn invalidate(&mut self, region: impl IntoIterator < Item = & 'a V >, tn_c: & TreeTN < Id , Symm , V >)` (impl FitEnvironment < Id , Symm , V >)

Invalidate all caches affected by updates to tensors in region T. For each `t ∈ T`: 1. Remove all `env[(t, *)]` (0th generation)

### ` fn invalidate_recursive(&mut self, from: & V, to: & V, tn_c: & TreeTN < Id , Symm , V >)` (impl FitEnvironment < Id , Symm , V >)

Recursively invalidate caches starting from env[(from, to)] towards leaves. If env[(from, to)] exists, remove it and propagate to env[(to, x)] for all x ≠ from.

### `pub fn verify_structural_consistency(&self, tn_c: & TreeTN < Id , Symm , V >) -> Result < () >` (impl FitEnvironment < Id , Symm , V >)

Verify cache structural consistency. For any `env[(x, x1)]` where `x` is not a leaf (has neighbors other than `x1`), all child environments `env[(y, x)]` for neighbors `y ≠ x1` must exist.

### `pub fn verify_value_consistency(&self, tn_a: & TreeTN < Id , Symm , V >, tn_b: & TreeTN < Id , Symm , V >, tn_c: & TreeTN < Id , Symm , V >, tol: f64) -> Result < () >` (impl FitEnvironment < Id , Symm , V >)

Verify cache value consistency by recomputing all entries. Creates a fresh empty cache, recomputes all entries lazily, and compares with the existing cache values.

### `pub fn verify_consistency(&self, tn_a: & TreeTN < Id , Symm , V >, tn_b: & TreeTN < Id , Symm , V >, tn_c: & TreeTN < Id , Symm , V >, tol: f64) -> Result < () >` (impl FitEnvironment < Id , Symm , V >)

Verify both structural and value consistency.

### ` fn default() -> Self` (impl FitEnvironment < Id , Symm , V >)

### ` fn compute_leaf_environment(node: & V, _towards: & V, tn_a: & TreeTN < Id , Symm , V >, tn_b: & TreeTN < Id , Symm , V >, tn_c: & TreeTN < Id , Symm , V >) -> Result < TensorDynLen < Id , Symm > >`

Compute environment for a leaf node (no children in subtree).

### ` fn compute_single_node_environment(node: & V, towards: & V, tn_a: & TreeTN < Id , Symm , V >, tn_b: & TreeTN < Id , Symm , V >, tn_c: & TreeTN < Id , Symm , V >, child_envs: & [TensorDynLen < Id , Symm >]) -> Result < TensorDynLen < Id , Symm > >`

Compute environment for a single node using child environments. This computes: child_envs × A[node] × B[node] × conj(C[node]) leaving open only the indices connecting to `towards`.

### `pub fn new(tn_a: TreeTN < Id , Symm , V >, tn_b: TreeTN < Id , Symm , V >, max_rank: Option < usize >, rtol: Option < f64 >) -> Self` (impl FitUpdater < Id , Symm , V >)

Create a new FitUpdater.

### `pub fn with_factorize_alg(mut self, alg: FactorizeAlg) -> Self` (impl FitUpdater < Id , Symm , V >)

Set the factorization algorithm.

### ` fn update(&mut self, subtree: TreeTN < Id , Symm , V >, step: & LocalUpdateStep < V >, full_treetn: & TreeTN < Id , Symm , V >) -> Result < TreeTN < Id , Symm , V > >` (impl FitUpdater < Id , Symm , V >)

### ` fn after_step(&mut self, step: & LocalUpdateStep < V >, full_treetn_after: & TreeTN < Id , Symm , V >) -> Result < () >` (impl FitUpdater < Id , Symm , V >)

### ` fn default() -> Self` (impl FitContractionOptions)

### `pub fn new(nsweeps: usize) -> Self` (impl FitContractionOptions)

Create new options with specified number of sweeps.

### `pub fn with_max_rank(mut self, max_rank: usize) -> Self` (impl FitContractionOptions)

Set maximum bond dimension.

### `pub fn with_rtol(mut self, rtol: f64) -> Self` (impl FitContractionOptions)

Set relative tolerance.

### `pub fn with_factorize_alg(mut self, alg: FactorizeAlg) -> Self` (impl FitContractionOptions)

Set factorization algorithm.

### `pub fn with_convergence_tol(mut self, tol: f64) -> Self` (impl FitContractionOptions)

Set convergence tolerance for early termination.

### `pub fn contract_fit(tn_a: & TreeTN < Id , Symm , V >, tn_b: & TreeTN < Id , Symm , V >, center: & V, options: FitContractionOptions) -> Result < TreeTN < Id , Symm , V > >`

Contract two TreeTNs using the fit (variational) algorithm.

### ` fn test_fit_environment_new()`

### ` fn test_fit_environment_insert_get()`

### ` fn test_fit_contraction_options_default()`

### ` fn test_fit_contraction_options_builder()`

## src/treetn/linsolve/environment.rs

### `pub fn neighbors(&self, node: & V) -> Self :: Neighbors < '_ >` (trait NetworkTopology)

Get neighbors of a node.

### `pub fn new() -> Self` (impl EnvironmentCache < Id , Symm , V >)

Create a new empty environment cache.

### `pub fn get(&self, from: & V, to: & V) -> Option < & TensorDynLen < Id , Symm > >` (impl EnvironmentCache < Id , Symm , V >)

Get a cached environment tensor if it exists.

### `pub fn insert(&mut self, from: V, to: V, env: TensorDynLen < Id , Symm >)` (impl EnvironmentCache < Id , Symm , V >)

Insert an environment tensor.

### `pub fn contains(&self, from: & V, to: & V) -> bool` (impl EnvironmentCache < Id , Symm , V >)

Check if environment exists for edge (from, to).

### `pub fn len(&self) -> usize` (impl EnvironmentCache < Id , Symm , V >)

Get the number of cached environments.

### `pub fn is_empty(&self) -> bool` (impl EnvironmentCache < Id , Symm , V >)

Check if the cache is empty.

### `pub fn clear(&mut self)` (impl EnvironmentCache < Id , Symm , V >)

Clear all cached environments.

### `pub fn invalidate(&mut self, region: impl IntoIterator < Item = & 'a V >, topology: & T)` (impl EnvironmentCache < Id , Symm , V >)

Invalidate all caches affected by updates to tensors in region. For each `t ∈ region`: 1. Remove all `env[(t, *)]` (0th generation)

### ` fn invalidate_recursive(&mut self, from: & V, to: & V, topology: & T)` (impl EnvironmentCache < Id , Symm , V >)

Recursively invalidate caches starting from env[(from, to)] towards leaves.

### ` fn default() -> Self` (impl EnvironmentCache < Id , Symm , V >)

### ` fn neighbors(&self, node: & NodeName) -> Self :: Neighbors < '_ >` (impl SiteIndexNetwork < NodeName , Id , Symm , Tags >)

### ` fn test_environment_cache_creation()`

## src/treetn/linsolve/linear_operator.rs

### `pub fn new(mpo: TreeTN < Id , Symm , V >, input_mapping: HashMap < V , IndexMapping < Id , Symm > >, output_mapping: HashMap < V , IndexMapping < Id , Symm > >) -> Self` (impl LinearOperator < Id , Symm , V >)

Create a new LinearOperator from an MPO and index mappings.

### `pub fn from_mpo_and_state(mpo: TreeTN < Id , Symm , V >, state: & TreeTN < Id , Symm , V >) -> Result < Self >` (impl LinearOperator < Id , Symm , V >)

Create a LinearOperator from an MPO and a reference state. This assumes: - The MPO has site indices that we need to map

### `pub fn apply(&self, state: & TreeTN < Id , Symm , V >) -> Result < TreeTN < Id , Symm , V > >` (impl LinearOperator < Id , Symm , V >)

Apply the operator to a state: compute `A|x⟩`. This handles index transformations automatically: 1. Replace state's site indices with MPO's input indices (s_in_tmp)

### `pub fn apply_local(&self, local_tensor: & TensorDynLen < Id , Symm >, region: & [V]) -> Result < TensorDynLen < Id , Symm > >` (impl LinearOperator < Id , Symm , V >)

Apply the operator to a local tensor at a specific region. This is used during the sweep for local updates.

### `pub fn mpo(&self) -> & TreeTN < Id , Symm , V >` (impl LinearOperator < Id , Symm , V >)

Get the internal MPO.

### `pub fn get_input_mapping(&self, node: & V) -> Option < & IndexMapping < Id , Symm > >` (impl LinearOperator < Id , Symm , V >)

Get input mapping for a node.

### `pub fn get_output_mapping(&self, node: & V) -> Option < & IndexMapping < Id , Symm > >` (impl LinearOperator < Id , Symm , V >)

Get output mapping for a node.

### ` fn site_indices(&self) -> HashSet < Index < Id , Symm > >` (impl LinearOperator < Id , Symm , V >)

### ` fn site_index_network(&self) -> & SiteIndexNetwork < V , Id , Symm >` (impl LinearOperator < Id , Symm , V >)

### ` fn node_names(&self) -> HashSet < V >` (impl LinearOperator < Id , Symm , V >)

### `pub fn input_site_indices(&self) -> HashSet < Index < Id , Symm > >` (impl LinearOperator < Id , Symm , V >)

Get all input site indices (true indices from state space).

### `pub fn output_site_indices(&self) -> HashSet < Index < Id , Symm > >` (impl LinearOperator < Id , Symm , V >)

Get all output site indices (true indices from result space).

### `pub fn input_mappings(&self) -> & HashMap < V , IndexMapping < Id , Symm > >` (impl LinearOperator < Id , Symm , V >)

Get all input mappings.

### `pub fn output_mappings(&self) -> & HashMap < V , IndexMapping < Id , Symm > >` (impl LinearOperator < Id , Symm , V >)

Get all output mappings.

## src/treetn/linsolve/local_linop.rs

### `pub fn new(projected_operator: Arc < RwLock < ProjectedOperator < Id , Symm , V > > >, region: Vec < V >, state: TreeTN < Id , Symm , V >, template: TensorDynLen < Id , Symm >, a0: f64, a1: f64) -> Self` (impl LocalLinOp < Id , Symm , V >)

Create a new LocalLinOp for V_in = V_out case.

### `pub fn with_bra_state(projected_operator: Arc < RwLock < ProjectedOperator < Id , Symm , V > > >, region: Vec < V >, state: TreeTN < Id , Symm , V >, bra_state: TreeTN < Id , Symm , V >, template: TensorDynLen < Id , Symm >, a0: f64, a1: f64) -> Self` (impl LocalLinOp < Id , Symm , V >)

Create a new LocalLinOp for V_in ≠ V_out case with explicit bra_state.

### `pub fn with_linear_operator(projected_operator: Arc < RwLock < ProjectedOperator < Id , Symm , V > > >, linear_operator: Arc < LinearOperator < Id , Symm , V > >, region: Vec < V >, state: TreeTN < Id , Symm , V >, bra_state: Option < TreeTN < Id , Symm , V > >, template: TensorDynLen < Id , Symm >, a0: f64, a1: f64) -> Self` (impl LocalLinOp < Id , Symm , V >)

Create a new LocalLinOp with a LinearOperator for index mapping.

### ` fn get_bra_state(&self) -> & TreeTN < Id , Symm , V >` (impl LocalLinOp < Id , Symm , V >)

Get the bra state for environment computation. Returns bra_state if set, otherwise returns state (V_in = V_out case).

### ` fn array_to_tensor(&self, x: & [f64]) -> TensorDynLen < Id , Symm >` (impl LocalLinOp < Id , Symm , V >)

Convert flat array to tensor.

### ` fn tensor_to_array(&self, tensor: & TensorDynLen < Id , Symm >) -> Vec < f64 >` (impl LocalLinOp < Id , Symm , V >)

Convert tensor to flat array.

### ` fn dims(&self) -> (usize , usize)` (impl LocalLinOp < Id , Symm , V >)

### ` fn matvec(&self, x: & [f64], y: & mut [f64])` (impl LocalLinOp < Id , Symm , V >)

### ` fn as_any(&self) -> & dyn Any` (impl LocalLinOp < Id , Symm , V >)

## src/treetn/linsolve/options.rs

### ` fn default() -> Self` (impl LinsolveOptions)

### `pub fn new(nsweeps: usize) -> Self` (impl LinsolveOptions)

Create new options with specified number of sweeps.

### `pub fn with_nsweeps(mut self, nsweeps: usize) -> Self` (impl LinsolveOptions)

Set number of sweeps.

### `pub fn with_truncation(mut self, truncation: TruncationOptions) -> Self` (impl LinsolveOptions)

Set truncation options.

### `pub fn with_max_rank(mut self, max_rank: usize) -> Self` (impl LinsolveOptions)

Set maximum bond dimension.

### `pub fn with_rtol(mut self, rtol: f64) -> Self` (impl LinsolveOptions)

Set relative tolerance for truncation.

### `pub fn with_krylov_tol(mut self, tol: f64) -> Self` (impl LinsolveOptions)

Set GMRES tolerance.

### `pub fn with_krylov_maxiter(mut self, maxiter: usize) -> Self` (impl LinsolveOptions)

Set maximum GMRES iterations.

### `pub fn with_krylov_dim(mut self, dim: usize) -> Self` (impl LinsolveOptions)

Set Krylov subspace dimension.

### `pub fn with_coefficients(mut self, a0: f64, a1: f64) -> Self` (impl LinsolveOptions)

Set coefficients a₀ and a₁.

### `pub fn with_convergence_tol(mut self, tol: f64) -> Self` (impl LinsolveOptions)

Set convergence tolerance for early termination.

### ` fn test_default_options()`

### ` fn test_builder_pattern()`

## src/treetn/linsolve/projected_operator.rs

### `pub fn new(operator: TreeTN < Id , Symm , V >) -> Self` (impl ProjectedOperator < Id , Symm , V >)

Create a new ProjectedOperator.

### `pub fn apply(&mut self, v: & TensorDynLen < Id , Symm >, region: & [V], ket_state: & TreeTN < Id , Symm , V >, bra_state: & TreeTN < Id , Symm , V >, topology: & T) -> Result < TensorDynLen < Id , Symm > >` (impl ProjectedOperator < Id , Symm , V >)

Apply the operator to a local tensor: compute `H|v⟩` at the current position.

### ` fn ensure_environments(&mut self, region: & [V], ket_state: & TreeTN < Id , Symm , V >, bra_state: & TreeTN < Id , Symm , V >, topology: & T) -> Result < () >` (impl ProjectedOperator < Id , Symm , V >)

Ensure environments are computed for neighbors of the region.

### ` fn compute_environment(&mut self, from: & V, to: & V, ket_state: & TreeTN < Id , Symm , V >, bra_state: & TreeTN < Id , Symm , V >, topology: & T) -> Result < TensorDynLen < Id , Symm > >` (impl ProjectedOperator < Id , Symm , V >)

Recursively compute environment for edge (from, to).

### `pub fn local_dimension(&self, region: & [V]) -> usize` (impl ProjectedOperator < Id , Symm , V >)

Compute the local dimension (size of the local Hilbert space).

### `pub fn invalidate(&mut self, region: & [V], topology: & T)` (impl ProjectedOperator < Id , Symm , V >)

Invalidate caches affected by updates to the given region.

## src/treetn/linsolve/projected_state.rs

### `pub fn new(rhs: TreeTN < Id , Symm , V >) -> Self` (impl ProjectedState < Id , Symm , V >)

Create a new ProjectedState.

### `pub fn local_constant_term(&mut self, region: & [V], ket_state: & TreeTN < Id , Symm , V >, topology: & T) -> Result < TensorDynLen < Id , Symm > >` (impl ProjectedState < Id , Symm , V >)

Compute the local constant term `<b|_local` for the given region. This returns the local RHS tensors contracted with environments.

### `pub fn local_constant_term_with_bra(&mut self, region: & [V], _ket_state: & TreeTN < Id , Symm , V >, bra_state: & TreeTN < Id , Symm , V >, topology: & T) -> Result < TensorDynLen < Id , Symm > >` (impl ProjectedState < Id , Symm , V >)

Compute the local constant term `<b|_local` for the given region with explicit bra state. For V_in ≠ V_out case, provides a reference state in V_out for environment computation.

### ` fn ensure_environments(&mut self, region: & [V], bra_state: & TreeTN < Id , Symm , V >, topology: & T) -> Result < () >` (impl ProjectedState < Id , Symm , V >)

Ensure environments are computed for neighbors of the region.

### ` fn compute_environment(&mut self, from: & V, to: & V, bra_state: & TreeTN < Id , Symm , V >, topology: & T) -> Result < TensorDynLen < Id , Symm > >` (impl ProjectedState < Id , Symm , V >)

Recursively compute environment for edge (from, to). Computes `<b|ref_out>` partial contraction at node `from`.

### `pub fn invalidate(&mut self, region: & [V], topology: & T)` (impl ProjectedState < Id , Symm , V >)

Invalidate caches affected by updates to the given region.

## src/treetn/linsolve/updater.rs

### ` fn default() -> Self` (impl LinsolveVerifyReport < V >)

### ` fn fmt(&self, f: & mut std :: fmt :: Formatter < '_ >) -> std :: fmt :: Result` (impl LinsolveVerifyReport < V >)

### `pub fn new(operator: TreeTN < Id , Symm , V >, rhs: TreeTN < Id , Symm , V >, options: LinsolveOptions) -> Self` (impl LinsolveUpdater < Id , Symm , V >)

Create a new LinsolveUpdater for V_in = V_out case.

### `pub fn with_reference_state(operator: TreeTN < Id , Symm , V >, rhs: TreeTN < Id , Symm , V >, reference_state_out: TreeTN < Id , Symm , V >, options: LinsolveOptions) -> Self` (impl LinsolveUpdater < Id , Symm , V >)

Create a new LinsolveUpdater for V_in ≠ V_out case with explicit reference state.

### `pub fn with_linear_operator(linear_operator: LinearOperator < Id , Symm , V >, rhs: TreeTN < Id , Symm , V >, reference_state_out: Option < TreeTN < Id , Symm , V > >, options: LinsolveOptions) -> Self` (impl LinsolveUpdater < Id , Symm , V >)

Create a new LinsolveUpdater with a LinearOperator for correct index handling. The LinearOperator wraps the MPO and handles the mapping between: - True site indices (from state x and b)

### `pub fn get_bra_state(&self, ket_state: & 'a TreeTN < Id , Symm , V >) -> & 'a TreeTN < Id , Symm , V >` (impl LinsolveUpdater < Id , Symm , V >)

Get the bra state for environment computation. Returns reference_state_out if set, otherwise returns the ket_state (V_in = V_out case).

### `pub fn verify(&self, state: & TreeTN < Id , Symm , V >) -> Result < LinsolveVerifyReport < V > >` (impl LinsolveUpdater < Id , Symm , V >)

Verify internal data consistency between operator, RHS, and state. This function checks that: 1. The operator's site space structure is compatible with the state

### ` fn contract_region(&self, subtree: & TreeTN < Id , Symm , V >, region: & [V]) -> Result < TensorDynLen < Id , Symm > >` (impl LinsolveUpdater < Id , Symm , V >)

Contract all tensors in the region into a single local tensor.

### ` fn build_subtree_topology(&self, solved_tensor: & TensorDynLen < Id , Symm >, region: & [V], full_treetn: & TreeTN < Id , Symm , V >) -> Result < TreeTopology < V > >` (impl LinsolveUpdater < Id , Symm , V >)

Build TreeTopology for the subtree region from the solved tensor. Maps each node to the positions of its indices in the solved tensor.

### ` fn copy_decomposed_to_subtree(&self, subtree: & mut TreeTN < Id , Symm , V >, decomposed: & TreeTN < Id , Symm , V >, region: & [V], full_treetn: & TreeTN < Id , Symm , V >) -> Result < () >` (impl LinsolveUpdater < Id , Symm , V >)

Copy decomposed tensors back to subtree, preserving original bond IDs.

### ` fn solve_local(&mut self, region: & [V], init: & TensorDynLen < Id , Symm >, state: & TreeTN < Id , Symm , V >) -> Result < TensorDynLen < Id , Symm > >` (impl LinsolveUpdater < Id , Symm , V >)

Solve the local linear problem using GMRES. Solves: (a₀ + a₁ * H_local) |x_local⟩ = |b_local⟩

### ` fn update(&mut self, subtree: TreeTN < Id , Symm , V >, step: & LocalUpdateStep < V >, full_treetn: & TreeTN < Id , Symm , V >) -> Result < TreeTN < Id , Symm , V > >` (impl LinsolveUpdater < Id , Symm , V >)

### ` fn after_step(&mut self, step: & LocalUpdateStep < V >, full_treetn_after: & TreeTN < Id , Symm , V >) -> Result < () >` (impl LinsolveUpdater < Id , Symm , V >)

## src/treetn/linsolve.rs

### ` fn validate_linsolve_inputs(operator: & TreeTN < Id , Symm , V >, rhs: & TreeTN < Id , Symm , V >, init: & TreeTN < Id , Symm , V >) -> Result < () >`

Validate that operator, rhs, and init have compatible structures for linsolve. Checks: 1. Operator can act on init (same topology)

### `pub fn linsolve(operator: & TreeTN < Id , Symm , V >, rhs: & TreeTN < Id , Symm , V >, init: TreeTN < Id , Symm , V >, center: & V, options: LinsolveOptions) -> Result < LinsolveResult < Id , Symm , V > >`

Solve the linear system `(a₀ + a₁ * H) |x⟩ = |b⟩` for TreeTN.

## src/treetn/localupdate.rs

### `pub fn from_treetn(treetn: & TreeTN < Id , Symm , V >, root: & V, nsite: usize) -> Option < Self >` (impl LocalUpdateSweepPlan < V >)

Generate a sweep plan from a TreeTN's topology. Convenience method that extracts the NodeNameNetwork topology from a TreeTN.

### `pub fn new(network: & NodeNameNetwork < V >, root: & V, nsite: usize) -> Option < Self >` (impl LocalUpdateSweepPlan < V >)

Generate a sweep plan from a NodeNameNetwork. Uses Euler tour traversal to visit all edges in both directions.

### `pub fn empty(nsite: usize) -> Self` (impl LocalUpdateSweepPlan < V >)

Create an empty sweep plan.

### `pub fn is_empty(&self) -> bool` (impl LocalUpdateSweepPlan < V >)

Check if the plan is empty.

### `pub fn len(&self) -> usize` (impl LocalUpdateSweepPlan < V >)

Number of update steps.

### `pub fn iter(&self) -> impl Iterator < Item = & LocalUpdateStep < V > >` (impl LocalUpdateSweepPlan < V >)

Iterate over the steps.

### `pub fn before_step(&mut self, _step: & LocalUpdateStep < V >, _full_treetn_before: & TreeTN < Id , Symm , V >) -> Result < () >` (trait LocalUpdater default)

Optional hook called before performing an update step. This is called with the full TreeTN state *before* the update is applied. Implementors can use it to validate assumptions or prefetch/update caches.

### `pub fn update(&mut self, subtree: TreeTN < Id , Symm , V >, step: & LocalUpdateStep < V >, full_treetn: & TreeTN < Id , Symm , V >) -> Result < TreeTN < Id , Symm , V > >` (trait LocalUpdater)

Update a local subtree.

### `pub fn after_step(&mut self, _step: & LocalUpdateStep < V >, _full_treetn_after: & TreeTN < Id , Symm , V >) -> Result < () >` (trait LocalUpdater default)

Optional hook called after an update step has been applied to the full TreeTN. This is called after: - The updated subtree has been inserted back into the full TreeTN

### `pub fn apply_local_update_sweep(treetn: & mut TreeTN < Id , Symm , V >, plan: & LocalUpdateSweepPlan < V >, updater: & mut U) -> Result < () >`

Apply a local update sweep to a TreeTN. This function orchestrates the sweep by: 1. Iterating through the sweep plan

### `pub fn new(max_rank: Option < usize >, rtol: Option < f64 >) -> Self` (impl TruncateUpdater)

Create a new truncation updater.

### ` fn update(&mut self, subtree: TreeTN < Id , Symm , V >, step: & LocalUpdateStep < V >, _full_treetn: & TreeTN < Id , Symm , V >) -> Result < TreeTN < Id , Symm , V > >` (impl TruncateUpdater)

### `pub fn extract_subtree(&self, node_names: & [V]) -> Result < Self >` (impl TreeTN < Id , Symm , V >)

Extract a sub-tree from this TreeTN. Creates a new TreeTN containing only the specified nodes and their connecting edges. Tensors are cloned into the new TreeTN.

### `pub fn replace_subtree(&mut self, node_names: & [V], replacement: & Self) -> Result < () >` (impl TreeTN < Id , Symm , V >)

Replace a sub-tree with another TreeTN of the same topology. This method replaces the tensors and ortho_towards directions for a subset of nodes with those from another TreeTN. The replacement TreeTN must have

### ` fn create_y_shape_treetn() -> (TreeTN < DynId , NoSymmSpace , String > , Index < DynId > , Index < DynId > , Index < DynId > , Index < DynId > ,)`

Create a 4-node Y-shape TreeTN: A |

### ` fn test_extract_subtree_single_node()`

### ` fn test_extract_subtree_two_nodes()`

### ` fn test_extract_subtree_disconnected_fails()`

### ` fn test_extract_subtree_preserves_consistency()`

### ` fn test_replace_subtree_same_appearance()`

### ` fn test_replace_subtree_two_nodes()`

### ` fn create_chain_network() -> NodeNameNetwork < String >`

Create a chain network: A - B - C

### ` fn create_y_network() -> NodeNameNetwork < String >`

Create a Y-shape network: A |

### ` fn test_sweep_plan_nsite1_chain()`

### ` fn test_sweep_plan_nsite2_chain()`

### ` fn test_sweep_plan_nsite1_y_shape()`

### ` fn test_sweep_plan_nsite2_y_shape()`

### ` fn test_sweep_plan_single_node()`

### ` fn test_sweep_plan_invalid_nsite()`

### ` fn test_sweep_plan_nonexistent_root()`

### ` fn create_chain_treetn() -> TreeTN < DynId , NoSymmSpace , String >`

Create a chain TreeTN: A - B - C Each node has a site index of dim 2, bonds of dim 4

### ` fn test_truncate_updater_basic()`

### ` fn test_apply_local_update_sweep_preserves_structure()`

### ` fn test_apply_local_update_sweep_requires_canonicalization()`

### ` fn test_sweep_plan_from_treetn()`

## src/treetn/mod.rs

### `pub fn new() -> Self` (impl TreeTN < Id , Symm , V >)

Create a new empty TreeTN. Use `add_tensor()` to add tensors and `connect()` to establish bonds manually.

### `pub fn from_tensors(tensors: Vec < TensorDynLen < Id , Symm > >, node_names: Vec < V >) -> Result < Self >` (impl TreeTN < Id , Symm , V >)

Create a TreeTN from a list of tensors and node names using einsum rule. This function connects tensors that share common indices (by ID). The algorithm is O(n) where n is the number of tensors:

### `pub fn add_tensor(&mut self, node_name: V, tensor: TensorDynLen < Id , Symm >) -> Result < NodeIndex >` (impl TreeTN < Id , Symm , V >)

Add a tensor to the network with a node name. Returns the NodeIndex for the newly added tensor. Also updates the site_index_network with the physical indices (all indices initially,

### `pub fn add_tensor_auto_name(&mut self, tensor: TensorDynLen < Id , Symm >) -> NodeIndex` (impl TreeTN < Id , Symm , V >)

Add a tensor to the network using NodeIndex as the node name. This method only works when `V = NodeIndex`. Returns the NodeIndex for the newly added tensor.

### `pub fn connect(&mut self, node_a: NodeIndex, index_a: & Index < Id , Symm >, node_b: NodeIndex, index_b: & Index < Id , Symm >) -> Result < EdgeIndex >` (impl TreeTN < Id , Symm , V >)

Connect two tensors via a specified pair of indices. The indices must have the same ID (Einsum mode).

### `pub(crate) fn add_tensor_internal(&mut self, node_name: V, tensor: TensorDynLen < Id , Symm >) -> Result < NodeIndex >` (impl TreeTN < Id , Symm , V >)

Internal method to add a tensor with a node name.

### `pub(crate) fn connect_internal(&mut self, node_a: NodeIndex, index_a: & Index < Id , Symm >, node_b: NodeIndex, index_b: & Index < Id , Symm >) -> Result < EdgeIndex >` (impl TreeTN < Id , Symm , V >)

Internal method to connect two tensors. In Einsum mode, `index_a` and `index_b` must have the same ID.

### `pub(crate) fn prepare_sweep_to_center(&mut self, canonical_center: impl IntoIterator < Item = V >, context_name: & str) -> Result < Option < SweepContext > >` (impl TreeTN < Id , Symm , V >)

Prepare context for sweep-to-center operations. This method: 1. Validates tree structure

### `pub(crate) fn sweep_edge(&mut self, src: NodeIndex, dst: NodeIndex, factorize_options: & FactorizeOptions, context_name: & str) -> Result < () >` (impl TreeTN < Id , Symm , V >)

Process one edge during a sweep operation. Factorizes the tensor at `src` node, absorbs the right factor into `dst` (parent), and updates the edge bond and ortho_towards.

### `pub fn tensor(&self, node: NodeIndex) -> Option < & TensorDynLen < Id , Symm > >` (impl TreeTN < Id , Symm , V >)

Get a reference to a tensor by NodeIndex.

### `pub fn tensor_mut(&mut self, node: NodeIndex) -> Option < & mut TensorDynLen < Id , Symm > >` (impl TreeTN < Id , Symm , V >)

Get a mutable reference to a tensor by NodeIndex.

### `pub fn replace_tensor(&mut self, node: NodeIndex, new_tensor: TensorDynLen < Id , Symm >) -> Result < Option < TensorDynLen < Id , Symm > > >` (impl TreeTN < Id , Symm , V >)

Replace a tensor at the given node with a new tensor. Validates that the new tensor contains all indices used in connections to this node. Returns an error if any connection index is missing.

### `pub fn bond_index(&self, edge: EdgeIndex) -> Option < & Index < Id , Symm > >` (impl TreeTN < Id , Symm , V >)

Get the bond index for a given edge.

### `pub fn bond_index_mut(&mut self, edge: EdgeIndex) -> Option < & mut Index < Id , Symm > >` (impl TreeTN < Id , Symm , V >)

Get a mutable reference to the bond index for a given edge.

### `pub fn edges_for_node(&self, node: NodeIndex) -> Vec < (EdgeIndex , NodeIndex) >` (impl TreeTN < Id , Symm , V >)

Get all edges connected to a node.

### `pub fn replace_edge_bond(&mut self, edge: EdgeIndex, new_bond_index: Index < Id , Symm >) -> Result < () >` (impl TreeTN < Id , Symm , V >)

Replace the bond index for an edge (e.g., after SVD creates a new bond index). Also updates site_index_network: the old bond index becomes physical again, and the new bond index is removed from physical indices.

### `pub fn set_ortho_towards(&mut self, index_id: & Id, dir: Option < V >)` (impl TreeTN < Id , Symm , V >)

Set the orthogonalization direction for an index (bond or site). The direction is specified as a node name (or None to clear).

### `pub fn ortho_towards_for_index(&self, index_id: & Id) -> Option < & V >` (impl TreeTN < Id , Symm , V >)

Get the node name that the orthogonalization points towards for an index. Returns None if ortho_towards is not set for this index.

### `pub fn set_edge_ortho_towards(&mut self, edge: petgraph :: stable_graph :: EdgeIndex, dir: Option < V >) -> Result < () >` (impl TreeTN < Id , Symm , V >)

Set the orthogonalization direction for an edge (by EdgeIndex). This is a convenience method that looks up the bond index ID and calls `set_ortho_towards`. The direction is specified as a node name (or None to clear).

### `pub fn ortho_towards_node(&self, edge: petgraph :: stable_graph :: EdgeIndex) -> Option < & V >` (impl TreeTN < Id , Symm , V >)

Get the node name that the orthogonalization points towards for an edge. Returns None if ortho_towards is not set for this edge's bond index.

### `pub fn ortho_towards_node_index(&self, edge: petgraph :: stable_graph :: EdgeIndex) -> Option < NodeIndex >` (impl TreeTN < Id , Symm , V >)

Get the NodeIndex that the orthogonalization points towards for an edge. Returns None if ortho_towards is not set for this edge's bond index.

### `pub fn validate_tree(&self) -> Result < () >` (impl TreeTN < Id , Symm , V >)

Validate that the graph is a tree (or forest). Checks: - The graph is connected (all nodes reachable from the first node)

### `pub fn node_count(&self) -> usize` (impl TreeTN < Id , Symm , V >)

Get the number of nodes in the network.

### `pub fn edge_count(&self) -> usize` (impl TreeTN < Id , Symm , V >)

Get the number of edges in the network.

### `pub fn node_index(&self, node_name: & V) -> Option < NodeIndex >` (impl TreeTN < Id , Symm , V >)

Get the NodeIndex for a node by name.

### `pub fn edge_between(&self, node_a: & V, node_b: & V) -> Option < EdgeIndex >` (impl TreeTN < Id , Symm , V >)

Get the EdgeIndex for the edge between two nodes by name. Returns `None` if either node doesn't exist or there's no edge between them.

### `pub fn node_indices(&self) -> Vec < NodeIndex >` (impl TreeTN < Id , Symm , V >)

Get all node indices in the tree tensor network.

### `pub fn node_names(&self) -> Vec < V >` (impl TreeTN < Id , Symm , V >)

Get all node names in the tree tensor network.

### `pub fn edges_to_canonicalize_by_names(&self, target: & V) -> Option < Vec < (V , V) > >` (impl TreeTN < Id , Symm , V >)

Compute edges to canonicalize from leaves to target, returning node names. Returns `(from, to)` pairs in the order they should be processed: - `from` is the node being factorized

### `pub fn canonical_center(&self) -> & HashSet < V >` (impl TreeTN < Id , Symm , V >)

Get a reference to the orthogonalization region (using node names). When empty, the network is not canonicalized.

### `pub fn is_canonicalized(&self) -> bool` (impl TreeTN < Id , Symm , V >)

Check if the network is canonicalized. Returns `true` if `canonical_center` is non-empty, `false` otherwise.

### `pub fn set_canonical_center(&mut self, region: impl IntoIterator < Item = V >) -> Result < () >` (impl TreeTN < Id , Symm , V >)

Set the orthogonalization region (using node names). Validates that all specified nodes exist in the graph.

### `pub fn clear_canonical_center(&mut self)` (impl TreeTN < Id , Symm , V >)

Clear the orthogonalization region (mark network as not canonicalized). Also clears the canonical form.

### `pub fn canonical_form(&self) -> Option < CanonicalForm >` (impl TreeTN < Id , Symm , V >)

Get the current canonical form. Returns `None` if not canonicalized.

### `pub fn add_to_canonical_center(&mut self, node_name: V) -> Result < () >` (impl TreeTN < Id , Symm , V >)

Add a node to the orthogonalization region. Validates that the node exists in the graph.

### `pub fn remove_from_canonical_center(&mut self, node_name: & V) -> bool` (impl TreeTN < Id , Symm , V >)

Remove a node from the orthogonalization region. Returns `true` if the node was in the region, `false` otherwise.

### `pub fn site_index_network(&self) -> & SiteIndexNetwork < V , Id , Symm >` (impl TreeTN < Id , Symm , V >)

Get a reference to the site index network. The site index network contains both topology (graph structure) and site space (physical indices).

### `pub fn site_index_network_mut(&mut self) -> & mut SiteIndexNetwork < V , Id , Symm >` (impl TreeTN < Id , Symm , V >)

Get a mutable reference to the site index network.

### `pub fn site_space(&self, node_name: & V) -> Option < & std :: collections :: HashSet < Index < Id , Symm > > >` (impl TreeTN < Id , Symm , V >)

Get a reference to the site space (physical indices) for a node.

### `pub fn site_space_mut(&mut self, node_name: & V) -> Option < & mut std :: collections :: HashSet < Index < Id , Symm > > >` (impl TreeTN < Id , Symm , V >)

Get a mutable reference to the site space (physical indices) for a node.

### `pub fn share_equivalent_site_index_network(&self, other: & Self) -> bool` (impl TreeTN < Id , Symm , V >)

Check if two TreeTNs share equivalent site index network structure. Two TreeTNs share equivalent structure if: - Same topology (nodes and edges)

### `pub fn same_topology(&self, other: & Self) -> bool` (impl TreeTN < Id , Symm , V >)

Check if two TreeTNs have the same topology (graph structure). This only checks that both networks have the same nodes and edges, not that they have the same site indices.

### `pub fn same_appearance(&self, other: & Self) -> bool` (impl TreeTN < Id , Symm , V >)

Check if two TreeTNs have the same "appearance". Two TreeTNs have the same appearance if: 1. They have the same topology (same nodes and edges)

### `pub fn verify_internal_consistency(&self) -> Result < () >` (impl TreeTN < Id , Symm , V >)

Verify internal data consistency by reconstructing the TreeTN from scratch. This function clones all tensors and node names, reconstructs a new TreeTN using `from_tensors`, and verifies that the reconstruction matches the original.

### `pub(crate) fn common_inds(inds_a: & [Index < Id , Symm >], inds_b: & [Index < Id , Symm >]) -> Vec < Index < Id , Symm > >`

Find common indices between two slices of indices.

### `pub(crate) fn compute_strides(dims: & [usize]) -> Vec < usize >`

Compute strides for row-major (C-order) indexing.

### `pub(crate) fn linear_to_multi_index(linear: usize, strides: & [usize], rank: usize) -> Vec < usize >`

Convert linear index to multi-index.

### `pub(crate) fn multi_to_linear_index(multi: & [usize], strides: & [usize]) -> usize`

Convert multi-index to linear index.

## src/treetn/operator_impl.rs

### ` fn site_indices(&self) -> HashSet < Index < Id , Symm > >` (impl TreeTN < Id , Symm , V >)

### ` fn site_index_network(&self) -> & SiteIndexNetwork < V , Id , Symm >` (impl TreeTN < Id , Symm , V >)

### ` fn node_names(&self) -> HashSet < V >` (impl TreeTN < Id , Symm , V >)

### ` fn make_index(dim: usize) -> DynIndex`

### ` fn create_chain_site_network(n: usize) -> SiteIndexNetwork < String , DynId >`

### ` fn test_treetn_operator_trait_site_indices()`

### ` fn test_treetn_operator_trait_node_names()`

### ` fn test_treetn_operator_trait_site_index_network()`

## src/treetn/ops.rs

### ` fn default() -> Self` (impl TreeTN < Id , Symm , V >)

### ` fn clone(&self) -> Self` (impl TreeTN < Id , Symm , V >)

### ` fn fmt(&self, f: & mut std :: fmt :: Formatter < '_ >) -> std :: fmt :: Result` (impl TreeTN < Id , Symm , V >)

### ` fn mul(mut self, a: f64) -> Self :: Output` (impl TreeTN < Id , Symm , V >)

Multiply the TreeTN by a scalar with distributed scaling.

### ` fn mul(mut self, a: Complex64) -> Self :: Output` (impl TreeTN < Id , Symm , V >)

Multiply the TreeTN by a complex scalar with distributed scaling.

### ` fn mul(self, a: f64) -> Self :: Output` (impl & TreeTN < Id , Symm , V >)

### ` fn mul(self, a: Complex64) -> Self :: Output` (impl & TreeTN < Id , Symm , V >)

### `pub fn log_norm(&mut self) -> Result < f64 >` (impl TreeTN < Id , Symm , V >)

Compute the natural logarithm of the Frobenius norm: ln(||TN||). **Warning**: This method may canonicalize the network if not already canonicalized to a single Unitary center. Use `log_norm` (without canonicalization) if you

## src/treetn/tensor_like.rs

### ` fn external_indices(&self) -> Vec < Index < Self :: Id , Self :: Symm , Self :: Tags > >` (impl TreeTN < Id , Symm , V >)

### ` fn num_external_indices(&self) -> usize` (impl TreeTN < Id , Symm , V >)

### ` fn to_tensor(&self) -> anyhow :: Result < tensor4all_core :: TensorDynLen < Self :: Id , Self :: Symm > >` (impl TreeTN < Id , Symm , V >)

### ` fn as_any(&self) -> & dyn std :: any :: Any` (impl TreeTN < Id , Symm , V >)

## src/treetn/transform.rs

### `pub fn fuse_to(&self, target: & SiteIndexNetwork < TargetV , Id , Symm >) -> Result < TreeTN < Id , Symm , TargetV > >` (impl TreeTN < Id , Symm , V >)

Fuse (merge) adjacent nodes to match the target structure. This operation contracts adjacent nodes that should be merged according to the target `SiteIndexNetwork`. The target structure must be a "coarsening"

### ` fn contract_node_group(&self, nodes: & HashSet < V >) -> Result < TensorDynLen < Id , Symm > >` (impl TreeTN < Id , Symm , V >)

Contract a group of nodes into a single tensor. The nodes must form a connected subtree in the current TreeTN. Contracts all internal bonds (bonds between nodes in the group),

### `pub fn split_to(&self, target: & SiteIndexNetwork < TargetV , Id , Symm >, options: & SplitOptions) -> Result < TreeTN < Id , Symm , TargetV > >` (impl TreeTN < Id , Symm , V >)

Split nodes to match the target structure. This operation splits nodes that contain site indices belonging to multiple target nodes. The target structure must be a "refinement" of the current

### ` fn split_tensor_for_targets(&self, tensor: & TensorDynLen < Id , Symm >, site_to_target: & HashMap < Id , TargetV >) -> Result < Vec < (TargetV , TensorDynLen < Id , Symm >) > >` (impl TreeTN < Id , Symm , V >)

Split a tensor into multiple tensors, one for each target node. This uses QR factorization to iteratively separate site indices belonging to different target nodes.

### ` fn test_fuse_to_identity_generic(site_network: & SiteIndexNetwork < String , DynId >, link_space: LinkSpace < String >, seed: u64, rtol: f64)`

Test fuse_to with identity transformation (same structure). The result should equal the original tensor.

### ` fn test_fuse_to_all_into_one_generic(site_network: & SiteIndexNetwork < String , DynId >, link_space: LinkSpace < String >, seed: u64, rtol: f64)`

Test fuse_to that merges all nodes into one. The result should be a single tensor equal to full contraction.

### ` fn test_fuse_to_pairwise_generic(site_network: & SiteIndexNetwork < String , DynId >, target_network: & SiteIndexNetwork < String , DynId >, link_space: LinkSpace < String >, seed: u64, rtol: f64)`

Test fuse_to that merges adjacent pairs. For a chain A-B-C-D, merge to AB-CD.

### ` fn create_4node_chain() -> SiteIndexNetwork < String , DynId >`

Create a 4-node chain: A -- B -- C -- D

### ` fn create_star() -> SiteIndexNetwork < String , DynId >`

Create a star topology: B is center, connected to A, C, D A |

### ` fn create_y_shape() -> SiteIndexNetwork < String , DynId >`

Create a Y-shape topology: A -- B -- C, B -- D A |

### ` fn test_fuse_to_identity_4node_chain()`

### ` fn test_fuse_to_all_into_one_4node_chain()`

### ` fn test_fuse_to_pairwise_4node_chain()`

### ` fn test_fuse_to_identity_star()`

### ` fn test_fuse_to_all_into_one_star()`

### ` fn test_fuse_to_star_merge_leaves()`

### ` fn test_fuse_to_identity_y_shape()`

### ` fn test_fuse_to_all_into_one_y_shape()`

### ` fn test_fuse_to_incompatible_error()`

### ` fn test_split_to_identity()`

Test split_to with identity transformation (same structure). The result should equal the original tensor.

### ` fn test_split_to_fused_to_chain()`

Test split_to: split a 2-node fused chain into 4 nodes. Before: AB -- CD (2 nodes, each with 2 site indices) After: A -- B -- C -- D (4 nodes)

### ` fn test_fuse_then_split_roundtrip()`

Test fuse_to followed by split_to (roundtrip). A -- B -- C -- D -> AB -- CD -> A -- B -- C -- D

### ` fn test_split_to_with_truncation()`

Test split_to with final_sweep option.

## src/treetn/truncate.rs

### `pub fn truncate(mut self, canonical_center: impl IntoIterator < Item = V >, options: TruncationOptions) -> Result < Self >` (impl TreeTN < Id , Symm , V >)

Truncate the network towards the specified center using options. This is the recommended unified API for truncation. It accepts: - Center nodes specified by their node names (V)

### `pub fn truncate_mut(&mut self, canonical_center: impl IntoIterator < Item = V >, options: TruncationOptions) -> Result < () >` (impl TreeTN < Id , Symm , V >)

Truncate the network in-place towards the specified center using options. This is the `&mut self` version of [`truncate`].

### `pub(crate) fn truncate_impl(&mut self, canonical_center: impl IntoIterator < Item = V >, form: CanonicalForm, rtol: Option < f64 >, max_rank: Option < usize >, context_name: & str) -> Result < () >` (impl TreeTN < Id , Symm , V >)

Internal implementation for truncation. Uses LocalUpdateSweepPlan with TruncateUpdater for full two-site sweeps.

