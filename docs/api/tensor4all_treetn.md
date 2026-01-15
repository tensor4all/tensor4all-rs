# tensor4all-treetn

## src/algorithm.rs

### `pub fn from_i32(value: i32) -> Option < Self >` (impl ContractionAlgorithm)

Create from C API integer representation. Returns `None` for invalid values.

### `pub fn to_i32(self) -> i32` (impl ContractionAlgorithm)

Convert to C API integer representation.

### `pub fn name(&self) -> & 'static str` (impl ContractionAlgorithm)

Get algorithm name as string.

### `pub fn from_i32(value: i32) -> Option < Self >` (impl CanonicalForm)

Create from C API integer representation. Returns `None` for invalid values.

### `pub fn to_i32(self) -> i32` (impl CanonicalForm)

Convert to C API integer representation.

### `pub fn name(&self) -> & 'static str` (impl CanonicalForm)

Get form name as string.

### `pub fn from_i32(value: i32) -> Option < Self >` (impl CompressionAlgorithm)

Create from C API integer representation. Returns `None` for invalid values.

### `pub fn to_i32(self) -> i32` (impl CompressionAlgorithm)

Convert to C API integer representation.

### `pub fn name(&self) -> & 'static str` (impl CompressionAlgorithm)

Get algorithm name as string.

### ` fn test_contraction_algorithm_roundtrip()`

### ` fn test_compression_algorithm_roundtrip()`

### ` fn test_canonical_form_roundtrip()`

### ` fn test_invalid_values()`

### ` fn test_default()`

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

## src/link_index_network.rs

### ` fn default() -> Self` (impl LinkIndexNetwork < I >)

### `pub fn new() -> Self` (impl LinkIndexNetwork < I >)

Create a new empty LinkIndexNetwork.

### `pub fn with_capacity(edges: usize) -> Self` (impl LinkIndexNetwork < I >)

Create with initial capacity.

### `pub fn insert(&mut self, edge: EdgeIndex, index: & I)` (impl LinkIndexNetwork < I >)

Register a link index for an edge.

### `pub fn remove(&mut self, index: & I) -> Option < EdgeIndex >` (impl LinkIndexNetwork < I >)

Remove a link index registration.

### `pub fn find_edge(&self, index: & I) -> Option < EdgeIndex >` (impl LinkIndexNetwork < I >)

Find the edge containing a given index.

### `pub fn find_edge_by_id(&self, id: & I :: Id) -> Option < EdgeIndex >` (impl LinkIndexNetwork < I >)

Find the edge containing an index by ID.

### `pub fn contains(&self, index: & I) -> bool` (impl LinkIndexNetwork < I >)

Check if an index is registered.

### `pub fn contains_id(&self, id: & I :: Id) -> bool` (impl LinkIndexNetwork < I >)

Check if an index ID is registered.

### `pub fn replace_index(&mut self, old_index: & I, new_index: & I, edge: EdgeIndex) -> Result < () , String >` (impl LinkIndexNetwork < I >)

Update the index for an edge (e.g., after SVD creates new bond index).

### `pub fn len(&self) -> usize` (impl LinkIndexNetwork < I >)

Number of registered link indices.

### `pub fn is_empty(&self) -> bool` (impl LinkIndexNetwork < I >)

Check if empty.

### `pub fn clear(&mut self)` (impl LinkIndexNetwork < I >)

Clear all registrations.

### `pub fn iter(&self) -> impl Iterator < Item = (& I :: Id , & EdgeIndex) >` (impl LinkIndexNetwork < I >)

Iterate over all (index_id, edge) pairs.

### ` fn test_basic_operations()`

### ` fn test_replace_index()`

### ` fn test_remove()`

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

### `pub fn are_exclusive_operators(target: & SiteIndexNetwork < V , T :: Index >, operators: & [& O]) -> bool`

Check if a set of operators are exclusive (non-overlapping) on the target network. Operators are exclusive if: 1. **Vertex-disjoint**: No two operators share a node

### ` fn check_path_exclusive(target: & SiteIndexNetwork < V , T :: Index >, set_a: & HashSet < V >, set_b: & HashSet < V >, all_sets: & [HashSet < V >]) -> bool`

Check if paths between two operator regions don't cross other operators.

### `pub fn compose_exclusive_linear_operators(target: & SiteIndexNetwork < V , T :: Index >, operators: & [& LinearOperator < T , V >], gap_site_indices: & HashMap < V , Vec < (T :: Index , T :: Index) > >) -> Result < LinearOperator < T , V > >`

Compose exclusive LinearOperators into a single LinearOperator. This function takes multiple non-overlapping operators and combines them into a single operator that acts on the full target space. Gap positions (nodes not

### `pub fn compose_exclusive_operators(_target: & SiteIndexNetwork < V , T :: Index >, _operators: & [& O], _gap_site_indices: & HashMap < V , Vec < (T :: Index , T :: Index) > >) -> Result < LinearOperator < T , V > >`

Compose exclusive operators into a single operator (convenience wrapper). This is a generic version that accepts any type implementing the Operator trait. For actual composition, use [`compose_exclusive_linear_operators`] with LinearOperator inputs.

### ` fn make_index(dim: usize) -> DynIndex`

### ` fn create_chain_site_network(n: usize) -> SiteIndexNetwork < String , DynIndex >`

### ` fn create_linear_operator_from_treetn(mpo: TreeTN < TensorDynLen , String >, input_indices: & [(String , DynIndex , DynIndex)], output_indices: & [(String , DynIndex , DynIndex)]) -> LinearOperator < TensorDynLen , String >`

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

### `pub fn build_identity_operator_tensor(site_indices: & [DynIndex], output_site_indices: & [DynIndex]) -> Result < TensorDynLen >`

Build an identity operator tensor for a gap node. For a node with site indices `{s1, s2, ...}` and bond indices `{l1, l2, ...}`, this creates an identity tensor where:

### `pub fn build_identity_operator_tensor_c64(site_indices: & [DynIndex], output_site_indices: & [DynIndex]) -> Result < TensorDynLen >`

Build an identity operator tensor with complex data type. Same as [`build_identity_operator_tensor`] but returns a complex tensor.

### ` fn make_index(dim: usize) -> DynIndex`

### ` fn get_f64_data(tensor: & TensorDynLen) -> & [f64]`

### ` fn get_c64_data(tensor: & TensorDynLen) -> & [Complex64]`

### ` fn test_identity_single_site()`

### ` fn test_identity_two_sites()`

### ` fn test_identity_dimension_mismatch()`

### ` fn test_identity_empty()`

### ` fn test_identity_c64()`

## src/operator/mod.rs

### `pub fn site_indices(&self) -> HashSet < T :: Index >` (trait Operator)

Get all site indices this operator acts on. Returns the union of site indices across all nodes.

### `pub fn site_index_network(&self) -> & SiteIndexNetwork < V , T :: Index >` (trait Operator)

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

### ` fn truncation_params(&self) -> & TruncationParams` (impl TruncationOptions)

### ` fn truncation_params_mut(&mut self) -> & mut TruncationParams` (impl TruncationOptions)

### `pub fn new() -> Self` (impl TruncationOptions)

Create options with default settings (no truncation limits).

### `pub fn with_max_rank(mut self, rank: usize) -> Self` (impl TruncationOptions)

Create options with a maximum rank.

### `pub fn with_rtol(mut self, rtol: f64) -> Self` (impl TruncationOptions)

Create options with a relative tolerance.

### `pub fn with_form(mut self, form: CanonicalForm) -> Self` (impl TruncationOptions)

Set the canonical form / algorithm.

### `pub fn rtol(&self) -> Option < f64 >` (impl TruncationOptions)

Get rtol (for backwards compatibility).

### `pub fn max_rank(&self) -> Option < usize >` (impl TruncationOptions)

Get max_rank (for backwards compatibility).

### ` fn default() -> Self` (impl SplitOptions)

### ` fn truncation_params(&self) -> & TruncationParams` (impl SplitOptions)

### ` fn truncation_params_mut(&mut self) -> & mut TruncationParams` (impl SplitOptions)

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

### `pub fn rtol(&self) -> Option < f64 >` (impl SplitOptions)

Get rtol (for backwards compatibility).

### `pub fn max_rank(&self) -> Option < usize >` (impl SplitOptions)

Get max_rank (for backwards compatibility).

### ` fn test_canonicalization_options_default()`

### ` fn test_canonicalization_options_new()`

### ` fn test_canonicalization_options_forced()`

### ` fn test_canonicalization_options_builder()`

### ` fn test_truncation_options_default()`

### ` fn test_truncation_options_new()`

### ` fn test_truncation_options_builder()`

### ` fn test_truncation_options_has_truncation_params()`

### ` fn test_split_options_default()`

### ` fn test_split_options_new()`

### ` fn test_split_options_builder()`

### ` fn test_split_options_has_truncation_params()`

## src/random.rs

### `pub fn uniform(dim: usize) -> Self` (impl LinkSpace < V >)

Create a uniform link space where all bonds have the same dimension.

### `pub fn per_edge(dims: HashMap < (V , V) , usize >) -> Self` (impl LinkSpace < V >)

Create a per-edge link space from a map of edge dimensions.

### `pub fn get(&self, a: & V, b: & V) -> Option < usize >` (impl LinkSpace < V >)

Get the dimension for an edge between two nodes. For `PerEdge`, the key is normalized to `(min(a, b), max(a, b))`.

### `pub fn random_treetn_f64(rng: & mut R, site_network: & SiteIndexNetwork < V , DefaultIndex >, link_space: LinkSpace < V >) -> TreeTN < TensorDynLen , V >`

Create a random f64 TreeTN from a site index network. Generates random tensors at each node with: - Site indices from the `site_network`

### `pub fn random_treetn_c64(rng: & mut R, site_network: & SiteIndexNetwork < V , DefaultIndex >, link_space: LinkSpace < V >) -> TreeTN < TensorDynLen , V >`

Create a random Complex64 TreeTN from a site index network. Similar to [`random_treetn_f64`], but generates complex-valued tensors where both real and imaginary parts are drawn from standard normal distribution.

### ` fn random_treetn_impl(rng: & mut R, site_network: & SiteIndexNetwork < V , DefaultIndex >, link_space: LinkSpace < V >, is_complex: bool) -> TreeTN < TensorDynLen , V >`

Internal implementation for creating random TreeTN.

### ` fn test_random_treetn_f64_two_nodes()`

### ` fn test_random_treetn_c64_chain()`

### ` fn test_link_space_per_edge()`

## src/site_index_network.rs

### `pub fn new() -> Self` (impl SiteIndexNetwork < NodeName , I >)

Create a new empty SiteIndexNetwork.

### `pub fn with_capacity(nodes: usize, edges: usize) -> Self` (impl SiteIndexNetwork < NodeName , I >)

Create a new SiteIndexNetwork with initial capacity.

### `pub fn add_node(&mut self, node_name: NodeName, site_space: impl Into < HashSet < I > >) -> Result < NodeIndex , String >` (impl SiteIndexNetwork < NodeName , I >)

Add a node with site space (physical indices).

### `pub fn has_node(&self, node_name: & NodeName) -> bool` (impl SiteIndexNetwork < NodeName , I >)

Check if a node exists.

### `pub fn site_space(&self, node_name: & NodeName) -> Option < & HashSet < I > >` (impl SiteIndexNetwork < NodeName , I >)

Get the site space (physical indices) for a node.

### `pub fn site_space_mut(&mut self, node_name: & NodeName) -> Option < & mut HashSet < I > >` (impl SiteIndexNetwork < NodeName , I >)

Get a mutable reference to the site space for a node. **Warning**: Direct modification of site space via this method does NOT update the reverse lookup (`index_to_node`). Use `add_site_index()`,

### `pub fn find_node_by_index(&self, index: & I) -> Option < & NodeName >` (impl SiteIndexNetwork < NodeName , I >)

Find the node containing a given site index.

### `pub fn find_node_by_index_id(&self, id: & I :: Id) -> Option < & NodeName >` (impl SiteIndexNetwork < NodeName , I >)

Find the node containing an index by ID.

### `pub fn contains_index(&self, index: & I) -> bool` (impl SiteIndexNetwork < NodeName , I >)

Check if a site index is registered.

### `pub fn add_site_index(&mut self, node_name: & NodeName, index: I) -> Result < () , String >` (impl SiteIndexNetwork < NodeName , I >)

Add a site index to a node's site space. Updates both the site space and the reverse lookup.

### `pub fn remove_site_index(&mut self, node_name: & NodeName, index: & I) -> Result < bool , String >` (impl SiteIndexNetwork < NodeName , I >)

Remove a site index from a node's site space. Updates both the site space and the reverse lookup.

### `pub fn replace_site_index(&mut self, node_name: & NodeName, old_index: & I, new_index: I) -> Result < () , String >` (impl SiteIndexNetwork < NodeName , I >)

Replace a site index in a node's site space. Updates both the site space and the reverse lookup.

### `pub fn set_site_space(&mut self, node_name: & NodeName, new_indices: HashSet < I >) -> Result < () , String >` (impl SiteIndexNetwork < NodeName , I >)

Replace all site indices for a node with a new set. Updates both the site space and the reverse lookup. This is an atomic operation that removes all old indices and adds all new ones.

### `pub fn site_space_by_index(&self, node: NodeIndex) -> Option < & HashSet < I > >` (impl SiteIndexNetwork < NodeName , I >)

Get the site space by NodeIndex.

### `pub fn add_edge(&mut self, n1: & NodeName, n2: & NodeName) -> Result < EdgeIndex , String >` (impl SiteIndexNetwork < NodeName , I >)

Add an edge between two nodes. Returns an error if either node doesn't exist.

### `pub fn node_index(&self, node_name: & NodeName) -> Option < NodeIndex >` (impl SiteIndexNetwork < NodeName , I >)

Get the NodeIndex for a node name.

### `pub fn node_name(&self, node: NodeIndex) -> Option < & NodeName >` (impl SiteIndexNetwork < NodeName , I >)

Get the node name for a NodeIndex.

### `pub fn node_names(&self) -> Vec < & NodeName >` (impl SiteIndexNetwork < NodeName , I >)

Get all node names.

### `pub fn node_count(&self) -> usize` (impl SiteIndexNetwork < NodeName , I >)

Get the number of nodes.

### `pub fn edge_count(&self) -> usize` (impl SiteIndexNetwork < NodeName , I >)

Get the number of edges.

### `pub fn topology(&self) -> & NodeNameNetwork < NodeName >` (impl SiteIndexNetwork < NodeName , I >)

Get a reference to the underlying topology (NodeNameNetwork).

### `pub fn edges(&self) -> impl Iterator < Item = (NodeName , NodeName) > + '_` (impl SiteIndexNetwork < NodeName , I >)

Get all edges as pairs of node names. Returns an iterator of `(NodeName, NodeName)` pairs.

### `pub fn neighbors(&self, node_name: & NodeName) -> impl Iterator < Item = NodeName > + '_` (impl SiteIndexNetwork < NodeName , I >)

Get all neighbors of a node. Returns an iterator of neighbor node names.

### `pub fn graph(&self) -> & StableGraph < () , () , Undirected >` (impl SiteIndexNetwork < NodeName , I >)

Get a reference to the internal graph.

### `pub fn graph_mut(&mut self) -> & mut StableGraph < () , () , Undirected >` (impl SiteIndexNetwork < NodeName , I >)

Get a mutable reference to the internal graph. **Warning**: Directly modifying the internal graph can break consistency.

### `pub fn share_equivalent_site_index_network(&self, other: & Self) -> bool` (impl SiteIndexNetwork < NodeName , I >)

Check if two SiteIndexNetworks share equivalent site index structure. Two networks are equivalent if: - Same topology (nodes and edges)

### `pub fn post_order_dfs(&self, root: & NodeName) -> Option < Vec < NodeName > >` (impl SiteIndexNetwork < NodeName , I >)

Perform a post-order DFS traversal starting from the given root node.

### `pub fn post_order_dfs_by_index(&self, root: NodeIndex) -> Vec < NodeIndex >` (impl SiteIndexNetwork < NodeName , I >)

Perform a post-order DFS traversal starting from the given root NodeIndex.

### `pub fn path_between(&self, from: NodeIndex, to: NodeIndex) -> Option < Vec < NodeIndex > >` (impl SiteIndexNetwork < NodeName , I >)

Find the shortest path between two nodes.

### `pub fn is_connected_subset(&self, nodes: & HashSet < NodeIndex >) -> bool` (impl SiteIndexNetwork < NodeName , I >)

Check if a subset of nodes forms a connected subgraph.

### `pub fn edges_to_canonicalize(&self, current_region: Option < & HashSet < NodeIndex > >, target: NodeIndex) -> CanonicalizeEdges` (impl SiteIndexNetwork < NodeName , I >)

Compute edges to canonicalize from current state to target.

### `pub fn edges_to_canonicalize_by_names(&self, target: & NodeName) -> Option < Vec < (NodeName , NodeName) > >` (impl SiteIndexNetwork < NodeName , I >)

Compute edges to canonicalize from leaves to target, returning node names. This is similar to `edges_to_canonicalize(None, target)` but returns `(from_name, to_name)` pairs instead of `(NodeIndex, NodeIndex)`.

### `pub fn edges_to_canonicalize_to_region(&self, target_region: & HashSet < NodeIndex >) -> CanonicalizeEdges` (impl SiteIndexNetwork < NodeName , I >)

Compute edges to canonicalize from leaves towards a connected region (multiple centers). See [`NodeNameNetwork::edges_to_canonicalize_to_region`] for details.

### `pub fn edges_to_canonicalize_to_region_by_names(&self, target_region: & HashSet < NodeName >) -> Option < Vec < (NodeName , NodeName) > >` (impl SiteIndexNetwork < NodeName , I >)

Compute edges to canonicalize towards a region, returning node names. See [`NodeNameNetwork::edges_to_canonicalize_to_region_by_names`] for details.

### `pub fn apply_operator_topology(&self, operator: & Self) -> Result < Self , String >` (impl SiteIndexNetwork < NodeName , I >)

Check if an operator can act on this state (as a ket). Returns `Ok(result_network)` if the operator can act on self, where `result_network` is the SiteIndexNetwork of the output state.

### `pub fn compatible_site_dimensions(&self, other: & Self) -> bool` (impl SiteIndexNetwork < NodeName , I >)

Check if this network has compatible site dimensions with another. Two networks have compatible site dimensions if: - Same topology (nodes and edges)

### ` fn default() -> Self` (impl SiteIndexNetwork < NodeName , I >)

### ` fn test_site_index_network_basic()`

### ` fn test_post_order_dfs_chain()`

### ` fn test_path_between_chain()`

### ` fn test_edges_to_canonicalize_full()`

### ` fn test_is_connected_subset()`

### ` fn test_share_equivalent_site_index_network()`

### ` fn test_apply_operator_topology()`

### ` fn test_compatible_site_dimensions()`

## src/treetn/addition.rs

### `pub fn compute_merged_bond_indices(&self, other: & Self) -> Result < HashMap < (V , V) , MergedBondInfo < T :: Index > > >` (impl TreeTN < T , V >)

Compute merged bond indices for direct-sum addition. For each edge in the network, compute the merged bond information containing dimensions from both networks and a new merged index.

### `pub fn add(&self, other: & Self) -> Result < Self >` (impl TreeTN < T , V >)

Add two TreeTNs using direct-sum construction. This creates a new TreeTN where each tensor is the direct sum of the corresponding tensors from self and other, with bond dimensions merged.

## src/treetn/canonicalize.rs

### `pub fn canonicalize(mut self, canonical_center: impl IntoIterator < Item = V >, options: CanonicalizationOptions) -> Result < Self >` (impl TreeTN < T , V >)

Canonicalize the network towards the specified center using options. This is the recommended unified API for canonicalization. It accepts: - Center nodes specified by their node names (V)

### `pub fn canonicalize_mut(&mut self, canonical_center: impl IntoIterator < Item = V >, options: CanonicalizationOptions) -> Result < () >` (impl TreeTN < T , V >)

Canonicalize the network in-place towards the specified center using options. This is the `&mut self` version of [`canonicalize`].

### `pub(crate) fn canonicalize_impl(&mut self, canonical_center: impl IntoIterator < Item = V >, form: CanonicalForm, context_name: & str) -> Result < () >` (impl TreeTN < T , V >)

Internal implementation for canonicalization. This is the core canonicalization logic that public methods delegate to.

## src/treetn/contraction.rs

### `pub fn sim_internal_inds(&self) -> Self` (impl TreeTN < T , V >)

Create a copy with all internal (link/bond) indices replaced by fresh IDs. External (site/physical) indices remain unchanged. This is useful when contracting two TreeTNs that might have overlapping internal index IDs.

### `pub fn contract_to_tensor(&self) -> Result < T >` (impl TreeTN < T , V >)

Contract the TreeTN to a single tensor. This method contracts all tensors in the network into a single tensor containing all physical indices. The contraction is performed using

### `pub fn contract_zipup(&self, other: & Self, center: & V, rtol: Option < f64 >, max_rank: Option < usize >) -> Result < Self >` (impl TreeTN < T , V >)

Contract two TreeTNs with the same topology using the zip-up algorithm. The zip-up algorithm traverses from leaves towards the center, contracting corresponding nodes from both networks and optionally truncating at each step.

### `pub fn contract_zipup_with(&self, other: & Self, center: & V, form: CanonicalForm, rtol: Option < f64 >, max_rank: Option < usize >) -> Result < Self >` (impl TreeTN < T , V >)

Contract two TreeTNs with the same topology using the zip-up algorithm with a specified form. See [`contract_zipup`](Self::contract_zipup) for details.

### `pub fn contract_zipup_tree_accumulated(&self, other: & Self, center: & V, form: CanonicalForm, rtol: Option < f64 >, max_rank: Option < usize >) -> Result < Self >` (impl TreeTN < T , V >)

Contract two TreeTNs using zip-up algorithm with accumulated intermediate tensors. This is an improved version of zip-up contraction that maintains intermediate tensors (environment tensors) as it processes from leaves towards the root, similar to

### `pub fn contract_naive(&self, other: & Self) -> Result < T >` (impl TreeTN < T , V >)

Contract two TreeTNs using naive full contraction. This is a reference implementation that: 1. Replaces internal indices with fresh IDs (sim_internal_inds)

### `pub fn validate_ortho_consistency(&self) -> Result < () >` (impl TreeTN < T , V >)

Validate that `canonical_center` and edge `ortho_towards` are consistent. Rules: - If `canonical_center` is empty (not canonicalized), all indices must have `ortho_towards == None`.

### ` fn find_common_indices(a: & T, b: & T) -> Vec < T :: Index >`

Find common indices between two tensors (by ID).

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

### `pub fn with_nfullsweeps(mut self, nfullsweeps: usize) -> Self` (impl ContractionOptions)

Set number of full sweeps for Fit method.

### `pub fn with_convergence_tol(mut self, tol: f64) -> Self` (impl ContractionOptions)

Set convergence tolerance for Fit method.

### `pub fn with_factorize_alg(mut self, alg: FactorizeAlg) -> Self` (impl ContractionOptions)

Set factorization algorithm for Fit method.

### `pub fn contract(tn_a: & TreeTN < T , V >, tn_b: & TreeTN < T , V >, center: & V, options: ContractionOptions) -> Result < TreeTN < T , V > >`

Contract two TreeTNs using the specified method. This is the main entry point for TreeTN contraction. It dispatches to the appropriate algorithm based on the options.

### `pub fn contract_naive_to_treetn(tn_a: & TreeTN < T , V >, tn_b: & TreeTN < T , V >, center: & V, _max_rank: Option < usize >, _rtol: Option < f64 >) -> Result < TreeTN < T , V > >`

Contract two TreeTNs using naive contraction, then decompose back to TreeTN. This method: 1. Contracts both networks to full tensors

## src/treetn/decompose.rs

### `pub fn new(nodes: HashMap < V , Vec < usize > >, edges: Vec < (V , V) >) -> Self` (impl TreeTopology < V >)

Create a new tree topology with the given nodes and edges.

### `pub fn validate(&self) -> Result < () >` (impl TreeTopology < V >)

Validate that this topology describes a tree.

### `pub fn factorize_tensor_to_treetn(tensor: & T, topology: & TreeTopology < V >) -> Result < TreeTN < T , V > >`

Decompose a dense tensor into a TreeTN using QR-based factorization. This function takes a dense tensor and a tree topology specification, then recursively decomposes the tensor using QR factorization to create a TreeTN.

### `pub fn factorize_tensor_to_treetn_with(tensor: & T, topology: & TreeTopology < V >, alg: FactorizeAlg) -> Result < TreeTN < T , V > >`

Factorize a dense tensor into a TreeTN using a specified factorization algorithm. This function takes a dense tensor and a tree topology specification, then recursively decomposes the tensor using the specified algorithm to create a TreeTN.

## src/treetn/fit.rs

### `pub fn new() -> Self` (impl FitEnvironment < T , V >)

Create an empty environment cache.

### `pub fn get(&self, from: & V, to: & V) -> Option < & T >` (impl FitEnvironment < T , V >)

Get the environment tensor for edge (from, to) if it exists.

### `pub(crate) fn insert(&mut self, from: V, to: V, env: T)` (impl FitEnvironment < T , V >)

Insert an environment tensor for edge (from, to). This is mainly for testing; normally use `get_or_compute` for lazy evaluation.

### `pub fn contains(&self, from: & V, to: & V) -> bool` (impl FitEnvironment < T , V >)

Check if environment exists for edge (from, to).

### `pub fn len(&self) -> usize` (impl FitEnvironment < T , V >)

Get the number of cached environments.

### `pub fn is_empty(&self) -> bool` (impl FitEnvironment < T , V >)

Check if the cache is empty.

### `pub fn clear(&mut self)` (impl FitEnvironment < T , V >)

Clear all cached environments.

### `pub fn get_or_compute(&mut self, from: & V, to: & V, tn_a: & TreeTN < T , V >, tn_b: & TreeTN < T , V >, tn_c: & TreeTN < T , V >) -> Result < T >` (impl FitEnvironment < T , V >)

Get or compute the environment tensor for edge (from, to). If the environment is cached, returns it directly. Otherwise, recursively computes it from child environments (towards leaves)

### `pub fn invalidate(&mut self, region: impl IntoIterator < Item = & 'a V >, tn_c: & TreeTN < T , V >)` (impl FitEnvironment < T , V >)

Invalidate all caches affected by updates to tensors in region T. For each `t ∈ T`: 1. Remove all `env[(t, *)]` (0th generation)

### ` fn invalidate_recursive(&mut self, from: & V, to: & V, tn_c: & TreeTN < T , V >)` (impl FitEnvironment < T , V >)

Recursively invalidate caches starting from env[(from, to)] towards leaves. If env[(from, to)] exists, remove it and propagate to env[(to, x)] for all x ≠ from.

### `pub fn verify_structural_consistency(&self, tn_c: & TreeTN < T , V >) -> Result < () >` (impl FitEnvironment < T , V >)

Verify cache structural consistency. For any `env[(x, x1)]` where `x` is not a leaf (has neighbors other than `x1`), all child environments `env[(y, x)]` for neighbors `y ≠ x1` must exist.

### ` fn default() -> Self` (impl FitEnvironment < T , V >)

### ` fn compute_leaf_environment(node: & V, _towards: & V, tn_a: & TreeTN < T , V >, tn_b: & TreeTN < T , V >, tn_c: & TreeTN < T , V >) -> Result < T >`

Compute environment for a leaf node (no children in subtree).

### ` fn compute_single_node_environment(node: & V, towards: & V, tn_a: & TreeTN < T , V >, tn_b: & TreeTN < T , V >, tn_c: & TreeTN < T , V >, child_envs: & [T]) -> Result < T >`

Compute environment for a single node using child environments. This computes: child_envs × A[node] × B[node] × conj(C[node]) leaving open only the indices connecting to `towards`.

### `pub fn new(tn_a: TreeTN < T , V >, tn_b: TreeTN < T , V >, max_rank: Option < usize >, rtol: Option < f64 >) -> Self` (impl FitUpdater < T , V >)

Create a new FitUpdater.

### `pub fn with_factorize_alg(mut self, alg: FactorizeAlg) -> Self` (impl FitUpdater < T , V >)

Set the factorization algorithm.

### ` fn update(&mut self, subtree: TreeTN < T , V >, step: & LocalUpdateStep < V >, full_treetn: & TreeTN < T , V >) -> Result < TreeTN < T , V > >` (impl FitUpdater < T , V >)

### ` fn after_step(&mut self, step: & LocalUpdateStep < V >, full_treetn_after: & TreeTN < T , V >) -> Result < () >` (impl FitUpdater < T , V >)

### ` fn default() -> Self` (impl FitContractionOptions)

### `pub fn new(nfullsweeps: usize) -> Self` (impl FitContractionOptions)

Create new options with specified number of full sweeps.

### `pub fn with_max_rank(mut self, max_rank: usize) -> Self` (impl FitContractionOptions)

Set maximum bond dimension.

### `pub fn with_rtol(mut self, rtol: f64) -> Self` (impl FitContractionOptions)

Set relative tolerance.

### `pub fn with_factorize_alg(mut self, alg: FactorizeAlg) -> Self` (impl FitContractionOptions)

Set factorization algorithm.

### `pub fn with_convergence_tol(mut self, tol: f64) -> Self` (impl FitContractionOptions)

Set convergence tolerance for early termination.

### `pub fn contract_fit(tn_a: & TreeTN < T , V >, tn_b: & TreeTN < T , V >, center: & V, options: FitContractionOptions) -> Result < TreeTN < T , V > >`

Contract two TreeTNs using the fit (variational) algorithm. This algorithm minimizes `||A*B - C||²` iteratively by optimizing each local tensor of C while keeping others fixed.

## src/treetn/linsolve/apply.rs

### ` fn default() -> Self` (impl ApplyOptions)

### `pub fn zipup() -> Self` (impl ApplyOptions)

Create options with ZipUp method (default).

### `pub fn fit() -> Self` (impl ApplyOptions)

Create options with Fit method.

### `pub fn naive() -> Self` (impl ApplyOptions)

Create options with Naive method.

### `pub fn with_max_rank(mut self, max_rank: usize) -> Self` (impl ApplyOptions)

Set maximum bond dimension.

### `pub fn with_rtol(mut self, rtol: f64) -> Self` (impl ApplyOptions)

Set relative tolerance.

### `pub fn with_nfullsweeps(mut self, nfullsweeps: usize) -> Self` (impl ApplyOptions)

Set number of full sweeps for Fit method.

### `pub fn apply_linear_operator(operator: & LinearOperator < T , V >, state: & TreeTN < T , V >, options: ApplyOptions) -> Result < TreeTN < T , V > >`

Apply a LinearOperator to a TreeTN state: compute `A|x⟩`. This function handles: - Partial operators (fills gaps with identity via compose_exclusive_linear_operators)

### ` fn extend_operator_to_full_space(operator: & LinearOperator < T , V >, state: & TreeTN < T , V >) -> Result < LinearOperator < T , V > >`

Extend a partial operator to cover the full state space. Uses `compose_exclusive_linear_operators` to fill gap nodes with identity operators.

### ` fn transform_state_to_input(operator: & LinearOperator < T , V >, state: & TreeTN < T , V >) -> Result < TreeTN < T , V > >`

Transform state's site indices to operator's input indices.

### ` fn transform_output_to_true(operator: & LinearOperator < T , V >, result: TreeTN < T , V >) -> Result < TreeTN < T , V > >`

Transform operator's output indices to true output indices.

### ` fn external_indices(&self) -> Vec < Self :: Index >` (impl LinearOperator < T , V >)

Return all external indices (true input and output indices).

### ` fn num_external_indices(&self) -> usize` (impl LinearOperator < T , V >)

### ` fn replaceind(&self, old_index: & Self :: Index, new_index: & Self :: Index) -> Result < Self >` (impl LinearOperator < T , V >)

Replace an external index (true index) in this operator. This updates the mapping but does NOT modify the internal MPO tensors.

### ` fn replaceinds(&self, old_indices: & [Self :: Index], new_indices: & [Self :: Index]) -> Result < Self >` (impl LinearOperator < T , V >)

Replace multiple external indices.

### `pub fn from_linear_operator(op: LinearOperator < T , V >) -> Self` (impl ArcLinearOperator < T , V >)

Create from an existing LinearOperator.

### `pub fn new(mpo: TreeTN < T , V >, input_mapping: HashMap < V , IndexMapping < T :: Index > >, output_mapping: HashMap < V , IndexMapping < T :: Index > >) -> Self` (impl ArcLinearOperator < T , V >)

Create a new ArcLinearOperator.

### `pub fn mpo_mut(&mut self) -> & mut TreeTN < T , V >` (impl ArcLinearOperator < T , V >)

Get a mutable reference to the MPO, cloning if necessary. This implements Copy-on-Write semantics: if this is the only reference, no copy is made. If there are other references, the MPO is cloned first.

### `pub fn mpo(&self) -> & TreeTN < T , V >` (impl ArcLinearOperator < T , V >)

Get an immutable reference to the MPO.

### `pub fn into_linear_operator(self) -> LinearOperator < T , V >` (impl ArcLinearOperator < T , V >)

Convert back to a LinearOperator (unwraps Arc if possible).

### `pub fn get_input_mapping(&self, node: & V) -> Option < & IndexMapping < T :: Index > >` (impl ArcLinearOperator < T , V >)

Get input mapping for a node.

### `pub fn get_output_mapping(&self, node: & V) -> Option < & IndexMapping < T :: Index > >` (impl ArcLinearOperator < T , V >)

Get output mapping for a node.

### `pub fn input_mappings(&self) -> & HashMap < V , IndexMapping < T :: Index > >` (impl ArcLinearOperator < T , V >)

Get all input mappings.

### `pub fn output_mappings(&self) -> & HashMap < V , IndexMapping < T :: Index > >` (impl ArcLinearOperator < T , V >)

Get all output mappings.

### `pub fn node_names(&self) -> HashSet < V >` (impl ArcLinearOperator < T , V >)

Get node names covered by this operator.

### ` fn make_index(dim: usize) -> DynIndex`

### ` fn create_chain_site_network(n: usize) -> SiteIndexNetwork < String , DynIndex >`

### ` fn test_apply_options_builder()`

### ` fn test_linear_operator_tensor_index()`

### ` fn test_arc_linear_operator_cow()`

## src/treetn/linsolve/environment.rs

### `pub fn neighbors(&self, node: & V) -> Self :: Neighbors < '_ >` (trait NetworkTopology)

Get neighbors of a node.

### `pub fn new() -> Self` (impl EnvironmentCache < T , V >)

Create a new empty environment cache.

### `pub fn get(&self, from: & V, to: & V) -> Option < & T >` (impl EnvironmentCache < T , V >)

Get a cached environment tensor if it exists.

### `pub fn insert(&mut self, from: V, to: V, env: T)` (impl EnvironmentCache < T , V >)

Insert an environment tensor.

### `pub fn contains(&self, from: & V, to: & V) -> bool` (impl EnvironmentCache < T , V >)

Check if environment exists for edge (from, to).

### `pub fn len(&self) -> usize` (impl EnvironmentCache < T , V >)

Get the number of cached environments.

### `pub fn is_empty(&self) -> bool` (impl EnvironmentCache < T , V >)

Check if the cache is empty.

### `pub fn clear(&mut self)` (impl EnvironmentCache < T , V >)

Clear all cached environments.

### `pub fn invalidate(&mut self, region: impl IntoIterator < Item = & 'a V >, topology: & NT)` (impl EnvironmentCache < T , V >)

Invalidate all caches affected by updates to tensors in region. For each `t ∈ region`: 1. Remove all `env[(t, *)]` (0th generation)

### ` fn invalidate_recursive(&mut self, from: & V, to: & V, topology: & NT)` (impl EnvironmentCache < T , V >)

Recursively invalidate caches starting from env[(from, to)] towards leaves.

### ` fn default() -> Self` (impl EnvironmentCache < T , V >)

### ` fn neighbors(&self, node: & NodeName) -> Self :: Neighbors < '_ >` (impl SiteIndexNetwork < NodeName , I >)

### ` fn test_environment_cache_creation()`

## src/treetn/linsolve/linear_operator.rs

### `pub fn new(mpo: TreeTN < T , V >, input_mapping: HashMap < V , IndexMapping < T :: Index > >, output_mapping: HashMap < V , IndexMapping < T :: Index > >) -> Self` (impl LinearOperator < T , V >)

Create a new LinearOperator from an MPO and index mappings.

### `pub fn from_mpo_and_state(mpo: TreeTN < T , V >, state: & TreeTN < T , V >) -> Result < Self >` (impl LinearOperator < T , V >)

Create a LinearOperator from an MPO and a reference state. This assumes: - The MPO has site indices that we need to map

### `pub fn apply(&self, state: & TreeTN < T , V >) -> Result < TreeTN < T , V > >` (impl LinearOperator < T , V >)

Apply the operator to a state: compute `A|x⟩`. This handles index transformations automatically: 1. Replace state's site indices with MPO's input indices (s_in_tmp)

### `pub fn apply_local(&self, local_tensor: & T, region: & [V]) -> Result < T >` (impl LinearOperator < T , V >)

Apply the operator to a local tensor at a specific region. This is used during the sweep for local updates.

### `pub fn mpo(&self) -> & TreeTN < T , V >` (impl LinearOperator < T , V >)

Get the internal MPO.

### `pub fn get_input_mapping(&self, node: & V) -> Option < & IndexMapping < T :: Index > >` (impl LinearOperator < T , V >)

Get input mapping for a node.

### `pub fn get_output_mapping(&self, node: & V) -> Option < & IndexMapping < T :: Index > >` (impl LinearOperator < T , V >)

Get output mapping for a node.

### ` fn site_indices(&self) -> HashSet < T :: Index >` (impl LinearOperator < T , V >)

### ` fn site_index_network(&self) -> & SiteIndexNetwork < V , T :: Index >` (impl LinearOperator < T , V >)

### ` fn node_names(&self) -> HashSet < V >` (impl LinearOperator < T , V >)

### `pub fn input_site_indices(&self) -> HashSet < T :: Index >` (impl LinearOperator < T , V >)

Get all input site indices (true indices from state space).

### `pub fn output_site_indices(&self) -> HashSet < T :: Index >` (impl LinearOperator < T , V >)

Get all output site indices (true indices from result space).

### `pub fn input_mappings(&self) -> & HashMap < V , IndexMapping < T :: Index > >` (impl LinearOperator < T , V >)

Get all input mappings.

### `pub fn output_mappings(&self) -> & HashMap < V , IndexMapping < T :: Index > >` (impl LinearOperator < T , V >)

Get all output mappings.

## src/treetn/linsolve/local_linop.rs

### `pub fn new(projected_operator: Arc < RwLock < ProjectedOperator < T , V > > >, region: Vec < V >, state: TreeTN < T , V >, a0: AnyScalar, a1: AnyScalar) -> Self` (impl LocalLinOp < T , V >)

Create a new LocalLinOp for V_in = V_out case.

### `pub fn with_bra_state(projected_operator: Arc < RwLock < ProjectedOperator < T , V > > >, region: Vec < V >, state: TreeTN < T , V >, bra_state: TreeTN < T , V >, a0: AnyScalar, a1: AnyScalar) -> Self` (impl LocalLinOp < T , V >)

Create a new LocalLinOp for V_in ≠ V_out case with explicit bra_state.

### ` fn get_bra_state(&self) -> & TreeTN < T , V >` (impl LocalLinOp < T , V >)

Get the bra state for environment computation. Returns bra_state if set, otherwise returns state (V_in = V_out case).

### `pub fn apply(&self, x: & T) -> Result < T >` (impl LocalLinOp < T , V >)

Apply the local linear operator: `y = a₀ * x + a₁ * H * x` This is used by `tensor4all_core::krylov::gmres` to solve the local problem.

## src/treetn/linsolve/options.rs

### ` fn default() -> Self` (impl LinsolveOptions)

### `pub fn new(nfullsweeps: usize) -> Self` (impl LinsolveOptions)

Create new options with specified number of full sweeps.

### `pub fn with_nfullsweeps(mut self, nfullsweeps: usize) -> Self` (impl LinsolveOptions)

Set number of full sweeps.

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

### `pub fn new(operator: TreeTN < T , V >) -> Self` (impl ProjectedOperator < T , V >)

Create a new ProjectedOperator.

### `pub fn with_index_mappings(operator: TreeTN < T , V >, input_mapping: HashMap < V , IndexMapping < T :: Index > >, output_mapping: HashMap < V , IndexMapping < T :: Index > >) -> Self` (impl ProjectedOperator < T , V >)

Create a new ProjectedOperator with index mappings from a LinearOperator. The mappings define how state's site indices relate to MPO's internal indices. This is required when the MPO uses internal indices (s_in_tmp, s_out_tmp)

### `pub fn apply(&mut self, v: & T, region: & [V], ket_state: & TreeTN < T , V >, bra_state: & TreeTN < T , V >, topology: & NT) -> Result < T >` (impl ProjectedOperator < T , V >)

Apply the operator to a local tensor: compute `H|v⟩` at the current position. If index mappings are set (via `with_index_mappings`), this method: 1. Transforms input `v`'s site indices to MPO's internal input indices

### ` fn ensure_environments(&mut self, region: & [V], ket_state: & TreeTN < T , V >, bra_state: & TreeTN < T , V >, topology: & NT) -> Result < () >` (impl ProjectedOperator < T , V >)

Ensure environments are computed for neighbors of the region.

### ` fn compute_environment(&mut self, from: & V, to: & V, ket_state: & TreeTN < T , V >, bra_state: & TreeTN < T , V >, topology: & NT) -> Result < T >` (impl ProjectedOperator < T , V >)

Recursively compute environment for edge (from, to).

### `pub fn local_dimension(&self, region: & [V]) -> usize` (impl ProjectedOperator < T , V >)

Compute the local dimension (size of the local Hilbert space).

### `pub fn invalidate(&mut self, region: & [V], topology: & NT)` (impl ProjectedOperator < T , V >)

Invalidate caches affected by updates to the given region.

## src/treetn/linsolve/projected_state.rs

### `pub fn new(rhs: TreeTN < T , V >) -> Self` (impl ProjectedState < T , V >)

Create a new ProjectedState.

### `pub fn local_constant_term(&mut self, region: & [V], ket_state: & TreeTN < T , V >, topology: & NT) -> Result < T >` (impl ProjectedState < T , V >)

Compute the local constant term `<b|_local` for the given region. This returns the local RHS tensors contracted with environments.

### `pub fn local_constant_term_with_bra(&mut self, region: & [V], _ket_state: & TreeTN < T , V >, bra_state: & TreeTN < T , V >, topology: & NT) -> Result < T >` (impl ProjectedState < T , V >)

Compute the local constant term `<b|_local` for the given region with explicit bra state. For V_in ≠ V_out case, provides a reference state in V_out for environment computation.

### ` fn ensure_environments(&mut self, region: & [V], bra_state: & TreeTN < T , V >, topology: & NT) -> Result < () >` (impl ProjectedState < T , V >)

Ensure environments are computed for neighbors of the region.

### ` fn compute_environment(&mut self, from: & V, to: & V, bra_state: & TreeTN < T , V >, topology: & NT) -> Result < T >` (impl ProjectedState < T , V >)

Recursively compute environment for edge (from, to). Computes `<b|ref_out>` partial contraction at node `from`.

### `pub fn invalidate(&mut self, region: & [V], topology: & NT)` (impl ProjectedState < T , V >)

Invalidate caches affected by updates to the given region.

## src/treetn/linsolve/updater.rs

### ` fn default() -> Self` (impl LinsolveVerifyReport < V >)

### ` fn fmt(&self, f: & mut std :: fmt :: Formatter < '_ >) -> std :: fmt :: Result` (impl LinsolveVerifyReport < V >)

### `pub fn new(operator: TreeTN < T , V >, rhs: TreeTN < T , V >, options: LinsolveOptions) -> Self` (impl LinsolveUpdater < T , V >)

Create a new LinsolveUpdater for V_in = V_out case.

### `pub fn with_reference_state(operator: TreeTN < T , V >, rhs: TreeTN < T , V >, reference_state_out: TreeTN < T , V >, options: LinsolveOptions) -> Self` (impl LinsolveUpdater < T , V >)

Create a new LinsolveUpdater for V_in ≠ V_out case with explicit reference state.

### `pub fn with_index_mappings(operator: TreeTN < T , V >, input_mapping: HashMap < V , IndexMapping < T :: Index > >, output_mapping: HashMap < V , IndexMapping < T :: Index > >, rhs: TreeTN < T , V >, reference_state_out: Option < TreeTN < T , V > >, options: LinsolveOptions) -> Self` (impl LinsolveUpdater < T , V >)

Create a new LinsolveUpdater with index mappings for correct index handling. Use this when the MPO uses internal indices (s_in_tmp, s_out_tmp) that differ from the state's site indices. The mappings define how to translate between them.

### `pub fn get_bra_state(&self, ket_state: & 'a TreeTN < T , V >) -> & 'a TreeTN < T , V >` (impl LinsolveUpdater < T , V >)

Get the bra state for environment computation. Returns reference_state_out if set, otherwise returns the ket_state (V_in = V_out case).

### `pub fn verify(&self, state: & TreeTN < T , V >) -> Result < LinsolveVerifyReport < V > >` (impl LinsolveUpdater < T , V >)

Verify internal data consistency between operator, RHS, and state. This function checks that: 1. The operator's site space structure is compatible with the state

### ` fn contract_region(&self, subtree: & TreeTN < T , V >, region: & [V]) -> Result < T >` (impl LinsolveUpdater < T , V >)

Contract all tensors in the region into a single local tensor.

### ` fn build_subtree_topology(&self, solved_tensor: & T, region: & [V], full_treetn: & TreeTN < T , V >) -> Result < TreeTopology < V > >` (impl LinsolveUpdater < T , V >)

Build TreeTopology for the subtree region from the solved tensor. Maps each node to the positions of its indices in the solved tensor.

### ` fn copy_decomposed_to_subtree(&self, subtree: & mut TreeTN < T , V >, decomposed: & TreeTN < T , V >, region: & [V], full_treetn: & TreeTN < T , V >) -> Result < () >` (impl LinsolveUpdater < T , V >)

Copy decomposed tensors back to subtree, preserving original bond IDs.

### ` fn solve_local(&mut self, region: & [V], init: & T, state: & TreeTN < T , V >) -> Result < T >` (impl LinsolveUpdater < T , V >)

Solve the local linear problem using GMRES. Solves: (a₀ + a₁ * H_local) |x_local⟩ = |b_local⟩

### ` fn update(&mut self, subtree: TreeTN < T , V >, step: & LocalUpdateStep < V >, full_treetn: & TreeTN < T , V >) -> Result < TreeTN < T , V > >` (impl LinsolveUpdater < T , V >)

### ` fn after_step(&mut self, step: & LocalUpdateStep < V >, full_treetn_after: & TreeTN < T , V >) -> Result < () >` (impl LinsolveUpdater < T , V >)

## src/treetn/linsolve.rs

### ` fn validate_linsolve_inputs(operator: & TreeTN < T , V >, rhs: & TreeTN < T , V >, init: & TreeTN < T , V >) -> Result < () >`

Validate that operator, rhs, and init have compatible structures for linsolve. Checks: 1. Operator can act on init (same topology)

### `pub fn linsolve(operator: & TreeTN < T , V >, rhs: & TreeTN < T , V >, init: TreeTN < T , V >, center: & V, options: LinsolveOptions) -> Result < LinsolveResult < T , V > >`

Solve the linear system `(a₀ + a₁ * H) |x⟩ = |b⟩` for TreeTN.

## src/treetn/localupdate.rs

### `pub fn from_treetn(treetn: & TreeTN < T , V >, root: & V, nsite: usize) -> Option < Self >` (impl LocalUpdateSweepPlan < V >)

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

### `pub fn before_step(&mut self, _step: & LocalUpdateStep < V >, _full_treetn_before: & TreeTN < T , V >) -> Result < () >` (trait LocalUpdater default)

Optional hook called before performing an update step. This is called with the full TreeTN state *before* the update is applied. Implementors can use it to validate assumptions or prefetch/update caches.

### `pub fn update(&mut self, subtree: TreeTN < T , V >, step: & LocalUpdateStep < V >, full_treetn: & TreeTN < T , V >) -> Result < TreeTN < T , V > >` (trait LocalUpdater)

Update a local subtree.

### `pub fn after_step(&mut self, _step: & LocalUpdateStep < V >, _full_treetn_after: & TreeTN < T , V >) -> Result < () >` (trait LocalUpdater default)

Optional hook called after an update step has been applied to the full TreeTN. This is called after: - The updated subtree has been inserted back into the full TreeTN

### `pub fn apply_local_update_sweep(treetn: & mut TreeTN < T , V >, plan: & LocalUpdateSweepPlan < V >, updater: & mut U) -> Result < () >`

Apply a local update sweep to a TreeTN. This function orchestrates the sweep by: 1. Iterating through the sweep plan

### `pub fn new(max_rank: Option < usize >, rtol: Option < f64 >) -> Self` (impl TruncateUpdater)

Create a new truncation updater.

### ` fn update(&mut self, subtree: TreeTN < T , V >, step: & LocalUpdateStep < V >, _full_treetn: & TreeTN < T , V >) -> Result < TreeTN < T , V > >` (impl TruncateUpdater)

### `pub fn extract_subtree(&self, node_names: & [V]) -> Result < Self >` (impl TreeTN < T , V >)

Extract a sub-tree from this TreeTN. Creates a new TreeTN containing only the specified nodes and their connecting edges. Tensors are cloned into the new TreeTN.

### `pub fn replace_subtree(&mut self, node_names: & [V], replacement: & Self) -> Result < () >` (impl TreeTN < T , V >)

Replace a sub-tree with another TreeTN of the same topology. This method replaces the tensors and ortho_towards directions for a subset of nodes with those from another TreeTN. The replacement TreeTN must have

### ` fn create_y_shape_treetn() -> (TreeTN < TensorDynLen , String > , DynIndex , DynIndex , DynIndex , DynIndex ,)`

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

### ` fn create_chain_treetn() -> TreeTN < TensorDynLen , String >`

Create a chain TreeTN: A - B - C Each node has a site index of dim 2, bonds of dim 4

### ` fn test_truncate_updater_basic()`

### ` fn test_apply_local_update_sweep_preserves_structure()`

### ` fn test_apply_local_update_sweep_requires_canonicalization()`

### ` fn test_sweep_plan_from_treetn()`

## src/treetn/mod.rs

### `pub fn new() -> Self` (impl TreeTN < T , V >)

Create a new empty TreeTN. Use `add_tensor()` to add tensors and `connect()` to establish bonds manually.

### `pub fn from_tensors(tensors: Vec < T >, node_names: Vec < V >) -> Result < Self >` (impl TreeTN < T , V >)

Create a TreeTN from a list of tensors and node names using einsum rule. This function connects tensors that share common indices (by ID). The algorithm is O(n) where n is the number of tensors:

### ` fn from_tensors_unchecked(tensors: Vec < T >, node_names: Vec < V >) -> Result < Self >` (impl TreeTN < T , V >)

Internal version of `from_tensors` that skips verification. Used by `verify_internal_consistency` to avoid infinite recursion.

### `pub fn add_tensor(&mut self, node_name: V, tensor: T) -> Result < NodeIndex >` (impl TreeTN < T , V >)

Add a tensor to the network with a node name. Returns the NodeIndex for the newly added tensor. Also updates the site_index_network with the physical indices (all indices initially,

### `pub fn add_tensor_auto_name(&mut self, tensor: T) -> NodeIndex` (impl TreeTN < T , V >)

Add a tensor to the network using NodeIndex as the node name. This method only works when `V = NodeIndex`. Returns the NodeIndex for the newly added tensor.

### `pub fn connect(&mut self, node_a: NodeIndex, index_a: & T :: Index, node_b: NodeIndex, index_b: & T :: Index) -> Result < EdgeIndex >` (impl TreeTN < T , V >)

Connect two tensors via a specified pair of indices. The indices must have the same ID (Einsum mode).

### `pub(crate) fn add_tensor_internal(&mut self, node_name: V, tensor: T) -> Result < NodeIndex >` (impl TreeTN < T , V >)

Internal method to add a tensor with a node name.

### `pub(crate) fn connect_internal(&mut self, node_a: NodeIndex, index_a: & T :: Index, node_b: NodeIndex, index_b: & T :: Index) -> Result < EdgeIndex >` (impl TreeTN < T , V >)

Internal method to connect two tensors. In Einsum mode, `index_a` and `index_b` must have the same ID.

### `pub(crate) fn prepare_sweep_to_center(&mut self, canonical_center: impl IntoIterator < Item = V >, context_name: & str) -> Result < Option < SweepContext > >` (impl TreeTN < T , V >)

Prepare context for sweep-to-center operations. This method: 1. Validates tree structure

### `pub(crate) fn sweep_edge(&mut self, src: NodeIndex, dst: NodeIndex, factorize_options: & FactorizeOptions, context_name: & str) -> Result < () >` (impl TreeTN < T , V >)

Process one edge during a sweep operation. Factorizes the tensor at `src` node, absorbs the right factor into `dst` (parent), and updates the edge bond and ortho_towards.

### `pub fn tensor(&self, node: NodeIndex) -> Option < & T >` (impl TreeTN < T , V >)

Get a reference to a tensor by NodeIndex.

### `pub fn tensor_mut(&mut self, node: NodeIndex) -> Option < & mut T >` (impl TreeTN < T , V >)

Get a mutable reference to a tensor by NodeIndex.

### `pub fn replace_tensor(&mut self, node: NodeIndex, new_tensor: T) -> Result < Option < T > >` (impl TreeTN < T , V >)

Replace a tensor at the given node with a new tensor. Validates that the new tensor contains all indices used in connections to this node. Returns an error if any connection index is missing.

### `pub fn bond_index(&self, edge: EdgeIndex) -> Option < & T :: Index >` (impl TreeTN < T , V >)

Get the bond index for a given edge.

### `pub fn bond_index_mut(&mut self, edge: EdgeIndex) -> Option < & mut T :: Index >` (impl TreeTN < T , V >)

Get a mutable reference to the bond index for a given edge.

### `pub fn edges_for_node(&self, node: NodeIndex) -> Vec < (EdgeIndex , NodeIndex) >` (impl TreeTN < T , V >)

Get all edges connected to a node.

### `pub fn replace_edge_bond(&mut self, edge: EdgeIndex, new_bond_index: T :: Index) -> Result < () >` (impl TreeTN < T , V >)

Replace the bond index for an edge (e.g., after SVD creates a new bond index). Also updates site_index_network: the old bond index becomes physical again, and the new bond index is removed from physical indices.

### `pub fn set_ortho_towards(&mut self, index: & T :: Index, dir: Option < V >)` (impl TreeTN < T , V >)

Set the orthogonalization direction for an index (bond or site). The direction is specified as a node name (or None to clear).

### `pub fn ortho_towards_for_index(&self, index: & T :: Index) -> Option < & V >` (impl TreeTN < T , V >)

Get the node name that the orthogonalization points towards for an index. Returns None if ortho_towards is not set for this index.

### `pub fn set_edge_ortho_towards(&mut self, edge: petgraph :: stable_graph :: EdgeIndex, dir: Option < V >) -> Result < () >` (impl TreeTN < T , V >)

Set the orthogonalization direction for an edge (by EdgeIndex). This is a convenience method that looks up the bond index and calls `set_ortho_towards`. The direction is specified as a node name (or None to clear).

### `pub fn ortho_towards_node(&self, edge: petgraph :: stable_graph :: EdgeIndex) -> Option < & V >` (impl TreeTN < T , V >)

Get the node name that the orthogonalization points towards for an edge. Returns None if ortho_towards is not set for this edge's bond index.

### `pub fn ortho_towards_node_index(&self, edge: petgraph :: stable_graph :: EdgeIndex) -> Option < NodeIndex >` (impl TreeTN < T , V >)

Get the NodeIndex that the orthogonalization points towards for an edge. Returns None if ortho_towards is not set for this edge's bond index.

### `pub fn validate_tree(&self) -> Result < () >` (impl TreeTN < T , V >)

Validate that the graph is a tree (or forest). Checks: - The graph is connected (all nodes reachable from the first node)

### `pub fn node_count(&self) -> usize` (impl TreeTN < T , V >)

Get the number of nodes in the network.

### `pub fn edge_count(&self) -> usize` (impl TreeTN < T , V >)

Get the number of edges in the network.

### `pub fn node_index(&self, node_name: & V) -> Option < NodeIndex >` (impl TreeTN < T , V >)

Get the NodeIndex for a node by name.

### `pub fn edge_between(&self, node_a: & V, node_b: & V) -> Option < EdgeIndex >` (impl TreeTN < T , V >)

Get the EdgeIndex for the edge between two nodes by name. Returns `None` if either node doesn't exist or there's no edge between them.

### `pub fn node_indices(&self) -> Vec < NodeIndex >` (impl TreeTN < T , V >)

Get all node indices in the tree tensor network.

### `pub fn node_names(&self) -> Vec < V >` (impl TreeTN < T , V >)

Get all node names in the tree tensor network.

### `pub fn edges_to_canonicalize_by_names(&self, target: & V) -> Option < Vec < (V , V) > >` (impl TreeTN < T , V >)

Compute edges to canonicalize from leaves to target, returning node names. Returns `(from, to)` pairs in the order they should be processed: - `from` is the node being factorized

### `pub fn canonical_center(&self) -> & HashSet < V >` (impl TreeTN < T , V >)

Get a reference to the orthogonalization region (using node names). When empty, the network is not canonicalized.

### `pub fn is_canonicalized(&self) -> bool` (impl TreeTN < T , V >)

Check if the network is canonicalized. Returns `true` if `canonical_center` is non-empty, `false` otherwise.

### `pub fn set_canonical_center(&mut self, region: impl IntoIterator < Item = V >) -> Result < () >` (impl TreeTN < T , V >)

Set the orthogonalization region (using node names). Validates that all specified nodes exist in the graph.

### `pub fn clear_canonical_center(&mut self)` (impl TreeTN < T , V >)

Clear the orthogonalization region (mark network as not canonicalized). Also clears the canonical form.

### `pub fn canonical_form(&self) -> Option < CanonicalForm >` (impl TreeTN < T , V >)

Get the current canonical form. Returns `None` if not canonicalized.

### `pub fn add_to_canonical_center(&mut self, node_name: V) -> Result < () >` (impl TreeTN < T , V >)

Add a node to the orthogonalization region. Validates that the node exists in the graph.

### `pub fn remove_from_canonical_center(&mut self, node_name: & V) -> bool` (impl TreeTN < T , V >)

Remove a node from the orthogonalization region. Returns `true` if the node was in the region, `false` otherwise.

### `pub fn site_index_network(&self) -> & SiteIndexNetwork < V , T :: Index >` (impl TreeTN < T , V >)

Get a reference to the site index network. The site index network contains both topology (graph structure) and site space (physical indices).

### `pub fn site_index_network_mut(&mut self) -> & mut SiteIndexNetwork < V , T :: Index >` (impl TreeTN < T , V >)

Get a mutable reference to the site index network.

### `pub fn site_space(&self, node_name: & V) -> Option < & std :: collections :: HashSet < T :: Index > >` (impl TreeTN < T , V >)

Get a reference to the site space (physical indices) for a node.

### `pub fn site_space_mut(&mut self, node_name: & V) -> Option < & mut std :: collections :: HashSet < T :: Index > >` (impl TreeTN < T , V >)

Get a mutable reference to the site space (physical indices) for a node.

### `pub fn share_equivalent_site_index_network(&self, other: & Self) -> bool` (impl TreeTN < T , V >)

Check if two TreeTNs share equivalent site index network structure. Two TreeTNs share equivalent structure if: - Same topology (nodes and edges)

### `pub fn same_topology(&self, other: & Self) -> bool` (impl TreeTN < T , V >)

Check if two TreeTNs have the same topology (graph structure). This only checks that both networks have the same nodes and edges, not that they have the same site indices.

### `pub fn same_appearance(&self, other: & Self) -> bool` (impl TreeTN < T , V >)

Check if two TreeTNs have the same "appearance". Two TreeTNs have the same appearance if: 1. They have the same topology (same nodes and edges)

### `pub fn verify_internal_consistency(&self) -> Result < () >` (impl TreeTN < T , V >)

Verify internal data consistency by checking structural invariants and reconstructing the TreeTN. This function performs two categories of checks:

### `pub(crate) fn common_inds(inds_a: & [I], inds_b: & [I]) -> Vec < I >`

Find common indices between two slices of indices.

### `pub(crate) fn compute_strides(dims: & [usize]) -> Vec < usize >`

Compute strides for row-major (C-order) indexing.

### `pub(crate) fn linear_to_multi_index(linear: usize, strides: & [usize], rank: usize) -> Vec < usize >`

Convert linear index to multi-index.

### `pub(crate) fn multi_to_linear_index(multi: & [usize], strides: & [usize]) -> usize`

Convert multi-index to linear index.

## src/treetn/operator_impl.rs

### ` fn site_indices(&self) -> HashSet < T :: Index >` (impl TreeTN < T , V >)

### ` fn site_index_network(&self) -> & SiteIndexNetwork < V , T :: Index >` (impl TreeTN < T , V >)

### ` fn node_names(&self) -> HashSet < V >` (impl TreeTN < T , V >)

## src/treetn/ops.rs

### ` fn default() -> Self` (impl TreeTN < T , V >)

### ` fn clone(&self) -> Self` (impl TreeTN < T , V >)

### ` fn fmt(&self, f: & mut std :: fmt :: Formatter < '_ >) -> std :: fmt :: Result` (impl TreeTN < T , V >)

### `pub fn log_norm(&mut self) -> Result < f64 >` (impl TreeTN < T , V >)

Compute log(||TreeTN||_F), the log of the Frobenius norm. Uses canonicalization to avoid numerical overflow: when canonicalized to a single site with Unitary form,

## src/treetn/tensor_like.rs

### ` fn external_indices(&self) -> Vec < Self :: Index >` (impl TreeTN < T , V >)

Return all external (site/physical) indices from all nodes. This collects all site indices from `site_index_network`. Bond indices are NOT included (they are internal to the network).

### ` fn num_external_indices(&self) -> usize` (impl TreeTN < T , V >)

### ` fn replaceind(&self, old_index: & Self :: Index, new_index: & Self :: Index) -> Result < Self >` (impl TreeTN < T , V >)

Replace an index in this TreeTN. Looks up the index location (site or link) and replaces it in: - The tensor containing it

### ` fn replaceinds(&self, old_indices: & [Self :: Index], new_indices: & [Self :: Index]) -> Result < Self >` (impl TreeTN < T , V >)

Replace multiple indices in this TreeTN.

## src/treetn/transform.rs

### `pub fn fuse_to(&self, target: & SiteIndexNetwork < TargetV , T :: Index >) -> Result < TreeTN < T , TargetV > >` (impl TreeTN < T , V >)

Fuse (merge) adjacent nodes to match the target structure. This operation contracts adjacent nodes that should be merged according to the target `SiteIndexNetwork`. The target structure must be a "coarsening"

### ` fn contract_node_group(&self, nodes: & HashSet < V >) -> Result < T >` (impl TreeTN < T , V >)

Contract a group of nodes into a single tensor. The nodes must form a connected subtree in the current TreeTN. Contracts all internal bonds (bonds between nodes in the group),

### `pub fn split_to(&self, target: & SiteIndexNetwork < TargetV , T :: Index >, options: & SplitOptions) -> Result < TreeTN < T , TargetV > >` (impl TreeTN < T , V >)

Split nodes to match the target structure. This operation splits nodes that contain site indices belonging to multiple target nodes. The target structure must be a "refinement" of the current

### ` fn split_tensor_for_targets(&self, tensor: & T, site_to_target: & HashMap < < T :: Index as IndexLike > :: Id , TargetV >) -> Result < Vec < (TargetV , T) > >` (impl TreeTN < T , V >)

Split a tensor into multiple tensors, one for each target node. This uses QR factorization to iteratively separate site indices belonging to different target nodes.

## src/treetn/truncate.rs

### `pub fn truncate(mut self, canonical_center: impl IntoIterator < Item = V >, options: TruncationOptions) -> Result < Self >` (impl TreeTN < T , V >)

Truncate the network towards the specified center using options. This is the recommended unified API for truncation. It accepts: - Center nodes specified by their node names (V)

### `pub fn truncate_mut(&mut self, canonical_center: impl IntoIterator < Item = V >, options: TruncationOptions) -> Result < () >` (impl TreeTN < T , V >)

Truncate the network in-place towards the specified center using options. This is the `&mut self` version of [`truncate`].

### `pub(crate) fn truncate_impl(&mut self, canonical_center: impl IntoIterator < Item = V >, form: CanonicalForm, rtol: Option < f64 >, max_rank: Option < usize >, context_name: & str) -> Result < () >` (impl TreeTN < T , V >)

Internal implementation for truncation. Uses LocalUpdateSweepPlan with TruncateUpdater for full two-site sweeps.

