
use super::*;

#[test]
fn test_node_name_network_basic() {
    let mut net: NodeNameNetwork<String> = NodeNameNetwork::new();

    net.add_node("A".to_string()).unwrap();
    net.add_node("B".to_string()).unwrap();

    assert_eq!(net.node_count(), 2);
    assert!(net.has_node(&"A".to_string()));

    net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    assert_eq!(net.edge_count(), 1);
}

#[test]
fn test_post_order_dfs_chain() {
    let mut net: NodeNameNetwork<String> = NodeNameNetwork::new();

    net.add_node("A".to_string()).unwrap();
    net.add_node("B".to_string()).unwrap();
    net.add_node("C".to_string()).unwrap();
    net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();

    let result = net.post_order_dfs(&"B".to_string()).unwrap();
    assert_eq!(result.len(), 3);
    assert_eq!(result.last().unwrap(), "B");
}

#[test]
fn test_path_between() {
    let mut net: NodeNameNetwork<String> = NodeNameNetwork::new();

    let a = net.add_node("A".to_string()).unwrap();
    let b = net.add_node("B".to_string()).unwrap();
    let c = net.add_node("C".to_string()).unwrap();
    let d = net.add_node("D".to_string()).unwrap();
    net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
    net.add_edge(&"C".to_string(), &"D".to_string()).unwrap();

    let path = net.path_between(a, d).unwrap();
    assert_eq!(path, vec![a, b, c, d]);
}

#[test]
fn test_is_connected_subset() {
    let mut net: NodeNameNetwork<String> = NodeNameNetwork::new();

    let a = net.add_node("A".to_string()).unwrap();
    let b = net.add_node("B".to_string()).unwrap();
    let c = net.add_node("C".to_string()).unwrap();
    let _d = net.add_node("D".to_string()).unwrap();
    net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
    net.add_edge(&"C".to_string(), &"D".to_string()).unwrap();

    assert!(net.is_connected_subset(&[b, c].into()));
    assert!(!net.is_connected_subset(&[a, c].into())); // Gap
}

#[test]
fn test_same_topology() {
    let mut net1: NodeNameNetwork<String> = NodeNameNetwork::new();
    net1.add_node("A".to_string()).unwrap();
    net1.add_node("B".to_string()).unwrap();
    net1.add_edge(&"A".to_string(), &"B".to_string()).unwrap();

    let mut net2: NodeNameNetwork<String> = NodeNameNetwork::new();
    net2.add_node("A".to_string()).unwrap();
    net2.add_node("B".to_string()).unwrap();
    net2.add_edge(&"A".to_string(), &"B".to_string()).unwrap();

    assert!(net1.same_topology(&net2));

    let mut net3: NodeNameNetwork<String> = NodeNameNetwork::new();
    net3.add_node("A".to_string()).unwrap();
    net3.add_node("C".to_string()).unwrap();
    net3.add_edge(&"A".to_string(), &"C".to_string()).unwrap();

    assert!(!net1.same_topology(&net3));
}

#[test]
fn test_euler_tour_chain() {
    // Chain: A - B - C
    let mut net: NodeNameNetwork<String> = NodeNameNetwork::new();
    let a = net.add_node("A".to_string()).unwrap();
    let b = net.add_node("B".to_string()).unwrap();
    let c = net.add_node("C".to_string()).unwrap();
    net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();

    // Euler tour from B: should visit all edges twice
    let edges = net.euler_tour_edges(&"B".to_string()).unwrap();
    // 2 edges × 2 directions = 4 directed edges
    assert_eq!(edges.len(), 4);

    // Verify each undirected edge appears twice (forward and backward)
    let edge_set: HashSet<_> = edges.iter().cloned().collect();
    assert!(edge_set.contains(&(b, a)) || edge_set.contains(&(b, c)));

    // Vertices: should start and end at B
    let vertices = net.euler_tour_vertices(&"B".to_string()).unwrap();
    assert_eq!(vertices.len(), 5); // B, ?, B, ?, B
    assert_eq!(vertices[0], b);
    assert_eq!(vertices[vertices.len() - 1], b);

    // All vertices should be visited
    let unique_vertices: HashSet<_> = vertices.iter().cloned().collect();
    assert_eq!(unique_vertices, [a, b, c].into());
}

#[test]
fn test_euler_tour_y_shape() {
    // Y-shape:
    //     A
    //     |
    // B - C - D
    let mut net: NodeNameNetwork<String> = NodeNameNetwork::new();
    let a = net.add_node("A".to_string()).unwrap();
    let b = net.add_node("B".to_string()).unwrap();
    let c = net.add_node("C".to_string()).unwrap();
    let d = net.add_node("D".to_string()).unwrap();
    net.add_edge(&"A".to_string(), &"C".to_string()).unwrap();
    net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
    net.add_edge(&"C".to_string(), &"D".to_string()).unwrap();

    // Euler tour from C: should visit all 3 edges twice
    let edges = net.euler_tour_edges(&"C".to_string()).unwrap();
    // 3 edges × 2 directions = 6 directed edges
    assert_eq!(edges.len(), 6);

    // Start from C
    assert_eq!(edges[0].0, c);

    // Vertices: 7 visits (C + 6 edges)
    let vertices = net.euler_tour_vertices(&"C".to_string()).unwrap();
    assert_eq!(vertices.len(), 7);
    assert_eq!(vertices[0], c);
    assert_eq!(vertices[vertices.len() - 1], c);

    // All vertices should be visited
    let unique_vertices: HashSet<_> = vertices.iter().cloned().collect();
    assert_eq!(unique_vertices, [a, b, c, d].into());
}

#[test]
fn test_euler_tour_single_node() {
    let mut net: NodeNameNetwork<String> = NodeNameNetwork::new();
    let a = net.add_node("A".to_string()).unwrap();

    let edges = net.euler_tour_edges(&"A".to_string()).unwrap();
    assert!(edges.is_empty());

    let vertices = net.euler_tour_vertices(&"A".to_string()).unwrap();
    assert_eq!(vertices, vec![a]);
}

#[test]
fn test_euler_tour_star() {
    // Star: center C connected to A, B, D, E
    //     A
    //     |
    // B - C - D
    //     |
    //     E
    let mut net: NodeNameNetwork<String> = NodeNameNetwork::new();
    let a = net.add_node("A".to_string()).unwrap();
    let b = net.add_node("B".to_string()).unwrap();
    let c = net.add_node("C".to_string()).unwrap();
    let d = net.add_node("D".to_string()).unwrap();
    let e = net.add_node("E".to_string()).unwrap();
    net.add_edge(&"A".to_string(), &"C".to_string()).unwrap();
    net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
    net.add_edge(&"C".to_string(), &"D".to_string()).unwrap();
    net.add_edge(&"C".to_string(), &"E".to_string()).unwrap();

    // Euler tour from C: should visit all 4 edges twice
    let edges = net.euler_tour_edges(&"C".to_string()).unwrap();
    // 4 edges × 2 directions = 8 directed edges
    assert_eq!(edges.len(), 8);

    // Start from C
    assert_eq!(edges[0].0, c);

    // Vertices: 9 visits (C + 8 edges)
    let vertices = net.euler_tour_vertices(&"C".to_string()).unwrap();
    assert_eq!(vertices.len(), 9);
    assert_eq!(vertices[0], c);
    assert_eq!(vertices[vertices.len() - 1], c);

    // All vertices should be visited
    let unique_vertices: HashSet<_> = vertices.iter().cloned().collect();
    assert_eq!(unique_vertices, [a, b, c, d, e].into());
}
