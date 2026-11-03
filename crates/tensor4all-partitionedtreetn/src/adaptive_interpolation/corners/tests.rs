use super::*;

fn sites(dims: &[usize]) -> Vec<DynIndex> {
    dims.iter().map(|&dim| DynIndex::new_dyn(dim)).collect()
}

#[test]
fn an_empty_variable_option_uses_one_group_of_all_active_sites() {
    let sites = sites(&[2, 3]);
    let active = [0, 1];
    let dims = [2, 3];
    let starts = corner_and_edge_starts(&sites, &active, &dims, &[], 0);
    // One group: the two corners (all zero, all high) and up to K_EDGE
    // edges, each point once.
    assert_eq!(starts[0], vec![0, 0]);
    assert_eq!(starts[1], vec![1, 2]);
    assert!((2..=2 + K_EDGE).contains(&starts.len()));
    // Every point is a valid point of the patch.
    assert!(starts.iter().all(|p| p[0] < 2 && p[1] < 3));
}

#[test]
fn variable_groups_mix_whole_variables() {
    // Two variables of two bits each: the corners set both bits of a
    // variable together, so the four corners are (0,0), (0,1), (1,0),
    // (1,1) as two-bit coordinates of x = s0 + 2 s1 and y = s2 + 2 s3.
    let sites = sites(&[2, 2, 2, 2]);
    let variables = vec![
        vec![sites[0].clone(), sites[1].clone()],
        vec![sites[2].clone(), sites[3].clone()],
    ];
    let starts = corner_and_edge_starts(&sites, &[0, 1, 2, 3], &[2, 2, 2, 2], &variables, 7);
    assert_eq!(
        &starts[..4],
        &[
            vec![0, 0, 0, 0],
            vec![1, 1, 0, 0],
            vec![0, 0, 1, 1],
            vec![1, 1, 1, 1],
        ]
    );
    // The single group of all four bits (the default) would also contain
    // the mixed corners [1, 0, 0, 0] and [0, 1, 0, 0].
    let one_group = corner_and_edge_starts(&sites, &[0, 1, 2, 3], &[2, 2, 2, 2], &[], 7);
    assert!(one_group.contains(&vec![1, 0, 0, 0]));
    assert!(!starts.contains(&vec![1, 0, 0, 0]));
}

#[test]
fn ungrouped_active_sites_form_one_more_group() {
    let sites = sites(&[2, 2, 2]);
    let variables = vec![vec![sites[0].clone()]];
    let starts = corner_and_edge_starts(&sites, &[0, 1, 2], &[2, 2, 2], &variables, 0);
    // Two groups: site 0 and sites 1, 2 together.
    assert_eq!(starts[0], vec![0, 0, 0]);
    assert!(starts.contains(&vec![0, 1, 1]));
    assert!(starts.contains(&vec![1, 0, 0]));
    assert!(starts.contains(&vec![1, 1, 1]));
}

#[test]
fn the_corner_list_is_a_fixed_function_of_the_stream() {
    // More groups than MAX_CORNERS allows, so the corners are drawn.
    let dims = vec![2usize; 7];
    let sites: Vec<DynIndex> = dims.iter().map(|&dim| DynIndex::new_dyn(dim)).collect();
    let variables: Vec<Vec<DynIndex>> = sites.iter().map(|site| vec![site.clone()]).collect();
    let active: Vec<usize> = (0..7).collect();
    let first = corner_and_edge_starts(&sites, &active, &dims, &variables, 11);
    let second = corner_and_edge_starts(&sites, &active, &dims, &variables, 11);
    assert_eq!(first, second);
    // Duplicate draws collapse, so the list is at most the drawn count.
    assert!(first.len() > MAX_CORNERS / 2);
    assert!(first.len() <= MAX_CORNERS + K_EDGE);
    let other = corner_and_edge_starts(&sites, &active, &dims, &variables, 12);
    assert_ne!(first, other);
    assert!(first.iter().all(|point| point.len() == 7));
}
