//! Exercise the public C boundary as a downstream consumer.
use std::ptr::{null, null_mut};
use tensor4all_capi::*;

#[test]
fn cached_evaluator_owns_its_network_after_input_handles_are_released() {
    let mut site = null_mut();
    let mut bond = null_mut();
    let mut other_site = null_mut();
    for out in [&mut site, &mut bond, &mut other_site] {
        assert_eq!(t4a_index_new(2, null(), 0, out), T4A_SUCCESS);
    }
    // A column-major identity joined to [[1, 3], [2, 4]].
    let mut left = null_mut();
    let mut right = null_mut();
    assert_eq!(
        t4a_tensor_new_dense_f64(
            2,
            [site as *const _, bond].as_ptr(),
            [1.0, 0.0, 0.0, 1.0].as_ptr(),
            4,
            &mut left
        ),
        T4A_SUCCESS
    );
    assert_eq!(
        t4a_tensor_new_dense_f64(
            2,
            [bond as *const _, other_site].as_ptr(),
            [1.0, 2.0, 3.0, 4.0].as_ptr(),
            4,
            &mut right
        ),
        T4A_SUCCESS
    );
    let mut tree = null_mut();
    assert_eq!(
        t4a_treetn_new([left as *const _, right].as_ptr(), 2, &mut tree),
        T4A_SUCCESS
    );
    t4a_tensor_release(left);
    t4a_tensor_release(right);
    t4a_index_release(bond);
    assert_eq!(t4a_treetn_is_assigned(tree), 1);

    let mut n = 0;
    assert_eq!(t4a_treetn_num_vertices(tree, &mut n), T4A_SUCCESS);
    assert_eq!(n, 2);
    let mut neighbors = [99usize];
    assert_eq!(
        t4a_treetn_neighbors(tree, 0, neighbors.as_mut_ptr(), 1, &mut n),
        T4A_SUCCESS
    );
    assert_eq!((neighbors, n), ([1], 1));
    let mut sites = [null_mut()];
    assert_eq!(
        t4a_treetn_site_indices(tree, 0, sites.as_mut_ptr(), 1, &mut n),
        T4A_SUCCESS
    );
    assert_eq!(n, 1);
    let mut dim = 0;
    assert_eq!(t4a_index_dim(sites[0], &mut dim), T4A_SUCCESS);
    assert_eq!(dim, 2);
    t4a_index_release(sites[0]);

    let points = [0, 0, 1, 0, 0, 1, 1, 1];
    let indices = [site as *const _, other_site];
    let mut values = [0.0; 4];
    let mut imaginary = [99.0; 4];
    assert_eq!(
        t4a_treetn_evaluate(
            tree,
            indices.as_ptr(),
            2,
            points.as_ptr(),
            4,
            values.as_mut_ptr(),
            imaginary.as_mut_ptr()
        ),
        T4A_SUCCESS
    );
    assert_eq!(values, [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(imaginary, [0.0; 4]);
    let mut norm = 0.0;
    assert_eq!(t4a_treetn_norm(tree, &mut norm), T4A_SUCCESS);
    assert!((norm - 30.0f64.sqrt()).abs() < 1e-12);
    let mut squared_norm = 0.0;
    let mut im = 99.0;
    assert_eq!(
        t4a_treetn_inner(tree, tree, &mut squared_norm, &mut im),
        T4A_SUCCESS
    );
    assert!((squared_norm - 30.0).abs() < 1e-12);
    assert_eq!(im, 0.0);

    let mut evaluator = null_mut();
    assert_eq!(
        t4a_treetn_evaluator_new(tree, indices.as_ptr(), 2, &mut evaluator),
        T4A_SUCCESS
    );
    let mut clone = null_mut();
    assert_eq!(
        t4a_treetn_evaluator_clone(evaluator, &mut clone),
        T4A_SUCCESS
    );
    t4a_treetn_evaluator_release(evaluator);
    t4a_treetn_release(tree);
    t4a_index_release(site);
    t4a_index_release(other_site);
    assert_eq!(t4a_treetn_evaluator_is_assigned(clone), 1);
    // Both calls must remain valid after every source handle has been released.
    for positions in [points, [1, 1, 0, 1, 1, 0, 0, 0]] {
        assert_eq!(
            t4a_treetn_evaluator_evaluate(
                clone,
                positions.as_ptr(),
                4,
                values.as_mut_ptr(),
                imaginary.as_mut_ptr()
            ),
            T4A_SUCCESS
        );
        let expected: Vec<_> = positions
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| 1.0 + p[0] as f64 + 2.0 * p[1] as f64)
            .collect();
        for (actual, expected) in values.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12);
        }
        assert_eq!(imaginary, [0.0; 4]);
    }
    t4a_treetn_evaluator_release(clone);
}
