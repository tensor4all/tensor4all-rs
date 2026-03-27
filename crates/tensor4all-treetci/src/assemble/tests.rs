use super::{assemble_global_point, assemble_points_column_major};
use crate::SubtreeKey;

#[test]
fn assemble_global_point_places_subtree_and_central_values_in_site_order() {
    let point = assemble_global_point(
        7,
        &[
            (&SubtreeKey::new(vec![0, 1, 2]), &vec![10, 11, 12]),
            (&SubtreeKey::new(vec![4, 6]), &vec![14, 16]),
        ],
        &[(3, 13), (5, 15)],
    )
    .unwrap();

    assert_eq!(point, vec![10, 11, 12, 13, 14, 15, 16]);
}

#[test]
fn assemble_points_column_major_packs_points_with_site_dimension_leading() {
    let batch = assemble_points_column_major(&[
        vec![0, 10, 20, 30],
        vec![1, 11, 21, 31],
        vec![2, 12, 22, 32],
    ])
    .unwrap();
    let view = batch.as_view();

    assert_eq!(view.n_sites(), 4);
    assert_eq!(view.n_points(), 3);
    assert_eq!(view.data(), &[0, 10, 20, 30, 1, 11, 21, 31, 2, 12, 22, 32]);
    assert_eq!(view.get(0, 2), Some(2));
    assert_eq!(view.get(3, 1), Some(31));
}
