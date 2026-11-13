use super::*;
#[test]
fn prefix_projection_has_known_values_and_an_unchanged_identity_buffer() {
    assert_eq!(
        CandidateBasisProjection::new(&[3, 3], &[2, 2])
            .unwrap()
            .project((1..=9).map(|x| x as f64).collect())
            .unwrap(),
        vec![1., 2., 4., 5.]
    );
    let values = vec![1., 2., 3., 4.];
    let allocation = values.as_ptr();
    let same = CandidateBasisProjection::new(&[2, 2], &[2, 2])
        .unwrap()
        .project(values)
        .unwrap();
    assert_eq!(same.as_ptr(), allocation);
    assert_eq!(
        CandidateBasisProjection::new(&[], &[])
            .unwrap()
            .project(vec![7.])
            .unwrap(),
        vec![7.]
    );
    assert!(CandidateBasisProjection::new(&[2], &[0])
        .unwrap()
        .project(vec![1., 2.])
        .unwrap()
        .is_empty());
}
#[test]
fn small_three_axis_projections_match_independent_column_major_coordinates() {
    for x in 1..=3 {
        for y in 1..=3 {
            for z in 1..=3 {
                let basis = [x, y, z];
                for mask in 0..8 {
                    let source = [x + (mask & 1), y + ((mask >> 1) & 1), z + ((mask >> 2) & 1)];
                    let values: Vec<f64> =
                        (1..=source.iter().product()).map(|x| x as f64).collect();
                    let mut expected = Vec::new();
                    for k in 0..z {
                        for j in 0..y {
                            for i in 0..x {
                                expected.push(values[i + source[0] * (j + source[1] * k)]);
                            }
                        }
                    }
                    assert_eq!(
                        CandidateBasisProjection::new(&source, &basis)
                            .unwrap()
                            .project(values)
                            .unwrap(),
                        expected
                    );
                }
            }
        }
    }
}
#[test]
fn invalid_extents_and_lengths_are_rejected_before_copying() {
    assert!(CandidateBasisProjection::new(&[1], &[2]).is_err());
    assert!(CandidateBasisProjection::new(&[1], &[1, 1]).is_err());
    assert!(CandidateBasisProjection::new(&[usize::MAX, 2], &[1, 1]).is_err());
    assert!(CandidateBasisProjection::new(&[2], &[1])
        .unwrap()
        .project(vec![1.])
        .is_err());
}
