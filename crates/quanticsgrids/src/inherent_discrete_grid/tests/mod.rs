use super::*;

#[test]
fn test_basic_1d_grid() {
    let grid = InherentDiscreteGrid::builder(&[3]).build().unwrap();
    assert_eq!(grid.ndims(), 1);
    assert_eq!(grid.len(), 3); // 3 sites for fused scheme
    assert_eq!(grid.base(), 2);
    assert_eq!(grid.rs(), &[3]);
    assert_eq!(grid.max_grididx(), &[8]);
}

#[test]
fn test_basic_2d_grid() {
    let grid = InherentDiscreteGrid::builder(&[3, 2])
        .with_variable_names(&["x", "y"])
        .build()
        .unwrap();

    assert_eq!(grid.ndims(), 2);
    assert_eq!(grid.variable_names(), &["x", "y"]);
    assert_eq!(grid.max_grididx(), &[8, 4]);
}

#[test]
fn test_grididx_to_quantics_roundtrip() {
    let grid = InherentDiscreteGrid::builder(&[3, 2])
        .with_variable_names(&["a", "b"])
        .build()
        .unwrap();

    let grididx = vec![5i64, 2];
    let quantics = grid.grididx_to_quantics(&grididx).unwrap();
    let back = grid.quantics_to_grididx(&quantics).unwrap();
    assert_eq!(back, grididx);
}

#[test]
fn test_all_grididx_roundtrip() {
    let grid = InherentDiscreteGrid::builder(&[2, 2]).build().unwrap();

    for x in 1..=4 {
        for y in 1..=4 {
            let grididx = vec![x, y];
            let quantics = grid.grididx_to_quantics(&grididx).unwrap();
            let back = grid.quantics_to_grididx(&quantics).unwrap();
            assert_eq!(back, grididx, "Failed for grididx {:?}", grididx);
        }
    }
}

#[test]
fn test_interleaved_scheme() {
    let grid = InherentDiscreteGrid::builder(&[2, 2])
        .with_unfolding_scheme(UnfoldingScheme::Interleaved)
        .build()
        .unwrap();

    // Interleaved: [1_1], [2_1], [1_2], [2_2]
    assert_eq!(grid.len(), 4);
    for x in 1..=4 {
        for y in 1..=4 {
            let grididx = vec![x, y];
            let quantics = grid.grididx_to_quantics(&grididx).unwrap();
            let back = grid.quantics_to_grididx(&quantics).unwrap();
            assert_eq!(back, grididx);
        }
    }
}

#[test]
fn test_base3_grid() {
    let grid = InherentDiscreteGrid::builder(&[2])
        .with_base(3)
        .build()
        .unwrap();

    assert_eq!(grid.base(), 3);
    assert_eq!(grid.max_grididx(), &[9]); // 3^2 = 9

    for x in 1..=9 {
        let grididx = vec![x];
        let quantics = grid.grididx_to_quantics(&grididx).unwrap();
        let back = grid.quantics_to_grididx(&quantics).unwrap();
        assert_eq!(back, grididx);
    }
}

#[test]
fn test_origcoord_conversion() {
    let grid = InherentDiscreteGrid::builder(&[2])
        .with_origin(&[0])
        .with_step(&[1])
        .build()
        .unwrap();

    // origin=0, step=1, max_grididx=4
    // grididx 1 -> origcoord 0
    // grididx 4 -> origcoord 3
    let coord = grid.grididx_to_origcoord(&[1]).unwrap();
    assert_eq!(coord, vec![0]);

    let coord = grid.grididx_to_origcoord(&[4]).unwrap();
    assert_eq!(coord, vec![3]);

    let idx = grid.origcoord_to_grididx(&[2]).unwrap();
    assert_eq!(idx, vec![3]);
}

#[test]
fn test_error_invalid_base() {
    let result = InherentDiscreteGrid::builder(&[3]).with_base(1).build();
    assert!(matches!(result, Err(QuanticsGridError::InvalidBase(1))));
}

#[test]
fn test_error_duplicate_variable_names() {
    let result = InherentDiscreteGrid::builder(&[2, 2])
        .with_variable_names(&["x", "x"])
        .build();
    assert!(matches!(
        result,
        Err(QuanticsGridError::DuplicateVariableName(_))
    ));
}

#[test]
fn test_error_quantics_out_of_range() {
    let grid = InherentDiscreteGrid::builder(&[2]).build().unwrap();
    // Rs=[2] with Fused creates 2 sites, each with dim 2
    // So quantics should have length 2, and each value should be in [1, 2]
    let result = grid.quantics_to_grididx(&[5, 1]); // 5 is out of range [1, 2]
    assert!(matches!(
        result,
        Err(QuanticsGridError::QuanticsOutOfRange { .. })
    ));
}

#[test]
fn test_error_grididx_out_of_bounds() {
    let grid = InherentDiscreteGrid::builder(&[2]).build().unwrap();
    let result = grid.grididx_to_quantics(&[5]); // max is 4
    assert!(matches!(
        result,
        Err(QuanticsGridError::GridIndexOutOfBounds { .. })
    ));
}

#[test]
fn test_local_dimensions() {
    let grid = InherentDiscreteGrid::builder(&[3, 2])
        .with_unfolding_scheme(UnfoldingScheme::Fused)
        .build()
        .unwrap();

    let dims = grid.local_dimensions();
    // Fused scheme with Rs=(3,2): sites have 2, 2, 1 indices each -> dims 4, 4, 2
    // Actually: bitnumber 0: [b, a] -> dim 4
    //           bitnumber 1: [b, a] -> dim 4
    //           bitnumber 2: [a] -> dim 2
    assert_eq!(dims, vec![4, 4, 2]);
}

#[test]
fn test_from_index_table() {
    // Create a custom index table like in Julia: [[(:a, 1), (:b, 2)], [(:a, 2)], [(:b, 1), (:a, 3)]]
    let index_table = vec![
        vec![("a".to_string(), 1), ("b".to_string(), 2)],
        vec![("a".to_string(), 2)],
        vec![("b".to_string(), 1), ("a".to_string(), 3)],
    ];

    let grid = InherentDiscreteGrid::from_index_table(&["a", "b"], index_table)
        .build()
        .unwrap();

    assert_eq!(grid.ndims(), 2);
    assert_eq!(grid.rs(), &[3, 2]); // a has 3 bits, b has 2 bits
    assert_eq!(grid.len(), 3);

    // Test roundtrip
    for x in 1..=8 {
        for y in 1..=4 {
            let grididx = vec![x, y];
            let quantics = grid.grididx_to_quantics(&grididx).unwrap();
            let back = grid.quantics_to_grididx(&quantics).unwrap();
            assert_eq!(back, grididx);
        }
    }
}
