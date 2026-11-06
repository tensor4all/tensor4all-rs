use super::*;

#[test]
fn square_site_mapping_errors_convert_to_tdvp_errors() {
    let err = TdvpError::from(SquareSiteMappingError::UnsupportedStateSiteCount {
        node: "site0".to_string(),
        count: 2,
    });
    assert!(matches!(
        err,
        TdvpError::UnsupportedStateSiteCount { node, count }
            if node == "site0" && count == 2
    ));

    let err = TdvpError::from(SquareSiteMappingError::UnsupportedMultipleSiteMappings {
        node: "site1".to_string(),
        role: "input",
        count: 3,
    });
    assert!(matches!(
        err,
        TdvpError::UnsupportedMultipleSiteMappings { node, role, count }
            if node == "site1" && role == "input" && count == 3
    ));

    let err = TdvpError::from(SquareSiteMappingError::MissingMapping {
        node: "site2".to_string(),
        role: "output",
    });
    assert!(matches!(
        err,
        TdvpError::MissingMapping { node, role }
            if node == "site2" && role == "output"
    ));

    let err = TdvpError::from(SquareSiteMappingError::InvalidMapping {
        node: "site3".to_string(),
        reason: "shape mismatch".to_string(),
    });
    assert!(matches!(
        err,
        TdvpError::InvalidMapping { node, reason }
            if node == "site3" && reason == "shape mismatch"
    ));
}
