//! Exact identity skips preserve tensors but require input certificates (#1290).

use super::*;
use crate::{CanonicalForm, CanonicalizationOptions};

#[test]
fn apply_zipup_identity_skip_claims_a_center_only_for_matching_input_certificates() {
    use crate::treetn::contraction::zipup_identity_node_skips;

    // Uncanonicalized; matching; opposite direction; a different form; and a
    // missing form or direction despite the remaining input metadata. The
    // last case has a certificate for a same-ID bond at another prime level.
    for case in 0..8 {
        let (mut state, operator) = build_four_node_star();
        if case != 0 {
            let center = if case == 2 { "b" } else { "r" };
            let form = match case {
                3 => CanonicalForm::LU,
                4 => CanonicalForm::CI,
                _ => CanonicalForm::Unitary,
            };
            state = state
                .canonicalize(
                    [center.to_string()],
                    CanonicalizationOptions::default().with_form(form),
                )
                .unwrap();
            state.validate_ortho_consistency().unwrap();
            if case == 5 {
                state.set_canonical_form(None);
            }
            if case >= 6 {
                let edge = state
                    .edge_between(&"b".to_string(), &"x".to_string())
                    .unwrap();
                let bond = state.bond_index(edge).unwrap().clone();
                state.set_edge_ortho_towards(edge, None).unwrap();
                if case == 7 {
                    let primed = bond.prime();
                    assert_eq!(primed.id(), bond.id());
                    assert_ne!(primed, bond);
                    state.set_ortho_towards(&primed, Some("x".to_string()));
                }
            }
        }
        let before = zipup_identity_node_skips();
        let result = apply_linear_operator(
            &operator,
            &state,
            ApplyOptions {
                center: Some("r".to_string()),
                svd_policy: Some(SvdTruncationPolicy::new(0.0)),
                ..ApplyOptions::zipup()
            },
        )
        .unwrap();
        assert_eq!(zipup_identity_node_skips() - before, 1);
        assert_node_passed_through(&result, &state, "b");
        if case == 1 {
            assert_eq!(
                result.canonical_region(),
                &["r".to_string()].into_iter().collect()
            );
            assert_eq!(result.canonical_form(), Some(CanonicalForm::Unitary));
            let edge = result
                .edge_between(&"b".to_string(), &"x".to_string())
                .unwrap();
            assert_eq!(result.ortho_towards_node(edge), Some(&"x".to_string()));
        } else {
            assert!(
                result.canonical_region().is_empty(),
                "case {case} claimed a center"
            );
            assert_eq!(result.canonical_form(), None);
            assert!(
                result.ortho_towards.is_empty(),
                "case {case} claimed an orientation"
            );
        }
        result.validate_ortho_consistency().unwrap();
        let naive = apply_linear_operator(&operator, &state, ApplyOptions::naive()).unwrap();
        let deviation = dense_deviation(&result, &naive);
        assert!(
            deviation <= 1e-12,
            "case {case}: identity apply deviation {deviation:e}"
        );
    }
}

#[test]
fn apply_zipup_mixed_identity_certificates_leave_all_canonical_metadata_unset() {
    use crate::treetn::contraction::zipup_identity_node_skips;

    let (state, sites) = build_comb_state();
    // The b branch points away from the requested hub; the d branch still
    // points towards it. An uncertified skip must not be masked by later ones.
    let state = state
        .canonicalize(["b1".to_string()], Default::default())
        .unwrap();
    state.validate_ortho_consistency().unwrap();
    let operator = build_branch_operator(&sites);
    let before = zipup_identity_node_skips();
    let result = apply_linear_operator(
        &operator,
        &state,
        ApplyOptions {
            center: Some("hub".to_string()),
            svd_policy: Some(SvdTruncationPolicy::new(0.0)),
            ..ApplyOptions::zipup()
        },
    )
    .unwrap();
    assert_eq!(zipup_identity_node_skips() - before, 4);
    for name in ["b0", "b1", "d0", "d1"] {
        assert_node_passed_through(&result, &state, name);
    }
    assert!(result.canonical_region().is_empty());
    assert_eq!(result.canonical_form(), None);
    assert!(result.ortho_towards.is_empty());
    result.validate_ortho_consistency().unwrap();
    let naive = apply_linear_operator(&operator, &state, ApplyOptions::naive()).unwrap();
    let deviation = dense_deviation(&result, &naive);
    assert!(
        deviation < 1e-12,
        "mixed-certificate identity apply deviation {deviation:e}"
    );
}
