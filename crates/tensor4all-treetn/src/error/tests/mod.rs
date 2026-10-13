use super::*;

#[test]
fn numbered_tag_selection_error_messages_are_stable() {
    assert_eq!(
        NumberedTagSelectionError::InvalidPrefix {
            tag_prefix: "k=1".to_string(),
        }
        .to_string(),
        "invalid numbered tag prefix k=1: prefix must not contain '='"
    );
    assert_eq!(
        NumberedTagSelectionError::RangeOverflow {
            start_index: usize::MAX,
            offset: 1,
        }
        .to_string(),
        format!(
            "numbered tag range overflows usize at offset 1 from start {}",
            usize::MAX
        )
    );
    assert_eq!(
        NumberedTagSelectionError::MissingTag {
            tag: "k=3".to_string(),
        }
        .to_string(),
        "no external index with tag k=3"
    );
    assert_eq!(
        NumberedTagSelectionError::AmbiguousTag {
            tag: "k=1".to_string(),
        }
        .to_string(),
        "found more than one external index with tag k=1"
    );
}

#[test]
fn linear_operator_binding_error_messages_are_stable() {
    assert_eq!(
        LinearOperatorIndexBindingError::DimensionMismatch {
            role: "input",
            old_dim: 2,
            new_dim: 3,
        }
        .to_string(),
        "input index shape mismatch: old dimension 2 vs new dimension 3"
    );
    assert_eq!(
        LinearOperatorIndexBindingError::DuplicateSourceIndex {
            role: "output",
            index: "i".to_string(),
        }
        .to_string(),
        "duplicate output index binding for i"
    );
    assert_eq!(
        LinearOperatorIndexBindingError::MissingSourceIndex {
            role: "input",
            index: "j".to_string(),
        }
        .to_string(),
        "input index j not found in operator mappings"
    );
}

#[test]
fn linear_operator_apply_error_messages_are_stable() {
    assert_eq!(
        LinearOperatorIndexApplyError::BindIndices(
            LinearOperatorIndexBindingError::MissingSourceIndex {
                role: "input",
                index: "i".to_string(),
            }
        )
        .to_string(),
        "failed to bind operator indices: input index i not found in operator mappings"
    );
    assert_eq!(
        LinearOperatorIndexApplyError::ApplyFailed {
            message: "contraction failed".to_string(),
        }
        .to_string(),
        "failed to apply rebound linear operator: contraction failed"
    );
}

#[test]
fn tagged_apply_error_messages_are_stable() {
    assert_eq!(
        LinearOperatorTaggedApplyError::SelectTaggedIndices(
            NumberedTagSelectionError::MissingTag {
                tag: "x=2".to_string(),
            }
        )
        .to_string(),
        "failed to select state indices by numbered tag: no external index with tag x=2"
    );
    assert_eq!(
        LinearOperatorTaggedApplyError::MappingCountMismatch {
            input_count: 2,
            output_count: 3,
        }
        .to_string(),
        "operator has 2 input mappings but 3 output mappings"
    );
    assert_eq!(
        LinearOperatorTaggedApplyError::ApplyToIndices(
            LinearOperatorIndexApplyError::ApplyFailed {
                message: "apply failed".to_string(),
            }
        )
        .to_string(),
        "failed to apply operator to tagged indices: failed to apply rebound linear operator: apply failed"
    );
}

#[test]
fn selected_index_contraction_error_messages_are_stable() {
    assert_eq!(
        SelectedIndexContractionError::DuplicateIndex {
            index: "i".to_string(),
        }
        .to_string(),
        "sum_over_indices: duplicate index i"
    );
    assert_eq!(
        SelectedIndexContractionError::IndexInMultipleSiteSpaces {
            index: "i".to_string(),
        }
        .to_string(),
        "sum_over_indices: index i appears in more than one site space"
    );
    assert_eq!(
        SelectedIndexContractionError::IndexNotFound {
            index: "j".to_string(),
        }
        .to_string(),
        "sum_over_indices: index j not found in external indices"
    );
    assert_eq!(
        SelectedIndexContractionError::BuildOnesTensor {
            node: "v0".to_string(),
            message: "shape mismatch".to_string(),
        }
        .to_string(),
        "sum_over_indices: failed to build ones tensor at node v0: shape mismatch"
    );
    assert_eq!(
        SelectedIndexContractionError::BuildWeightsTree {
            message: "invalid tree".to_string(),
        }
        .to_string(),
        "sum_over_indices: failed to build ones TreeTN: invalid tree"
    );
    assert_eq!(
        SelectedIndexContractionError::PartialContractFailed {
            message: "zipup failed".to_string(),
        }
        .to_string(),
        "selected-index contraction failed: zipup failed"
    );
}

#[test]
fn format_anyhow_error_preserves_context_chain() {
    let error = anyhow::anyhow!("root").context("outer context");

    assert_eq!(format_anyhow_error(error), "outer context: root");
}
