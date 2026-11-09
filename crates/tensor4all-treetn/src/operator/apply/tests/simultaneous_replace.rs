use super::*;

fn true_indices(operator: &LinearOperator<IdxTensor, String>, node: &str) -> (DynIndex, DynIndex) {
    let node = node.to_string();
    (
        operator
            .get_input_mapping(&node)
            .unwrap()
            .true_index
            .clone(),
        operator
            .get_output_mapping(&node)
            .unwrap()
            .true_index
            .clone(),
    )
}

#[test]
fn swaps_and_cycles_rename_every_mapping_once() {
    let a = make_index(2);
    let b = make_index(2);
    let c = make_index(2);
    let operator = build_identity_operator(&[
        ("A".to_string(), a.clone()),
        ("B".to_string(), b.clone()),
        ("C".to_string(), c.clone()),
    ]);

    let swapped = operator
        .replace_indices(&[a.clone(), b.clone()], &[b.clone(), a.clone()])
        .unwrap();
    assert_eq!(true_indices(&swapped, "A"), (b.clone(), b.clone()));
    assert_eq!(true_indices(&swapped, "B"), (a.clone(), a.clone()));
    assert_eq!(true_indices(&swapped, "C"), (c.clone(), c.clone()));

    let cycled = operator
        .replace_indices(
            &[a.clone(), b.clone(), c.clone()],
            &[b.clone(), c.clone(), a.clone()],
        )
        .unwrap();
    assert_eq!(true_indices(&cycled, "A"), (b.clone(), b.clone()));
    assert_eq!(true_indices(&cycled, "B"), (c.clone(), c.clone()));
    assert_eq!(true_indices(&cycled, "C"), (a.clone(), a.clone()));

    // The source operator is untouched.
    assert_eq!(true_indices(&operator, "A"), (a.clone(), a));
}

#[test]
fn missing_source_and_dimension_mismatch_are_rejected() {
    let a = make_index(2);
    let b = make_index(2);
    let operator =
        build_identity_operator(&[("A".to_string(), a.clone()), ("B".to_string(), b.clone())]);

    let missing = make_index(2);
    let error = operator
        .replace_indices(&[a.clone(), missing], &[b.clone(), a.clone()])
        .unwrap_err();
    assert!(error.to_string().contains("not found"));

    let error = operator
        .replace_indices(std::slice::from_ref(&a), &[make_index(3)])
        .unwrap_err();
    assert!(error.to_string().contains("Index space mismatch"));

    assert_eq!(true_indices(&operator, "A"), (a.clone(), a));
    assert_eq!(true_indices(&operator, "B"), (b.clone(), b));
}
