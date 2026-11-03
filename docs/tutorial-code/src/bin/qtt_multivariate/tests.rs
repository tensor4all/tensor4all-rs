use super::*;

#[test]
fn target_function_lives_with_the_tutorial_binary() {
    let target_path = std::any::type_name_of_val(&multivariate_target);
    assert!(
        target_path.contains("qtt_multivariate::multivariate_target"),
        "target path was {target_path}"
    );
}
