use super::BOOK_INTRODUCTION;

#[test]
fn introduction_uses_repo_relative_rustdoc_link() {
    assert!(
        BOOK_INTRODUCTION.contains("[rustdoc API reference](rustdoc/tensor4all_core/)"),
        "top-level mdBook page must link to rustdoc within the published /tensor4all-rs/ site"
    );
    assert!(
        !BOOK_INTRODUCTION.contains("[rustdoc API reference](../rustdoc/tensor4all_core/)"),
        "top-level mdBook page must not climb above the published site root"
    );
}
