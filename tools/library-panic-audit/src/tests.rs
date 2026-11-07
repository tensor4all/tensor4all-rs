use super::*;

fn parse(arguments: &[&str]) -> Result<Option<Args>> {
    parse_args_from(arguments.iter().map(|argument| (*argument).to_owned()))
}

#[test]
fn default_audit_has_no_configuration() {
    let args = parse(&[]).unwrap().unwrap();
    assert!(args.configuration.is_none());
    let args = parse(&["--root", "/tmp/root"]).unwrap().unwrap();
    assert!(args.configuration.is_none());
    assert_eq!(args.root, Some(PathBuf::from("/tmp/root")));
}

#[test]
fn configuration_parses_cargo_selection_flags() {
    let args = parse(&[
        "--config-name",
        "system-blas",
        "-p",
        "tensor4all-core",
        "--package",
        "tensor4all-treetn",
        "--no-default-features",
        "--features",
        "tenferro-system-blas",
    ])
    .unwrap()
    .unwrap();
    let configuration = args.configuration.unwrap();
    assert_eq!(configuration.name, "system-blas");
    assert_eq!(
        configuration.packages,
        vec!["tensor4all-core".to_owned(), "tensor4all-treetn".to_owned()]
    );
    assert!(configuration.no_default_features);
    assert_eq!(
        configuration.features.as_deref(),
        Some("tenferro-system-blas")
    );
}

#[test]
fn configuration_flags_without_a_name_are_rejected() {
    for arguments in [
        vec!["--package", "tensor4all-core"],
        vec!["--no-default-features"],
        vec!["--features", "diagnostics"],
    ] {
        let error = parse(&arguments).unwrap_err().to_string();
        assert!(error.contains("require --config-name"), "{error}");
    }
}

#[test]
fn update_baseline_parses_but_rejects_a_configuration_selection() {
    let args = parse(&["--update-baseline"]).unwrap().unwrap();
    assert!(args.update_baseline);
    assert!(!parse(&[]).unwrap().unwrap().update_baseline);
    let error = parse(&[
        "--update-baseline",
        "--config-name",
        "production",
        "--package",
        "demo",
    ])
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("cannot be combined with --config-name"),
        "{error}"
    );
}

#[test]
fn unknown_arguments_still_fail() {
    let error = parse(&["--workspace"]).unwrap_err().to_string();
    assert!(error.contains("unknown argument"), "{error}");
    let error = parse(&["--features", "a", "--features", "b"])
        .unwrap_err()
        .to_string();
    assert!(error.contains("only be given once"), "{error}");
}
