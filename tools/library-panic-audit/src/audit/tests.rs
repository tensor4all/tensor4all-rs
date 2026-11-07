use super::*;

#[test]
fn configuration_requires_a_name_and_a_package() {
    let error = Configuration::new(String::new(), vec!["demo".to_owned()], false, None)
        .unwrap_err()
        .to_string();
    assert!(error.contains("--config-name must not be empty"), "{error}");
    let error = Configuration::new("system-blas".to_owned(), Vec::new(), false, None)
        .unwrap_err()
        .to_string();
    assert!(error.contains("requires at least one --package"), "{error}");
    let error = Configuration::new("system-blas".to_owned(), vec![" ".to_owned()], false, None)
        .unwrap_err()
        .to_string();
    assert!(error.contains("--package must not be empty"), "{error}");
}

#[test]
fn configuration_rejects_empty_features_and_keeps_explicit_selection() {
    let error = Configuration::new(
        "diagnostics".to_owned(),
        vec!["tensor4all-treetn".to_owned()],
        true,
        Some(" ".to_owned()),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("--features must list"), "{error}");

    let configuration = Configuration::new(
        "system-blas".to_owned(),
        vec!["tensor4all-core".to_owned(), "tensor4all-treetn".to_owned()],
        true,
        Some("tenferro-system-blas".to_owned()),
    )
    .unwrap();
    assert_eq!(
        configuration.label(),
        "configuration `system-blas` --no-default-features \
         --features tenferro-system-blas -p tensor4all-core -p tensor4all-treetn"
    );
}
use serde_json::json;
use std::time::{SystemTime, UNIX_EPOCH};

struct TempRoot {
    path: PathBuf,
}

impl TempRoot {
    fn new() -> Self {
        // Tests run in parallel, so the clock alone is not a unique name.
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before epoch")
            .as_nanos();
        let sequence = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "t4a-panic-audit-{}-{nonce}-{sequence}",
            std::process::id()
        ));
        fs::create_dir_all(path.join("crates/demo/src")).expect("create fixture root");
        fs::write(path.join("crates/demo/src/lib.rs"), "pub fn demo() {}\n")
            .expect("write fixture source");
        // The audit compares canonical paths, and the temporary directory
        // can sit behind a symlink (macOS: /var -> /private/var).
        let path = fs::canonicalize(&path).expect("canonicalize fixture root");
        Self { path }
    }
}

impl Drop for TempRoot {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn diagnostic(code: &str, span: Value) -> Value {
    json!({
        "reason": "compiler-message",
        "message": {
            "code": {"code": code},
            "spans": [span]
        }
    })
}

fn local_span(file_name: &str, line: u64) -> Value {
    json!({
        "file_name": file_name,
        "line_start": line,
        "line_end": line,
        "is_primary": true
    })
}

fn artifact() -> Value {
    json!({
        "reason": "compiler-artifact",
        "package_id": "pkg",
        "target": {"name": "demo", "kind": ["cdylib", "rlib"]},
        "profile": {"opt_level": "0"},
        "filenames": ["target/debug/deps/libdemo-hash.rmeta"]
    })
}

#[test]
fn cargo_json_boundary_is_strict_and_tracks_lib_like_targets() {
    let root = TempRoot::new();
    let expected = HashSet::from([TargetKey {
        package_id: "pkg".to_owned(),
        target_name: "demo".to_owned(),
        kind: "cdylib".to_owned(),
    }]);
    let finished = json!({"reason": "build-finished", "success": true});
    let output = format!("{}\n{}\n", artifact(), finished);
    let (seen, artifacts, findings) =
        parse_clippy_output(output.as_bytes(), &root.path, &expected).expect("valid Cargo JSON");
    assert_eq!(seen, expected);
    assert_eq!(artifacts.len(), 1);
    assert!(findings.is_empty());

    for malformed in [
        b"\xff".as_slice(),
        b"null\n".as_slice(),
        b"{}\n".as_slice(),
        b"{\"reason\":null}\n".as_slice(),
        b"{\"reason\":\"unknown\"}\n".as_slice(),
        b"{\"reason\":\"build-finished\",\"success\":null}\n".as_slice(),
    ] {
        assert!(parse_clippy_output(malformed, &root.path, &HashSet::new()).is_err());
    }

    let missing_message =
        br#"{"reason":"compiler-message","package_id":"pkg","target":{"name":"demo","kind":["cdylib"]},"message":null}
"#;
    assert!(parse_clippy_output(missing_message, &root.path, &HashSet::new()).is_err());
    let missing_finished = serde_json::to_vec(&artifact()).expect("serialize artifact");
    assert!(parse_clippy_output(&missing_finished, &root.path, &expected).is_err());
}

#[test]
fn compiler_json_accepts_only_exact_codes() {
    let root = TempRoot::new();
    let message = diagnostic(
        "clippy::unwrap_used",
        local_span("crates/demo/src/lib.rs", 1),
    );
    let finding = parse_compiler_message(&message, &root.path)
        .expect("valid diagnostic")
        .expect("known code");
    assert_eq!(finding.entry(), "crates/demo/src/lib.rs:1:unwrap");
    let unknown = diagnostic("clippy::unwrap", local_span("crates/demo/src/lib.rs", 1));
    assert!(parse_compiler_message(&unknown, &root.path).is_err());
}

#[test]
fn compiler_json_follows_expansion_to_local_call_site() {
    let root = TempRoot::new();
    let message = diagnostic(
        "clippy::panic",
        json!({
            "file_name": "/rustc/library/std/src/panic.rs",
            "line_start": 7,
            "is_primary": true,
            "expansion": {
                "span": local_span("crates/demo/src/lib.rs", 1)
            }
        }),
    );
    let finding = parse_compiler_message(&message, &root.path)
        .expect("valid expansion")
        .expect("known code");
    assert_eq!(finding.entry(), "crates/demo/src/lib.rs:1:panic");
}

#[test]
fn compiler_json_rejects_missing_or_outside_local_span() {
    let root = TempRoot::new();
    for span in [
        json!({"file_name": "crates/demo/src/lib.rs", "is_primary": false}),
        local_span("/tmp/outside.rs", 1),
    ] {
        let message = diagnostic("clippy::expect_used", span);
        assert!(parse_compiler_message(&message, &root.path).is_err());
    }
}

#[test]
fn dep_info_parser_handles_make_escapes_and_filters_external_sources() {
    let root = TempRoot::new();
    let spaced = root.path.join("crates/demo/src/with space.rs");
    let hashed = root.path.join("crates/demo/src/with#hash.rs");
    fs::write(&spaced, "pub fn spaced() {}\n").expect("write spaced source");
    fs::write(&hashed, "pub fn hashed() {}\n").expect("write hashed source");
    let dep_info = root.path.join("demo.d");
    fs::write(
        &dep_info,
        "demo: crates/demo/src/lib.rs crates/demo/src/with\\ space.rs \\\n crates/demo/src/with#hash.rs /outside/external.rs Cargo.toml\n",
    )
    .expect("write dep-info");
    let parsed = parse_make_dep_info(&dep_info, &root.path).expect("parse dep-info");
    let sources = parsed.all_sources();
    assert_eq!(sources.len(), 3);
    assert!(sources.contains(&fs::canonicalize(spaced).unwrap()));
    assert!(sources.contains(&fs::canonicalize(hashed).unwrap()));
}

#[test]
fn dep_info_parser_handles_rustc_encoded_paths() {
    let root = TempRoot::new();
    let dollar = root.path.join("crates/demo/src/with$dollar.rs");
    let colon = root.path.join("crates/demo/src/with:colon.rs");
    let hashed = root.path.join("crates/demo/src/with#hash.rs");
    for path in [&dollar, &colon, &hashed] {
        fs::write(path, "pub fn selected() {}\n").expect("write selected source");
    }
    let dep_info = root.path.join("demo.d");
    // This is rustc's Make encoding: '$' is written as '$$', while '#'
    // is an ordinary path character and spaces/colons are backslash escaped.
    fs::write(
        &dep_info,
        "demo: crates/demo/src/with$$dollar.rs crates/demo/src/with\\:colon.rs crates/demo/src/with#hash.rs\\\n",
    )
    .expect("write rustc-style dep-info");
    let parsed = parse_make_dep_info(&dep_info, &root.path).expect("parse dep-info");
    let sources = parsed.all_sources();
    assert!(sources.contains(&fs::canonicalize(dollar).unwrap()));
    assert!(sources.contains(&fs::canonicalize(colon).unwrap()));
    assert!(sources.contains(&fs::canonicalize(hashed).unwrap()));
}

#[test]
fn dep_info_continuation_separates_adjacent_paths_without_spaces() {
    let root = TempRoot::new();
    let first = root.path.join("crates/demo/src/first.rs");
    let second = root.path.join("crates/demo/src/second.rs");
    fs::write(&first, "pub fn first() {}\n").expect("write first source");
    fs::write(&second, "pub fn second() {}\n").expect("write second source");
    let dep_info = root.path.join("demo.d");
    fs::write(
        &dep_info,
        "target/debug/deps/libdemo.rmeta: crates/demo/src/first.rs\\\ncrates/demo/src/second.rs\n",
    )
    .expect("write continuation dep-info");

    let parsed = parse_make_dep_info(&dep_info, &root.path).expect("parse dep-info");
    assert_eq!(
        parsed.all_sources(),
        BTreeSet::from([
            fs::canonicalize(first).unwrap(),
            fs::canonicalize(second).unwrap(),
        ])
    );
}

#[test]
fn dep_info_candidates_cover_hashed_artifact_names() {
    let root = TempRoot::new();
    for artifact_name in [
        "libdemo-hash.rlib",
        "libdemo-hash.rmeta",
        "libdemo-hash.so",
        "demo-hash",
        "libdemo-hash.dll",
    ] {
        let artifact = root.path.join(artifact_name);
        let candidates = dep_info_candidates(&artifact);
        assert!(candidates.iter().any(|candidate| {
            candidate.file_name().and_then(|name| name.to_str()) == Some("demo-hash.d")
        }));
    }
}

#[test]
fn dep_info_lookup_rejects_stale_heuristic_targets() {
    let root = TempRoot::new();
    let stale_source = root.path.join("crates/demo/src/stale.rs");
    let valid_source = root.path.join("crates/demo/src/valid.rs");
    fs::write(&stale_source, "pub fn stale() {}\n").expect("write stale source");
    fs::write(&valid_source, "pub fn valid() {}\n").expect("write valid source");
    let deps = root.path.join("target/debug/deps");
    fs::create_dir_all(&deps).expect("create target deps");
    fs::write(
        deps.join("demo-hash.d"),
        "target/debug/deps/demo-hash.d: crates/demo/src/stale.rs\n",
    )
    .expect("write stale self-rule dep-info");
    fs::write(
        deps.join("libdemo-hash.d"),
        "target/debug/deps/libdemo-hash.rlib: crates/demo/src/valid.rs\n",
    )
    .expect("write valid artifact dep-info");
    let key = TargetKey {
        package_id: "pkg".to_owned(),
        target_name: "demo".to_owned(),
        kind: "rlib".to_owned(),
    };
    let artifacts = vec![CompilerArtifact {
        key: key.clone(),
        filenames: BTreeSet::from([PathBuf::from("target/debug/deps/libdemo-hash.rlib")]),
        target: json!({"name": "demo", "kind": ["rlib"], "src_path": "crates/demo/src/valid.rs"}),
        profile: json!({"opt_level": "0"}),
    }];
    let sources = dep_info_sources(&root.path, &artifacts, &HashSet::from([key]))
        .expect("select matching dep-info");
    assert!(sources.contains(&fs::canonicalize(valid_source).unwrap()));
    assert!(!sources.contains(&fs::canonicalize(stale_source).unwrap()));
}

#[test]
fn dep_info_rejects_artifact_output_without_its_own_sources() {
    // The rule whose output matches the artifact carries no sources, and
    // the real origins live only on an unrelated self-rule. The unrelated
    // rule must not satisfy the artifact's source validation.
    let root = TempRoot::new();
    let unrelated = root.path.join("crates/demo/src/unrelated.rs");
    fs::write(&unrelated, "pub fn unrelated() {}\n").expect("write unrelated source");
    let deps = root.path.join("target/debug/deps");
    fs::create_dir_all(&deps).expect("create target deps");
    fs::write(
        deps.join("libdemo-hash.d"),
        "target/debug/deps/libdemo-hash.rlib:\ntarget/debug/deps/demo-hash.d: crates/demo/src/unrelated.rs\n",
    )
    .expect("write flattened dep-info");
    let key = TargetKey {
        package_id: "pkg".to_owned(),
        target_name: "demo".to_owned(),
        kind: "rlib".to_owned(),
    };
    let artifacts = vec![CompilerArtifact {
        key: key.clone(),
        filenames: BTreeSet::from([PathBuf::from("target/debug/deps/libdemo-hash.rlib")]),
        target: json!({"name": "demo", "kind": ["rlib"], "src_path": "crates/demo/src/lib.rs"}),
        profile: json!({"opt_level": "0"}),
    }];
    let error = dep_info_sources(&root.path, &artifacts, &HashSet::from([key]))
        .expect_err("artifact rule with no sources must fail closed");
    assert!(error
        .to_string()
        .contains("contained no workspace-local Rust sources"));
}

#[test]
fn dep_info_lookup_requires_artifact_source_in_selected_sources() {
    let root = TempRoot::new();
    let selected = root.path.join("crates/demo/src/selected.rs");
    fs::write(&selected, "pub fn selected() {}\n").expect("write selected source");
    let deps = root.path.join("target/debug/deps");
    fs::create_dir_all(&deps).expect("create target deps");
    fs::write(
        deps.join("libdemo-hash.d"),
        "target/debug/deps/libdemo-hash.rlib: crates/demo/src/selected.rs\n",
    )
    .expect("write dep-info");
    let key = TargetKey {
        package_id: "pkg".to_owned(),
        target_name: "demo".to_owned(),
        kind: "rlib".to_owned(),
    };
    let artifacts = vec![CompilerArtifact {
        key: key.clone(),
        filenames: BTreeSet::from([PathBuf::from("target/debug/deps/libdemo-hash.rlib")]),
        target: json!({"name": "demo", "kind": ["rlib"], "src_path": "crates/demo/src/lib.rs"}),
        profile: json!({"opt_level": "0"}),
    }];
    let error = dep_info_sources(&root.path, &artifacts, &HashSet::from([key]))
        .expect_err("missing artifact source must fail closed");
    assert!(error
        .to_string()
        .contains("dep-info sources omitted compiler artifact source"));
}

#[test]
fn cargo_json_preserves_each_artifact_record() {
    let root = TempRoot::new();
    let expected = HashSet::from([TargetKey {
        package_id: "pkg".to_owned(),
        target_name: "demo".to_owned(),
        kind: "cdylib".to_owned(),
    }]);
    let first = json!({
        "reason": "compiler-artifact",
        "package_id": "pkg",
        "target": {"name": "demo", "kind": ["cdylib", "rlib"], "src_path": "crates/demo/src/lib.rs"},
        "profile": {"opt_level": "0"},
        "filenames": ["target/debug/deps/libdemo-debug.rmeta"]
    });
    let second = json!({
        "reason": "compiler-artifact",
        "package_id": "pkg",
        "target": {"name": "demo", "kind": ["cdylib", "rlib"], "src_path": "crates/demo/src/lib.rs"},
        "profile": {"opt_level": "3"},
        "filenames": ["target/release/deps/libdemo-release.rmeta"]
    });
    let artifact_source = root.path.join("crates/demo/src/lib.rs");
    let debug_source = root.path.join("crates/demo/src/debug.rs");
    let release_source = root.path.join("crates/demo/src/release.rs");
    fs::write(&artifact_source, "pub fn artifact() {}\n").expect("write artifact source");
    fs::write(&debug_source, "pub fn debug() {}\n").expect("write debug source");
    fs::write(&release_source, "pub fn release() {}\n").expect("write release source");
    fs::create_dir_all(root.path.join("target/debug/deps")).expect("create debug deps");
    fs::create_dir_all(root.path.join("target/release/deps")).expect("create release deps");
    fs::write(
        root.path.join("target/debug/deps/libdemo-debug.d"),
        "target/debug/deps/libdemo-debug.rmeta: crates/demo/src/debug.rs crates/demo/src/lib.rs\n",
    )
    .expect("write debug dep-info");
    fs::write(
        root.path.join("target/release/deps/libdemo-release.d"),
        "target/release/deps/libdemo-release.rmeta: crates/demo/src/release.rs crates/demo/src/lib.rs\n",
    )
    .expect("write release dep-info");
    let output = format!(
        "{}\n{}\n{}\n",
        first,
        second,
        json!({"reason": "build-finished", "success": true})
    );
    let (_, artifacts, _) = parse_clippy_output(output.as_bytes(), &root.path, &expected)
        .expect("valid duplicate-key artifact records");
    assert_eq!(artifacts.len(), 2);
    let sources =
        dep_info_sources(&root.path, &artifacts, &expected).expect("union dep-info sources");
    assert!(sources.contains(&fs::canonicalize(debug_source).unwrap()));
    assert!(sources.contains(&fs::canonicalize(release_source).unwrap()));
}

#[test]
fn dep_info_parser_and_artifact_lookup_fail_closed() {
    let root = TempRoot::new();
    let malformed = root.path.join("malformed.d");
    fs::write(&malformed, b"target crates/demo/src/lib.rs\n").expect("write malformed dep-info");
    assert!(parse_make_dep_info(&malformed, &root.path).is_err());
    fs::write(&malformed, b"target: crates/demo/src/lib.rs \\").expect("write dangling escape");
    assert!(parse_make_dep_info(&malformed, &root.path).is_err());

    let key = TargetKey {
        package_id: "pkg".to_owned(),
        target_name: "demo".to_owned(),
        kind: "rlib".to_owned(),
    };
    let artifacts = vec![CompilerArtifact {
        key: key.clone(),
        filenames: BTreeSet::from([PathBuf::from("target/debug/deps/libdemo-missing.rmeta")]),
        target: json!({"name": "demo", "kind": ["rlib"], "src_path": "crates/demo/src/lib.rs"}),
        profile: json!({"opt_level": "0"}),
    }];
    let expected = HashSet::from([key]);
    let error = dep_info_sources(&root.path, &artifacts, &expected).unwrap_err();
    assert!(error.to_string().contains("no dep-info"));
}

#[test]
fn assertion_scan_handles_fragments_and_macro_tokens() {
    let root = TempRoot::new();
    let parsed = root.path.join("crates/demo/src/lib.rs");
    let fragment = root.path.join("crates/demo/src/fragment.rs");
    let source = r#"macro_rules! matcher_only {
(assert!($($argument:tt)*)) => { () };
}
macro_rules! transcriber {
($assert:ident) => {
    $assert!();
    assert!(true);
};
}
macro_rules! passthrough { ($value:expr) => { $value }; }
pub fn public() { passthrough!(assert!(true)); }
macro_rules! nested_outer {
() => {
    macro_rules! nested_inner {
        (assert!($value:expr)) => { assert!($value); };
    }
};
}
pub fn nested_use() { assert!(true); }
"#;
    fs::write(&parsed, source).expect("write parsed source");
    fs::write(&fragment, "let value = assert!(true);\n").expect("write included fragment");
    let sources = BTreeSet::from([
        fs::canonicalize(parsed).unwrap(),
        fs::canonicalize(fragment).unwrap(),
    ]);
    let findings = scan_sources(&root.path, &sources)
        .expect("scan selected sources")
        .assertions;
    let parsed_findings = findings
        .iter()
        .filter(|finding| finding.path.ends_with("lib.rs"))
        .map(Location::entry)
        .collect::<Vec<_>>();
    assert_eq!(
        parsed_findings,
        vec![
            "crates/demo/src/lib.rs:7:assert",
            "crates/demo/src/lib.rs:11:assert",
            "crates/demo/src/lib.rs:15:assert",
            "crates/demo/src/lib.rs:19:assert",
        ]
    );
    assert!(findings
        .iter()
        .any(|finding| finding.path.ends_with("fragment.rs") && finding.kind == "assert"));
}

#[test]
fn cfg_evaluator_only_skips_definitely_test_only_content() {
    let parse = |source: &str| syn::parse_file(source).expect("parse source");
    assert!(definitely_test_only(&parse("#![cfg(test)]\npub fn f() {}\n").attrs).unwrap());
    let nested = parse("#[cfg_attr(not(test), cfg_attr(not(test), cfg(test)))]\npub fn f() {}\n");
    assert!(definitely_test_only(item_attrs(&nested.items[0])).unwrap());
    let unknown = parse("#[cfg(target_arch = \"never\")]\npub fn f() {}\n");
    assert!(!definitely_test_only(item_attrs(&unknown.items[0])).unwrap());
}

#[test]
fn assertion_kind_recognizes_all_macro_variants() {
    for name in [
        "assert",
        "assert_eq",
        "assert_ne",
        "debug_assert",
        "debug_assert_eq",
        "debug_assert_ne",
    ] {
        assert_eq!(assertion_kind(name), Some(name));
        // Raw identifiers keep the same classification.
        assert_eq!(assertion_kind(&format!("r#{name}")), Some(name));
    }
    assert_eq!(assertion_kind("assert_ge"), None);
    assert_eq!(assertion_kind("unwrap"), None);
}

#[test]
fn sanctioned_invariant_panic_requires_one_direct_panic() {
    let parse = |source: &str| syn::parse_file(source).expect("parse source");
    let path = SANCTIONED_INVARIANT_PATH;
    let tail =
        parse("pub fn invariant_violated(context: &str) -> ! {\n    panic!(\"{context}\")\n}\n");
    assert_eq!(
        sanctioned_invariant_panic(path, &tail.items)
            .expect("sanctioned")
            .entry(),
        "crates/tensor4all-tensorbackend/src/lib.rs:2:panic"
    );
    let semicolon =
        parse("pub fn invariant_violated(context: &str) -> ! {\n    panic!(\"{context}\");\n}\n");
    assert!(sanctioned_invariant_panic(path, &semicolon.items).is_some());
    let extra = parse(
        "pub fn invariant_violated(context: &str) -> ! {\n    if context.is_empty() { panic!(\"empty\"); }\n    panic!(\"{context}\")\n}\n",
    );
    assert!(sanctioned_invariant_panic(path, &extra.items).is_none());
    let nested = parse(
        "pub fn invariant_violated(context: &str) -> ! {\n    let fail = || panic!(\"nested\");\n    fail();\n    panic!(\"{context}\")\n}\n",
    );
    assert!(sanctioned_invariant_panic(path, &nested.items).is_none());
}

#[test]
fn sanctioned_invariant_panic_counts_occurrences_not_lines() {
    let parse = |source: &str| syn::parse_file(source).expect("parse source");
    let path = SANCTIONED_INVARIANT_PATH;
    // Two panics on the one allowlisted line must not both pass.
    let same_line = parse(
        "pub fn invariant_violated(c: &str) -> ! {\n    if c.is_empty() { panic!(\"e\") } panic!(\"{c}\")\n}\n",
    );
    assert!(sanctioned_invariant_panic(path, &same_line.items).is_none());
    // A panic hidden in another macro's arguments counts too.
    let in_macro = parse(
        "pub fn invariant_violated(c: &str) -> ! {\n    let _v = vec![panic!(\"x\")];\n    panic!(\"{c}\")\n}\n",
    );
    assert!(sanctioned_invariant_panic(path, &in_macro.items).is_none());
}

#[test]
fn invariant_call_visitor_sees_macro_arguments_and_rejects_renames() {
    let parse = |source: &str| syn::parse_file(source).expect("parse source");
    let mut calls = BTreeSet::new();
    let mut visitor = InvariantCallVisitor {
        path: "crates/demo/src/lib.rs",
        calls: &mut calls,
        error: None,
    };
    visitor.visit_file(&parse(
        "fn f(x: Option<u8>) -> u8 {\n    passthrough!(x.unwrap_or_else(|| invariant_violated(\"c\", \"e\")))\n}\n",
    ));
    assert!(visitor.error.is_none());
    assert_eq!(
        calls.into_iter().collect::<Vec<_>>(),
        vec![("crates/demo/src/lib.rs".to_owned(), 2)]
    );

    let mut calls = BTreeSet::new();
    let mut visitor = InvariantCallVisitor {
        path: "crates/demo/src/lib.rs",
        calls: &mut calls,
        error: None,
    };
    visitor.visit_file(&parse(
        "use tensor4all_tensorbackend::invariant_violated as iv;\n",
    ));
    assert!(visitor.error.is_some());
}

#[test]
fn assertion_only_helper_ignores_format_arguments() {
    let block = |source: &str| match &syn::parse_file(source).expect("parse source").items[0] {
        Item::Fn(function) => function.block.clone(),
        _ => panic!("expected function"),
    };
    let path = "crates/demo/src/lib.rs";
    assert!(assertion_only_helper(
        path,
        &block("fn check(v: u32) { assert!(true, \"{}\", v); }\n")
    )
    .is_none());
    assert!(assertion_only_helper(
        path,
        &block("fn check(v: u32) { assert_eq!(1, 1, \"{}\", v); }\n")
    )
    .is_none());
    assert!(assertion_only_helper(
        path,
        &block("fn check(v: u32) { assert_eq!(v, 1, \"{}\", v); }\n")
    )
    .is_some());
}

#[test]
fn sanctioned_invariant_panic_ignores_nested_same_name_function() {
    let parse = |source: &str| syn::parse_file(source).expect("parse source");
    let nested = parse(
        "mod inner {\n    pub fn invariant_violated() -> ! {\n        panic!(\"nested\")\n    }\n}\n",
    );
    assert!(sanctioned_invariant_panic(SANCTIONED_INVARIANT_PATH, &nested.items).is_none());
    let unrelated = parse("pub fn other() { panic!(\"other\"); }\n");
    assert!(sanctioned_invariant_panic(SANCTIONED_INVARIANT_PATH, &unrelated.items).is_none());
}

#[test]
fn assertion_only_helper_flags_only_single_runtime_assertions() {
    let parse = |source: &str| syn::parse_file(source).expect("parse source");
    let block = |source: &str| match &parse(source).items[0] {
        Item::Fn(function) => function.block.clone(),
        _ => panic!("expected function"),
    };
    let path = "crates/demo/src/lib.rs";
    // Tail-expression and semicolon forms with a runtime operand are flagged.
    assert_eq!(
        assertion_only_helper(path, &block("fn check(ok: bool) { assert!(ok) }\n"))
            .expect("tail expression flagged")
            .entry(),
        "crates/demo/src/lib.rs:1:assert"
    );
    assert_eq!(
        assertion_only_helper(path, &block("fn check(ok: bool) { debug_assert!(ok); }\n"))
            .expect("semicolon flagged")
            .kind,
        "debug_assert"
    );
    assert!(assertion_only_helper(
        path,
        &block("fn check(a: i32, b: i32) { assert_eq!(a, b); }\n")
    )
    .is_some());
    // Literal-only assertions cannot launder a runtime invariant.
    assert!(assertion_only_helper(path, &block("fn check() { assert!(true); }\n")).is_none());
    assert!(assertion_only_helper(path, &block("fn check() { assert_eq!(1, 1); }\n")).is_none());
    // Any other statement, or a second assertion, leaves the helper out of scope.
    assert!(assertion_only_helper(
        path,
        &block("fn check(ok: bool) { assert!(ok); assert!(ok); }\n")
    )
    .is_none());
    assert!(assertion_only_helper(
        path,
        &block("fn check(ok: bool) { let _ = ok; assert!(ok); }\n")
    )
    .is_none());
}

#[test]
fn scan_sources_sanctions_the_helper_and_reports_production_call_sites() {
    let root = TempRoot::new();
    let helper = root.path.join("crates/tensor4all-tensorbackend/src/lib.rs");
    fs::create_dir_all(helper.parent().expect("helper parent")).expect("create helper dir");
    let source = "pub fn invariant_violated(context: &str) -> ! {\n    panic!(\"{context}\")\n}\n\npub fn caller(flag: bool) -> u32 {\n    if flag {\n        1\n    } else {\n        invariant_violated(\"bad\")\n    }\n}\n\n#[cfg(test)]\nmod tests {\n    fn test_only() {\n        invariant_violated(\"test\");\n    }\n}\n";
    fs::write(&helper, source).expect("write helper source");
    let sources = BTreeSet::from([fs::canonicalize(helper).expect("canonical helper")]);
    let scanned = scan_sources(&root.path, &sources).expect("scan");
    assert_eq!(
        scanned.sanctioned_invariant_panics.len(),
        1,
        "the one direct panic is sanctioned"
    );
    assert_eq!(
        scanned.invariant_calls.into_iter().collect::<Vec<_>>(),
        vec![("crates/tensor4all-tensorbackend/src/lib.rs".to_owned(), 9)],
        "only the production call site is reported"
    );
    assert!(scanned.assertions.is_empty());
}

/// A keyed baseline entry, spelled the way a reviewed baseline holds it.
fn keyed(path: &str, kind: &str, item: &str, index: usize, line: usize, content: &str) -> Finding {
    Finding {
        key: FindingKey {
            path: path.to_owned(),
            kind: kind.to_owned(),
            item: item.to_owned(),
            index,
        },
        line,
        content: content.to_owned(),
    }
}

fn legacy(value: &str) -> BaselineEntry {
    BaselineEntry::Legacy(parse_legacy_entry(value).expect("valid legacy entry"))
}

/// The baseline a reviewer would commit for this fixture: the findings it
/// produces today.
fn reviewed_baseline(root: &TempRoot, source: &str) -> Vec<BaselineEntry> {
    scanned_findings(root, source)
        .into_iter()
        .map(BaselineEntry::Keyed)
        .collect()
}

fn scanned_findings(root: &TempRoot, source: &str) -> BTreeSet<Finding> {
    let path = root.path.join("crates/demo/src/lib.rs");
    fs::write(&path, source).expect("write fixture source");
    let sources = BTreeSet::from([fs::canonicalize(&path).expect("canonicalize fixture source")]);
    let scan = scan_sources(&root.path, &sources).expect("scan fixture source");
    key_findings(&scan.items, scan.assertions)
}

fn report_for(root: &TempRoot, source: &str, entries: Vec<BaselineEntry>) -> AuditReport {
    let path = root.path.join("crates/demo/src/lib.rs");
    fs::write(&path, source).expect("write fixture source");
    let sources = BTreeSet::from([fs::canonicalize(&path).expect("canonicalize fixture source")]);
    let scan = scan_sources(&root.path, &sources).expect("scan fixture source");
    let actual = key_findings(&scan.items, scan.assertions);
    build_report(
        &actual,
        entries,
        &BTreeSet::new(),
        BTreeSet::new(),
        None,
        &root.path.join("baseline.json"),
    )
}

const FIXTURE_ASSERTION: &str = "pub fn check(value: usize, limit: usize) -> usize {\n    \
                                  assert!(value > 0);\n    assert!(value < limit);\n    value\n}\n";

const FIXTURE_TWO_ITEMS: &str = "pub fn first(ok: bool) { assert!(ok); }\n\n\
                                 pub fn second(ok: bool) { assert!(ok); }\n";

const FIXTURE_TWO_ITEMS_SWAPPED: &str = "pub fn first() {}\n\n\
                                         pub fn second(ok: bool) {\n    assert!(ok);\n    assert!(ok);\n}\n";

const FIXTURE_ONE_LINE: &str =
    "pub fn check(ok: bool, other: bool) { assert!(ok); assert!(other); }\n";

const FIXTURE_ONE_LINE_INSERTED: &str = "// inserted\n\
                                         pub fn check(ok: bool, other: bool) { assert!(ok); assert!(other); }\n";

const FIXTURE_ONE_LINE_SWAPPED: &str =
    "pub fn check(ok: bool, other: bool) { assert!(other); assert!(ok); }\n";

#[test]
fn rendering_ignores_layout_but_keeps_literal_text() {
    let render = |source: &str| render_tokens(&source.parse::<TokenStream>().expect("tokens"));
    assert_eq!(render("|& row | row < m . nrows"), "|&row|row<m.nrows");
    assert_eq!(render("|\n&row|\nrow < m.nrows"), "|&row|row<m.nrows");
    assert_eq!(render("a as u8"), "a as u8");
    assert_eq!(render("value<=limit"), "value<=limit");
    assert_eq!(render("ok && probe()"), "ok&&probe()");
    assert_eq!(render("ok & &probe()"), "ok& &probe()");
    assert_ne!(
        render("ok && probe()"),
        render("ok & &probe()"),
        "a compound operator must not fuse into two standalone tokens"
    );
    assert_ne!(
        render("\"a + b\""),
        render("\"a+b\""),
        "a literal keeps its exact text"
    );
}

#[test]
fn reviewed_baseline_entries_carry_the_enclosing_item_and_ordinal() {
    let root = TempRoot::new();
    let keys = |entries: &[BaselineEntry]| {
        entries
            .iter()
            .map(|entry| match entry {
                BaselineEntry::Keyed(finding) => finding.key_text(),
                BaselineEntry::Legacy(_) => "<legacy>".to_owned(),
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        keys(&reviewed_baseline(&root, FIXTURE_ASSERTION)),
        vec!["check#0".to_owned(), "check#1".to_owned()]
    );
    assert_eq!(
        keys(&reviewed_baseline(&root, FIXTURE_TWO_ITEMS)),
        vec!["first#0".to_owned(), "second#0".to_owned()],
        "each item keeps its own key; findings must not all land in one scope"
    );
}

#[test]
fn an_assertion_moved_between_items_fails() {
    let root = TempRoot::new();
    let entries = reviewed_baseline(&root, FIXTURE_TWO_ITEMS);
    let report = report_for(&root, FIXTURE_TWO_ITEMS_SWAPPED, entries);
    assert_eq!(report.stale.len(), 1, "{:?}", report.stale);
    assert!(
        report.stale[0].contains("\"item\":\"first\""),
        "the entry of the item the assertion left is stale: {:?}",
        report.stale
    );
    assert_eq!(
        report
            .unbaselined
            .iter()
            .map(Finding::key_text)
            .collect::<Vec<_>>(),
        vec!["second#1".to_owned()],
        "the copy that appeared in the other item is added"
    );
    assert_eq!(report.exit_code_and_print(), 1);
}

#[test]
fn assertions_on_one_line_keep_their_own_keys() {
    let root = TempRoot::new();
    let entries = reviewed_baseline(&root, FIXTURE_ONE_LINE);
    assert_eq!(entries.len(), 2, "both assertions on the line are reviewed");
    let moved = report_for(&root, FIXTURE_ONE_LINE_INSERTED, entries);
    assert!(
        moved.unbaselined.is_empty() && moved.stale.is_empty(),
        "a line inserted above the item is not a change: {:?}",
        moved.stale
    );
    assert_eq!(moved.moved.len(), 2, "both entries moved");
    let swapped = report_for(
        &root,
        FIXTURE_ONE_LINE_SWAPPED,
        reviewed_baseline(&root, FIXTURE_ONE_LINE),
    );
    assert_eq!(
        swapped.changed.len(),
        2,
        "swapping the two assertions on one line changes both keys"
    );
    assert_eq!(swapped.exit_code_and_print(), 1);
}

#[test]
fn an_unaligned_recorded_ordinal_is_not_accepted() {
    let root = TempRoot::new();
    let mut entries = reviewed_baseline(&root, FIXTURE_ASSERTION);
    for entry in &mut entries {
        if let BaselineEntry::Keyed(finding) = entry {
            if finding.key.index == 0 {
                finding.key.index = 7;
            }
        }
    }
    let report = report_for(&root, FIXTURE_ASSERTION, entries);
    assert_eq!(
        report.unbaselined.len(),
        1,
        "the finding whose ordinal no longer matches is added"
    );
    assert_eq!(report.stale.len(), 1, "and its entry is stale");
    assert_eq!(report.exit_code_and_print(), 1);
    assert!(
        report.update_baseline(None).is_err(),
        "the updater must not silently renumber a changed key set"
    );
}

#[test]
fn assertions_swapped_inside_one_item_are_reported_as_changed() {
    let root = TempRoot::new();
    let entries = reviewed_baseline(&root, FIXTURE_ASSERTION);
    let swapped = FIXTURE_ASSERTION
        .replace("assert!(value > 0)", "assert!(TMP)")
        .replace("assert!(value < limit)", "assert!(value > 0)")
        .replace("assert!(TMP)", "assert!(value < limit)");
    let report = report_for(&root, &swapped, entries);
    assert_eq!(report.changed.len(), 2, "a reordering is not line movement");
    assert_eq!(report.exit_code_and_print(), 1);
}

#[test]
fn lines_inserted_above_a_baselined_assertion_are_green_and_reported_as_moved() {
    let root = TempRoot::new();
    let entries = reviewed_baseline(&root, FIXTURE_ASSERTION);
    assert_eq!(entries.len(), 2, "both fixture assertions are reviewed");
    let moved_source =
        format!("// inserted line\n// inserted line\n// inserted line\n{FIXTURE_ASSERTION}");
    let report = report_for(&root, &moved_source, entries);
    assert!(
        report.unbaselined.is_empty() && report.stale.is_empty() && report.changed.is_empty(),
        "a pure line shift must not add, stale, or change an entry"
    );
    assert_eq!(
        report
            .moved
            .iter()
            .map(|(finding, baseline_line)| (finding.entry(), *baseline_line))
            .collect::<Vec<_>>(),
        vec![
            ("crates/demo/src/lib.rs:5:assert".to_owned(), 2),
            ("crates/demo/src/lib.rs:6:assert".to_owned(), 3),
        ]
    );
    assert_eq!(report.exit_code_and_print(), 0);
}

#[test]
fn an_added_assertion_fails_and_names_the_new_assertion_only() {
    let root = TempRoot::new();
    let entries = reviewed_baseline(&root, FIXTURE_ASSERTION);
    let with_new = FIXTURE_ASSERTION.replace(
        "    assert!(value < limit);\n",
        "    assert!(limit > 0);\n    assert!(value < limit);\n",
    );
    let report = report_for(&root, &with_new, entries);
    assert_eq!(
        report
            .unbaselined
            .iter()
            .map(|finding| (finding.entry(), finding.key_text()))
            .collect::<Vec<_>>(),
        vec![(
            "crates/demo/src/lib.rs:3:assert".to_owned(),
            "check#1".to_owned()
        )]
    );
    assert!(
        report.stale.is_empty(),
        "the reviewed assertions kept their entries; only the new one is added"
    );
    assert_eq!(report.exit_code_and_print(), 1);
}

#[test]
fn a_deleted_assertion_fails_as_stale() {
    let root = TempRoot::new();
    let entries = reviewed_baseline(&root, FIXTURE_ASSERTION);
    let without_one = FIXTURE_ASSERTION.replace("    assert!(value < limit);\n", "");
    let report = report_for(&root, &without_one, entries);
    assert!(report.unbaselined.is_empty());
    assert_eq!(
        report.stale,
        vec![keyed(
            "crates/demo/src/lib.rs",
            "assert",
            "check",
            1,
            3,
            "value<limit"
        )
        .json()]
    );
    assert_eq!(report.exit_code_and_print(), 1);
}

#[test]
fn a_rewritten_assertion_fails_as_changed() {
    let root = TempRoot::new();
    let entries = reviewed_baseline(&root, FIXTURE_ASSERTION);
    let rewritten = FIXTURE_ASSERTION.replace("assert!(value < limit)", "assert!(value <= limit)");
    let report = report_for(&root, &rewritten, entries);
    assert!(report.unbaselined.is_empty() && report.stale.is_empty());
    assert_eq!(report.changed.len(), 1);
    assert_eq!(report.changed[0].0.content, "value<=limit");
    assert_eq!(report.exit_code_and_print(), 1);
}

#[test]
fn a_legacy_entry_migrates_while_it_matches_its_line() {
    let root = TempRoot::new();
    let report = report_for(
        &root,
        FIXTURE_ASSERTION,
        vec![legacy("crates/demo/src/lib.rs:2:assert")],
    );
    assert_eq!(report.matched.len(), 1);
    assert_eq!(report.resolved.len(), 1);
    assert_eq!(
        report.resolved[0].previous, "crates/demo/src/lib.rs:2:assert",
        "the diff of a migrated entry shows the shape it replaced"
    );
    // The other fixture assertion has no entry, so the run still fails.
    assert_eq!(report.unbaselined.len(), 1);
}

#[test]
fn a_legacy_entry_that_moved_without_matching_is_stale() {
    let root = TempRoot::new();
    let report = report_for(
        &root,
        FIXTURE_ASSERTION,
        vec![legacy("crates/demo/src/lib.rs:99:assert")],
    );
    assert_eq!(report.stale, vec!["crates/demo/src/lib.rs:99:assert"]);
    assert!(report.matched.is_empty());
}

#[test]
fn legacy_entries_cannot_collapse_onto_one_finding() {
    let root = TempRoot::new();
    // Both legacy entries claim the same line, so only one may migrate; the
    // other stays stale instead of silently duplicating the survivor.
    let report = report_for(
        &root,
        FIXTURE_ASSERTION,
        vec![
            legacy("crates/demo/src/lib.rs:2:assert"),
            legacy("crates/demo/src/lib.rs:2:assert"),
        ],
    );
    assert_eq!(report.matched.len(), 1);
    assert_eq!(report.stale.len(), 1);
}

#[test]
fn report_removes_sanctioned_panics_before_baselining() {
    let sanctioned = keyed(
        "crates/tensor4all-tensorbackend/src/lib.rs",
        "panic",
        "invariant_violated",
        0,
        150,
        "",
    );
    let extra = keyed(
        "crates/tensor4all-tensorbackend/src/lib.rs",
        "panic",
        "invariant_violated",
        1,
        151,
        "",
    );
    let actual = [sanctioned.clone(), extra.clone()]
        .into_iter()
        .collect::<BTreeSet<_>>();
    let report = build_report(
        &actual,
        Vec::new(),
        &BTreeSet::from([sanctioned.clone()]),
        BTreeSet::new(),
        None,
        Path::new("baseline.json"),
    );
    assert_eq!(report.sanctioned, vec![sanctioned]);
    assert_eq!(report.unbaselined, vec![extra]);
}

#[test]
fn a_sanctioned_key_at_another_line_is_not_subtracted() {
    let sanctioned = keyed(
        "crates/tensor4all-tensorbackend/src/lib.rs",
        "panic",
        "invariant_violated",
        0,
        150,
        "",
    );
    let unsanctioned = keyed(
        "crates/tensor4all-tensorbackend/src/lib.rs",
        "panic",
        "invariant_violated",
        0,
        151,
        "",
    );
    let actual = BTreeSet::from([unsanctioned.clone()]);
    let report = build_report(
        &actual,
        Vec::new(),
        &BTreeSet::from([sanctioned]),
        BTreeSet::new(),
        None,
        Path::new("baseline.json"),
    );
    assert_eq!(
        report.unbaselined,
        vec![unsanctioned],
        "only the exact sanctioned location is allowlisted"
    );
}

#[test]
fn baseline_entries_reject_noncanonical_numbers_and_unknown_fields() {
    let parse = |value: Value| parse_baseline_entry(&value);
    assert!(parse(json!("crates/demo/src/lib.rs:01:assert")).is_err());
    assert!(parse(json!("crates/demo/src/lib.rs:0:assert")).is_err());
    assert!(parse(json!("crates/demo/src/lib.rs:1:unknown")).is_err());
    assert!(parse(json!("demo/src/lib.rs:1:assert")).is_err());
    assert!(parse(json!({
        "path": "crates/demo/src/lib.rs", "kind": "assert", "item": "check",
        "index": 0, "line": 0, "content": "value > 0"
    }))
    .is_err());
    assert!(parse(json!({
        "path": "crates/demo/src/lib.rs", "kind": "assert", "item": "",
        "index": 0, "line": 1, "content": "value > 0"
    }))
    .is_err());
    assert!(parse(json!({
        "path": "crates/demo/src/lib.rs", "kind": "assert", "item": "check",
        "index": 0, "line": 1,
    }))
    .is_err());
    assert!(parse(json!({
        "path": "crates/demo/src/lib.rs", "kind": "assert", "item": "check",
        "index": 0, "line": 1, "content": "value > 0", "extra": 1
    }))
    .is_err());
}

#[test]
fn report_prints_sanctioned_and_call_sites_without_failing() {
    let sanctioned = keyed(
        "crates/tensor4all-tensorbackend/src/lib.rs",
        "panic",
        "invariant_violated",
        0,
        150,
        "",
    );
    let passing = AuditReport {
        configuration: None,
        baseline_path: PathBuf::from("baseline.json"),
        matched: Vec::new(),
        moved: Vec::new(),
        changed: Vec::new(),
        unbaselined: Vec::new(),
        stale: Vec::new(),
        forbidden_baseline: Vec::new(),
        sanctioned: vec![sanctioned],
        invariant_calls: vec![("crates/demo/src/lib.rs".to_owned(), 9)],
        resolved: Vec::new(),
    };
    assert_eq!(passing.exit_code_and_print(), 0);
    let failing = AuditReport {
        configuration: None,
        baseline_path: PathBuf::from("baseline.json"),
        matched: Vec::new(),
        moved: Vec::new(),
        changed: Vec::new(),
        unbaselined: vec![keyed(
            "crates/demo/src/lib.rs",
            "assert",
            "check",
            0,
            1,
            "value > 0",
        )],
        stale: Vec::new(),
        forbidden_baseline: Vec::new(),
        sanctioned: Vec::new(),
        invariant_calls: Vec::new(),
        resolved: Vec::new(),
    };
    assert_eq!(failing.exit_code_and_print(), 1);
}

#[test]
fn report_marks_new_and_stale_entries_without_baselining_raw_paths() {
    let matched = keyed(
        "crates/demo/src/lib.rs",
        "assert",
        "check",
        0,
        1,
        "value > 0",
    );
    let stale = keyed(
        "crates/demo/src/lib.rs",
        "assert",
        "check",
        1,
        2,
        "value < limit",
    );
    let raw = keyed("crates/demo/src/lib.rs", "panic", "check", 0, 3, "");
    let new = keyed("crates/demo/src/lib.rs", "unwrap", "check", 0, 4, "");
    let actual = [matched.clone(), new.clone()]
        .into_iter()
        .collect::<BTreeSet<_>>();
    let baseline = [matched.clone(), stale.clone(), raw.clone()]
        .into_iter()
        .map(BaselineEntry::Keyed)
        .collect::<Vec<_>>();
    let report = build_report(
        &actual,
        baseline,
        &BTreeSet::new(),
        BTreeSet::new(),
        None,
        Path::new("baseline.json"),
    );
    assert_eq!(report.matched, vec![matched]);
    assert_eq!(report.unbaselined, vec![new]);
    assert_eq!(
        report.stale.into_iter().collect::<BTreeSet<_>>(),
        BTreeSet::from([stale.json(), raw.json()])
    );
    assert_eq!(report.forbidden_baseline, vec![raw.json()]);
}

#[test]
fn configuration_selection_does_not_report_unmatched_baseline_entries() {
    let matched = keyed(
        "crates/demo/src/lib.rs",
        "assert",
        "check",
        0,
        1,
        "value > 0",
    );
    let out_of_selection = keyed(
        "crates/demo/src/lib.rs",
        "assert",
        "check",
        1,
        2,
        "value < limit",
    );
    let actual = [matched.clone()].into_iter().collect::<BTreeSet<_>>();
    let baseline = [matched, out_of_selection]
        .into_iter()
        .map(BaselineEntry::Keyed)
        .collect::<Vec<_>>();
    let configuration = Configuration::new(
        "diagnostics".to_owned(),
        vec!["tensor4all-treetn".to_owned()],
        false,
        Some("diagnostics".to_owned()),
    )
    .unwrap();
    let report = build_report(
        &actual,
        baseline,
        &BTreeSet::new(),
        BTreeSet::new(),
        Some(&configuration),
        Path::new("baseline.json"),
    );
    assert!(report.stale.is_empty());
    assert_eq!(report.configuration.as_deref(), Some("diagnostics"));
}

#[test]
fn update_baseline_rewrites_moved_entries_and_prints_the_diff() {
    let root = TempRoot::new();
    let entries = reviewed_baseline(&root, FIXTURE_ASSERTION);
    let moved_source = format!("// inserted line\n// inserted line\n{FIXTURE_ASSERTION}");
    let report = report_for(&root, &moved_source, entries);
    let baseline = root.path.join("baseline.json");
    fs::write(&baseline, "[]\n").expect("write baseline");
    let diff = report
        .update_baseline(None)
        .expect("baseline update succeeds");
    assert_eq!(
        diff.len(),
        4,
        "one removed and one added line per entry whose recorded line moved: {diff:?}"
    );
    assert!(diff[0].starts_with("- {"), "{diff:?}");
    assert!(diff[1].starts_with("+ {"), "{diff:?}");
    assert!(diff[1].contains("\"line\":4"), "{diff:?}");
    let text = fs::read_to_string(&baseline).expect("read baseline");
    let values: Vec<Value> = serde_json::from_str(&text).expect("keyed baseline");
    assert_eq!(values.len(), 2);
    assert_eq!(
        values
            .iter()
            .map(|value| value["line"].as_u64())
            .collect::<Vec<_>>(),
        vec![Some(4), Some(5)],
        "the rewritten baseline records the current lines"
    );
}

#[test]
fn update_baseline_refuses_in_ci_and_when_the_keys_changed() {
    let root = TempRoot::new();
    let entries = reviewed_baseline(&root, FIXTURE_ASSERTION);
    let baseline = root.path.join("baseline.json");
    fs::write(&baseline, "[]\n").expect("write baseline");
    let unchanged = report_for(&root, FIXTURE_ASSERTION, entries.clone());
    let error = unchanged
        .update_baseline(Some("true"))
        .expect_err("CI must refuse the update");
    assert!(error.to_string().contains("in CI"), "{error}");
    assert!(
        unchanged.update_baseline(Some("0")).is_ok(),
        "an explicit non-CI value is not CI"
    );

    let with_new = FIXTURE_ASSERTION.replace(
        "    assert!(value < limit);\n",
        "    assert!(limit > 0);\n    assert!(value < limit);\n",
    );
    let changed = report_for(&root, &with_new, entries);
    let error = changed
        .update_baseline(None)
        .expect_err("an added assertion must not be absorbed into the baseline");
    assert!(error.to_string().contains("keys changed"), "{error}");
}
