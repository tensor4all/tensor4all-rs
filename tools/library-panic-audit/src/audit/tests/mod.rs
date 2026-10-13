use super::*;

#[test]
fn enabled_features_drops_only_the_excluded_names() {
    let available = ["alpha".to_owned(), "beta".to_owned(), "gamma".to_owned()];
    let excluded = BTreeSet::from(["beta".to_owned()]);
    assert_eq!(
        enabled_features(&available, &excluded).unwrap(),
        "alpha,gamma"
    );
    assert!(enabled_features(&available, &BTreeSet::new()).is_ok());
}

#[test]
fn enabled_features_rejects_unknown_or_complete_exclusions() {
    let available = ["alpha".to_owned()];
    let unknown = BTreeSet::from(["nope".to_owned()]);
    let error = enabled_features(&available, &unknown)
        .unwrap_err()
        .to_string();
    assert!(error.contains("does not define"), "{error}");
    let everything = BTreeSet::from(["alpha".to_owned()]);
    let error = enabled_features(&available, &everything)
        .unwrap_err()
        .to_string();
    assert!(error.contains("nothing left"), "{error}");
}
use serde_json::json;
use std::time::{SystemTime, UNIX_EPOCH};

struct TempRoot {
    path: PathBuf,
}

impl TempRoot {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("t4a-panic-audit-{nonce}"));
        fs::create_dir_all(path.join("crates/demo/src")).expect("create fixture root");
        fs::write(path.join("crates/demo/src/lib.rs"), "pub fn demo() {}\n")
            .expect("write fixture source");
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
    let findings = scan_assertions(&root.path, &sources).expect("scan selected sources");
    let parsed_findings = findings
        .iter()
        .filter(|finding| finding.path.ends_with("lib.rs"))
        .map(Finding::entry)
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
fn baseline_lines_reject_noncanonical_numbers() {
    assert!(parse_entry("crates/demo/src/lib.rs:01:assert").is_err());
    assert!(parse_entry("crates/demo/src/lib.rs:0:assert").is_err());
    assert!(parse_entry("crates/demo/src/lib.rs:1:unknown").is_err());
    assert!(parse_entry("demo/src/lib.rs:1:assert").is_err());
}

#[test]
fn report_marks_new_and_stale_entries_without_baselining_raw_paths() {
    let matched = parse_entry("crates/demo/src/lib.rs:1:assert").expect("matched");
    let stale = parse_entry("crates/demo/src/lib.rs:2:assert").expect("stale");
    let raw = parse_entry("crates/demo/src/lib.rs:3:panic").expect("raw");
    let new = parse_entry("crates/demo/src/lib.rs:4:unwrap").expect("new");
    let actual = [matched.clone(), new].into_iter().collect::<BTreeSet<_>>();
    let baseline = [matched, stale, raw].into_iter().collect::<BTreeSet<_>>();
    let report = build_report(&actual, &baseline);
    assert_eq!(
        report.matched,
        vec![parse_entry("crates/demo/src/lib.rs:1:assert").unwrap()]
    );
    assert_eq!(
        report.unbaselined,
        vec![parse_entry("crates/demo/src/lib.rs:4:unwrap").unwrap()]
    );
    assert_eq!(
        report.stale,
        vec![
            parse_entry("crates/demo/src/lib.rs:2:assert").unwrap(),
            parse_entry("crates/demo/src/lib.rs:3:panic").unwrap()
        ]
    );
    assert_eq!(
        report.forbidden_baseline,
        vec![parse_entry("crates/demo/src/lib.rs:3:panic").unwrap()]
    );
}
