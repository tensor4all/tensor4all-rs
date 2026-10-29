//! Compiler-backed production panic audit.
//!
//! Raw panic findings come from Clippy diagnostics emitted while Cargo checks
//! the workspace's production library and binary targets. Assertion findings
//! are scanned only in the source files selected by rustc's dep-info for those
//! same compiler runs; the source visitor does not resolve modules or evaluate
//! target and feature configuration. Dep-info can also list `.rs` files used as
//! `include!`, `include_str!`, or `include_bytes!` data rather than complete
//! modules: complete files use the public visitor, while parse failures are
//! conservatively token-scanned for literal `assert!`-family calls (`assert!`,
//! `assert_eq!`, `assert_ne!`, `debug_assert!`, `debug_assert_eq!`,
//! `debug_assert_ne!`). Macro invocation arguments and `macro_rules!`
//! transcribers are likewise scanned only for those literal assertion calls;
//! no semantic macro resolver is attempted. The public-path assertion scanner
//! covers public free functions, public inherent methods, trait defaults, and
//! trait-implementation methods. Assertions in other private helpers are
//! outside that surface, with one exception: a private free function or
//! private inherent method whose entire body is a single `assert!`-family call
//! on a runtime value is recorded, because that helper is the laundering route
//! for a panic the public-path scan cannot see. The single sanctioned
//! `panic!` inside `tensor4all_tensorbackend::invariant_violated` is
//! allowlisted by exact normalized path and top-level function name only while
//! that body holds exactly one direct `panic!`; every call to that helper is
//! printed for review. Any additional or nested panic in that file or function
//! remains a finding.

use anyhow::{anyhow, bail, Context, Result};
use proc_macro2::{TokenStream, TokenTree};
use serde_json::Value;
use std::collections::{BTreeSet, HashSet};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::visit::Visit;
use syn::{Arm, Attribute, Expr, ImplItem, Item, Meta, Stmt, Token, TraitItem, Visibility};

const RAW_CODES: [(&str, &str); 4] = [
    ("clippy::panic", "panic"),
    ("clippy::unreachable", "unreachable"),
    ("clippy::unwrap_used", "unwrap"),
    ("clippy::expect_used", "expect"),
];
const RAW_KINDS: [&str; 4] = ["panic", "unreachable", "unwrap", "expect"];
const ASSERTION_KINDS: [&str; 6] = [
    "assert",
    "assert_eq",
    "assert_ne",
    "debug_assert",
    "debug_assert_eq",
    "debug_assert_ne",
];
/// Exact normalized path of the one sanctioned invariant-panic helper.
const SANCTIONED_INVARIANT_PATH: &str = "crates/tensor4all-tensorbackend/src/lib.rs";
/// Top-level function whose single direct `panic!` is allowlisted.
const SANCTIONED_INVARIANT_FUNCTION: &str = "invariant_violated";
const PRODUCTION_KINDS: [&str; 7] = [
    "lib",
    "rlib",
    "dylib",
    "cdylib",
    "staticlib",
    "proc-macro",
    "bin",
];
// Each feature pass gets ten minutes; the wrapper allows both passes and the
// CI step leaves room for the wrapper build and process teardown.
const CLIPPY_TIMEOUT: Duration = Duration::from_secs(600);

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct Finding {
    path: String,
    line: usize,
    kind: String,
}

impl Finding {
    fn entry(&self) -> String {
        format!("{}:{}:{}", self.path, self.line, self.kind)
    }
}

#[derive(Debug)]
pub(crate) struct AuditReport {
    configuration: Option<String>,
    matched: Vec<Finding>,
    unbaselined: Vec<Finding>,
    stale: Vec<Finding>,
    forbidden_baseline: Vec<Finding>,
    sanctioned: Vec<Finding>,
    invariant_calls: Vec<(String, usize)>,
}

impl AuditReport {
    pub(crate) fn exit_code_and_print(&self) -> i32 {
        if let Some(configuration) = &self.configuration {
            println!("Audit configuration: `{configuration}`");
        }
        for finding in &self.matched {
            println!("Baseline matched: {}", finding.entry());
        }
        for finding in &self.sanctioned {
            println!("Sanctioned invariant panic: {}", finding.entry());
        }
        for finding in &self.unbaselined {
            println!("{}", finding.entry());
        }
        for finding in &self.stale {
            println!("Stale baseline: {}", finding.entry());
        }
        for finding in &self.forbidden_baseline {
            eprintln!(
                "Baseline contains forbidden raw panic-style entry: {}",
                finding.entry()
            );
        }

        println!(
            "invariant_violated call sites: {}",
            self.invariant_calls.len()
        );
        for (path, line) in &self.invariant_calls {
            println!("invariant_violated call site: {path}:{line}");
        }

        if !self.unbaselined.is_empty()
            || !self.stale.is_empty()
            || !self.forbidden_baseline.is_empty()
        {
            eprintln!(
                "Audit failed: {} {}, {} {}.",
                self.unbaselined.len(),
                plural(self.unbaselined.len(), "unbaselined finding", None),
                self.stale.len(),
                plural(
                    self.stale.len(),
                    "stale baseline entry",
                    Some("stale baseline entries")
                )
            );
            1
        } else {
            println!(
                "Audit passed: 0 unbaselined findings, {} stale baseline entries",
                self.stale.len()
            );
            0
        }
    }
}

fn plural(count: usize, singular: &str, plural: Option<&str>) -> String {
    if count == 1 {
        singular.to_owned()
    } else {
        plural
            .map(str::to_owned)
            .unwrap_or_else(|| format!("{singular}s"))
    }
}

#[derive(Clone, Debug)]
struct ProductionTarget {
    package_id: String,
    package_name: String,
    target_name: String,
    kind: String,
}

impl ProductionTarget {
    fn key(&self) -> TargetKey {
        TargetKey {
            package_id: self.package_id.clone(),
            target_name: self.target_name.clone(),
            kind: self.kind.clone(),
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct TargetKey {
    package_id: String,
    target_name: String,
    kind: String,
}

#[derive(Clone, Debug)]
struct CompilerArtifact {
    key: TargetKey,
    filenames: BTreeSet<PathBuf>,
    target: Value,
    profile: Value,
}

type CompilerArtifacts = Vec<CompilerArtifact>;
type ClippyParse = (HashSet<TargetKey>, CompilerArtifacts, BTreeSet<Finding>);

/// One Cargo feature configuration selected by the caller.
///
/// The default pass audits every workspace production target with its default
/// features. A configuration pass instead audits exactly the listed packages
/// with one explicit Cargo feature selection, because this workspace's CPU
/// backend features are mutually exclusive and no single all-features build
/// exists (issue #915 D6).
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Configuration {
    pub(crate) name: String,
    pub(crate) packages: Vec<String>,
    pub(crate) no_default_features: bool,
    pub(crate) features: Option<String>,
}

impl Configuration {
    /// Build one named feature configuration from parsed command-line values.
    pub(crate) fn new(
        name: String,
        packages: Vec<String>,
        no_default_features: bool,
        features: Option<String>,
    ) -> Result<Self> {
        let name = name.trim().to_owned();
        if name.is_empty() {
            bail!("--config-name must not be empty");
        }
        if packages.is_empty() {
            bail!("--config-name {name} requires at least one --package");
        }
        let packages = packages
            .into_iter()
            .map(|package| package.trim().to_owned())
            .collect::<Vec<_>>();
        if packages.iter().any(String::is_empty) {
            bail!("--package must not be empty");
        }
        let features = match features {
            Some(features) => {
                let features = features.trim().to_owned();
                if features.is_empty() {
                    bail!("--features must list at least one feature");
                }
                Some(features)
            }
            None => None,
        };
        Ok(Self {
            name,
            packages,
            no_default_features,
            features,
        })
    }

    fn label(&self) -> String {
        let mut selection = format!("configuration `{}`", self.name);
        if self.no_default_features {
            selection.push_str(" --no-default-features");
        }
        if let Some(features) = &self.features {
            selection.push_str(&format!(" --features {features}"));
        }
        selection.push_str(&format!(" -p {}", self.packages.join(" -p ")));
        selection
    }
}

#[derive(Debug)]
struct ProductionSelection {
    root: PathBuf,
    targets: Vec<ProductionTarget>,
    excluded_packages: Vec<String>,
}

impl ProductionSelection {
    fn load(root: &Path, configuration: Option<&Configuration>) -> Result<Self> {
        let root = canonical_root(root)?;
        let output = Command::new("cargo")
            .args(["metadata", "--format-version", "1", "--no-deps"])
            .current_dir(&root)
            .output()
            .context("failed to execute cargo metadata")?;
        if !output.status.success() {
            bail!(
                "cargo metadata failed: {}",
                command_detail(&output.stdout, &output.stderr)
            );
        }
        let metadata: Value = serde_json::from_slice(&output.stdout)
            .context("cargo metadata returned malformed JSON")?;
        let workspace_members = metadata
            .get("workspace_members")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("cargo metadata omitted workspace_members"))?;
        let member_ids = workspace_members
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| anyhow!("cargo metadata contains a non-string workspace member"))
            })
            .collect::<Result<HashSet<_>>>()?;
        let packages = metadata
            .get("packages")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("cargo metadata omitted packages"))?;

        let crates_root = root.join("crates");
        let requested_packages = configuration.map(|configuration| {
            configuration
                .packages
                .iter()
                .map(|package| package.trim().to_owned())
                .collect::<BTreeSet<_>>()
        });
        let mut targets = Vec::new();
        let mut all_workspace_packages = Vec::new();
        for package in packages {
            let package_id = package
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("cargo metadata package omitted id"))?;
            if !member_ids.contains(package_id) {
                continue;
            }
            let package_name = package
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("cargo metadata package omitted name"))?;
            all_workspace_packages.push(package_name.to_owned());
            if let Some(requested) = &requested_packages {
                if !requested.contains(package_name) {
                    continue;
                }
            }
            let manifest = package
                .get("manifest_path")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("cargo metadata package omitted manifest_path"))?;
            let manifest = ensure_inside(&root, Path::new(manifest), "package manifest")?;
            if !manifest.starts_with(&crates_root) {
                continue;
            }
            let package_targets = package
                .get("targets")
                .and_then(Value::as_array)
                .ok_or_else(|| anyhow!("cargo metadata package omitted targets"))?;
            for target in package_targets {
                let target_kind = target
                    .get("kind")
                    .and_then(Value::as_array)
                    .ok_or_else(|| anyhow!("cargo metadata target omitted kind"))?;
                let Some(kind) = cargo_target_kind(target_kind)? else {
                    continue;
                };
                let target_name = target
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow!("cargo metadata target omitted name"))?;
                targets.push(ProductionTarget {
                    package_id: package_id.to_owned(),
                    package_name: package_name.to_owned(),
                    target_name: target_name.to_owned(),
                    kind,
                });
            }
        }

        if let (Some(configuration), Some(requested)) = (configuration, &requested_packages) {
            let selected_package_names = targets
                .iter()
                .map(|target| target.package_name.as_str())
                .collect::<HashSet<_>>();
            let missing = requested
                .iter()
                .filter(|name| !selected_package_names.contains(name.as_str()))
                .cloned()
                .collect::<Vec<_>>();
            if !missing.is_empty() {
                bail!(
                    "configuration `{}` names packages without production targets under crates/: {}",
                    configuration.name,
                    missing.join(", ")
                );
            }
        }
        if targets.is_empty() {
            bail!("cargo metadata selected no production Cargo targets under crates/");
        }
        let selected_package_names = targets
            .iter()
            .map(|target| target.package_name.as_str())
            .collect::<HashSet<_>>();
        let excluded_packages = all_workspace_packages
            .into_iter()
            .filter(|name| !selected_package_names.contains(name.as_str()))
            .collect::<Vec<_>>();

        Ok(Self {
            root,
            targets,
            excluded_packages,
        })
    }

    fn expected_keys(&self) -> HashSet<TargetKey> {
        self.targets.iter().map(ProductionTarget::key).collect()
    }
}

fn cargo_target_kind(kinds: &[Value]) -> Result<Option<String>> {
    if kinds.is_empty() {
        bail!("Cargo target omitted kind entries");
    }
    let kinds = kinds
        .iter()
        .map(|kind| {
            kind.as_str()
                .ok_or_else(|| anyhow!("Cargo target kind was not a string"))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(kinds
        .into_iter()
        .find(|kind| PRODUCTION_KINDS.contains(kind))
        .map(str::to_owned))
}

fn canonical_root(root: &Path) -> Result<PathBuf> {
    let root = fs::canonicalize(root)
        .with_context(|| format!("audit root does not exist: {}", root.display()))?;
    if !root.is_dir() {
        bail!("audit root is not a directory: {}", root.display());
    }
    Ok(root)
}

fn ensure_inside(root: &Path, path: &Path, description: &str) -> Result<PathBuf> {
    let candidate = if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    };
    let canonical = fs::canonicalize(&candidate)
        .with_context(|| format!("{description} does not exist: {}", candidate.display()))?;
    if !canonical.starts_with(root) {
        bail!(
            "{description} is outside audit root: {}",
            canonical.display()
        );
    }
    Ok(canonical)
}

fn command_detail(stdout: &[u8], stderr: &[u8]) -> String {
    let stderr = String::from_utf8_lossy(stderr).trim().to_owned();
    if !stderr.is_empty() {
        return stderr;
    }
    String::from_utf8_lossy(stdout).trim().to_string()
}

fn run_clippy(
    selection: &ProductionSelection,
    configuration: Option<&Configuration>,
) -> Result<(BTreeSet<PathBuf>, BTreeSet<Finding>)> {
    let mut command = Command::new("cargo");
    command.arg("clippy");
    match configuration {
        Some(configuration) => {
            for package in &configuration.packages {
                command.args(["--package", package.trim()]);
            }
        }
        None => {
            command.arg("--workspace");
            for package in &selection.excluded_packages {
                command.args(["--exclude", package]);
            }
        }
    }
    command.args(["--lib", "--bins", "--message-format=json"]);
    if let Some(configuration) = configuration {
        if configuration.no_default_features {
            command.arg("--no-default-features");
        }
        if let Some(features) = &configuration.features {
            command.args(["--features", features]);
        }
    }
    command.args(["--"]);
    command.args(["-A", "clippy::all", "--cap-lints", "warn"]);
    for (code, _) in RAW_CODES {
        // `--force-warn` is deliberately stronger than `-W`: a source-level
        // `allow` must not hide a production panic finding.
        command.args(["--force-warn", code]);
    }

    command.current_dir(&selection.root);
    let output = command_output_with_timeout(&mut command, CLIPPY_TIMEOUT)
        .context("failed to execute cargo clippy")?;
    if !output.status.success() {
        let label =
            configuration.map_or_else(|| "(default features)".to_owned(), Configuration::label);
        bail!(
            "cargo clippy {label} failed: {}",
            command_detail(&output.stdout, &output.stderr)
        );
    }

    let expected = selection.expected_keys();
    let (_, artifacts, findings) = parse_clippy_output(&output.stdout, &selection.root, &expected)?;
    let source_files = dep_info_sources(&selection.root, &artifacts, &expected)?;
    Ok((source_files, findings))
}

fn command_output_with_timeout(command: &mut Command, timeout: Duration) -> Result<Output> {
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow!("cargo clippy stdout pipe was unavailable"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow!("cargo clippy stderr pipe was unavailable"))?;
    let stdout_reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        let mut stdout = stdout;
        stdout.read_to_end(&mut bytes).map(|_| bytes)
    });
    let stderr_reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        let mut stderr = stderr;
        stderr.read_to_end(&mut bytes).map(|_| bytes)
    });
    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if start.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            let _ = stdout_reader.join();
            let _ = stderr_reader.join();
            bail!("cargo clippy timed out after {} seconds", timeout.as_secs());
        }
        thread::sleep(Duration::from_millis(50));
    };
    let stdout = stdout_reader
        .join()
        .map_err(|_| anyhow!("cargo clippy stdout reader panicked"))??;
    let stderr = stderr_reader
        .join()
        .map_err(|_| anyhow!("cargo clippy stderr reader panicked"))??;
    Ok(Output {
        status,
        stdout,
        stderr,
    })
}

fn parse_clippy_output(
    stdout: &[u8],
    root: &Path,
    expected: &HashSet<TargetKey>,
) -> Result<ClippyParse> {
    let stdout = std::str::from_utf8(stdout).context("cargo clippy emitted invalid UTF-8")?;
    let mut seen_targets = HashSet::new();
    let mut artifacts = CompilerArtifacts::new();
    let mut findings = BTreeSet::new();
    let mut build_finished = false;
    for (line_number, line) in stdout.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let message: Value = serde_json::from_str(line).with_context(|| {
            format!(
                "cargo clippy emitted malformed JSON on line {}",
                line_number + 1
            )
        })?;
        if !message.is_object() {
            bail!(
                "cargo clippy emitted a non-object JSON record on line {}",
                line_number + 1
            );
        }
        let reason = message
            .get("reason")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("cargo clippy JSON record omitted string reason"))?;
        match reason {
            "compiler-artifact" => {
                if let Some(key) = selected_target_key(&message, expected)? {
                    let filenames = message
                        .get("filenames")
                        .and_then(Value::as_array)
                        .ok_or_else(|| {
                            anyhow!("selected compiler-artifact record omitted filenames")
                        })?;
                    if filenames.is_empty() {
                        bail!("selected compiler-artifact record omitted artifact filenames");
                    }
                    let filenames = filenames
                        .iter()
                        .map(|filename| {
                            let filename = filename.as_str().ok_or_else(|| {
                                anyhow!("compiler-artifact filename was not a string")
                            })?;
                            if filename.is_empty() {
                                bail!("compiler-artifact filename was empty");
                            }
                            Ok(PathBuf::from(filename))
                        })
                        .collect::<Result<BTreeSet<_>>>()?;
                    let target = message
                        .get("target")
                        .cloned()
                        .ok_or_else(|| anyhow!("compiler-artifact record omitted target"))?;
                    let profile = message
                        .get("profile")
                        .cloned()
                        .ok_or_else(|| anyhow!("compiler-artifact record omitted profile"))?;
                    if !profile.is_object() {
                        bail!("compiler-artifact profile was not an object");
                    }
                    artifacts.push(CompilerArtifact {
                        key: key.clone(),
                        filenames,
                        target,
                        profile,
                    });
                    seen_targets.insert(key);
                }
            }
            "compiler-message" => {
                let diagnostic = message
                    .get("message")
                    .and_then(Value::as_object)
                    .ok_or_else(|| anyhow!("cargo compiler-message record omitted message"))?;
                if diagnostic.is_empty() {
                    bail!("cargo compiler-message record contained an empty message");
                }
                if selected_target_key(&message, expected)?.is_some() {
                    if let Some(finding) = parse_compiler_message(&message, root)? {
                        findings.insert(finding);
                    }
                }
            }
            "build-finished" => {
                let success = message
                    .get("success")
                    .and_then(Value::as_bool)
                    .ok_or_else(|| anyhow!("cargo build-finished record omitted success"))?;
                if !success {
                    bail!("cargo clippy reported an incomplete build");
                }
                build_finished = true;
            }
            "build-script-executed" => {}
            other => bail!("cargo clippy emitted unknown JSON reason: {other:?}"),
        }
    }
    if !build_finished {
        bail!("cargo clippy output did not contain a successful build-finished record");
    }
    if seen_targets != *expected {
        let missing = expected
            .difference(&seen_targets)
            .map(|key| format!("{}:{}", key.target_name, key.kind))
            .collect::<Vec<_>>();
        bail!("cargo clippy did not compile every production target: {missing:?}");
    }
    let missing_artifacts = expected
        .iter()
        .filter(|key| !artifacts.iter().any(|artifact| artifact.key == **key))
        .map(|key| format!("{}:{}", key.target_name, key.kind))
        .collect::<Vec<_>>();
    if !missing_artifacts.is_empty() {
        bail!("Cargo omitted compiler artifacts for production targets: {missing_artifacts:?}");
    }
    Ok((seen_targets, artifacts, findings))
}

fn selected_target_key(
    message: &Value,
    expected: &HashSet<TargetKey>,
) -> Result<Option<TargetKey>> {
    let package_id = message
        .get("package_id")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("cargo JSON record omitted package_id"))?;
    let target = message
        .get("target")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow!("cargo JSON record omitted target"))?;
    let target_name = target
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("cargo JSON target omitted name"))?;
    let kinds = target
        .get("kind")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("cargo JSON target omitted kind"))?;
    let Some(kind) = cargo_target_kind(kinds)? else {
        return Ok(None);
    };
    let key = TargetKey {
        package_id: package_id.to_owned(),
        target_name: target_name.to_owned(),
        kind,
    };
    Ok(expected.get(&key).cloned())
}

fn parse_compiler_message(message: &Value, root: &Path) -> Result<Option<Finding>> {
    let diagnostic = message
        .get("message")
        .ok_or_else(|| anyhow!("cargo compiler-message record omitted message"))?;
    let code = diagnostic
        .get("code")
        .and_then(Value::as_object)
        .and_then(|code| code.get("code"))
        .and_then(Value::as_str);
    let Some(code) = code else {
        return Ok(None);
    };
    let Some((_, kind)) = RAW_CODES.iter().find(|(known, _)| *known == code) else {
        if code.starts_with("clippy::") {
            bail!("unsupported Clippy diagnostic code: {code}");
        }
        return Ok(None);
    };
    let spans = diagnostic
        .get("spans")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("{code:?} diagnostic omitted spans"))?;
    let primary_spans = spans
        .iter()
        .filter(|span| span.get("is_primary").and_then(Value::as_bool) == Some(true))
        .collect::<Vec<_>>();
    if primary_spans.is_empty() {
        bail!("{code:?} diagnostic has no primary span");
    }
    for span in primary_spans {
        if let Some((path, line)) = resolve_call_site(span, root)? {
            return Ok(Some(Finding {
                path,
                line,
                kind: (*kind).to_owned(),
            }));
        }
    }
    bail!("{code:?} diagnostic has no local primary or expansion call-site span")
}

fn resolve_call_site(span: &Value, root: &Path) -> Result<Option<(String, usize)>> {
    if let Some(expansion) = span.get("expansion").and_then(Value::as_object) {
        if let Some(call_site) = expansion.get("span") {
            if let Some(found) = resolve_call_site(call_site, root)? {
                return Ok(Some(found));
            }
        }
    }
    let Some(file_name) = span.get("file_name").and_then(Value::as_str) else {
        return Ok(None);
    };
    let Some(line) = span.get("line_start").and_then(Value::as_u64) else {
        return Ok(None);
    };
    if line == 0 || line > usize::MAX as u64 {
        bail!("diagnostic span has invalid line number: {line}");
    }
    let path = Path::new(file_name);
    let candidate = if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    };
    let Ok(canonical) = fs::canonicalize(&candidate) else {
        return Ok(None);
    };
    if !canonical.starts_with(root) {
        return Ok(None);
    }
    let relative = canonical
        .strip_prefix(root)
        .map_err(|_| anyhow!("diagnostic span path escaped audit root"))?;
    let relative = relative
        .to_str()
        .ok_or_else(|| anyhow!("diagnostic span path is not UTF-8"))?
        .replace(std::path::MAIN_SEPARATOR, "/");
    let source = fs::read_to_string(&canonical)
        .with_context(|| format!("failed to read diagnostic source {}", canonical.display()))?;
    let line = line as usize;
    if source.lines().count() < line {
        bail!("diagnostic span line is outside source file: {relative}:{line}");
    }
    Ok(Some((relative, line)))
}

#[derive(Debug, Default)]
struct ParsedDepInfo {
    rules: Vec<MakeRule>,
}

/// A single Make rule: outputs are associated only with the sources on the
/// same rule line. Stale or synthesized dep-infos may split an artifact
/// output from its real dependencies across rules, so sources are resolved
/// per rule against the artifact output and never flattened.
#[derive(Debug, Default)]
struct MakeRule {
    outputs: BTreeSet<PathBuf>,
    sources: BTreeSet<PathBuf>,
}

#[cfg(test)]
impl ParsedDepInfo {
    fn all_sources(&self) -> BTreeSet<PathBuf> {
        self.rules
            .iter()
            .flat_map(|rule| rule.sources.iter().cloned())
            .collect()
    }
}

fn dep_info_sources(
    root: &Path,
    artifacts: &CompilerArtifacts,
    expected: &HashSet<TargetKey>,
) -> Result<BTreeSet<PathBuf>> {
    let mut sources = BTreeSet::new();
    let mut accounted = HashSet::new();
    for artifact in artifacts {
        if !expected.contains(&artifact.key) {
            continue;
        }
        let dep_infos = locate_dep_infos(root, artifact)?;
        let mut artifact_sources = BTreeSet::new();
        for (dep_info, rule_sources) in dep_infos {
            if rule_sources.is_empty() {
                bail!(
                    "dep-info {} contained no workspace-local Rust sources for the artifact output {}:{}",
                    dep_info.display(),
                    artifact.key.target_name,
                    artifact.key.kind
                );
            }
            artifact_sources.extend(rule_sources);
        }
        if let Some(src_path) = artifact_source_path(root, artifact)? {
            if !artifact_sources.contains(&src_path) {
                bail!(
                    "dep-info sources omitted compiler artifact source {} for {}:{}",
                    src_path.display(),
                    artifact.key.target_name,
                    artifact.key.kind
                );
            }
        }
        sources.extend(artifact_sources);
        accounted.insert(artifact.key.clone());
    }
    let missing = expected
        .difference(&accounted)
        .map(|key| format!("{}:{}", key.target_name, key.kind))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        bail!("missing compiler artifacts for production targets: {missing:?}");
    }
    Ok(sources)
}

fn artifact_identity(artifact: &CompilerArtifact) -> String {
    let source = artifact
        .target
        .get("src_path")
        .and_then(Value::as_str)
        .unwrap_or("<unknown source>");
    let profile = artifact
        .profile
        .get("opt_level")
        .and_then(Value::as_str)
        .unwrap_or("<unknown profile>");
    format!(
        "{}:{} ({source}, opt-level {profile})",
        artifact.key.target_name, artifact.key.kind
    )
}

fn artifact_source_path(root: &Path, artifact: &CompilerArtifact) -> Result<Option<PathBuf>> {
    let Some(source) = artifact.target.get("src_path") else {
        return Ok(None);
    };
    let source = source
        .as_str()
        .ok_or_else(|| anyhow!("compiler artifact target src_path was not a string"))?;
    ensure_inside(root, Path::new(source), "compiler artifact source").map(Some)
}

fn locate_dep_infos(
    root: &Path,
    artifact: &CompilerArtifact,
) -> Result<Vec<(PathBuf, BTreeSet<PathBuf>)>> {
    let expected_outputs = artifact
        .filenames
        .iter()
        .map(|filename| normalize_path(root, filename))
        .collect::<BTreeSet<_>>();
    let mut candidates = BTreeSet::new();
    for filename in &artifact.filenames {
        let artifact_path = if filename.is_absolute() {
            filename.clone()
        } else {
            root.join(filename)
        };
        candidates.extend(dep_info_candidates(&artifact_path));
    }

    let mut matching = Vec::new();
    for candidate in candidates {
        if !candidate.is_file() {
            continue;
        }
        let canonical = fs::canonicalize(&candidate).with_context(|| {
            format!(
                "failed to canonicalize dep-info candidate {}",
                candidate.display()
            )
        })?;
        let parsed = parse_make_dep_info(&canonical, root)?;
        // Only sources listed on rules whose outputs include an artifact
        // output belong to this artifact; an unrelated rule must never
        // satisfy the artifact's source validation.
        let mut rule_sources = BTreeSet::new();
        let mut matched = false;
        for rule in &parsed.rules {
            if rule
                .outputs
                .iter()
                .any(|output| expected_outputs.contains(output))
            {
                matched = true;
                rule_sources.extend(rule.sources.iter().cloned());
            }
        }
        if matched {
            matching.push((canonical, rule_sources));
        }
    }
    if matching.is_empty() {
        let filenames = artifact
            .filenames
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>();
        bail!(
            "no dep-info rule matched production artifact {} from compiler artifacts {filenames:?}",
            artifact_identity(artifact)
        );
    }
    Ok(matching)
}

fn normalize_path(root: &Path, path: &Path) -> PathBuf {
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    };
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            std::path::Component::RootDir => normalized.push(std::path::MAIN_SEPARATOR_STR),
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if !normalized.pop() {
                    normalized.push(component.as_os_str());
                }
            }
            std::path::Component::Normal(value) => normalized.push(value),
        }
    }
    normalized
}

fn dep_info_candidates(artifact: &Path) -> Vec<PathBuf> {
    let Some(file_name) = artifact.file_name().and_then(|name| name.to_str()) else {
        return Vec::new();
    };
    let stem = artifact
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or(file_name);
    let without_lib = stem.strip_prefix("lib").unwrap_or(stem);
    let mut candidates = Vec::new();
    if artifact.extension().and_then(|ext| ext.to_str()) == Some("d") {
        candidates.push(artifact.to_path_buf());
    }
    for name in [format!("{without_lib}.d"), format!("{stem}.d")] {
        let candidate = artifact.with_file_name(name);
        if !candidates.contains(&candidate) {
            candidates.push(candidate);
        }
    }
    candidates
}

fn parse_make_dep_info(path: &Path, root: &Path) -> Result<ParsedDepInfo> {
    let bytes =
        fs::read(path).with_context(|| format!("failed to read dep-info {}", path.display()))?;
    let source = std::str::from_utf8(&bytes)
        .with_context(|| format!("dep-info is not UTF-8: {}", path.display()))?;
    let source = join_make_lines(source)?;
    let mut saw_rule = false;
    let mut parsed = ParsedDepInfo::default();
    for (line_number, line) in source.split('\n').enumerate() {
        let trimmed = line.trim_start();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let separator = make_rule_separator(line).with_context(|| {
            format!(
                "dep-info line {} has no Make rule separator",
                line_number + 1
            )
        })?;
        saw_rule = true;
        let mut rule = MakeRule::default();
        for output in parse_make_words(&line[..separator]).with_context(|| {
            format!("dep-info line {} has invalid Make outputs", line_number + 1)
        })? {
            rule.outputs
                .insert(normalize_path(root, Path::new(&output)));
        }
        for dependency in parse_make_words(&line[separator + 1..]).with_context(|| {
            format!(
                "dep-info line {} has invalid Make dependencies",
                line_number + 1
            )
        })? {
            if Path::new(&dependency)
                .extension()
                .and_then(|ext| ext.to_str())
                != Some("rs")
            {
                continue;
            }
            let candidate = Path::new(&dependency);
            let candidate = if candidate.is_absolute() {
                candidate.to_path_buf()
            } else {
                root.join(candidate)
            };
            let canonical = match fs::canonicalize(&candidate) {
                Ok(path) => path,
                Err(error) if candidate.starts_with(root) => {
                    return Err(error).with_context(|| {
                        format!(
                            "workspace source from dep-info does not exist: {}",
                            candidate.display()
                        )
                    });
                }
                Err(_) => continue,
            };
            if canonical.starts_with(root) && canonical.is_file() {
                rule.sources.insert(canonical);
            }
        }
        parsed.rules.push(rule);
    }
    if !saw_rule {
        bail!("dep-info contained no Make rules: {}", path.display());
    }
    Ok(parsed)
}

fn join_make_lines(source: &str) -> Result<String> {
    let mut joined = String::with_capacity(source.len());
    let mut chars = source.chars().peekable();
    while let Some(character) = chars.next() {
        if character == '\\' {
            match chars.peek().copied() {
                Some('\n') => {
                    chars.next();
                    joined.push(' ');
                }
                Some('\r') => {
                    chars.next();
                    if chars.peek() == Some(&'\n') {
                        chars.next();
                    }
                    joined.push(' ');
                }
                _ => joined.push(character),
            }
        } else {
            joined.push(character);
        }
    }
    Ok(joined)
}

fn make_rule_separator(line: &str) -> Result<usize> {
    let bytes = line.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'\\' {
            index = index.saturating_add(2);
            continue;
        }
        if bytes[index] == b':' {
            return Ok(index);
        }
        index += 1;
    }
    bail!("missing unescaped rule colon")
}

fn parse_make_words(source: &str) -> Result<Vec<String>> {
    let mut words = Vec::new();
    let mut word = String::new();
    let mut chars = source.chars().peekable();
    while let Some(character) = chars.next() {
        if character.is_whitespace() {
            if !word.is_empty() {
                words.push(std::mem::take(&mut word));
            }
        } else if character == '\\' {
            let escaped = chars
                .next()
                .ok_or_else(|| anyhow!("Make word ended with an escape"))?;
            word.push(escaped);
        } else if character == '$' {
            if chars.peek() == Some(&'$') {
                chars.next();
            }
            word.push('$');
        } else {
            // rustc emits unescaped '#' in source names; unlike a general
            // Make parser, dep-info must preserve it as ordinary path data.
            word.push(character);
        }
    }
    if !word.is_empty() {
        words.push(word);
    }
    Ok(words)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CfgState {
    False,
    Unknown,
    True,
}

impl CfgState {
    fn not(self) -> Self {
        match self {
            Self::False => Self::True,
            Self::Unknown => Self::Unknown,
            Self::True => Self::False,
        }
    }
}

fn cfg_state(meta: &Meta) -> Result<CfgState> {
    match meta {
        Meta::Path(path) if path.is_ident("test") => Ok(CfgState::False),
        Meta::List(list) if list.path.is_ident("all") => {
            let metas = parse_meta_list(list)?;
            if metas.is_empty() {
                return Ok(CfgState::True);
            }
            let mut unknown = false;
            for meta in metas {
                match cfg_state(&meta)? {
                    CfgState::False => return Ok(CfgState::False),
                    CfgState::Unknown => unknown = true,
                    CfgState::True => {}
                }
            }
            Ok(if unknown {
                CfgState::Unknown
            } else {
                CfgState::True
            })
        }
        Meta::List(list) if list.path.is_ident("any") => {
            let metas = parse_meta_list(list)?;
            if metas.is_empty() {
                return Ok(CfgState::False);
            }
            let mut unknown = false;
            for meta in metas {
                match cfg_state(&meta)? {
                    CfgState::True => return Ok(CfgState::True),
                    CfgState::Unknown => unknown = true,
                    CfgState::False => {}
                }
            }
            Ok(if unknown {
                CfgState::Unknown
            } else {
                CfgState::False
            })
        }
        Meta::List(list) if list.path.is_ident("not") => {
            let metas = parse_meta_list(list)?;
            if metas.len() != 1 {
                return Ok(CfgState::Unknown);
            }
            Ok(cfg_state(&metas[0])?.not())
        }
        _ => Ok(CfgState::Unknown),
    }
}

fn parse_meta_list(list: &syn::MetaList) -> Result<Vec<Meta>> {
    list.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)
        .map(|metas| metas.into_iter().collect())
        .context("cfg attribute has invalid predicate syntax")
}

fn definitely_test_only(attrs: &[Attribute]) -> Result<bool> {
    for attr in attrs {
        if attr.path().is_ident("cfg") {
            let Meta::List(list) = &attr.meta else {
                continue;
            };
            let metas = parse_meta_list(list)?;
            if metas.len() == 1 && cfg_state(&metas[0])? == CfgState::False {
                return Ok(true);
            }
        } else if attr.path().is_ident("cfg_attr") && cfg_attr_definitely_false(&attr.meta)? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn cfg_attr_definitely_false(meta: &Meta) -> Result<bool> {
    let Meta::List(list) = meta else {
        return Ok(false);
    };
    if !list.path.is_ident("cfg_attr") {
        return Ok(false);
    }
    let metas = parse_meta_list(list)?;
    let Some(predicate) = metas.first() else {
        return Ok(false);
    };
    if cfg_state(predicate)? != CfgState::True {
        return Ok(false);
    }
    for generated in metas.iter().skip(1) {
        if generated.path().is_ident("cfg") {
            let Meta::List(list) = generated else {
                continue;
            };
            let predicates = parse_meta_list(list)?;
            if predicates.len() == 1 && cfg_state(&predicates[0])? == CfgState::False {
                return Ok(true);
            }
        } else if generated.path().is_ident("cfg_attr") && cfg_attr_definitely_false(generated)? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn assertion_kind(name: &str) -> Option<&'static str> {
    match name.strip_prefix("r#").unwrap_or(name) {
        "assert" => Some("assert"),
        "assert_eq" => Some("assert_eq"),
        "assert_ne" => Some("assert_ne"),
        "debug_assert" => Some("debug_assert"),
        "debug_assert_eq" => Some("debug_assert_eq"),
        "debug_assert_ne" => Some("debug_assert_ne"),
        _ => None,
    }
}

fn record_assertion(path: &str, line: usize, kind: &str, findings: &mut BTreeSet<Finding>) {
    if line != 0 {
        findings.insert(Finding {
            path: path.to_owned(),
            line,
            kind: kind.to_owned(),
        });
    }
}

fn scan_assertion_tokens(path: &str, tokens: TokenStream, findings: &mut BTreeSet<Finding>) {
    let tokens = tokens.into_iter().collect::<Vec<_>>();
    for (index, token) in tokens.iter().enumerate() {
        match token {
            TokenTree::Ident(ident) => {
                let metavariable = matches!(tokens.get(index.wrapping_sub(1)), Some(TokenTree::Punct(p)) if p.as_char() == '$');
                if !metavariable {
                    if let Some(kind) = assertion_kind(&ident.to_string()) {
                        if matches!(tokens.get(index + 1), Some(TokenTree::Punct(p)) if p.as_char() == '!')
                        {
                            record_assertion(path, ident.span().start().line, kind, findings);
                        }
                    }
                }
            }
            TokenTree::Group(group) => {
                scan_assertion_tokens(path, group.stream(), findings);
            }
            TokenTree::Punct(_) | TokenTree::Literal(_) => {}
        }
    }
}

fn scan_macro_rules(path: &str, tokens: TokenStream, findings: &mut BTreeSet<Finding>) {
    let tokens = tokens.into_iter().collect::<Vec<_>>();
    let mut index = 0;
    while index + 2 < tokens.len() {
        let arrow = matches!(
            (&tokens[index], &tokens[index + 1]),
            (TokenTree::Punct(equal), TokenTree::Punct(greater))
                if equal.as_char() == '=' && greater.as_char() == '>'
        );
        if arrow {
            if let Some(TokenTree::Group(transcriber)) = tokens.get(index + 2) {
                scan_macro_transcriber(path, transcriber.stream(), findings);
                index += 3;
                continue;
            }
        }
        index += 1;
    }
}

/// Scan a `macro_rules!` transcriber for actual assertion invocations while
/// recognizing nested `macro_rules!` definitions: a nested name in matcher
/// position is a pattern, not a call, so only the nested rules' transcribers
/// are scanned.
fn scan_macro_transcriber(path: &str, tokens: TokenStream, findings: &mut BTreeSet<Finding>) {
    let tokens = tokens.into_iter().collect::<Vec<_>>();
    let mut index = 0;
    while index < tokens.len() {
        match &tokens[index] {
            TokenTree::Ident(ident) if ident == "macro_rules" => {
                if let (
                    Some(TokenTree::Punct(bang)),
                    Some(TokenTree::Ident(_)),
                    Some(TokenTree::Group(body)),
                ) = (
                    tokens.get(index + 1),
                    tokens.get(index + 2),
                    tokens.get(index + 3),
                ) {
                    if bang.as_char() == '!' {
                        scan_macro_rules(path, body.stream(), findings);
                        index += 4;
                        continue;
                    }
                }
                index += 1;
            }
            TokenTree::Group(group) => {
                scan_macro_transcriber(path, group.stream(), findings);
                index += 1;
            }
            TokenTree::Ident(ident) => {
                let metavariable = matches!(
                    tokens.get(index.wrapping_sub(1)),
                    Some(TokenTree::Punct(p)) if p.as_char() == '$'
                );
                if !metavariable {
                    if let Some(kind) = assertion_kind(&ident.to_string()) {
                        if matches!(tokens.get(index + 1), Some(TokenTree::Punct(p)) if p.as_char() == '!')
                        {
                            record_assertion(path, ident.span().start().line, kind, findings);
                        }
                    }
                }
                index += 1;
            }
            TokenTree::Punct(_) | TokenTree::Literal(_) => index += 1,
        }
    }
}

fn scan_item_macro(path: &str, item: &syn::ItemMacro, findings: &mut BTreeSet<Finding>) {
    if item
        .mac
        .path
        .segments
        .last()
        .is_some_and(|segment| segment.ident == "macro_rules")
    {
        scan_macro_rules(path, item.mac.tokens.clone(), findings);
    } else {
        scan_assertion_tokens(path, item.mac.tokens.clone(), findings);
    }
}

fn line_starts(source: &str) -> Vec<usize> {
    let mut starts = vec![0];
    for (offset, byte) in source.as_bytes().iter().enumerate() {
        if *byte == b'\n' {
            starts.push(offset + 1);
        }
    }
    starts
}

fn source_line(starts: &[usize], offset: usize) -> usize {
    starts.partition_point(|start| *start <= offset)
}

fn skip_quoted(bytes: &[u8], start: usize, quote: u8) -> usize {
    let mut index = start + 1;
    while index < bytes.len() {
        match bytes[index] {
            b'\\' => index = index.saturating_add(2),
            value if value == quote => return index + 1,
            b'\n' if quote == b'\'' => return index,
            _ => index += 1,
        }
    }
    bytes.len()
}

fn raw_string_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut index = start;
    if bytes.get(index) == Some(&b'b') {
        index += 1;
    }
    if bytes.get(index) != Some(&b'r') {
        return None;
    }
    index += 1;
    while bytes.get(index) == Some(&b'#') {
        index += 1;
    }
    let hashes = index - start - 1 - usize::from(bytes.get(start) == Some(&b'b'));
    if bytes.get(index) != Some(&b'"') {
        return None;
    }
    index += 1;
    while index < bytes.len() {
        if bytes[index] == b'"'
            && bytes.get(index + 1..index + 1 + hashes) == Some(&vec![b'#'; hashes][..])
        {
            return Some(index + 1 + hashes);
        }
        index += 1;
    }
    Some(bytes.len())
}

fn skip_block_comment(bytes: &[u8], start: usize) -> usize {
    let mut depth = 1;
    let mut index = start + 2;
    while index + 1 < bytes.len() {
        if bytes[index] == b'/' && bytes[index + 1] == b'*' {
            depth += 1;
            index += 2;
        } else if bytes[index] == b'*' && bytes[index + 1] == b'/' {
            depth -= 1;
            index += 2;
            if depth == 0 {
                return index;
            }
        } else {
            index += 1;
        }
    }
    bytes.len()
}

fn skip_space_and_comments(bytes: &[u8], mut index: usize) -> usize {
    loop {
        while bytes.get(index).is_some_and(u8::is_ascii_whitespace) {
            index += 1;
        }
        if bytes.get(index..index + 2) == Some(b"//") {
            index += 2;
            while bytes.get(index).is_some_and(|byte| *byte != b'\n') {
                index += 1;
            }
            continue;
        }
        if bytes.get(index..index + 2) == Some(b"/*") {
            index = skip_block_comment(bytes, index);
            continue;
        }
        return index;
    }
}

fn scan_literal_assertions_text(source: &str, path: &str, findings: &mut BTreeSet<Finding>) {
    let bytes = source.as_bytes();
    let starts = line_starts(source);
    let mut index = 0;
    while index < bytes.len() {
        if bytes.get(index..index + 2) == Some(b"//") {
            index += 2;
            while bytes.get(index).is_some_and(|byte| *byte != b'\n') {
                index += 1;
            }
            continue;
        }
        if bytes.get(index..index + 2) == Some(b"/*") {
            index = skip_block_comment(bytes, index);
            continue;
        }
        if let Some(end) = raw_string_end(bytes, index) {
            index = end;
            continue;
        }
        if bytes[index] == b'"' {
            index = skip_quoted(bytes, index, b'"');
            continue;
        }
        if bytes[index] == b'\'' {
            let end = skip_quoted(bytes, index, b'\'');
            index = if end > index + 1 { end } else { index + 1 };
            continue;
        }
        if bytes[index] == b'b' && matches!(bytes.get(index + 1), Some(b'"' | b'\'')) {
            index = skip_quoted(bytes, index + 1, bytes[index + 1]);
            continue;
        }
        if bytes[index].is_ascii_alphabetic() || bytes[index] == b'_' {
            let start = index;
            index += 1;
            while bytes
                .get(index)
                .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
            {
                index += 1;
            }
            let name = &source[start..index];
            if let Some(kind) = assertion_kind(name) {
                let bang = skip_space_and_comments(bytes, index);
                if bytes.get(bang) == Some(&b'!') {
                    record_assertion(path, source_line(&starts, start), kind, findings);
                }
            }
            continue;
        }
        index += 1;
    }
}

struct PublicAssertionVisitor<'a> {
    path: &'a str,
    findings: &'a mut BTreeSet<Finding>,
    in_public_context: bool,
    error: Option<anyhow::Error>,
}

/// Does a token stream mention an identifier other than the boolean literals
/// `true`/`false`? Only such a helper can launder a runtime invariant through
/// an assertion.
fn tokens_name_runtime_value(tokens: TokenStream) -> bool {
    tokens.into_iter().any(|token| match token {
        TokenTree::Ident(ident) => !matches!(ident.to_string().as_str(), "true" | "false"),
        TokenTree::Group(group) => tokens_name_runtime_value(group.stream()),
        TokenTree::Punct(_) | TokenTree::Literal(_) => false,
    })
}

/// The first `count` top-level comma-separated arguments of a macro.
fn assertion_operands(tokens: TokenStream, count: usize) -> Vec<TokenStream> {
    let mut operands = vec![TokenStream::new()];
    for token in tokens {
        if matches!(&token, TokenTree::Punct(punct) if punct.as_char() == ',') {
            if operands.len() == count {
                break;
            }
            operands.push(TokenStream::new());
            continue;
        }
        if let Some(current) = operands.last_mut() {
            current.extend(std::iter::once(token));
        }
    }
    operands
}

/// Lines of every `name!` or `name(` occurrence in a token stream (including
/// nested groups), where `bang` selects the macro form.
fn token_occurrences(tokens: TokenStream, name: &str, bang: bool, lines: &mut Vec<usize>) {
    let mut iter = tokens.into_iter().peekable();
    while let Some(token) = iter.next() {
        match token {
            TokenTree::Ident(ident) if ident == name => {
                let follows = match iter.peek() {
                    Some(TokenTree::Punct(punct)) => bang && punct.as_char() == '!',
                    Some(TokenTree::Group(group)) => {
                        !bang && group.delimiter() == proc_macro2::Delimiter::Parenthesis
                    }
                    _ => false,
                };
                let line = ident.span().start().line;
                if follows && line != 0 {
                    lines.push(line);
                }
            }
            TokenTree::Group(group) => token_occurrences(group.stream(), name, bang, lines),
            _ => {}
        }
    }
}

/// The `Macro` of a statement that is a single assertion invocation, whether
/// written with a trailing semicolon or as the block's tail expression.
fn assertion_macro_statement(stmt: &Stmt) -> Option<&syn::Macro> {
    match stmt {
        Stmt::Expr(Expr::Macro(expr_macro), _) => Some(&expr_macro.mac),
        Stmt::Macro(stmt_macro) => Some(&stmt_macro.mac),
        _ => None,
    }
}

/// A private helper whose entire body is one assertion on a runtime value is
/// the laundering route for an invisible panic, so record that assertion.
/// Literal-only helpers and helpers with any other statement are left to the
/// public-path scanner.
fn assertion_only_helper(path: &str, block: &syn::Block) -> Option<Finding> {
    if block.stmts.len() != 1 {
        return None;
    }
    let mac = assertion_macro_statement(&block.stmts[0])?;
    let kind = assertion_kind(&mac.path.segments.last()?.ident.to_string())?;
    // Only the condition decides whether the assertion can fail: the first
    // argument of `assert!`, the first two of `assert_eq!`/`assert_ne!`.
    // Format arguments after them never do.
    let operands = if kind.ends_with("_eq") || kind.ends_with("_ne") {
        2
    } else {
        1
    };
    if !assertion_operands(mac.tokens.clone(), operands)
        .into_iter()
        .any(tokens_name_runtime_value)
    {
        return None;
    }
    let line = mac.span().start().line;
    if line == 0 {
        return None;
    }
    Some(Finding {
        path: path.to_owned(),
        line,
        kind: kind.to_owned(),
    })
}

fn skip_test_only_attrs(attrs: &[Attribute], error: &mut Option<anyhow::Error>) -> bool {
    match definitely_test_only(attrs) {
        Ok(skip) => skip,
        Err(err) => {
            if error.is_none() {
                *error = Some(err);
            }
            true
        }
    }
}

impl PublicAssertionVisitor<'_> {
    fn scan_items(&mut self, items: &[Item]) -> Result<()> {
        for item in items {
            if definitely_test_only(item_attrs(item))? {
                continue;
            }
            match item {
                Item::Fn(function) if is_public(&function.vis) => {
                    self.scan_block(&function.block)?;
                }
                Item::Fn(function) => {
                    if let Some(finding) = assertion_only_helper(self.path, &function.block) {
                        self.findings.insert(finding);
                    }
                }
                Item::Trait(trait_item) if is_public(&trait_item.vis) => {
                    for trait_item in &trait_item.items {
                        if definitely_test_only(trait_item_attrs(trait_item))? {
                            continue;
                        }
                        if let TraitItem::Fn(function) = trait_item {
                            if let Some(block) = &function.default {
                                self.scan_block(block)?;
                            }
                        }
                    }
                }
                Item::Impl(impl_item) => {
                    let trait_impl = impl_item.trait_.is_some();
                    for impl_item in &impl_item.items {
                        if definitely_test_only(impl_item_attrs(impl_item))? {
                            continue;
                        }
                        if let ImplItem::Fn(function) = impl_item {
                            if trait_impl || is_public(&function.vis) {
                                self.scan_block(&function.block)?;
                            } else if let Some(finding) =
                                assertion_only_helper(self.path, &function.block)
                            {
                                self.findings.insert(finding);
                            }
                        }
                    }
                }
                Item::Macro(item_macro) => {
                    scan_item_macro(self.path, item_macro, self.findings);
                }
                Item::Mod(module) => {
                    if let Some((_, items)) = &module.content {
                        self.scan_items(items)?;
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn scan_block(&mut self, block: &syn::Block) -> Result<()> {
        let previous = self.in_public_context;
        self.in_public_context = true;
        self.visit_block(block);
        self.in_public_context = previous;
        if let Some(error) = self.error.take() {
            return Err(error);
        }
        Ok(())
    }

    fn skip_attrs(&mut self, attrs: &[Attribute]) -> bool {
        skip_test_only_attrs(attrs, &mut self.error)
    }
}

impl<'ast> Visit<'ast> for PublicAssertionVisitor<'_> {
    fn visit_item(&mut self, node: &'ast Item) {
        if self.skip_attrs(item_attrs(node)) {
            return;
        }
        syn::visit::visit_item(self, node);
    }

    fn visit_stmt(&mut self, node: &'ast Stmt) {
        let attrs = match node {
            Stmt::Local(local) => &local.attrs,
            Stmt::Item(item) => item_attrs(item),
            Stmt::Expr(expr, _) => expr_attrs(expr),
            Stmt::Macro(mac) => &mac.attrs,
        };
        if self.skip_attrs(attrs) {
            return;
        }
        syn::visit::visit_stmt(self, node);
    }

    fn visit_expr(&mut self, node: &'ast Expr) {
        if self.skip_attrs(expr_attrs(node)) {
            return;
        }
        syn::visit::visit_expr(self, node);
    }

    fn visit_arm(&mut self, node: &'ast Arm) {
        if self.skip_attrs(&node.attrs) {
            return;
        }
        syn::visit::visit_arm(self, node);
    }

    fn visit_item_macro(&mut self, node: &'ast syn::ItemMacro) {
        scan_item_macro(self.path, node, self.findings);
    }

    fn visit_macro(&mut self, node: &'ast syn::Macro) {
        if !self.in_public_context {
            return;
        }
        if let Some(segment) = node.path.segments.last() {
            if let Some(kind) = assertion_kind(&segment.ident.to_string()) {
                record_assertion(self.path, node.span().start().line, kind, self.findings);
            }
        }
        scan_assertion_tokens(self.path, node.tokens.clone(), self.findings);
    }
}

/// Collects the assertion findings, the sanctioned invariant panic, and every
/// `invariant_violated` call site found while parsing the production sources.
#[derive(Debug, Default)]
struct SourceScan {
    assertions: BTreeSet<Finding>,
    sanctioned_invariant_panics: BTreeSet<Finding>,
    invariant_calls: BTreeSet<(String, usize)>,
}

fn is_panic_macro(mac: &syn::Macro) -> bool {
    mac.path
        .segments
        .last()
        .is_some_and(|segment| segment.ident == "panic")
}

/// The line of a `panic!` statement written directly in a block.
fn direct_panic_line(stmt: &Stmt) -> Option<usize> {
    let mac = match stmt {
        Stmt::Expr(Expr::Macro(expr_macro), _) => &expr_macro.mac,
        Stmt::Macro(stmt_macro) => &stmt_macro.mac,
        _ => return None,
    };
    if !is_panic_macro(mac) {
        return None;
    }
    let line = mac.span().start().line;
    (line != 0).then_some(line)
}

/// Every `panic!` occurrence (not distinct lines): parsed macro invocations
/// plus `panic!` tokens hidden inside other macros' arguments, which `syn`
/// does not parse.
#[derive(Debug, Default)]
struct PanicMacroVisitor {
    lines: Vec<usize>,
}

impl<'ast> Visit<'ast> for PanicMacroVisitor {
    fn visit_macro(&mut self, node: &'ast syn::Macro) {
        if is_panic_macro(node) {
            let line = node.span().start().line;
            if line != 0 {
                self.lines.push(line);
            }
        }
        token_occurrences(node.tokens.clone(), "panic", true, &mut self.lines);
        syn::visit::visit_macro(self, node);
    }
}

/// The single direct `panic!` of the sanctioned invariant helper, if the body
/// still holds exactly one panic and it is a direct statement. An extra direct
/// panic or a panic nested in a closure, block, or nested item makes the
/// function unsanctioned, so every panic it contains stays a finding.
fn sanctioned_invariant_panic(path: &str, items: &[Item]) -> Option<Finding> {
    let function = items.iter().find_map(|item| match item {
        Item::Fn(function) if function.sig.ident == SANCTIONED_INVARIANT_FUNCTION => Some(function),
        _ => None,
    })?;
    let mut visitor = PanicMacroVisitor::default();
    visitor.visit_block(&function.block);
    let direct = function
        .block
        .stmts
        .iter()
        .filter_map(direct_panic_line)
        .collect::<Vec<_>>();
    if visitor.lines.len() != 1 || direct != visitor.lines {
        return None;
    }
    let line = visitor.lines[0];
    Some(Finding {
        path: path.to_owned(),
        line,
        kind: "panic".to_owned(),
    })
}

struct InvariantCallVisitor<'a> {
    path: &'a str,
    calls: &'a mut BTreeSet<(String, usize)>,
    error: Option<anyhow::Error>,
}

impl<'ast> Visit<'ast> for InvariantCallVisitor<'_> {
    fn visit_item(&mut self, node: &'ast Item) {
        if skip_test_only_attrs(item_attrs(node), &mut self.error) {
            return;
        }
        syn::visit::visit_item(self, node);
    }

    fn visit_stmt(&mut self, node: &'ast Stmt) {
        let attrs = match node {
            Stmt::Local(local) => &local.attrs,
            Stmt::Item(item) => item_attrs(item),
            Stmt::Expr(expr, _) => expr_attrs(expr),
            Stmt::Macro(mac) => &mac.attrs,
        };
        if skip_test_only_attrs(attrs, &mut self.error) {
            return;
        }
        syn::visit::visit_stmt(self, node);
    }

    fn visit_expr(&mut self, node: &'ast Expr) {
        if skip_test_only_attrs(expr_attrs(node), &mut self.error) {
            return;
        }
        syn::visit::visit_expr(self, node);
    }

    fn visit_arm(&mut self, node: &'ast Arm) {
        if skip_test_only_attrs(&node.attrs, &mut self.error) {
            return;
        }
        syn::visit::visit_arm(self, node);
    }

    fn visit_macro(&mut self, node: &'ast syn::Macro) {
        // `syn` does not parse macro arguments, so find calls in their tokens.
        let mut lines = Vec::new();
        token_occurrences(
            node.tokens.clone(),
            SANCTIONED_INVARIANT_FUNCTION,
            false,
            &mut lines,
        );
        for line in lines {
            self.calls.insert((self.path.to_owned(), line));
        }
        syn::visit::visit_macro(self, node);
    }

    fn visit_use_rename(&mut self, node: &'ast syn::UseRename) {
        // A renamed import would hide call sites from this report.
        if node.ident == SANCTIONED_INVARIANT_FUNCTION && self.error.is_none() {
            self.error = Some(anyhow!(
                "{}: `{}` must not be imported under another name",
                self.path,
                SANCTIONED_INVARIANT_FUNCTION
            ));
        }
        syn::visit::visit_use_rename(self, node);
    }

    fn visit_expr_call(&mut self, node: &'ast syn::ExprCall) {
        if let Expr::Path(function) = &*node.func {
            if function
                .path
                .segments
                .last()
                .is_some_and(|segment| segment.ident == SANCTIONED_INVARIANT_FUNCTION)
            {
                let line = node.span().start().line;
                if line != 0 {
                    self.calls.insert((self.path.to_owned(), line));
                }
            }
        }
        syn::visit::visit_expr_call(self, node);
    }
}

fn scan_sources(root: &Path, source_files: &BTreeSet<PathBuf>) -> Result<SourceScan> {
    let mut scan = SourceScan::default();
    for path in source_files {
        let bytes = fs::read(path)
            .with_context(|| format!("failed to read production source {}", path.display()))?;
        let source = String::from_utf8_lossy(&bytes);
        let relative = path
            .strip_prefix(root)
            .map_err(|_| anyhow!("production source escaped audit root"))?
            .to_str()
            .ok_or_else(|| anyhow!("production source path is not UTF-8"))?
            .replace(std::path::MAIN_SEPARATOR, "/");
        match syn::parse_file(&source) {
            Ok(file) => {
                if definitely_test_only(&file.attrs)? {
                    continue;
                }
                let mut visitor = PublicAssertionVisitor {
                    path: &relative,
                    findings: &mut scan.assertions,
                    in_public_context: false,
                    error: None,
                };
                visitor.scan_items(&file.items)?;
                {
                    let mut call_visitor = InvariantCallVisitor {
                        path: &relative,
                        calls: &mut scan.invariant_calls,
                        error: None,
                    };
                    call_visitor.visit_file(&file);
                    if let Some(error) = call_visitor.error.take() {
                        return Err(error);
                    }
                }
                if relative == SANCTIONED_INVARIANT_PATH {
                    if let Some(finding) = sanctioned_invariant_panic(&relative, &file.items) {
                        scan.sanctioned_invariant_panics.insert(finding);
                    }
                }
            }
            Err(_) => {
                // Dep-info includes include! fragments and .rs data files in
                // addition to complete modules. They are part of a successful
                // compiler build, so parse failure is not a configuration error.
                // Scan tokens when possible and fall back to a string/comment
                // aware lexical check for incomplete fragments.
                if let Ok(tokens) = source.parse::<TokenStream>() {
                    scan_assertion_tokens(&relative, tokens, &mut scan.assertions);
                } else {
                    scan_literal_assertions_text(&source, &relative, &mut scan.assertions);
                }
            }
        }
    }
    Ok(scan)
}

fn expr_attrs(expr: &Expr) -> &[Attribute] {
    match expr {
        Expr::Array(expr) => &expr.attrs,
        Expr::Assign(expr) => &expr.attrs,
        Expr::Async(expr) => &expr.attrs,
        Expr::Await(expr) => &expr.attrs,
        Expr::Binary(expr) => &expr.attrs,
        Expr::Block(expr) => &expr.attrs,
        Expr::Break(expr) => &expr.attrs,
        Expr::Call(expr) => &expr.attrs,
        Expr::Cast(expr) => &expr.attrs,
        Expr::Closure(expr) => &expr.attrs,
        Expr::Const(expr) => &expr.attrs,
        Expr::Continue(expr) => &expr.attrs,
        Expr::Field(expr) => &expr.attrs,
        Expr::ForLoop(expr) => &expr.attrs,
        Expr::Group(expr) => &expr.attrs,
        Expr::If(expr) => &expr.attrs,
        Expr::Index(expr) => &expr.attrs,
        Expr::Infer(expr) => &expr.attrs,
        Expr::Let(expr) => &expr.attrs,
        Expr::Lit(expr) => &expr.attrs,
        Expr::Loop(expr) => &expr.attrs,
        Expr::Macro(expr) => &expr.attrs,
        Expr::Match(expr) => &expr.attrs,
        Expr::MethodCall(expr) => &expr.attrs,
        Expr::Paren(expr) => &expr.attrs,
        Expr::Path(expr) => &expr.attrs,
        Expr::Range(expr) => &expr.attrs,
        Expr::RawAddr(expr) => &expr.attrs,
        Expr::Reference(expr) => &expr.attrs,
        Expr::Repeat(expr) => &expr.attrs,
        Expr::Return(expr) => &expr.attrs,
        Expr::Struct(expr) => &expr.attrs,
        Expr::Try(expr) => &expr.attrs,
        Expr::TryBlock(expr) => &expr.attrs,
        Expr::Tuple(expr) => &expr.attrs,
        Expr::Unary(expr) => &expr.attrs,
        Expr::Unsafe(expr) => &expr.attrs,
        Expr::While(expr) => &expr.attrs,
        Expr::Yield(expr) => &expr.attrs,
        Expr::Verbatim(_) => &[],
        _ => &[],
    }
}

fn item_attrs(item: &Item) -> &[Attribute] {
    match item {
        Item::Const(item) => &item.attrs,
        Item::Enum(item) => &item.attrs,
        Item::ExternCrate(item) => &item.attrs,
        Item::Fn(item) => &item.attrs,
        Item::ForeignMod(item) => &item.attrs,
        Item::Impl(item) => &item.attrs,
        Item::Macro(item) => &item.attrs,
        Item::Mod(item) => &item.attrs,
        Item::Static(item) => &item.attrs,
        Item::Struct(item) => &item.attrs,
        Item::Trait(item) => &item.attrs,
        Item::TraitAlias(item) => &item.attrs,
        Item::Type(item) => &item.attrs,
        Item::Union(item) => &item.attrs,
        Item::Use(item) => &item.attrs,
        Item::Verbatim(_) => &[],
        _ => &[],
    }
}

fn trait_item_attrs(item: &TraitItem) -> &[Attribute] {
    match item {
        TraitItem::Const(item) => &item.attrs,
        TraitItem::Fn(item) => &item.attrs,
        TraitItem::Type(item) => &item.attrs,
        TraitItem::Macro(item) => &item.attrs,
        TraitItem::Verbatim(_) => &[],
        _ => &[],
    }
}

fn impl_item_attrs(item: &ImplItem) -> &[Attribute] {
    match item {
        ImplItem::Const(item) => &item.attrs,
        ImplItem::Fn(item) => &item.attrs,
        ImplItem::Type(item) => &item.attrs,
        ImplItem::Macro(item) => &item.attrs,
        ImplItem::Verbatim(_) => &[],
        _ => &[],
    }
}

fn is_public(visibility: &Visibility) -> bool {
    matches!(visibility, Visibility::Public(_))
}

fn parse_entry(value: &str) -> Result<Finding> {
    let parts = value.split(':').collect::<Vec<_>>();
    if parts.len() != 3 {
        bail!("invalid baseline entry: {value:?}");
    }
    let path = parts[0];
    let line = parts[1];
    let kind = parts[2];
    if !path.starts_with("crates/")
        || path.contains('\\')
        || path.contains("//")
        || path
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
        || Path::new(path).is_absolute()
    {
        bail!("baseline path is not normalized: {value:?}");
    }
    let parsed_line = line.parse::<usize>().ok();
    if parsed_line.filter(|line| *line > 0).is_none()
        || parsed_line.map(|line| line.to_string()).as_deref() != Some(line)
    {
        bail!("invalid baseline entry: {value:?}");
    }
    if !RAW_KINDS.contains(&kind) && !ASSERTION_KINDS.contains(&kind) {
        bail!("invalid baseline entry: {value:?}");
    }
    Ok(Finding {
        path: path.to_owned(),
        line: parsed_line.ok_or_else(|| anyhow!("invalid baseline entry: {value:?}"))?,
        kind: kind.to_owned(),
    })
}

fn load_baseline(path: &Path) -> Result<BTreeSet<Finding>> {
    let source = fs::read_to_string(path)
        .with_context(|| format!("failed to read panic baseline {}", path.display()))?;
    let values: Vec<String> = serde_json::from_str(&source).with_context(|| {
        format!(
            "panic baseline is not a JSON array of strings: {}",
            path.display()
        )
    })?;
    let mut baseline = BTreeSet::new();
    for value in values {
        let finding = parse_entry(&value)?;
        if !baseline.insert(finding) {
            bail!("panic baseline contains duplicate entries");
        }
    }
    Ok(baseline)
}

fn build_report(
    actual: &BTreeSet<Finding>,
    baseline: &BTreeSet<Finding>,
    sanctioned: &BTreeSet<Finding>,
    invariant_calls: BTreeSet<(String, usize)>,
    configuration: Option<&Configuration>,
) -> AuditReport {
    let actual = actual
        .difference(sanctioned)
        .cloned()
        .collect::<BTreeSet<_>>();
    let matched = baseline.intersection(&actual).cloned().collect::<Vec<_>>();
    let unbaselined = actual.difference(baseline).cloned().collect::<Vec<_>>();
    // A configuration selects a subset of the workspace, so baseline entries
    // outside that selection are expected to stay unmatched; only the default
    // pass owns the completeness of the baseline.
    let stale = if configuration.is_some() {
        Vec::new()
    } else {
        baseline.difference(&actual).cloned().collect::<Vec<_>>()
    };
    let forbidden_baseline = baseline
        .iter()
        .filter(|finding| RAW_KINDS.contains(&finding.kind.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    AuditReport {
        configuration: configuration.map(|configuration| configuration.name.clone()),
        matched,
        unbaselined,
        stale,
        forbidden_baseline,
        sanctioned: sanctioned.iter().cloned().collect(),
        invariant_calls: invariant_calls.into_iter().collect(),
    }
}

pub(crate) fn audit(
    root_arg: &Path,
    baseline_path: &Path,
    configuration: Option<&Configuration>,
) -> Result<AuditReport> {
    let selection = ProductionSelection::load(root_arg, configuration)?;
    let baseline = load_baseline(baseline_path)?;
    let (source_files, mut actual) = run_clippy(&selection, configuration)?;
    let scan = scan_sources(&selection.root, &source_files)?;
    actual.extend(scan.assertions);
    Ok(build_report(
        &actual,
        &baseline,
        &scan.sanctioned_invariant_panics,
        scan.invariant_calls,
        configuration,
    ))
}

#[cfg(test)]
mod tests {
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
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock before epoch")
                .as_nanos();
            let path = std::env::temp_dir().join(format!("t4a-panic-audit-{nonce}"));
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
            parse_clippy_output(output.as_bytes(), &root.path, &expected)
                .expect("valid Cargo JSON");
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
        fs::write(&malformed, b"target crates/demo/src/lib.rs\n")
            .expect("write malformed dep-info");
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
        let nested =
            parse("#[cfg_attr(not(test), cfg_attr(not(test), cfg(test)))]\npub fn f() {}\n");
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
        let tail = parse(
            "pub fn invariant_violated(context: &str) -> ! {\n    panic!(\"{context}\")\n}\n",
        );
        assert_eq!(
            sanctioned_invariant_panic(path, &tail.items)
                .expect("sanctioned")
                .entry(),
            "crates/tensor4all-tensorbackend/src/lib.rs:2:panic"
        );
        let semicolon = parse(
            "pub fn invariant_violated(context: &str) -> ! {\n    panic!(\"{context}\");\n}\n",
        );
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
        assert!(
            assertion_only_helper(path, &block("fn check() { assert_eq!(1, 1); }\n")).is_none()
        );
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

    #[test]
    fn report_removes_sanctioned_panics_before_baselining() {
        let sanctioned = parse_entry("crates/tensor4all-tensorbackend/src/lib.rs:150:panic")
            .expect("sanctioned");
        let extra =
            parse_entry("crates/tensor4all-tensorbackend/src/lib.rs:151:panic").expect("extra");
        let actual = [sanctioned.clone(), extra.clone()]
            .into_iter()
            .collect::<BTreeSet<_>>();
        let report = build_report(
            &actual,
            &BTreeSet::new(),
            &BTreeSet::from([sanctioned.clone()]),
            BTreeSet::new(),
            None,
        );
        assert_eq!(report.sanctioned, vec![sanctioned]);
        assert_eq!(report.unbaselined, vec![extra]);
    }

    #[test]
    fn baseline_lines_reject_noncanonical_numbers() {
        assert!(parse_entry("crates/demo/src/lib.rs:01:assert").is_err());
        assert!(parse_entry("crates/demo/src/lib.rs:0:assert").is_err());
        assert!(parse_entry("crates/demo/src/lib.rs:1:unknown").is_err());
        assert!(parse_entry("demo/src/lib.rs:1:assert").is_err());
    }

    #[test]
    fn report_prints_sanctioned_and_call_sites_without_failing() {
        let sanctioned =
            parse_entry("crates/tensor4all-tensorbackend/src/lib.rs:150:panic").expect("panic");
        let passing = AuditReport {
            configuration: None,
            matched: Vec::new(),
            unbaselined: Vec::new(),
            stale: Vec::new(),
            forbidden_baseline: Vec::new(),
            sanctioned: vec![sanctioned],
            invariant_calls: vec![("crates/demo/src/lib.rs".to_owned(), 9)],
        };
        assert_eq!(passing.exit_code_and_print(), 0);
        let failing = AuditReport {
            configuration: None,
            matched: Vec::new(),
            unbaselined: vec![parse_entry("crates/demo/src/lib.rs:1:assert").expect("new")],
            stale: Vec::new(),
            forbidden_baseline: Vec::new(),
            sanctioned: Vec::new(),
            invariant_calls: Vec::new(),
        };
        assert_eq!(failing.exit_code_and_print(), 1);
    }

    #[test]
    fn report_marks_new_and_stale_entries_without_baselining_raw_paths() {
        let matched = parse_entry("crates/demo/src/lib.rs:1:assert").expect("matched");
        let stale = parse_entry("crates/demo/src/lib.rs:2:assert").expect("stale");
        let raw = parse_entry("crates/demo/src/lib.rs:3:panic").expect("raw");
        let new = parse_entry("crates/demo/src/lib.rs:4:unwrap").expect("new");
        let actual = [matched.clone(), new].into_iter().collect::<BTreeSet<_>>();
        let baseline = [matched, stale, raw].into_iter().collect::<BTreeSet<_>>();
        let report = build_report(&actual, &baseline, &BTreeSet::new(), BTreeSet::new(), None);
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

    #[test]
    fn configuration_selection_does_not_report_unmatched_baseline_entries() {
        let matched = parse_entry("crates/demo/src/lib.rs:1:assert").expect("matched");
        let out_of_selection = parse_entry("crates/demo/src/lib.rs:2:assert").expect("outside");
        let actual = [matched.clone()].into_iter().collect::<BTreeSet<_>>();
        let baseline = [matched, out_of_selection]
            .into_iter()
            .collect::<BTreeSet<_>>();
        let configuration = Configuration::new(
            "diagnostics".to_owned(),
            vec!["tensor4all-treetn".to_owned()],
            false,
            Some("diagnostics".to_owned()),
        )
        .unwrap();
        let report = build_report(
            &actual,
            &baseline,
            &BTreeSet::new(),
            BTreeSet::new(),
            Some(&configuration),
        );
        assert!(report.stale.is_empty());
        assert_eq!(report.configuration.as_deref(), Some("diagnostics"));
    }
}
