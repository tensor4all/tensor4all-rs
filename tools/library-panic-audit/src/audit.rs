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
//!
//! A baseline entry is keyed on the normalized path, the kind, the enclosing
//! item, and the ordinal of that kind inside the item, ordered by line. The line
//! is recorded but never matched on: code inserted above an assertion keeps the
//! entry, which is reported as moved, while an added or a deleted assertion
//! still fails. The normalized assertion text is the secondary check, so an
//! assertion whose arguments changed is reported as changed rather than
//! accepted. `--update-baseline` rewrites the recorded lines and text; the
//! previous `path:line:kind` shape is still read and is migrated to the keyed
//! shape while it still matches the finding at its line.

use anyhow::{anyhow, bail, Context, Result};
use proc_macro2::{TokenStream, TokenTree};
use serde_json::Value;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashSet};
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

/// One discovered panic-path or assertion location, before it is keyed.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct Location {
    path: String,
    line: usize,
    /// Source column, used to place the finding inside its item. Two assertions
    /// written on one line stay distinct because their columns differ.
    column: usize,
    kind: String,
    /// Normalized assertion argument tokens, empty for Clippy findings and for
    /// text-scanned fragments where no token stream is available.
    content: String,
}

impl Location {
    fn entry(&self) -> String {
        format!("{}:{}:{}", self.path, self.line, self.kind)
    }
}

/// The reviewed identity of a finding: an entry keeps it when code above it
/// moves, and loses it when the assertion is added or deleted. The line number
/// is deliberately absent.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct FindingKey {
    path: String,
    kind: String,
    item: String,
    index: usize,
}

/// A keyed finding: the stable identity plus the reported line and the
/// normalized assertion text used as the secondary check.
#[derive(Clone, Debug)]
pub(crate) struct Finding {
    key: FindingKey,
    line: usize,
    content: String,
}

impl Finding {
    fn entry(&self) -> String {
        format!("{}:{}:{}", self.key.path, self.line, self.key.kind)
    }

    /// The key as reported next to a finding, e.g. `IdxTensor::permute#0`.
    fn key_text(&self) -> String {
        format!("{}#{}", self.key.item, self.key.index)
    }

    /// The baseline spelling of this entry.
    fn json(&self) -> String {
        serde_json::json!({
            "path": &self.key.path,
            "kind": &self.key.kind,
            "item": &self.key.item,
            "index": self.key.index,
            "line": self.line,
            "content": &self.content,
        })
        .to_string()
    }
}

// The line and the assertion text are reported, not matched on: set operations
// compare the key alone, so a baseline entry and the finding it still describes
// stay equal after the code above them moves.
impl Ord for Finding {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key.cmp(&other.key)
    }
}

impl PartialOrd for Finding {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Finding {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl Eq for Finding {}

/// One reviewed baseline entry. `Legacy` is the previous `path:line:kind`
/// shape, accepted for one release and resolved on load.
#[derive(Clone, Debug)]
enum BaselineEntry {
    Legacy(Location),
    Keyed(Finding),
}

impl BaselineEntry {
    /// How the entry is spelled in the baseline file.
    fn previous(&self) -> String {
        match self {
            BaselineEntry::Legacy(location) => location.entry(),
            BaselineEntry::Keyed(finding) => finding.json(),
        }
    }

    fn kind(&self) -> &str {
        match self {
            BaselineEntry::Legacy(location) => &location.kind,
            BaselineEntry::Keyed(finding) => &finding.key.kind,
        }
    }
}

/// A baseline entry together with the finding it describes today.
#[derive(Debug)]
struct ResolvedEntry {
    previous: String,
    current: Finding,
}

#[derive(Debug)]
pub(crate) struct AuditReport {
    configuration: Option<String>,
    baseline_path: PathBuf,
    matched: Vec<Finding>,
    /// Findings that kept their entry but moved: reported, never fatal.
    moved: Vec<(Finding, usize)>,
    /// Same key, different assertion text: the reviewed assertion changed.
    changed: Vec<(Finding, Finding)>,
    unbaselined: Vec<Finding>,
    stale: Vec<String>,
    forbidden_baseline: Vec<String>,
    sanctioned: Vec<Finding>,
    invariant_calls: Vec<(String, usize)>,
    resolved: Vec<ResolvedEntry>,
}

impl AuditReport {
    fn failed(&self) -> bool {
        !self.unbaselined.is_empty()
            || !self.stale.is_empty()
            || !self.changed.is_empty()
            || !self.forbidden_baseline.is_empty()
    }

    pub(crate) fn exit_code_and_print(&self) -> i32 {
        if let Some(configuration) = &self.configuration {
            println!("Audit configuration: `{configuration}`");
        }
        for finding in &self.matched {
            println!("Baseline matched: {}", finding.entry());
        }
        for (finding, baseline_line) in &self.moved {
            println!(
                "Moved baseline entry: {} [{}] (baseline recorded line {baseline_line})",
                finding.entry(),
                finding.key_text()
            );
        }
        for (finding, baseline) in &self.changed {
            println!(
                "Changed assertion: {} [{}] (baseline `{}`, now `{}`)",
                finding.entry(),
                finding.key_text(),
                baseline.content,
                finding.content
            );
        }
        for finding in &self.sanctioned {
            println!("Sanctioned invariant panic: {}", finding.entry());
        }
        for finding in &self.unbaselined {
            println!("{}", finding.entry());
        }
        for entry in &self.stale {
            println!("Stale baseline: {entry}");
        }
        for entry in &self.forbidden_baseline {
            eprintln!("Baseline contains forbidden raw panic-style entry: {entry}");
        }

        println!(
            "invariant_violated call sites: {}",
            self.invariant_calls.len()
        );
        for (path, line) in &self.invariant_calls {
            println!("invariant_violated call site: {path}:{line}");
        }
        if !self.unbaselined.is_empty() {
            println!("Unbaselined baseline entries:");
            for finding in &self.unbaselined {
                println!("  {}", finding.json());
            }
        }

        if self.failed() {
            let mut message = format!(
                "Audit failed: {} {}, {} {}",
                self.unbaselined.len(),
                plural(self.unbaselined.len(), "unbaselined finding", None),
                self.stale.len(),
                plural(
                    self.stale.len(),
                    "stale baseline entry",
                    Some("stale baseline entries")
                )
            );
            if !self.changed.is_empty() {
                message.push_str(&format!(
                    ", {} {}",
                    self.changed.len(),
                    plural(self.changed.len(), "changed assertion", None)
                ));
            }
            eprintln!("{message}.");
            1
        } else {
            let mut message = format!(
                "Audit passed: 0 unbaselined findings, {} stale baseline entries",
                self.stale.len()
            );
            if !self.moved.is_empty() {
                message.push_str(&format!(
                    ", {} moved {}",
                    self.moved.len(),
                    plural(self.moved.len(), "entry", None)
                ));
            }
            println!("{message}");
            0
        }
    }

    /// Rewrites the baseline in the keyed format, printing every entry it
    /// changed and returning those lines. Line movement and format migration are
    /// mechanical; a changed set of keys is not, so that case is refused.
    pub(crate) fn update_baseline(&self, ci: Option<&str>) -> Result<Vec<String>> {
        if ci_environment(ci) {
            bail!(
                "refusing to update {} in CI: run it locally and commit the result",
                self.baseline_path.display()
            );
        }
        if self.failed() {
            bail!(
                "refusing to update {}: the baseline keys changed; resolve the added, stale, \
                 changed, and forbidden entries by hand first",
                self.baseline_path.display()
            );
        }
        let mut entries = Vec::new();
        let mut diff = Vec::new();
        for entry in &self.resolved {
            let current = entry.current.json();
            if entry.previous != current {
                diff.push(format!("- {}", entry.previous));
                diff.push(format!("+ {current}"));
            }
            entries.push(current);
        }
        let text = format!("[\n  {}\n]\n", entries.join(",\n  "));
        fs::write(&self.baseline_path, text).with_context(|| {
            format!(
                "failed to write panic baseline {}",
                self.baseline_path.display()
            )
        })?;
        for line in &diff {
            println!("{line}");
        }
        println!(
            "Baseline updated: {} of {} entries rewritten in {}",
            diff.len() / 2,
            self.resolved.len(),
            self.baseline_path.display()
        );
        Ok(diff)
    }
}

/// CI is detected from the conventional `CI` variable: any non-empty value
/// other than `0` or `false`.
fn ci_environment(value: Option<&str>) -> bool {
    value.is_some_and(|value| {
        let value = value.trim();
        !value.is_empty()
            && !value.eq_ignore_ascii_case("0")
            && !value.eq_ignore_ascii_case("false")
    })
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
type ClippyParse = (HashSet<TargetKey>, CompilerArtifacts, BTreeSet<Location>);

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
) -> Result<(BTreeSet<PathBuf>, BTreeSet<Location>)> {
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

fn parse_compiler_message(message: &Value, root: &Path) -> Result<Option<Location>> {
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
        if let Some(position) = resolve_call_site(span, root)? {
            return Ok(Some(Location {
                path: position.path,
                line: position.line,
                column: position.column,
                kind: (*kind).to_owned(),
                content: String::new(),
            }));
        }
    }
    bail!("{code:?} diagnostic has no local primary or expansion call-site span")
}

/// A resolved call site: the normalized path, and the position Clippy reported
/// (column 0 when the diagnostic carried none).
struct CallSite {
    path: String,
    line: usize,
    column: usize,
}

fn resolve_call_site(span: &Value, root: &Path) -> Result<Option<CallSite>> {
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
    // Clippy reports 1-based columns; proc-macro2 counts from 0.
    let column = span
        .get("column_start")
        .and_then(Value::as_u64)
        .and_then(|column| usize::try_from(column).ok())
        .unwrap_or(1)
        .saturating_sub(1);
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
    Ok(Some(CallSite {
        path: relative,
        line,
        column,
    }))
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

/// The normalized assertion arguments. `TokenStream` re-renders its tokens, so
/// source formatting and line breaks never reach the reviewed text.
fn assertion_content(tokens: Option<&TokenTree>) -> String {
    match tokens {
        Some(TokenTree::Group(group)) => render_tokens(&group.stream()),
        _ => String::new(),
    }
}

fn record_assertion(
    path: &str,
    position: proc_macro2::LineColumn,
    kind: &str,
    content: String,
    findings: &mut BTreeSet<Location>,
) {
    if position.line != 0 {
        findings.insert(Location {
            path: path.to_owned(),
            line: position.line,
            column: position.column,
            kind: kind.to_owned(),
            content,
        });
    }
}

/// The two-character punctuation pairs that spell one Rust operator. Only these
/// keep the token boundary in the rendering, so an accidental adjacency such as
/// `|&` renders the same however it is spaced.
const COMPOUND_PUNCT: [(char, char); 20] = [
    ('&', '&'),
    ('&', '='),
    ('*', '='),
    ('+', '='),
    ('-', '='),
    ('-', '>'),
    ('.', '.'),
    ('/', '='),
    (':', ':'),
    ('<', '<'),
    ('<', '='),
    ('=', '='),
    ('=', '>'),
    ('>', '='),
    ('>', '>'),
    ('%', '='),
    ('^', '='),
    ('|', '='),
    ('|', '|'),
    ('!', '='),
];

/// Renders tokens with a single space only where two tokens would otherwise
/// merge: between two word-like tokens, and where two standalone punctuation
/// characters spell a compound operator. The result never depends on source
/// layout (`|&row|` and `| & row |` render alike), a literal keeps its exact
/// text, and `&&` stays distinct from `& &`.
fn render_tokens(tokens: &TokenStream) -> String {
    fn word(token: &TokenTree) -> bool {
        matches!(token, TokenTree::Ident(_) | TokenTree::Literal(_))
    }
    let mut rendered = String::new();
    let mut previous_word = false;
    let mut previous_punct: Option<(char, proc_macro2::Spacing)> = None;
    for token in tokens.clone() {
        let text = match &token {
            TokenTree::Group(group) => {
                let inner = render_tokens(&group.stream());
                match group.delimiter() {
                    proc_macro2::Delimiter::Parenthesis => format!("({inner})"),
                    proc_macro2::Delimiter::Brace => format!("{{{inner}}}"),
                    proc_macro2::Delimiter::Bracket => format!("[{inner}]"),
                    proc_macro2::Delimiter::None => inner,
                }
            }
            TokenTree::Ident(ident) => ident.to_string(),
            TokenTree::Literal(literal) => literal.to_string(),
            TokenTree::Punct(punct) => punct.as_char().to_string(),
        };
        let current_word = word(&token);
        if previous_word && current_word {
            rendered.push(' ');
        } else if let TokenTree::Punct(punct) = &token {
            let separates = previous_punct.is_some_and(|(previous, spacing)| {
                spacing == proc_macro2::Spacing::Alone
                    && COMPOUND_PUNCT.contains(&(previous, punct.as_char()))
            });
            if separates {
                rendered.push(' ');
            }
        }
        rendered.push_str(&text);
        previous_word = current_word;
        previous_punct = match &token {
            TokenTree::Punct(punct) => Some((punct.as_char(), punct.spacing())),
            _ => None,
        };
    }
    rendered
}

fn scan_assertion_tokens(path: &str, tokens: TokenStream, findings: &mut BTreeSet<Location>) {
    let tokens = tokens.into_iter().collect::<Vec<_>>();
    for (index, token) in tokens.iter().enumerate() {
        match token {
            TokenTree::Ident(ident) => {
                let metavariable = matches!(tokens.get(index.wrapping_sub(1)), Some(TokenTree::Punct(p)) if p.as_char() == '$');
                if !metavariable {
                    if let Some(kind) = assertion_kind(&ident.to_string()) {
                        if matches!(tokens.get(index + 1), Some(TokenTree::Punct(p)) if p.as_char() == '!')
                        {
                            record_assertion(
                                path,
                                ident.span().start(),
                                kind,
                                assertion_content(tokens.get(index + 2)),
                                findings,
                            );
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

fn scan_macro_rules(path: &str, tokens: TokenStream, findings: &mut BTreeSet<Location>) {
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
fn scan_macro_transcriber(path: &str, tokens: TokenStream, findings: &mut BTreeSet<Location>) {
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
                            record_assertion(
                                path,
                                ident.span().start(),
                                kind,
                                assertion_content(tokens.get(index + 2)),
                                findings,
                            );
                        }
                    }
                }
                index += 1;
            }
            TokenTree::Punct(_) | TokenTree::Literal(_) => index += 1,
        }
    }
}

fn scan_item_macro(path: &str, item: &syn::ItemMacro, findings: &mut BTreeSet<Location>) {
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

fn scan_literal_assertions_text(source: &str, path: &str, findings: &mut BTreeSet<Location>) {
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
                    let line = source_line(&starts, start);
                    record_assertion(
                        path,
                        proc_macro2::LineColumn {
                            line,
                            column: start - starts[line - 1],
                        },
                        kind,
                        String::new(),
                        findings,
                    );
                }
            }
            continue;
        }
        index += 1;
    }
}

struct PublicAssertionVisitor<'a> {
    path: &'a str,
    findings: &'a mut BTreeSet<Location>,
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
fn assertion_only_helper(path: &str, block: &syn::Block) -> Option<Location> {
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
    let position = mac.span().start();
    if position.line == 0 {
        return None;
    }
    Some(Location {
        path: path.to_owned(),
        line: position.line,
        column: position.column,
        kind: kind.to_owned(),
        content: render_tokens(&mac.tokens),
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
                record_assertion(
                    self.path,
                    node.span().start(),
                    kind,
                    render_tokens(&node.tokens),
                    self.findings,
                );
            }
        }
        scan_assertion_tokens(self.path, node.tokens.clone(), self.findings);
    }
}

/// One named Rust item and the source positions it covers.
#[derive(Debug)]
struct ItemRange {
    start: proc_macro2::LineColumn,
    end: proc_macro2::LineColumn,
    name: String,
}

/// Item name for a file whose syntax tree could not be parsed, and for a
/// finding outside every item.
const FILE_SCOPE: &str = "<file>";

/// The name of the item a finding sits in. A trait implementation carries the
/// trait as well, because one type can implement the same method name both
/// inherently and through a trait.
fn item_segment(item: &Item) -> Option<String> {
    Some(match item {
        Item::Const(item) => item.ident.to_string(),
        Item::Enum(item) => item.ident.to_string(),
        Item::Fn(item) => item.sig.ident.to_string(),
        Item::Impl(item) => {
            let ty = tokens_label(item.self_ty.as_ref());
            match &item.trait_ {
                Some((_, path, _)) => format!("{ty} as {}", tokens_label(path)),
                None => ty,
            }
        }
        Item::Macro(item) => item.ident.as_ref()?.to_string(),
        Item::Mod(item) => item.ident.to_string(),
        Item::Static(item) => item.ident.to_string(),
        Item::Struct(item) => item.ident.to_string(),
        Item::Trait(item) => item.ident.to_string(),
        Item::Type(item) => item.ident.to_string(),
        Item::Union(item) => item.ident.to_string(),
        _ => return None,
    })
}

/// Renders a syntactic fragment completely, then collapses the whitespace
/// `TokenStream` adds around punctuation. Keeping the full syntax means
/// `Foo<u8>::check`, `&mut u8`, and `[u8; 4]` stay different items.
fn tokens_label(tokens: impl quote::ToTokens) -> String {
    render_tokens(&tokens.to_token_stream())
}

fn collect_block_items(block: &syn::Block, prefix: &str, ranges: &mut Vec<ItemRange>) {
    for statement in &block.stmts {
        if let Stmt::Item(item) = statement {
            collect_items(std::slice::from_ref(item), prefix, ranges);
        }
    }
}

fn qualified(prefix: &str, segment: &str) -> String {
    if prefix.is_empty() {
        segment.to_owned()
    } else {
        format!("{prefix}::{segment}")
    }
}

fn push_range(prefix: &str, segment: &str, span: proc_macro2::Span, ranges: &mut Vec<ItemRange>) {
    let start = span.start();
    let end = span.end();
    if start.line != 0 && end >= start {
        ranges.push(ItemRange {
            start,
            end,
            name: qualified(prefix, segment),
        });
    }
}

fn push_function(
    prefix: &str,
    ident: &proc_macro2::Ident,
    span: proc_macro2::Span,
    block: &syn::Block,
    ranges: &mut Vec<ItemRange>,
) {
    let segment = ident.to_string();
    push_range(prefix, &segment, span, ranges);
    collect_block_items(block, &qualified(prefix, &segment), ranges);
}

fn collect_items(items: &[Item], prefix: &str, ranges: &mut Vec<ItemRange>) {
    for item in items {
        let Some(segment) = item_segment(item) else {
            continue;
        };
        push_range(prefix, &segment, item.span(), ranges);
        let name = qualified(prefix, &segment);
        match item {
            Item::Fn(function) => {
                collect_block_items(&function.block, &name, ranges);
            }
            Item::Impl(impl_item) => {
                for impl_item in &impl_item.items {
                    if let ImplItem::Fn(function) = impl_item {
                        let segment = function.sig.ident.to_string();
                        push_range(&name, &segment, function.span(), ranges);
                        collect_block_items(&function.block, &qualified(&name, &segment), ranges);
                    }
                }
            }
            Item::Mod(module) => {
                if let Some((_, items)) = &module.content {
                    collect_items(items, &name, ranges);
                }
            }
            Item::Trait(trait_item) => {
                for trait_item in &trait_item.items {
                    if let TraitItem::Fn(function) = trait_item {
                        if let Some(block) = &function.default {
                            push_function(
                                &name,
                                &function.sig.ident,
                                function.span(),
                                block,
                                ranges,
                            );
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

/// The innermost item whose span contains `position`. Comparing positions, not
/// lines, keeps two items written on one line apart.
fn enclosing_item(ranges: &[ItemRange], position: proc_macro2::LineColumn) -> Option<&str> {
    ranges
        .iter()
        .filter(|range| range.start <= position && position <= range.end)
        .max_by_key(|range| range.start)
        .map(|range| range.name.as_str())
}

/// Gives every discovered location its reviewed key: the enclosing item and the
/// ordinal of the finding among the same kind inside that item, ordered by line.
/// Only the relative order matters, so code inserted above an assertion cannot
/// change its key.
fn key_findings(
    items: &BTreeMap<String, Vec<ItemRange>>,
    locations: BTreeSet<Location>,
) -> BTreeSet<Finding> {
    let mut grouped: BTreeMap<(String, String, String), Vec<Location>> = BTreeMap::new();
    for location in locations {
        let item = items
            .get(&location.path)
            .and_then(|ranges| {
                enclosing_item(
                    ranges,
                    proc_macro2::LineColumn {
                        line: location.line,
                        column: location.column,
                    },
                )
            })
            .unwrap_or(FILE_SCOPE)
            .to_owned();
        grouped
            .entry((location.path.clone(), location.kind.clone(), item))
            .or_default()
            .push(location);
    }
    let mut findings = BTreeSet::new();
    for ((_, _, item), mut group) in grouped {
        group.sort_by_key(|location| (location.line, location.column));
        for (index, location) in group.into_iter().enumerate() {
            findings.insert(Finding {
                key: FindingKey {
                    path: location.path,
                    kind: location.kind,
                    item: item.clone(),
                    index,
                },
                line: location.line,
                content: location.content,
            });
        }
    }
    findings
}

/// Collects the assertion findings, the sanctioned invariant panic, and every
/// `invariant_violated` call site found while parsing the production sources.
#[derive(Debug, Default)]
struct SourceScan {
    assertions: BTreeSet<Location>,
    items: BTreeMap<String, Vec<ItemRange>>,
    sanctioned_invariant_panics: BTreeSet<Location>,
    invariant_calls: BTreeSet<(String, usize)>,
}

fn is_panic_macro(mac: &syn::Macro) -> bool {
    mac.path
        .segments
        .last()
        .is_some_and(|segment| segment.ident == "panic")
}

/// The position of a `panic!` statement written directly in a block.
fn direct_panic_position(stmt: &Stmt) -> Option<proc_macro2::LineColumn> {
    let mac = match stmt {
        Stmt::Expr(Expr::Macro(expr_macro), _) => &expr_macro.mac,
        Stmt::Macro(stmt_macro) => &stmt_macro.mac,
        _ => return None,
    };
    if !is_panic_macro(mac) {
        return None;
    }
    let position = mac.span().start();
    (position.line != 0).then_some(position)
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
fn sanctioned_invariant_panic(path: &str, items: &[Item]) -> Option<Location> {
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
        .filter_map(direct_panic_position)
        .collect::<Vec<_>>();
    if visitor.lines.len() != 1
        || direct
            .iter()
            .map(|position| position.line)
            .ne(visitor.lines.iter().copied())
    {
        return None;
    }
    // The real position, so this entry keys like the Clippy finding for the same
    // panic even when the function shares a line with another item.
    let position = direct.first()?;
    Some(Location {
        path: path.to_owned(),
        line: position.line,
        column: position.column,
        kind: "panic".to_owned(),
        content: String::new(),
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
                let mut ranges = Vec::new();
                collect_items(&file.items, "", &mut ranges);
                scan.items.insert(relative.clone(), ranges);
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

/// One reviewed baseline entry: the legacy `path:line:kind` string or the keyed
/// object. Both shapes are accepted for one release of the keyed format.
fn parse_baseline_entry(value: &Value) -> Result<BaselineEntry> {
    match value {
        Value::String(text) => Ok(BaselineEntry::Legacy(parse_legacy_entry(text)?)),
        Value::Object(map) => Ok(BaselineEntry::Keyed(parse_keyed_entry(map)?)),
        _ => bail!("invalid baseline entry: {value}"),
    }
}

fn validate_path(path: &str, value: &Value) -> Result<()> {
    if !path.starts_with("crates/")
        || path.contains('\\')
        || path.contains("//")
        || path
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
        || Path::new(path).is_absolute()
    {
        bail!("baseline path is not normalized: {value}");
    }
    Ok(())
}

fn validate_kind(kind: &str, value: &Value) -> Result<()> {
    if !RAW_KINDS.contains(&kind) && !ASSERTION_KINDS.contains(&kind) {
        bail!("invalid baseline entry: {value}");
    }
    Ok(())
}

fn parse_legacy_entry(value: &str) -> Result<Location> {
    let parts = value.split(':').collect::<Vec<_>>();
    if parts.len() != 3 {
        bail!("invalid baseline entry: {value:?}");
    }
    let (path, line, kind) = (parts[0], parts[1], parts[2]);
    let parsed = Value::String(value.to_owned());
    validate_path(path, &parsed)?;
    let parsed_line = line.parse::<usize>().ok();
    if parsed_line.filter(|line| *line > 0).is_none()
        || parsed_line.map(|line| line.to_string()).as_deref() != Some(line)
    {
        bail!("invalid baseline entry: {value:?}");
    }
    validate_kind(kind, &parsed)?;
    Ok(Location {
        path: path.to_owned(),
        line: parsed_line.ok_or_else(|| anyhow!("invalid baseline entry: {value:?}"))?,
        column: 0,
        kind: kind.to_owned(),
        content: String::new(),
    })
}

fn entry_field<'a>(map: &'a serde_json::Map<String, Value>, name: &str) -> Result<&'a Value> {
    map.get(name)
        .ok_or_else(|| anyhow!("baseline entry omitted {name:?}"))
}

fn entry_number(map: &serde_json::Map<String, Value>, name: &str) -> Result<usize> {
    let value = entry_field(map, name)?;
    let number = value
        .as_u64()
        .and_then(|number| usize::try_from(number).ok())
        .ok_or_else(|| anyhow!("baseline entry {name:?} is not a non-negative integer"))?;
    Ok(number)
}

fn entry_text<'a>(map: &'a serde_json::Map<String, Value>, name: &str) -> Result<&'a str> {
    entry_field(map, name)?
        .as_str()
        .ok_or_else(|| anyhow!("baseline entry {name:?} is not a string"))
}

fn parse_keyed_entry(map: &serde_json::Map<String, Value>) -> Result<Finding> {
    let known = ["path", "kind", "item", "index", "line", "content"];
    if let Some(unknown) = map.keys().find(|key| !known.contains(&key.as_str())) {
        bail!("baseline entry has an unknown field: {unknown:?}");
    }
    let previous = Value::Object(map.clone());
    let path = entry_text(map, "path")?;
    let kind = entry_text(map, "kind")?;
    let item = entry_text(map, "item")?;
    let content = entry_text(map, "content")?;
    let index = entry_number(map, "index")?;
    let line = entry_number(map, "line")?;
    validate_path(path, &previous)?;
    validate_kind(kind, &previous)?;
    if item.is_empty() {
        bail!("baseline entry item is empty: {previous}");
    }
    if line == 0 {
        bail!("baseline entry line is not positive: {previous}");
    }
    Ok(Finding {
        key: FindingKey {
            path: path.to_owned(),
            kind: kind.to_owned(),
            item: item.to_owned(),
            index,
        },
        line,
        content: content.to_owned(),
    })
}

fn load_baseline(path: &Path) -> Result<Vec<BaselineEntry>> {
    let source = fs::read_to_string(path)
        .with_context(|| format!("failed to read panic baseline {}", path.display()))?;
    let values: Vec<Value> = serde_json::from_str(&source).with_context(|| {
        format!(
            "panic baseline is not a JSON array of baseline entries: {}",
            path.display()
        )
    })?;
    let mut baseline = Vec::new();
    let mut keys = BTreeSet::new();
    let mut legacy = BTreeSet::new();
    for value in values {
        let entry = parse_baseline_entry(&value)?;
        let duplicate = match &entry {
            BaselineEntry::Keyed(finding) => !keys.insert(finding.clone()),
            BaselineEntry::Legacy(location) => {
                !legacy.insert((location.path.clone(), location.line, location.kind.clone()))
            }
        };
        if duplicate {
            bail!("panic baseline contains duplicate entries");
        }
        baseline.push(entry);
    }
    Ok(baseline)
}

/// Resolves each baseline entry to the finding it describes now. A legacy entry
/// has no item or ordinal to key on, so it can only be migrated while it still
/// matches the finding at its line; otherwise it is reported stale rather than
/// guessed, because the line alone no longer identifies an assertion. Keyed
/// entries claim their identity first, so mixing the two shapes in one file
/// cannot make the outcome depend on the order of the entries.
fn resolve_baseline(
    entries: Vec<BaselineEntry>,
    actual: &BTreeSet<Finding>,
) -> (BTreeMap<Finding, String>, Vec<BaselineEntry>) {
    let mut resolved = BTreeMap::new();
    let mut legacy = Vec::new();
    for entry in entries {
        match entry {
            BaselineEntry::Keyed(finding) => {
                let previous = finding.json();
                resolved.insert(finding, previous);
            }
            BaselineEntry::Legacy(location) => legacy.push(location),
        }
    }
    let mut unresolved = Vec::new();
    for location in legacy {
        let found = actual.iter().find(|finding| {
            finding.key.path == location.path
                && finding.line == location.line
                && finding.key.kind == location.kind
                && !resolved.contains_key(*finding)
        });
        match found {
            Some(finding) => {
                resolved.insert(finding.clone(), location.entry());
            }
            None => unresolved.push(BaselineEntry::Legacy(location)),
        }
    }
    (resolved, unresolved)
}

#[derive(Debug, Default)]
struct Paired {
    /// A current finding and the baseline entry it keeps.
    kept: Vec<(Finding, Finding)>,
    added: Vec<Finding>,
    stale: Vec<Finding>,
}

fn group_key(finding: &Finding) -> (&str, &str, &str) {
    (&finding.key.path, &finding.key.kind, &finding.key.item)
}

/// Actual findings and baseline entries sharing one (path, kind, item).
type FindingGroups<'a> =
    BTreeMap<(&'a str, &'a str, &'a str), (Vec<&'a Finding>, Vec<&'a Finding>)>;

/// One item's findings and entries while they are being paired.
struct GroupPairing<'a> {
    actuals: Vec<&'a Finding>,
    entries: Vec<&'a Finding>,
    actual_used: Vec<bool>,
    entry_used: Vec<bool>,
}

impl<'a> GroupPairing<'a> {
    /// Pairs every still-unpaired entry with the finding of the same recorded
    /// ordinal, when that finding is still free.
    fn pair_ordinals(&mut self, paired: &mut Paired) {
        for entry_index in 0..self.entries.len() {
            if self.entry_used[entry_index] {
                continue;
            }
            let index = self.entries[entry_index].key.index;
            let found = self
                .actuals
                .iter()
                .enumerate()
                .position(|(position, actual)| {
                    !self.actual_used[position] && actual.key.index == index
                });
            self.pair(entry_index, found, paired);
        }
    }

    /// Pairs every still-unpaired entry carrying the same normalized assertion
    /// text, which is how an insertion or a deletion is told apart from its
    /// neighbours.
    fn pair_text(&mut self, paired: &mut Paired) {
        for entry_index in 0..self.entries.len() {
            let content = &self.entries[entry_index].content;
            if self.entry_used[entry_index] || content.is_empty() {
                continue;
            }
            let found = self
                .actuals
                .iter()
                .enumerate()
                .position(|(position, actual)| {
                    !self.actual_used[position] && actual.content == *content
                });
            self.pair(entry_index, found, paired);
        }
    }

    fn pair(&mut self, entry_index: usize, found: Option<usize>, paired: &mut Paired) {
        if let Some(actual_index) = found {
            self.actual_used[actual_index] = true;
            self.entry_used[entry_index] = true;
            paired.kept.push((
                (*self.actuals[actual_index]).clone(),
                (*self.entries[entry_index]).clone(),
            ));
        }
    }

    fn record_leftovers(&mut self, paired: &mut Paired) {
        for (index, actual) in self.actuals.iter().enumerate() {
            if !self.actual_used[index] {
                paired.added.push((*actual).clone());
            }
        }
        for (index, entry) in self.entries.iter().enumerate() {
            if !self.entry_used[index] {
                paired.stale.push((*entry).clone());
            }
        }
    }
}

/// Pairs baseline entries with actual findings. With as many findings as
/// entries, the recorded ordinal decides, so a rewrite or a reordering inside
/// one item is visible. When the counts differ, equal normalized assertion text
/// pairs first, so an insertion or a deletion reports the assertion that
/// actually appeared or disappeared instead of blaming a neighbour.
fn pair_findings(actual: &BTreeSet<Finding>, baseline: &BTreeSet<Finding>) -> Paired {
    let mut groups = FindingGroups::new();
    for finding in actual {
        groups
            .entry(group_key(finding))
            .or_default()
            .0
            .push(finding);
    }
    for finding in baseline {
        groups
            .entry(group_key(finding))
            .or_default()
            .1
            .push(finding);
    }
    let mut paired = Paired::default();
    for (_, (actuals, entries)) in groups {
        let mut group = GroupPairing {
            actual_used: vec![false; actuals.len()],
            entry_used: vec![false; entries.len()],
            actuals,
            entries,
        };
        if group.actuals.len() == group.entries.len() {
            group.pair_ordinals(&mut paired);
        } else {
            group.pair_text(&mut paired);
            group.pair_ordinals(&mut paired);
        }
        group.record_leftovers(&mut paired);
    }
    paired
}

fn build_report(
    actual: &BTreeSet<Finding>,
    entries: Vec<BaselineEntry>,
    sanctioned: &BTreeSet<Finding>,
    invariant_calls: BTreeSet<(String, usize)>,
    configuration: Option<&Configuration>,
    baseline_path: &Path,
) -> AuditReport {
    // The sanction is one exact source location. Keys alone cannot carry it,
    // because two same-named functions in one file share an item name.
    let actual = actual
        .iter()
        .filter(|finding| {
            !sanctioned
                .iter()
                .any(|sanctioned| sanctioned.key == finding.key && sanctioned.line == finding.line)
        })
        .cloned()
        .collect::<BTreeSet<_>>();
    let (resolved_map, unresolved) = resolve_baseline(entries, &actual);
    let baseline = resolved_map.keys().cloned().collect::<BTreeSet<_>>();
    let paired = pair_findings(&actual, &baseline);

    let mut matched = Vec::new();
    let mut moved = Vec::new();
    let mut changed = Vec::new();
    let mut resolved = Vec::new();
    for (current, recorded) in paired.kept {
        if current.content != recorded.content {
            changed.push((current, recorded));
            continue;
        }
        if current.line != recorded.line || current.key.index != recorded.key.index {
            moved.push((current.clone(), recorded.line));
        } else {
            matched.push(current.clone());
        }
        let previous = resolved_map.get(&recorded).cloned().unwrap_or_default();
        resolved.push(ResolvedEntry { previous, current });
    }

    // A configuration selects a subset of the workspace, so baseline entries
    // outside that selection are expected to stay unmatched; only the default
    // pass owns the completeness of the baseline.
    let mut stale = if configuration.is_some() {
        Vec::new()
    } else {
        paired
            .stale
            .iter()
            .map(|finding| finding.json())
            .collect::<Vec<_>>()
    };
    if configuration.is_none() {
        stale.extend(unresolved.iter().map(BaselineEntry::previous));
    }
    stale.sort();

    let mut forbidden_baseline = resolved_map
        .keys()
        .filter(|finding| RAW_KINDS.contains(&finding.key.kind.as_str()))
        .map(|finding| finding.json())
        .chain(
            unresolved
                .iter()
                .filter(|entry| RAW_KINDS.contains(&entry.kind()))
                .map(BaselineEntry::previous),
        )
        .collect::<Vec<_>>();
    forbidden_baseline.sort();

    let mut unbaselined = paired.added;
    sort_by_location(&mut unbaselined);
    sort_by_location(&mut matched);
    moved.sort_by(|left, right| location_order(&left.0, &right.0));
    changed.sort_by(|left, right| location_order(&left.0, &right.0));
    resolved.sort_by(|left, right| left.current.cmp(&right.current));

    AuditReport {
        configuration: configuration.map(|configuration| configuration.name.clone()),
        baseline_path: baseline_path.to_path_buf(),
        matched,
        moved,
        changed,
        unbaselined,
        stale,
        forbidden_baseline,
        sanctioned: sanctioned.iter().cloned().collect(),
        invariant_calls: invariant_calls.into_iter().collect(),
        resolved,
    }
}

/// Prints findings in `path:line:kind` order, with the line compared as a
/// number rather than as text.
fn location_order(left: &Finding, right: &Finding) -> Ordering {
    (&left.key.path, left.line, &left.key.kind).cmp(&(&right.key.path, right.line, &right.key.kind))
}

fn sort_by_location(findings: &mut [Finding]) {
    findings.sort_by(location_order);
}

pub(crate) fn audit(
    root_arg: &Path,
    baseline_path: &Path,
    configuration: Option<&Configuration>,
) -> Result<AuditReport> {
    let selection = ProductionSelection::load(root_arg, configuration)?;
    let baseline = load_baseline(baseline_path)?;
    let (source_files, mut locations) = run_clippy(&selection, configuration)?;
    let scan = scan_sources(&selection.root, &source_files)?;
    locations.extend(scan.assertions);
    let actual = key_findings(&scan.items, locations);
    let sanctioned = key_findings(&scan.items, scan.sanctioned_invariant_panics);
    Ok(build_report(
        &actual,
        baseline,
        &sanctioned,
        scan.invariant_calls,
        configuration,
        baseline_path,
    ))
}

#[cfg(test)]
mod tests;
