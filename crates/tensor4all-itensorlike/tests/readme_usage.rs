use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

fn extract_first_rust_code_block(readme: &str) -> Option<&str> {
    let start = readme.find("```rust")?;
    let body_start = readme[start..].find('\n')? + start + 1;
    let end = readme[body_start..].find("\n```")? + body_start;
    Some(&readme[body_start..end])
}

fn render_doctest_snippet(snippet: &str) -> String {
    let mut rendered = String::new();
    for line in snippet.lines() {
        if let Some(unhidden) = line.strip_prefix("# ") {
            rendered.push_str(unhidden);
        } else if line == "#" {
            // Rustdoc treats a bare "#" as a hidden blank line.
        } else {
            rendered.push_str(line);
        }
        rendered.push('\n');
    }
    rendered
}

fn make_temp_project_dir() -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "tensor4all-itensorlike-readme-{}-{timestamp}",
        std::process::id()
    ))
}

fn write_temp_project(project_dir: &Path, snippet: &str) {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let core_dir = manifest_dir.parent().unwrap().join("tensor4all-core");

    fs::create_dir_all(project_dir.join("src")).unwrap();
    fs::write(
        project_dir.join("Cargo.toml"),
        format!(
            r#"[package]
name = "tensor4all_itensorlike_readme_usage"
version = "0.1.0"
edition = "2021"

[dependencies]
anyhow = "1"
tensor4all-core = {{ path = "{}" }}
tensor4all-itensorlike = {{ path = "{}" }}
"#,
            core_dir.display(),
            manifest_dir.display()
        ),
    )
    .unwrap();
    fs::write(project_dir.join("src/main.rs"), snippet).unwrap();
}

#[test]
fn readme_usage_example_compiles() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();
    let readme = fs::read_to_string(manifest_dir.join("README.md")).unwrap();
    let snippet =
        extract_first_rust_code_block(&readme).expect("README should contain a Rust example");
    let rendered_snippet = render_doctest_snippet(snippet);

    let project_dir = make_temp_project_dir();
    write_temp_project(&project_dir, &rendered_snippet);

    let output = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string()))
        .arg("check")
        .arg("--release")
        .arg("--offline")
        .arg("--manifest-path")
        .arg(project_dir.join("Cargo.toml"))
        .arg("--target-dir")
        .arg(workspace_root.join("target/readme-usage"))
        .output()
        .unwrap();

    let _ = fs::remove_dir_all(&project_dir);

    assert!(
        output.status.success(),
        "README usage example failed to compile\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
