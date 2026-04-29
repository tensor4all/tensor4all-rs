use std::path::{Path, PathBuf};

/// Environment variable that overrides where demo CSV files are written.
pub const DATA_DIR_ENV: &str = "TENSOR4ALL_DATA_DIR";

/// Resolve the data directory for demo output.
///
/// If `env_override` is set, that path wins. Otherwise we fall back to the
/// repository's `docs/data` directory so the tutorials keep their current
/// default behavior.
pub fn resolve_data_dir(manifest_dir: &Path, env_override: Option<&str>) -> PathBuf {
    match env_override
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        Some(path) => PathBuf::from(path),
        None => manifest_dir.join("docs").join("data"),
    }
}

/// Resolve the demo data directory using the current process environment.
pub fn data_dir() -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    resolve_data_dir(manifest_dir, std::env::var(DATA_DIR_ENV).ok().as_deref())
}
