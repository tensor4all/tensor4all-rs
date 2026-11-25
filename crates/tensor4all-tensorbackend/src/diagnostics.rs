//! Process-wide diagnostics, resolved once instead of during each contraction.

use std::sync::OnceLock;

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct NativeEinsumDiagnostics {
    pub(crate) profile: bool,
    pub(crate) trace_paths: bool,
    pub(crate) compare_balanced: bool,
    pub(crate) trace_pool: bool,
    pub(crate) path_min_bytes: usize,
    pub(crate) path_max_signatures: usize,
    pub(crate) pool_min_output_bytes: usize,
    pub(crate) pool_min_retained_bytes: usize,
}

impl NativeEinsumDiagnostics {
    fn from_lookup(mut lookup: impl FnMut(&str) -> Option<String>) -> Self {
        Self {
            profile: lookup("T4A_PROFILE_NATIVE_EINSUM").is_some(),
            trace_paths: lookup("T4A_TRACE_NATIVE_EINSUM_PATHS").is_some(),
            compare_balanced: lookup("T4A_TRACE_NATIVE_EINSUM_COMPARE_BALANCED").is_some(),
            trace_pool: lookup("T4A_TRACE_NATIVE_EINSUM_POOL").is_some(),
            path_min_bytes: lookup("T4A_TRACE_NATIVE_EINSUM_MIN_BYTES")
                .and_then(|value| value.parse().ok())
                .unwrap_or(0),
            path_max_signatures: lookup("T4A_TRACE_NATIVE_EINSUM_MAX_SIGNATURES")
                .and_then(|value| value.parse().ok())
                .unwrap_or(64),
            pool_min_output_bytes: lookup("T4A_TRACE_NATIVE_EINSUM_POOL_MIN_OUTPUT_BYTES")
                .and_then(|value| value.parse().ok())
                .unwrap_or(0),
            pool_min_retained_bytes: lookup("T4A_TRACE_NATIVE_EINSUM_POOL_MIN_RETAINED_BYTES")
                .and_then(|value| value.parse().ok())
                .unwrap_or(0),
        }
    }
}

pub(crate) fn native_einsum_diagnostics() -> &'static NativeEinsumDiagnostics {
    static CONFIG: OnceLock<NativeEinsumDiagnostics> = OnceLock::new();
    CONFIG.get_or_init(|| NativeEinsumDiagnostics::from_lookup(|key| std::env::var(key).ok()))
}

#[cfg(test)]
mod tests;
