use super::*;

#[test]
fn missing_settings_disable_diagnostics_and_keep_default_limits() {
    let config = NativeEinsumDiagnostics::from_lookup(|_| None);
    assert_eq!(
        config,
        NativeEinsumDiagnostics {
            profile: false,
            trace_paths: false,
            compare_balanced: false,
            trace_pool: false,
            path_min_bytes: 0,
            path_max_signatures: 64,
            pool_min_output_bytes: 0,
            pool_min_retained_bytes: 0,
        }
    );
}

#[test]
fn flags_use_presence_even_for_empty_or_false_values() {
    for key in [
        "T4A_PROFILE_NATIVE_EINSUM",
        "T4A_TRACE_NATIVE_EINSUM_PATHS",
        "T4A_TRACE_NATIVE_EINSUM_COMPARE_BALANCED",
        "T4A_TRACE_NATIVE_EINSUM_POOL",
    ] {
        for value in ["", "0", "false", "1"] {
            let config =
                NativeEinsumDiagnostics::from_lookup(|name| (name == key).then(|| value.into()));
            assert_eq!(config.profile, key == "T4A_PROFILE_NATIVE_EINSUM");
            assert_eq!(config.trace_paths, key == "T4A_TRACE_NATIVE_EINSUM_PATHS");
            assert_eq!(
                config.compare_balanced,
                key == "T4A_TRACE_NATIVE_EINSUM_COMPARE_BALANCED"
            );
            assert_eq!(config.trace_pool, key == "T4A_TRACE_NATIVE_EINSUM_POOL");
        }
    }
}

#[test]
fn numeric_limits_are_independent_and_accept_zero() {
    let config = NativeEinsumDiagnostics::from_lookup(|key| {
        Some(
            match key {
                "T4A_TRACE_NATIVE_EINSUM_MIN_BYTES" => "128",
                "T4A_TRACE_NATIVE_EINSUM_MAX_SIGNATURES" => "0",
                "T4A_TRACE_NATIVE_EINSUM_POOL_MIN_OUTPUT_BYTES" => "256",
                "T4A_TRACE_NATIVE_EINSUM_POOL_MIN_RETAINED_BYTES" => "512",
                _ => return None,
            }
            .into(),
        )
    });
    assert_eq!(config.path_min_bytes, 128);
    assert_eq!(config.path_max_signatures, 0);
    assert_eq!(config.pool_min_output_bytes, 256);
    assert_eq!(config.pool_min_retained_bytes, 512);
    assert!(!config.profile);
    assert!(!config.trace_paths);
    assert!(!config.compare_balanced);
    assert!(!config.trace_pool);
}

#[test]
fn invalid_numeric_limits_keep_the_defaults() {
    let defaults = NativeEinsumDiagnostics::from_lookup(|_| None);
    for value in [
        "",
        "invalid",
        "-1",
        "999999999999999999999999999999999999999999999",
    ] {
        let config = NativeEinsumDiagnostics::from_lookup(|key| {
            key.contains("BYTES")
                .then(|| value.into())
                .or_else(|| key.ends_with("MAX_SIGNATURES").then(|| value.into()))
        });
        assert_eq!(config, defaults, "invalid value: {value}");
    }
}

#[test]
fn process_config_reuses_the_same_snapshot() {
    let config = native_einsum_diagnostics();
    assert!(std::ptr::eq(config, native_einsum_diagnostics()));
    let other_thread = std::thread::spawn(native_einsum_diagnostics)
        .join()
        .unwrap();
    assert!(std::ptr::eq(config, other_thread));
}
