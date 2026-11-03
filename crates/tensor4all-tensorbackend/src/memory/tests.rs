use super::release_process_allocator_cached_memory;

#[test]
fn reports_platform_support() {
    let report = release_process_allocator_cached_memory();
    assert_eq!(
        report.supported,
        cfg!(any(target_os = "macos", target_os = "linux"))
    );
}
