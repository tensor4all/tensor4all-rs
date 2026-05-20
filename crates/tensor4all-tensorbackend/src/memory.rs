//! Process-level memory pressure helpers.

/// Result reported by [`release_process_allocator_cached_memory`].
///
/// The fields are platform-specific because allocator pressure-relief APIs do
/// not expose a uniform contract. On macOS, `released_bytes` is the value
/// returned by `malloc_zone_pressure_relief`. On Linux, `success` is the boolean
/// result returned by `malloc_trim(0)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AllocatorPressureRelief {
    /// Whether tensor4all-rs has a platform hook for the current target.
    pub supported: bool,
    /// Platform-reported released bytes when the allocator exposes that value.
    pub released_bytes: Option<usize>,
    /// Platform-reported success when the allocator exposes only a status bit.
    pub success: Option<bool>,
}

/// Ask the process allocator to return cached/free memory to the operating system.
///
/// This is a diagnostic and memory-pressure hook for the platform system
/// allocator only. It does not release memory that is still owned by live
/// tensors or explicit buffer pools, and it may have no effect if the program is
/// built with a custom global allocator.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::release_process_allocator_cached_memory;
///
/// let report = release_process_allocator_cached_memory();
/// assert_eq!(
///     report.supported,
///     cfg!(any(target_os = "macos", target_os = "linux"))
/// );
/// ```
pub fn release_process_allocator_cached_memory() -> AllocatorPressureRelief {
    allocator_pressure_relief()
}

#[cfg(target_os = "macos")]
fn allocator_pressure_relief() -> AllocatorPressureRelief {
    use std::ffi::c_void;

    extern "C" {
        fn malloc_default_zone() -> *mut c_void;
        fn malloc_zone_pressure_relief(zone: *mut c_void, goal: usize) -> usize;
    }

    unsafe {
        let zone = malloc_default_zone();
        if zone.is_null() {
            AllocatorPressureRelief {
                supported: true,
                released_bytes: Some(0),
                success: Some(false),
            }
        } else {
            let released_bytes = malloc_zone_pressure_relief(zone, 0);
            AllocatorPressureRelief {
                supported: true,
                released_bytes: Some(released_bytes),
                success: Some(released_bytes > 0),
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn allocator_pressure_relief() -> AllocatorPressureRelief {
    extern "C" {
        fn malloc_trim(pad: usize) -> i32;
    }

    let success = unsafe { malloc_trim(0) != 0 };
    AllocatorPressureRelief {
        supported: true,
        released_bytes: None,
        success: Some(success),
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn allocator_pressure_relief() -> AllocatorPressureRelief {
    AllocatorPressureRelief {
        supported: false,
        released_bytes: None,
        success: None,
    }
}

#[cfg(test)]
mod tests {
    use super::release_process_allocator_cached_memory;

    #[test]
    fn reports_platform_support() {
        let report = release_process_allocator_cached_memory();
        assert_eq!(
            report.supported,
            cfg!(any(target_os = "macos", target_os = "linux"))
        );
    }
}
