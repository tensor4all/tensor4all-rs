#![cfg(feature = "partitioned")]
#[path = "../examples/support/large_fourier.rs"]
mod large_fourier;

use large_fourier::Runs;

// The full 1D/2D R=10/20 grid has up to 2^40 entries, but these tests use
// interpolative construction and 35 cached sample points, never a dense DFT.
#[test]
#[ignore = "heavy: runs in the scheduled heavy-tests workflow (see CONTRIBUTING.md)"]
fn heavy_gaussian_1d_r10_r20() {
    for bits in [10, 20] {
        let measured =
            large_fourier::run_case(bits, 1, false, Runs::ScheduleAndFrontEnd, None).unwrap();
        assert!(measured.reconstruction_bound > 0.0);
        assert!(measured.compressions > 0);
    }
}
#[test]
#[ignore = "heavy: runs in the scheduled heavy-tests workflow (see CONTRIBUTING.md)"]
fn heavy_gaussian_2d_grouped_r10_r20() {
    for bits in [10, 20] {
        let measured =
            large_fourier::run_case(bits, 2, false, Runs::ScheduleAndFrontEnd, None).unwrap();
        assert!(measured.reconstruction_bound > 0.0);
        assert!(measured.compressions > 0);
    }
}

/// Address-space ceiling of the interleaved cases, in KiB: the 8 GiB under
/// which they were accepted (#1279, #1286). The R = 10 case peaks at 4.0 GiB RSS.
const INTERLEAVED_CEILING_KIB: u64 = 8 * 1024 * 1024;
/// Set in the child process that runs an interleaved case under the ceiling.
const INTERLEAVED_CHILD: &str = "T4A_HEAVY_INTERLEAVED_CHILD";

/// Peak resident set size of this process in KiB (`VmHWM`), if the platform
/// reports it.
fn peak_rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
    line.split_whitespace().nth(1)?.parse().ok()
}

/// Run `case` under the address-space ceiling. On Linux the test re-executes
/// itself as a child with `ulimit -v`, so an allocation above the ceiling fails
/// the child and so this test, instead of the runner's OOM killer ending the
/// job; elsewhere, and inside the child, `case` runs directly and the peak RSS
/// is reported.
fn under_interleaved_ceiling(test: &str, case: impl FnOnce()) {
    if std::env::var_os(INTERLEAVED_CHILD).is_some() || !cfg!(target_os = "linux") {
        case();
        match peak_rss_kib() {
            Some(peak) => eprintln!(
                "peak RSS {:.2} GiB (address-space ceiling {} GiB)",
                peak as f64 / (1024.0 * 1024.0),
                INTERLEAVED_CEILING_KIB / (1024 * 1024)
            ),
            None => eprintln!("peak RSS not reported on this platform"),
        }
        return;
    }
    // Test names omit the crate, the first segment of `module_path!`.
    let path = module_path!();
    let module = path.split_once("::").map_or(path, |(_, rest)| rest);
    let name = format!("{module}::{test}");
    let status = std::process::Command::new("sh")
        .arg("-c")
        .arg(format!(
            "ulimit -v {INTERLEAVED_CEILING_KIB} && exec \"$0\" \"$@\""
        ))
        .arg(std::env::current_exe().unwrap())
        .args([
            "--exact",
            &name,
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(INTERLEAVED_CHILD, "1")
        .status()
        .unwrap();
    assert!(
        status.success(),
        "{test} failed under the {} GiB ceiling: {status}",
        INTERLEAVED_CEILING_KIB / (1024 * 1024)
    );
}

// Two interleaved axes at R = 10: the exact reference application used to
// form input bond x operator bond = 50 x 144 = 7200 wide site tensors and run
// out of memory; the zip-up keeps every bond at its physical bound, at most
// 1024 here (#1279). Its peak follows that physical bound, not the compressed
// rank, so the case runs under the 8 GiB ceiling. Since the exact images
// factorize by QR (#1285) each schedule takes about 6 minutes with one thread,
// so the case runs the separately timed schedule and the front end, as the
// other cases do.
#[test]
#[ignore = "heavy: runs in the scheduled heavy-tests workflow (see CONTRIBUTING.md)"]
fn heavy_gaussian_2d_interleaved_r10() {
    under_interleaved_ceiling("heavy_gaussian_2d_interleaved_r10", || {
        let measured =
            large_fourier::run_case(10, 2, true, Runs::ScheduleAndFrontEnd, None).unwrap();
        assert!(measured.reconstruction_bound > 0.0);
        assert!(measured.compressions > 0);
    });
}

// Two interleaved axes at R = 20, where the exact image itself can be 7200
// wide over the middle of the chain: the leaves are applied with the certified
// truncation (#1286), which never forms that image, given half of the
// allowance. `run_case` checks the sampled spectrum against the reported bound,
// which now includes the certified application bounds. Only the front end runs,
// because each leaf's exact factors take minutes to build.
#[test]
#[ignore = "heavy: runs in the scheduled heavy-tests workflow (see CONTRIBUTING.md)"]
fn heavy_gaussian_2d_interleaved_r20_certified() {
    under_interleaved_ceiling("heavy_gaussian_2d_interleaved_r20_certified", || {
        let measured = large_fourier::run_case(20, 2, true, Runs::FrontEndOnly, Some(0.5)).unwrap();
        // `run_case` already checks the bound against the allowance.
        assert!(measured.reconstruction_bound > 0.0);
        assert_eq!(measured.certified_share, Some(0.5));
    });
}
