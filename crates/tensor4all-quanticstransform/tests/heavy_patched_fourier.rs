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
        let measured = large_fourier::run_case(bits, 1, false, Runs::ScheduleAndFrontEnd).unwrap();
        assert!(measured.reconstruction_bound > 0.0);
        assert!(measured.compressions > 0);
    }
}
#[test]
#[ignore = "heavy: runs in the scheduled heavy-tests workflow (see CONTRIBUTING.md)"]
fn heavy_gaussian_2d_grouped_r10_r20() {
    for bits in [10, 20] {
        let measured = large_fourier::run_case(bits, 2, false, Runs::ScheduleAndFrontEnd).unwrap();
        assert!(measured.reconstruction_bound > 0.0);
        assert!(measured.compressions > 0);
    }
}

/// Address-space ceiling of the interleaved case, in KiB: the 8 GiB under which
/// it was accepted (#1279). Its measured peak is 4.0 GiB RSS.
const INTERLEAVED_CEILING_KIB: u64 = 8 * 1024 * 1024;
/// Set in the child process that runs the interleaved case under the ceiling.
const INTERLEAVED_CHILD: &str = "T4A_HEAVY_INTERLEAVED_CHILD";

/// Peak resident set size of this process in KiB (`VmHWM`), if the platform
/// reports it.
fn peak_rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
    line.split_whitespace().nth(1)?.parse().ok()
}

// Two interleaved axes at R = 10: the exact reference application used to
// form input bond x operator bond = 50 x 144 = 7200 wide site tensors and run
// out of memory; the zip-up keeps every bond at its physical bound, at most
// 1024 here (#1279). Its peak follows that physical bound, not the compressed
// rank, so the case runs in a child process under an 8 GiB address-space
// ceiling: an allocation above it fails the child and so this test, instead of
// the runner's OOM killer ending the job. Each schedule spends about 6
// minutes on the exact images with one thread (QR since #1285), so the case
// runs the front end only. R = 20 interleaved, where the exact image itself can be 7200 wide, is
// not covered.
#[test]
#[ignore = "heavy: runs in the scheduled heavy-tests workflow (see CONTRIBUTING.md)"]
fn heavy_gaussian_2d_interleaved_r10() {
    if std::env::var_os(INTERLEAVED_CHILD).is_some() || !cfg!(target_os = "linux") {
        let measured = large_fourier::run_case(10, 2, true, Runs::FrontEndOnly).unwrap();
        assert!(measured.reconstruction_bound > 0.0);
        assert!(measured.compressions > 0);
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
    let name = format!("{module}::heavy_gaussian_2d_interleaved_r10");
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
        "the interleaved case failed under the {} GiB ceiling: {status}",
        INTERLEAVED_CEILING_KIB / (1024 * 1024)
    );
}
