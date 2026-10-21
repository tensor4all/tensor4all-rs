//! Default CPU-pool precedence: the setter, then Rayon's default, then the
//! main thread's allowed CPUs.
//!
//! Each scenario runs in a fresh process because the first [`CpuBackend::new`]
//! fixes the process-wide default. Environment variables are set on the child
//! command (`RAYON_NUM_THREADS` is read before the pool is built), and Rayon's
//! own `RAYON_RS_NUM_CPUS` fallback is cleared in the baseline child.

use std::process::{Command, Output};

use tensor4all_tensorbackend::CpuBackend;

const CHILD: &str = "T4A_CPU_THREADS_CHILD";
const EXPECT: &str = "T4A_CPU_THREADS_EXPECT";

/// Run one scenario in a fresh test process.
fn child(scenario: &str, environment: &[(&str, &str)]) -> Output {
    let mut command = Command::new(std::env::current_exe().expect("the test binary"));
    command
        .args(["--exact", "cpu_threads_child", "--nocapture"])
        .env(CHILD, scenario)
        .env_remove("RAYON_NUM_THREADS")
        .env_remove("RAYON_RS_NUM_CPUS");
    for (key, value) in environment {
        command.env(key, value);
    }
    let output = command.output().expect("the child test process starts");
    assert!(
        output.status.success(),
        "scenario {scenario:?} failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

/// The child-side assertions. Without [`CHILD`] this is an ordinary passing
/// test; [`child`] selects one scenario in a fresh process.
#[test]
fn cpu_threads_child() {
    let Ok(scenario) = std::env::var(CHILD) else {
        return;
    };
    match scenario.as_str() {
        "setter" => {
            // The setter, not the inherited RAYON_NUM_THREADS=2, fixes the count.
            tensor4all_tensorbackend::set_default_cpu_threads(
                std::num::NonZeroUsize::new(3).expect("three workers"),
            )
            .expect("the first configuration wins");
            assert_eq!(CpuBackend::new().num_threads(), 3);
        }
        "env" => {
            let expected: usize = std::env::var(EXPECT)
                .expect("the parent passes the expected count")
                .parse()
                .expect("the expected count is a number");
            let backend = CpuBackend::new();
            assert_eq!(backend.num_threads(), expected);
            if expected == 1 {
                assert!(backend.engine().pool().is_none());
            }
        }
        "affinity" => {
            let expected = std::thread::available_parallelism()
                .expect("the platform reports parallelism")
                .get();
            assert_eq!(CpuBackend::new().num_threads(), expected);
        }
        other => panic!("unknown child scenario {other:?}"),
    }
}

/// `set_default_cpu_threads` wins over `RAYON_NUM_THREADS`.
#[test]
fn setter_wins_over_rayon_num_threads() {
    child("setter", &[("RAYON_NUM_THREADS", "2")]);
}

/// `RAYON_NUM_THREADS` wins over the CPUs in the main thread's mask.
#[cfg(target_os = "linux")]
#[test]
fn rayon_num_threads_wins_over_the_affinity_count() {
    let allowed = CpuBackend::allowed_cpus().expect("the Linux affinity snapshot");
    // Pick a Rayon count that differs from the affinity count, whichever way
    // this host is provisioned: a value above one needs a pool; one resolves
    // to the serial engine.
    let expected = if allowed.len() == 1 { "2" } else { "1" };
    child(
        "env",
        &[("RAYON_NUM_THREADS", expected), (EXPECT, expected)],
    );
}

/// With neither variable set, the count is the affinity count that Rayon's
/// own `available_parallelism` reports.
#[cfg(target_os = "linux")]
#[test]
fn without_env_the_affinity_count_is_used() {
    child("affinity", &[]);
}

/// Rayon's own handling of zero or malformed values is left in place: both
/// fall back to `available_parallelism`, and the pool still builds.
#[cfg(target_os = "linux")]
#[test]
fn invalid_rayon_num_threads_uses_rayons_default() {
    child("affinity", &[("RAYON_NUM_THREADS", "0")]);
    child("affinity", &[("RAYON_NUM_THREADS", "not-a-number")]);
}
