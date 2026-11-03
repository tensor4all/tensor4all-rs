use super::{experiment, relative_error, run, Protocol};

fn protocol(target: f64, algorithms: [&'static str; 2]) -> Protocol<'static> {
    Protocol {
        cases: &[("chain", 4, 2), ("binary", 3, 1), ("star", 4, 1)],
        seeds: &[1, 2],
        blocks: 1,
        samples: 16,
        target,
        algorithms,
    }
}

fn records(
    target: f64,
    algorithms: [&'static str; 2],
    tag: &str,
) -> (bool, Vec<serde_json::Value>) {
    let dir = std::env::temp_dir().join(format!("tree-rsi-bench-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let path = dir.join("out.jsonl");
    let ok = experiment(
        &protocol(target, algorithms),
        path.to_str().unwrap(),
        "src",
        "bin",
    )
    .is_ok();
    let lines = std::fs::read_to_string(&path)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    std::fs::remove_dir_all(&dir).unwrap();
    (ok, lines)
}

#[test]
fn exact_caps_pass_and_reach_the_timing_phase() {
    // Caps chi^2 hold the exact product, so both algorithms are exact.
    let (ok, lines) = records(1e-8, ["rsi", "treeaci"], "pass");
    assert!(ok);
    assert_eq!(lines[0]["type"], "protocol");
    assert_eq!(lines[0]["source_sha256"], "src");
    let observations = lines.iter().filter(|l| l["type"] == "observation");
    let timing = observations
        .clone()
        .filter(|l| l["phase"] == "timing")
        .count();
    assert_eq!(timing, 3 * 2 * 2); // cases x seeds x algorithms, one block
    for record in observations {
        assert!(record["error"].as_f64().unwrap() <= 1e-8, "{record}");
        assert!(!record["diagnostics"]["edge_ranks"]
            .as_array()
            .unwrap()
            .is_empty());
    }
}

#[test]
fn missed_targets_and_failures_skip_timing_and_reject_the_run() {
    let (ok, lines) = records(-1.0, ["rsi", "treeaci"], "miss");
    assert!(!ok);
    assert!(lines.iter().all(|l| l["phase"] != "timing"));
    let (ok, lines) = records(1e-8, ["rsi", "unknown"], "fail");
    assert!(!ok);
    assert!(lines
        .iter()
        .any(|l| l["type"] == "failure" && l["failure"] == "unknown algorithm"));
    assert!(run("unknown", &[], 1, 0, 0).is_err());
}
#[test]
fn accuracy_metric_uses_amplitude_differences_and_rejects_invalid_records() {
    assert_eq!(relative_error(&[2.0, 4.0], &[1.0, 2.0]), Some(1.0));
    assert_eq!(
        relative_error(&[1e-250, 2e-250], &[1e-250, 2e-250]),
        Some(0.0)
    );
    assert_eq!(relative_error(&[f64::NAN], &[1.0]), None);
    assert_eq!(relative_error(&[0.0], &[0.0]), Some(0.0));
    assert_eq!(relative_error(&[1.0], &[0.0]), None);
    assert_eq!(relative_error(&[], &[]), None);
}
