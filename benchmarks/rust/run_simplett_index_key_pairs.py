#!/usr/bin/env python3
"""Run the paired TTCache index-key experiment for the simplett cache (#627).

Builds the `cache_overhead` Criterion benchmark at three pinned revisions,
runs the `cache_keys` group in rotated arm order, applies the predeclared noise
and timing gates, and writes the evidence files described in
`benchmarks/results/2026-11-05-simplett-index-keys/README.md`.

Run from a checkout that contains all three revisions (for example after
`git fetch origin pull/1146/head`). The script refuses to time anything if the
host fails the idle check, and the decision is INCONCLUSIVE whenever a noise
gate fails; it never retries or drops cells.

    python3 benchmarks/rust/run_simplett_index_key_pairs.py --output DIR

Only the standard library is required.
"""

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import pathlib
import platform
import shutil
import statistics
import subprocess
import sys
import time

HARNESS = "crates/tensor4all-simplett/benches/cache_overhead.rs"
HARNESS_SHA256 = "790df1082fd016ecdaaeb42f540330103580af9538c15dacfe50342ec7168125"

# The original revision predates the harness; it is built with the harness file
# from the final revision copied over its own, and its library unchanged.
ARMS = {
    "original": "097e2927d640ea68ff365ac56886be7676e79c8d",
    "shared": "6527870794612c9ef0dfcf659082f89e2af787c5",
    "final": "a740d6c7f287090d7681eb3c3d7d2bcbe031938b",
}
HARNESS_SOURCE = "final"

ROUND_ORDERS = [
    ["original", "shared", "final"],
    ["final", "original", "shared"],
    ["shared", "final", "original"],
]

THREAD_ENV = {
    name: "1"
    for name in [
        "RAYON_NUM_THREADS",
        "BLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]
}

ROUTES = ["cold_fixed", "warm_fixed", "warm_auto", "left_hit", "right_hit"]
BATCH_ROUTES = {"cold_fixed", "warm_fixed", "warm_auto"}
HIT_ROUTES = {"left_hit", "right_hit"}
FULL_CELL_COUNT = 30  # six cases x five routes

GATES = {
    "maximum_load_1m": 8.0,
    # Two-sided: (ci95_high - ci95_low) / (2 * mean).
    "maximum_relative_ci_halfwidth": 0.25,
    # Sample standard deviation (divisor n - 1) of the per-round means.
    "maximum_rep_mean_cv": 0.20,
    "final_batch_max_ratio_vs_original": 1.10,
    "final_batch_max_ratio_vs_shared": 1.10,
    "final_hit_geomean_max_ratio_vs_shared": 0.95,
}


def run(cmd, **kwargs):
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kwargs).stdout


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(pathlib.Path(path).read_bytes())


def host_description():
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
    }
    if sys.platform == "darwin":
        info["cpu"] = run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
    elif pathlib.Path("/proc/cpuinfo").exists():
        for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["cpu"] = line.split(":", 1)[1].strip()
                break
    return info


def thermal_snapshot():
    if sys.platform == "darwin":
        try:
            return run(["pmset", "-g", "therm"]).strip()
        except (OSError, subprocess.CalledProcessError) as err:
            return f"unavailable: {err}"
    return "unavailable on this platform"


def observe():
    return {"time": time.time(), "loadavg": list(os.getloadavg()), "thermal": thermal_snapshot()}


def export_revision(repo, rev, dest):
    """Write the tree of `rev` into `dest` without touching the repository's worktrees."""
    dest.mkdir(parents=True)
    archive = subprocess.Popen(["git", "archive", "--format=tar", rev], cwd=repo, stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", str(dest)], stdin=archive.stdout, check=True)
    archive.stdout.close()
    if archive.wait() != 0:
        raise SystemExit(f"git archive {rev} failed")


def build_arm(repo, arm, rev, harness_bytes, work, target_dir):
    src = work / "src" / arm
    export_revision(repo, rev, src)
    harness_path = src / HARNESS
    if arm != HARNESS_SOURCE:
        harness_path.write_bytes(harness_bytes)
    if sha256_file(harness_path) != HARNESS_SHA256:
        raise SystemExit(f"{arm}: harness hash mismatch after export")
    cmd = [
        "cargo", "bench", "--locked", "-p", "tensor4all-simplett",
        "--bench", "cache_overhead", "--no-run", "--message-format=json",
        "--target-dir", str(target_dir),
    ]
    print(f"building {arm} at {rev[:12]}", flush=True)
    out = subprocess.run(cmd, cwd=src, check=True, text=True, stdout=subprocess.PIPE).stdout
    executables = [
        msg["executable"]
        for msg in map(json.loads, out.splitlines())
        if msg.get("reason") == "compiler-artifact"
        and msg.get("target", {}).get("name") == "cache_overhead"
        and msg.get("executable")
    ]
    if len(executables) != 1:
        raise SystemExit(f"{arm}: expected one cache_overhead executable, found {executables}")
    binary = work / "bin" / arm
    binary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(executables[0], binary)
    return {
        "commit": rev,
        "harness_sha256": HARNESS_SHA256,
        "harness_overlaid": arm != HARNESS_SOURCE,
        "binary_sha256": sha256_file(binary),
        "rustc": run(["rustc", "-vV"], cwd=src).strip(),
        "cargo": run(["cargo", "-V"], cwd=src).strip(),
    }, binary


def idle_check(seconds, max_load):
    before = observe()
    time.sleep(seconds)
    after = observe()
    passed = max(before["loadavg"][0], after["loadavg"][0]) <= max_load
    return {"seconds": seconds, "maximum_load_1m": max_load, "before": before, "after": after, "passed": passed}


def read_cells(criterion_home, baseline):
    cells = {}
    for estimates in criterion_home.glob(f"**/{baseline}/estimates.json"):
        cell_dir = estimates.parent
        full_id = json.loads((cell_dir / "benchmark.json").read_text())["full_id"]
        mean = json.loads(estimates.read_text())["mean"]
        cells[full_id] = {
            "mean_ns": mean["point_estimate"],
            "ci95_low_ns": mean["confidence_interval"]["lower_bound"],
            "ci95_high_ns": mean["confidence_interval"]["upper_bound"],
            "samples": json.loads((cell_dir / "sample.json").read_text()),
        }
    return cells


def run_arm(binary, arm, round_no, out, cpu, bench_filter):
    label = f"{arm}-r{round_no}"
    criterion_home = out / "criterion" / label
    env = dict(os.environ, **THREAD_ENV, CRITERION_HOME=str(criterion_home))
    preexec = None
    if cpu is not None:
        preexec = lambda: os.sched_setaffinity(0, {cpu})  # noqa: E731
    before = observe()
    start = time.monotonic()
    proc = subprocess.run(
        [str(binary), bench_filter, "--bench", "--noplot", "--save-baseline", label],
        env=env, text=True, capture_output=True, preexec_fn=preexec,
    )
    elapsed = time.monotonic() - start
    after = observe()
    (out / "logs").mkdir(exist_ok=True)
    (out / "logs" / f"{label}.txt").write_text(proc.stdout + "\n" + proc.stderr)
    run_record = {
        "round": round_no, "arm": arm, "exit_code": proc.returncode,
        "elapsed_seconds": elapsed, "before": before, "after": after,
    }
    if proc.returncode != 0:
        raise SystemExit(f"{label} exited with {proc.returncode}; see logs/{label}.txt")
    run_record["cells"] = read_cells(criterion_home, label)
    return run_record


def summarize(runs):
    by_cell = {}
    for r in runs:
        for cell, values in r["cells"].items():
            by_cell.setdefault(cell, {}).setdefault(r["arm"], {})[r["round"]] = values

    def ratio(cell, num, den, round_no):
        return by_cell[cell][num][round_no]["mean_ns"] / by_cell[cell][den][round_no]["mean_ns"]

    round_numbers = sorted({r["round"] for r in runs})

    cases = []
    for cell in sorted(by_cell):
        route = cell.split("/")[1]
        entry = {"case": cell, "route": route}
        for arm in ARMS:
            means = [by_cell[cell][arm][k]["mean_ns"] for k in round_numbers]
            entry[f"{arm}_median_ns"] = statistics.median(means)
            entry[f"{arm}_cv"] = statistics.stdev(means) / statistics.mean(means)
        for num, den in [("final", "original"), ("final", "shared")]:
            values = [ratio(cell, num, den, k) for k in round_numbers]
            entry[f"{num}_over_{den}"] = {
                "per_round": values, "median": statistics.median(values),
                "min": min(values), "max": max(values),
            }
        cases.append(entry)

    halfwidths = [
        (v["ci95_high_ns"] - v["ci95_low_ns"]) / (2 * v["mean_ns"])
        for r in runs for v in r["cells"].values()
    ]
    upward = [(v["ci95_high_ns"] - v["mean_ns"]) / v["mean_ns"] for r in runs for v in r["cells"].values()]
    loads = [x for r in runs for x in (r["before"]["loadavg"][0], r["after"]["loadavg"][0])]
    cvs = [c[f"{arm}_cv"] for c in cases for arm in ARMS]

    noise = {
        "maximum_load_1m": max(loads),
        "maximum_relative_ci_halfwidth": max(halfwidths),
        "maximum_upward_ci_halfwidth": max(upward),
        "maximum_rep_mean_cv": max(cvs),
    }
    noise_pass = {
        "load": noise["maximum_load_1m"] <= GATES["maximum_load_1m"],
        "ci_halfwidth": noise["maximum_relative_ci_halfwidth"] <= GATES["maximum_relative_ci_halfwidth"],
        "cv": noise["maximum_rep_mean_cv"] <= GATES["maximum_rep_mean_cv"],
    }

    batch = [c for c in cases if c["route"] in BATCH_ROUTES]
    hits = [c for c in cases if c["route"] in HIT_ROUTES]
    timing = {
        "final_batch_max_ratio_vs_original": max((c["final_over_original"]["median"] for c in batch), default=None),
        "final_batch_max_ratio_vs_shared": max((c["final_over_shared"]["median"] for c in batch), default=None),
        "final_hit_geomean_ratio_vs_shared": (
            math.exp(statistics.mean(math.log(c["final_over_shared"]["median"]) for c in hits)) if hits else None
        ),
    }
    timing_pass = {
        "batch_vs_original": timing["final_batch_max_ratio_vs_original"] is not None
        and timing["final_batch_max_ratio_vs_original"] <= GATES["final_batch_max_ratio_vs_original"],
        "batch_vs_shared": timing["final_batch_max_ratio_vs_shared"] is not None
        and timing["final_batch_max_ratio_vs_shared"] <= GATES["final_batch_max_ratio_vs_shared"],
        "hit_geomean_vs_shared": timing["final_hit_geomean_ratio_vs_shared"] is not None
        and timing["final_hit_geomean_ratio_vs_shared"] <= GATES["final_hit_geomean_max_ratio_vs_shared"],
    }
    if not all(noise_pass.values()):
        decision = "INCONCLUSIVE"
    elif all(timing_pass.values()):
        decision = "PASS"
    else:
        decision = "FAIL"
    return {
        "decision": decision, "noise": noise, "noise_pass": noise_pass,
        "timing": timing, "timing_pass": timing_pass, "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=pathlib.Path, required=True, help="new directory for all results")
    parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[2])
    parser.add_argument("--idle-seconds", type=float, default=120.0)
    parser.add_argument("--idle-max-load", type=float, default=6.0)
    parser.add_argument(
        "--target-dir", type=pathlib.Path, default=None,
        help="parent of per-arm Cargo target directories to reuse; by default a temporary one inside --output, "
        "removed after building",
    )
    parser.add_argument("--cpu", type=int, default=None, help="pin each run to this CPU (Linux only)")
    parser.add_argument(
        "--filter", default="cache_keys",
        help="Criterion filter; anything other than the default yields a partial, smoke-test-only result",
    )
    args = parser.parse_args()

    if args.cpu is not None and not hasattr(os, "sched_setaffinity"):
        raise SystemExit("--cpu needs os.sched_setaffinity, which this platform lacks")
    out = args.output.resolve()
    if out.exists():
        raise SystemExit(f"{out} already exists; choose a new directory")
    repo = args.repo.resolve()
    for arm, rev in ARMS.items():
        if subprocess.run(["git", "cat-file", "-e", f"{rev}^{{commit}}"], cwd=repo).returncode != 0:
            raise SystemExit(f"{arm} revision {rev} is missing; fetch the PR head first")
    harness_bytes = subprocess.run(
        ["git", "show", f"{ARMS[HARNESS_SOURCE]}:{HARNESS}"], cwd=repo, check=True, capture_output=True
    ).stdout
    if sha256_bytes(harness_bytes) != HARNESS_SHA256:
        raise SystemExit("harness at the final revision does not match the recorded SHA-256")

    out.mkdir(parents=True)
    work = out / "build"
    target_dir = args.target_dir.resolve() if args.target_dir else work / "target"
    arms, binaries = {}, {}
    for arm, rev in ARMS.items():
        # One target directory per arm: Cargo hashes workspace crates without
        # their location and `git archive` stamps files with the commit time,
        # so a shared directory can reuse another arm's build as up to date.
        arms[arm], binaries[arm] = build_arm(repo, arm, rev, harness_bytes, work, target_dir / arm)
    shutil.rmtree(work / "src")
    hashes = [info["binary_sha256"] for info in arms.values()]
    if len(set(hashes)) != len(hashes):
        raise SystemExit(f"arms produced identical executables: {dict(zip(arms, hashes))}")
    if args.target_dir is None:
        shutil.rmtree(target_dir)

    protocol = {
        "arms": arms,
        "host": host_description(),
        "thread_environment": THREAD_ENV,
        "cpu_pin": args.cpu,
        "filter": args.filter,
        "round_orders": ROUND_ORDERS,
        "routes": ROUTES,
        "gates": GATES,
        "idle_gate": {"seconds": args.idle_seconds, "maximum_load_1m": args.idle_max_load},
        "statistic": "median of per-round Criterion mean estimates; paired ratios per round and their median",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=1) + "\n")

    idle = idle_check(args.idle_seconds, args.idle_max_load)
    (out / "idle-check.json").write_text(json.dumps(idle, indent=1) + "\n")
    if not idle["passed"]:
        raise SystemExit(f"idle check failed (load above {args.idle_max_load}); nothing was timed")

    runs = []
    for round_no, order in enumerate(ROUND_ORDERS, start=1):
        for arm in order:
            record = run_arm(binaries[arm], arm, round_no, out, args.cpu, args.filter)
            runs.append(record)
            print(f"round {round_no} {arm}: {len(record['cells'])} cells, "
                  f"load {record['before']['loadavg'][0]:.2f} -> {record['after']['loadavg'][0]:.2f}", flush=True)

    cell_sets = {frozenset(r["cells"]) for r in runs}
    if len(cell_sets) != 1 or not next(iter(cell_sets)):
        raise SystemExit("runs measured different or empty cell sets; see experiment.json.gz")
    complete = len(next(iter(cell_sets))) == FULL_CELL_COUNT and args.filter == "cache_keys"

    with gzip.open(out / "experiment.json.gz", "wt") as fh:
        json.dump({"protocol": protocol, "runs": runs}, fh)
    with open(out / "experiment.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["case", "arm", "round", "mean_ns", "ci95_low_ns", "ci95_high_ns"])
        for r in runs:
            for cell in sorted(r["cells"]):
                v = r["cells"][cell]
                writer.writerow([cell, r["arm"], r["round"], v["mean_ns"], v["ci95_low_ns"], v["ci95_high_ns"]])

    summary = summarize(runs)
    summary["complete_suite"] = complete
    if not complete:
        summary["decision"] = "PARTIAL"
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("decision", "noise", "noise_pass", "timing", "timing_pass")}, indent=1))


if __name__ == "__main__":
    main()
