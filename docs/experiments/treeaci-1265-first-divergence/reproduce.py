#!/usr/bin/env python3
"""Reproduce the rejected #1265 prototype in disposable, private worktrees.

This runs numerical diagnostics, not a performance benchmark. It never pushes,
changes the caller's checkout, or installs a new public API. Artifacts are kept.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

BASE = "5909b1848439a2f09df365c40fe97ed97d636002"
BUNDLE = Path(__file__).resolve().parent


def run_command(args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def replace_once(root, path, old, new):
    file = root / "crates/tensor4all-treeaci" / path
    source = file.read_text()
    if source.count(old) != 1:
        raise RuntimeError(f"unexpected source at {file}: {old!r}")
    file.write_text(source.replace(old, new))


def instrument(worktree, output):
    replace_once(worktree, "Cargo.toml", "[dependencies]", "[dependencies]\nserde_json.workspace = true")
    lib = worktree / "crates/tensor4all-treeaci/src/lib.rs"
    lib.write_text(lib.read_text() + f'\n#[path = {json.dumps(str(BUNDLE / "trace.rs"))}]\nmod divergence_trace;\n')
    hooks = [
        ("src/schedule.rs", "    for pass in 0..options.max_sweeps {", "    crate::divergence_trace::init(state);\n    for pass in 0..options.max_sweeps {\n        crate::divergence_trace::pass(pass+1);"),
        ("src/schedule.rs", "        stable_rank_passes =\n", '        crate::divergence_trace::metrics(state, exec, "pass_error");\n        crate::divergence_trace::state(state, "after_pass");\n        stable_rank_passes =\n'),
        ("src/schedule.rs", "            known_guard_pivots = search.pivots;", '            crate::divergence_trace::state(state, "after_guard");\n            known_guard_pivots = search.pivots;'),
        ("src/transaction.rs", "    let proposal = materialize_and_factor_edge(", "    crate::divergence_trace::before_edge(state,forward);\n    let proposal = materialize_and_factor_edge("),
        ("src/transaction.rs", "    commit_edge_proposal(state, forward, proposal, exec)", "    let report=commit_edge_proposal(state, forward, proposal, exec)?;\n    crate::divergence_trace::after_edge(state,exec);\n    Ok(report)"),
        ("src/local_update.rs", "    let mut matrix = Matrix::from_col_major_vec(row_count, col_count, local_values);", "    crate::divergence_trace::matrix(problem,forward,&row_candidates,&col_candidates,&local_values);\n    let mut matrix = Matrix::from_col_major_vec(row_count, col_count, local_values);"),
        ("src/local_update.rs", "    let (left, right, row_indices, col_indices) = if factors.rank == 0 {", "    crate::divergence_trace::factors(&factors,sampled_scale);\n    let (left, right, row_indices, col_indices) = if factors.rank == 0 {"),
        ("src/global_guard.rs", "    drop(start_coordinates);", "    crate::divergence_trace::guard_starts(&starts,&start_errors,nsearch,threshold);\n    drop(start_coordinates);"),
        ("src/global_guard.rs", "    Ok(GlobalSearchReport {\n        pivots,", "    let pivots=crate::divergence_trace::guard(pivots);\n    Ok(GlobalSearchReport {\n        pivots,"),
    ]
    for path, old, new in hooks:
        replace_once(worktree, path, old, new)
    probe = (BUNDLE / "probe.rs").read_text().replace(
        '#[path = "g0.rs"]',
        f'#[path = {json.dumps(str(BUNDLE / "g0.rs"))}]',
    )
    (worktree / "crates/tensor4all-treeaci/examples/guard_1265_trace.rs").write_text(probe)
    patch = subprocess.check_output(["git", "diff", "--binary"], cwd=worktree)
    (output / "instrumented.diff").write_bytes(patch)


def build(repo, output, arm, target):
    worktree = output / f"worktree-{arm}"
    owned = False
    try:
        run_command(["git", "worktree", "add", "--detach", str(worktree), BASE], cwd=repo)
        owned = True
        if arm == "candidate":
            run_command(["git", "apply", str(BUNDLE / "candidate.patch")], cwd=worktree)
        arm_output = output / f"sources-{arm}"
        arm_output.mkdir()
        instrument(worktree, arm_output)
        env = os.environ.copy()
        env["CARGO_TARGET_DIR"] = str(target)
        with (output / f"build-{arm}.log").open("w") as log:
            run_command(["cargo", "build", "--locked", "--release", "-p", "tensor4all-treeaci", "--example", "guard_1265_trace"], cwd=worktree, env=env, stdout=log, stderr=log)
        binary = output / arm
        shutil.copy2(target / "release/examples/guard_1265_trace", binary)
        (output / f"{arm}.binary.json").write_text(json.dumps({
            "base": BASE,
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "bundle_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in BUNDLE.iterdir() if p.is_file()},
        }, indent=2) + "\n")
    finally:
        if owned:
            # The script created this disposable worktree at a new, exclusive
            # output path; force removes only its generated diagnostic overlay.
            run_command(["git", "worktree", "remove", "--force", str(worktree)], cwd=repo)


def run_cell(output, arm, name, cpu, replay=None, prefix=None, focus=False, pass_errors=True):
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("T4A_DIAG_"):
            del env[key]
    env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "RAYON_NUM_THREADS": "1", "T4A_ROOT_IMPORTANCE": "on", "T4A_ROOT_REPAIR": "off", "T4A_DIAG_VALIDATION": str(output / "validation.json")})
    if replay:
        env["T4A_DIAG_REPLAY"] = str(replay)
    if prefix:
        env["T4A_DIAG_MAX_SWEEPS"] = str(prefix)
    if not pass_errors:
        env["T4A_DIAG_PASS_METRICS"] = "off"
    if focus:
        env.update({"T4A_DIAG_EDGE_PASSES": "2,4", "T4A_DIAG_EDGE_FULL": "2:23,2:24,2:25,2:26", "T4A_DIAG_DUMP": "2:24", "T4A_DIAG_DUMP_PATH": str(output / f"{name}-p2s24.bin")})
    command = ["taskset", "-c", str(cpu), str(output / arm), "9", "nblock", "0.01", "2026", "true"]
    with (output / f"{name}.stdout").open("w") as stdout, (output / f"{name}.trace.jsonl").open("w") as stderr:
        result = subprocess.run(command, env=env, stdout=stdout, stderr=stderr)
    (output / f"{name}.run.json").write_text(json.dumps({"command": command, "returncode": result.returncode, "diagnostic_env": {k: v for k, v in env.items() if k.startswith("T4A_")}}, indent=2) + "\n")
    guards = {}
    with (output / f"{name}.trace.jsonl").open() as trace:
        for line in trace:
            if line.startswith("{"):
                event = json.loads(line)
                if event["kind"] == "guard":
                    guards[str(event["pass"])] = event["injected"]
    (output / f"{name}-guard.json").write_text(json.dumps(guards, indent=2) + "\n")
    values = [json.loads(line) for line in (output / f"{name}.stdout").read_text().splitlines() if line.startswith("{")]
    if not values:
        raise RuntimeError(f"{name} produced no numerical output; inspect its retained trace")
    value = values[-1]
    print(name, result.returncode, value["termination"], len(value["ranks"]), value["importance"]["rel_l2"], flush=True)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path, help="new directory for retained sources, binaries and numerical records")
    parser.add_argument("--target-dir", type=Path, help="explicit reusable Cargo target; default: owned out-dir/cargo-target")
    parser.add_argument("--cpu", default=14, type=int)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--skip-pass-errors", action="store_true", help="keep final validation and metadata but omit expensive per-pass held-out evaluation")
    parser.add_argument("--focus", action="store_true", help="also run incomplete five-sweep probes; their unchanged accuracy assertions are expected to fail")
    args = parser.parse_args()
    repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    target = args.target_dir.resolve() if args.target_dir else output / "cargo-target"
    for arm in ["baseline", "candidate"]:
        build(repo, output, arm, target)
    env = os.environ.copy()
    env["T4A_DIAG_MAKE_VALIDATION"] = str(output / "validation.json")
    run_command(["taskset", "-c", str(args.cpu), str(output / "baseline"), "9", "nblock", "0.01", "2026", "true"], env=env)
    if args.build_only:
        return
    results = {}
    for arm in ["baseline", "candidate"]:
        name = arm + "-natural"
        results[name] = run_cell(output, arm, name, args.cpu, pass_errors=not args.skip_pass_errors)
    for arm, source in [("baseline", "baseline"), ("candidate", "candidate"), ("candidate", "baseline"), ("baseline", "candidate")]:
        name = f"{arm}-replay-{source}-natural"
        results[name] = run_cell(output, arm, name, args.cpu, replay=output / f"{source}-natural-guard.json", pass_errors=False)
    for cutoff in [2, 3]:
        for arm, source in [("baseline", "candidate"), ("candidate", "baseline")]:
            points = json.loads((output / f"{source}-natural-guard.json").read_text())
            replay = output / f"{source}-prefix{cutoff}-guard.json"
            replay.write_text(json.dumps({k: v for k, v in points.items() if int(k) <= cutoff}, indent=2) + "\n")
            name = f"{arm}-replay-{source}-prefix{cutoff}"
            results[name] = run_cell(output, arm, name, args.cpu, replay=replay, pass_errors=False)
    if args.focus:
        for arm in ["baseline", "candidate"]:
            results[arm + "-focus"] = run_cell(output, arm, arm + "-focus", args.cpu, prefix=5, focus=True)
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
