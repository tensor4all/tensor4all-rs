#!/usr/bin/env python3
"""Issue #1463 pilot: build matrix_tci2.rs against HEAD in target/issue-1463/ and run 16 cases.

Usage (repository root, see README.md): python3 docs/experiments/issue-1463/run.py
Writes head.txt, host.txt, rustc.txt and results.csv here; removes the workspace afterwards.
"""
import os, shutil, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SCRATCH = ROOT / "target" / "issue-1463"
EXAMPLE = Path("crates/tensor4all-tensorci/examples/issue1463_matrix.rs")
CASES = [(k, r, c, rank, piv, cost) for piv in ("full", "rook") for cost in ("cheap", "burn")
         for (k, rank) in (("corr", 2), ("unrel", 0)) for (r, c) in ((2, 3), (8, 8))]
REPS = 3  # after one warm-up round


def out(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


head = out("git", "-C", str(ROOT), "rev-parse", "HEAD").strip()
work = SCRATCH / "work"
shutil.rmtree(SCRATCH, ignore_errors=True)
work.mkdir(parents=True)
subprocess.run(f"git -C {ROOT} archive {head} | tar -x -C {work}", shell=True, check=True)
shutil.copyfile(HERE / "matrix_tci2.rs", work / EXAMPLE)
(HERE / "head.txt").write_text(head + "\n")
(HERE / "rustc.txt").write_text(out("rustc", "-Vv"))
(HERE / "host.txt").write_text(out("sysctl", "-n", "machdep.cpu.brand_string") + out("sw_vers") + out("uname", "-m"))
tgt = SCRATCH / "tgt"
subprocess.run(["cargo", "build", "--locked", "--release", "--jobs", "2", "--target-dir", str(tgt),
                "-p", "tensor4all-tensorci", "--example", "issue1463_matrix"], cwd=work, check=True)
env = dict(os.environ, RAYON_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
rows = ["target,shape,rank,pivot,cost,variant,source_calls,max_bond,err,tt_bytes,median_ms"]
for case in CASES:
    r = subprocess.run([str(tgt / "release/examples/issue1463_matrix"), *map(str, case), str(REPS)],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr  # includes the A == B and C == D assertions
    rows += [l[4:] for l in r.stdout.splitlines() if l.startswith("ROW,")]
    print(*case, flush=True)
(HERE / "results.csv").write_text("\n".join(rows) + "\n")
shutil.rmtree(SCRATCH, ignore_errors=True)
