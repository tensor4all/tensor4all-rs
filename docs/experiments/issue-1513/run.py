#!/usr/bin/env python3
"""Issue #1513 step 2: compare the matrix readout and slab extraction at three revisions.

Revisions: pre-#1516 (element-wise slabs), #1516 (contiguous slab copies), and the
checked-out HEAD (borrowed in-place slabs). Each is built from `git archive` in a
disposable workspace under target/issue-1513/ (the shared target dir of that
workspace is removed at the end) with readout.rs as an example, each with its own target dir.

Usage (repository root, commit HEAD first; the harness is read from this directory):
  RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    nice -n 15 python3 docs/experiments/issue-1513/run.py [--pre REV] [--mid REV] [--passes N] [--reps N]

Writes readout.csv, micro.csv, host.txt, rustc.txt, revisions.txt here. Every
revision must produce the same output hash for every case (bit identity).
"""
import argparse, os, re, shutil, subprocess
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SCRATCH = ROOT / "target" / "issue-1513"
EXAMPLE = Path("crates/tensor4all-tensorci/examples/issue1513_readout.rs")
CASES = [("corr", 2, 3, 2), ("corr", 8, 8, 2), ("unrel", 2, 3, 0), ("unrel", 8, 8, 0)]


def out(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def strip(source, region):
    return re.sub(rf"(?s)[ \t]*// {region}-BEGIN\n.*?// {region}-END\n", "", source)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", default="62c3b606^")
    ap.add_argument("--mid", default="62c3b606")
    ap.add_argument("--reuse-binaries", action="store_true", help="skip building; use readout-* left in target/issue-1513/")
    ap.add_argument("--passes", type=int, default=3, help="processes per revision and case (median of their medians)")
    ap.add_argument("--reps", type=int, default=7, help="timed rounds per process after one warm-up round")
    args = ap.parse_args()
    revs = {"pre-1516": out("git", "-C", str(ROOT), "rev-parse", args.pre).strip(),
            "1516": out("git", "-C", str(ROOT), "rev-parse", args.mid).strip(),
            "step2": out("git", "-C", str(ROOT), "rev-parse", "HEAD").strip()}
    if not args.reuse_binaries:
        shutil.rmtree(SCRATCH, ignore_errors=True)
        SCRATCH.mkdir(parents=True)
    source = (HERE / "readout.rs").read_text()
    bins = {}
    for name, rev in revs.items():
        bins[name] = SCRATCH / f"readout-{name}"
        if args.reuse_binaries and bins[name].exists():
            continue
        work = SCRATCH / name
        work.mkdir()
        # One target dir per revision: git archive stamps every file with the commit time, so
        # cargo's mtime fingerprints would reuse another revision's workspace crates.
        tgt = SCRATCH / f"tgt-{name}"
        subprocess.run(f"git -C {ROOT} archive {rev} | tar -x -C {work}", shell=True, check=True)
        has_borrow = "fn as_col_major_slice" in (work / "crates/tensor4all-simplett/src/tensor.rs").read_text()
        has_fold = (work / "crates/tensor4all-tensorci/src/tensorci2/matrix/folded.rs").exists()
        text = source if has_borrow else strip(source, "BORROW")
        text = text if has_fold else strip(text, "FOLDED")
        (work / EXAMPLE).write_text(text)
        subprocess.run(["cargo", "build", "--locked", "--release", "--jobs", "2", "--target-dir", str(tgt),
                        "-p", "tensor4all-tensorci", "--example", "issue1513_readout"], cwd=work, check=True)
        shutil.copy2(tgt / "release/examples/issue1513_readout", bins[name])
        shutil.rmtree(work)
        shutil.rmtree(tgt)  # the build output is no longer needed
    (HERE / "revisions.txt").write_text("".join(f"{n} {r}\n" for n, r in revs.items()))
    (HERE / "rustc.txt").write_text(out("rustc", "-Vv"))
    (HERE / "host.txt").write_text(out("sysctl", "-n", "machdep.cpu.brand_string") + out("sw_vers") + out("uname", "-m"))
    env = dict(os.environ, RAYON_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")

    def run(name, *a):
        r = subprocess.run(["nice", "-n", "15", str(bins[name]), *map(str, a)], env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        return [l.split(",") for l in r.stdout.splitlines()]

    rows = ["target,shape,rank,revision,median_ms,folded_ms,hash"]
    for kind, r, c, rank in CASES:
        samples = {n: [] for n in bins}
        folded = []
        hashes = {}
        for p in range(args.passes):
            order = list(bins)
            order = order[p % 3:] + order[:p % 3]  # rotate to cancel drift
            for name in order:
                (row,) = run(name, kind, r, c, rank, args.reps)
                samples[name].append(float(row[5]))
                hashes.setdefault(name, set()).add(row[6])
                if len(row) > 7:
                    folded.append(float(row[7]))
        assert len({h for hs in hashes.values() for h in hs}) == 1, f"outputs differ at {kind} {r}x{c}: {hashes}"
        for name in bins:
            f = f"{median(folded):.2f}" if name == "step2" else ""
            rows.append(f"{kind},{r}x{c},{rank},{name},{median(samples[name]):.2f},{f},{next(iter(hashes[name]))}")
        print(rows[-3:], flush=True)
    (HERE / "readout.csv").write_text("\n".join(rows) + "\n")

    micro = ["shape,revision,slice_site_ns,slice_site_then_sum_ns,borrow_then_sum_ns"]
    acc = {}
    for p in range(args.passes):
        order = list(bins)
        order = order[p % 3:] + order[:p % 3]
        for name in order:
            for row in run(name, "micro", 2 * args.reps):
                acc.setdefault((row[1], name), []).append([float(v) for v in row[2:]])
    for (shape, name), vals in acc.items():
        cols = [f"{median(v[i] for v in vals):.1f}" for i in range(len(vals[0]))]
        micro.append(f"{shape},{name},{cols[0]},{cols[1]},{cols[2] if len(cols) > 2 else ''}")
    (HERE / "micro.csv").write_text("\n".join(micro) + "\n")
    shutil.rmtree(SCRATCH, ignore_errors=True)


if __name__ == "__main__":
    main()
