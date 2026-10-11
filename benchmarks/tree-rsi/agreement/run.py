"""Agreement benchmark: Rust tree RSI versus the author's TT RSI and TreeACI.

Run from the repository root with an interpreter that has NumPy, h5py and
TensorLy (the author's code imports TensorLy), passing the author checkout:

    python -B benchmarks/tree-rsi/agreement/run.py --author PATH [--cases a,b]

Outputs go only to ignored ``target/tree-rsi/agreement-<timestamp>/``.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tn  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
VALID = 1e-6  # relative L2 error at which an output counts as valid


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- fixtures

def read_author_mps(path):
    with h5py.File(path, "r") as f:
        n = int(f["num_sites"][()])
        mps = [f[f"tensor_{i}"][:] for i in range(1, n + 1)]
    mps.reverse()  # same orientation as the author's readh5_mps
    mps[0] = mps[0].reshape(1, *mps[0].shape)
    mps[-1] = mps[-1].reshape(*mps[-1].shape, 1)
    return [np.ascontiguousarray(c, dtype=np.float64) for c in mps]


def bits_grid(nbits, naxes):
    """Dense coordinate grids (values in [0,1)) with node axis i = (axis i % naxes,
    bit i // naxes), most significant bit first (gw-rs interleaved/comb site map)."""
    n = nbits * naxes
    grids = []
    for a in range(naxes):
        coord = np.zeros((2,) * n)
        for b in range(nbits):
            shape = [1] * n
            shape[a + naxes * b] = 2
            coord = coord + np.arange(2).reshape(shape) * 2.0 ** (-(b + 1))
        grids.append(coord)
    return grids


def comb_edges(nbits):
    edges = [(0, 1), (1, 2)]
    for start in range(3):
        for bit in range(1, nbits):
            edges.append((start + 3 * (bit - 1), start + 3 * bit))
    return edges


def chain_edges(n):
    return [(i, i + 1) for i in range(n - 1)]


def g0_like_pair(nbits, eta=0.08):
    kx, ky, w = bits_grid(nbits, 3)
    eps = -0.5 * (np.cos(2 * np.pi * kx) + np.cos(2 * np.pi * ky))
    omega = 2.0 * (w - 0.5)
    f = eta / ((omega - eps) ** 2 + eta ** 2)
    g = np.exp(-((omega - 0.2) ** 2) / 0.3) * (1.0 + 0.5 * np.cos(2 * np.pi * kx) * np.cos(2 * np.pi * ky))
    return f, g


def make_cases(author):
    """Each case: name, kind (tt/tree), operands (fixture cores), edges, root,
    caps, reference mode, author sketch-width rule."""
    cases = {}
    dmrg = Path(author) / "datasets" / "itensor_dmrg_mps"

    def dmrg_case(name, file, caps, ref):
        psi = read_author_mps(dmrg / file)
        return dict(kind="tt", tt=[psi, psi], caps=caps, ref=ref,
                    note=f"author DMRG MPS {file}, max bond {max(c.shape[2] for c in psi)}")

    cases["dmrg_n10_chi20"] = lambda: dmrg_case("dmrg_n10_chi20", "n10_system/psi_maxdim20_n10.h5",
                                                [10, 20, 40, 80, 120], "dense")
    for chi in (10, 20, 30):
        cases[f"dmrg_n20_chi{chi}"] = (lambda chi=chi: dmrg_case(
            f"dmrg_n20_chi{chi}", f"n20_system/psi_maxdim{chi}_n20.h5", [10, 20, 40, 80, 120], "tt"))
    for chi in (30, 60, 80, 150):
        cases[f"dmrg_n50_chi{chi}"] = (lambda chi=chi: dmrg_case(
            f"dmrg_n50_chi{chi}", f"n50_system/psi_maxdim{chi}_n50.h5", [20, 40, 80, 120, 160, 200],
            "sampled"))

    def gauss():
        n = 22
        (x,) = bits_grid(n, 1)
        s = 0.15
        f = np.exp(-((x - 0.25) ** 2) / (2 * s * s))
        g = np.exp(-((x - 0.75) ** 2) / (2 * s * s))
        return dict(kind="tt", dense=[f, g], edges=chain_edges(n), caps=[4, 8, 16, 32], ref="dense",
                    note="paper IV.C Gaussians centred at 0.25/0.75, sigma 0.15, 2^22 grid")

    def ring2d():
        n = 22
        x, y = bits_grid(11, 2)
        r2 = (x - 0.5) ** 2 + (y - 0.5) ** 2
        f = 1.0 / (1.0 + (r2 - 0.09) ** 2 / 4e-4)
        g = np.cos(2 * np.pi * (3 * x + 5 * y)) * np.exp(-((x - 0.4) ** 2 + (y - 0.6) ** 2) / 0.1)
        return dict(kind="tt", dense=[f, g], edges=chain_edges(n), caps=[16, 32, 64, 128, 192],
                    ref="dense", note="2D ring Lorentzian x oscillating Gaussian, interleaved 11+11 bits")

    def g0_chain():
        f, g = g0_like_pair(7)
        return dict(kind="tt", dense=[f, g], edges=chain_edges(21), caps=[16, 32, 64, 128, 192],
                    ref="dense", note="G0-like (kx,ky,omega) 3x7 bits, gw-rs interleaved chain")

    def g0_comb():
        f, g = g0_like_pair(7)
        return dict(kind="tree", dense=[f, g], edges=comb_edges(7), caps=[16, 32, 64, 128, 192],
                    ref="dense", note="G0-like (kx,ky,omega) 3x7 bits, gw-rs comb (CTTN), junction degree 3")

    def g0_grouped():
        r = 8
        f, g = g0_like_pair(r)
        # gw-rs grouped chain: kx fine->coarse, omega coarse->fine, ky coarse->fine.
        order = [3 * b for b in range(r)][::-1] + [2 + 3 * b for b in range(r)] + [1 + 3 * b for b in range(r)]
        return dict(kind="tt", dense=[np.transpose(f, order), np.transpose(g, order)],
                    edges=chain_edges(3 * r), caps=[32, 64, 128, 192, 256, 384], ref="dense",
                    input_rtol=1e-8, note="G0-like (kx,ky,omega) 3x8 bits, gw-rs grouped chain, inputs at 1e-8")

    def g0_comb8():
        f, g = g0_like_pair(8)
        return dict(kind="tree", dense=[f, g], edges=comb_edges(8), caps=[32, 64, 128, 192, 256, 384],
                    ref="dense", input_rtol=1e-8,
                    note="G0-like (kx,ky,omega) 3x8 bits, gw-rs comb (CTTN), junction degree 3, inputs at 1e-8")

    def comb_case(nbits, pair, note, caps, root=1):
        f, g = pair
        return dict(kind="tree", dense=[f, g], edges=comb_edges(nbits), caps=caps, ref="dense",
                    input_rtol=1e-8, root=root, junction=1, aci_max_working_bytes=6 << 30,
                    note=note + f", gw-rs comb (CTTN) {nbits} bits/axis, junction ky0 degree 3, root {root}, inputs at 1e-8")

    def ball_pair(nbits):
        kx, ky, w = bits_grid(nbits, 3)
        f = 1.0 / (1.0 + 20.0 * ((kx - 0.5) ** 2 + (ky - 0.45) ** 2 + (w - 0.55) ** 2))
        g = np.cos(2 * np.pi * (kx + 2 * ky + 3 * w)) * np.exp(-((kx - 0.3) ** 2 + (w - 0.6) ** 2) / 0.2)
        return f, g

    cases["g0_comb8_big"] = lambda: comb_case(8, g0_like_pair(8), "G0-like eta 0.08", [256, 384])
    cases["g0_comb8_leafroot"] = lambda: comb_case(8, g0_like_pair(8), "G0-like eta 0.08",
                                                   [64, 128, 192, 256], root=2 + 3 * 7)
    cases["g0_comb8_eta02"] = lambda: comb_case(8, g0_like_pair(8, 0.2), "G0-like eta 0.2",
                                                [32, 64, 96, 128, 192])
    cases["ball_comb8"] = lambda: comb_case(8, ball_pair(8), "3D Lorentzian ball x 3D plane wave",
                                            [32, 64, 96, 128, 192, 256])
    cases["g0_comb9"] = lambda: comb_case(9, g0_like_pair(9), "G0-like eta 0.08", [64, 128, 192, 256, 384])
    cases["g0_grouped"] = g0_grouped
    cases["g0_comb8"] = g0_comb8
    cases["qtt_gauss"] = gauss
    cases["qtt_ring2d"] = ring2d
    cases["g0_interleaved"] = g0_chain
    cases["g0_comb"] = g0_comb
    return cases


def materialize(case):
    """Fill operands (fixture cores), root and the reference."""
    if "tt" in case:
        case["operands"] = [tn.fixture_from_tt(t) for t in case["tt"]]
        n = len(case["tt"][0])
        case["edges"] = chain_edges(n)
    else:
        n = case["dense"][0].ndim
        root = n - 1 if case["kind"] == "tt" else 1
        rtol = case.get("input_rtol", 1e-12)
        case["operands"] = [tn.tree_from_dense(d, case["edges"], root, rtol) for d in case["dense"]]
        case["input_error"] = [float(np.linalg.norm(tn.tree_to_dense(o) - d) / np.linalg.norm(d))
                               for o, d in zip(case["operands"], case["dense"])]
        del case["dense"]
    case["n"] = n
    case["physical_dims"] = [int(c["array"].shape[-1]) for c in case["operands"][0]]
    # Root: the paper's left-to-right order on chains; the degree-3 junction on combs
    # unless the case names another root.
    case.setdefault("root", n - 1 if case["kind"] == "tt" else 1)
    if case["kind"] == "tt":
        case["tt"] = [tn.tt_from_fixture(o) for o in case["operands"]]
    if case["ref"] == "dense":
        a, b = (tn.tree_to_dense(o) for o in case["operands"])
        case["ref_dense"] = a * b
        case["ref_norm"] = float(np.linalg.norm(case["ref_dense"]))
    elif case["ref"] == "tt":
        exact = tn.tt_hadamard(*case["tt"])
        case["ref_tt"] = tn.tt_round(exact, 1e-14)
        case["ref_norm"] = tn.tt_norm(case["ref_tt"])
    else:
        rng = np.random.default_rng(12345)
        psi = case["tt"][0]
        case["samples"] = tn.tt_born_samples(psi, 20000, rng)
        case["sample_ref"] = tn.tt_eval(case["tt"][0], case["samples"]) * tn.tt_eval(case["tt"][1], case["samples"])
    case["input_ranks"] = [sorted({int(s) for c in o for s in c["array"].shape[:-1]}) for o in case["operands"]]
    return case


def write_fixture(case, path):
    path.mkdir(parents=True, exist_ok=True)
    flat, operands = [], []
    offset = 0
    for op in case["operands"]:
        meta = []
        for c in op:
            data = np.asarray(c["array"], dtype=np.float64).ravel(order="F")
            meta.append({"neighbors": c["neighbors"], "shape": list(c["array"].shape),
                         "offset": offset, "len": int(data.size)})
            flat.append(data)
            offset += data.size
        operands.append(meta)
    np.concatenate(flat).astype("<f8").tofile(path / "inputs.bin")
    (path / "fixture.json").write_text(json.dumps(
        {"n": case["n"], "physical_dims": case["physical_dims"], "operands": operands}))


def read_output(prefix):
    record = json.loads(prefix.with_suffix(".json").read_text())
    values = np.fromfile(prefix.with_suffix(".bin"), dtype="<f8")
    cores = [{"neighbors": m["neighbors"],
              "array": values[m["offset"]: m["offset"] + m["len"]].reshape(m["shape"], order="F")}
             for m in record["cores"]]
    return record, cores


# ---------------------------------------------------------------- algorithms

def run_worker(worker, fixture, algorithm, request, prefix):
    req = prefix.with_suffix(".request.json")
    req.write_text(json.dumps(request))
    env = dict(os.environ, RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1")
    cmd = ["taskset", "-c", os.environ.get("AGREEMENT_CPU", "2"), str(worker), str(fixture),
           algorithm, str(req), str(prefix)]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        return {"error": (proc.stderr or proc.stdout).strip()[-2000:]}, None
    return read_output(prefix)


def run_author(author_src, tt, cap, k, seed):
    """The author's HadamardTT_RSI with eps=0, recording its probe draws."""
    if str(author_src) not in sys.path:
        sys.path.insert(0, str(author_src))
    from multiply_rsi import HadamardTT_RSI
    draws = []
    original = np.random.normal

    def recording(*args, **kwargs):
        value = original(*args, **kwargs)
        draws.append(np.array(value, copy=True))
        return value

    np.random.seed(seed)
    np.random.normal = recording
    try:
        start = time.perf_counter()
        out = HadamardTT_RSI(tt[0], tt[1], 2, cap, 0, k, seed, verbose=False)
        seconds = time.perf_counter() - start
    finally:
        np.random.normal = original
    return out, draws, seconds


def author_probe_file(draws, n, k, path):
    """Map recorded draws (input-major, then cores 2..n-1) to Rust probe blocks."""
    tail = n - 2
    assert len(draws) == 2 * tail, (len(draws), tail)
    layout, flat, offset = [], [], 0
    for a in range(2):
        for i in range(tail):
            block = draws[a * tail + i]  # (d, k), rows = site value, columns = probe label
            assert block.shape[1] == k
            layout.append([a, 2 + i, int(block.shape[0]), k, offset])
            flat.append(block.ravel(order="F"))
            offset += block.size
    np.concatenate(flat).astype("<f8").tofile(path)
    return layout


# ---------------------------------------------------------------- metrics

def error_of(case, cores):
    """Relative L2 error against the case reference (``sampled`` is an estimate)."""
    if case["ref"] == "dense":
        return float(np.linalg.norm(tn.tree_to_dense(cores) - case["ref_dense"]) / case["ref_norm"])
    tt = tn.tt_from_fixture(cores)
    if case["ref"] == "tt":
        return tn.tt_diff_norm(tt, case["ref_tt"]) / case["ref_norm"]
    # Born samples x ~ psi^2 = P: mean((A-P)^2/P) / mean(P) estimates ||A-P||^2/||P||^2.
    p = case["sample_ref"]
    a = tn.tt_eval(tt, case["samples"])
    return float(np.sqrt(np.mean((a - p) ** 2 / p) / np.mean(p)))


def difference(case, x, y):
    """Exact ``||x - y|| / max(||x||, ||y||)`` between two outputs."""
    if case["kind"] == "tree" or case["ref"] == "dense":
        dx, dy = tn.tree_to_dense(x), tn.tree_to_dense(y)
        return float(np.linalg.norm(dx - dy) / max(np.linalg.norm(dx), np.linalg.norm(dy)))
    tx, ty = tn.tt_from_fixture(x), tn.tt_from_fixture(y)
    return tn.tt_diff_norm(tx, ty) / max(tn.tt_norm(tx), tn.tt_norm(ty))


# ---------------------------------------------------------------- driver

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--author", required=True)
    parser.add_argument("--cases", default="")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--caps", default="", help="override every selected case's caps")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    author_src = Path(args.author) / "src"
    out = Path(args.out) if args.out else REPO / "target" / "tree-rsi" / time.strftime("agreement-%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]

    subprocess.run(["cargo", "build", "--release", "-p", "tree-rsi-benchmark", "--bin", "agreement-worker"],
                   cwd=REPO, check=True)
    worker_src = REPO / "target" / "release" / "agreement-worker"
    worker = out / "agreement-worker"
    worker.write_bytes(worker_src.read_bytes())
    worker.chmod(0o755)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True).stdout
    author_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=args.author, capture_output=True,
                                 text=True).stdout.strip()
    (out / "provenance.json").write_text(json.dumps({
        "repo_head": head, "repo_dirty": dirty.splitlines(), "worker_sha256": sha256(worker),
        "author_head": author_head, "numpy": np.__version__, "valid_threshold": VALID,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=1))

    registry = make_cases(args.author)
    names = [c for c in args.cases.split(",") if c] or list(registry)
    log = open(out / "results.jsonl", "a")
    for name in names:
        case = materialize(registry[name]())
        if args.caps:
            case["caps"] = [int(c) for c in args.caps.split(",")]
        fixture = out / name / "fixture"
        write_fixture(case, fixture)
        header = {"case": name, "kind": case["kind"], "n": case["n"], "note": case["note"], "root": case["root"],
                  "physical_dims": sorted(set(case["physical_dims"])), "input_ranks": case["input_ranks"],
                  "reference": case["ref"], "input_error": case.get("input_error"),
                  "input_rtol": case.get("input_rtol", 1e-12),
                  "ref_norm": case.get("ref_norm")}
        log.write(json.dumps({"header": header}) + "\n")
        log.flush()
        print(json.dumps(header), flush=True)
        for cap in case["caps"]:
            row = {"case": name, "cap": cap}
            outputs = {}
            # Library-default Rust RSI, independent seeds.
            for seed in seeds:
                rec, cores = run_worker(worker, fixture, "rsi", {"cap": cap, "seed": seed, "root": case["root"]},
                                        out / name / f"rsi-c{cap}-s{seed}")
                row[f"rsi_s{seed}"] = summarize(case, rec, cores)
                outputs[f"rsi_s{seed}"] = cores
            # TreeACI from the same source tree.
            aci_request = {"cap": cap, "seed": 1, "root": case["root"]}
            if "aci_max_working_bytes" in case:
                aci_request["max_working_bytes"] = case["aci_max_working_bytes"]
            rec, cores = run_worker(worker, fixture, "aci", aci_request,
                                    out / name / f"aci-c{cap}")
            row["aci"] = summarize(case, rec, cores)
            outputs["aci"] = cores
            if case["kind"] == "tt":
                k = cap // 2 + 10  # the author's DMRG/function scripts: floor(cap/2) + oversampling 10
                tt_out, draws, secs = run_author(author_src, case["tt"], cap, k, 1)
                acores = tn.fixture_from_tt(tt_out)
                row["author"] = {"seconds": secs, "ranks": [int(c.shape[2]) for c in tt_out[:-1]],
                                 "error": error_of(case, acores), "sketch_dim": k}
                outputs["author"] = acores
                probes = out / name / f"probes-c{cap}.bin"
                layout = author_probe_file(draws, case["n"], k, probes)
                rec, cores = run_worker(worker, fixture, "rsi",
                                        {"cap": cap, "seed": 0, "root": case["root"], "sketch_dim": k,
                                         "rel_tol": 0.0, "probes": str(probes), "probe_layout": layout},
                                        out / name / f"replay-c{cap}")
                row["replay"] = summarize(case, rec, cores)
                outputs["replay"] = cores
                if cores is not None:
                    row["replay_vs_author"] = difference(case, cores, acores)
                    row["replay_ranks_equal"] = row["replay"]["ranks"] == row["author"]["ranks"]
            diffs = {}
            for key in [k for k in outputs if k != "aci" and outputs[k] is not None]:
                if outputs["aci"] is not None:
                    diffs[f"{key}_vs_aci"] = difference(case, outputs[key], outputs["aci"])
            row["diffs"] = diffs
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(compact(row)), flush=True)
    log.close()


def summarize(case, rec, cores):
    if cores is None:
        return rec
    ranks = [r for (_, _, r) in rec["edge_ranks"]]
    out = {"seconds": rec["seconds"], "max_rank": max(ranks, default=1), "error": error_of(case, cores),
           "ranks": chain_ranks(rec) if case["kind"] == "tt" else ranks}
    d = rec["diagnostics"]
    if "termination" in d:
        out["termination"] = d["termination"]
    else:
        out["sketch_dim"] = d["sketch_dim"]
        out["sketched_edges"] = sum(1 for e in d["edges"] if not e["exact_columns"])
        if "junction" in case:
            j = case["junction"]
            out["junction_edges"] = [[e["child"], e["parent"], "exact" if e["exact_columns"] else "sketched"]
                                     for e in d["edges"] if j in (e["child"], e["parent"])]
    return out


def chain_ranks(rec):
    return [r for (_, _, r) in sorted(rec["edge_ranks"])]


def compact(row):
    def fmt(v):
        if isinstance(v, dict) and all(isinstance(x, float) for x in v.values()):
            return {k: f"{x:.2e}" for k, x in v.items()}
        if isinstance(v, dict):
            return {k: (f"{x:.2e}" if isinstance(x, float) else x) for k, x in v.items()
                    if k in ("error", "seconds", "max_rank", "termination")}
        return f"{v:.2e}" if isinstance(v, float) else v
    return {k: fmt(v) for k, v in row.items()}


if __name__ == "__main__":
    main()
