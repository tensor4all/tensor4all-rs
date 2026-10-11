"""SVD truncation error at each cap: a quasi-optimal rank-cap reference.

    python -B benchmarks/tree-rsi/agreement/floor.py --author PATH RESULTS_DIR [--cases a,b]

Appends ``floor.jsonl`` to RESULTS_DIR for every (case, cap) present in its
``results.jsonl``. Chains use TT-SVD of the exact product (optimal within a
factor sqrt(n-1)); the tree uses hierarchical SVD from the dense product.
Sampled references (n50 DMRG) have no affordable exact product and are skipped.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run  # noqa: E402
import tn  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--author", required=True)
    parser.add_argument("results")
    parser.add_argument("--cases", default="")
    args = parser.parse_args()
    out = Path(args.results)
    caps = {}
    for line in open(out / "results.jsonl"):
        rec = json.loads(line)
        if "cap" in rec:
            caps.setdefault(rec["case"], set()).add(rec["cap"])
    wanted = [c for c in args.cases.split(",") if c] or sorted(caps)
    registry = run.make_cases(args.author)
    with open(out / "floor.jsonl", "a") as log:
        for name in wanted:
            case = registry[name]()
            if case["ref"] == "sampled":
                continue
            case = run.materialize(case)
            for cap in sorted(caps.get(name, [])):
                if case["ref"] == "tt":
                    trunc = tn.tt_round(case["ref_tt"], 0.0, cap)
                    err = tn.tt_diff_norm(trunc, case["ref_tt"]) / case["ref_norm"]
                else:
                    cores = tn.tree_from_dense(case["ref_dense"], case["edges"], case["root"], 0.0, cap)
                    err = float(np.linalg.norm(tn.tree_to_dense(cores) - case["ref_dense"]) / case["ref_norm"])
                rec = {"case": name, "cap": cap, "svd_error": err}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
