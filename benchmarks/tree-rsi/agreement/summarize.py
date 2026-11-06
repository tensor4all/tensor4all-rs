"""Summarize one or more agreement ``results.jsonl`` files as Markdown tables.

    python benchmarks/tree-rsi/agreement/summarize.py target/tree-rsi/agreement-*/results.jsonl
"""
import json
import sys
from pathlib import Path

VALID = 1e-6


def e(x):
    return "—" if x is None else f"{x:.1e}"


def main(paths):
    headers, rows, floor = {}, {}, {}
    for path in paths:
        fpath = Path(path).parent / "floor.jsonl"
        if fpath.exists():
            for line in open(fpath):
                rec = json.loads(line)
                floor[(rec["case"], rec["cap"])] = rec["svd_error"]
        for line in open(path):
            rec = json.loads(line)
            if "header" in rec:
                headers[rec["header"]["case"]] = rec["header"]
            else:
                rows[(rec["case"], rec["cap"])] = rec  # later files override earlier caps
    for case, h in headers.items():
        caps = sorted(c for (n, c) in rows if n == case)
        print(f"\n### {case}\n")
        print(f"{h['note']}; n = {h['n']}, d = {h['physical_dims']}, input bond dims {h['input_ranks'][0]}, "
              f"reference: {h['reference']}.\n")
        tt = h["kind"] == "tt"
        cols = ["cap", "SVD trunc. error", "RSI error (3 seeds)", "RSI max rank", "ACI error", "ACI max rank", "ACI stop",
                "max ‖RSI−ACI‖", "both valid"]
        if tt:
            cols[2:2] = ["author error", "replay error", "‖replay−author‖", "ranks equal"]
        cols += ["RSI s (median)", "ACI s"]
        print("| " + " | ".join(cols) + " |")
        print("|" + "---|" * len(cols))
        for cap in caps:
            r = rows[(case, cap)]
            rsi = [r[f"rsi_s{s}"] for s in (1, 2, 3) if f"rsi_s{s}" in r]
            ok = [x for x in rsi if "error" in x and isinstance(x["error"], float)]
            errs = sorted(x["error"] for x in ok)
            aci = r.get("aci", {})
            aci_err = aci.get("error")
            if isinstance(aci_err, str):  # worker failure message
                aci = {"termination": aci_err.split("{")[0].replace("Error:", "").strip() or "error"}
                aci_err = None
            rsi_aci = [v for k, v in (r.get("diffs") or {}).items() if k.startswith("rsi_s")]
            both = bool(errs) and aci_err is not None and max(errs) <= VALID and aci_err <= VALID
            cells = [str(cap), e(floor.get((case, cap))), "/".join(e(x) for x in errs) if errs else "error",
                     str(max(x["max_rank"] for x in ok)) if ok else "—",
                     e(aci_err) if aci_err is not None else "error", str(aci.get("max_rank", "—")),
                     aci.get("termination", "—"), e(max(rsi_aci)) if rsi_aci else "—",
                     "yes" if both else "no"]
            if tt:
                cells[2:2] = [e(r.get("author", {}).get("error")), e(r.get("replay", {}).get("error")),
                              e(r.get("replay_vs_author")), str(r.get("replay_ranks_equal", "—"))]
            times = sorted(x["seconds"] for x in ok)
            cells += [f"{times[len(times) // 2]:.3f}" if times else "—",
                      f"{aci['seconds']:.3f}" if "seconds" in aci else "—"]
            print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
