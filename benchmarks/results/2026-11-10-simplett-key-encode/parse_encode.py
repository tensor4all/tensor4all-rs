"""Summarize encode-only rounds: median of per-round Criterion mean estimates (ns)."""
import json, pathlib, statistics, sys

out = pathlib.Path(sys.argv[1])
UNIT = {"ps": 1e-3, "ns": 1.0, "µs": 1e3, "us": 1e3, "ms": 1e6}
data = {}  # (variant, bench) -> [mean per round]
for crit in sorted(out.glob("criterion-*-r*")):
    variant = crit.name.split("-")[1]
    for est in crit.glob("index_key_encode_*/binary/*/new/estimates.json"):
        group = est.parts[-5].removeprefix("index_key_encode_")
        width = int(est.parts[-3])
        mean = json.loads(est.read_text())["mean"]["point_estimate"]
        data.setdefault((variant, group, width), []).append(mean)

widths = sorted({k[2] for k in data})
groups = ["encode", "vec_key_hash"]
rows = []
print("| width | group | " + " | ".join(f"{v} (r1/r2/r3 → median)" for v in ["before", "after", "unbox"]) + " | after/before |")
for g in groups:
    for w in widths:
        cells = []
        med = {}
        for v in ["before", "after", "unbox"]:
            xs = data.get((v, g, w), [])
            med[v] = statistics.median(xs) if xs else float("nan")
            cells.append("/".join(f"{x:.1f}" for x in xs) + f" → **{med[v]:.1f}**")
        print(f"| {w} | {g} | " + " | ".join(cells) + f" | {med['after']/med['before']:.3f} |")
        rows.append({"width": w, "group": g, **{v: data.get((v, g, w), []) for v in ["before", "after", "unbox"]}})
(out / "summary.json").write_text(json.dumps(rows, indent=1))
