"""Print the driver summary as a markdown table (µs, per-round ratio medians, unrounded min/max)."""
import json, pathlib, sys

s = json.loads(pathlib.Path(sys.argv[1], "summary.json").read_text())
print("decision:", s["decision"])
print("noise:", json.dumps(s["noise"]), json.dumps(s["noise_pass"]))
print("timing:", json.dumps(s["timing"]), json.dumps(s["timing_pass"]))
print()
print("| case | original µs | shared µs | final µs | final/original median [min, max] | final/shared median |")
print("|---|---:|---:|---:|---:|---:|")
order = {"cold_fixed": 0, "warm_fixed": 1, "warm_auto": 2, "left_hit": 3, "right_hit": 4}
for c in sorted(s["cases"], key=lambda c: (order[c["route"]], c["case"])):
    fo, fs = c["final_over_original"], c["final_over_shared"]
    print(f"| {c['case'].split('/',1)[1]} | {c['original_median_ns']/1e3:.4f} | {c['shared_median_ns']/1e3:.4f} | "
          f"{c['final_median_ns']/1e3:.4f} | {fo['median']:.4f} [{fo['min']:.4f}, {fo['max']:.4f}] | {fs['median']:.4f} |")
