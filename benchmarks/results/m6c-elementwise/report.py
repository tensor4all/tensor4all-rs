#!/usr/bin/env python3
"""Generate the tables of the M6c element-wise benchmark report.

Reads the raw JSONL of `benchmark_m6c_elementwise` under `raw/` (one
directory per run group, each with a PROVENANCE file) and rewrites every
`<!-- BEGIN GENERATED: name -->` ... `<!-- END GENERATED: name -->` block of
the report, so no table in it is typed by hand.

Usage: python3 report.py [report.md]
"""

import gzip
import json
import math
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
REPORT = HERE.parent / "2026-11-11-m6c-elementwise.md"


def lines(path, kind=None):
    out = []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            d = json.loads(line)
            if kind is None or d.get("kind") == kind:
                out.append(d)
    return out


def config(path):
    return lines(path, "config")[0]


def variant_value(variant, key, default):
    for pair in (variant or "").split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            if k == key:
                return v
    return default


def g(x, digits=2):
    if x is None:
        return "-"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return str(x)
    return f"{x:.{digits}g}" if isinstance(x, float) else str(x)


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def operation_error(acc, key):
    """The error against f of the given inputs: `*_vs_inputs` for products,
    the analytic G0 for the Dyson map (its inputs are exact)."""
    return acc.get(f"{key}_vs_inputs", acc.get(key))


# ---------------------------------------------------------------------------


def exhaustive():
    rows = []
    for name, label in [
        ("g0-r6-k1", "G0"),
        ("g0-r7", "G0"),
        ("g0sq-r6", "G0 * G0"),
        ("mixed-k1", "G0 * G0, differently partitioned"),
    ]:
        path = RAW / "r6-regen" / f"{name}.jsonl"
        c = config(path)
        for r in lines(path, "result"):
            acc = r["accuracy"]
            rtol = float(variant_value(r["variant"], "rtol", c["rtol"]))
            l2 = operation_error(acc, "rel_l2")
            mx = operation_error(acc, "rel_max")
            rows.append([label, c["r"], c["temperature"], c["cap"], g(rtol), r["patches"],
                         g(l2), g(mx), g(l2 / rtol)])
    return table(["workload", "R", "T", "cap", "rtol", "patches", "exact rel. L2 error",
                  "exact rel. max error", "L2 error / rtol"], rows)


def layouts():
    rows = []
    for group, name in [("r6-regen", "mixed-k1"), ("r12-mixed", "mixed")]:
        path = RAW / group / f"{name}.jsonl"
        if not path.exists():
            continue
        c = config(path)
        lay = lines(path, "input_layout")[0]
        for r in lines(path, "result"):
            rtol = float(variant_value(r["variant"], "rtol", c["rtol"]))
            rows.append([c["r"], c["temperature"], f"{lay['patches'][0]} / {lay['patches'][1]}",
                         lay["shared_projectors"], lay["common_refinement"], g(rtol),
                         r["patches"], g(operation_error(r["accuracy"], "rel_l2"))])
    return table(["R", "T", "input patches", "shared projectors", "common refinement", "rtol",
                  "output patches", "rel. L2 error vs inputs"], rows)


def tolerance_factor():
    rows = []
    for workload, prefix in [("G0 * G0, differently partitioned", "mixed"), ("G0", "g0-r6")]:
        for k in (1, 3, 10):
            path = RAW / "r6-regen" / f"{prefix}-k{k}.jsonl"
            for r in lines(path, "result"):
                e, acc = r["engine"], r["accuracy"]
                rtol = float(variant_value(r["variant"], "rtol", 0))
                l2 = operation_error(acc, "rel_l2")
                rows.append([workload, k, g(rtol), r["patches"], e["verification_failures"],
                             e["engine_retries"], e["kernel"]["calls"], f"{r['seconds']:.1f}",
                             r["params"], g(l2), g(l2 / rtol)])
    return table(["workload (R = 6, T = 0.1, cap 16)", "k", "rtol", "patches",
                  "failed measurements", "retries", "kernel runs", "time [s]", "parameters",
                  "exact rel. L2 error", "error / rtol"], rows)


def audit():
    rows = []
    for name in ["audit-rtol1e-3-m10", "audit-rtol1e-4-m10", "audit-rtol1e-3-m1"]:
        path = RAW / "r6-regen" / f"{name}.jsonl.gz"
        c = config(path)
        calls = [d for d in lines(path, "kernel_call") if "audit_error" not in d]
        errors = len(lines(path, "kernel_call")) - len(calls)
        tau = calls[0]["tau"]
        conv = [d for d in calls if d["termination"] == "Converged"]
        false_conv = [d for d in conv if d["exact_rms_err"] > tau]
        iter_limit = [d for d in calls if d["termination"] == "IterationLimit"]
        bad = false_conv + iter_limit
        ratios, ratios3 = [], []
        for d in bad:
            t = tau * math.sqrt(2 ** d["active"])
            s2 = max((cut["singular"][1] if len(cut["singular"]) > 1 else 0.0) for cut in d["cuts"])
            s3 = max((cut["singular"][2] if len(cut["singular"]) > 2 else 0.0) for cut in d["cuts"])
            ratios.append(s2 / t)
            ratios3.append(s3 / t)
        near = sum(0.3 <= x <= 3 for x in ratios)
        rank2 = sum(d["rank"] == 2 for d in bad)
        est = statistics.median(d["engine_estimate"] / tau for d in bad) if bad else None
        rms = statistics.median(d["exact_rms_err"] / tau for d in bad) if bad else None
        # The config line does not carry the margin; the run name does.
        margin = float(re.search(r"-m(\d+)$", name).group(1))
        rows.append([g(c["rtol"]), g(margin), len(calls), len(conv), len(false_conv),
                     len(iter_limit), f"{rank2} / {len(bad)}", g(est), g(rms),
                     g(statistics.median(ratios)) if ratios else "-",
                     f"{near} / {len(bad)}",
                     g(statistics.median(ratios3)) if ratios3 else "-", errors])
    return table(["rtol", "TreeACI margin", "kernel runs", "Converged",
                  "Converged with exact RMS > tau", "IterationLimit",
                  "rank 2, failing runs", "median estimate / tau, failing runs",
                  "median exact RMS / tau, failing runs",
                  "median sigma_2 / (tau sqrt N), failing runs", "within [0.3, 3]",
                  "median sigma_3 / (tau sqrt N), failing runs", "audit errors"], rows)


def stable_from(ranks):
    top = max(ranks)
    for i in range(len(ranks)):
        if max(ranks[i:]) - min(ranks[i:]) <= 0.03 * top:
            return i + 1
    return len(ranks)


def last_change(ranks):
    """The last sweep at which the maximum rank changed (1-based)."""
    changes = [i + 1 for i in range(1, len(ranks)) if ranks[i] != ranks[i - 1]]
    return max(changes, default=1)


def topology_scan():
    rows = []
    for t in ("0.1", "0.05", "0.03", "0.01"):
        params = {}
        for topo in ("cttn", "tt_nblock"):
            path = RAW / "topology-scan" / f"{topo}-T{t}.jsonl"
            r = lines(path, "result")[0]
            e = r["engine"]
            params[topo] = r["params"]
            rows.append([t, topo, r["max_rank"], r["params"], e["termination"], e["sweeps"],
                         stable_from(e["max_ranks"]), last_change(e["max_ranks"]),
                         f"{r['seconds']:.0f}", g(r["accuracy"]["rel_l2"])])
        rows.append([t, "cttn / tt_nblock", "", g(params["cttn"] / params["tt_nblock"], 3),
                     "", "", "", "", "", ""])
    return table(["T", "topology", "max rank", "parameters", "termination", "sweeps",
                  "max rank within 3% from sweep", "last change of max rank at sweep",
                  "time [s]", "sampled rel. L2 error"], rows)


def probe_784():
    rows = []
    for mode, label in [("rel", "relative 1e-4"), ("abs", "absolute 3.183e-3")]:
        for arm in ("base", "fix"):
            path = RAW / "784probe" / f"r10-{mode}-{arm}.jsonl"
            r = lines(path, "result")[0]
            e = r["engine"]
            rows.append([label, arm, f"{r['seconds']:.1f}", e["termination"], e["sweeps"],
                         r["max_rank"], stable_from(e["max_ranks"]), last_change(e["max_ranks"]),
                         r["params"], g(r["accuracy"]["rel_l2"], 3)])
    return table(["tolerance", "arm", "time [s]", "termination", "sweeps", "max rank",
                  "max rank within 3% from sweep", "last change of max rank at sweep",
                  "parameters", "sampled rel. L2 error"], rows)


def comb_runs():
    out = {}
    for path in sorted((RAW / "comb-T0.1-R12").glob("s*-*.jsonl")):
        seed, method = path.stem.split("-", 1)
        c = config(path)
        for r in lines(path, "result"):
            out.setdefault((int(seed[1:]), method), []).append((c, r))
    return out


def comb_table():
    runs = comb_runs()
    rows = []
    for (seed, method), items in sorted(runs.items()):
        for c, r in items:
            e, acc = r["engine"], r["accuracy"]
            if method == "unpatched":
                setting = f"tol {variant_value(r['variant'], 'tol', c['tol'])}"
                detail = f"{e['termination']}, {e['sweeps']} sweeps"
            else:
                rtol = variant_value(r["variant"], "rtol", c["rtol"])
                k = variant_value(r["variant"], "engine-tol-factor", "1")
                setting = f"rtol {rtol}, k {k}" if method == "patched" else f"rtol {c['rtol']}"
                detail = (f"{e['splits']} splits, {e['verification_failures']} failed "
                          f"measurements; kernel {e['kernel']['seconds']:.1f} s, "
                          f"measurement {e['residual_engine']['seconds']:.1f} s")
            rows.append([seed, method, setting, f"{r['seconds']:.1f}", r["patches"], r["max_rank"],
                         r["params"], f"{g(acc['rel_l2'], 3)} ± {g(acc['rel_l2_se'], 1)}", detail])
    return table(["seed", "method", "setting", "time [s]", "patches", "max rank", "parameters",
                  "sampled rel. L2 error ± s.e.", "engine"], rows)


def comb_matched():
    """Each patched run against the fastest unpatched run of the same seed
    whose sampled error is not larger."""
    runs = comb_runs()
    rows = []
    for (seed, method), items in sorted(runs.items()):
        if method != "patched":
            continue
        unpatched = runs.get((seed, "unpatched"), [])
        for c, r in items:
            err = r["accuracy"]["rel_l2"]
            match = [u for _, u in unpatched if u["accuracy"]["rel_l2"] <= err]
            rtol = variant_value(r["variant"], "rtol", c["rtol"])
            k = variant_value(r["variant"], "engine-tol-factor", "1")
            if not match:
                rows.append([seed, f"rtol {rtol}, k {k}", g(err, 3), "no unpatched run this accurate",
                             "", "", ""])
                continue
            u = min(match, key=lambda u: u["seconds"])
            rows.append([seed, f"rtol {rtol}, k {k}", g(err, 3),
                         f"tol {variant_value(u['variant'], 'tol', '')} ({g(u['accuracy']['rel_l2'], 3)})",
                         g(r["seconds"] / u["seconds"], 3), g(r["params"] / u["params"], 3),
                         r["patches"]])
    return table(["seed", "patched run", "its error", "unpatched match (error)",
                  "time patched / unpatched", "parameters patched / unpatched",
                  "patches"], rows)


def b0_status():
    rows = []
    for path in sorted((RAW / "comb-T0.1-R12").glob("s*-b0.jsonl")):
        results = lines(path, "result")
        time_file = path.with_suffix(".time")
        text = time_file.read_text() if time_file.exists() else ""
        status = re.search(r"Exit status: (\d+)", text)
        if results:
            r = results[0]
            rows.append([path.stem, f"{r['seconds']:.0f}", r["patches"], r["params"],
                         g(r["accuracy"]["rel_l2"], 3)])
        else:
            state = f"exit {status.group(1)}" if status else "no exit record"
            rows.append([path.stem, "no result", "", "", state])
    return table(["run", "time [s]", "patches", "parameters", "sampled rel. L2 error / status"],
                 rows)


GENERATORS = {
    "exhaustive": exhaustive,
    "layouts": layouts,
    "tolerance-factor": tolerance_factor,
    "audit": audit,
    "topology-scan": topology_scan,
    "probe-784": probe_784,
    "comb-runs": comb_table,
    "comb-matched": comb_matched,
    "comb-b0": b0_status,
}


def main():
    report = Path(sys.argv[1]) if len(sys.argv) > 1 else REPORT
    text = report.read_text()
    for name, generate in GENERATORS.items():
        pattern = re.compile(
            rf"(<!-- BEGIN GENERATED: {re.escape(name)} -->\n).*?(<!-- END GENERATED: {re.escape(name)} -->)",
            re.S,
        )
        if not pattern.search(text):
            sys.exit(f"missing block {name}")
        text = pattern.sub(lambda m: m.group(1) + generate() + "\n" + m.group(2), text)
    report.write_text(text)


if __name__ == "__main__":
    main()
