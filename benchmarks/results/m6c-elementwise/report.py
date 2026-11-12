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


def group_runs(group):
    out = {}
    for path in sorted((RAW / group).glob("s*-*.jsonl")):
        seed, method = path.stem.split("-", 1)
        c = config(path)
        for r in lines(path, "result"):
            out.setdefault((int(seed[1:]), method), []).append((c, r))
    return out


def run_setting(method, c, r):
    if method == "unpatched":
        return f"tol {variant_value(r['variant'], 'tol', c['tol'])}"
    rtol = variant_value(r["variant"], "rtol", c["rtol"])
    k = variant_value(r["variant"], "engine-tol-factor", "1")
    return f"rtol {rtol}, k {k}" if method == "patched" else f"rtol {c['rtol']}"


def runs_table(group):
    runs = group_runs(group)
    rows = []
    for (seed, method), items in sorted(runs.items()):
        for c, r in items:
            e, acc = r["engine"], r["accuracy"]
            if method == "unpatched":
                ranks = e["max_ranks"]
                detail = (f"{e['termination']}, {e['sweeps']} sweeps; max rank within 3% from "
                          f"sweep {stable_from(ranks)}, last change at sweep {last_change(ranks)}")
            else:
                detail = (f"{e['splits']} splits, {e['verification_failures']} failed "
                          f"measurements; kernel {e['kernel']['seconds']:.1f} s, "
                          f"measurement {e['residual_engine']['seconds']:.1f} s")
            rows.append([seed, method, run_setting(method, c, r), f"{r['seconds']:.1f}",
                         r["patches"], r["max_rank"], r["params"],
                         f"{g(acc['rel_l2'], 3)} ± {g(acc['rel_l2_se'], 1)}", detail])
    return table(["seed", "method", "setting", "time [s]", "patches", "max rank", "parameters",
                  "sampled rel. L2 error ± s.e.", "engine"], rows)


def matched_table(group):
    """Each patched run against the fastest unpatched run of the same seed
    whose sampled error is not larger."""
    runs = group_runs(group)
    rows = []
    for (seed, method), items in sorted(runs.items()):
        if method != "patched":
            continue
        unpatched = runs.get((seed, "unpatched"), [])
        for c, r in items:
            err = r["accuracy"]["rel_l2"]
            match = [u for _, u in unpatched if u["accuracy"]["rel_l2"] <= err]
            setting = run_setting(method, c, r)
            if not match:
                rows.append([seed, setting, g(err, 3), "no unpatched run this accurate",
                             "", "", "", ""])
                continue
            u = min(match, key=lambda u: u["seconds"])
            uerr = u["accuracy"]["rel_l2"]
            rows.append([seed, setting, g(err, 3),
                         f"tol {variant_value(u['variant'], 'tol', '')} ({g(uerr, 3)})",
                         g(err / uerr, 2), g(r["seconds"] / u["seconds"], 3),
                         g(r["params"] / u["params"], 3), r["patches"]])
    return table(["seed", "patched run", "its error", "unpatched match (error)",
                  "its error / match error", "time patched / unpatched",
                  "parameters patched / unpatched", "patches"], rows)


def processes(group):
    """Wall clock, peak memory and exit status of each process, from its
    `/usr/bin/time -v` record (whole process: inputs, runs and accuracy checks)."""
    rows = []
    for path in sorted((RAW / group).glob("s*-*.time")):
        text = path.read_text()
        wall = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", text)
        rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
        status = re.search(r"Exit status: (\d+)", text)
        rows.append([path.stem, wall.group(1) if wall else "-",
                     f"{int(rss.group(1)) / 1024 ** 2:.2f}" if rss else "-",
                     status.group(1) if status else "no exit record"])
    return table(["process", "wall clock", "peak RSS [GiB]", "exit status"], rows)


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


def unpatched_by_sweeps(seed):
    """The unpatched runs of one seed on tt_nblock, T = 0.01, R = 12: the
    50-sweep runs and the same runs stopped after 10, 20 and 30 sweeps, as
    (tol, sweeps, result). A stopped run must be the prefix of its 50-sweep
    run (same seed), which is checked here."""
    full = {variant_value(r["variant"], "tol", c["tol"]): r
            for c, r in group_runs("nblock-T0.01-R12").get((seed, "unpatched"), [])}
    out = [(tol, 50, r) for tol, r in full.items()]
    for c, r in group_runs("nblock-T0.01-R12-sweeps").get((seed, "unpatched"), []):
        tol = variant_value(r["variant"], "tol", c["tol"])
        n = int(variant_value(r["variant"], "max-sweeps", "50"))
        f = full[tol]["engine"]
        if (r["engine"]["max_ranks"] != f["max_ranks"][:n]
                or r["engine"]["max_errors"] != f["max_errors"][:n]):
            raise SystemExit(f"seed {seed} tol {tol} sweeps {n} is not a prefix of its 50-sweep run")
        out.append((tol, n, r))
    return sorted(out, key=lambda t: (-float(t[0]), t[1]))


def nblock_sweeps():
    rows = []
    for seed in (42, 2026, 7):
        for tol, n, r in unpatched_by_sweeps(seed):
            rows.append([seed, tol, n, f"{r['seconds']:.1f}", r["max_rank"], r["params"],
                         f"{g(r['accuracy']['rel_l2'], 3)} ± {g(r['accuracy']['rel_l2_se'], 1)}"])
    return table(["seed", "tol", "sweeps", "time [s]", "max rank", "parameters",
                  "sampled rel. L2 error ± s.e."], rows)


def nblock_matched_sweeps():
    """Each patched run against the fastest unpatched run of the same seed,
    at any sweep limit, whose sampled error is not larger."""
    rows = []
    for (seed, method), items in sorted(group_runs("nblock-T0.01-R12").items()):
        if method != "patched":
            continue
        unpatched = unpatched_by_sweeps(seed)
        for c, r in items:
            err = r["accuracy"]["rel_l2"]
            match = [(tol, n, u) for tol, n, u in unpatched if u["accuracy"]["rel_l2"] <= err]
            tol, n, u = min(match, key=lambda t: t[2]["seconds"])
            uerr = u["accuracy"]["rel_l2"]
            rows.append([seed, run_setting(method, c, r), g(err, 3),
                         f"tol {tol}, {n} sweeps ({g(uerr, 3)})", g(err / uerr, 2),
                         g(r["seconds"] / u["seconds"], 3), g(r["params"] / u["params"], 3)])
    return table(["seed", "patched run", "its error", "unpatched match (error)",
                  "its error / match error", "time patched / unpatched",
                  "parameters patched / unpatched"], rows)


def nblock_trace():
    """The time of the traced patched run by engine role and termination.
    A residual run is attributed to the kernel run it follows: the first
    after an accepted run is its acceptance measurement, the second its
    audit."""
    path = RAW / "nblock-T0.01-R12-trace" / "s42-patched-trace.jsonl"
    calls = lines(path, "engine_call")
    result = lines(path, "result")[0]
    parts = {}
    def add(name, call):
        n, t, pts = parts.get(name, (0, 0.0, 0))
        parts[name] = (n + 1, t + call["seconds"], pts + call["points"])
    last = None
    following = 0
    for call in calls:
        if call["role"] == "kernel":
            last, following = call, 0
            add(f"kernel, {call['termination']}", call)
        else:
            following += 1
            accepted = last is not None and last["termination"] == "Converged"
            role = ("acceptance" if following == 1 else "audit") if accepted else "other"
            add(f"residual TT, {role}", call)
    total = result["seconds"]
    rows = [[name, n, f"{t:.1f}", f"{100 * t / total:.0f} %", pts]
            for name, (n, t, pts) in sorted(parts.items(), key=lambda kv: -kv[1][1])]
    rest = total - sum(t for _, t, _ in parts.values())
    rows.append(["rest of the driver", "", f"{rest:.1f}", f"{100 * rest / total:.1f} %", ""])
    rows.append(["total", "", f"{total:.1f}", "", ""])
    return table(["part", "runs", "time [s]", "share", "points evaluated"], rows)


# ---------------------------------------------------------------------------
# #1274: acceptance, per-pass history and the max-norm reference


def rtol_of(result):
    return variant_value(result["variant"], "rtol", "")


def kernel_terminations(path):
    """Per result of `path`: (result, {termination: [runs, seconds]}) of its
    kernel `engine_call` lines, which precede the result line."""
    out, current = [], {}
    for d in lines(path):
        if d.get("kind") == "engine_call" and d["role"] == "kernel":
            n, t = current.get(d["termination"], (0, 0.0))
            current[d["termination"]] = (n + 1, t + d["seconds"])
        elif d.get("kind") == "result":
            out.append((d, current))
            current = {}
    return out


def acceptance_runs():
    rows = []
    for seed in (42, 2026, 7):
        for name, label in [("l2-noaudit", "L2, k 3, audit off"),
                            ("sampledmax", "sampled-max (pinned), k 1")]:
            path = RAW / "nblock-T0.01-R12-acceptance" / f"s{seed}-{name}.jsonl"
            for r, kernel in kernel_terminations(path):
                total = sum(t for _, t in kernel.values())
                n_it, t_it = kernel.get("IterationLimit", (0, 0.0))
                rows.append([seed, label, rtol_of(r), f"{r['seconds']:.0f}", r["patches"],
                             r["params"], g(r["accuracy"]["rel_l2"], 3),
                             f"{n_it} / {100 * t_it / total:.0f} %",
                             f"{r['engine']['residual_engine']['seconds']:.0f}"])
    return table(["seed", "acceptance", "rtol", "time [s]", "patches", "parameters",
                  "sampled rel. L2 error", "kernel IterationLimit runs / share of kernel time",
                  "measurement [s]"], rows)


def history():
    """Which TreeACI stop condition fails in the kernel runs that end at the
    sweep limit, over passes 11 to 50. Convergence after pass c needs (1)
    error <= tolerance, (2) no edge rank growth in the last `min_sweeps`
    passes and (3) no Guard pivots in them; where (1) and (3) hold and the
    run went on, (2) failed."""
    rows, total = [], None
    for seed in (42, 2026, 7):
        path = RAW / "nblock-T0.01-R12-history" / f"s{seed}-history.jsonl"
        result = lines(path, "result")[0]
        calls = lines(path, "kernel_history")
        c = dict(below=len(calls), repro=sum(d["reproduced"] for d in calls), it=0, passes=0,
                 f1=0, f3=0, f2=0, changed=0, up=0, down=0)
        for d in calls:
            if d["termination"] != "IterationLimit":
                continue
            c["it"] += 1
            tol, m = d["engine_tolerance"], d["min_sweeps"]
            e, gp, rk = d["max_errors"], d["global_pivots_found"], d["max_ranks"]
            for i in range(10, len(e)):
                c1 = e[i] <= tol
                c3 = all(x == 0 for x in gp[i + 1 - m:i + 1])
                c["passes"] += 1
                c["f1"] += not c1
                c["f3"] += not c3
                c["f2"] += c1 and c3
                c["changed"] += rk[i] != rk[i - 1]
                c["up"] += rk[i] > rk[i - 1]
                c["down"] += rk[i] < rk[i - 1]
        total = {k: (total or {}).get(k, 0) + v for k, v in c.items()}
        rows.append([seed, result["patches"], f"{c['below']} ({c['repro']})", c["it"], c["passes"],
                     c["f1"], c["f3"], c["f2"], c["changed"], f"{c['up']} / {c['down']}"])
    t = total
    rows.append(["all", "", f"{t['below']} ({t['repro']})", t["it"], t["passes"], t["f1"],
                 t["f3"], t["f2"], t["changed"], f"{t['up']} / {t['down']}"])
    return table(["seed", "patches", "kernel runs below the cap (reproduced by the rerun)",
                  "IterationLimit runs", "passes 11 to 50 of those runs",
                  "(1) error > tolerance", "(3) Guard pivots in the last 2 passes",
                  "(1) and (3) hold, so (2) fails", "max rank changed",
                  "max rank up / down"], rows)


def reference_group(name, seed, k):
    """(reference line, results) of a sampled-max process; the k = 1 runs of
    the pinned group must reproduce those of `nblock-T0.01-R12-acceptance`."""
    path = RAW / f"nblock-T0.01-R12-{name}" / f"s{seed}-{'sampledmax' if name == 'reference' else 'given'}-k{k}.jsonl"
    results = lines(path, "result")
    if name == "reference" and k == 1:
        first = lines(RAW / "nblock-T0.01-R12-acceptance" / f"s{seed}-sampledmax.jsonl", "result")
        key = lambda r: (r["patches"], r["params"], r["accuracy"]["rel_l2"])
        if [key(r) for r in results] != [key(r) for r in first]:
            raise SystemExit(f"seed {seed}: the pinned k = 1 rerun differs from the first run")
    return lines(path, "reference")[0], results


def reference_pinned():
    rows = []
    for seed in (42, 2026, 7):
        ref, results = reference_group("reference", seed, 1)
        norm = results[0]["engine"]["norm"]
        rows.append([seed, g(norm["max_reference"], 3), norm["source"],
                     f"{ref['max_abs'] / norm['max_reference']:.0f}"])
    return table(["seed", "pinned max_reference", "source", "exact max abs(G0) / pinned"], rows)


def reference_spread():
    rows = []
    for name, label in [("reference", "pinned"), ("given", "exact (given)")]:
        for k in (1, 3):
            errors = {}
            for seed in (42, 2026, 7):
                _, results = reference_group(name, seed, k)
                for r in results:
                    errors.setdefault(rtol_of(r), []).append(r["accuracy"]["rel_l2"])
            for rtol, es in errors.items():
                rows.append([label, k, rtol, ", ".join(g(e, 3) for e in es),
                             f"{max(es) / min(es):.3g}"])
    return table(["reference", "k", "rtol", "errors of seeds 42, 2026, 7", "largest / smallest"],
                 rows)


def given_matched():
    """Each exact-reference sampled-max run with k = 1 against the L2 run
    (k = 3, audit off) of the same seed whose error is the closest one not
    smaller than its own, within a factor 2."""
    rows = []
    for seed in (42, 2026, 7):
        l2 = lines(RAW / "nblock-T0.01-R12-acceptance" / f"s{seed}-l2-noaudit.jsonl", "result")
        _, given = reference_group("given", seed, 1)
        for s in given:
            es = s["accuracy"]["rel_l2"]
            matches = [l for l in l2 if es <= l["accuracy"]["rel_l2"] <= 2 * es]
            if not matches:
                rows.append([seed, f"{rtol_of(s)}, {g(es, 3)}, {s['seconds']:.0f} s",
                             "none within a factor 2", "", "", ""])
                continue
            l = min(matches, key=lambda l: l["accuracy"]["rel_l2"])
            el = l["accuracy"]["rel_l2"]
            rows.append([seed, f"{rtol_of(s)}, {g(es, 3)}, {s['seconds']:.0f} s",
                         f"{rtol_of(l)}, {g(el, 3)}, {l['seconds']:.0f} s", g(el / es, 3),
                         g(s["seconds"] / l["seconds"], 2), g(s["params"] / l["params"], 3)])
    return table(["seed", "exact reference, k 1: rtol, error, time",
                  "L2, k 3, audit off: rtol, error, time", "L2 error / its error",
                  "time / L2 time", "parameters / L2 parameters"], rows)


def acceptance_memory():
    def rss(path):
        found = re.search(r"Maximum resident set size \(kbytes\): (\d+)", path.read_text())
        return f"{int(found.group(1)) / 1024 ** 2:.2f}"
    rows = []
    for seed in (42, 2026, 7):
        rows.append([seed,
                     rss(RAW / "nblock-T0.01-R12-acceptance" / f"s{seed}-l2-noaudit.time"),
                     rss(RAW / "nblock-T0.01-R12-reference" / f"s{seed}-sampledmax-k1.time"),
                     rss(RAW / "nblock-T0.01-R12-reference" / f"s{seed}-sampledmax-k3.time"),
                     rss(RAW / "nblock-T0.01-R12-given" / f"s{seed}-given-k1.time"),
                     rss(RAW / "nblock-T0.01-R12-given" / f"s{seed}-given-k3.time")])
    return table(["seed", "L2, k 3, audit off", "sampled-max pinned, k 1",
                  "sampled-max pinned, k 3", "sampled-max exact, k 1", "sampled-max exact, k 3"],
                 rows)


GENERATORS = {
    "exhaustive": exhaustive,
    "layouts": layouts,
    "tolerance-factor": tolerance_factor,
    "audit": audit,
    "topology-scan": topology_scan,
    "probe-784": probe_784,
    "comb-runs": lambda: runs_table("comb-T0.1-R12"),
    "comb-matched": lambda: matched_table("comb-T0.1-R12"),
    "comb-b0": b0_status,
    "nblock-runs": lambda: runs_table("nblock-T0.01-R12"),
    "nblock-matched": lambda: matched_table("nblock-T0.01-R12"),
    "nblock-processes": lambda: processes("nblock-T0.01-R12"),
    "nblock-sweeps": nblock_sweeps,
    "nblock-matched-sweeps": nblock_matched_sweeps,
    "nblock-trace": nblock_trace,
    "acceptance-runs": acceptance_runs,
    "history": history,
    "reference-pinned": reference_pinned,
    "reference-spread": reference_spread,
    "given-matched": given_matched,
    "acceptance-memory": acceptance_memory,
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
