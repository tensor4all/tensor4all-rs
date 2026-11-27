#!/usr/bin/env python3
"""Recompute the serial granularity report from every accepted process."""
import collections
import json
from pathlib import Path
import statistics

import run as diagnostic

HERE = Path(__file__).resolve().parent


def summarize(directory, exit_code):
    sample = diagnostic.parse(directory, exit_code)
    assert sample["guard_verdict"] == "CLEAN" and exit_code == 0
    assert not sample["unclosed_probes"] and not sample["unclosed_candidates"]
    result = sample["result"]
    assert result["merge_refine"]["error_bound"] <= result["merge_refine"]["absolute_tolerance"]
    assert result["accuracy"]["rel_l2"] <= 1e-6
    total_ns = result["seconds"] * 1e9
    widths = collections.Counter(r["level"] for r in sample["regions"])
    candidates = [(r, c) for r in sample["regions"] for c in r["candidates"]]
    assert all(c["complete"] for _, c in candidates)
    probes = [(r, c, p) for r, c in candidates for p in c["probes"]]
    parents_ns = sum(g["ns"] for _, _, p in probes for g in p["parents"])
    eligible = [(r, c, p) for r, c, p in probes
                if sum(g["additions"] > 0 for g in p["parents"]) >= 2]
    eligible_ns = sum(g["ns"] for _, _, p in eligible for g in p["parents"] if g["additions"] > 0)
    by_label = collections.defaultdict(lambda: {"calls": 0, "ns": 0})
    for _, candidate in candidates:
        by_label[candidate["label"]]["calls"] += 1
        by_label[candidate["label"]]["ns"] += candidate["ns"]
    hottest = max(eligible, key=lambda row: row[2]["ns"], default=None)
    hot = None
    if hottest:
        region, candidate, probe = hottest
        parent_ns = sum(p["ns"] for p in probe["parents"])
        hot = {"level": region["level"], "region": region["ordinal"],
               "candidate": candidate["label"], "input_items": probe["input_items"],
               "parents": len(probe["parents"]),
               "parents_with_additions": sum(p["additions"] > 0 for p in probe["parents"]),
               "probe_seconds": probe["ns"] / 1e9,
               "parent_seconds": parent_ns / 1e9,
               "largest_parent_seconds": max(p["ns"] for p in probe["parents"]) / 1e9,
               "largest_parent_fraction": max(p["ns"] for p in probe["parents"]) / parent_ns}
    regions_ns = sum(r["ns"] for r in sample["regions"])
    axis_waves = [[c for c in r["candidates"] if c["label"].startswith("axis-")]
                  for r in sample["regions"]]
    eligible_axis_ns = sum(c["ns"] for wave in axis_waves if len(wave) >= 2 for c in wave)
    assert parents_ns <= sum(c["ns"] for _, c in candidates) <= regions_ns <= total_ns
    return {"directory": str(directory.relative_to(HERE)), "frontend_seconds": result["seconds"],
            "region_widths": dict(sorted(widths.items())), "regions_seconds": regions_ns / 1e9,
            "candidate_labels": dict(by_label), "parent_seconds": parents_ns / 1e9,
            "maximum_axis_candidate_width": max(map(len, axis_waves), default=0),
            "eligible_axis_frontend_share": eligible_axis_ns / total_ns,
            "parent_frontend_share": parents_ns / total_ns,
            "eligible_parent_seconds": eligible_ns / 1e9,
            "eligible_parent_frontend_share": eligible_ns / total_ns,
            "eligible_probes": len(eligible),
            "maximum_addition_parent_width": max((sum(g["additions"] > 0 for g in p["parents"]) for _, _, p in probes), default=0),
            "hottest_eligible_probe": hot,
            "emission_seconds": sample["emission_ns"] / 1e9,
            "emission_frontend_share": sample["emission_ns"] / total_ns,
            "os_threads_at_boundaries": sorted({r["os_threads"] for r in sample["regions"] if r["os_threads"] is not None}),
            "numerical_result": result}


def main():
    protocol = json.loads((HERE / "protocol.json").read_text())
    attempts = json.loads((HERE / "raw/attempts.json").read_text())
    accepted = next(a for a in attempts if a["all_clean"] and a["success"])
    rows = [summarize(HERE / "raw" / s["directory"], s["exit"]) for s in accepted["samples"]]
    assert len(rows) == 6
    cells = []
    for case in protocol["cases"]:
        samples = [r for r, s in zip(rows, accepted["samples"]) if s["case"] == case["id"]]
        assert len(samples) == case["repeats"] == 3
        med = lambda key: statistics.median(s[key] for s in samples)
        cell = {"case": case["id"], "processes": 3,
                "frontend_seconds": med("frontend_seconds"), "regions_seconds": med("regions_seconds"),
                "parent_seconds": med("parent_seconds"), "eligible_parent_seconds": med("eligible_parent_seconds"),
                "parent_frontend_share": med("parent_frontend_share"),
                "eligible_parent_frontend_share": med("eligible_parent_frontend_share"),
                "maximum_region_width": max(max(s["region_widths"].values()) for s in samples),
                "maximum_addition_parent_width": max(s["maximum_addition_parent_width"] for s in samples),
                "maximum_emission_share": max(s["emission_frontend_share"] for s in samples),
                "parent_need_screen": med("eligible_parent_frontend_share") >= 0.2,
                "eligible_axis_frontend_share": med("eligible_axis_frontend_share"),
                "maximum_axis_candidate_width": max(s["maximum_axis_candidate_width"] for s in samples),
                "samples": samples}
        cells.append(cell)
    (HERE / "summary.json").write_text(json.dumps(cells, indent=2) + "\n")
    timing = ["| Case (G0 cap / transform goal) | Fourier seconds | Region seconds | Parent-operation share | Eligible addition-parent share | Max incoming regions | Max addition-parent groups in one probe |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    hotspots = ["| Case, fixed repetition 0 | Hottest eligible child probe | Parent groups / with additions | Parent seconds | Largest parent seconds | Largest parent's fraction |",
                "| --- | --- | ---: | ---: | ---: | ---: |"]
    labels = ["| Case, fixed repetition 0 | Candidate label | Calls | Seconds |", "| --- | --- | ---: | ---: |"]
    for cell in cells:
        name = cell["case"]
        timing.append(f"| {name} | {cell['frontend_seconds']:.3f} | {cell['regions_seconds']:.3f} | {cell['parent_frontend_share']:.1%} | {cell['eligible_parent_frontend_share']:.1%} | {cell['maximum_region_width']} | {cell['maximum_addition_parent_width']} |")
        sample = cell["samples"][0]
        hot = sample["hottest_eligible_probe"]
        if hot:
            hotspots.append(f"| {name} | level {hot['level']}, region {hot['region']}, {hot['candidate']} | {hot['parents']} / {hot['parents_with_additions']} | {hot['parent_seconds']:.3f} | {hot['largest_parent_seconds']:.3f} | {hot['largest_parent_fraction']:.1%} |")
        for label, value in sorted(sample["candidate_labels"].items()):
            labels.append(f"| {name} | {label} | {value['calls']} | {value['ns']/1e9:.3f} |")
    axis_note = "\n".join(
        f"- {c['case']}: axis-candidate width {c['maximum_axis_candidate_width']}; "
        f"same-region multi-axis candidate work {c['eligible_axis_frontend_share']:.1%} of Fourier time."
        for c in cells)
    text = """# Serial merge granularity diagnostic (#1497)

## Scope and validity

This follows the [whole-region baseline](../1497-baseline/README.md). It is a
single-revision serial diagnostic, not a parallel implementation or a speedup
comparison. Source is pinned to `5fbd73e5`, which integrates #1407/#1511 and
contains upstream `62c3b606`. [Protocol](protocol.json), [source/binary/tool
pins](raw/manifest.json), [all matrix outcomes](raw/attempts.json).

All six processes in the declared two-cell matrix complete and are CLEAN.
The accepted preflight also has three CLEAN processes: R4 tracing off/on and
the counts-only tiny fixture. Off/on signatures agree on every reported field
except the declared frontend/producer seconds. The first preflight is retained
under [raw-preflight-contaminated-1](raw-preflight-contaminated-1/manifest.json):
its tiny fixture passed numerically but was CONTAMINATED (4.1% foreign CPU,
3% limit). The entire preflight was rerun before starting the matrix, with
unchanged cases, thresholds and repetitions. Its timings do not enter tables.

The tiny two-bit Exact fixture directly records incoming widths `[1, 2]` and
asserts four final regions and dense values `[0.5, 0.5, 0.5, 0.5]`. This pins
the count's meaning; it has no input-sibling additions and establishes no
performance opportunity. One original leaf still means one leaf application.

Both real-workload-derived cells keep the original M6a generator, seed 42,
T=0.1, `rtol=1e-6` and certified share 0.5. R4 uses G0 cap 8 / transform goal
16 and full 4096-point DFT checking; R6 uses G0 cap 16 / goal 64 and the existing
4096-sample/four-slice check. IR and production R10 are outside this follow-up;
their original failures remain in the earlier baseline.

## Measurement definition

One incoming region job is one `(spectator cell, per-axis prefix)` map entry
before an executed transition. Its scope contains the complete decision for
that key. Candidate scopes cover own-region work, an axis probe including its
children, or joint/uniform children. These scopes are serial in this run.
An own candidate can return early when it meets the rank goal.

Within each child `restrict_and_merge` call, a parent group is all source
items mapping to one input parent. The probe records each original item's
restriction and any sibling addition in the original traversal order, then
aggregates their durations by parent for inspection. It never reorders the
input loop or balances a sum. Per-probe item/addition totals and nested time
accounting are checked, including agreement with the public additions counter.

"Eligible addition-parent share" sums parent groups containing additions,
only in child probes with at least two such groups, divided by that process's
Fourier timer. It implements the declared 20% need screen with actual same-probe
width. It is a screen for further design, not a scaling result or memory-admission
proof. Sequential child probes/candidates are never pooled into concurrent width.

## Results

Times/shares are medians of the three per-process values, not pooled job medians.
Widths are observed maxima across the retained processes.

""" + "\n".join(timing) + """

The following is the fixed first repetition, not the fastest sample. The
hottest eligible child probe is chosen by its own elapsed time. All repetitions
and every group duration are retained in raw stderr; [summary.json](summary.json)
contains per-process aggregates and the same hotspot selection.

""" + "\n".join(hotspots) + "\n\n" + "\n".join(labels) + "\n\n" + axis_note + """

Both cells pass the declared parent-work need screen. Prioritize a design of
input-parent jobs within one child probe: that opportunity includes the own
candidate as well as axis probes. R6 also has a narrower axis-candidate option;
its two candidates belong to one region decision, rather than two whole-region
jobs. R4 takes the own-candidate early exit without axis work. These observed
durations do not make either option safe to dispatch concurrently.

## Interpretation and limits

The observed parent-group distribution can justify refining that seam's design;
it cannot establish a lane speedup. A parent job would need to preserve its
original sibling fold, while the coordinator replays scalar contributions in
canonical source/term order. Group subtotals can change floating-point results.
Dispatch/join must fit each child probe's admission and early-exit boundary.
No worker borrows a peer engine, and aggregate memory admission remains unsolved.

Parent times exclude collector-map updates and logging; candidate/region and
frontend times include nested diagnostic overhead. Explicit emission durations
are retained, but clock/metadata overhead is not fully removed. The largest
parent fraction describes a cost distribution, not an ideal speedup prediction.
Thread counts at region boundaries and actual command affinities are recorded;
OS thread counts are not a numerical worker budget. One logical CPU was used.

The new serial source includes the reviewed policy/API integration, which is
still under review. These numbers do not re-time the old `8ce72e4e` or `d60f0751`
campaigns and do not promise production-resolution or IR performance.

## Reproduction

Create a detached checkout at the protocol's full baseline commit, then apply
[profiling.patch](profiling.patch) (which includes the diagnostic lazy reader),
or run [instrument.py](instrument.py) on that clean checkout. Numerical calls
and their order are preserved. The helper uses only the standard library; no
new production dependency or lane implementation is added.

Run `taskset -c 0-7 python3 run.py --checkout /path/to/detached-checkout
--target-dir /path/to/owned-target` from this experiment directory, in a fresh
copy without existing `raw`. Builds use `ai-run build --mem 6G`; all finished
binary executions use exclusive `ai-run measure --mem 8G`, CPU 8, with 300-second
process caps. `ai-run` is external; its entry-point hash is in the manifest.
Choose and declare appropriate CPU sets before a campaign on another host.

`python3 report.py` regenerates tables and derived per-process summaries from
the committed stdout/stderr/guard records. Those derived summaries and Cargo's
full artifact listing are local caches; the authoritative measurement records,
build logs and pins are committed. Keep the contaminated preflight too.

`python3 verify.py` checks retained controls, tool/source-patch pins and process
counts. Add `--checkout /path/to/instrumented-checkout` to compare all captured
source bytes, and `--binaries` while the original binaries remain available.
Both patch application and instrumentation generation were independently
reproduced at the pinned baseline and matched all eight captured source hashes.
"""
    (HERE / "README.md").write_text(text)
    print("Generated two-cell diagnostic from all six CLEAN processes")


if __name__ == "__main__":
    main()
