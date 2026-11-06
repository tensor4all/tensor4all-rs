#!/usr/bin/env python3
"""Recompute accuracy from raw arrays; reject incomplete or contradictory records."""

import argparse
import collections
import json
import math
import pathlib
import statistics

import numpy as np

import compare as c
import supplement as s


ACCURACY_GATE = 1e-8
SPECTRAL_RANK_RTOL = 1e-12


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def normalized_edge(a, b):
    return tuple(sorted((int(a), int(b))))


def structural_rank_bounds(manifest):
    """Return the product-rank upper bound for every tree edge."""
    n = int(manifest["n"])
    edge_list = [normalized_edge(a, b) for a, b, _ in manifest["edges"]]
    expected_edges = set(edge_list)
    if (
        len(expected_edges) != n - 1
        or any(a == b or a < 0 or b >= n for a, b in expected_edges)
    ):
        raise RuntimeError("fixture bond graph is not a simple n-node tree")
    adjacency = [[] for _ in range(n)]
    for a, b in expected_edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    reached = set()
    todo = [0] if n else []
    while todo:
        node = todo.pop()
        if node in reached:
            continue
        reached.add(node)
        todo.extend(adjacency[node])
    if len(reached) != n:
        raise RuntimeError("fixture bond graph is disconnected")

    operand_maps = []
    for operand in manifest["operand_edges"]:
        ranks = {}
        for a, b, rank in operand:
            edge = normalized_edge(a, b)
            if edge in ranks or rank < 1:
                raise RuntimeError("invalid or duplicate input bond in fixture manifest")
            ranks[edge] = int(rank)
        if set(ranks) != expected_edges:
            raise RuntimeError("operand bond edges differ from the fixture tree")
        operand_maps.append(ranks)
    if len(operand_maps) != 2:
        raise RuntimeError("expected two operand bond manifests")
    return {
        edge: operand_maps[0][edge] * operand_maps[1][edge]
        for edge in expected_edges
    }


def cut_rank_diagnostics(expected, manifest, edge, observed_rank):
    """Compute a cut-SVD rank and error lower bound from the dense oracle."""
    n = int(manifest["n"])
    physical_dim = round(expected.size ** (1.0 / n))
    if physical_dim < 1 or physical_dim**n != expected.size:
        raise RuntimeError("dense oracle size is not a uniform physical-dimension power")

    adjacency = [[] for _ in range(n)]
    for a, b, _ in manifest["edges"]:
        adjacency[a].append(b)
        adjacency[b].append(a)
    start, blocked = edge
    sites = set()
    todo = [start]
    while todo:
        node = todo.pop()
        if node in sites:
            continue
        sites.add(node)
        todo.extend(neighbor for neighbor in adjacency[node] if neighbor != blocked)
    if blocked in sites or not sites or len(sites) == n:
        raise RuntimeError("reported RSI edge does not split the fixture tree")
    left = sorted(sites)
    right = sorted(set(range(n)) - sites)
    tensor = expected.reshape([physical_dim] * n, order="F")
    matrix = np.transpose(tensor, left + right).reshape(
        physical_dim ** len(left), -1, order="F"
    )
    singular = np.linalg.svd(matrix, compute_uv=False)
    norm = float(np.linalg.norm(singular))
    if not len(singular) or norm == 0.0 or not math.isfinite(norm):
        raise RuntimeError("invalid singular spectrum for reported RSI edge")
    numerical_rank = int(np.count_nonzero(singular > singular[0] * SPECTRAL_RANK_RTOL))
    lower_bound = float(np.linalg.norm(singular[observed_rank:]) / norm)
    return numerical_rank, lower_bound


def diagnose_rsi_failure(records, expected_arrays, manifests, caps):
    """Find the largest observed RSI failure with a structurally sufficient cap.

    The selected edge is only a descriptive local diagnostic. Its reported
    pivot is not a residual certificate and does not establish causality.
    """
    candidates = []
    for record in records:
        if record["algorithm"] != "rsi" or record["status"] != "completed":
            continue
        error = float(record["relative_l2"])
        if not math.isfinite(error) or error < 0.0:
            raise RuntimeError("invalid saved RSI relative error")
        if error <= ACCURACY_GATE:
            continue
        case = record["case"]
        cap = caps[case]
        bounds = structural_rank_bounds(manifests[case])
        rank_bound = max(bounds.values(), default=1)
        if rank_bound > cap:
            continue

        edge_reports = record["diagnostics"].get("edges", [])
        output_ranks = {
            normalized_edge(a, b): int(rank)
            for a, b, rank in record["output_ranks"]
        }
        if set(output_ranks) != set(bounds):
            raise RuntimeError("RSI output edges differ from fixture edges")
        reports = {}
        for report in edge_reports:
            edge = normalized_edge(report["child"], report["parent"])
            if edge in reports or edge not in bounds:
                raise RuntimeError("duplicate or unexpected RSI diagnostic edge")
            if int(report["rank"]) != output_ranks[edge]:
                raise RuntimeError("RSI edge diagnostic rank differs from returned tree")
            pivot = float(report["relative_pivot"])
            if not math.isfinite(pivot) or pivot < 0.0:
                raise RuntimeError("invalid RSI local pivot diagnostic")
            reports[edge] = report
        if set(reports) != set(bounds):
            raise RuntimeError("RSI diagnostic edge set differs from fixture tree")

        compressed = [
            (float(report["relative_pivot"]), edge, report)
            for edge, report in reports.items()
            if not report["exact_columns"]
            and int(report["rows"]) > int(report["rank"])
            and int(report["rank"]) < cap
        ]
        focus = min(compressed, key=lambda item: (item[0], item[1])) if compressed else None
        if focus is None:
            focus_edge = None
            focus_rank = None
            focus_pivot = None
            cut_rank = None
            cut_lower_bound = None
        else:
            focus_pivot, focus_edge, report = focus
            focus_rank = int(report["rank"])
            cut_rank, cut_lower_bound = cut_rank_diagnostics(
                expected_arrays[case], manifests[case], focus_edge, focus_rank
            )
            if error + 1e-12 < cut_lower_bound:
                raise RuntimeError("observed RSI error contradicts the cut-SVD lower bound")

        candidates.append(
            dict(
                case=case,
                seed=record["seed"],
                block=record["block"],
                configured_cap=cap,
                structural_rank_upper_bound=rank_bound,
                relative_l2=error,
                accuracy_gate=ACCURACY_GATE,
                focus_edge=list(focus_edge) if focus_edge is not None else None,
                observed_edge_rank=focus_rank,
                edge_relative_pivot=focus_pivot,
                reference_cut_rank_1e12=cut_rank,
                observed_rank_cut_relative_l2_lower_bound=cut_lower_bound,
            )
        )

    if not candidates:
        return dict(
            status="no_observed_cap_sufficient_rsi_accuracy_failure",
            accuracy_gate=ACCURACY_GATE,
            explanation=(
                "No completed RSI observation failed the accuracy gate while the "
                "input-bond product bounds fit within the configured cap."
            ),
        )
    worst = max(candidates, key=lambda item: (item["relative_l2"], item["case"], item["seed"], item["block"]))
    return dict(
        status="observed_cap_sufficient_rsi_accuracy_failure",
        selection="largest completed full-grid relative L2 error among structurally cap-sufficient failures",
        local_edge_note=(
            "The focus edge is the non-exact compressed edge with the smallest "
            "reported relative pivot. This local pivot is not an error estimate "
            "and the association does not establish causality."
        ),
        **worst,
    )


def optional_median(values):
    return statistics.median(values) if values else None


def optional_min(values):
    return min(values) if values else None


def optional_max(values):
    return max(values) if values else None


def summarize_measurements(group, seeds, blocks):
    """Summarize successful measurements while retaining failed attempts."""
    complete = [record for record in group if record["status"] == "completed"]
    times = [float(record["seconds"]) for record in complete]
    errors = [float(record["relative_l2"]) for record in complete]
    cvs = []
    for seed in seeds:
        repeats = [
            float(record["seconds"])
            for record in complete
            if record["seed"] == seed
        ]
        cvs.append(
            statistics.stdev(repeats) / statistics.mean(repeats)
            if len(repeats) == blocks and len(repeats) > 1
            else None
        )
    summary = dict(
        attempts=len(group),
        completed=len(complete),
        failed=len(group) - len(complete),
        accuracy_pass=sum(error <= ACCURACY_GATE for error in errors),
        seconds_median=optional_median(times),
        seconds_min=optional_min(times),
        seconds_max=optional_max(times),
        relative_l2_min=optional_min(errors),
        relative_l2_max=optional_max(errors),
        repeat_cv_by_seed=cvs,
        timing_stable=all(value is not None and value <= 0.10 for value in cvs),
        actual_max_bond_ranks=sorted(
            {
                max(value[2] for value in record["output_ranks"])
                for record in complete
            }
        ),
    )
    return complete, summary


def fmt(value, digits=2, scale=1.0):
    return "n/a" if value is None else f"{value * scale:.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("initial", type=pathlib.Path)
    args = parser.parse_args()
    initial = args.initial.resolve()
    directory = initial / "supplement"
    records = load(directory / "observations.jsonl")
    traces = load(initial / "trace" / "observations.jsonl")

    protocols = [
        json.loads((path / "protocol.json").read_text())
        for path in (initial, directory)
    ]
    seeds = protocols[0]["seeds"]
    blocks = int(protocols[1]["blocks"])
    if seeds != protocols[1]["seeds"] or blocks < 1 or not seeds:
        raise RuntimeError("benchmark and supplement repetition protocols disagree")

    fixtures = {}
    caps = {}
    for base, cases in [(initial, c.CASES), (directory, s.SUPPLEMENT)]:
        for topology, n, chi, cap in cases:
            path = base / s.name(topology, n, chi)
            case = f"{path.name}-cap{cap}"
            fixtures[case] = path
            caps[case] = int(cap)

    expected_keys = {
        (case, algorithm, seed, block)
        for case in fixtures
        for algorithm in ("treeaci", "rsi")
        for seed in seeds
        for block in range(blocks)
    }
    keys = [(r["case"], r["algorithm"], r["seed"], r["block"]) for r in records]
    if len(set(keys)) != len(keys) or set(keys) != expected_keys:
        raise RuntimeError("missing/duplicate/unexpected observations")
    allowed_statuses = {"completed", "error", "timeout"}
    if any(r.get("status") not in allowed_statuses for r in records):
        raise RuntimeError("observation has an unknown status")

    trace_keys = [(r["case"], r["seed"]) for r in traces]
    expected_trace_keys = {(case, seed) for case in fixtures for seed in seeds}
    if len(set(trace_keys)) != len(trace_keys) or set(trace_keys) != expected_trace_keys:
        raise RuntimeError("incomplete diagnostic replay")
    if not all(r["matches_untouched_main"] for r in traces):
        raise RuntimeError("diagnostic copy changed output")
    for path in (initial, directory):
        if not json.loads((path / "completion.json").read_text())["source_unchanged"]:
            raise RuntimeError("candidate changed during benchmark")
        for algorithm, metadata in json.loads((path / "binaries.json").read_text()).items():
            if c.sha(path / algorithm / "worker") != metadata["sha256"]:
                raise RuntimeError("binary hash mismatch")

    expected_arrays = {}
    input_arrays = {}
    manifests = {}
    for case, path in fixtures.items():
        manifest = json.loads((path / "fixture.json").read_text())
        manifests[case] = manifest
        if c.sha(path / "inputs.bin") != manifest["data_sha256"] or c.sha(
            path / "expected.bin"
        ) != manifest["expected_sha256"]:
            raise RuntimeError("fixture hash mismatch")
        expected_arrays[case] = np.fromfile(path / "expected.bin", dtype="<f8")
        input_arrays[case] = [
            np.fromfile(path / f"input{i}.bin", dtype="<f8") for i in range(2)
        ]
        if not np.array_equal(
            expected_arrays[case], input_arrays[case][0] * input_arrays[case][1]
        ):
            raise RuntimeError("reference is not exact dense Hadamard product")

    for record in records:
        if record["status"] != "completed":
            continue
        stem = f"{record['case']}-{record['algorithm']}-seed{record['seed']}"
        folder = directory / f"block{record['block']}"
        output = folder / f"{stem}.bin"
        if c.sha(output) != record["output_sha256"]:
            raise RuntimeError("output hash mismatch")
        actual = np.fromfile(output, dtype="<f8")
        exact = expected_arrays[record["case"]]
        error = c.error(actual, exact)
        if error is None or not math.isclose(
            error, record["relative_l2"], rel_tol=1e-13, abs_tol=1e-25
        ):
            raise RuntimeError("saved accuracy differs from raw arrays")
        if (error <= ACCURACY_GATE) != record["accuracy_pass"]:
            raise RuntimeError("incorrect saved acceptance")
        if not math.isfinite(float(record["seconds"])) or record["seconds"] <= 0:
            raise RuntimeError("invalid completed-call timing")
        for i in range(2):
            input_error = c.error(
                np.fromfile(folder / f"{stem}.input{i}.bin", dtype="<f8"),
                input_arrays[record["case"]][i],
            )
            if input_error is None or input_error > 1e-12:
                raise RuntimeError("native input differs from independent oracle")
        cap = caps[record["case"]]
        ranks = record["output_ranks"]
        if len(ranks) != manifests[record["case"]]["n"] - 1 or not all(
            0 < rank <= cap for _, _, rank in ranks
        ):
            raise RuntimeError("invalid actual output ranks")
        lower = manifests[record["case"]]["relative_l2_lower_bounds"][str(cap)]
        if error + 1e-12 < lower:
            raise RuntimeError("observed error violates cut SVD lower bound")

    groups = collections.defaultdict(list)
    for record in records:
        groups[record["case"], record["algorithm"]].append(record)
    rows = []
    for case in fixtures:
        manifest = manifests[case]
        entry = dict(
            case=case,
            cut_lower_bound=manifest["relative_l2_lower_bounds"][str(caps[case])],
            input_numerical_cut_ranks=[
                item["numerical_rank_1e12"] for item in manifest["input_cut_spectra"]
            ],
            product_numerical_cut_rank=manifest["product_cut_rank_1e12"],
        )
        for algorithm in ("treeaci", "rsi"):
            group = groups[case, algorithm]
            complete, d = summarize_measurements(group, seeds, blocks)
            if algorithm == "treeaci":
                matching = [record for record in traces if record["case"] == case]
                shapes = [
                    (a, b)
                    for record in matching
                    for _, a, b in record["local_matrix_shapes"]
                ]
                d.update(
                    termination=sorted(
                        {record["diagnostics"]["termination"] for record in complete}
                    ),
                    intermediate_peak=max(
                        (record["peak_active_bond_rank"] for record in matching),
                        default=None,
                    ),
                    max_local_matrix_elements=max(
                        (a * b for a, b in shapes), default=None
                    ),
                    max_local_matrix_shape=max(
                        shapes, key=lambda shape: shape[0] * shape[1], default=None
                    ),
                )
            else:
                edges = [
                    edge
                    for record in complete
                    for edge in record["diagnostics"]["edges"]
                ]
                shapes = [(edge["rows"], edge["columns"]) for edge in edges]
                d.update(
                    intermediate_peak=max(
                        (edge["rank"] for edge in edges), default=None
                    ),
                    max_local_matrix_elements=max(
                        (rows * columns for rows, columns in shapes), default=None
                    ),
                    max_local_matrix_shape=max(
                        shapes, key=lambda shape: shape[0] * shape[1], default=None
                    ),
                    nontrivial_sketch_edges=sorted(
                        {
                            (
                                edge["child"],
                                edge["parent"],
                                edge["rows"],
                                edge["columns"],
                                edge["rank"],
                            )
                            for edge in edges
                            if not edge["exact_columns"] and edge["rows"] > edge["rank"]
                        }
                    ),
                )
                d["nontrivial_sketch_edges"] = [
                    dict(child=child, parent=parent, rows=rows, columns=columns, rank=rank)
                    for child, parent, rows, columns, rank in d["nontrivial_sketch_edges"]
                ]
            entry[algorithm] = d
        rows.append(entry)

    failure = diagnose_rsi_failure(records, expected_arrays, manifests, caps)
    report = dict(
        baseline=protocols[0]["baseline"],
        candidate_head=protocols[0]["candidate_head"],
        candidate_source_sha256=protocols[0]["candidate_source_sha256"],
        total_timed_comparisons=len(records),
        completed_timed_comparisons=sum(record["status"] == "completed" for record in records),
        diagnostic_replays=len(traces),
        raw_arrays_revalidated=True,
        rows=rows,
        failure_cut=failure,
    )
    c.save(initial / "summary.json", report)

    total_completed = report["completed_timed_comparisons"]
    lines = [
        "# High-rank main TreeACI / worktree RSI measurements",
        "",
        f"Baseline: `{report['baseline']}`. Candidate source SHA-256: `{report['candidate_source_sha256']}`.",
        "",
        f"{len(records)} calls attempted; {total_completed} completed and revalidated; "
        f"{len(traces)} separate diagnostic replays. See protocols and raw observations beside this report.",
        "",
        f"Times are medians of completed calls (`completed/attempts` is shown per algorithm), in milliseconds. "
        f"Errors are worst full-grid relative L2 over completed calls. Output ranks are actual, not configured caps. "
        f"ACI trace peaks are active intermediate bond ranks; RSI constructs each edge once.",
        "",
        "| Case | cap lower bound | ACI ms | RSI ms | ACI error | RSI error | ACI / RSI pass | ACI / RSI final max rank | ACI / RSI peak rank | ACI stop | stable ACI / RSI |",
        "|---|---:|---:|---:|---:|---:|---|---|---|---|---|",
    ]
    for row in rows:
        a, b = row["treeaci"], row["rsi"]
        lines.append(
            f"| {row['case']} | {row['cut_lower_bound']:.3e} | "
            f"{fmt(a['seconds_median'], scale=1000)} | {fmt(b['seconds_median'], scale=1000)} | "
            f"{fmt(a['relative_l2_max'], digits=3)} | {fmt(b['relative_l2_max'], digits=3)} | "
            f"{a['accuracy_pass']}/{a['completed']} completed of {a['attempts']}, "
            f"{b['accuracy_pass']}/{b['completed']} completed of {b['attempts']} | "
            f"{a['actual_max_bond_ranks']}, {b['actual_max_bond_ranks']} | "
            f"{a['intermediate_peak']}, {b['intermediate_peak']} | "
            f"{', '.join(a['termination']) or 'n/a'} | "
            f"{a['timing_stable']}, {b['timing_stable']} |"
        )

    insufficient = [
        f"{row['case']} ({row['cut_lower_bound']:.3e})"
        for row in rows
        if row["cut_lower_bound"] > ACCURACY_GATE
    ]
    lines += [
        "",
        "## Interpretation",
        "",
        "- A cut lower bound above the full-grid gate shows that the configured rank cap cannot meet the gate on that cut; it does not by itself establish an implementation defect.",
        "- Cases with a cut lower bound above the full-grid gate: "
        + (", ".join(insufficient) if insufficient else "none in these fixtures."),
        "- Incomplete calls remain in the attempted counts. Timing and accuracy summaries use completed calls only; empty statistics are reported as `n/a` / `null`.",
        "- Unstable timing means at least one seed lacks a complete repeat set or has repeat CV > 10%; retain every measurement and avoid confident speedup claims. Timing ranges and CV are in summary.json.",
        "- This single-host f64 synthetic experiment does not accept downstream GW results or prove a general tree-RSI error guarantee.",
        "",
        "## Post hoc RSI failure observation",
        "",
    ]
    if failure["status"] == "observed_cap_sufficient_rsi_accuracy_failure":
        edge = failure["focus_edge"]
        if edge is None:
            focus_text = "No non-exact compressed edge below the configured cap was available for a local focus diagnostic."
        else:
            focus_text = (
                f"The selected local edge is {edge}, with observed rank {failure['observed_edge_rank']}, "
                f"relative pivot {failure['edge_relative_pivot']:.3e}, cut numerical rank "
                f"{failure['reference_cut_rank_1e12']}, and cut error lower bound "
                f"{failure['observed_rank_cut_relative_l2_lower_bound']:.3e}."
            )
        lines.append(
            f"The largest completed cap-sufficient RSI failure is case `{failure['case']}`, "
            f"seed {failure['seed']}, block {failure['block']}: relative L2 "
            f"{failure['relative_l2']:.3e} at cap {failure['configured_cap']} "
            f"(input-bond product upper bound {failure['structural_rank_upper_bound']}). "
            f"{focus_text} This is an observation, not a causal diagnosis or an error certificate."
        )
    else:
        lines.append(failure["explanation"])
    lines += ["", json.dumps(failure, indent=2), ""]
    (initial / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
