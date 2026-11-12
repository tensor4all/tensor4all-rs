#!/usr/bin/env python3
"""Collect merge-refine intended-regime cells into one CSV (issue #777 item F4).

Reads the `<cell>.<mode>.jsonl` / `<cell>.<mode>.err` pairs written by one
`benchmark_merge_refine_intended_regime` process per (cell, method) pair, where
`<mode>` is `baseline-only`, `only-exact` or `only-adaptive`, and `.err` holds a
single `/usr/bin/time -f "wall_s %e cpu_pct %P maxrss_kb %M"` line. Peak RSS and
the wall clock therefore describe one method's process, not the whole cell.

    python3 benchmarks/rust/collect_merge_refine_intended_regime.py \
      --dir target/f4 --out benchmarks/results/evidence/<date>-merge-refine-intended-regime.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

FIELDS = [
    "cell",
    "mode",
    "kind",
    "method",
    "bits",
    "input_depth",
    "output_depth",
    "target_bond_dim",
    "rtol",
    "chirp_turns",
    "repeats",
    "deviation_enabled",
    "input_rank",
    "input_max_deviation_from_analytic",
    "elapsed_ms",
    "cumulative_ms",
    "delta_ms",
    "level_count",
    "max_bond_dim",
    "max_transient_bond_dim",
    "region_count",
    "term_count",
    "applied_operator_count",
    "additions",
    "projections",
    "compression_attempts",
    "compressions",
    "dropped_terms",
    "refined_regions",
    "stopped_regions",
    "logical_parameters",
    "error_bound",
    "deviation_from_reference",
    "l2_deviation_from_reference",
    "reference_l2",
    "wall_s",
    "cpu_pct",
    "maxrss_kb",
]


def timing(err_path: Path) -> dict[str, float]:
    text = err_path.read_text(encoding="utf-8", errors="replace")
    found = {}
    match = re.search(r"wall_s (\S+) cpu_pct (\S+) maxrss_kb (\S+)", text)
    if match:
        found["wall_s"] = float(match.group(1))
        found["cpu_pct"] = float(match.group(2).rstrip("%"))
        found["maxrss_kb"] = float(match.group(3))
    return found


def rows(dir_path: Path) -> list[dict[str, object]]:
    collected = []
    for path in sorted(dir_path.glob("*.jsonl")):
        cell, mode = path.name[: -len(".jsonl")].split(".")
        per_process = timing(path.with_suffix(".err"))
        build = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            kind = record.get("kind")
            if kind == "build":
                # Every process of a cell rebuilds the same input; keep one row.
                build = {
                    "bits": record["bits"],
                    "input_depth": record["input_depth"],
                    "output_depth": record["output_depth"],
                    "target_bond_dim": record["target_bond_dim"],
                    "rtol": record["rtol"],
                    "chirp_turns": record["chirp_turns"],
                    "repeats": record["repeats"],
                    "deviation_enabled": record["deviation"],
                }
            elif kind == "input":
                collected.append(
                    {
                        **build,
                        "cell": cell,
                        "mode": mode,
                        "kind": "input",
                        "method": "input",
                        "input_rank": record["input_rank"],
                        "input_max_deviation_from_analytic": record[
                            "input_max_deviation_from_analytic"
                        ],
                        **per_process,
                    }
                )
            else:
                row = {**build, "cell": cell, "mode": mode, **per_process}
                row["kind"] = kind
                row["method"] = record["method"]
                row["output_depth"] = record["output_depth"]
                for field in (
                    "elapsed_ms",
                    "cumulative_ms",
                    "delta_ms",
                    "level_count",
                    "max_bond_dim",
                    "max_transient_bond_dim",
                    "region_count",
                    "term_count",
                    "applied_operator_count",
                    "additions",
                    "projections",
                    "compression_attempts",
                    "compressions",
                    "dropped_terms",
                    "refined_regions",
                    "stopped_regions",
                    "logical_parameters",
                    "error_bound",
                    "deviation_from_reference",
                    "l2_deviation_from_reference",
                    "reference_l2",
                ):
                    row[field] = record.get(field)
                collected.append(row)
    return collected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    arguments = parser.parse_args()
    collected = rows(arguments.dir)
    if not collected:
        raise SystemExit(f"no cells under {arguments.dir}")
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    with arguments.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in collected:
            writer.writerow({field: row.get(field, "") for field in FIELDS})
    print(f"wrote {len(collected)} rows to {arguments.out}")


if __name__ == "__main__":
    main()
