#!/usr/bin/env python3
"""Compare every reported acceptance cell, failing closed on missing cells."""

import argparse
import json
from pathlib import Path
import re

RECORD = re.compile(
    r"issue1441 (?P<case>.+?) seed=(?P<seed>\d+) rule=(?P<rule>\w+) "
    r"points=(?P<points>\d+) iterations=(?P<iterations>\d+) "
    r"ranks=(?P<ranks>\[[^\]]*\]) termination=(?P<termination>\w+) error=(?P<error>\S+)"
)


def read(path):
    records = {}
    for match in RECORD.finditer(Path(path).read_text()):
        record = match.groupdict()
        key = (record.pop("case"), int(record.pop("seed")), record.pop("rule"))
        assert key not in records, f"duplicate cell: {key}"
        for field in ("points", "iterations"):
            record[field] = int(record[field])
        record["ranks"] = json.loads(record["ranks"])
        record["error"] = float(record["error"])
        records[key] = record
    assert len(records) == 70, f"expected 70 cells, found {len(records)}"
    expected = {
        (case, seed, rule)
        for case, seeds in [
            ("box r=6", range(1, 12)),
            ("box r=7", range(1, 12)),
            ("plane=16 binary=false", range(1, 12)),
            ("comb delta=1", [11]),
            ("comb delta=4", [11]),
        ]
        for seed in seeds
        for rule in ["Sampled", "PerBond"]
    }
    assert records.keys() == expected, "wrong acceptance cases"
    assert all(r["termination"] == "Converged" for r in records.values())
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    baseline, candidate = read(args.baseline), read(args.candidate)
    cells = [
        dict(case=key[0], seed=key[1], rule=key[2], baseline=baseline[key],
             candidate=candidate[key], identical=baseline[key] == candidate[key])
        for key in sorted(baseline)
    ]
    result = dict(cells=cells, all_identical=all(cell["identical"] for cell in cells))
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(f"{len(cells)} cells; all costs, trajectories and errors identical: {result['all_identical']}")
    return 0 if result["all_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
