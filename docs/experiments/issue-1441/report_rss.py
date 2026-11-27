#!/usr/bin/env python3
"""Regenerate peak-RSS statistics from all cells of the accepted CLEAN attempt."""
import argparse
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent / "rss"


def summarize(root):
    attempts = json.loads((root / "raw/attempts.json").read_text())
    accepted = next(a for a in attempts if a["clean"] and a["success"] and a["same_observations"])
    protocol = json.loads((root / "protocol.json").read_text())
    samples = accepted["samples"]
    assert len(samples) == 2 * protocol["pairs"]
    assert {(s["pair"], s["arm"]) for s in samples} == {
        (pair, arm) for pair in range(protocol["pairs"]) for arm in ["baseline", "candidate"]}
    observations, by_arm = [], {"baseline": [], "candidate": []}
    for sample in samples:
        directory = root / "raw" / f"attempt-{accepted['attempt']}" / f"{sample['pair']}-{sample['arm']}"
        guard = json.loads((directory / "ai-run.json").read_text())
        assert sample["exit"] == guard["exit"] == 0
        assert sample["verdict"] == guard["verdict"] == "CLEAN"
        observed = json.loads((directory / "stdout.txt").read_text())
        assert observed == sample["observations"]
        observations.append(observed)
        peak = next(int(line.rsplit(":", 1)[1]) for line in (directory / "time.txt").read_text().splitlines()
                    if "Maximum resident set size" in line)
        assert peak == sample["peak_rss_kib"]
        by_arm[sample["arm"]].append(peak / 1024)
    assert all(o == observations[0] for o in observations)
    assert observations[0]["points"] == 30877
    assert observations[0]["checksum"] == 15356029994682581386
    assert all(len(values) == protocol["pairs"] for values in by_arm.values())
    return {"all_clean": True, "processes": len(samples), "point_streams_identical": True,
            "peak_rss": {arm: {"median_mib": statistics.median(values),
                               "range_mib": [min(values), max(values)]}
                         for arm, values in by_arm.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE)
    args = parser.parse_args()
    result = summarize(args.results)
    (args.results / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Generated RSS summary from all {result['processes']} CLEAN processes")


if __name__ == "__main__":
    main()
