#!/usr/bin/env python3
"""Verify retained controls and pins, optionally against an instrumented checkout."""
import argparse
import json
from pathlib import Path

import run as diagnostic

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path)
    parser.add_argument("--binaries", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((HERE / "raw/manifest.json").read_text())
    for field, filename in [("protocol", "protocol.json"), ("controller", "run.py"),
                            ("instrument", "instrument.py"), ("profiler", "profiler.rs"),
                            ("patch", "profiling.patch")]:
        assert diagnostic.sha(HERE / filename) == manifest[field + "_sha256"], filename
    if args.checkout:
        for filename, digest in manifest["source_sha256"].items():
            assert diagnostic.sha(args.checkout / filename) == digest, filename
    if args.binaries:
        for binary in manifest["binaries"].values():
            assert diagnostic.sha(Path(binary["path"])) == binary["sha256"], binary["path"]
    for name in ["raw", "raw-preflight-contaminated-1"]:
        root = HERE / name
        archived = json.loads((root / "manifest.json").read_text())
        assert {k: v for k, v in archived.items() if k != "at_utc"} == {
            k: v for k, v in manifest.items() if k != "at_utc"}
        controls = [diagnostic.parse(root / f"validation-{mode}", 0) for mode in [0, 1]]
        assert all(c["guard_verdict"] == "CLEAN" and c["result"] for c in controls)
        signatures = [{k: v for k, v in c["result"].items() if k not in {"seconds", "g0_seconds"}}
                      for c in controls]
        assert signatures[0] == signatures[1]
        tiny = diagnostic.parse(root / "tiny", 0)
        assert [sum(r["level"] == level for r in tiny["regions"]) for level in [1, 2]] == [1, 2]
        assert "1 passed; 0 failed" in (root / "tiny/stdout.txt").read_text()
        assert tiny["guard_verdict"] == ("CLEAN" if name == "raw" else "CONTAMINATED")
    records = list(HERE.glob("raw/**/ai-run.json")) + list(HERE.glob("raw-preflight-contaminated-1/**/ai-run.json"))
    verdicts = [json.loads(p.read_text())["verdict"] for p in records]
    assert len(verdicts) == 12 and verdicts.count("CLEAN") == 11 and verdicts.count("CONTAMINATED") == 1
    print("Pins verified; both off/on controls agree; tiny widths [1,2]; 12 retained processes: 11 CLEAN, 1 CONTAMINATED")


if __name__ == "__main__":
    main()
