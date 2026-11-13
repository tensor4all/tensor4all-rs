"""Searched root reference (#1274) against the given exact reference and the
old pinned reference, on the M6c tt_nblock workload (T = 0.01, R = 12, cap 64).

Reads this directory's s<seed>-searched-k<k>.jsonl, the given-reference runs
in ../nblock-T0.01-R12-given and the pinned-reference runs in
../nblock-T0.01-R12-reference. Prints one row per seed, k and rtol.
"""
import json
import re
import sys
from pathlib import Path

EXACT = 31.83098861215053

HERE = Path(__file__).resolve().parent
GIVEN = HERE.parent / "nblock-T0.01-R12-given"
PINNED = HERE.parent / "nblock-T0.01-R12-reference"


def results(path):
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if not line.startswith("{"):
            continue
        record = json.loads(line)
        if record.get("kind") != "result":
            continue
        rtol = next(
            part.split("=")[1] for part in record["variant"].split(",") if part.startswith("rtol=")
        )
        out[rtol] = record
    return out


def cell(record):
    if record is None:
        return "—"
    norm = record["engine"]["norm"]
    return (
        f"ref {norm['max_reference']:.4g} ({norm['source']}), L2 {record['accuracy']['rel_l2']:.3g}, "
        f"{record['patches']} patches, {record['params']} params, {record['seconds']:.0f} s"
    )


def peak_rss_gib(path):
    if not path.exists():
        return None
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", path.read_text())
    return int(m.group(1)) / 2**20 if m else None


def search_table():
    print("seed k | searched reference | rel. to exact 31.83 | starts | sweeps | evaluations | cache hits | search s | process peak RSS GiB")
    for seed in (42, 2026, 7):
        for k in (1, 3):
            runs = results(HERE / f"s{seed}-searched-k{k}.jsonl")
            first = runs.get("1e-5")
            if first is None:
                continue
            norm = first["engine"]["norm"]
            search = norm.get("search") or {}
            rss = peak_rss_gib(HERE / f"s{seed}-searched-k{k}.time")
            print(
                f"{seed} {k} | {norm['max_reference']!r} | {norm['max_reference'] / EXACT - 1:+.2e} | "
                f"{len(search.get('starts', []))} | {search.get('sweeps')} | {search.get('evaluations')} | "
                f"{search.get('cache_hits')} | {search.get('seconds', float('nan')):.3f} | "
                f"{rss if rss is None else round(rss, 2)}"
            )


def determinism(preview_dir):
    preview_dir = Path(preview_dir)
    same = True
    for seed in (42, 2026, 7):
        for k in (1, 3):
            now = results(HERE / f"s{seed}-searched-k{k}.jsonl")
            before = results(preview_dir / f"s{seed}-searched-k{k}.jsonl")
            for rtol, record in now.items():
                if rtol not in before:
                    continue
                a, b = record, before[rtol]
                key = lambda r: (r["engine"]["norm"]["max_reference"], r["accuracy"]["rel_l2"], r["patches"], r["params"])
                if key(a) != key(b):
                    same = False
                    print("DIFFERS", seed, k, rtol, key(a), key(b))
    print("preview and rerun identical in reference, L2, patches and parameters:", same)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "search":
        return search_table()
    if len(sys.argv) > 2 and sys.argv[1] == "determinism":
        return determinism(sys.argv[2])
    print("seed k rtol | searched | given (exact 31.83) | pinned")
    for seed in (42, 2026, 7):
        for k in (1, 3):
            searched = results(HERE / f"s{seed}-searched-k{k}.jsonl")
            given = results(GIVEN / f"s{seed}-given-k{k}.jsonl")
            pinned = results(PINNED / f"s{seed}-sampledmax-k{k}.jsonl")
            for rtol in ("1e-5", "1e-6", "1e-7"):
                print(f"{seed} {k} {rtol} | {cell(searched.get(rtol))} | {cell(given.get(rtol))} | {cell(pinned.get(rtol))}")


if __name__ == "__main__":
    main()
