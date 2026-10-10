#!/usr/bin/env python3
"""Convert the #784 DMRG data fixture to followup_quality chain JSON.

Requires h5py and NumPy. Reverses file site order as in the original
reproduction; writes two identical real inputs in column-major order.
Only reads fixture data; it runs no compression/interpolation algorithm.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with h5py.File(args.input, "r") as source:
        count = int(source["num_sites"][()])
        cores = [np.asarray(source[f"tensor_{i}"][:]) for i in range(1, count + 1)]
    if count != 50 or any(np.iscomplexobj(c) or not np.isfinite(c).all() for c in cores):
        parser.error("expected the real, finite n50 DMRG fixture")
    cores.reverse()
    cores[0] = cores[0].reshape(1, *cores[0].shape)
    cores[-1] = cores[-1].reshape(*cores[-1].shape, 1)
    if any(c.ndim != 3 or c.shape[1] != 3 for c in cores):
        parser.error("expected physical dimension three at every site")
    records = [{"shape": list(c.shape), "values": c.flatten(order="F").tolist()} for c in cores]
    args.output.write_text(json.dumps({"dims": [c.shape[1] for c in cores], "inputs": [records, records]}))


if __name__ == "__main__":
    main()
