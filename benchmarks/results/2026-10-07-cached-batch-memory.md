# Cached TreeTN batch memory results

All final gates passed: affinity valid, steal fraction 0.0, analytic output checks passed for every run. Code commit `09b2c2c8`; the code is unchanged by evidence/formatting follow-ups. Exact binary checksum and host observations are in the [raw record](2026-10-07-cached-batch-memory.json).

Release/default faer, CPU 2, all provider/Rayon thread counts 1, AMD Ryzen 9 6900HX/WSL2. Seven complete pairs per case, fresh processes. The [protocol](2026-10-07-cached-batch-memory-protocol.md) defines both fixtures and statistics. All modes include the #780 fix; these are policy comparisons, not revision-wide speedups.

| Path | Rank | Points | Legacy RSS MiB | Bounded RSS MiB | Time ratio (95% CI) | RSS ratio (95% CI) |
|---|---:|---:|---:|---:|---|---|
| raw | 8 | 256 | 8.43 | 8.43 | 1.028 (0.999–1.061) | 0.993 (0.981–1.000) |
| raw | 8 | 1024 | 9.85 | 9.84 | 1.044 (1.037–1.069) | 0.996 (0.979–1.010) |
| raw | 8 | 4096 | 15.50 | 15.52 | 1.028 (1.024–1.044) | 1.004 (0.995–1.011) |
| raw | 17 | 256 | 8.65 | 8.58 | 1.021 (0.993–1.051) | 0.996 (0.987–1.008) |
| raw | 17 | 1024 | 10.09 | 10.01 | 1.058 (1.048–1.065) | 0.992 (0.975–0.998) |
| raw | 17 | 4096 | 15.71 | 15.75 | 1.019 (1.012–1.027) | 1.001 (0.999–1.006) |
| raw | 33 | 256 | 9.31 | 9.27 | 1.010 (0.988–1.032) | 0.997 (0.980–1.000) |
| raw | 33 | 1024 | 10.54 | 10.70 | 1.029 (1.004–1.040) | 1.014 (0.999–1.022) |
| raw | 33 | 4096 | 16.34 | 16.37 | 1.018 (1.007–1.020) | 1.002 (0.989–1.008) |
| generic | 8 | 256 | 16.32 | 16.30 | 0.977 (0.913–0.998) | 0.999 (0.984–1.005) |
| generic | 8 | 1024 | 27.95 | 18.10 | 0.860 (0.801–0.906) | 0.655 (0.641–0.658) |
| generic | 8 | 4096 | 76.40 | 21.77 | 0.650 (0.645–0.706) | 0.285 (0.284–0.293) |
| generic | 17 | 256 | 41.94 | 41.95 | 0.997 (0.897–1.021) | 1.003 (0.999–1.009) |
| generic | 17 | 1024 | 128.25 | 52.86 | 0.653 (0.597–0.693) | 0.412 (0.410–0.413) |
| generic | 17 | 4096 | 444.28 | 53.22 | 0.420 (0.348–0.579) | 0.120 (0.118–0.130) |
| generic | 33 | 256 | 221.55 | 54.43 | 0.532 (0.408–0.629) | 0.246 (0.244–0.247) |
| generic | 33 | 1024 | 840.71 | 55.28 | 0.299 (0.253–0.541) | 0.066 (0.065–0.066) |
| generic | 33 | 4096 | 3036.55 | 55.61 | 0.333 (0.318–0.590) | 0.018 (0.018–0.019) |

Generic rank33/4096 RSS falls from approximately 3 GiB to 56 MiB, with about threefold throughput improvement. Raw streaming paths retain similar memory; their worst measured median slowdown is within the predeclared 10% gate. No general workspace speedup is claimed.

The original [raw-center experiment](2026-10-07-cached-batch-memory-raw-first.json) failed the memory gate and fixed-16 throughput gate; its startup affinity observation also makes it inconclusive. The [exploratory record](2026-10-07-cached-batch-memory-tuning.json) adds the generic path and records every fixed-cap case; it also has the startup-affinity probe defect and is not confirmatory evidence. The final experiment corrects that observation and repeats all final cases.

The largest case is 4096 points; the downstream 20k/rank33 case was not repeated because its reported 17 GB footprint approaches this host’s RAM. Input/output vectors still scale with point count, and finite payload budgets exclude key metadata and spare capacity.
