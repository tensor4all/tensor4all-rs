# Opt-in confirmation results

All 84 checked cells and 24 timed cells completed. All eight measurement guards are CLEAN, with zero exits and identical deterministic outputs across checked/timed repetitions. Candidate/off matches main on all 21 cells, including full error bits, ranks, termination and oracle/request counts. The separate real/complex seed-4 #1352 regressions and the heavy survey pass their unchanged dense accuracy checks.

Measured candidate: `ed000a66c7fde91e40a29a23c59dcc46d53b3596`. Exact references, lock/source/binary hashes, CPU/compiler/thread settings and dirty states are in [manifest.json](data/manifest.json) and [compiler.txt](data/compiler.txt). The reference checkouts have only the listed untracked diagnostic sources; their tracked library sources are unchanged. [Protocol](README.md), [all checked cells](checked.csv), [all timing samples](timed.csv), [raw data](data/).

## Same-library comb cost comparison

Every row is the same smooth 60-site target at tolerance 1e-8 and the stated seed. Reported and independently sampled errors come from the same interpolation. Independent validation uses the registered 20,000 points, not an exhaustive domain bound. Historical runs converge; main and both candidate modes reach all 30 sweeps. The final recorded errors can be below tolerance while the complete ordinary convergence window fails.

| Seed | Revision / mode | Sweeps | Max rank | Oracle points | Reported error | Sampled relative max | Sampled relative L2 | Confirmations / failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Historical 501ebb4a | 5 | 44 | 6,360,596 | 9.91169e-09 | 1.42219e-08 | 2.26408e-08 | 0 / 0 |
| 1 | Main 35174fdd | 30 | 44 | 59,932,373 | 7.68478e-09 | 1.25859e-08 | 1.84401e-08 | 0 / 0 |
| 1 | Candidate/off | 30 | 44 | 59,932,373 | 7.68478e-09 | 1.25859e-08 | 1.84401e-08 | 0 / 0 |
| 1 | Candidate/on | 30 | 44 | 59,932,373 | 7.68478e-09 | 1.25859e-08 | 1.84401e-08 | 0 / 0 |
| 2 | Historical 501ebb4a | 5 | 43 | 6,099,323 | 9.80264e-09 | 1.62345e-08 | 2.26159e-08 | 0 / 0 |
| 2 | Main 35174fdd | 30 | 44 | 59,344,799 | 8.73063e-09 | 1.56599e-08 | 2.70026e-08 | 0 / 0 |
| 2 | Candidate/off | 30 | 44 | 59,344,799 | 8.73063e-09 | 1.56599e-08 | 2.70026e-08 | 0 / 0 |
| 2 | Candidate/on | 30 | 44 | 59,344,799 | 8.73063e-09 | 1.56599e-08 | 2.70026e-08 | 0 / 0 |
| 3 | Historical 501ebb4a | 6 | 44 | 8,361,963 | 9.82558e-09 | 1.95819e-08 | 2.73325e-08 | 0 / 0 |
| 3 | Main 35174fdd | 30 | 45 | 61,831,775 | 9.73257e-09 | 2.27835e-08 | 3.04164e-08 | 0 / 0 |
| 3 | Candidate/off | 30 | 45 | 61,831,775 | 9.73257e-09 | 2.27835e-08 | 3.04164e-08 | 0 / 0 |
| 3 | Candidate/on | 30 | 45 | 61,831,775 | 9.73257e-09 | 2.27835e-08 | 3.04164e-08 | 0 / 0 |

Timer scope is the whole interpolation call, including any confirmation replay and requested final materialization, with context construction, identical priming and validation excluded. **These three comb runs execute zero confirmations, so no replay actually occurs**. The option-on difference measures history recording/storage on this workload; it does not measure replay throughput. Gaussian checked cells below exercise actual replay. Confirmation remains sampled and is never a dense error certificate.

Times are medians of two complete paired rounds in seconds; ratios are medians of the two within-round ratios. Two samples are descriptive observations, with no confidence interval or speedup acceptance claim. Raw samples and guard observables remain available.

| Seed | Historical s | Main s | Off s | On s | Off / main | On / off | On / historical |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.1174 | 9.6260 | 9.5934 | 30.1365 | 0.997 | 3.141 | 26.971 |
| 2 | 1.0562 | 9.5221 | 9.4597 | 29.7208 | 0.993 | 3.142 | 28.139 |
| 3 | 1.4012 | 9.8321 | 9.8077 | 31.1755 | 0.998 | 3.179 | 22.250 |

## Same-library Gaussian counts and accuracy

Totals aggregate the six registered seeds at each tolerance. The CSV retains every seed, reported/dense error, verdict and confirmation count; raw JSON also retains complete rank/error-bit histories. No Gaussian wall times are reported. A dense miss means `Converged` with independent relative max error above tolerance.

| Tolerance | Revision / mode | Oracle points | Converged | Dense misses among converged | Confirmation checks / failures |
| --- | --- | ---: | ---: | ---: | ---: |
| 1e-04 | Historical 501ebb4a | 433,152 | 6 / 6 | 3 | 0 / 0 |
| 1e-04 | Main 35174fdd | 1,362,136 | 5 / 6 | 2 | 0 / 0 |
| 1e-04 | Candidate/off | 1,362,136 | 5 / 6 | 2 | 0 / 0 |
| 1e-04 | Candidate/on | 1,607,991 | 4 / 6 | 0 | 6 / 2 |
| 1e-06 | Historical 501ebb4a | 634,331 | 6 / 6 | 4 | 0 / 0 |
| 1e-06 | Main 35174fdd | 2,135,123 | 5 / 6 | 2 | 0 / 0 |
| 1e-06 | Candidate/off | 2,135,123 | 5 / 6 | 2 | 0 / 0 |
| 1e-06 | Candidate/on | 2,551,767 | 3 / 6 | 0 | 9 / 6 |
| 1e-08 | Historical 501ebb4a | 753,077 | 6 / 6 | 5 | 0 / 0 |
| 1e-08 | Main 35174fdd | 1,641,678 | 6 / 6 | 6 | 0 / 0 |
| 1e-08 | Candidate/off | 1,641,678 | 6 / 6 | 6 | 0 / 0 |
| 1e-08 | Candidate/on | 4,393,113 | 1 / 6 | 1 | 18 / 17 |

The option-on Gaussian seed 2 at 1e-8 still reports `Converged`: reported error 8.451641799880936e-9 versus dense relative max 1.0037372266680222e-8 after one passing confirmation. This negative accuracy result is retained. Confirmation resolves the registered #1352 frozen-plane cases, but does not solve the general #1371 numerical-policy problem or guarantee an unseen point. Per-bond stopping/local-error policy remains #1405.

## Reproduction

Use the exact candidate and reference commits in the manifest. Copy the registered driver into each reference checkout, removing only its unavailable confirmation/history option fields and replacing confirmation counters with zero; the exact adapted sources are retained in `data/*-driver.rs`. Build each release example through `ai-run build --mem 6G --jobs 2`, with separate `target/1372-reference` directories. Then invoke `run.py` with a fresh output directory and the three checkout paths, from an observer process confined to CPUs 0–7. The runner rejects changed sources/binaries, checks main/off equality and all repeated outputs, and only writes `VALID.json` after all eight CLEAN reports. It does not publish or merge anything.
