#!/usr/bin/env python3
"""Generate the registered result tables: analyze.py DATA_DIRECTORY OUTPUT_DIRECTORY."""
import csv
import json
from pathlib import Path
from statistics import median
import sys

src, out = map(Path, sys.argv[1:])
assert json.loads((src/'VALID.json').read_text())['status'] == 'VALID'
out.mkdir(parents=True, exist_ok=True)
manifest=json.loads((src/'manifest.json').read_text())
arms=['historical','main','off','on']
labels={'historical':'Historical 501ebb4a','main':'Main 35174fdd','off':'Candidate/off','on':'Candidate/on'}
checked={a:[json.loads(s) for s in (src/(a+'-checked.jsonl')).read_text().splitlines()] for a in arms}
timed={(n,a):[json.loads(s) for s in (src/f'round{n}-{a}.jsonl').read_text().splitlines()] for n in [1,2] for a in arms}
assert len(list(src.glob('*.guard.json'))) == 8
assert all(len(v) == 21 for v in checked.values())
assert all(len(v) == 3 for v in timed.values())
for f in src.glob('*.guard.json'):
 assert json.loads(f.read_text())['verdict']=='CLEAN'
fields=['arm','case','seed','tolerance','termination','sweeps','max_rank','oracle_points','requested_points','reported_error','validation_relmax','validation_rel_l2','validation_points','checks','failures','injections']
with (out/'checked.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader()
 for a in arms:
  for r in checked[a]:
   w.writerow({**{k:r[k] for k in ['case','seed','tolerance','termination','oracle_points','requested_points','reported_error','validation_relmax','validation_rel_l2','validation_points']}, 'arm':a,'sweeps':len(r['ranks']),'max_rank':max(r['ranks']),'checks':r['confirmation_counts'][0],'failures':r['confirmation_counts'][1],'injections':r['confirmation_counts'][2]})
with (out/'timed.csv').open('w') as f:
 w=csv.writer(f,lineterminator="\n");w.writerow(['round','arm','seed','seconds'])
 for (n,a),rows in timed.items():
  for r in rows:w.writerow([n,a,r['seed'],r['seconds']])
def secs(a,seed,n):return next(r['seconds'] for r in timed[n,a] if r['seed']==seed)
lines=['# Opt-in confirmation results', '',
'All 84 checked cells and 24 timed cells completed. All eight measurement guards are CLEAN, with zero exits and identical deterministic outputs across checked/timed repetitions. Candidate/off matches main on all 21 cells, including full error bits, ranks, termination and oracle/request counts. The separate real/complex seed-4 #1352 regressions and the heavy survey pass their unchanged dense accuracy checks.', '',
f"Measured candidate: `{manifest['arms']['off']['head']}`. Exact references, lock/source/binary hashes, CPU/compiler/thread settings and dirty states are in [manifest.json](data/manifest.json) and [compiler.txt](data/compiler.txt). The reference checkouts have only the listed untracked diagnostic sources; their tracked library sources are unchanged. [Protocol](README.md), [all checked cells](checked.csv), [all timing samples](timed.csv), [raw data](data/).", '',
'## Same-library comb cost comparison', '',
'Every row is the same smooth 60-site target at tolerance 1e-8 and the stated seed. Reported and independently sampled errors come from the same interpolation. Independent validation uses the registered 20,000 points, not an exhaustive domain bound. Historical runs converge; main and both candidate modes reach all 30 sweeps. The final recorded errors can be below tolerance while the complete ordinary convergence window fails.', '',
'| Seed | Revision / mode | Sweeps | Max rank | Oracle points | Reported error | Sampled relative max | Sampled relative L2 | Confirmations / failures |',
'| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for seed in [1,2,3]:
 for a in arms:
  r=next(r for r in checked[a] if r['case']=='comb60' and r['seed']==seed)
  lines.append(f"| {seed} | {labels[a]} | {len(r['ranks'])} | {max(r['ranks'])} | {r['oracle_points']:,} | {r['reported_error']:.5e} | {r['validation_relmax']:.5e} | {r['validation_rel_l2']:.5e} | {r['confirmation_counts'][0]} / {r['confirmation_counts'][1]} |")
lines += ['', 'Timer scope is the whole interpolation call, including any confirmation replay and requested final materialization, with context construction, identical priming and validation excluded. **These three comb runs execute zero confirmations, so no replay actually occurs**. The option-on difference measures history recording/storage on this workload; it does not measure replay throughput. Gaussian checked cells below exercise actual replay. Confirmation remains sampled and is never a dense error certificate.', '',
'Times are medians of two complete paired rounds in seconds; ratios are medians of the two within-round ratios. Two samples are descriptive observations, with no confidence interval or speedup acceptance claim. Raw samples and guard observables remain available.', '',
'| Seed | Historical s | Main s | Off s | On s | Off / main | On / off | On / historical |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
ratios={}
for seed in [1,2,3]:
 med=[median([secs(a,seed,n) for n in [1,2]]) for a in arms]
 ratio=[median([secs(a,seed,n)/secs(b,seed,n) for n in [1,2]]) for a,b in [('off','main'),('on','off'),('on','historical')]]
 ratios[seed]={'median_seconds':dict(zip(arms,med)),'paired_median_ratios':dict(zip(['off/main','on/off','on/historical'],ratio))}
 lines.append('| '+str(seed)+' | '+' | '.join(f'{v:.4f}' for v in med)+' | '+' | '.join(f'{v:.3f}' for v in ratio)+' |')
lines += ['', '## Same-library Gaussian counts and accuracy', '',
'Totals aggregate the six registered seeds at each tolerance. The CSV retains every seed, reported/dense error, verdict and confirmation count; raw JSON also retains complete rank/error-bit histories. No Gaussian wall times are reported. A dense miss means `Converged` with independent relative max error above tolerance.', '',
'| Tolerance | Revision / mode | Oracle points | Converged | Dense misses among converged | Confirmation checks / failures |', '| --- | --- | ---: | ---: | ---: | ---: |']
for tol in [1e-4,1e-6,1e-8]:
 for a in arms:
  rs=[r for r in checked[a] if r['case']=='gaussian16' and r['tolerance']==tol]
  conv=[r for r in rs if r['termination']=='Converged']
  lines.append(f"| {tol:.0e} | {labels[a]} | {sum(r['oracle_points'] for r in rs):,} | {len(conv)} / 6 | {sum(r['validation_relmax']>tol for r in conv)} | {sum(r['confirmation_counts'][0] for r in rs)} / {sum(r['confirmation_counts'][1] for r in rs)} |")
lines += ['', 'The option-on Gaussian seed 2 at 1e-8 still reports `Converged`: reported error 8.451641799880936e-9 versus dense relative max 1.0037372266680222e-8 after one passing confirmation. This negative accuracy result is retained. Confirmation resolves the registered #1352 frozen-plane cases, but does not solve the general #1371 numerical-policy problem or guarantee an unseen point. Per-bond stopping/local-error policy remains #1405.', '',
'## Reproduction', '',
'Use the exact candidate and reference commits in the manifest. Copy the registered driver into each reference checkout, removing only its unavailable confirmation/history option fields and replacing confirmation counters with zero; the exact adapted sources are retained in `data/*-driver.rs`. Build each release example through `ai-run build --mem 6G --jobs 2`, with separate `target/1372-reference` directories. Then invoke `run.py` with a fresh output directory and the three checkout paths, from an observer process confined to CPUs 0–7. The runner rejects changed sources/binaries, checks main/off equality and all repeated outputs, and only writes `VALID.json` after all eight CLEAN reports. It does not publish or merge anything. Regenerate the derived Markdown/CSV/summary using `analyze.py DATA_DIRECTORY OUTPUT_DIRECTORY`.', '']
(out/'RESULTS.md').write_text('\n'.join(lines))
(out/'summary.json').write_text(json.dumps(ratios,indent=2)+'\n')
print(json.dumps(ratios,indent=2))
