#!/usr/bin/env python3
"""Render a complete clean-pinned scaling study only after ai-run says CLEAN."""
import json,random,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def median_ci(values):
    rng=random.Random(1347)
    samples=sorted(statistics.median(rng.choices(values,k=len(values))) for _ in range(10000))
    return [samples[249],samples[9749]]
def main():
    directory=Path(sys.argv[1]).resolve()
    gate=json.loads((directory/'ai-run.json').read_text())
    assert gate['verdict']=='CLEAN' and gate['exit']==0,'No timing conclusions from an invalid run'
    validation=json.loads((directory/'validation.json').read_text())
    assert validation['complete'] and validation['work_and_values_agree']
    rows=json.loads((directory/'samples.json').read_text());assert len(rows)==130
    source=json.loads((directory/'provenance.json').read_text())
    cells=[]
    for scalar in ('real','complex'):
        base=sorted((r for r in rows if r['scalar']==scalar and r['family']=='engine' and r['b']==1),key=lambda r:r['block'])
        for family in ('engine','lanes','mixed'):
            configs=sorted({(r['p'],r['b']) for r in rows if r['scalar']==scalar and r['family']==family},key=lambda c:c[0]*c[1])
            for p,b in configs:
                samples=sorted((r for r in rows if (r['scalar'],r['family'],r['p'],r['b'])==(scalar,family,p,b)),key=lambda r:r['block'])
                assert len(samples)==5 and {r['block'] for r in samples}==set(range(5))
                wall=[r['sample']['wall_ms'] for r in samples];mid=statistics.median(wall)
                ratio=statistics.median(r['sample']['wall_ms'] for r in base)/mid
                paired=[a['sample']['wall_ms']/z['sample']['wall_ms'] for a,z in zip(base,samples)]
                ci=median_ci(paired);mad=statistics.median(abs(v-mid) for v in wall)/mid
                cells.append(dict(scalar=scalar,family=family,p=p,b=b,total_threads=p*b,wall_ms=mid,
                    process_cpu_ms=statistics.median(r['sample']['process_cpu_ns']/1e6 for r in samples),
                    relative_mad=mad,speedup_over_single=ratio,paired_speedups=paired,paired_speedup_median_95=ci,
                    interpretation='imprecise' if mad>.15 else ('faster' if ratio>=1.10 and ci[0]>1 else
                                    'slower' if ratio<=1/1.10 and ci[1]<1 else 'no_clear_difference')))
    narrowed=[r for r in rows if r['sample']['initial_mask']!=r['sample']['after_construction_mask']]
    mixed_masks=sorted({mask for r in rows if r['family']=='mixed' for mask in r['sample']['callback_masks']})
    report=dict(provenance=source,ai_run=gate,validation=validation,cells=cells,samples=rows,
                constructor_mask_narrowing_processes=len(narrowed),mixed_callback_masks=mixed_masks)
    output=ROOT/'benchmarks/results';stem='2026-11-17-clean-pins'
    (output/(stem+'-ai-run.json')).write_text((directory/'ai-run.json').read_text())
    (output/(stem+'-record.json')).write_text(json.dumps(report,indent=2)+'\n')
    text=['# Same-library native scaling on clean downstream-pinned dependencies','',
          f"Resource gate: **{gate['verdict']}**, exit **{gate['exit']}**; all **130** independent processes and numerical checks completed.",'',
          '[Frozen protocol](2026-11-17-clean-pins-protocol.md) · [All samples and source audit](2026-11-17-clean-pins-record.json) · [Original ai-run report](2026-11-17-clean-pins-ai-run.json).','',
          f"Production base `{source['root_base']}`; harness `{source['harness_commit']}`. Production code and numerical dependency identities are unchanged from the downstream lock. Tenferro is the downstream Git pin `32a4e98c` (0.10), Hataori `9d655157`, and threadteam registry 0.5.1; no local dependency override or repair is used. This is not a latest-upstream comparison.",'',
          'Four physical cores are CPUs 8,10,12,14. Eight and sixteen compute participants intentionally oversubscribe those cores. Each process constructs its resources outside timing, primes for two seconds, runs a checked warmup, then times one public QTT API call. Each scalar has its own pool-free one-thread baseline. These are internal modes of the same library, not competing implementations.','']
    for scalar in ('real','complex'):
        text += [f'## {scalar.title()} evaluator','',
                 '| Mode | Lanes P | Threads/engine B | Total compute participants | Wall median ms | CPU median ms | Speedup / single | Paired 95% speedup interval | Relative MAD | Precision |',
                 '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |']
        for c in cells:
            if c['scalar']!=scalar:continue
            lo,hi=c['paired_speedup_median_95']
            text.append(f"| {c['family']} | {c['p']} | {c['b']} | {c['total_threads']} | {c['wall_ms']:.3f} | {c['process_cpu_ms']:.3f} | {c['speedup_over_single']:.3f}× | [{lo:.3f}, {hi:.3f}] | {c['relative_mad']:.2%} | {c['interpretation']} |")
        text+=['']
    text+=['## Correctness, placement and limits','',
           'Every warmup and timed output passed the independent 16×16 grid oracle at the original 1e-4 tolerance, exact grid ownership and canonical prefix order. Warmup/time results agree; callback counts, evaluator points, prefixes, core bytes and all grid values agree across configurations within each scalar (existing 1e-10 diagnostic comparison tolerance). The full values and every raw observation remain in the record.','',
           f"Constructor masks narrowed in **{len(narrowed)}/130** processes. This is recorded behavior of the unmodified constructors, not silently repaired by the harness. Initial, per-constructor and warmup callback masks are retained for every process. The configurations request resource budgets inside one shared process CPU allocation; they do not promise disjoint per-engine domains.",'',
           f"All 30 mixed-mode processes observed only these callback masks: `{mixed_masks}`. Their first B2 construction narrowed the creator from the four-core allocation to CPU8, and subsequent constructors retained CPU8. This directly records loss of the intended allocation in the clean pinned stack; it is not evidence about latest upstream. It also prevents interpreting P×B as evidence that all four physical cores actually executed evaluator work.",'',
           'The process-CPU clock brackets the primary wall timer. Callback counters and affinity queries are absent from the timed callback monomorphization. Setup, priming, warmup and dense verification are outside primary timing. No previous withdrawn sample is reused; no local repair, MPI speedup or latest-upstream result is inferred. More than 15% relative MAD is marked imprecise while retaining every sample.','']
    (output/(stem+'.md')).write_text('\n'.join(text))
    print('CLEAN verified; report generated; constructor mask narrowing:',len(narrowed))
    for c in cells:print(c['scalar'],c['family'],c['p'],c['b'],round(c['wall_ms'],3),round(c['speedup_over_single'],3),c['interpretation'])
if __name__=='__main__':main()
