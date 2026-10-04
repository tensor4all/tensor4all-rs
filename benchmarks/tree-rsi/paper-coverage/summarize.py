"""Recompute completeness and acceptance from raw measurements, not saved prose."""
import collections,json,math,statistics
from fixtures import OUT,save,sha
from evidence import verify_schedule

def records(folder):
    protocol=json.loads((folder/'protocol.json').read_text())
    completion=json.loads((folder/'completion.json').read_text())
    rows=list(map(json.loads,(folder/'observations.jsonl').read_text().splitlines()))
    verify_schedule(protocol,completion,rows)
    corrections={}
    if (folder/'validation-corrections.json').exists():
        corrections={r['tag']:r['validation'] for r in json.loads((folder/'validation-corrections.json').read_text())}
    global_checks={}
    if (folder/'global-checks.jsonl').exists():global_checks={r['tag']:r for r in map(json.loads,(folder/'global-checks.jsonl').read_text().splitlines())}
    gpe_checks={}
    if (folder/'gpe-global-checks.jsonl').exists():gpe_checks={r['tag']:r for r in map(json.loads,(folder/'gpe-global-checks.jsonl').read_text().splitlines())}
    bounds=json.loads((OUT/'reference-truncation-bounds.json').read_text()) if (OUT/'reference-truncation-bounds.json').exists() else {}
    for r in rows:
        r['run_directory']=folder.name
        if r['tag'] in corrections:r['validation']=corrections[r['tag']];r['validation_corrected']=True
        if r['tag'] in global_checks:r['global_check']=global_checks[r['tag']]
        if r['tag'] in gpe_checks:r['gpe_global_check']=gpe_checks[r['tag']]
        r['acceptance_scope']='not validated';r['accepted']=False;r['measured_error']=None
        if r['status']!='completed':continue
        v=r['validation']
        if 'gpe_global_check' in r:
            error=r['gpe_global_check']['full_grid_relative_l2'];r.update(acceptance_scope='all2^26 entries, blocked exact half-chain contractions',measured_error=error,accepted=math.isfinite(error) and error<=1e-8)
        elif 'global_check' in r:
            g=r['global_check'];error=g['relative_l2_vs_reference_tt'];bound=g['relative_reference_discarded_norm']
            r.update(acceptance_scope='global QR reference with measured SVD truncation',measured_error=error,accepted=math.isfinite(error) and error+bound<=1e-8)
        elif 'full_grid_relative_l2' in v:
            error=v['full_grid_relative_l2'];r.update(acceptance_scope='full physical grid',measured_error=error,accepted=math.isfinite(error) and error<=1e-8)
        elif 'relative_l2_vs_reference_tt' in v:
            error=v['relative_l2_vs_reference_tt'];bound=bounds.get(r['case']['fixture'],{}).get('relative_svd_truncation_upper_bound')
            r.update(acceptance_scope='bounded rounded reference TT; formula sample checked separately',measured_error=error,
                reference_relative_truncation_bound=bound,accepted=bound is not None and math.isfinite(error) and error+bound<=1e-8)
        elif 'importance_relative_l2_estimate' in v:
            error=v['importance_relative_l2_estimate'];checks=[error,v['norm_abs_error']]
            if 'hzz_abs_error' in v:checks.append(v['hzz_abs_error'])
            r.update(acceptance_scope='observables and held-out estimate only, no global certificate',measured_error=error,accepted=all(math.isfinite(e) and e<=1e-8 for e in checks))
        if any(edge[2]>r['case']['cap'] for edge in r.get('output_ranks',[])):raise ValueError('reported output exceeds cap')
    return rows

def main():
    all_rows=[];groups={}
    for group in ('dmrg','functions','tree'):
        folder=sorted(OUT.glob('run-'+group+'-*'))[-1];rows=records(folder);all_rows+=rows
        groups[group]=dict(directory=folder.name,observations=len(rows),status=dict(collections.Counter(r['status'] for r in rows)),
                           accepted=dict(collections.Counter(r['algorithm'] for r in rows if r['accepted'])))
    for experiment in ('supplement','gpe-tolerance'):
        extra=sorted(OUT.glob(experiment+'-*'))
        if extra and (extra[-1]/'completion.json').exists():
            rows=records(extra[-1]);all_rows+=rows
            groups[experiment]=dict(directory=extra[-1].name,observations=len(rows),status=dict(collections.Counter(r['status'] for r in rows)),accepted=dict(collections.Counter(r['algorithm'] for r in rows if r['accepted'])))
    trace={}
    trace_dirs=sorted(OUT.glob('trace-*'))
    if trace_dirs and (trace_dirs[-1]/'completion.json').exists():
        for r in map(json.loads,(trace_dirs[-1]/'observations.jsonl').read_text().splitlines()):
            if not r['matches_untouched_main']:raise ValueError('instrumentation changed output')
            trace[r['case']['fixture'],r['case']['cap'],r['case']['options'].get('local_tolerance',1e-12)]=r['peak_active_bond_rank']
    grouped=collections.defaultdict(list)
    for r in all_rows:grouped[r['case']['fixture'],r['case']['cap'],json.dumps(r['case']['options'],sort_keys=True),r['algorithm']].append(r)
    summary=[]
    for (fixture,cap,options_json,algorithm),rows in grouped.items():
        options=json.loads(options_json);p=options.get('oversampling',5);tolerance=options.get('local_tolerance',1e-12)
        manifest=json.loads((OUT/'fixtures'/fixture/'fixture.json').read_text())
        input_rank=max(max(core['shape'][:-1],default=1) for operand in manifest['operands'] for core in operand)
        done=[r for r in rows if r['status']=='completed'];errors=[r['measured_error'] for r in done if r['measured_error'] is not None]
        ranks=[max(e[2] for e in r['output_ranks']) for r in done]
        peak=max((max(e['rank'] for e in r['diagnostics']['edges']) for r in done),default=None) if algorithm=='rsi' else trace.get((fixture,cap,tolerance))
        widths=[r['diagnostics']['sketch_dim'] for r in done] if algorithm=='rsi' else []
        summary.append(dict(fixture=fixture,cap=cap,oversampling=p,local_tolerance=tolerance,options=options,algorithm=algorithm,observations=len(rows),
            input_max_rank=input_rank,
            sketch_width_range=[min(widths),max(widths)] if widths else None,
            accepted=sum(r['accepted'] for r in rows),seconds_median=statistics.median(r['seconds'] for r in done) if done else None,
            error_range=[min(errors),max(errors)] if errors else None,output_rank_range=[min(ranks),max(ranks)] if ranks else None,
            intermediate_rank=peak,intermediate_scope='all constructed edge ranks' if algorithm=='rsi' else 'seed1 diagnostic replay',
            acceptance_scope=done[0]['acceptance_scope'] if done else 'no completed call'))
    timing=[]
    for experiment in ('repeat','matched'):
        repeat_dirs=sorted(p for p in OUT.glob(experiment+'-*') if p.is_dir())
        if not repeat_dirs or not (repeat_dirs[-1]/'completion.json').exists():continue
        repeated=records(repeat_dirs[-1]);buckets=collections.defaultdict(list)
        for r in repeated:buckets[r['case']['fixture'],r['algorithm'],r['seed']].append(r)
        for key,rows in buckets.items():
            seconds=[r['seconds'] for r in rows if r['status']=='completed'];cv=statistics.stdev(seconds)/statistics.mean(seconds) if len(seconds)>1 else None
            timing.append(dict(experiment=experiment,fixture=key[0],algorithm=key[1],seed=key[2],calls=len(rows),accepted=sum(r['accepted'] for r in rows),seconds_median=statistics.median(seconds),coefficient_of_variation=cv,unstable=cv is None or cv>.10))
    save(OUT/'summary.json',dict(groups=groups,case_summary=summary,timing=timing,
        limitations=['rank curves include intentionally insufficient caps; acceptance counts are not algorithm rankings',
        'n50 DMRG remains observables/samples only; GPE global checks apply only where a completed full-grid record exists',
        'function reference is independently rounded; input/formula errors are separate',
        'intermediate rank refers to the working output network; original input ranks and local matrix shapes are separate quantities',
        'paper Julia timings and unpublished convolution parameters are not reproduced',
        'nonlinear ReLU/reciprocal and actual native GW pipeline remain unsupported/unaccepted']))
    def number(v):return 'unavailable' if v is None else f'{v:.4g}'
    lines=['# Recomputed benchmark observations','',
        'Generated from raw observations and validation corrections. Rank-limited failures do not identify implementation bugs. Timing medians below are descriptive coverage calls, not accepted speedup claims.','',
        '| Fixture | Cap | p / actual k | Local tolerance | Algorithm | Pass / 3 | Error min–max | Seconds median | Output rank | Intermediate rank |',
        '|---|---:|---|---:|---|---:|---:|---:|---|---:|']
    for r in summary:
        err='–'.join(map(number,r['error_range'])) if r['error_range'] else 'unavailable'
        lines.append(f"| {r['fixture']} | {r['cap']} | {r['oversampling']} / {r['sketch_width_range']} | {r['local_tolerance']} | {r['algorithm']} | {r['accepted']}/{r['observations']} | {err} | {number(r['seconds_median'])} | {r['output_rank_range']} | {r['intermediate_rank']} |")
    lines+=['','## Repeated timing variability','', '| Experiment | Fixture | Algorithm | Seed | Median seconds | CV | Unstable (>10%) |','|---|---|---|---:|---:|---:|---|']
    for r in timing:lines.append(f"| {r['experiment']} | {r['fixture']} | {r['algorithm']} | {r['seed']} | {number(r['seconds_median'])} | {number(r['coefficient_of_variation'])} | {r['unstable']} |")
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n');print(json.dumps(groups,indent=2));print('timing groups',len(timing))
if __name__=='__main__':main()
