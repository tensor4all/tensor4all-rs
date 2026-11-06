#!/usr/bin/env python3
"""Predeclared high-rank sketch cases plus three timing blocks for every case."""
import compare as c
import argparse, os, pathlib, shutil, json, datetime
SUPPLEMENT = [(t,20,chi,cap) for t in ('chain','balanced') for chi in ((16,16),(128,2)) for cap in (256,512)]
def name(t,n,chi): return f'{t}-n{n}-chi'+('x'.join(map(str,chi)) if isinstance(chi,tuple) else str(chi))
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('initial',type=pathlib.Path); args=parser.parse_args()
    original=args.initial.resolve(); directory=original/'supplement'; directory.mkdir()
    initial=c.json.loads((original/'protocol.json').read_text()); cpu=initial['affinity'][0]
    source=c.fingerprint()
    c.save(directory/'protocol.json',dict(cases=SUPPLEMENT,original_cases=c.CASES,blocks=3,seeds=c.SEEDS,
        baseline=initial['baseline'],candidate_source_sha256=source,driver_sha256=c.sha(c.SOURCE/'compare.py'),supplement_sha256=c.sha(c.SOURCE/'supplement.py'),worker_sha256=c.sha(c.SOURCE/'worker.rs'),
        rationale='Initial accurate cases used exact complement on the hardest edge; add true rank-256 products with central complement dimension 1024, requiring randomized sketch compression.',
        gates='same local tolerance 1e-12, full-grid relative L2 gate 1e-8, phase timeout 180s; no cases removed',
        timing='three blocks of all three seeds, alternating order; per-seed CV <= 0.10 required for stable timing label; no formal population speedup claim',
        fixture='each operand has its own true bond dimensions; 16 x 16 and 128 x 2 are bounded by rank 256; explicit dense oracle limit 2^20 entries',
        intermediate='separate diagnostic build after timing; numerical output hashes must match untouched main',
        affinity=[cpu]))
    for filename in ('compare.py','supplement.py','worker.rs'): shutil.copy2(c.SOURCE/filename,directory/filename)
    workers={}
    for algorithm,src in [('treeaci',original/'main'),('rsi',c.ROOT)]:
        print('BUILD',algorithm,flush=True); workers[algorithm]=c.build(directory/algorithm,src,algorithm,c.ROOT/'target')
    c.save(directory/'binaries.json',{a:dict(sha256=c.sha(p),lock_sha256=c.sha(p.parent/'Cargo.lock')) for a,p in workers.items()})
    os.sched_setaffinity(0,{cpu})
    for t,n,chi,cap in SUPPLEMENT:
        path=directory/name(t,n,chi)
        if not path.exists(): c.fixture(path,t,n,chi)
    all_cases=[(original,t,n,chi,cap) for t,n,chi,cap in c.CASES]+[(directory,t,n,chi,cap) for t,n,chi,cap in SUPPLEMENT]
    with (directory/'observations.jsonl').open('w') as file:
        for block in range(3):
            outputs=directory/f'block{block}';outputs.mkdir()
            for i,(base,t,n,chi,cap) in enumerate(all_cases):
                fixture=base/name(t,n,chi);case=f'{fixture.name}-cap{cap}'
                for seed in c.SEEDS:
                    order=['treeaci','rsi'] if (i+seed+block)%2 else ['rsi','treeaci']
                    for algorithm in order:
                        record=c.run(workers[algorithm],fixture,case,algorithm,seed,outputs); record['block']=block
                        file.write(json.dumps(record)+'\n');file.flush()
                        print('RESULT',block,case,algorithm,seed,record['status'],record.get('seconds'),record.get('relative_l2'),flush=True)
    c.save(directory/'completion.json',dict(source_unchanged=source==c.fingerprint(),observations=3*len(all_cases)*3*2,completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print('FINISHED',directory,flush=True)
if __name__=='__main__': main()
