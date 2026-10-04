"""Labeled extra validation and sensitivity cases; do not replace original curves."""
import argparse,datetime,json,os,pathlib,shutil,sys
import run
from evidence import checked_workers,snapshot_harness,workers_unchanged,harness_unchanged
from build import BASELINE
from fixtures import OUT,ROOT,MAIN_COMMIT,save,sha

CASES=[dict(fixture='complex-convolution-n16',cap=k,options={}) for k in (16,32,64,128,256)]
CASES += [dict(fixture='gpe-91-vortices',cap=k,options={}) for k in (512,1024)]
CASES += [dict(fixture='dmrg-n20-chi30',cap=470,options=dict(local_tolerance=1e-14,oversampling=p)) for p in (5,64)]
CASES += [dict(fixture='dmrg-n20-chi30',cap=470,options=dict(local_tolerance=0.0,oversampling=5)),
          dict(fixture='dmrg-n20-chi30',cap=470,options=dict(local_tolerance=0.0,sketch_dim=245))]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpe-tolerance',action='store_true');args=parser.parse_args()
    selected=[dict(fixture='gpe-91-vortices',cap=k,options=dict(local_tolerance=t)) for k in (256,512,1024) for t in (1e-14,0.0)] if args.gpe_tolerance else CASES
    algorithms=['rsi'] if args.gpe_tolerance else ['treeaci','rsi']
    repeated=sorted(OUT.glob('repeat-*'))[-1]
    if not (repeated/'completion.json').exists():raise RuntimeError('finish fixed repetitions first')
    prefix='gpe-tolerance-' if args.gpe_tolerance else 'supplement-'
    directory=OUT/(prefix+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));directory.mkdir()
    cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu})
    workers,evidence=checked_workers(OUT,dict(treeaci=BASELINE,rsi=ROOT),ROOT,algorithms)
    source_files=snapshot_harness(directory,pathlib.Path(__file__).parent)
    save(directory/'protocol.json',dict(cases=selected,algorithms=algorithms,seeds=run.SEEDS,baseline=MAIN_COMMIT,
        **evidence,phase='supplement',source_files=source_files,threads=1,affinity=[cpu],
        selection=('post-observation GPE tolerance sensitivity; only RSI rerun, untouched-main baselines retained from earlier batches' if args.gpe_tolerance else 'additional bounded convolution, larger GPE caps, and explicitly post-observation DMRG tolerance/sketch sensitivity; original failures retained'),
        timing='coverage only; not a replacement for fixed repetitions'))
    cache={};count=0
    with (directory/'observations.jsonl').open('w') as log:
        for i,case in enumerate(selected):
            for seed in run.SEEDS:
                for algorithm in (algorithms if (i+seed)%2 else list(reversed(algorithms))):
                    row=run.attempt(workers[algorithm],case,algorithm,seed,directory,'supplement',validation_cache=cache)
                    log.write(json.dumps(row)+'\n');log.flush();count+=1
                    print('SUPPLEMENT',count,case,algorithm,seed,row['status'],row.get('seconds'),row.get('validation'),flush=True)
    save(directory/'completion.json',dict(observations=count,expected=len(selected)*len(algorithms)*len(run.SEEDS),candidate_source_unchanged=workers_unchanged(OUT,evidence['worker_builds'],ROOT) and harness_unchanged(pathlib.Path(__file__).parent,source_files)))
    print('FINISHED',directory,flush=True)
if __name__=='__main__':main()
