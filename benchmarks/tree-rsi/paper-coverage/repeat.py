"""Five fixed timing blocks; failures and accuracy are retained in every block."""
import datetime,json,os,pathlib,sys,shutil
import run
from evidence import checked_workers,snapshot_harness,workers_unchanged,harness_unchanged
from build import BASELINE
from fixtures import OUT,ROOT,MAIN_COMMIT,save,sha

SELECTION=[
    dict(fixture='dmrg-n20-chi30',cap=470,options={}),
    dict(fixture='dmrg-n50-chi150',cap=300,options={}),
    dict(fixture='gaussian-0.49-0.51-0.01',cap=12,options=dict(oversampling=5)),
    dict(fixture='oscillatory-m2',cap=30,options=dict(oversampling=10)),
    dict(fixture='gpe-91-vortices',cap=160,options={}),
    dict(fixture='complex-tree-n20-m2-chi16-root0',cap=256,options={}),
]

def main():
    for group in ('dmrg','functions','tree'):
        latest=sorted(OUT.glob('run-'+group+'-*'))[-1]
        if not (latest/'completion.json').exists():raise RuntimeError('finish coverage timing first')
    directory=OUT/('repeat-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));directory.mkdir()
    cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu})
    algorithms=['treeaci','rsi']
    workers,evidence=checked_workers(OUT,dict(treeaci=BASELINE,rsi=ROOT),ROOT,algorithms)
    source_files=snapshot_harness(directory,pathlib.Path(__file__).parent)
    save(directory/'protocol.json',dict(algorithms=algorithms,cases=SELECTION,seeds=run.SEEDS,blocks=5,threads=1,affinity=[cpu],
        baseline=MAIN_COMMIT,**evidence,phase='repeat',source_files=source_files,
        selection='fixed physical, function and tree cases including both passes and failures; no exclusion by observed timing',
        timing='one API call per process; warm backend via bounded input evaluation; export and validation excluded',
        variability='coefficient of variation per fixed seed; above10% labeled unstable; median ratios descriptive only',
        acceptance='same1e-8 gates and independent references as coverage; a failed product has no accepted speedup'))
    count=0;validation_cache={}
    with (directory/'observations.jsonl').open('w') as log:
        for block in range(5):
            for i,case in enumerate(SELECTION):
                for seed in run.SEEDS:
                    for algorithm in (['treeaci','rsi'] if (block+i+seed)%2 else ['rsi','treeaci']):
                        record=run.attempt(workers[algorithm],case,algorithm,seed,directory,'repeat',block,validation_cache)
                        log.write(json.dumps(record)+'\n');log.flush();count+=1
                        print('REPEAT',count,case,algorithm,seed,block,record['status'],record.get('seconds'),flush=True)
    save(directory/'completion.json',dict(observations=count,expected=180,candidate_source_unchanged=workers_unchanged(OUT,evidence['worker_builds'],ROOT) and harness_unchanged(pathlib.Path(__file__).parent,source_files)))
    print('FINISHED',directory,flush=True)
if __name__=='__main__':main()
