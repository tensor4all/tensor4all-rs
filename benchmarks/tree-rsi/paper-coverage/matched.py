"""Post-diagnosis matched-accuracy timing; original failed settings stay visible."""
import datetime,json,os,shutil,pathlib,sys
from evidence import checked_workers,snapshot_harness,workers_unchanged,harness_unchanged
from build import BASELINE
from fixtures import OUT,ROOT,MAIN_COMMIT,save,sha
import run

CASE=dict(fixture='dmrg-n20-chi30',cap=470,options=dict(local_tolerance=1e-14,oversampling=64))

def main():
    prior=sorted(OUT.glob('supplement-*'))[-1]
    if not (prior/'completion.json').exists():raise RuntimeError('complete sensitivity run first')
    checks={r['tag']:r for r in map(json.loads,(prior/'global-checks.jsonl').read_text().splitlines())}
    selected=[r for r in map(json.loads,(prior/'observations.jsonl').read_text().splitlines()) if r['case']==CASE]
    if len(selected)!=6 or not all(checks[r['tag']]['gate_1e8'] for r in selected):raise RuntimeError('chosen setting did not pass independent global checks for all seeds/algorithms')
    directory=OUT/('matched-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));directory.mkdir()
    cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu})
    algorithms=['treeaci','rsi']
    workers,evidence=checked_workers(OUT,dict(treeaci=BASELINE,rsi=ROOT),ROOT,algorithms)
    source_files=snapshot_harness(directory,pathlib.Path(__file__).parent)
    save(directory/'protocol.json',dict(algorithms=algorithms,cases=[CASE],seeds=run.SEEDS,blocks=5,threads=1,affinity=[cpu],
        baseline=MAIN_COMMIT,**evidence,phase='matched',source_files=source_files,
        selection='post-diagnosis setting with all6 independent global error checks below1e-8; not a preselected broad success rate',
        acceptance='every result retains independent validation; global n20 checks required after run',
        timing='same API timing and process warmup as primary experiment;5 fixed blocks; keep failures and report variability'))
    cache={};count=0
    with (directory/'observations.jsonl').open('w') as log:
        for block in range(5):
            for seed in run.SEEDS:
                for algorithm in (['treeaci','rsi'] if (block+seed)%2 else ['rsi','treeaci']):
                    row=run.attempt(workers[algorithm],CASE,algorithm,seed,directory,'matched',block,cache)
                    log.write(json.dumps(row)+'\n');log.flush();count+=1
                    print('MATCHED',count,algorithm,seed,block,row['status'],row.get('seconds'),flush=True)
    save(directory/'completion.json',dict(observations=count,expected=30,candidate_source_unchanged=workers_unchanged(OUT,evidence['worker_builds'],ROOT) and harness_unchanged(pathlib.Path(__file__).parent,source_files)))
    print('FINISHED',directory,flush=True)
if __name__=='__main__':main()
