"""Untimed replay of every fixture/cap/ACI-option setting, seed1, with rank hooks."""
import ast,json,os,datetime,sys,shutil,difflib
from pathlib import Path
from build import BASELINE_ROOT,build
from fixtures import OUT,ROOT,MAIN_COMMIT,save,sha
from evidence import checked_workers,snapshot_harness,workers_unchanged,harness_unchanged
from summarize import records
from run import attempt

def main():
    repeated=sorted(OUT.glob('repeat-*'))[-1]
    if not (repeated/'completion.json').exists():raise RuntimeError('finish timing before diagnostic build')
    initial=BASELINE_ROOT
    source=initial/'trace/main'
    if not source.exists():
        # Reuse the same four hooks without running the older benchmark suite.
        sys.path.insert(0,str(ROOT/'benchmarks/tree-rsi/main-comparison'))
        from trace import HOOKS
        source.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(initial/'main',source);patch=[]
        for filename,replacements in HOOKS.items():
            path=source/'crates/tensor4all-treeaci/src'/filename;old=path.read_text();new=old
            for before,after in replacements:
                if new.count(before)!=1:raise RuntimeError('rank trace hook no longer matches baseline')
                new=new.replace(before,after)
            path.write_text(new);patch.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+filename,tofile='b/'+filename))
        (source.parent/'instrumentation.patch').write_text(''.join(patch))
    # This copy is the previously verified pristine baseline plus four
    # eprintln hooks; its patch is preserved and hashed in this run.
    build('treeaci-trace',source,'worker.rs','treeaci')
    directory=OUT/('trace-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));directory.mkdir()
    rows=[];seen=set()
    folders=[sorted(OUT.glob('run-'+group+'-*'))[-1] for group in ('dmrg','functions','tree')]
    extra=sorted(OUT.glob('supplement-*'))
    if extra:folders.append(extra[-1])
    for folder in folders:
        if not (folder/'completion.json').exists():raise RuntimeError('finish all measured calls first')
        for row in records(folder):
            if row['algorithm']!='treeaci' or row['seed']!=1:continue
            case=row['case'];options={k:v for k,v in case['options'].items() if k not in ('oversampling','sketch_dim')}
            key=(case['fixture'],case['cap'],json.dumps(options,sort_keys=True))
            if key in seen:continue
            seen.add(key);rows.append((folder,row))
    workers,evidence=checked_workers(OUT,{'treeaci-trace':source},ROOT,['treeaci-trace'])
    evidence['worker_builds']={'treeaci':evidence['worker_builds']['treeaci-trace']}
    evidence['worker_hashes']={'treeaci':evidence['worker_hashes']['treeaci-trace']}
    worker=workers['treeaci-trace'];cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu})
    source_files=snapshot_harness(directory,Path(__file__).parent)
    save(directory/'protocol.json',dict(selection='every fixture/cap/ACI-option combination at seed1; RSI-only options deduplicated; not every seed',
        baseline=MAIN_COMMIT,**evidence,patch_sha256=sha(initial/'trace/instrumentation.patch'),
        cases=[row['case'] for _,row in rows],seeds=[1],algorithms=['treeaci'],phase='trace',
        source_files=source_files,
        timing='diagnostic times excluded',parity='exact exported tensor bytes, output ranks and diagnostics against untouched-main coverage'))
    shutil.copy2(initial/'trace/instrumentation.patch',directory/'instrumentation.patch')
    cache={}
    for folder,row in rows:
        if row['status']=='completed':
            fixture=OUT/'fixtures'/row['case']['fixture'];binary=folder/(row['tag']+'.bin');value=row['validation']
            corrections=folder/'validation-corrections.json'
            if corrections.exists():
                for correction in json.loads(corrections.read_text()):
                    if correction['tag']==row['tag']:value=correction['validation']
            cache[(sha(fixture/'inputs.bin'),sha(fixture/'fixture.json'),sha(binary),sha(binary.with_suffix('.cores.json')))]=(row['tag'],value)
    with (directory/'observations.jsonl').open('w') as log:
        for folder,baseline in rows:
            r=attempt(worker,baseline['case'],'treeaci',1,directory,'trace',validation_cache=cache)
            peak=1;matrices=[];commits=guards=0
            for line in (directory/(r['tag']+'.stderr')).read_text().splitlines():
                if not line.startswith('RANKTRACE '):continue
                kind,data=line[10:].split(' ',1)
                if kind in ('init','guard'):
                    peak=max(peak,max(ast.literal_eval(data),default=1));guards+=kind=='guard'
                elif kind=='commit':peak=max(peak,int(data.split()[1]));commits+=1
                elif kind=='matrix':matrices.append(list(map(int,data.split())))
            r.update(peak_active_bond_rank=peak,local_matrix_shapes=matrices,edge_commit_count=commits,guard_injection_count=guards,
                baseline_directory=folder.name,baseline_tag=baseline['tag'],
                matches_untouched_main=(r.get('validation',{}).get('output_sha256')==baseline.get('validation',{}).get('output_sha256') and
                    all(r.get(k)==baseline.get(k) for k in ('output_ranks','diagnostics','status'))))
            log.write(json.dumps(r)+'\n');log.flush();print('TRACE',r['tag'],'rank',peak,'parity',r['matches_untouched_main'],flush=True)
    save(directory/'completion.json',dict(observations=len(rows),expected=len(rows),candidate_source_unchanged=workers_unchanged(OUT,evidence['worker_builds'],ROOT) and harness_unchanged(Path(__file__).parent,source_files)))
    print('FINISHED',directory,flush=True)
if __name__=='__main__':main()
