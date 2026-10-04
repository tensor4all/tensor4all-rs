#!/usr/bin/env python3
"""Fixed paper workload coverage. Every failure is an observation, not dropped."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS'):os.environ[key]='1'
import argparse,datetime,hashlib,json,pathlib,selectors,subprocess,time,sys,shutil
import numpy as np
import reference as ref
from evidence import checked_workers,snapshot_harness,workers_unchanged,harness_unchanged
from build import BASELINE
from fixtures import OUT,ROOT,MAIN_COMMIT,read_export,read_inputs,save,sha
SEEDS=[1,2,3]
TIMEOUT=180

def cases():
    out=[]
    def add(name,caps,**opts):
        for cap in caps:out.append(dict(fixture=name,cap=cap,options=opts))
    add('dmrg-n10-chi20',[20,40,80,160])
    for chi,caps in [(10,[10,20,30,40,50,55]),(15,[15]),(20,[20,50,100,150,200,210]),(25,[25]),(30,[30,100,200,300,400,470])]:add(f'dmrg-n20-chi{chi}',caps)
    for chi,caps in [(20,[20,50,100,150]),(40,[40,50,100,150]),(60,[50,60,150,250]),(80,[50,80,150,250]),(100,[100,200,300]),(150,[150,300,450])]:add(f'dmrg-n50-chi{chi}',caps)
    for name in ['gaussian-0.4-0.6-0.15','gaussian-0.25-0.75-0.15','gaussian-0.1-0.9-0.15','gaussian-0.49-0.51-0.01']:add(name,[4,6,8,10,12],oversampling=5)
    for suffix in ['', '-m3','-m4']:add('gaussian-0.4-0.6-0.15'+suffix,[15],oversampling=5)
    for m,caps in [(2,[5,10,15,20,25,30]),(3,[10,15,20,25,30,35,40,45]),(4,[10,20,30,40,50,60])]:
        for p in [0,5,10]:add(f'oscillatory-m{m}',caps,oversampling=p)
    add('active-matter-Dxx',[5,10,20,30])
    add('gpe-91-vortices',[40,80,160,256])
    for count,chi,roots in [(2,16,[0,7,19]),(4,4,[0,19])]:
        for root in roots:add(f'complex-tree-n20-m{count}-chi{chi}-root{root}',[128,256,512])
    return out

def validate(folder,output):
    manifest=json.loads((folder/'fixture.json').read_text());kind=manifest['metadata']['kind']
    if kind=='tree':
        from tree_cases import validate as validate_tree
        return validate_tree(folder,output)
    cores=read_export(output)
    d=dict(validation_scope=kind,output_sha256=sha(output),actual_max_rank=max(a.shape[2] for a in cores))
    if not all(np.isfinite(a).all() for a in cores):return dict(d,finite=False,acceptance=False)
    d['finite']=True
    if (folder/'expected-dense.npy').exists():
        exact=np.load(folder/'expected-dense.npy');actual=ref.dense(cores)
        if kind=='active_matter':
            # Original HDF5 uses float32. The worker receives f64 cores:
            # contract those exact decoded inputs in f64 for the product gate.
            inputs=read_inputs(folder);exact=np.ones_like(actual)
            for operand in inputs:exact*=ref.dense(operand)
            d['reference_precision']='f64 contraction/product of exact worker inputs; supersedes initial float32 dense reference'
        d.update(full_grid_relative_l2=float(np.linalg.norm(actual-exact)/np.linalg.norm(exact)),full_grid_relative_maxabs=float(np.max(np.abs(actual-exact))/np.max(np.abs(exact))))
        d['accuracy_pass']=d['full_grid_relative_l2']<=1e-8
        if (folder/'physical-dense.npy').exists():
            physical=np.load(folder/'physical-dense.npy')
            if kind=='active_matter':
                import h5py
                with h5py.File(OUT/'author/datasets/qtensor_well/active_matter_t50_Dxx.hdf5') as file:physical=file['quantics_tensor'][()].astype(np.float64).reshape(-1)**2
            d['physical_field_relative_l2']=float(np.linalg.norm(actual-physical)/np.linalg.norm(physical))
        if kind=='convolution':
            truth=np.load(folder/'physical-convolution.npy');inverse=np.fft.ifft(actual)
            d['convolution_relative_l2']=float(np.linalg.norm(inverse-truth)/np.linalg.norm(truth))
    if kind in ('dmrg','wavefunction'):
        true_norm=manifest['metadata']['norm']
        if kind=='dmrg':
            true_hzz=manifest['metadata']['hzz'];norm,hzz=ref.diagonal_observables(cores)
        else:
            state=np.ones(1,dtype=cores[0].dtype)
            for a in cores:state=state@a.sum(axis=1)
            norm=complex(state[0])
        sample=np.load(folder/'samples.npz');expected=sample['expected'];actual=ref.evaluate(cores,sample['points']);b=int(sample['born_count'])
        def rel(x,y):
            den=float(np.linalg.norm(y));return float(np.linalg.norm(x-y)/den) if den else None
        # Equal stratified samples from |psi|^2 / norm and uniform distribution.
        q=.5*expected/true_norm+.5*np.exp(-sum(np.log(manifest['physical_dims'])))
        estimate=float(np.sqrt(np.sum(np.abs(actual-expected)**2/q)/np.sum(expected**2/q)))
        d.update(probability_sum=[norm.real,norm.imag],norm_abs_error=float(abs(norm-true_norm)),
            born_sample_relative_l2=rel(actual[:b],expected[:b]),uniform_sample_relative_l2=rel(actual[b:],expected[b:]),importance_relative_l2_estimate=estimate,
            sample_min_probability=float(np.min(actual.real)),sample_max_imaginary=float(np.max(np.abs(actual.imag))),
            observable_and_sample_gate=abs(norm-true_norm)<=1e-8 and estimate<=1e-8,
            global_error_certified='full_grid_relative_l2' in d)
        if kind=='dmrg':
            d.update(hzz=[hzz.real,hzz.imag],hzz_abs_error=float(abs(hzz-true_hzz)))
            d['observable_and_sample_gate'] &= abs(hzz-true_hzz)<=1e-8
    if kind=='analytic':
        archive=np.load(folder/'reference-cores.npz');exact=[archive[f'v{i}'] for i in range(len(cores))];reference=json.loads((folder/'reference.json').read_text())
        d['relative_l2_vs_reference_tt']=ref.norm(ref.difference(cores,exact))/reference['norm']
        d['reference_svd_discarded_norms']=reference['svd_discarded_norms']
        samples=np.load(folder/'samples.npz');actual=ref.evaluate(cores,samples['points']);expected=samples['expected']
        d['relative_l2_vs_formula_uniform_sample']=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
        d['accuracy_pass']=d['relative_l2_vs_reference_tt']<=1e-8
        d['formula_sample_pass']=d['relative_l2_vs_formula_uniform_sample']<=1e-8
    return d

def attempt(worker,case,algorithm,seed,run_folder,phase='coverage',block=0,validation_cache=None):
    tag=f"{case['fixture']}-cap{case['cap']}-p{case['options'].get('oversampling',5)}-{algorithm}-seed{seed}-{phase}{block}"
    if set(case['options'])-{'oversampling'}:
        tag+='-opts'+hashlib.sha256(json.dumps(case['options'],sort_keys=True).encode()).hexdigest()[:12]
    folder=OUT/'fixtures'/case['fixture'];output=run_folder/(tag+'.bin');options=run_folder/(tag+'.options.json');save(options,case['options'])
    if output.exists():raise ValueError(f'refusing to overwrite previous output: {tag}')
    record=dict(case=case,algorithm=algorithm,seed=seed,phase=phase,block=block,tag=tag)
    if not (folder/'fixture.json').exists():return dict(record,status='missing_fixture')
    events=[];started=time.monotonic();last=started;current='setup';peak=0
    with (run_folder/(tag+'.stderr')).open('w') as stderr:
        p=subprocess.Popen([str(worker),str(folder),str(case['cap']),str(seed),str(output),str(options)],stdout=subprocess.PIPE,stderr=stderr,text=True,bufsize=1)
        poll=selectors.DefaultSelector();poll.register(p.stdout,selectors.EVENT_READ);timeout=False
        while True:
            for key,_ in poll.select(.1):
                line=key.fileobj.readline()
                if not line:poll.unregister(key.fileobj);continue
                event=json.loads(line);events.append(event)
                if event['phase']=='algorithm_start':current='algorithm';last=time.monotonic()
                elif event['phase']=='algorithm_done':current='export';last=time.monotonic()
            try:
                for line in pathlib.Path(f'/proc/{p.pid}/status').read_text().splitlines():
                    if line.startswith('VmHWM:'):peak=max(peak,int(line.split()[1]))
            except FileNotFoundError:pass
            if time.monotonic()-last>TIMEOUT:p.kill();p.wait();timeout=True;break
            if p.poll() is not None and not poll.get_map():break
        poll.close();p.wait()
    record.update(events=events,wall_seconds=time.monotonic()-started,exit_code=p.returncode,peak_process_rss_kib=peak)
    done=next((e for e in events if e['phase']=='algorithm_done'),None)
    if done:record.update(seconds=done['seconds'],output_ranks=done['output_ranks'],diagnostics=done['diagnostics'])
    if timeout:return dict(record,status='timeout',timeout_phase=current)
    if p.returncode or not done:return dict(record,status='algorithm_error')
    try:
        key=(sha(folder/'inputs.bin'),sha(folder/'fixture.json'),sha(output),sha(output.with_suffix('.cores.json')))
        if validation_cache is not None and key in validation_cache:
            original_tag,value=validation_cache[key]
            record['validation']=dict(value);record['validation_reused_from']=original_tag
        else:
            record['validation']=validate(folder,output)
            if validation_cache is not None:validation_cache[key]=(tag,dict(record['validation']))
        record['status']='completed'
    except Exception as error:record.update(status='validation_error',validation_error=repr(error))
    return record

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--group',choices=['dmrg','functions','tree','all'],default='all');args=parser.parse_args()
    def group(case):
        if case['fixture'].startswith('dmrg'):return 'dmrg'
        if case['fixture'].startswith('complex-tree'):return 'tree'
        return 'functions'
    selected=[c for c in cases() if args.group=='all' or group(c)==args.group]
    directory=OUT/('run-'+args.group+'-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));directory.mkdir()
    cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu})
    algorithms=['treeaci','rsi']
    workers,evidence=checked_workers(OUT,dict(treeaci=BASELINE,rsi=ROOT),ROOT,algorithms)
    protocol=dict(cases=selected,seeds=SEEDS,baseline=MAIN_COMMIT,**evidence,algorithms=algorithms,phase='coverage',
        author_revision='153b25a8aa059d0147b45955d0842b2f32fa5d1d',
        local_tolerance=1e-12,default_options='ACI 20 maximum/2 minimum sweeps, enabled global guard; RSI default oversampling5 unless overridden',
        acceptance='global relative L2 1e-8 when dense or bounded reference TT available; DMRG large n has separate observable/sample gate, not a global norm certificate',
        timeout_per_phase=TIMEOUT,affinity=[cpu],threads=1,selection='all fixed cases, seeds and failures retained; curves deliberately include insufficient caps',
        timing='coverage: one call per seed; no stable speedup claims; preparation, serialization and independent validation excluded from timer',
        source_files=snapshot_harness(directory,pathlib.Path(__file__).parent))
    save(directory/'protocol.json',protocol)
    print('DIRECTORY',directory,flush=True)
    count=0
    with (directory/'observations.jsonl').open('w') as file:
        for i,case in enumerate(selected):
            for seed in SEEDS:
                order=['treeaci','rsi'] if (i+seed)%2 else ['rsi','treeaci']
                for algorithm in order:
                    result=attempt(workers[algorithm],case,algorithm,seed,directory);file.write(json.dumps(result)+'\n');file.flush();count+=1
                    print('RESULT',count,case['fixture'],case['cap'],case['options'],algorithm,seed,result['status'],result.get('seconds'),result.get('validation',{}),flush=True)
    save(directory/'completion.json',dict(observations=count,expected=len(selected)*6,candidate_source_unchanged=workers_unchanged(OUT,evidence['worker_builds'],ROOT) and harness_unchanged(pathlib.Path(__file__).parent,protocol['source_files'])))
    print('FINISHED',directory,flush=True)
if __name__=='__main__':main()
