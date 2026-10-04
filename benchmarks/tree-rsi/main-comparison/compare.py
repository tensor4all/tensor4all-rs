#!/usr/bin/env python3
"""Reproducible isolated main TreeACI / worktree RSI experiment. Outputs are ignored.

Run: python3 benchmarks/tree-rsi/main-comparison/compare.py
Requires numpy; builds release workers then runs the entire fixed case list.
"""
import os
# Set before importing numpy; one CPU for both providers and the independent oracle.
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS'):
    os.environ[key] = '1'
import argparse, datetime, hashlib, json, pathlib, platform, selectors, shutil, subprocess, sys, time
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[3]
SOURCE = pathlib.Path(__file__).resolve().parent
CASES = [('chain',16,128,128),('chain',16,128,256),('binary',19,128,128),('binary',19,128,256),('chain',18,256,256),('binary',19,256,256)]
SEEDS = [1,2,3]
TIMEOUT = 180

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path, value): path.write_text(json.dumps(value, indent=2)+'\n')
def command(args, **kwargs): return subprocess.check_output(args, cwd=ROOT, text=True, **kwargs).strip()
def fingerprint():
    h = hashlib.sha256()
    paths = [ROOT/'Cargo.toml', ROOT/'Cargo.lock']
    for crate in ('tensor4all-treersi','tensor4all-treetn','tensor4all-core','tensor4all-tensorbackend'):
        paths += sorted((ROOT/'crates'/crate).rglob('*.rs')) + [ROOT/'crates'/crate/'Cargo.toml']
    for p in paths: h.update(str(p.relative_to(ROOT)).encode()+b'\0'+p.read_bytes())
    return h.hexdigest()
def fixture(directory, topology, n, chi):
    directory.mkdir()
    pairs = [((v-1 if topology=='chain' else (v-1)//2),v) for v in range(1,n)]
    if topology == 'balanced':
        assert n % 2 == 0
        h=n//2
        pairs=[(0,h)]+[((v-1)//2+off,v+off) for off in (0,h) for v in range(1,h)]
    neighbors = [[] for _ in range(n)]
    for a,b in pairs: neighbors[a].append(b); neighbors[b].append(a)
    def component(v, parent):
        result=[v]
        for w in neighbors[v]:
            if w != parent: result += component(w,v)
        return result
    chis = [chi,chi] if isinstance(chi,int) else list(chi)
    edges = [(a,b,min(max(chis),2**min(len(component(b,a)),len(component(a,b))))) for a,b in pairs]
    dims = {(a,b):r for a,b,r in edges}
    bond_labels = {(a,b):n+i for i,(a,b,_) in enumerate(edges)}
    cut = max(pairs,key=lambda ab:min(len(component(ab[0],ab[1])),len(component(ab[1],ab[0]))))
    left = sorted(component(cut[1],cut[0])); right = sorted(set(range(n))-set(left))
    arrays=[]; operands=[]; dense=[]; offset=0; spectra=[]; paths=[]
    operand_edges=[]
    for operand_index, seed in enumerate((11,29)):
        dims = {(a,b):min(chis[operand_index],r) for a,b,r in edges}
        operand_edges.append([(a,b,r) for (a,b),r in dims.items()])
        rng=np.random.Generator(np.random.PCG64(seed)); cores=[]; args=[]
        for v in range(n):
            ns=sorted(neighbors[v]); shape=[dims[min(v,w),max(v,w)] for w in ns]+[2]
            children=[w for w in ns if w>v]
            scale=np.sqrt(np.prod([dims[min(v,w),max(v,w)] for w in children],dtype=np.float64))
            core=rng.standard_normal(shape)/scale
            flat=core.ravel(order='F').astype('<f8'); arrays.append(flat)
            cores.append(dict(neighbors=ns,shape=shape,offset=offset,len=len(flat))); offset+=len(flat)
            args.extend([core,[bond_labels[min(v,w),max(v,w)] for w in ns]+[v]])
        args.append(list(range(n)))
        path, info=np.einsum_path(*args,optimize='greedy'); paths.append(info)
        full=np.einsum(*args,optimize=path); dense.append(full)
        full.ravel(order='F').astype('<f8').tofile(directory/f'input{len(dense)-1}.bin')
        matrix=np.transpose(full,left+right).reshape(2**len(left),-1)
        s=np.linalg.svd(matrix,compute_uv=False)
        spectra.append(dict(numerical_rank_1e12=int(np.count_nonzero(s>s[0]*1e-12)), relative_singular_values=(s/s[0]).tolist()))
        operands.append(cores)
    np.concatenate(arrays).tofile(directory/'inputs.bin')
    expected=dense[0]*dense[1]; expected.ravel(order='F').astype('<f8').tofile(directory/'expected.bin')
    s=np.linalg.svd(np.transpose(expected,left+right).reshape(2**len(left),-1),compute_uv=False)
    total=np.linalg.norm(s)
    assert abs(total/np.linalg.norm(expected.ravel())-1)<1e-12
    manifest=dict(topology=topology,n=n,chi=chi,root=(n-1 if topology=='chain' else 0),edges=edges,operand_edges=operand_edges,operands=operands,
        generation='NumPy PCG64 normal(0,1), seeds 11 and 29; per-core division by sqrt(product child ranks)',
        cut=cut,cut_sites=left,input_cut_spectra=spectra,product_cut_rank_1e12=int(np.count_nonzero(s>s[0]*1e-12)),
        product_relative_singular_values=(s/total).tolist(),
        relative_l2_lower_bounds={str(cap):float(np.linalg.norm(s[cap:])/total) for cap in (128,256,512)},
        data_sha256=sha(directory/'inputs.bin'),expected_sha256=sha(directory/'expected.bin'),einsum_paths=paths)
    save(directory/'fixture.json',manifest)
    print('FIXTURE',directory.name,'input ranks',[v['numerical_rank_1e12'] for v in spectra], 'product rank',manifest['product_cut_rank_1e12'],'bounds',manifest['relative_l2_lower_bounds'],flush=True)

def build(directory, source, algorithm, target):
    directory.mkdir(exist_ok=True)
    if (directory/"build.log").exists(): shutil.copy2(directory/"build.log",directory/"previous-build.log")
    shutil.copy2(SOURCE/'worker.rs',directory/'worker.rs')
    shutil.copy2(source/'Cargo.lock',directory/'Cargo.lock')
    package=f'isolated-{algorithm}-benchmark'
    deps = '\n'.join(f'tensor4all-{name} = {{ path = "{source}/crates/tensor4all-{name}", default-features = false, features = ["tenferro-cpu-faer"] }}' for name in ('core','treetn'))
    deps += f'\ntensor4all-{("treersi" if algorithm=="rsi" else "treeaci")} = {{ path = "{source}/crates/tensor4all-{("treersi" if algorithm=="rsi" else "treeaci")}" }}'
    (directory/'Cargo.toml').write_text(f'''[workspace]
[package]
name = "{package}"
version = "0.0.0"
edition = "2024"
[features]
rsi = []
treeaci = []
[dependencies]
serde_json = "1"
{deps}
[[bin]]
name = "{package}"
path = "worker.rs"
[profile.release]
debug = 0
''')
    with (directory/'build.log').open('w') as log:
        subprocess.run(['cargo','build','--release','--offline','--manifest-path',str(directory/'Cargo.toml'),'--features',algorithm,'--target-dir',str(target)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    shutil.copy2(target/'release'/package,directory/'worker')
    return directory/'worker'

def error(actual, expected):
    if actual.shape!=expected.shape or not np.isfinite(actual).all(): return None
    return float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
def run(worker, fixture_dir, case, algorithm, seed, directory):
    stem=f'{case}-{algorithm}-seed{seed}'; output=directory/(stem+'.bin')
    record=dict(case=case,algorithm=algorithm,seed=seed)
    cap=int(case.split('cap')[-1]); events=[]; peak_kib=0
    start=time.monotonic(); phase_start=start; phase='setup'
    with (directory/(stem+'.stderr')).open('w') as log:
        p=subprocess.Popen([str(worker),str(fixture_dir),str(cap),str(seed),str(output)],stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
        selector=selectors.DefaultSelector(); selector.register(p.stdout,selectors.EVENT_READ)
        timed_out=False
        while True:
            for key,_ in selector.select(.05):
                line=key.fileobj.readline()
                if line:
                    event=json.loads(line); events.append(event)
                    print(stem,event['phase'],event.get('seconds',''),flush=True)
                    if event['phase']=='algorithm_start': phase='algorithm'; phase_start=time.monotonic()
                    elif event['phase']=='algorithm_done': phase='validation'; phase_start=time.monotonic()
                else: selector.unregister(key.fileobj)
            try:
                status=pathlib.Path(f'/proc/{p.pid}/status').read_text()
                for line in status.splitlines():
                    if line.startswith('VmHWM:'): peak_kib=max(peak_kib,int(line.split()[1]))
            except FileNotFoundError: pass
            if time.monotonic()-phase_start>TIMEOUT:
                timed_out=True; p.kill(); p.wait(); break
            if p.poll() is not None and not selector.get_map(): break
        selector.close(); p.wait()
    record.update(events=events,exit_code=p.returncode,wall_seconds=time.monotonic()-start,peak_process_rss_kib=peak_kib,timeout_phase=phase if timed_out else None)
    for i in range(2):
        path=output.with_suffix(f'.input{i}.bin')
        record[f'input{i}_relative_l2']=error(np.fromfile(path,dtype='<f8'),np.fromfile(fixture_dir/f'input{i}.bin',dtype='<f8')) if path.exists() else None
    done=next((e for e in events if e['phase']=='algorithm_done'),None)
    if done: record.update(seconds=done['seconds'],diagnostics=done['diagnostics'],output_ranks=done['output_ranks'])
    if output.exists() and not timed_out and p.returncode==0:
        actual=np.fromfile(output,dtype='<f8'); expected=np.fromfile(fixture_dir/'expected.bin',dtype='<f8')
        record['relative_l2']=error(actual,expected)
        record['relative_maxabs']=float(np.max(np.abs(actual-expected))/np.max(np.abs(expected))) if actual.shape==expected.shape and np.isfinite(actual).all() else None
        record['output_sha256']=sha(output)
        record['status']='completed'
    else: record['status']='timeout' if timed_out else 'error'
    record['accuracy_pass']=record.get('relative_l2') is not None and record['relative_l2']<=1e-8
    record['input_validation_pass']=all(record[f'input{i}_relative_l2'] is not None and record[f'input{i}_relative_l2']<=1e-12 for i in range(2))
    return record

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--directory',type=pathlib.Path); parser.add_argument('--resume-setup',action='store_true'); args=parser.parse_args()
    directory=args.directory or ROOT/'target'/'tree-rsi'/('main-comparison-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    directory=directory.resolve(); directory.mkdir(parents=True,exist_ok=args.resume_setup)
    if args.resume_setup and (directory/"observations.jsonl").exists(): raise RuntimeError("cannot resume setup after observations")
    cpu=min(os.sched_getaffinity(0)); os.sched_setaffinity(0,{cpu})
    baseline=command(['git','rev-parse','origin/main']); before=fingerprint()
    protocol=dict(baseline=baseline,candidate_head=command(['git','rev-parse','HEAD']),candidate_source_sha256=before,
        worker_sha256=sha(SOURCE/'worker.rs'),driver_sha256=sha(SOURCE/'compare.py'),cases=CASES,seeds=SEEDS,
        repetitions='one public call per seed, 3 seeds per case; descriptive timing, no speedup significance claim',
        local_tolerance=1e-12,full_grid_relative_l2_gate=1e-8,timeout_seconds_per_phase=TIMEOUT,
        warmup='input dense contractions initialize backend before measured product call',
        treeaci='main defaults except cap, seed, root, tolerance: max_sweeps=20, min_sweeps=2, global guard enabled',
        rsi='worktree defaults except cap, seed, root, rel_tol: k=ceil(cap/2)+5; no global certificate',
        ordering='case order fixed; paired algorithm order alternates with case index + seed',
        timing_region='public hadamard_many call including RNG; fixture construction, dense export and validation excluded',
        accuracy_selection='all cases and errors recorded; no rejection before timing and no retuning',
        numeric_backend='Rust tenferro CPU faer, f64; independent NumPy dense contraction and SVD',
        affinity=[cpu],threads={k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS')},
        host=platform.uname()._asdict(),numpy=np.__version__,rustc=command(['rustc','--version']),
        git_status=command(['git','status','--short']),intermediate_rank_scope='ACI sweep maxima are observations, not intra-sweep peaks; RSI edge construction ranks and matrix shapes recorded')
    save(directory/'protocol.json',protocol); print('DIRECTORY',directory,flush=True)
    # Snapshot source without touching branch, index, or old worktrees.
    baseline_dir=directory/'main'; baseline_dir.mkdir(exist_ok=args.resume_setup)
    archive=directory/'main.tar'
    with archive.open('wb') as file: subprocess.run(['git','archive',baseline],cwd=ROOT,stdout=file,check=True)
    subprocess.run(['tar','-xf',str(archive),'-C',str(baseline_dir)],check=True); archive.unlink()
    # Allow build parallelism on available CPUs; restore the fixed benchmark CPU afterwards.
    available=set(range(os.cpu_count())); os.sched_setaffinity(0,available)
    workers={}
    for algorithm,source in [('treeaci',baseline_dir),('rsi',ROOT)]:
        print('BUILD',algorithm,flush=True)
        workers[algorithm]=build(directory/algorithm,source,algorithm,ROOT/'target')
    os.sched_setaffinity(0,{cpu})
    save(directory/'binaries.json',{a:dict(sha256=sha(p),lock_sha256=sha(p.parent/'Cargo.lock')) for a,p in workers.items()})
    for topology,n,chi,cap in CASES:
        path=directory/f'{topology}-n{n}-chi{chi}'
        if not path.exists(): fixture(path,topology,n,chi)
    with (directory/'observations.jsonl').open('w') as out:
        for i,(topology,n,chi,cap) in enumerate(CASES):
            fixture_dir=directory/f'{topology}-n{n}-chi{chi}'; case=f'{fixture_dir.name}-cap{cap}'
            for seed in SEEDS:
                order=['treeaci','rsi'] if (i+seed)%2 else ['rsi','treeaci']
                for algorithm in order:
                    record=run(workers[algorithm],fixture_dir,case,algorithm,seed,directory)
                    out.write(json.dumps(record)+'\n'); out.flush()
                    print('RESULT',case,algorithm,seed,record['status'],record.get('seconds'),record.get('relative_l2'),flush=True)
    save(directory/'completion.json',dict(source_unchanged=before==fingerprint(),observations=36,completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print('FINISHED',directory,flush=True)
if __name__=='__main__': main()
