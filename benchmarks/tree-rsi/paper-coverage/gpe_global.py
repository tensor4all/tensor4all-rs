"""All 2^26 GPE entries, using bounded half-chain frames and blocked GEMM.

No contraction is repeated independently for each physical point. Exact
probabilities are generated once and memory-mapped for all output comparisons.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,time
import numpy as np
from fixtures import OUT,read_inputs,read_export,save,sha

def frames(cores,cut):
    left_size=int(np.prod([a.shape[1] for a in cores[:cut]],dtype=object))
    right_size=int(np.prod([a.shape[1] for a in cores[cut:]],dtype=object))
    rank=cores[cut-1].shape[2]
    if (left_size+right_size)*rank*16>512*1024**2:raise ValueError('half-frame memory bound exceeds512MiB')
    left=np.ones((1,1),dtype=cores[0].dtype)
    for a in cores[:cut]:left=(left@a.reshape(a.shape[0],-1)).reshape(-1,a.shape[2])
    right=np.ones((1,1),dtype=cores[0].dtype)
    for a in cores[cut:][::-1]:right=(a.reshape(-1,a.shape[2])@right).reshape(a.shape[0],-1)
    return left,right

def main():
    folders=[sorted(OUT.glob(pattern))[-1] for pattern in ('run-functions-*','supplement-*','gpe-tolerance-*')]
    if not all((f/'completion.json').exists() for f in folders):raise RuntimeError('finish all GPE measurements first')
    fixture=OUT/'fixtures/gpe-91-vortices';psi=read_inputs(fixture)[0];left,right=frames(psi,13)
    directory=OUT/'gpe-global';directory.mkdir(exist_ok=True);truth_path=directory/'probabilities.npy';block=128
    if not truth_path.exists():
        temporary=directory/'probabilities.partial.npy'
        truth=np.lib.format.open_memmap(temporary,mode='w+',dtype='<f8',shape=(left.shape[0],right.shape[1]))
        for start in range(0,len(left),block):truth[start:start+block]=np.abs(left[start:start+block]@right)**2
        truth.flush();del truth;temporary.replace(truth_path)
    if (directory/'protocol.json').exists():
        prior=json.loads((directory/'protocol.json').read_text())
        if prior['input_sha256']!=sha(fixture/'inputs.bin') or prior['truth_sha256']!=sha(truth_path):raise RuntimeError('reference cache provenance mismatch')
    truth=np.load(truth_path,mmap_mode='r');norm2=sum(float(np.sum(truth[i:i+block]**2)) for i in range(0,len(truth),block))
    mass=sum(float(np.sum(truth[i:i+block])) for i in range(0,len(truth),block))
    if abs(mass-1)>1e-11:raise ValueError('full-grid probability normalization mismatch')
    save(directory/'protocol.json',dict(physical_entries=int(truth.size),cut=13,rows_per_block=block,
        input_sha256=sha(fixture/'inputs.bin'),truth_sha256=sha(truth_path),probability_sum=mass,
        reference='exact double-precision contraction of input wavefunction, squared magnitudes; no TT reference truncation',
        workspace='half frames together limited to512MiB; probability cache512MiB memory-mapped; blocks128x8192',
        timing='independent validation only; never included in algorithm timing'))
    cache={}
    for folder in folders:
        results=[]
        for row in map(json.loads,(folder/'observations.jsonl').read_text().splitlines()):
            if row['case']['fixture']!='gpe-91-vortices' or row['status']!='completed':continue
            binary=folder/(row['tag']+'.bin');key=(sha(binary),sha(binary.with_suffix('.cores.json')));started=time.monotonic()
            if key in cache:result=dict(cache[key]);result['validation_reused_from']=result['tag']
            else:
                a,b=frames(read_export(binary),13);error2=0.;maxabs=0.
                for start in range(0,len(a),block):
                    delta=a[start:start+block]@b-truth[start:start+block]
                    error2+=float(np.vdot(delta.ravel(),delta.ravel()).real);maxabs=max(maxabs,float(np.max(np.abs(delta))))
                error=float(np.sqrt(error2/norm2))
                result=dict(full_grid_relative_l2=error,full_grid_maxabs_error=maxabs,gate_1e8=error<=1e-8,
                    physical_entries=int(truth.size),reference_relative_truncation=0.,seconds=time.monotonic()-started)
            result.update(tag=row['tag'],algorithm=row['algorithm'],seed=row['seed'],case=row['case']);cache[key]=result;results.append(result)
            print('GPE GLOBAL',row['tag'],result['full_grid_relative_l2'],time.monotonic()-started,flush=True)
            # Preserve completed global checks even if a later case is interrupted.
            (folder/'gpe-global-checks.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
    save(directory/'completion.json',dict(unique_outputs=len(cache),physical_entries=int(truth.size)))
if __name__=='__main__':main()
