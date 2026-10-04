"""Pinned author inputs and independently checked native-chain fixtures."""
from pathlib import Path
import json,hashlib,subprocess
import h5py
import reference as ref
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'target/tree-rsi/paper-coverage'
MAIN_COMMIT='b881f39d9e72e32c43b9841fe10433fa5f63b167'
AUTHOR=OUT/'author'

def save(path,data):path.write_text(json.dumps(data,indent=2)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write_fixture(name,operands,metadata):
    folder=OUT/'fixtures'/name;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'fixture.json').exists():raise RuntimeError(f'refusing to overwrite fixture {name}')
    n=len(operands[0]);entries=[];offset=0;chunks=[]
    for cores in operands:
        rows=[]
        for v,a in enumerate(cores):
            neighbors=([v-1] if v else [])+([v+1] if v<n-1 else [])
            block=np.moveaxis(a,1,-1)
            if v==0:block=block[0]
            if v==n-1:block=block[...,0,:]
            values=block.ravel(order='F').astype('<c16')
            rows.append(dict(neighbors=neighbors,shape=list(block.shape),offset=offset,len=values.size))
            chunks.append(values);offset+=values.size
        entries.append(rows)
    np.concatenate(chunks).tofile(folder/'inputs.bin')
    manifest=dict(n=n,root=n-1,physical_dims=[a.shape[1] for a in operands[0]],operands=entries,dtype='c64' if any(np.iscomplexobj(a) and np.any(a.imag) for op in operands for a in op) else 'f64',metadata=metadata,source_sha256=sha(folder/'inputs.bin'))
    save(folder/'fixture.json',manifest)
    np.savez(folder/'input-cores.npz',**{f'a{j}_v{i}':a for j,op in enumerate(operands) for i,a in enumerate(op)})
    return folder

def read_export(binary,metadata=None):
    if metadata is None:metadata=json.loads(binary.with_suffix('.cores.json').read_text())
    values=np.fromfile(binary,dtype='<c16');n=len(metadata);cores=[]
    for v,m in enumerate(metadata):
        block=values[m['offset']:m['offset']+m['len']].reshape(m['shape'],order='F')
        if v==0:block=block[None,...]
        if v==n-1:block=block[:,None,:]
        cores.append(np.moveaxis(block,-1,1))
    if max(np.max(np.abs(a.imag)) for a in cores)==0:cores=[a.real.copy() for a in cores]
    return cores

def read_inputs(folder):
    m=json.loads((folder/'fixture.json').read_text())
    return [read_export(folder/'inputs.bin',op) for op in m['operands']]

def dmrg(n,chi):
    path=AUTHOR/f'datasets/itensor_dmrg_mps/n{n}_system/psi_maxdim{chi}_n{n}.h5'
    with h5py.File(path) as f:
        actual_n=int(f['num_sites'][()]);cores=[f[f'tensor_{i}'][()] for i in range(actual_n,0,-1)]
        cores[0]=cores[0].reshape(1,*cores[0].shape);cores[-1]=cores[-1].reshape(*cores[-1].shape,1)
        original_hzz=float(f['energy_diag'][()])
    actual_chi=max(a.shape[2] for a in cores)
    if actual_n!=n or actual_chi!=chi:raise ValueError(f'mislabeled author data: {path}: actual n={actual_n},chi={actual_chi}')
    norm,hzz=ref.wavefunction_observables(cores)
    if abs(norm-1)>1e-10 or abs(hzz-original_hzz)>1e-9:raise ValueError(f'author state reference mismatch {norm} {hzz} {original_hzz}')
    folder=write_fixture(f'dmrg-n{n}-chi{chi}',[cores,[a.conj() for a in cores]],dict(kind='dmrg',source=str(path.relative_to(OUT)),source_sha256=sha(path),norm=norm,hzz=hzz,file_hzz=original_hzz,actual_input_max_bond=actual_chi))
    rng=np.random.default_rng(20261001)
    born=ref.born_samples(cores,1024,20261001);uniform=rng.integers(0,3,size=(1024,n))
    points=np.concatenate([born,uniform]);true=np.abs(ref.evaluate(cores,points))**2
    np.savez(folder/'samples.npz',points=points,expected=true,born_count=1024)
    if n<=10:
        full=ref.dense(cores);np.save(folder/'expected-dense.npy',np.abs(full)**2)
    print('FIXTURE',folder.name,'norm',norm,'Hzz',hzz,flush=True)
    return folder

def formula(config,x):
    name=config['function']
    if name=='gaussian':return np.exp(-(x-config['mu'])**2/(2*config['sigma']**2))
    if name=='osc1':return np.cos(1024*x)*np.exp(-x*x)+4*np.exp(x)-3*x*x+10*x
    if name=='osc2':return np.sin(1024*x)*(np.exp(x*x)+5*x+2)-4*x
    if name=='relu_input':return np.cos(10*x)*np.sin(32*x)*(np.exp(x*x)+5*x+2)-4*x
    raise ValueError(name)

def analytic_input(name,config):
    folder=OUT/'prepared';folder.mkdir(exist_ok=True);file=folder/(name+'.bin');config_file=folder/(name+'.json');save(config_file,config)
    with (folder/(name+'.log')).open('w') as log:
        subprocess.run([str(OUT/'prepare/worker'),str(config_file),str(file)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    cores=read_export(file)
    rng=np.random.default_rng(8837);n=config['n'];points=rng.integers(0,2,size=(4096,n));x=points@(2.0**-np.arange(1,n+1));exact=formula(config,x);actual=ref.evaluate(cores,points)
    error=float(np.linalg.norm(actual-exact)/np.linalg.norm(exact))
    save(folder/(name+'-validation.json'),dict(relative_l2_uniform_sample=error,points=4096,seed=8837,actual_max_bond=max(a.shape[2] for a in cores),config=config))
    print('ANALYTIC INPUT',name,'error',error,'rank',max(a.shape[2] for a in cores),flush=True)
    return cores,error

def analytic_product(name,operands,configs,errors):
    folder=write_fixture(name,operands,dict(kind='analytic',input_configs=configs,input_sample_errors=errors))
    expected=operands[0];discarded=[]
    # The independent reference is built once, never used as an RSI/ACI input.
    # QR/SVD compression keeps the reference bounded; record every truncation.
    for op in operands[1:]:
        exact=ref.product(expected,op,max_total_elements=10_000_000)
        expected,loss=ref.round_chain(exact,128,1e-15);discarded.append(loss)
    np.savez(folder/'reference-cores.npz',**{f'v{i}':a for i,a in enumerate(expected)})
    rng=np.random.default_rng(20261001);n=len(expected);points=rng.integers(0,2,size=(8192,n));x=points@(2.0**-np.arange(1,n+1))
    truth=np.ones(len(x))
    for cfg in configs:truth*=formula(cfg,x)
    np.savez(folder/'samples.npz',points=points,expected=truth)
    save(folder/'reference.json',dict(norm=ref.norm(expected),svd_discarded_norms=discarded,construction='bounded exact local product then QR/SVD at 1e-15, reference cap 128; never algorithm input',max_rank=max(a.shape[2] for a in expected)))
    print('FIXTURE',name,'reference rank',max(a.shape[2] for a in expected),flush=True)

def prepare_all():
    for n,chis in [(10,[20]),(20,[10,15,20,25,30]),(50,[20,40,60,80,100,150])]:
        for chi in chis:dmrg(n,chi)
    for pair in [(.4,.6,.15),(.25,.75,.15),(.1,.9,.15),(.49,.51,.01)]:
        a,b,sigma=pair;inputs=[];configs=[];errors=[]
        for mu in (a,b):
            config=dict(function='gaussian',mu=mu,sigma=sigma,n=25,cap=10);op,err=analytic_input(f'gaussian-{mu}-{sigma}',config)
            inputs.append(op);configs.append(config);errors.append(err)
        prefix=f'gaussian-{a}-{b}-{sigma}'
        analytic_product(prefix,inputs,configs,errors)
        if pair==(.4,.6,.15):
            for indices in ([0,1,1],[0,0,1,1]):analytic_product(prefix+f'-m{len(indices)}',[inputs[i] for i in indices],[configs[i] for i in indices],[errors[i] for i in indices])
    inputs=[];configs=[];errors=[]
    for name in ['osc1','osc2']:
        cfg=dict(function=name,n=25,cap=10);op,err=analytic_input(name,cfg);inputs.append(op);configs.append(cfg);errors.append(err)
    for indices in ([0,1],[0,1,1],[0,0,1,1]):analytic_product(f'oscillatory-m{len(indices)}',[inputs[i] for i in indices],[configs[i] for i in indices],[errors[i] for i in indices])
    path=AUTHOR/'datasets/qtensor_well/active_matter_t50_Dxx.hdf5'
    with h5py.File(path) as file:full=file['quantics_tensor'][()].reshape(-1)
    cores=ref.dense_to_chain(full,[2]*16,20);represented=ref.dense(cores)
    folder=write_fixture('active-matter-Dxx',[cores,cores],dict(kind='active_matter',source_sha256=sha(path),input_relative_l2=float(np.linalg.norm(represented-full)/np.linalg.norm(full)),input_preparation='independent bounded SVD, cap20; original author input QTT not published'))
    np.save(folder/'expected-dense.npy',represented**2);np.save(folder/'physical-dense.npy',full**2)
    print('FIXTURE active-matter-Dxx input error',np.linalg.norm(represented-full)/np.linalg.norm(full),flush=True)
if __name__=='__main__':prepare_all()
