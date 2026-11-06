"""Complete fixture coverage without dense 2^25 grids or executable pickle globals."""
import pathlib,pickle,json,math
import numpy as np
import h5py
import reference as ref
from fixtures import OUT,AUTHOR,write_fixture,analytic_input,analytic_product,read_export,formula,save,sha


def narrow_gaussian(mu,sigma,n=25,cap=10):
    # Independent real Fourier representation of the periodized Gaussian.
    # Here the omitted period images are < exp(-1200) on [0,1); Fourier
    # frequencies above 160 have negligible tails for sigma=0.01.
    modes=160;k=np.arange(1,modes+1);amp=np.sqrt(2*np.pi)*sigma*np.exp(-.5*(2*np.pi*sigma*k)**2)
    width=2*modes+1;first=np.zeros(width);first[0]=np.sqrt(2*np.pi)*sigma
    first[1::2]=2*amp*np.cos(-2*np.pi*k*mu);first[2::2]=2*amp*np.sin(-2*np.pi*k*mu)
    last=np.zeros(width);last[0]=1;last[1::2]=1;cores=[]
    for site in range(n):
        a=np.zeros((width,2,width));a[:,0,:]=np.eye(width);a[0,1,0]=1
        theta=2*np.pi*k*2.0**(-site-1);cs=np.cos(theta);sn=np.sin(theta);i=2*np.arange(modes)+1
        a[i,1,i]=cs;a[i,1,i+1]=sn;a[i+1,1,i]=-sn;a[i+1,1,i+1]=cs
        if site==0:a=np.einsum('a,asb->sb',first,a)[None,:,:]
        if site==n-1:a=np.einsum('asb,b->as',a,last)[:,:,None]
        cores.append(a)
    rounded,discarded=ref.round_chain(cores,cap)
    cfg=dict(function='gaussian',mu=mu,sigma=sigma,n=n,cap=cap)
    rng=np.random.default_rng(8837);points=rng.integers(0,2,size=(8192,n));x=points@(2.0**-np.arange(1,n+1));true=formula(cfg,x)
    error=float(np.linalg.norm(ref.evaluate(rounded,points)-true)/np.linalg.norm(true))
    save(OUT/f'narrow-{mu}-preparation.json',dict(config=cfg,method='real Fourier rotation cores, frequencies0..160; QR/SVD cap10; no full physical grid',discarded_norm=discarded,formula_sample_relative_l2=error))
    print('NARROW INPUT',mu,error,flush=True)
    return rounded,error


def complete_functions():
    inputs=[];configs=[];errors=[]
    for mu in [.49,.51]:
        op,err=narrow_gaussian(mu,.01);inputs.append(op);errors.append(err);configs.append(dict(function='gaussian',mu=mu,sigma=.01,n=25,cap=10))
    analytic_product('gaussian-0.49-0.51-0.01',inputs,configs,errors)
    inputs=[];configs=[];errors=[]
    for name in ['osc1','osc2']:
        cfg=dict(function=name,n=25,cap=10);op,err=oscillatory_input(cfg);inputs.append(op);configs.append(cfg);errors.append(err)
    for ids in ([0,1],[0,1,1],[0,0,1,1]):analytic_product(f'oscillatory-m{len(ids)}',[inputs[i] for i in ids],[configs[i] for i in ids],[errors[i] for i in ids])
    path=AUTHOR/'datasets/qtensor_well/active_matter_t50_Dxx.hdf5'
    with h5py.File(path) as f:full=f['quantics_tensor'][()].reshape(-1)
    cores=ref.dense_to_chain(full,[2]*16,20);represented=ref.dense(cores)
    folder=write_fixture('active-matter-Dxx',[cores,cores],dict(kind='active_matter',source_sha256=sha(path),input_relative_l2=float(np.linalg.norm(represented-full)/np.linalg.norm(full)),input_preparation='independent bounded SVD cap20; author input QTT not published'))
    np.save(folder/'expected-dense.npy',represented**2);np.save(folder/'physical-dense.npy',full**2)
    print('ACTIVE MATTER INPUT',np.linalg.norm(represented-full)/np.linalg.norm(full),flush=True)


def polynomial(coefficients,n=25):
    """Exact additive-bit polynomial representation, no full grid."""
    width=len(coefficients);cores=[]
    for site in range(n):
        delta=2.0**(-site-1);a=np.zeros((width,2,width));a[:,0,:]=np.eye(width)
        for j in range(width):
            for k in range(j,width):a[j,1,k]=math.comb(k,j)*delta**(k-j)
        if site==0:a=a[:1]
        if site==n-1:a=np.einsum('asb,b->as',a,coefficients)[:,:,None]
        cores.append(a)
    return ref.round_chain(cores,width,1e-15)[0]


def oscillatory_input(cfg):
    # Build a Taylor/rotation TT, then enforce cap10.
    name=cfg['function'];n=cfg['n'];c=np.zeros(61)
    for k in range(31):c[2*k]=((-1)**k if name=='osc1' else 1)/math.factorial(k)
    if name=='osc2':c[0]+=2;c[1]+=5
    envelope=polynomial(c,n);rotation=[]
    for site in range(n):
        t=1024*2.0**(-site-1);a=np.stack([np.eye(2),[[np.cos(t),np.sin(t)],[-np.sin(t),np.cos(t)]]],axis=1)
        if site==0:a=a[:1]
        if site==n-1:a=a[:,:,(0 if name=='osc1' else 1):][:,:,:1]
        rotation.append(a)
    oscillation=ref.product(envelope,rotation)
    c=np.zeros(31)
    if name=='osc1':
        for k in range(31):c[k]=4/math.factorial(k)
        c[1]+=10;c[2]-=3
    else:c[1]=-4
    smooth=polynomial(c,n);negative=[a.copy() for a in smooth];negative[0]*=-1
    result,discarded=ref.round_chain(ref.difference(oscillation,negative),10)
    points=np.random.default_rng(8837).integers(0,2,size=(8192,n));x=points@(2.0**-np.arange(1,n+1));true=formula(cfg,x)
    error=float(np.linalg.norm(ref.evaluate(result,points)-true)/np.linalg.norm(true))
    save(OUT/f'{name}-preparation.json',dict(config=cfg,method='Taylor through degree60 for exp(+/-x^2), degree30 for exp(x); exact trigonometric rotations; QR/SVD cap10',discarded_norm=discarded,formula_sample_relative_l2=error))
    print('OSC INPUT',name,error,flush=True)
    return result,error


class ArrayOnlyUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        allowed={('numpy','ndarray'):np.ndarray,('numpy','dtype'):np.dtype,
                 ('numpy.core.multiarray','_reconstruct'):np._core.multiarray._reconstruct,
                 ('numpy._core.multiarray','_reconstruct'):np._core.multiarray._reconstruct}
        if (module,name) not in allowed:raise ValueError(f'unsupported pickle global {module}.{name}')
        return allowed[module,name]
    def persistent_load(self,pid):raise ValueError('persistent references prohibited')


def gpe():
    path=AUTHOR/'gpe_density/91v.pkl'
    with path.open('rb') as file:cores=ArrayOnlyUnpickler(file).load()
    if not isinstance(cores,list) or not 1<len(cores)<=100:raise ValueError('invalid MPS list')
    for a in cores:
        if not isinstance(a,np.ndarray) or a.ndim!=3 or a.dtype.kind not in 'fc' or a.size>2**24 or not np.isfinite(a).all():raise ValueError('invalid MPS core')
    scale=ref.norm(cores);cores=[a.copy() for a in cores];cores[0]/=scale
    folder=write_fixture('gpe-91-vortices',[cores,[a.conj() for a in cores]],dict(kind='wavefunction',source_sha256=sha(path),original_norm=scale,normalization='divide first core by independently computed QR norm',norm=1.0,actual_input_max_bond=max(a.shape[2] for a in cores)))
    born=ref.born_samples(cores,1024,20261001);uniform=np.random.default_rng(20261001).integers(0,2,size=(1024,len(cores)));points=np.concatenate([born,uniform]);expected=np.abs(ref.evaluate(cores,points))**2
    np.savez(folder/'samples.npz',points=points,expected=expected,born_count=1024)
    print('GPE',len(cores),'actual rank',max(a.shape[2] for a in cores),'original norm',scale,flush=True)

if __name__=='__main__':
    complete_functions();gpe()
