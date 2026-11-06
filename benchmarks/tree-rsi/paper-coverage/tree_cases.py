"""Independent full-grid oracles for genuine branching, complex high-rank trees."""
import json, pathlib
import numpy as np
from fixtures import OUT, save, sha


def materialize(cores, metadata):
    n=len(cores)
    if n>20:raise ValueError('full tree oracle limited to 2^20 entries')
    edges=sorted({tuple(sorted((v,w))) for v,m in enumerate(metadata) for w in m['neighbors']})
    labels={edge:n+i for i,edge in enumerate(edges)};args=[]
    for v,(a,m) in enumerate(zip(cores,metadata)):
        args.extend([a,[labels[tuple(sorted((v,w)))] for w in m['neighbors']]+[v]])
    args.append(list(range(n)))
    path,_=np.einsum_path(*args,optimize='greedy')
    # Explicitly bound every intermediate chosen by numpy, including physical
    # open indices; never assume a small final tensor implies small workspace.
    dimensions={v:2 for v in range(n)}
    for a,m in zip(cores,metadata):
        for axis,w in enumerate(m['neighbors']):
            v=m['node'];dimensions[labels[tuple(sorted((v,w)))]]=a.shape[axis]
    remaining=[set(args[i]) for i in range(1,len(args)-1,2)]
    for positions in path[1:]:
        selected=[remaining[i] for i in positions];other=[s for i,s in enumerate(remaining) if i not in positions]
        union=set.union(*selected);needed=set(range(n)).union(*other)
        result=union & needed
        size=np.prod([dimensions[k] for k in result],dtype=object)
        if size>16_777_216:raise ValueError(f'oracle intermediate too large: {size}')
        remaining=other+[result]
    return np.einsum(*args,optimize=path)


def read_blocks(binary,metadata):
    flat=np.fromfile(binary,dtype='<c16')
    return [flat[m['offset']:m['offset']+m['len']].reshape(m['shape'],order='F') for m in metadata]


def validate(folder,output):
    metadata=json.loads(output.with_suffix('.cores.json').read_text())
    for v,m in enumerate(metadata):m['node']=v
    cores=read_blocks(output,metadata);expected=np.load(folder/'expected-dense.npy')
    if not all(np.isfinite(a).all() for a in cores):return dict(finite=False,accuracy_pass=False)
    actual=materialize(cores,metadata);error=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
    return dict(validation_scope='full branching-tree grid',finite=True,full_grid_relative_l2=error,
                full_grid_relative_maxabs=float(np.max(np.abs(actual-expected))/np.max(np.abs(expected))),
                accuracy_pass=error<=1e-8,output_sha256=sha(output),
                actual_max_rank=max(max(m['shape'][:-1]) for m in metadata))


def prepare():
    n=20;half=n//2
    pairs=[(0,half)]+[((v-1)//2+off,v+off) for off in (0,half) for v in range(1,half)]
    neighbors=[[] for _ in range(n)]
    for a,b in pairs:neighbors[a].append(b);neighbors[b].append(a)
    def size(v,parent):return 1+sum(size(w,v) for w in neighbors[v] if w!=parent)
    for count,chi,roots in [(2,16,[0,7,19]),(4,4,[0,19])]:
        chunks=[];operands=[];offset=0;expected=np.ones([2]*n,dtype=complex)
        for operand in range(count):
            rng=np.random.default_rng(61001+operand);cores=[];metadata=[]
            for v in range(n):
                ns=sorted(neighbors[v]);shape=[min(chi,2**min(size(v,w),size(w,v))) for w in ns]+[2]
                core=(rng.standard_normal(shape)+1j*rng.standard_normal(shape))/np.sqrt(np.prod(shape))
                cores.append(core);metadata.append(dict(node=v,neighbors=ns,shape=shape))
            full=materialize(cores,metadata);scale=np.linalg.norm(full)/np.sqrt(full.size)
            cores[0]/=scale;full/=scale;expected*=full
            for a,m in zip(cores,metadata):
                flat=a.ravel(order='F').astype('<c16');m.update(offset=offset,len=len(flat));offset+=len(flat);chunks.append(flat)
            operands.append(metadata)
        # Central edge splits the graph into two ten-site branches.
        s=np.linalg.svd(expected.reshape(1024,1024),compute_uv=False)
        meta=dict(kind='tree',input_chi=chi,operand_count=count,
                  generation='independent complex Gaussian entries, seeds61001+operand; normalized full-grid input RMS=1',
                  central_cut_rank_1e12=int(np.count_nonzero(s>s[0]*1e-12)),
                  rank_error_lower_bounds={str(k):float(np.linalg.norm(s[k:])/np.linalg.norm(s)) for k in (128,256,512)})
        for root in roots:
            folder=OUT/'fixtures'/f'complex-tree-n20-m{count}-chi{chi}-root{root}';folder.mkdir(exist_ok=True)
            np.concatenate(chunks).tofile(folder/'inputs.bin');np.save(folder/'expected-dense.npy',expected)
            save(folder/'fixture.json',dict(n=n,root=root,physical_dims=[2]*n,operands=operands,dtype='c64',metadata=meta,source_sha256=sha(folder/'inputs.bin')))
            print('TREE FIXTURE',folder.name,meta,flush=True)


if __name__=='__main__':prepare()
