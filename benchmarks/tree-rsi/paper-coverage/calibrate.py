"""Rank feasibility and sample calibration, computed after measured calls."""
import json
import numpy as np
import reference as ref
from fixtures import OUT,save

def spectrum(cores,cut):
    work=[a.copy() for a in cores]
    for i in range(len(work)-1,0,-1):
        a=work[i];q,r=np.linalg.qr(a.reshape(a.shape[0],-1).conj().T,mode='reduced')
        work[i]=q.conj().T.reshape(q.shape[1],a.shape[1],a.shape[2]);work[i-1]=np.tensordot(work[i-1],r.conj().T,axes=(2,0))
    carry=np.ones((1,1),dtype=work[0].dtype)
    for a in work[:cut]:
        block=(carry@a.reshape(a.shape[0],-1)).reshape(-1,a.shape[2]);_,carry=np.linalg.qr(block,mode='reduced')
    return np.linalg.svd(carry,compute_uv=False)

def main():
    folder=sorted(OUT.glob('run-dmrg-*'))[-1]
    rows=list(map(json.loads,(folder/'observations.jsonl').read_text().splitlines()))
    globals={r['tag']:r for r in map(json.loads,(folder/'global-checks.jsonl').read_text().splitlines())}
    calibration=[]
    for row in rows:
        if row['status']!='completed':continue
        v=row['validation'];g=globals.get(row['tag']);true=g['relative_l2_vs_reference_tt'] if g else v.get('full_grid_relative_l2')
        if true is None:continue
        estimate=v['importance_relative_l2_estimate']
        calibration.append(dict(tag=row['tag'],global_error=true,importance_estimate=estimate,estimate_over_global=estimate/true if true else None,
            observable_and_sample_gate=v['observable_and_sample_gate'],global_gate=true<=1e-8))
    save(folder/'sample-calibration.json',calibration)
    rank_bounds={}
    for chi in (10,15,20,25,30):
        f=OUT/'fixtures'/f'dmrg-n20-chi{chi}';archive=np.load(f/'reference-cores.npz');cores=[archive[f'v{i}'] for i in range(20)]
        s=spectrum(cores,10);total=np.linalg.norm(s)
        if abs(total/ref.norm(cores)-1)>1e-10:raise ValueError('Schmidt spectrum norm mismatch')
        caps=sorted({r['case']['cap'] for r in rows if r['case']['fixture']==f.name})
        rank_bounds[f.name]=dict(cut_after_site9=True,relative_singular_values=(s/total).tolist(),
            reference_relative_truncation=json.loads((f/'reference.json').read_text())['relative_discarded_norm'],
            reference_best_rank_tail={str(cap):float(np.linalg.norm(s[cap:])/total) for cap in caps},
            interpretation='tail of independently rounded reference; account for its recorded truncation and floating roundoff; lower bound only, not an algorithm guarantee')
    save(folder/'rank-feasibility.json',rank_bounds)
    print('sample/global disagreement',sum(r['observable_and_sample_gate'] and not r['global_gate'] for r in calibration),'/',len(calibration))
if __name__=='__main__':main()
