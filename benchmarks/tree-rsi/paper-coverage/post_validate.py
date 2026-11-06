"""Stronger n20 checks after timing: bounded explicit products and stable QR norms."""
import reference as ref
from fixtures import OUT,read_inputs,read_export,save
import numpy as np
import pathlib,json,time

def main():
    runs=sorted(OUT.glob('run-dmrg-*'));directory=runs[-1]
    if not (directory/'completion.json').exists():raise RuntimeError('finish timed phase first')
    targets=[r for r in map(json.loads,(directory/'observations.jsonl').read_text().splitlines()) if r['status']=='completed' and r['case']['fixture'].startswith('dmrg-n20')]
    with (directory/'global-checks.jsonl').open('w') as log:
        for chi in [10,15,20,25,30]:
            folder=OUT/'fixtures'/f'dmrg-n20-chi{chi}';inputs=read_inputs(folder)
            started=time.monotonic();exact=ref.product(*inputs,max_total_elements=50_000_000)
            reference,discarded=ref.round_chain(exact,chi*(chi+1)//2,1e-15)
            reference_norm=ref.norm(reference);del exact
            np.savez(folder/'reference-cores.npz',**{f'v{i}':a for i,a in enumerate(reference)})
            metadata=dict(reference_norm=reference_norm,discarded_norm=discarded,relative_discarded_norm=discarded/reference_norm,seconds=time.monotonic()-started,
                method='bounded exact core product followed by QR/SVD; cap is symmetric-square dimension; measured residual uses block-difference QR, no subtraction of Gram norms')
            save(folder/'reference.json',metadata);print('GLOBAL REFERENCE',chi,metadata,flush=True)
            for record in targets:
                if record['case']['fixture']!=folder.name:continue
                result=read_export(directory/(record['tag']+'.bin'))
                error=ref.norm(ref.difference(result,reference))/reference_norm
                row=dict(tag=record['tag'],case=record['case'],algorithm=record['algorithm'],seed=record['seed'],relative_l2_vs_reference_tt=error,relative_reference_discarded_norm=discarded/reference_norm,
                    gate_1e8=error+discarded/reference_norm<=1e-8)
                log.write(json.dumps(row)+'\n');log.flush();print('GLOBAL',record['tag'],error,flush=True)
if __name__=='__main__':main()
