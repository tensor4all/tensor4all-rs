"""Apply stable global n20 validation to repeats/sensitivity without duplicate work."""
import json
import numpy as np
import reference as ref
from fixtures import OUT,read_export,read_inputs,save,sha

def main():
    baseline=sorted(OUT.glob('run-dmrg-*'))[-1]
    measured={r['tag']:r for r in map(json.loads,(baseline/'global-checks.jsonl').read_text().splitlines())}
    cache={};references={}
    for row in map(json.loads,(baseline/'observations.jsonl').read_text().splitlines()):
        if row['tag'] not in measured:continue
        binary=baseline/(row['tag']+'.bin');key=(row['case']['fixture'],sha(binary),sha(binary.with_suffix('.cores.json')))
        cache[key]=measured[row['tag']]
    for pattern in ('repeat-*','supplement-*','matched-*'):
        folders=sorted(p for p in OUT.glob(pattern) if p.is_dir())
        if not folders:continue
        folder=folders[-1]
        if not (folder/'completion.json').exists():raise RuntimeError('finish measured calls first')
        checks=[]
        for row in map(json.loads,(folder/'observations.jsonl').read_text().splitlines()):
            fixture=row['case']['fixture']
            if row['status']!='completed' or not fixture.startswith('dmrg-n20'):continue
            binary=folder/(row['tag']+'.bin');key=(fixture,sha(binary),sha(binary.with_suffix('.cores.json')))
            if key in cache:
                record=dict(cache[key]);record['validation_reused_from']=record['tag']
            else:
                if fixture not in references:
                    f=OUT/'fixtures'/fixture;archive=np.load(f/'reference-cores.npz')
                    references[fixture]=([archive[f'v{i}'] for i in range(20)],json.loads((f/'reference.json').read_text()))
                cores,metadata=references[fixture];error=ref.norm(ref.difference(read_export(binary),cores))/metadata['reference_norm']
                record=dict(relative_l2_vs_reference_tt=error,relative_reference_discarded_norm=metadata['relative_discarded_norm'],
                            gate_1e8=error+metadata['relative_discarded_norm']<=1e-8)
            record.update(tag=row['tag'],case=row['case'],algorithm=row['algorithm'],seed=row['seed'])
            cache[key]=record;checks.append(record)
        (folder/'global-checks.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in checks));print('GLOBAL FOLLOWUPS',folder.name,len(checks),flush=True)
    reference_bounds()

def reference_bounds():
    bounds={}
    for folder in (OUT/'fixtures').iterdir():
        m=json.loads((folder/'fixture.json').read_text())
        if m['metadata']['kind']!='analytic':continue
        inputs=read_inputs(folder);metadata=json.loads((folder/'reference.json').read_text());bound=0.
        operand_bounds=[min(ref.norm(operand),ref.maxabs_upper_bound(operand)) for operand in inputs[1:]]
        for maximum,loss in zip(operand_bounds,metadata['svd_discarded_norms']):bound=bound*maximum+loss
        bounds[folder.name]=dict(relative_svd_truncation_upper_bound=bound/metadata['norm'],
            subsequent_operand_maxabs_bounds=operand_bounds,
            method='accumulate local SVD losses multiplied by subsequent operand entrywise upper bounds (bounded prefix row norms times remaining local spectral norms after right QR, capped by Frobenius norm); not a floating-point interval certificate')
    save(OUT/'reference-truncation-bounds.json',bounds)
if __name__=='__main__':main()
