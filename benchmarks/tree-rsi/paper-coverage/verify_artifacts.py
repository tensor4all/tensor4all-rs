"""Verify raw output hashes, fixture identities, primary-source blobs and counts."""
import hashlib,json,pathlib
from fixtures import OUT,AUTHOR,ROOT,MAIN_COMMIT,sha,save
from summarize import records
from evidence import expected_schedule,verify_schedule,verify_archived_worker

def expected_observation_count(protocol):
    return len(expected_schedule(protocol))

def verify_source_snapshot(folder,protocol):
    source_files=protocol.get('source_files')
    if not isinstance(source_files,dict) or not source_files:raise ValueError('invalid source-file manifest')
    snapshot=folder/'source-snapshot'
    for name,digest in source_files.items():
        if not isinstance(name,str) or name in ('.','..') or pathlib.PurePath(name).name!=name:raise ValueError('unsafe source-snapshot path')
        path=snapshot/name
        if not path.is_file() or sha(path)!=digest:raise ValueError(f'source snapshot mismatch: {name}')

def verify_worker_hashes(protocol,out=OUT):
    workers=protocol.get('worker_hashes');builds=protocol.get('worker_builds')
    algorithms=set(protocol['algorithms'])
    if not isinstance(workers,dict) or set(workers)!=algorithms or not isinstance(builds,dict) or set(builds)!=algorithms:
        raise ValueError('protocol has incomplete worker build receipts')
    for algorithm,build_id in builds.items():
        if not isinstance(build_id,str) or len(build_id)!=64 or any(c not in '0123456789abcdef' for c in build_id):
            raise ValueError('invalid build identity')
        receipt=verify_archived_worker(out/'builds'/build_id,build_id)
        if workers[algorithm]!=receipt['worker_sha256']:
            raise ValueError(f'worker hash mismatch: {algorithm}')
        if algorithm=='rsi' and protocol.get('candidate_source_sha256')!=receipt['source_sha256']:
            raise ValueError('candidate source differs from worker build receipt')

def verify_run_evidence(folder,protocol,completion,rows,out=OUT):
    verify_schedule(protocol,completion,rows)
    verify_source_snapshot(folder,protocol)
    verify_worker_hashes(protocol,out)

def verify_trace_parity(rows,out):
    baselines={}
    for row in rows:
        name=row.get('baseline_directory')
        if not isinstance(name,str) or pathlib.Path(name).name!=name or name in ('.','..'):
            raise ValueError('invalid trace baseline path')
        if name not in baselines:
            baselines[name]={r['tag']:r for r in records(out/name)}
        baseline=baselines[name].get(row.get('baseline_tag'))
        fields=('case','seed','algorithm','status','output_ranks','diagnostics')
        if baseline is None or row.get('matches_untouched_main') is not True or any(row.get(k)!=baseline.get(k) for k in fields):
            raise ValueError('trace parity failure')
        if row['status']=='completed' and row['validation']['output_sha256']!=baseline['validation']['output_sha256']:
            raise ValueError('trace output parity failure')

def main():
    inventory=json.loads((OUT/'author-tree.json').read_text());verified_author=0
    for entry in inventory['tree']:
        path=AUTHOR/entry['path']
        if entry['type']!='blob' or not path.is_file():continue
        data=path.read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if blob!=entry['sha']:raise ValueError(f'author blob mismatch: {path}')
        verified_author+=1
    fixtures=0
    for folder in (OUT/'fixtures').iterdir():
        if not folder.is_dir():continue
        manifest=json.loads((folder/'fixture.json').read_text())
        if sha(folder/'inputs.bin')!=manifest['source_sha256']:raise ValueError('fixture input bytes changed')
        fixtures+=1
    result={};checks=0;inode_cache={}
    patterns=['run-dmrg-*','run-functions-*','run-tree-*','repeat-*','supplement-*','matched-*','gpe-tolerance-*']
    for pattern in patterns:
        folder=sorted(p for p in OUT.glob(pattern) if p.is_dir())[-1]
        protocol=json.loads((folder/'protocol.json').read_text())
        if protocol.get('baseline')!=MAIN_COMMIT:raise ValueError(f'{folder.name}: stale TreeACI baseline')
        rows=records(folder)
        completion=json.loads((folder/'completion.json').read_text())
        verify_run_evidence(folder,protocol,completion,rows)
        for row in rows:
            if row['status']!='completed':continue
            binary=folder/(row['tag']+'.bin');stat=binary.stat();inode=(stat.st_dev,stat.st_ino,stat.st_size)
            if inode not in inode_cache:inode_cache[inode]=sha(binary)
            if inode_cache[inode]!=row['validation']['output_sha256']:raise ValueError('output differs from recorded bytes')
            checks+=1
        result[folder.name]=dict(observations=len(rows),completed=sum(r['status']=='completed' for r in rows),
                                source_unchanged=completion['candidate_source_unchanged'])
    trace=sorted(OUT.glob('trace-*'))[-1];rows=list(map(json.loads,(trace/'observations.jsonl').read_text().splitlines()))
    trace_protocol=json.loads((trace/'protocol.json').read_text())
    trace_completion=json.loads((trace/'completion.json').read_text())
    if trace_protocol.get('baseline')!=MAIN_COMMIT:raise ValueError('stale trace baseline')
    verify_run_evidence(trace,trace_protocol,trace_completion,rows)
    patch=trace/'instrumentation.patch'
    if not patch.is_file() or sha(patch)!=trace_protocol.get('patch_sha256'):
        raise ValueError('trace instrumentation patch hash mismatch')
    verify_trace_parity(rows,OUT)
    for row in rows:
        if not row['matches_untouched_main']:raise ValueError('trace parity failure')
        if row['status']=='completed' and sha(trace/(row['tag']+'.bin'))!=row['validation']['output_sha256']:raise ValueError('trace output hash mismatch')
    save(OUT/'artifact-verification.json',dict(author_git_blobs_verified=verified_author,fixture_inputs_verified=fixtures,
        measured_outputs_verified=checks,experiments=result,diagnostic_replays=len(rows),completed_diagnostic_outputs=sum(r['status']=='completed' for r in rows),
        primary_author_revision='153b25a8aa059d0147b45955d0842b2f32fa5d1d',baseline=MAIN_COMMIT))
    print('VERIFIED',verified_author,'author blobs;',fixtures,'fixtures;',checks,'measured output hashes;',len(rows),'trace outcomes')
if __name__=='__main__':main()
