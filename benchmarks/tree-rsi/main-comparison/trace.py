#!/usr/bin/env python3
"""Untimed main instrumentation replay. Never substitute these times for baseline."""
import compare as c
import supplement as s
import argparse, ast, difflib, json, os, pathlib, shutil
HOOKS = {
    'state.rs': [('        let edge_ranks = initial_edge_ranks;', '        let edge_ranks = initial_edge_ranks;\n        eprintln!("RANKTRACE init {:?}", edge_ranks);')],
    'transaction.rs': [('    state.edge_ranks[edge_number] = state.pivots.rank(edge_number);','    state.edge_ranks[edge_number] = state.pivots.rank(edge_number);\n    eprintln!("RANKTRACE commit {} {}", edge_number, state.edge_ranks[edge_number]);')],
    'global_guard.rs': [('    state.edge_ranks = next_edge_ranks;','    state.edge_ranks = next_edge_ranks;\n    eprintln!("RANKTRACE guard {:?}", state.edge_ranks);')],
    'local_update.rs': [('    let col_count = col_candidates.len();','    let col_count = col_candidates.len();\n    eprintln!("RANKTRACE matrix {} {} {}", forward, row_count, col_count);')],
}
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('initial',type=pathlib.Path); args=parser.parse_args()
    initial=args.initial.resolve(); supplement=initial/'supplement'
    if not (supplement/'completion.json').exists(): raise RuntimeError('timing must finish before diagnostic build')
    directory=initial/'trace';directory.mkdir(); source=directory/'main';shutil.copytree(initial/'main',source)
    patch=[]
    for filename,replacements in HOOKS.items():
        p=source/'crates/tensor4all-treeaci/src'/filename; old=p.read_text(); new=old
        for before,after in replacements:
            if new.count(before)!=1: raise RuntimeError('instrumentation hook mismatch: '+filename)
            new=new.replace(before,after)
        p.write_text(new)
        patch.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+filename,tofile='b/'+filename))
    (directory/'instrumentation.patch').write_text(''.join(patch))
    c.save(directory/'protocol.json',dict(baseline=json.loads((initial/'protocol.json').read_text())['baseline'],patch_sha256=c.sha(directory/'instrumentation.patch'),
        purpose='All initial active ranks, every committed edge update and global-guard rank injection; every local matrix shape. No algorithm decisions changed.',
        comparison='Exact output byte hash, all output ranks, termination and diagnostics must match untouched main for the same case/seed.',
        timing='Diagnostic runtimes excluded from all timing summaries.'))
    worker=c.build(directory/'worker',source,'treeaci',c.ROOT/'target')
    c.save(directory/'binary.json',dict(sha256=c.sha(worker)))
    os.sched_setaffinity(0,set(json.loads((initial/'protocol.json').read_text())['affinity']))
    baseline={}
    for line in (supplement/'observations.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r['algorithm']=='treeaci' and r['block']==0: baseline[r['case'],r['seed']]=r
    all_cases=[(initial,t,n,chi,cap) for t,n,chi,cap in c.CASES]+[(supplement,t,n,chi,cap) for t,n,chi,cap in s.SUPPLEMENT]
    with (directory/'observations.jsonl').open('w') as file:
        for base,t,n,chi,cap in all_cases:
            fixture=base/s.name(t,n,chi);case=f'{fixture.name}-cap{cap}'
            for seed in c.SEEDS:
                r=c.run(worker,fixture,case,'treeaci',seed,directory)
                peak=0;matrices=[];commits=0;guards=0
                for line in (directory/f'{case}-treeaci-seed{seed}.stderr').read_text().splitlines():
                    if not line.startswith('RANKTRACE '):continue
                    kind,data=line[len('RANKTRACE '):].split(' ',1)
                    if kind in ('init','guard'):
                        peak=max(peak,max(ast.literal_eval(data),default=1));guards+=kind=='guard'
                    elif kind=='commit':peak=max(peak,int(data.split()[1]));commits+=1
                    elif kind=='matrix': matrices.append(list(map(int,data.split())))
                b=baseline[case,seed]
                r.update(peak_active_bond_rank=peak,local_matrix_shapes=matrices,edge_commit_count=commits,guard_injection_count=guards,
                    matches_untouched_main=all(r.get(k)==b.get(k) for k in ('output_sha256','output_ranks','diagnostics','status')))
                file.write(json.dumps(r)+'\n');file.flush()
                print('TRACE',case,seed,'peak',peak,'matrix',max((a*b for _,a,b in matrices),default=0),'parity',r['matches_untouched_main'],flush=True)
    c.save(directory/'completion.json',dict(observations=len(all_cases)*3))
if __name__=='__main__':main()
