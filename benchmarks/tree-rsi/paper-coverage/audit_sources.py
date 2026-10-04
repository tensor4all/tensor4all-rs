"""Capture source evidence and active dependency graphs, including untracked files."""
import json,pathlib,re,subprocess
from fixtures import OUT,ROOT,AUTHOR,save,sha

def evidence(path,patterns):
    text=path.read_text();lines=text.splitlines()
    return dict(path=str(path),sha256=sha(path),matches=[dict(line=i+1,text=line) for i,line in enumerate(lines) if any(re.search(p,line) for p in patterns)])

def main():
    result={}
    for algorithm in ('treeaci','rsi'):
        manifest=OUT/algorithm/'Cargo.toml'
        tree=subprocess.check_output(['cargo','tree','--offline','--locked','--manifest-path',str(manifest),'-e','normal,build'],cwd=ROOT,text=True)
        (OUT/f'{algorithm}-active-dependencies.txt').write_text(tree)
        prohibited=['tensor4all-simplett','tensor4all-tensorci','tensor4all-aci']+(['tensor4all-treeaci'] if algorithm=='rsi' else [])
        result[algorithm]=dict(active_dependency_graph_sha256=sha(OUT/f'{algorithm}-active-dependencies.txt'),prohibited_present=[p for p in prohibited if re.search(r'\b'+re.escape(p)+r' v',tree)])
        if result[algorithm]['prohibited_present']:raise ValueError(f'prohibited active dependencies in {algorithm}')
    result['author_sketch']=evidence(AUTHOR/'src/sketch.py',[r'def tt_sketching_cache',r'rd.seed',r'np.random.normal'])
    result['author_dmrg']=evidence(AUTHOR/'test/mps_dmrg.py',[r'fulleval_thres',r'tt_to_tensor',r'eps=',r'oversampling =',r'skdim ='])
    result['author_lu']=evidence(AUTHOR/'src/rank_revealing.py',[r'def prrldu',r'k = min\(Nr, Nc\)',r'while s < k',r'for s in range\(min\(k, maxdim\)\)',r'np.outer'])
    result['rust_options']=evidence(ROOT/'crates/tensor4all-treersi/src/options.rs',[r'ceil\(',r'rel_tol:',r'oversampling:',r'sketch_dim:'])
    gw=pathlib.Path('/root/projects/gw-rs')
    if gw.exists():
        result['downstream']={
            'g0':evidence(gw/'g0_rsi/src/main.rs',[r'tree_rsi_elementwise',r'N_TEST:',r'VAL_TOL:',r'Validation: max relerr',r'abs_tol:']),
            'product':evidence(gw/'sgw_rsi/src/ops/rsi.rs',[r'tree_rsi_elementwise',r'local_pivot_error',r'abs_tol:',r'let error']),
            'provenance':evidence(gw/'sgw_rsi/src/provenance.rs',[r'run_git.*diff',r'Untracked files',r'diff_hash']),
            'dependencies':evidence(gw/'sgw_rsi/Cargo.toml',[r'tensor4all-(simplett|aci|treersi)']),
        }
        result['downstream_scope']='source/API audit only; no old pipeline execution or numerical acceptance; source hashes identify current untracked files'
    save(OUT/'source-audit.json',result);print('Active dependency prohibitions checked; source evidence recorded.')
if __name__=='__main__':main()
