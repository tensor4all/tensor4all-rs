#!/usr/bin/env python3
"""Audit and freeze the exact downstream-pinned measurement source graph."""
import difflib,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def git(*args,cwd=ROOT):return subprocess.check_output(['git',*args],cwd=cwd,text=True).strip()
def main():
    metadata=json.loads(Path(sys.argv[1]).read_text())
    baseline=git('merge-base','origin/main','HEAD')
    original=git('show',baseline+':Cargo.lock').splitlines()
    current=(ROOT/'Cargo.lock').read_text().strip().splitlines()
    # Compare the complete lock text: only the fixture's two dev references
    # may be added. No package/source/version/checksum entry may change.
    delta=[line for line in difflib.unified_diff(original,current,n=0)
           if line.startswith(('+','-')) and not line.startswith(('+++','---'))]
    assert delta==['+ "libc",','+ "serde_json",'],delta
    relevant=lambda p:p['name'].startswith(('tenferro','threadteam','hataori','t4a-faer','cpueinsum','tcontract','strided','tlinalg'))
    production=git('diff',baseline,'--','Cargo.toml','.cargo/config.toml','crates/*/src')
    assert not production,'production source/config was changed'
    assert not git('status','--porcelain'),'commit the frozen harness before recording'
    sources=[];checkouts={}
    for p in metadata['packages']:
        if not relevant(p):continue
        assert p['source'] and p['source'].startswith(('git+','registry+')),p
        sources.append({k:p[k] for k in ('name','version','source','manifest_path')})
        if p['source'].startswith('git+'):
            directory=git('rev-parse','--show-toplevel',cwd=Path(p['manifest_path']).parent)
            head=git('rev-parse','HEAD',cwd=directory)
            assert head==p['source'].rsplit('#',1)[1]
            status=git('status','--porcelain',cwd=directory).splitlines()
            # Cargo itself writes this fetch-completion marker; it is not source.
            assert all(line=='?? .cargo-ok' for line in status),status
            checkouts[directory]=dict(commit=head,tracked_source_clean=True,cargo_generated_markers=status)
    binary=Path(sys.argv[2]).resolve()
    record=dict(root_base=baseline,harness_commit=git('rev-parse','HEAD'),dirty=False,production_unchanged=True,
        dependency_identities_match_downstream_lock=True,dependency_sources=sources,clean_git_checkouts=checkouts,
        binary=str(binary),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        provider='native/faer',features=['parallel'],profile='release',build_memory_cap='8G',build_jobs=2,
        rustc=Path('/tmp/1342-clean-pins-rustc.txt').read_text(),
        cpu_topology=subprocess.check_output(['lscpu','-e=CPU,CORE,SOCKET'],text=True),
        observer_cpus=list(range(8)),measurement_domain=list(range(8,16)),compute_cpus=[8,10,12,14],
        kernel=os.uname().release,host=os.uname().nodename,
        requested_environment={k:os.environ.get(k) for k in ('RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','RAYON_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')})
    Path(sys.argv[3]).write_text(json.dumps(record,indent=2)+'\n')
    print('Verified clean dependency graph and unchanged production code:',len(sources),'packages')
if __name__=='__main__':main()
