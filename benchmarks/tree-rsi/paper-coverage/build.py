"""Build fully isolated public-API workers; reuse existing release dependencies."""
from pathlib import Path
import json,shutil,subprocess
ROOT=Path(__file__).resolve().parents[3]
SOURCE=Path(__file__).resolve().parent
from fixtures import MAIN_COMMIT,OUT
from evidence import archive_worker,source_inputs
OUTPUT=OUT
BASELINE_ROOT=OUTPUT/f'main-comparison-{MAIN_COMMIT[:7]}'
BASELINE=BASELINE_ROOT/'main'
def build(name,source,filename,algorithm=None):
    directory=OUTPUT/name;directory.mkdir(exist_ok=True)
    shutil.copy2(SOURCE/filename,directory/'main.rs');shutil.copy2(source/'Cargo.lock',directory/'Cargo.lock')
    deps='\n'.join(f'tensor4all-{p} = {{ path = "{source}/crates/tensor4all-{p}", default-features = false, features = ["tenferro-cpu-faer"] }}' for p in ('core','treetn'))
    extra='treetci' if algorithm is None else 'treersi' if algorithm=='rsi' else 'treeaci'
    deps+=f'\ntensor4all-{extra} = {{path = "{source}/crates/tensor4all-{extra}"}}'
    (directory/'Cargo.toml').write_text(f'''[workspace]
[package]
name="paper-{name}"
version="0.0.0"
edition="2024"
[features]
rsi=[]
treeaci=[]
[dependencies]
serde_json="1"
anyhow="1"
num-complex="0.4"
{deps}
[[bin]]
name="paper-{name}"
path="main.rs"
[profile.release]
debug=0
''')
    before=source_inputs(directory,source,ROOT,algorithm,SOURCE/filename)
    args=['cargo','build','--release','--offline','--locked','--message-format=json-render-diagnostics','--manifest-path',str(directory/'Cargo.toml'),'--target-dir',str(ROOT/'target')]
    if algorithm:args+=['--features',algorithm]
    result=subprocess.run(args,cwd=ROOT,capture_output=True,text=True)
    (directory/'build.log').write_text(result.stdout+result.stderr)
    result.check_returncode()
    executables=[event['executable'] for event in map(json.loads,result.stdout.splitlines())
        if event.get('reason')=='compiler-artifact' and event['target']['name']==f'paper-{name}' and event.get('executable')]
    if len(executables)!=1:raise ValueError('Cargo did not identify exactly one worker executable')
    if before!=source_inputs(directory,source,ROOT,algorithm,SOURCE/filename):raise ValueError('source or build settings changed while building worker')
    shutil.copy2(executables[0],directory/'worker')
    archive_worker(directory,OUTPUT,before)
if __name__=='__main__':
    for name,src,file,algorithm in [('treeaci',BASELINE,'worker.rs','treeaci'),('rsi',ROOT,'worker.rs','rsi'),('prepare',ROOT,'prepare.rs',None)]:
        print('BUILD',name,flush=True);build(name,src,file,algorithm)
