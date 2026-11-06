"""Fetch pinned primary inputs into ignored storage; never execute author code."""
import hashlib,json,pathlib,subprocess,urllib.request
from fixtures import OUT,ROOT,AUTHOR,MAIN_COMMIT,save,sha
from build import BASELINE,BASELINE_ROOT
REVISION='153b25a8aa059d0147b45955d0842b2f32fa5d1d'

def fetch(url):
    request=urllib.request.Request(url,headers={'User-Agent':'tensor4all-paper-audit'})
    with urllib.request.urlopen(request,timeout=120) as response:return response.read()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    tree=json.loads(fetch(f'https://api.github.com/repos/zmeng137/Recursive-Sketched-Interpolation/git/trees/{REVISION}?recursive=1'))
    if tree.get('truncated'):raise RuntimeError('truncated author inventory')
    save(OUT/'author-tree.json',tree);receipt=[];total=0
    for entry in tree['tree']:
        name=entry['path']
        selected=name.endswith(('.py','.md','.h5','.hdf5')) or name=='gpe_density/91v.pkl'
        if entry['type']!='blob' or not selected:continue
        total+=entry.get('size',0)
        if total>512*1024**2:raise RuntimeError('author download exceeds512MiB budget')
        relative=pathlib.PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('unsafe author path')
        path=AUTHOR/name;path.parent.mkdir(parents=True,exist_ok=True)
        url=f'https://raw.githubusercontent.com/zmeng137/Recursive-Sketched-Interpolation/{REVISION}/{name}'
        data=fetch(url)
        if path.exists() and path.read_bytes()!=data:raise RuntimeError(f'existing author artifact differs: {name}')
        if not path.exists():path.write_bytes(data)
        receipt.append(dict(path=name,url=url,bytes=len(data),sha256=sha(path)))
    save(OUT/'download-receipt.json',receipt)
    # git archive fixes the baseline independently of any local main changes.
    if not BASELINE.exists():
        BASELINE_ROOT.mkdir(parents=True,exist_ok=True)
        BASELINE.mkdir(parents=True,exist_ok=True)
        archive=OUT/f'baseline-main-{MAIN_COMMIT[:7]}.tar'
        with archive.open('wb') as f:subprocess.run(['git','archive',MAIN_COMMIT],cwd=ROOT,stdout=f,check=True)
        subprocess.run(['tar','-xf',str(archive),'-C',str(BASELINE)],check=True);archive.unlink()
    paper=OUT/'paper.pdf'
    if not paper.exists():paper.write_bytes(fetch('https://arxiv.org/pdf/2602.17974v1'))
    save(OUT/'primary-sources.json',dict(author=REVISION,main=MAIN_COMMIT,paper_sha256=sha(paper),author_files=len(receipt)))
    print('Pinned primary inputs verified:',len(receipt))
if __name__=='__main__':main()
