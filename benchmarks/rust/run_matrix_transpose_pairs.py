#!/usr/bin/env python3
"""Repeat complete baseline/candidate transpose suites under fixed affinity.

Build the shared benchmark harness in both worktrees, then pass the resulting
immutable binaries. The protocol fixes nine pairs and bootstrap seed 738.
"""
import argparse
import csv,io,json,math,os,random,statistics,subprocess,time,hashlib
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--baseline',type=Path,required=True)
parser.add_argument('--candidate',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--cpu',type=int,default=2)
args=parser.parse_args()
bins={'baseline':str(args.baseline.resolve()),'candidate':str(args.candidate.resolve())}
env=os.environ.copy()
for name in ['RAYON_NUM_THREADS','BLAS_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:env[name]='1'
def observe():return {'cpu':list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:])), 'load':Path('/proc/loadavg').read_text().strip(),'mem_available':next(s for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))}
record={'before':observe(),'runs':[],'affinity_valid':True,'binary_sha256':{k:hashlib.sha256(Path(p).read_bytes()).hexdigest() for k,p in bins.items()}}
for pair in range(9):
 for side in (['baseline','candidate'] if pair%2==0 else ['candidate','baseline']):
  proc=subprocess.Popen([bins[side]],preexec_fn=lambda:os.sched_setaffinity(0,{args.cpu}),env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
  while proc.poll() is None:
   try:
    for task in Path(f'/proc/{proc.pid}/task').iterdir():
     try:record['affinity_valid'] &= os.sched_getaffinity(int(task.name))=={args.cpu}
     except ProcessLookupError:pass
   except FileNotFoundError:pass
   time.sleep(.01)
  out,err=proc.communicate()
  if proc.returncode:raise RuntimeError((side,proc.returncode,err))
  rows=list(csv.DictReader(io.StringIO(out)))
  assert len(rows)==60,len(rows)
  record['runs'].append({'pair':pair,'side':side,'rows':rows})
  args.output.write_text(json.dumps(record,indent=2)+'\n')
 print('pair',pair+1,'complete',flush=True)
record['after']=observe();a=record['before']['cpu'];b=record['after']['cpu'];delta=[y-x for x,y in zip(a,b)];record['steal_fraction']=delta[7]/sum(delta)
keys=[(r['kind'],r['dtype'],r['rows'],r['cols']) for r in record['runs'][0]['rows']]
summary=[];rng=random.Random(738)
for key in keys:
 ratios=[]
 for pair in range(9):
  vals={run['side']:float(next(r['seconds_per_call'] for r in run['rows'] if (r['kind'],r['dtype'],r['rows'],r['cols'])==key)) for run in record['runs'] if run['pair']==pair}
  ratios.append(vals['candidate']/vals['baseline'])
 boot=sorted(statistics.median(rng.choices(ratios,k=len(ratios))) for _ in range(10000))
 summary.append({'key':key,'ratios':ratios,'median_ratio':statistics.median(ratios),'ci95':[boot[250],boot[9749]]})
controls=[r['median_ratio'] for r in summary if r['key'][0]=='micro' and r['key'][2:] == ('8','8')]
record['valid']=record['affinity_valid'] and record['steal_fraction']<.01 and all(.9<=r<=1.1 for r in controls)
primary=[r['median_ratio'] for r in summary if r['key'][0]=='micro' and r['key'][2]==r['key'][3] and int(r['key'][2])>=256]
record['primary_geomean']=math.exp(statistics.mean(map(math.log,primary)))
record['summary']=summary
args.output.write_text(json.dumps(record,indent=2)+'\n')
print('valid',record['valid'],'primary_geomean',record['primary_geomean'],'controls',controls)
for row in summary:print(row['key'],round(row['median_ratio'],4),[round(x,4) for x in row['ci95']])
