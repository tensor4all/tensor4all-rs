#!/usr/bin/env python3
import argparse, os, subprocess, json, hashlib, time, re, statistics, random, math, pathlib
parser = argparse.ArgumentParser(description='Run the complete predeclared TreeTCI memo paired experiment.')
parser.add_argument('--baseline', type=pathlib.Path, required=True)
parser.add_argument('--candidate', type=pathlib.Path, required=True)
parser.add_argument('--memo-candidate', type=pathlib.Path, required=True)
parser.add_argument('--output', type=pathlib.Path, required=True)
args = parser.parse_args()
root = pathlib.Path(__file__).resolve().parents[2]
out = args.output.resolve()
out.mkdir(parents=True,exist_ok=True)
env=dict(os.environ)
for name in ['RAYON_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:env[name]='1'
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
assert not subprocess.check_output(['git','diff','HEAD'],cwd=root)
manifest={'baseline':'f11e30d847fadb3739aa2cc35f1bd0951186254d','candidate':commit,'cpu':2,'environment':{k:env[k] for k in ['RAYON_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']},'sources':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['benchmarks/rust/benchmark_treetci_global_search.rs','benchmarks/rust/benchmark_treetci_memo.rs','benchmarks/2026-10-08-treetci-memo-protocol.md']}}
manifest['binary_sha256'] = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in [('baseline', args.baseline), ('candidate', args.candidate), ('memo_candidate', args.memo_candidate)]}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
rows=[]
def cpu_ticks():
 return list(map(int,next(line for line in pathlib.Path('/proc/stat').read_text().splitlines() if line.startswith('cpu2 ')).split()[1:]))
def parse(text,kind):
 result={}
 lines=text.splitlines()
 for index,line in enumerate(lines):
  if not line.startswith('case='):continue
  case=re.search(r'case=(\S+)',line)[1]
  if kind=='global':
   seconds=float(re.search(r'median_s=(\S+)',line)[1]);signature=re.sub(r' (min_s|median_s)=\S+','',line)+'\n'+lines[index+1]+'\n'+lines[index+2]
   result[case]={'seconds':seconds,'signature':signature}
  else:
   seconds=float(re.search(r'seconds=(\S+)',line)[1]);signature=line[line.index('ranks='):]
   result[case]={'seconds':seconds,'signature':signature,**{k:int(re.search(r'\b'+k+r'=(\d+)',line)[1]) for k in ['requested','evaluated','entries','bytes','drops']}}
 return result
for round_no in range(7):
 for kind,binaries in [('global',[(str(args.baseline.resolve()),None,'baseline'),(str(args.candidate.resolve()),None,'candidate')]),('memo',[(str(args.memo_candidate.resolve()),'plain','baseline'),(str(args.memo_candidate.resolve()),'memo','candidate')])]:
  pair=[]
  for binary,arg,label in (binaries if round_no%2==0 else binaries[::-1]):
   load=os.getloadavg()[0];before=cpu_ticks();prefix=f'{kind}-{round_no}-{label}'
   cmd=['/usr/bin/time','-f','max_rss_kib=%M',binary]+([] if arg is None else [arg])
   p=subprocess.Popen(cmd,env=env,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,preexec_fn=lambda:os.sched_setaffinity(0,{2}))
   affinity_ok=True
   while p.poll() is None:
    try:
     pending=[p.pid]
     while pending:
      pid=pending.pop()
      for task in pathlib.Path(f'/proc/{pid}/task').iterdir():
       status=(task/'status').read_text()
       affinity_ok &= re.search(r'Cpus_allowed_list:\s*(\S+)',status)[1]=='2'
       pending.extend(map(int,(task/'children').read_text().split()))
    except FileNotFoundError:pass
    time.sleep(.05)
   stdout,stderr=p.communicate();after=cpu_ticks();delta=[a-b for a,b in zip(after,before)];steal=delta[7]/max(1,sum(delta));load_after=os.getloadavg()[0]
   (out/f'{prefix}.txt').write_text(stdout+'\n'+stderr)
   row={'kind':kind,'round':round_no,'label':label,'exit':p.returncode,'load_before':load,'load_after':load_after,'steal_fraction':steal,'affinity_ok':affinity_ok,'max_rss_kib':int(re.search(r'max_rss_kib=(\d+)',stderr)[1]),'cases':parse(stdout,kind)}
   rows.append(row);pair.append(row);(out/'rounds.json').write_text(json.dumps(rows,indent=2)+'\n')
   print(prefix,'exit',p.returncode,'cases',len(row['cases']),'RSS',row['max_rss_kib'],flush=True)
  if any(r['exit'] or len(r['cases'])!=(3 if kind=='global' else 8) for r in pair):
   (out/'failed.json').write_text(json.dumps({'reason':'invalid complete suite','rows':pair},indent=2));raise SystemExit(1)
valid=all(r['affinity_ok'] and max(r['load_before'],r['load_after'])<=8 and r['steal_fraction']<=.02 for r in rows)
ratios={};correct=True;reductions={};rss={}
for kind in ['global','memo']:
 for round_no in range(7):
  b=next(r for r in rows if r['kind']==kind and r['round']==round_no and r['label']=='baseline');c=next(r for r in rows if r['kind']==kind and r['round']==round_no and r['label']=='candidate')
  for case,base in b['cases'].items():
   cand=c['cases'][case];key=kind+'/'+case;ratios.setdefault(key,[]).append(cand['seconds']/base['seconds']);correct &= cand['signature']==base['signature']
   rss.setdefault(key,[]).append([b['max_rss_kib'],c['max_rss_kib']])
   if kind=='memo':
    correct &= cand['requested']==base['requested'] and cand['bytes']<=256*1024*1024 and cand['drops']==0
    reductions.setdefault(key,[]).append(cand['evaluated']/base['evaluated'])
    if case.startswith('expensive'):correct &= cand['evaluated']<=.5*base['evaluated']
rng=random.Random(802);samples=[[rng.randrange(7) for _ in range(7)] for _ in range(10000)]
def ci(values):
 vals=sorted(values);return [vals[249],vals[9749]]
summary={}
for key,values in ratios.items():
 interval=ci([statistics.median([values[i] for i in sample]) for sample in samples]);cv=statistics.pstdev(values)/statistics.mean(values);valid &= cv<=.20
 summary[key]={'ratios':values,'median_ratio':statistics.median(values),'ci95':interval,'relative_sd':cv,'evaluation_ratios':reductions.get(key),'paired_process_rss_kib':rss[key]}
expensive=[values for key,values in ratios.items() if key.startswith('memo/expensive')]
geom=lambda vals:math.exp(statistics.mean(math.log(x) for x in vals))
primary=geom([statistics.median(v) for v in expensive]);primary_ci=ci([geom([statistics.median([v[i] for i in sample]) for v in expensive]) for sample in samples])
nonregression=all(v['ci95'][1]<=1.10 for k,v in summary.items() if k.startswith('global/'))
result={'decision':'INCONCLUSIVE' if not valid else 'PASS' if correct and nonregression and primary_ci[1]<=.8 else 'FAIL','valid':valid,'correct':correct,'default_nonregression':nonregression,'primary_ratio':primary,'primary_ci95':primary_ci,'cases':summary}
(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
