import json,sys
# usage: table.py <topology:tree|chain> <mono_file> <file>... ; depth-0 baseline from mono_file
def load(f):
  out={}; pt={}
  for l in open('raw/%s.jsonl'%f):
    d=json.loads(l)
    if d['kind']=='patch': pt.setdefault(d['depth'],[]).append(d)
    if d['kind']=='depth': d['_p']=pt.get(d['depth'],[]); out[d['depth']]=d
  return out
topo=sys.argv[1]; base=load(sys.argv[2])[0]; key='sum_r3' if topo=='tree' else 'sum_r2'
print('| k | patches (feat/non/zero) | max rank feat / non | sum evals (x mono) | sum TCI s (x mono) | sum r^%s (x mono) | params (x mono) | rel L2 | max err/max |'%('3' if topo=='tree' else '2'))
print('|---|---|---|---|---|---|---|---|---|')
rows={}
for f in sys.argv[2:]:
  for k,d in load(f).items(): rows[k]=d
for k in sorted(rows):
  d=rows[k]; ps=d['_p']
  fe=[p['rank'] for p in ps if p['feature']]; no=[p['rank'] for p in ps if not p['feature']]
  nz=sum(1 for p in ps if p['termination']=='AllSamplesZero')
  print('| %d | %d (%d/%d/%d) | %s / %s | %d (%.2f) | %.2f (%.2f) | %.3g (%.2f) | %d (%.2f) | %.1e | %.1e |'%(k,d['patches'],len(fe),len(no)-nz,nz,
    max(fe) if fe else '-', ('%d-%d'%(min(no),max(no))) if no else '-',
    d['sum_evals_unique'],d['sum_evals_unique']/base['sum_evals_unique'],d['sum_tci_seconds'],d['sum_tci_seconds']/base['sum_tci_seconds'],
    d[key],d[key]/base[key],d['sum_params'],d['sum_params']/base['sum_params'],d['rel_l2_sampled'],d['max_err_over_max']))
