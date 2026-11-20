from pathlib import Path
import json,struct,sys,numpy as np
out=Path(sys.argv[1]) if len(sys.argv)>1 else Path('target/issue1265/divergence');records={}
for arm in ['baseline','candidate']:
 name=arm+'-focus';binary=(out/(name+'-p2s24.bin')).read_bytes();m,n=struct.unpack('<QQ',binary[:16]);a=np.frombuffer(binary[16:],dtype='<c16').reshape((m,n),order='F');meta={}
 for l in (out/(name+'.trace.jsonl')).open():
  try:t=json.loads(l)
  except json.JSONDecodeError:continue
  if (t['pass'],t['step'])==(2,24):meta[t['kind']]=t
 i,j=meta['factors']['row_indices'],meta['factors']['col_indices'];s=np.linalg.svd(a,compute_uv=False);p=a[np.ix_(i,j)];sp=np.linalg.svd(p,compute_uv=False)
 approx=a[:,j]@np.linalg.solve(p,a[i,:]);res=np.max(np.abs(a-approx))
 records[arm]={'matrix':a,'meta':meta,'result':{'shape':[m,n],'singular_values':s.tolist(),'selected_cross_singular_values':sp.tolist(),'selected_cross_condition':float(sp[0]/sp[-1]),'local_cross_reconstruction_max_residual':float(res),'rank':len(i),'normalized_pivot_errors':(np.array(meta['factors']['pivot_errors'])/meta['factors']['sampled_scale']).tolist()}}
a,b=records['baseline'],records['candidate'];ac,bc=a['matrix'],b['matrix'];ar,br=a['meta']['matrix'],b['meta']['matrix'];firstcols=list(dict.fromkeys(ar['cols']));ai=[ar['cols'].index(k) for k in firstcols];bi=[br['cols'].index(k) for k in firstcols];common=bc[:,bi]
result={'baseline':a['result'],'candidate':b['result'],'common_column_keys':firstcols,'added_column_keys':list(set(br['cols'])-set(ar['cols'])),'common_columns_max_abs_difference':float(np.max(np.abs(ac[:,ai]-common))),'candidate_restricted_to_baseline_columns_singular_values':np.linalg.svd(common,compute_uv=False).tolist(),'scope':'Diagnostic SVD of actual raw local matrices, not a change to production linalg or truncation. First divergent candidate grid: pass2 step24 cut44 (21->22).'}
(out/'condition.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
