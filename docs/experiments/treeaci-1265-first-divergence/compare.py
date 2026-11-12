from pathlib import Path
import json,sys
out=Path(sys.argv[3]) if len(sys.argv)>3 else Path('target/issue1265/divergence')
def load(name):
 records={};matrix=None
 for line in (out/(name+'.trace.jsonl')).open():
  try: t=json.loads(line)
  except json.JSONDecodeError:continue
  key=(t['pass'],t['step'],t['kind'])
  if t['kind']=='matrix':matrix=t
  if t['kind']=='factors':
   t['selected_rows']=[matrix['rows'][i] for i in t['row_indices']]
   t['selected_cols']=[matrix['cols'][i] for i in t['col_indices']]
  records[key]=t
 return records
left=sys.argv[1] if len(sys.argv)>1 else 'baseline-natural';right=sys.argv[2] if len(sys.argv)>2 else 'candidate-natural'
a,b=load(left),load(right);first={};all_diffs={}
def unique(v):return list(dict.fromkeys(v))
for key in sorted(a.keys()&b.keys()):
 x,y=a[key],b[key];kind=x['kind'];checks={}
 if kind in ['before_edge','after_guard','after_pass']:
  checks['raw_candidates']=x['candidates']==y['candidates']
  checks['distinct_candidates']=[unique(v) for v in x['candidates']]==[unique(v) for v in y['candidates']]
  checks['candidate_membership']=[set(v) for v in x['candidates']]==[set(v) for v in y['candidates']]
  checks['selected_state']=x['pivots']==y['pivots']
 if kind=='matrix':
  checks['raw_matrix']=x['matrix_hash']==y['matrix_hash'] and x['rows']==y['rows'] and x['cols']==y['cols']
  checks['distinct_grid']=unique(x['rows'])==unique(y['rows']) and unique(x['cols'])==unique(y['cols'])
 if kind=='factors':
  checks['rank']=x['rank']==y['rank']
  checks['pivot_selection']=x['selected_rows']==y['selected_rows'] and x['selected_cols']==y['selected_cols']
  checks['pivot_errors']=x['pivot_errors']==y['pivot_errors']
 if kind=='guard':checks['guard_points']=x['natural']==y['natural']
 if kind=='pass_error': checks['held_out_error']=abs(x['importance_l2']-y['importance_l2'])<=1e-14
 for check,passed in checks.items():
  if not passed:
   first.setdefault(check,{'pass':x['pass'],'step':x['step'],'cut':x['cut'],'kind':kind})
   all_diffs.setdefault(check,[]).append([x['pass'],x['step'],x['cut'],kind])
result={'left':left,'right':right,'common_records':len(a.keys()&b.keys()),'first_differences':first,'difference_counts':{k:len(v) for k,v in all_diffs.items()}}
(out/(left+'-vs-'+right+'.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
for check in ['distinct_candidates','pivot_selection','held_out_error']:
 if check not in first:continue
 key=first[check];k=(key['pass'],key['step'],key['kind']);x,y=a[k],b[k]
 detail={'left':x,'right':y}
 if key['kind']=='after_guard':
  detail['changed_cuts']=[{'cut':i,'left':u,'right':v,'left_distinct':unique(u),'right_distinct':unique(v)} for i,(u,v) in enumerate(zip(x['candidates'],y['candidates'])) if unique(u)!=unique(v)]
 (out/(left+'-vs-'+right+'-first-'+check+'.json')).write_text(json.dumps(detail,indent=2)+'\n')
