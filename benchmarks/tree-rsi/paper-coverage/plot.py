"""Export scientific plots from independently validated raw observations."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fixtures import OUT
from summarize import records

def main():
    folders=[sorted(OUT.glob(pattern))[-1] for pattern in ('run-dmrg-*','run-functions-*','run-tree-*','supplement-*','gpe-tolerance-*')]
    rows=[r for folder in folders for r in records(folder)]
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    fixtures=['dmrg-n20-chi30','gpe-91-vortices','complex-tree-n20-m2-chi16-root0']
    titles=['Spin-1 DMRG: n=20, input rank30','Complex GPE: n=26, input rank65','Complex branching tree: n=20, input16x16']
    for ax,fixture,title in zip(axes,fixtures,titles):
        for algorithm,color in [('treeaci','#1764ab'),('rsi','#bf4b20')]:
            selected=[r for r in rows if r['case']['fixture']==fixture and r['algorithm']==algorithm and r['case']['options'].get('local_tolerance',1e-12)==1e-12 and r['status']=='completed']
            caps=sorted({r['case']['cap'] for r in selected});values=[[r['measured_error'] for r in selected if r['case']['cap']==cap] for cap in caps]
            median=[np.median(v) for v in values];low=[min(v) for v in values];high=[max(v) for v in values]
            ax.plot(caps,median,'o-',color=color,label=algorithm+' local tol1e-12');ax.fill_between(caps,low,high,color=color,alpha=.16)
        if fixture=='gpe-91-vortices':
            selected=[r for r in rows if r['case']['fixture']==fixture and r['algorithm']=='rsi' and r['case']['options'].get('local_tolerance')==0 and r['status']=='completed']
            caps=sorted({r['case']['cap'] for r in selected});values=[[r['measured_error'] for r in selected if r['case']['cap']==cap] for cap in caps]
            ax.plot(caps,[np.median(v) for v in values],'s--',color='#96329a',label='rsi local tol0');ax.fill_between(caps,[min(v) for v in values],[max(v) for v in values],color='#96329a',alpha=.12)
        ax.axhline(1e-8,color='0.4',ls=':',label='global target1e-8');ax.set_yscale('log');ax.set_title(title);ax.set_xlabel('Configured output rank cap');ax.set_ylabel('Global relative L2 error');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Primary settings and labeled sensitivity; bands span all3 seeds. Insufficient caps are retained.',fontsize=11)
    fig.savefig(OUT/'global-error-curves.svg');fig.savefig(OUT/'global-error-curves.png',dpi=180);plt.close(fig)
    folder=sorted(p for p in OUT.glob('matched-*') if p.is_dir())[-1];timing=records(folder)
    if len(timing)!=30 or not all(r['accepted'] for r in timing):raise ValueError('matched timing requires30 accepted global checks')
    fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
    for algorithm,color,offset in [('treeaci','#1764ab',-.12),('rsi','#bf4b20',.12)]:
        for seed in (1,2,3):
            times=[r['seconds'] for r in timing if r['algorithm']==algorithm and r['seed']==seed]
            x=seed+offset;ax.scatter(x+np.linspace(-.035,.035,len(times)),times,c=color,s=22,label=algorithm if seed==1 else None)
            ax.plot([x-.07,x+.07],[np.median(times)]*2,color=color,lw=2)
    ax.set_xticks([1,2,3]);ax.set_xlabel('Seed');ax.set_ylabel('Public API seconds');ax.set_yscale('log');ax.grid(alpha=.2);ax.legend()
    ax.set_title('DMRG n20 chi30, cap470; local tol1e-14; RSI k221\nAll30 calls pass global1e-8;5 measured calls per seed')
    fig.savefig(OUT/'matched-dmrg-timing.svg');fig.savefig(OUT/'matched-dmrg-timing.png',dpi=180);plt.close(fig)
if __name__=='__main__':main()
