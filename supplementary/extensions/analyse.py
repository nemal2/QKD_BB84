"""Canonical paired analysis and scientific plots for every asymmetry test cell."""
import csv
import json
import numpy as np
from scipy.stats import binomtest,spearmanr
from common import OUT,P,entropy_benchmark
from provenance import atomic_json
from inventory import check_inventory, record_fingerprint

METHODS=['pooled','separate','shrink','selected','oracle_pool','oracle_basis']


def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n;c=(p+z*z/(2*n))/den
    d=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(max(0,c-d)),float(min(1,c+d))]


def paired(base,other,ybase=1.,yother=1.,rng=None):
    rng=np.random.default_rng(38124) if rng is None else rng
    a=np.asarray(base,bool);b=np.asarray(other,bool);F=len(a)
    ya=np.broadcast_to(ybase,a.shape);yb=np.broadcast_to(yother,b.shape)
    delta=b*yb-a*ya
    # Resample joint outcomes; constant row counts admit an exact multinomial shortcut.
    if np.ptp(ya)==0 and np.ptp(yb)==0:
        ct=np.bincount(a.astype(int)*2+b.astype(int),minlength=4)
        draws=rng.multinomial(F,ct/F,size=10000)/F
        means=yb[0]*(draws[:,1]+draws[:,3])-ya[0]*(draws[:,2]+draws[:,3])
    else:
        means=np.concatenate([delta[rng.integers(F,size=(100,F))].mean(1) for _ in range(100)])
    rescued=int(np.sum(~a & b));harmed=int(np.sum(a & ~b))
    p=binomtest(rescued,rescued+harmed,.5).pvalue if rescued+harmed else 1.
    return dict(frames=F,base_failures=int((~a).sum()),failures=int((~b).sum()),
                fer=float((~b).mean()),fer_ci=wilson(int((~b).sum()),F),
                rescued=rescued,harmed=harmed,p_exact=float(p),
                yield_mean=float(np.mean(b*yb)),base_yield=float(np.mean(a*ya)),
                delta_yield=float(delta.mean()),delta_yield_ci=np.quantile(means,[.025,.975]).tolist())


def holm(rows):
    order=sorted(range(len(rows)),key=lambda i:rows[i]['p_exact']);running=0.
    for j,i in enumerate(order):
        running=max(running,min(1.,rows[i]['p_exact']*(len(rows)-j)))
        rows[i]['p_holm']=running


def csvfile(name,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/name).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,keys);writer.writeheader();writer.writerows(rows)


def asymmetry():
    rows=[];rng=np.random.default_rng(390823)
    for qi,q in enumerate(P['pilot']['qber']):
        for ai,a in enumerate(P['pilot']['asymmetry']):
            for si,s in enumerate(P['pilot']['sample']):
                tag=f'confirm_q{qi}_a{ai}_s{si}'
                d=np.load(OUT/f'{tag}_frames.npz',allow_pickle=False)
                e=np.load(OUT/f'{tag}_evaluation.npz',allow_pickle=False)
                n=int(e['n']);m=int(e['m']);yield_factor=(n-m-64)/(n+s)
                ent=entropy_benchmark(d['transitions'])
                theoretical_variance=2*sum(t*(1-t) for t in d['transitions'].mean(1))/s
                for j,method in enumerate(METHODS):
                    row=dict(qber=q,asymmetry=a,sample=s,n=n,m=m,matrix_seed=29,
                             method=method,G=ent['G'],nG=n*ent['G'],
                             population_SNR2=(2*q*a)**2/theoretical_variance,
                             **paired(e['correct'][0],e['correct'][j],yield_factor,yield_factor,rng))
                    row['mean_iterations']=float(e['iterations'][j].mean())
                    row['undetected_errors']=int(np.sum(e['accepted'][j] & ~e['correct'][j]))
                    rows.append(row)
    for method in METHODS:holm([r for r in rows if r['method']==method])
    csvfile('asymmetry_results.csv',rows)
    summary=[]
    for method in METHODS[1:]:
        rr=[r for r in rows if r['method']==method];eq=[r for r in rr if r['asymmetry']==0]
        summary.append(dict(method=method,cells=len(rr),significant_gains=sum(r['p_holm']<.05 and r['rescued']>r['harmed'] for r in rr),
                            significant_harms=sum(r['p_holm']<.05 and r['rescued']<r['harmed'] for r in rr),
                            mean_delta_yield=float(np.mean([r['delta_yield'] for r in rr])),
                            symmetric_mean_delta_yield=float(np.mean([r['delta_yield'] for r in eq])),
                            best_cell=max(rr,key=lambda r:r['delta_yield']),
                            worst_cell=min(rr,key=lambda r:r['delta_yield'])))
    correlations=[]
    for method in ['separate','shrink','selected','oracle_basis']:
        rr=[r for r in rows if r['method']==method]
        for variable in ['asymmetry','nG','population_SNR2']:
            correlations.append(dict(method=method,variable=variable,
                spearman_rho=float(spearmanr([r[variable] for r in rr],[r['delta_yield'] for r in rr]).statistic)))
    result=dict(cells=75,frames=37500,decoder_outputs=225000,method_summaries=summary,
                record_fingerprint=record_fingerprint(),
                descriptive_correlations=correlations,
                scope='Independent frames conditional on one fixed matrix per QBER; cross-matrix generalization requires the transfer experiment.')
    atomic_json(OUT/'asymmetry_summary.json',result)
    return rows,result


def plots(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    images=OUT/'figures';images.mkdir(exist_ok=True)
    methods=['separate','shrink','selected']
    limit=max(abs(100*r['delta_yield']) for r in rows if r['method'] in methods)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,3,figsize=(12,10),layout='constrained')
    for qi,q in enumerate(P['pilot']['qber']):
        for j,method in enumerate(methods):
            ax=axes[qi,j]
            rr=[r for r in rows if r['qber']==q and r['method']==method]
            grid=np.array([[next(r['delta_yield'] for r in rr if r['asymmetry']==a and r['sample']==s)
                            for a in P['pilot']['asymmetry']] for s in P['pilot']['sample']])*100
            im=ax.imshow(grid,cmap='RdBu',norm=TwoSlopeNorm(0,-limit,limit),origin='lower',aspect='auto')
            ax.set_xticks(range(5),P['pilot']['asymmetry']);ax.set_yticks(range(5),P['pilot']['sample'])
            ax.set_title(f'{100*q:.0f}% QBER · {method.replace("selected","frozen rule")}')
            ax.set_xlabel('Basis asymmetry a');ax.set_ylabel('Disclosed sample s')
            for r in rr:
                if r['p_holm']<.05:ax.text(P['pilot']['asymmetry'].index(r['asymmetry']),P['pilot']['sample'].index(r['sample']),'*',ha='center',va='center',fontsize=16)
    fig.colorbar(im,ax=axes,label='Paired reconciliation yield change (percentage points)',shrink=.75)
    fig.suptitle('Finite-sample basis information: benefit and estimation cost\n* Holm-adjusted paired test p < 0.05 within each method’s 75 cells',fontsize=14)
    fig.savefig(images/'asymmetry_heatmaps.png',dpi=180);fig.savefig(images/'asymmetry_heatmaps.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    colors={.02:'#0072B2',.06:'#D55E00',.10:'#009E73'}
    for method,marker in [('separate','x'),('selected','o')]:
        for q,color in colors.items():
            rr=[r for r in rows if r['method']==method and r['qber']==q]
            for ax,key in zip(axes,['nG','population_SNR2']):
                ax.scatter([r[key] for r in rr],[100*r['delta_yield'] for r in rr],s=30,marker=marker,alpha=.7,color=color,label=f'{method}, {100*q:.0f}%')
    for ax,label in zip(axes,['Information opportunity nG (bits)','Population basis-contrast SNR²']):
        ax.set_xscale('symlog',linthresh=.1);ax.axhline(0,color='.5',linewidth=.8)
        ax.set_xlabel(label);ax.set_ylabel('Paired yield change (pp)')
    axes[1].legend(fontsize=8,ncols=2);fig.suptitle('Descriptive channel benchmarks; these are not policy inputs')
    fig.savefig(images/'information_benchmark.png',dpi=180);fig.savefig(images/'information_benchmark.pdf');plt.close(fig)


def main():
    check_inventory()
    rows,summary=asymmetry();plots(rows)
    for row in summary['method_summaries']:
        print(row['method'],row['significant_gains'],row['significant_harms'],100*row['mean_delta_yield'],flush=True)


if __name__=='__main__':main()
