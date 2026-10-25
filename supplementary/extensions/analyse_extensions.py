"""Cross-matrix uncertainty, sample costs, physical controls and diagnostics."""
import json
import numpy as np
from scipy.stats import t as student_t
from common import OUT,P,entropy_benchmark
from analyse import METHODS,paired,csvfile,wilson
from provenance import atomic_json
from inventory import check_inventory, record_fingerprint


def matrix_interval(values):
    values=np.asarray(values,float);K=len(values);mean=float(values.mean())
    radius=float(student_t.ppf(.975,K-1)*values.std(ddof=1)/np.sqrt(K)) if K>1 else None
    return dict(matrices=K,mean=mean,ci=[mean-radius,mean+radius] if radius is not None else None,
                matrix_values=values.tolist())


def code_transfer():
    rng=np.random.default_rng(49823);rows=[];pipeline=[]
    for family in ['gallager','irregular']:
        for n in P['codes']['lengths']:
            for k in P['codes']['test_seeds']:
                for ci,c in enumerate(P['transfer']['conditions']):
                    tag=f'transfer_{family}_n{n}_k{k}_c{ci}'
                    e=np.load(OUT/f'{tag}_evaluation.npz',allow_pickle=False);m=int(e['m']);s=c['s']
                    for j,method in enumerate(METHODS):
                        rows.append(dict(family=family,n=n,matrix_seed=k,condition=ci,asymmetry=c['a'],sample=s,
                            method=method,m=m,**paired(e['correct'][0],e['correct'][j],(n-m-64)/(n+s),(n-m-64)/(n+s),rng)))
                    u=np.load(OUT/f'{tag}_pipeline_pooled.npz',allow_pickle=False)
                    a=np.load(OUT/f'{tag}_pipeline_selected.npz',allow_pickle=False)
                    pipeline.append(dict(family=family,n=n,matrix_seed=k,condition=ci,asymmetry=c['a'],sample=s,
                        mean_rows_pooled=float(u['m'].mean()),mean_rows_selected=float(a['m'].mean()),
                        mean_iterations_pooled=float(u['iterations'].mean()),mean_iterations_selected=float(a['iterations'].mean()),
                        **paired(u['correct'],a['correct'],(n-u['m']-64)/(n+s),(n-a['m']-64)/(n+s),rng)))
    csvfile('transfer_results.csv',rows);csvfile('pipeline_results.csv',pipeline)
    aggregate=[]
    for family in ['gallager','irregular']:
        for n in P['codes']['lengths']:
            for ci in range(3):
                for method in ['separate','shrink','selected','oracle_basis']:
                    rr=[r for r in rows if r['family']==family and r['n']==n and r['condition']==ci and r['method']==method]
                    aggregate.append(dict(family=family,n=n,condition=ci,method=method,
                                          **matrix_interval([r['delta_yield'] for r in rr])))
                rr=[r for r in pipeline if r['family']==family and r['n']==n and r['condition']==ci]
                aggregate.append(dict(family=family,n=n,condition=ci,method='adaptive_pipeline',
                                      **matrix_interval([r['delta_yield'] for r in rr])))
    atomic_json(OUT/'transfer_summary.json',aggregate)
    return rows,pipeline,aggregate


def other_controls():
    rng=np.random.default_rng(49824);fixed=[];physical=[]
    for family in ['gallager','irregular']:
        for k in P['codes']['test_seeds']:
            for s in P['fixed_total']['sample']:
                tag=f'fixedtotal_{family}_k{k}_s{s}';n=2000-s
                u=np.load(OUT/f'{tag}_pipeline_pooled.npz',allow_pickle=False)
                a=np.load(OUT/f'{tag}_pipeline_selected.npz',allow_pickle=False)
                fixed.append(dict(family=family,matrix_seed=k,sample=s,n=n,
                    mean_rows_pooled=float(u['m'].mean()),mean_rows_selected=float(a['m'].mean()),
                    **paired(u['correct'],a['correct'],(n-u['m']-64)/2000,(n-a['m']-64)/2000,rng)))
            for model in ['homogeneous','gate_dependent','pauli','thermal','readout']:
                for setting in ['fixed','matched']:
                    tag=f'physical_{family}_k{k}_{model}_{setting}'
                    e=np.load(OUT/f'{tag}_evaluation.npz',allow_pickle=False)
                    d=np.load(OUT/f'{tag}_frames.npz',allow_pickle=False);m=int(e['m'])
                    ent=entropy_benchmark(d['transitions'])
                    for j,method in enumerate(e['methods']):
                        physical.append(dict(family=family,matrix_seed=k,model=model,setting=setting,
                            method=str(method),m=m,qbar=ent['qbar'],G=ent['G'],
                            conditional_entropy=ent['conditional_entropy'],
                            **paired(e['correct'][0],e['correct'][j],(1000-m-64)/1200,(1000-m-64)/1200,rng)))
    csvfile('fixed_total_results.csv',fixed);csvfile('physical_results.csv',physical)
    summary=[]
    for family in ['gallager','irregular']:
        for s in P['fixed_total']['sample']:
            rr=[r for r in fixed if r['family']==family and r['sample']==s]
            summary.append(dict(experiment='fixed_total',family=family,sample=s,
                delta=matrix_interval([r['delta_yield'] for r in rr]),
                pooled_yield=matrix_interval([r['base_yield'] for r in rr]),
                selected_yield=matrix_interval([r['yield_mean'] for r in rr])))
        for model in ['homogeneous','gate_dependent','pauli','thermal','readout']:
            for setting in ['fixed','matched']:
                for method in ['separate','shrink','selected','transition']:
                    rr=[r for r in physical if r['family']==family and r['model']==model and r['setting']==setting and r['method']==method]
                    summary.append(dict(experiment='physical',family=family,model=model,setting=setting,method=method,
                                        **matrix_interval([r['delta_yield'] for r in rr])))
    atomic_json(OUT/'controls_summary.json',summary)
    return fixed,physical,summary


def diagnostics():
    rows=[];rng=np.random.default_rng(49825)
    for pattern in ['diagnostic_p*.npz','devicecase_*.npz','diagnostic_stress_*.npz']:
        for path in sorted(OUT.glob(pattern)):
            d=np.load(path,allow_pickle=False);truth=float(d['target']);v=d['estimates'];se=d['standard_errors']
            for j,method in enumerate(d['methods']):
                error=v[:,j]-truth;covered=abs(error)<=1.959963984540054*se[:,j]
                boot=np.concatenate([abs(error)[rng.integers(len(error),size=(100,len(error)))].mean(1) for _ in range(20)])
                paired_difference=abs(error)-abs(v[:,4]-truth)
                pairboot=np.concatenate([paired_difference[rng.integers(len(error),size=(100,len(error)))].mean(1) for _ in range(20)])
                mae=float(np.mean(abs(error)));coverage=float(covered.mean())
                rows.append(dict(file=path.name,experiment='grid' if pattern.startswith('diagnostic_p') else 'device_case' if pattern.startswith('devicecase') else 'stress',
                    method=str(method),p=float(d['p']),attack=float(d['r']),pE=float(d['pE']),case=str(d['case']),stress=str(d['stress']),
                    budget=int(d['budget']),replicates=len(error),target=truth,ideal_attack=float(d['ideal_attack']),
                    MAE=mae,MAE_ci=np.quantile(boot,[.025,.975]).tolist(),RMSE=float(np.sqrt(np.mean(error**2))),
                    MAE_minus_calibrated_gate=float(paired_difference.mean()),
                    MAE_difference_ci=np.quantile(pairboot,[.025,.975]).tolist(),
                    bias=float(error.mean()),coverage=coverage,coverage_ci=wilson(int(covered.sum()),len(error)),
                    mean_interval_width=float(np.mean(2*1.959963984540054*se[:,j])),
                    reliable=mae<=.01 and coverage>=.9))
    csvfile('diagnostic_results.csv',rows)
    atomic_json(OUT/'diagnostic_summary.json',rows)
    return rows


def plots(transfer,controls,diagnostic):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    images=OUT/'figures';images.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
    for ci,ax in enumerate(axes):
        for family,color in [('gallager','#0072B2'),('irregular','#D55E00')]:
            for method,marker,offset in [('selected','o',-.03),('adaptive_pipeline','s',.03)]:
                rr=[r for r in transfer if r['family']==family and r['condition']==ci and r['method']==method]
                x=np.array([r['n'] for r in rr],float)*10**offset;y=100*np.array([r['mean'] for r in rr])
                lo=100*np.array([r['ci'][0] for r in rr]);hi=100*np.array([r['ci'][1] for r in rr])
                ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt=marker+'-',capsize=3,color=color,alpha=.8,label=f'{family}, {method.replace("adaptive_pipeline","pipeline")}')
        c=P['transfer']['conditions'][ci];ax.set_title(f'a={c["a"]}, s={c["s"]}');ax.set_xscale('log',base=4)
        ax.set_xticks([1000,4000,16000],['1000','4000','16000']);ax.axhline(0,color='.5',lw=.8)
        ax.set_xlabel('Decoded block length');ax.set_ylabel('Paired yield change (pp)')
    axes[0].legend(fontsize=7);fig.suptitle('Transfer to unseen codes: 95% t intervals across five matrix seeds')
    fig.savefig(images/'code_transfer.png',dpi=180);fig.savefig(images/'code_transfer.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
    for ax,family in zip(axes,['gallager','irregular']):
        rr=[r for r in controls if r['experiment']=='fixed_total' and r['family']==family]
        for key,color in [('pooled_yield','#777777'),('selected_yield','#0072B2')]:
            x=[r['sample'] for r in rr];y=np.array([r[key]['mean'] for r in rr])*100
            lo=np.array([r[key]['ci'][0] for r in rr])*100;hi=np.array([r[key]['ci'][1] for r in rr])*100
            ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt='o-',capsize=3,color=color,label=key.replace('_yield',''))
        ax.set_title(family);ax.set_xlabel('Sacrificed sample');ax.set_ylabel('Absolute reconciliation yield (%)');ax.legend()
    fig.suptitle('Fixed total of 2000 sifted bits: sample accuracy versus retained length')
    fig.savefig(images/'sample_cost.png',dpi=180);fig.savefig(images/'sample_cost.pdf');plt.close(fig)
    methods=['linear_four','gate_four','gate_two','calibrated_BSC','calibrated_gate','Fiorini']
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for ax,method in zip(axes.flat,methods):
        rr=[r for r in diagnostic if r['experiment']=='grid' and r['method']==method]
        grid=np.array([[sum(r['reliable'] for r in rr if r['p']==p and r['budget']==b) for b in P['diagnostics']['budgets']] for p in P['diagnostics']['p_legitimate']])
        ax.imshow(grid,origin='lower',vmin=0,vmax=4,cmap='Blues',aspect='auto')
        for i in range(4):
            for j in range(3):ax.text(j,i,f'{grid[i,j]}/4',ha='center',va='center',color='white' if grid[i,j]>=3 else 'black')
        ax.set_xticks(range(3),P['diagnostics']['budgets']);ax.set_yticks(range(4),P['diagnostics']['p_legitimate'])
        ax.set_title(method.replace('_',' '));ax.set_xlabel('Transmitted signals');ax.set_ylabel('Legitimate depolarizing p')
    fig.suptitle('Diagnostic reliability across four attack fractions\nCells count MAE ≤ 1 pp and empirical individual-interval coverage ≥ 90%')
    fig.savefig(images/'diagnostic_reliability.png',dpi=180);fig.savefig(images/'diagnostic_reliability.pdf');plt.close(fig)


def main():
    check_inventory()
    _,_,transfer=code_transfer();_,_,controls=other_controls();diag=diagnostics();plots(transfer,controls,diag)
    atomic_json(OUT/'analysis_provenance.json',dict(record_fingerprint=record_fingerprint(),
        transfer_rows=len(transfer),control_rows=len(controls),diagnostic_rows=len(diag)))
    print('All extension tables and figures generated',flush=True)


if __name__=='__main__':main()
