"""Summarize target accuracy, empirical interval coverage, and weak identification."""
from pathlib import Path
import csv,json
import numpy as np
from experiment import wilson

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'


def metric(estimates,intervals,target):
    covered=(intervals[:,0]<=target+1e-10)&(intervals[:,1]>=target-1e-10)
    lo,hi=wilson(covered.sum(),len(covered))
    result={'coverage_pct':100*float(covered.mean()),'coverage_MC_CI_pct':[100*float(lo),100*float(hi)],
            'width_median_pp':100*float(np.median(intervals[:,1]-intervals[:,0])),
            'width_p95_pp':100*float(np.quantile(intervals[:,1]-intervals[:,0],.95)),
            'width_mean_pp':100*float(np.mean(intervals[:,1]-intervals[:,0]))}
    if estimates is not None:
        error=abs(estimates-target)
        half=1.959963984540054*np.std(error,ddof=1)/np.sqrt(len(error)) if len(error)>1 else np.nan
        result.update(MAE_pp=100*float(error.mean()),MAE_MC_CI_pp=[100*float(error.mean()-half),100*float(error.mean()+half)],bias_pp=100*float(np.mean(estimates-target)))
    return result


def safe_inverse(counts):
    """Undefined finite-sample inverses are flagged rather than silently projected."""
    c=np.asarray(counts,float)
    if np.any(c[:,1]<=0):return None,'empty_basis'
    y=c[:,0]/c[:,1];denominator=1-2*y[0]
    if abs(denominator)<=1e-12:return None,'singular_denominator'
    discriminant=1-2*(y[1]-y[0])/denominator
    if discriminant<=1e-12:return None,'nonpositive_or_singular_discriminant'
    p=1-np.sqrt(discriminant);r=(4*y[0]-p)/(1-p/2)
    return (p,r),'valid'


def main():
    rows=[];profiles=[]
    for path in sorted(OUT.glob('*fit.npz'))+sorted(OUT.glob('retained_*_profile.npz')):
        with np.load(path) as z:d={k:z[k] for k in z.files}
        target=float(d['effective_target']);estimates=d.get('estimates');intervals=d['intervals']
        row={'record':path.name,'circuit':str(d.get('circuit','PRIMARY')),'scenario':str(d['scenario']),
             'method':str(d['method']),'replicates':len(intervals),'readout':float(d['readout']),
             'target':target,'point_identified':estimates is not None,**metric(estimates,intervals,target)}
        if float(d['readout'])>0:
            row['Eve_only_target']=metric(estimates,intervals,float(d['target']))
        if 'condition' in d:
            row.update(condition_median=float(np.median(d['condition'])),
                       condition_p95=float(np.quantile(d['condition'],.95,method='higher')),
                       rank_deficient_pct=100*float(np.mean(d['information_rank']<(2 if str(d['method'])=='basis' else 3))),
                       noise_boundary_pct=100*float(d['boundary'].mean()),
                       target_boundary_pct=100*float(d['target_boundary'].mean()),
                       parameter_se_median=np.median(d['parameter_standard_errors'],axis=0).tolist())
            row['boundary_strata']={}
            for label,mask in [('boundary',d['boundary']),('interior',~d['boundary'])]:
                row['boundary_strata'][label]={'count':int(mask.sum())}
                if mask.any():row['boundary_strata'][label].update(metric(estimates[mask],intervals[mask],target))
        if str(d['method'])=='retained_profile':
            oldi=d['old_estimates'][:,None]+1.959963984540054*d['old_standard_errors'][:,None]*np.array([-1,1])
            row['WLS_Wald']=metric(d['old_estimates'],oldi,target)
            row['old_parameter_se_median']=np.median(d['old_parameter_standard_errors'],axis=0).tolist()
            nuisance=path.with_name(path.stem.replace('_profile','_nuisance')+'.npz')
            with np.load(nuisance) as z:ni=z['intervals']
            truth={'equal':(.03,.03),'off_grid':(.025,.025),'alice_low':(.01,.05),
                   'bob_low':(.05,.01),'curvature':(.03,.03)}[str(d['scenario'])]
            row['device_profiles']={name:metric(d['parameters'][:,i],ni[:,i,:],truth[i])
                                    for i,name in enumerate(['pA','pB'])}
            profiles.append(row)
        else:rows.append(row)
    validity={}
    folder=HERE.parent/'diagnostic_validation/results'
    for path in sorted(folder.glob('*basis_single.npz')):
        with np.load(path) as z:
            for c in z['counts']:
                _,status=safe_inverse(c);validity[status]=validity.get(status,0)+1
    assert validity=={'valid':31000},validity
    edge_cases=[([[0,0],[1,20]],'empty_basis'),([[5,10],[2,10]],'singular_denominator'),
                ([[0,10],[8,10]],'nonpositive_or_singular_discriminant')]
    for counts,expected in edge_cases:assert safe_inverse(counts)[1]==expected
    result={'cross_circuit':rows,'retained_profiles':profiles,'inverse_validity':validity,
            'inverse_edge_cases_checked':len(edge_cases),
            'scope':'Fixed cells, equal transmitted budgets; uncertainty is Monte Carlo precision within this design. Profile sets use physical bounds and an asymptotic cutoff, with observed coverage reported.'}
    contrasts=[]
    for path in sorted(OUT.glob('*_full_fit.npz')):
        with np.load(path) as z:
            if 'estimates' not in z:continue
            target=float(z['effective_target']);first=abs(z['estimates']-target)
        with np.load(path.with_name(path.name.replace('_full_fit','_scaled_fit'))) as z:
            second=abs(z['estimates']-target)
        delta=float(second.mean()-first.mean())
        half=1.959963984540054*np.sqrt((np.var(first,ddof=1)+np.var(second,ddof=1))/len(first))
        contrasts.append({'record':path.name,'scaled_minus_full_MAE_pp':100*delta,
                          'MC95CI_pp':[100*(delta-half),100*(delta+half)]})
    result['independent_acquisition_contrasts']=contrasts
    def clean(value):
        if isinstance(value,float) and not np.isfinite(value):return None
        if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
        if isinstance(value,list):return [clean(v) for v in value]
        return value
    (OUT/'summary.json').write_text(json.dumps(clean(result),indent=2,allow_nan=False)+'\n')
    fields=['circuit','scenario','method','readout','point_identified','MAE_pp','coverage_pct','width_median_pp','width_p95_pp','noise_boundary_pct','condition_median']
    with (OUT/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(clean(rows+profiles))
    for row in rows+profiles:
        print(row['circuit'],row['scenario'],row['method'],row['readout'],
              'MAE',round(row.get('MAE_pp',float('nan')),3),'coverage',round(row['coverage_pct'],1),
              'widths',round(row['width_median_pp'],2),round(row['width_p95_pp'],2),flush=True)


if __name__=='__main__':main()
