"""Summaries and Monte Carlo uncertainty for the frozen diagnostic validation."""
from pathlib import Path
import csv
import json
import numpy as np
from scipy.stats import norm
from models import unequal_response,unequal_jacobian,equal_jacobian,observations
from experiment import P,OUT

Z=norm.ppf(.975)


def wilson(k,n):
    p=k/n;den=1+Z*Z/n;centre=(p+Z*Z/(2*n))/den
    half=Z*np.sqrt(p*(1-p)/n+Z*Z/(4*n*n))/den
    return [float(centre-half),float(centre+half)]


def mean_interval(values):
    values=np.asarray(values);mean=float(values.mean());se=float(values.std(ddof=1)/np.sqrt(len(values)))
    return mean,[mean-Z*se,mean+Z*se]


def cell_summary(d):
    error=d['estimates']-float(d['target']);se=d['standard_errors'];covered=abs(error)<=Z*se
    assert len(error)==P['replicates'] and np.isfinite(error).all() and np.isfinite(se).all() and (se>0).all()
    mae,ci=mean_interval(abs(error)*100)
    return dict(MAE_pp=mae,MAE_MC_CI_pp=ci,bias_pp=float(error.mean()*100),
        RMSE_pp=float(np.sqrt(np.mean(error**2))*100),coverage=float(covered.mean()),
        coverage_MC_CI=wilson(int(covered.sum()),len(error)),mean_interval_width_pp=float(200*Z*se.mean()))


def grid_summary():
    rows=[];errors={};moments={}
    for stage in ['nominal','off_grid']:
        cfg=P[stage]
        for pi,p in enumerate(cfg['p']):
            for ri,r in enumerate(cfg['r']):
                for bi,budget in enumerate(cfg['budgets']):
                    key=(stage,pi,ri,bi)
                    for method in P['methods']:
                        path=OUT/f'{stage}_p{pi}_r{ri}_b{bi}_{method}.npz'
                        with np.load(path,allow_pickle=False) as file:d={k:file[k] for k in file.files}
                        assert (float(d['p']),float(d['r']),int(d['budget']))==(p,r,budget)
                        row=dict(file=path.name,stage=stage,p=p,r=r,budget=budget,method=method,replicates=P['replicates'],**cell_summary(d))
                        rows.append(row);errors[key,method]=abs(d['estimates']-r/4)
                        if method=='basis_single':
                            assert d['moment_valid'].all(),'Report invalid analytic-inverse draws before summarizing.'
                            alt=dict(d,estimates=d['moment_estimates'],standard_errors=d['moment_standard_errors'])
                            moments[key]=cell_summary(alt)
                            rows.append(dict(row,method='basis_inverse',**moments[key]))
                            errors[key,'basis_inverse']=abs(d['moment_estimates']-r/4)
    aggregate=[]
    for stage in ['nominal','off_grid']:
        for budget in [None,*P[stage]['budgets']]:
            keys=sorted({key for key,method in errors if key[0]==stage and (budget is None or P[stage]['budgets'][key[3]]==budget)})
            for method in [*P['methods'],'basis_inverse']:
                sample=[errors[key,method] for key in keys]
                mean=float(np.mean(sample));se=float(np.sqrt(sum(x.var(ddof=1)/len(x) for x in sample))/len(sample))
                cells=[row for row in rows if row['stage']==stage and row['method']==method and (budget is None or row['budget']==budget)]
                out=dict(stage=stage,budget=budget,method=method,cells=len(keys),MAE_pp=100*mean,
                    MAE_MC_CI_pp=[100*(mean-Z*se),100*(mean+Z*se)],
                    average_coverage=float(np.mean([v['coverage'] for v in cells])),
                    minimum_cell_coverage=min(v['coverage'] for v in cells),
                    maximum_cell_coverage=max(v['coverage'] for v in cells))
                for comparator in ['basis_single','calibrated']:
                    differences=[errors[key,method]-errors[key,comparator] for key in keys]
                    d=float(np.mean(differences));ds=float(np.sqrt(sum(x.var(ddof=1)/len(x) for x in differences))/len(keys))
                    out['minus_'+comparator]=dict(MAE_pp=100*d,MC_CI_pp=[100*(d-Z*ds),100*(d+Z*ds)])
                aggregate.append(out)
    moment_summary={}
    for stage in ['nominal','off_grid']:
        rr=[v for key,v in moments.items() if key[0]==stage]
        moment_summary[stage]=dict(MAE_pp=float(np.mean([r['MAE_pp'] for r in rr])),
            average_coverage=float(np.mean([r['coverage'] for r in rr])),invalid_replicates=0)
    return rows,aggregate,moment_summary


def holdout_summary():
    rows=[]
    for scenario in P['holdout']['scenarios']:
        for method in ['basis_single','aggregate_two_cap4','aggregate_four_cap4','independent_devices']:
            path=OUT/f'holdout_{scenario["name"]}_{method}.npz'
            with np.load(path,allow_pickle=False) as file:d={k:file[k] for k in file.files}
            f=float(d['validation_scale']);pars=d['parameters'];cov=d['covariance']
            if method=='independent_devices':
                pred=np.array([unequal_response(f,*p) for p in pars])
                jac=np.array([unequal_jacobian(f,*p) for p in pars])
                bound=np.any((pars[:,:2]<1e-7)|(pars[:,:2]>min(.3,.999/4)-1e-6),axis=1)
            else:
                pred=unequal_response(f,pars[:,0],pars[:,0],pars[:,1])
                jac=equal_jacobian(pars[:,0],pars[:,1],[f])[:,0,:]
                bound=(pars[:,0]<1e-7)|(pars[:,0]>min(.3,.999/float(d['scales'].max()))-1e-6)
            prediction_variance=np.einsum('ni,nij,nj->n',jac,cov,jac)
            assert (prediction_variance>=0).all()
            observed,variance=observations(d['validation'])
            prediction_covered=abs(observed-pred)<=Z*np.sqrt(variance+prediction_variance)
            rows.append(dict(scenario=scenario['name'],method=method,replicates=P['replicates'],
                total_signals=int(d['training_budget'])+int(d['validation_budget']),
                **cell_summary(d),prediction_coverage=float(prediction_covered.mean()),
                prediction_coverage_MC_CI=wilson(int(prediction_covered.sum()),len(prediction_covered)),
                prediction_bias_pp=float(100*np.mean(pred-float(d['true_validation_qber']))),
                prediction_MAE_pp=float(100*np.mean(abs(pred-float(d['true_validation_qber'])))),
                noise_bound_fraction=float(bound.mean())))
    return rows


def main():
    cells,aggregate,moments=grid_summary();holdout=holdout_summary()
    report=dict(status='passed',grid_cells=cells,grid_aggregate=aggregate,
                analytic_inverse_sensitivity=moments,held_out=holdout,
                uncertainty='95% Monte Carlo intervals condition on the listed cells and frozen designs. They exclude design selection and deployment variation; comparisons are not equivalence tests.',
                inventory=dict(nominal_cells=48,off_grid_cells=9,replicates_per_cell=500,
                    grid_replicate_blocks=28500,grid_budgeted_acquisitions=228000,
                    held_out_scenarios=5,held_out_replicate_blocks=2500,
                    held_out_budgeted_training_acquisitions=7500,
                    held_out_acquisitions_reused_for_four_setting_model_fits=True))
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    for name,rows in [('cell-results.csv',cells),('held-out-results.csv',holdout)]:
        with (OUT/name).open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({'status':'passed','methods':len(P['methods']),'cells':len(cells),'held_out':len(holdout),'inventory':report['inventory']},indent=2))


if __name__=='__main__':main()
