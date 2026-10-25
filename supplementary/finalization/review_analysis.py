"""Additional reporting from retained records; no new stochastic experiments.

Uncertainty in grid averages is Monte Carlo uncertainty conditional on the fixed
design cells. Transfer timing is retained batch wall time, not decoder latency.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import numpy as np
from scipy.stats import t

HERE=Path(__file__).resolve().parent
SUP=HERE.parent
RESULTS=SUP/'extensions/results'
sys.path.insert(0,str(SUP/'extensions'))
from common import h2,priors
from analyse import wilson


def analyse():
    inputs=set()
    def data(path):
        inputs.add(path)
        with np.load(path,allow_pickle=False) as d:
            return {k:d[k] for k in d.files}
    def read(path):
        inputs.add(path)
        return json.loads(path.read_text())
    diagnostic={};differences=[]
    for path in sorted(RESULTS.glob('diagnostic_p*.npz')):
        d=data(path);truth=float(d['target']);names=list(d['methods'])
        errors=abs(d['estimates']-truth)
        for j,method in enumerate(names):
            e=errors[:,j];covered=e<=1.959963984540054*d['standard_errors'][:,j]
            diagnostic.setdefault(method,[]).append(dict(
                mean=float(e.mean()),variance_of_mean=float(e.var(ddof=1)/len(e)),
                coverage=float(covered.mean()),coverage_ci=wilson(int(covered.sum()),len(e))))
        differences.append(errors[:,names.index('gate_two')]-errors[:,names.index('calibrated_gate')])
    summary={}
    for method,cells in diagnostic.items():
        assert len(cells)==48
        mean=float(np.mean([v['mean'] for v in cells]))
        se=float(np.sqrt(sum(v['variance_of_mean'] for v in cells))/len(cells))
        summary[method]=dict(MAE_pp=100*mean,MC_CI_pp=[100*(mean-1.959964*se),100*(mean+1.959964*se)],
            average_coverage=float(np.mean([v['coverage'] for v in cells])),
            coverage_lower_bound_at_least_90=sum(v['coverage_ci'][0]>=.9 for v in cells),
            cells_whose_coverage_interval_contains_90=sum(v['coverage_ci'][0]<=.9<=v['coverage_ci'][1] for v in cells))
    mean=float(np.mean(differences));se=float(np.sqrt(sum(v.var(ddof=1)/len(v) for v in differences))/48)
    difference=dict(two_minus_calibrated_MAE_pp=100*mean,MC_CI_pp=[100*(mean-1.959964*se),100*(mean+1.959964*se)],
                    scope='Paired replicate-level differences; independent fixed cells, not deployment or design uncertainty.')

    transfer=[]
    conditions=[(0,50),(.25,200),(.75,1000)]
    for family in ['gallager','irregular']:
        for n in [1000,4000,16000]:
            for ci,(a,s) in enumerate(conditions):
                rows={k:[] for k in ['fixed_pooled','fixed_selected','pipeline_pooled','pipeline_selected']}
                for seed in [211,223,227,229,233]:
                    tag=f'transfer_{family}_n{n}_k{seed}_c{ci}'
                    fixed=data(RESULTS/f'{tag}_evaluation.npz')
                    times=read(RESULTS/f'{tag}_timing.json')
                    for mode in ['fixed','pipeline']:
                        for method in ['pooled','selected']:
                            if mode=='fixed':
                                j=list(fixed['methods']).index(method)
                                m=np.full(500,int(fixed['m']));correct=fixed['correct'][j]
                                its=fixed['iterations'][j];seconds=times['fixed'][j]
                            else:
                                d=data(RESULTS/f'{tag}_pipeline_{method}.npz')
                                m=d['m'];correct=d['correct'];its=d['iterations'];seconds=times[method]
                            assert len(correct)==500
                            row=dict(seed=seed,failures=int((~correct).sum()),FER=float((~correct).mean()),
                                yield_mean=float(np.mean(correct*(n-m-64)/(n+s))),
                                row_fraction=float(np.mean(m)/n),f_EC=float(np.mean(m+64)/(n*h2(.06))),
                                iterations=float(np.mean(its)),batch_ms_per_frame=float(1000*seconds/len(correct)))
                            rows[f'{mode}_{method}'].append(row)
                for method,values in rows.items():
                    result=dict(family=family,n=n,asymmetry=a,sample=s,method=method,frames=2500,
                        failures=sum(v['failures'] for v in values))
                    for key in ['FER','yield_mean','row_fraction','f_EC','iterations','batch_ms_per_frame']:
                        x=np.array([v[key] for v in values]);avg=float(x.mean());half=float(t.ppf(.975,4)*x.std(ddof=1)/np.sqrt(5))
                        result[key]=avg;result[key+'_matrix_CI']=[avg-half,avg+half]
                    result['FER_Wilson_conditional']=wilson(result['failures'],2500)
                    transfer.append(result)

    rule=read(RESULTS/'frozen_rule.json');clipping={}
    for cohort,paths,methods in [
        ('confirmation',sorted(RESULTS.glob('confirm_*_frames.npz')),['pooled','separate','shrink','selected']),
        ('physical',sorted(RESULTS.glob('physical_gallager_*_frames.npz')),['pooled','separate','shrink','selected','transition'])]:
        clipping[cohort]={method:dict(frames=0,bits=0,upper_frames=0,upper_bits=0,lower_frames=0,lower_bits=0,maximum_raw_prior=0.) for method in methods}
        for path in paths:
            d=data(path)
            for method in methods:
                pp=priors(d,method,rule);stats=clipping[cohort][method]
                upper=pp>.499;lower=pp<1e-6
                stats['frames']+=len(pp);stats['bits']+=pp.size
                stats['upper_frames']+=int(upper.any(axis=1).sum());stats['upper_bits']+=int(upper.sum())
                stats['lower_frames']+=int(lower.any(axis=1).sum());stats['lower_bits']+=int(lower.sum())
                stats['maximum_raw_prior']=max(stats['maximum_raw_prior'],float(pp.max()))
    inputs.update(SUP/'extensions'/f for f in ['common.py','analyse.py'])
    inputs.add(SUP/'study/reconciliation_v2.py')
    return dict(status='passed',new_stochastic_experiments=False,
        diagnostic_grid=summary,diagnostic_paired_difference=difference,
        coverage_450_of_500_wilson=wilson(450,500),absolute_transfer=transfer,
        clipping=clipping,input_sha256={str(p.relative_to(SUP)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(inputs)})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true')
    args=parser.parse_args();result=analyse();path=HERE/'review-analysis.json'
    text=json.dumps(result,indent=2)+'\n'
    if args.check:assert path.read_text()==text
    else:
        path.write_text(text)
        rows=result['absolute_transfer'];keys=[k for k in rows[0] if not k.endswith('_CI') and 'Wilson' not in k]
        with (HERE/'absolute-transfer.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:v for k,v in result.items() if k not in ['input_sha256','absolute_transfer']},indent=2))


if __name__=='__main__':main()
