"""Equal-budget development policies, independent code transfer and sample costs.

The irregular family uses published variable-degree fractions with the existing
bounded local-cycle-search, degree-balanced check realization. This does not
reproduce the published check distribution or its infinite-length thresholds.
"""
import argparse
import json
import time
import numpy as np
from scipy.optimize import minimize
from common import (OUT,P,LAMBDAS,seed,code,frame_data,evaluate,selected_lambda,h2)
from provenance import atomic_json,atomic_npz,guarded_run

FAMILIES=['gallager','irregular']
RATES=np.array(P['pipeline_development']['rate_pool'])
METHODS=['pooled','separate','shrink','selected','oracle_pool','oracle_basis']


def features(data,beta):
    """Observable sample summaries only: no true channel or decoded-block values."""
    q=data['pool'];z2=np.minimum(100,data['diff']**2/np.maximum(data['variance'],1e-30))
    h=h2(q);rate=np.broadcast_to(beta,q.shape)
    return np.column_stack([h,q,np.log1p(z2),np.full(len(q),1/np.sqrt(int(data['s']))),
                            np.full(len(q),np.log(int(data['n'])/1000)),rate,
                            rate-h,(rate-h)**2,h*np.log1p(z2)])


def fit_policy(x,y):
    mean=x.mean(0);scale=np.maximum(x.std(0),1e-8)
    X=np.column_stack([np.ones(len(x)),(x-mean)/scale]);y=np.asarray(y,float)
    penalty=np.r_[0.,np.ones(x.shape[1])]
    def objective(w):
        z=X@w;p=1/(1+np.exp(-np.clip(z,-40,40)))
        loss=np.sum(np.logaddexp(0,z)-y*z)+.5*np.sum(penalty*w*w)
        gradient=X.T@(p-y)+penalty*w
        return loss,gradient
    result=minimize(objective,np.zeros(X.shape[1]),jac=True,method='L-BFGS-B',
                    options=dict(maxiter=1000,ftol=1e-12,gtol=1e-7))
    if not result.success:raise RuntimeError(result.message)
    return dict(mean=mean.tolist(),scale=scale.tolist(),weights=result.x.tolist(),
                development_rows=len(y),failures=int(y.sum()),
                regularization=1.,optimization=str(result.message))


def predict(policy,x):
    z=np.column_stack([np.ones(len(x)),(x-policy['mean'])/policy['scale']])@policy['weights']
    return 1/(1+np.exp(-np.clip(z,-40,40)))


def candidates(q):
    return np.unique([RATES[np.argmin(abs(RATES-alpha*float(h2(q))))]
                      for alpha in P['pipeline_development']['candidate_margins']])


def store(path,result,methods):
    result=dict(result)
    result['estimates_packed']=np.packbits(result.pop('estimates'),axis=-1,bitorder='little')
    result['methods']=np.array(methods,dtype=str)
    atomic_npz(path,**result)


def train():
    path=OUT/'frozen_pipeline_policies.json'
    if path.exists():return json.loads(path.read_text())
    rule=json.loads((OUT/'frozen_rule.json').read_text());allx={};ally={};calibration={}
    for fi,family in enumerate(FAMILIES):
        allx[family]=[];ally[family]=[[],[]]
        for ni,n in enumerate(P['codes']['lengths']):
            for matrix_seed in P['codes']['development_seeds']:
                for qi,q in enumerate(P['pipeline_development']['qber']):
                    for ai,a in enumerate(P['pipeline_development']['asymmetry']):
                        for si,s in enumerate(P['pipeline_development']['sample']):
                            tag=f'develop_{family}_n{n}_k{matrix_seed}_q{qi}_a{ai}_s{si}'
                            fpath=OUT/f'{tag}_frames.npz'
                            if fpath.exists():data=dict(np.load(fpath,allow_pickle=False))
                            else:
                                # Same source frames across code families, rates and prior methods.
                                data=frame_data(q,a,s,n,100,seed('development',ni,matrix_seed,qi,ai,si))
                                atomic_npz(fpath,**data)
                            for beta in candidates(q):
                                c=code(family,n,float(beta),matrix_seed);m=c[2]
                                epath=OUT/f'{tag}_m{m}_evaluation.npz'
                                if epath.exists():r=dict(np.load(epath,allow_pickle=False))
                                else:
                                    r,t=evaluate(c,data,['pooled','selected'],rule)
                                    store(epath,r,['pooled','selected'])
                                    atomic_json(epath.with_suffix('.timing.json'),t)
                                allx[family].append(features(data,m/n))
                                for j in range(2):ally[family][j].append(~r['correct'][j])
                                if q==.06:
                                    key=f'{family}_{n}_{float(beta)}'
                                    calibration.setdefault(key,[]).extend((r['correct'][0]*(n-m-64)/(n+s)).tolist())
                            print(tag,flush=True)
    policies={}
    for family in FAMILIES:
        x=np.concatenate(allx[family]);policies[family]={}
        for j,method in enumerate(['pooled','selected']):
            policies[family][method]=fit_policy(x,np.concatenate(ally[family][j]))
    fixed={}
    for family in FAMILIES:
        for n in P['codes']['lengths']:
            rates=candidates(.06)
            means={str(float(b)):float(np.mean(calibration[f'{family}_{n}_{float(b)}'])) for b in rates}
            fixed[f'{family}_{n}']=dict(beta=max(rates,key=lambda b:(means[str(float(b))],-b)),
                                        development_yields=means)
    result=dict(policies=policies,fixed_reference=fixed,
                feature_names=['pooled entropy','pooled QBER','log1p sample z2','inverse sqrt sample',
                               'log block length relative to1000','row fraction','row fraction minus entropy',
                               'squared row margin','entropy times log1p z2'],
                tuning='Identical features, ridge penalty 1, optimizer, rows and code candidates for both methods.',
                source='Elkouss et al., arXiv:0901.2140, Table I variable-degree fractions; check distributions differ.',
                deployment='Maximize predicted verified reconciliation yield over the shared row-rate pool.',
                seed_scope='Only development seeds101,103; no holdout outcomes read.')
    atomic_json(path,result)
    print('pipeline policies frozen',flush=True)
    return result


def pipeline(data,family,matrix_seed,method,policies,rule):
    n=int(data['n']);s=int(data['s']);F=len(data['pool'])
    actual=np.array([3*round(b*n/3) if family=='gallager' else round(b*n) for b in RATES])
    scores=np.stack([(1-predict(policies[family][method],features(data,m/n)))*(n-m-64)/(n+s) for m in actual])
    choice=scores.argmax(0);result={k:np.empty(F,dtype=bool if k in ['correct','accepted'] else np.int64)
                                  for k in ['correct','accepted','iterations','m']}
    estimates=np.zeros((F,n),np.uint8);timing=0.
    for j in np.unique(choice):
        mask=choice==j;part={k:(v[mask] if isinstance(v,np.ndarray) and v.ndim and len(v)==F else v)
                           for k,v in data.items()}
        c=code(family,n,float(RATES[j]),matrix_seed)
        r,t=evaluate(c,part,[method],rule);timing+=t[0]
        for k in ['correct','accepted','iterations']:result[k][mask]=r[k][0]
        result['m'][mask]=c[2];estimates[mask]=r['estimates'][0]
    result['estimates_packed']=np.packbits(estimates,axis=-1,bitorder='little')
    return result,timing


def transfer(policies):
    rule=json.loads((OUT/'frozen_rule.json').read_text())
    plan=json.loads((OUT/'frozen_confirmation_plan.json').read_text())
    for ni,n in enumerate(P['codes']['lengths']):
        for matrix_seed in P['codes']['test_seeds']:
            for ci,condition in enumerate(plan):
                a=condition['a'];s=condition['s'];F=condition['frames_per_matrix']
                data=frame_data(.06,a,s,n,F,seed('transfer',ni,matrix_seed,ci))
                for family in FAMILIES:
                    tag=f'transfer_{family}_n{n}_k{matrix_seed}_c{ci}'
                    path=OUT/f'{tag}_evaluation.npz'
                    if path.exists() and all((OUT/f'{tag}_pipeline_{method}.npz').exists()
                                             for method in ['pooled','selected']):continue
                    atomic_npz(OUT/f'{tag}_frames.npz',**data)
                    beta=float(policies['fixed_reference'][f'{family}_{n}']['beta'])
                    r,t=evaluate(code(family,n,beta,matrix_seed),data,METHODS,rule)
                    store(path,r,METHODS)
                    timed=dict(fixed=t)
                    for method in ['pooled','selected']:
                        r,tt=pipeline(data,family,matrix_seed,method,policies['policies'],rule)
                        atomic_npz(OUT/f'{tag}_pipeline_{method}.npz',**r);timed[method]=tt
                    atomic_json(OUT/f'{tag}_timing.json',timed)
                    print(tag,'completed',flush=True)


def fixed_total(policies):
    rule=json.loads((OUT/'frozen_rule.json').read_text());cfg=P['fixed_total'];N=cfg['N']
    for matrix_seed in P['codes']['test_seeds']:
        for si,s in enumerate(cfg['sample']):
            n=N-s;data=frame_data(cfg['qber'],cfg['asymmetry'],s,n,500,seed('fixed_total',matrix_seed,si))
            for family in FAMILIES:
                tag=f'fixedtotal_{family}_k{matrix_seed}_s{s}';path=OUT/f'{tag}_frames.npz'
                if all((OUT/f'{tag}_pipeline_{method}.npz').exists() for method in ['pooled','selected']):continue
                atomic_npz(path,**data)
                for method in ['pooled','selected']:
                    r,t=pipeline(data,family,matrix_seed,method,policies['policies'],rule)
                    atomic_npz(OUT/f'{tag}_pipeline_{method}.npz',**r)
                    atomic_json(OUT/f'{tag}_{method}_timing.json',t)
                print(tag,'completed',flush=True)


def physical(policies):
    from physics import physical_transition,matched_transition
    rule=json.loads((OUT/'frozen_rule.json').read_text());models=['homogeneous','gate_dependent','pauli','thermal','readout']
    methods=['pooled','separate','shrink','selected','transition']
    for matrix_seed in P['codes']['test_seeds']:
        for ci,model in enumerate(models):
            for matched in [False,True]:
                transitions,scale=matched_transition(model) if matched else (physical_transition(model),1.)
                # Same-rate comparator selected on independent Q=6% development frames,
                # using both asymmetries (0 and 0.5) and all three sample sizes.
                data=frame_data(float(transitions.mean()),0,200,1000,500,
                                seed('physical',matrix_seed,ci,int(matched)),transitions)
                for family in FAMILIES:
                    tag=f'physical_{family}_k{matrix_seed}_{model}_{"matched" if matched else "fixed"}'
                    path=OUT/f'{tag}_evaluation.npz'
                    if path.exists():continue
                    atomic_npz(OUT/f'{tag}_frames.npz',**data)
                    beta=float(policies['fixed_reference'][f'{family}_1000']['beta'])
                    r,t=evaluate(code(family,1000,beta,matrix_seed),data,methods,rule)
                    store(path,r,methods)
                    atomic_json(OUT/f'{tag}_timing.json',dict(seconds=t,physical_scale=scale))
                    print(tag,'completed',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['development','transfer','fixed-total','physical','all'],default='all')
    args=parser.parse_args()
    with guarded_run('pipelines',['protocol.json','common.py','pipelines.py','physics.py',
                                 'results/frozen_rule.json','results/frozen_confirmation_plan.json',
                                 '../study/reconciliation_v2.py']):
        policies=train()
        if args.stage in ['transfer','all']:transfer(policies)
        if args.stage in ['fixed-total','all']:fixed_total(policies)
        if args.stage in ['physical','all']:physical(policies)


if __name__=='__main__':main()
