"""Frozen equal-budget diagnostic comparisons and independent response validation."""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import sys
import numpy as np
from models import (basis_response,basis_noise,unequal_response,unequal_jacobian,
                    equal_jacobian,fit_basis,fit_aggregate,fit_unequal,observations,calibrated)

HERE=Path(__file__).resolve().parent
P=json.loads((HERE/'protocol.json').read_text())
OUT=HERE/'results'


def seed(*coordinates):
    return int(np.random.SeedSequence([P['seed_base'],*coordinates]).generate_state(1)[0])


def counts(rng,signals,q):
    n=rng.binomial(int(signals),.5,P['replicates'])
    return np.column_stack([rng.binomial(n,q),n])


def basis_counts(rng,signals,q):
    n=rng.multinomial(int(signals),[.25,.25,.5],size=P['replicates'])[:,:2]
    return np.stack([rng.binomial(n,np.asarray(q)),n],axis=-1)


def allocate(signals,weights):
    raw=signals*np.asarray(weights);n=np.floor(raw).astype(int)
    for ix in np.argsort(-(raw-n))[:int(signals-n.sum())]:n[ix]+=1
    assert n.sum()==signals and (n>0).all()
    return n


def configurations():
    """Finite design optimization reads no Monte Carlo outcomes."""
    OUT.mkdir(exist_ok=True)
    path=OUT/'design.json'
    digest=hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest()
    if path.exists():
        result=json.loads(path.read_text());assert result['protocol_sha256']==digest
        return result
    grid=list(itertools.product(P['nominal']['p'],P['nominal']['r']))
    units=P['design']['allocation_units'];methods={};candidates=[]
    for cap in P['scaling_caps']:
        for settings in [2,4]:
            allocations=np.array([v for v in itertools.product(range(1,units),repeat=settings-1)
                                  if sum(v)<units])
            allocations=np.column_stack([allocations,units-allocations.sum(axis=1)])/units
            interiors=np.linspace(1,cap,7)[1:-1]
            designs=[[1.,cap]] if settings==2 else [[1.,*middle,cap] for middle in itertools.combinations(interiors,2)]
            best=None
            for scales in designs:
                worst=np.zeros(len(allocations))
                for p,r in grid:
                    q=unequal_response(np.array(scales),p,p,r)
                    j=equal_jacobian(np.array([p]),np.array([r]),scales)[0]
                    w=4000*allocations/(q*(1-q))[None,:]
                    A=w@(j[:,0]**2);B=w@(j[:,0]*j[:,1]);C=w@(j[:,1]**2)
                    variance=A/(A*C-B*B)/16
                    worst=np.maximum(worst,variance)
                i=int(np.argmin(worst))
                item=dict(scales=list(map(float,scales)),weights=allocations[i].tolist(),worst_variance=float(worst[i]))
                candidates.append(dict(cap=cap,settings=settings,**item))
                if best is None or item['worst_variance']<best['worst_variance']-1e-15:best=item
            name=f'aggregate_{"two" if settings==2 else "four"}_cap{cap:g}'
            methods[name]=best
    result=dict(protocol_sha256=digest,methods=methods,best_allocation_for_each_setting_set=candidates,
                selection='Nominal-grid minimax residual Fisher variance; fixed finite candidate search, before acquisitions.')
    path.write_text(json.dumps(result,indent=2)+'\n')
    return result


def save(path,**data):
    temporary=path.with_suffix('.tmp.npz');np.savez_compressed(temporary,**data);temporary.replace(path)


def grid():
    designs=configurations()['methods'];methods=P['methods']
    for stage,key in enumerate(['nominal','off_grid']):
        cfg=P[key]
        for pi,p in enumerate(cfg['p']):
            for ri,r in enumerate(cfg['r']):
                for bi,budget in enumerate(cfg['budgets']):
                    tag=f'{key}_p{pi}_r{ri}_b{bi}'
                    for mi,method in enumerate(methods):
                        path=OUT/f'{tag}_{method}.npz'
                        if path.exists():continue
                        rngseed=seed(stage,pi,ri,bi,mi);rng=np.random.default_rng(rngseed)
                        common=dict(stage=key,p=p,r=r,budget=budget,target=r/4,method=method,seed=rngseed)
                        if method=='basis_single':
                            c=basis_counts(rng,budget,basis_response(p,r));fit=fit_basis(c)
                            save(path,counts=c,**fit,**common)
                        elif method=='calibrated':
                            nc=round(budget*P['design']['calibration_share'])
                            cal=counts(rng,nc,unequal_response(1,p,p,0.))
                            attacked=counts(rng,budget-nc,unequal_response(1,p,p,r))
                            values,se=calibrated(cal,attacked)
                            save(path,calibration=cal,signal=attacked,estimates=values[:,1],standard_errors=se[:,1],**common)
                        else:
                            d=designs[method];scales=np.array(d['scales']);signals=allocate(budget,d['weights'])
                            c=np.stack([counts(rng,n,unequal_response(f,p,p,r)) for n,f in zip(signals,scales)],axis=1)
                            fit=fit_aggregate(c,scales)
                            save(path,counts=c,scales=scales,signals=signals,**fit,**common)
                    print(tag,flush=True)


def basis_general(a,b,r):
    noise=np.array([a/4,(2-(1-b)*(1-a)*(2-a))/4])
    return r/4+(1-r/2)*noise


def holdout():
    designs=configurations()['methods'];cfg=P['holdout'];T=cfg['training_budget'];Tv=cfg['validation_budget'];fv=cfg['validation_scale']
    for si,scenario in enumerate(cfg['scenarios']):
        a,b,r,epsilon=[scenario[k] for k in ['pA','pB','r','curvature']]
        def effective(f):return f+epsilon*f*(f-1)
        validation=counts(np.random.default_rng(seed(2,si,99)),Tv,unequal_response(effective(fv),a,b,r))
        for mi,method in enumerate(['basis_single','aggregate_two_cap4','aggregate_four_cap4']):
            path=OUT/f'holdout_{scenario["name"]}_{method}.npz'
            rngseed=seed(2,si,mi);rng=np.random.default_rng(rngseed)
            if path.exists():
                with np.load(path,allow_pickle=False) as z:c=z['counts'];scales=z['scales'] if 'scales' in z else None
            else:
                if method=='basis_single':
                    c=basis_counts(rng,T,basis_general(a,b,r));fit=fit_basis(c);scales=np.array([1.])
                else:
                    d=designs[method];scales=np.array(d['scales']);signals=allocate(T,d['weights'])
                    c=np.stack([counts(rng,n,unequal_response(effective(f),a,b,r)) for n,f in zip(signals,scales)],axis=1)
                    fit=fit_aggregate(c,scales)
                save(path,counts=c,scales=scales,validation=validation,validation_scale=fv,
                     training_budget=T,validation_budget=Tv,target=r/4,pA=a,pB=b,r=r,curvature=epsilon,
                     true_validation_qber=unequal_response(effective(fv),a,b,r),method=method,scenario=scenario['name'],seed=rngseed,**fit)
            if method=='aggregate_four_cap4':
                independent=OUT/f'holdout_{scenario["name"]}_independent_devices.npz'
                if not independent.exists():
                    fit=fit_unequal(c,scales)
                    save(independent,counts=c,scales=scales,validation=validation,validation_scale=fv,
                         training_budget=T,validation_budget=Tv,target=r/4,pA=a,pB=b,r=r,curvature=epsilon,
                         true_validation_qber=unequal_response(effective(fv),a,b,r),method='independent_devices',scenario=scenario['name'],seed=rngseed,**fit)
        print('holdout',scenario['name'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['design','grid','holdout','all'],default='all');args=parser.parse_args()
    configurations()
    guard={str(path.relative_to(HERE.parent)):hashlib.sha256(path.read_bytes()).hexdigest()
           for path in [HERE/'protocol.json',HERE/'models.py',HERE/'experiment.py',HERE.parent/'extensions/diagnostics.py',HERE.parent/'extensions/physics.py']}
    provenance=OUT/'source_guard.json'
    if provenance.exists():assert json.loads(provenance.read_text())==guard,'Source changed; inspect before reusing acquisitions.'
    else:provenance.write_text(json.dumps(guard,indent=2)+'\n')
    if args.stage in ['grid','all']:grid()
    if args.stage in ['holdout','all']:holdout()


if __name__=='__main__':main()
