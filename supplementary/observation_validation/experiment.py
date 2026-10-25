"""Focused cross-circuit acquisitions and retained-cohort likelihood analysis."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
from scipy.optimize import brentq
from scipy.special import xlogy
from likelihood import fit_batch,information,probabilities,PMAX,CUTOFF

HERE=Path(__file__).resolve().parent
P=json.loads((HERE/'protocol.json').read_text())
OUT=HERE/'results'


# Compute finite-count subset intervals, including empty subsets.
def wilson(e,n):
    n=np.asarray(n);p=np.divide(e,n,out=np.zeros_like(n,dtype=float),where=n>0)
    nn=np.maximum(n,1);z=1.959963984540054
    centre=(p+z*z/(2*nn))/(1+z*z/nn)
    half=z*np.sqrt(p*(1-p)/nn+z*z/(4*nn*nn))/(1+z*z/nn)
    lo=np.maximum(0,centre-half);hi=np.minimum(1,centre+half)
    return np.where(n>0,lo,0.),np.where(n>0,hi,1.)


def summarize_fit(counts,paths,groups,scales,equal=False):
    fit=fit_batch(counts,paths,groups,scales,equal)
    diagnostics=[information(row[:3],paths,groups,scales,c,equal) for row,c in zip(fit,counts)]
    return dict(parameters=fit[:,:3],nll=fit[:,3],estimates=fit[:,2]/4,
        intervals=fit[:,4:],condition=np.array([x[0] for x in diagnostics]),
        information_rank=np.array([x[1] for x in diagnostics]),
        parameter_standard_errors=np.array([x[2] for x in diagnostics]),
        boundary=np.any((fit[:,:2]<1e-6)|(fit[:,:2]>PMAX-1e-6),axis=1),
        target_boundary=(fit[:,2]<1e-6)|(fit[:,2]>1-1e-6))


# Report the identified residual set for balanced paths, without choosing an arbitrary point estimate.
def nonidentified(counts):
    e=counts[:,:,0].sum(axis=1);n=counts[:,:,1].sum(axis=1)
    intervals=[];qmax=(1-(1-PMAX)**3)/2
    for ee,nn in zip(e,n):
        q=ee/nn
        def dev(x):
            return 2*(xlogy(ee,q/max(x,1e-300))+xlogy(nn-ee,(1-q)/max(1-x,1e-300)))-CUTOFF
        lo=brentq(dev,1e-15,q) if ee else 0.
        hi=brentq(dev,q,1-1e-15) if ee<nn else 1.
        intervals.append([max(0.,(lo-qmax)/(1-2*qmax)),min(.25,hi)])
    return dict(intervals=np.array(intervals),identified=False)


def save(path,**data):
    tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**data);tmp.replace(path)


# Generate independently budgeted acquisitions, then fit each disclosed-data reduction.
def run_cell(index,circuit,scenario,readout=0.):
    paths=np.asarray(P['circuits'][circuit],dtype=np.int64)
    a,b,r=[scenario[k] for k in ['pA','pB','r']]
    tag=f'{circuit.replace("+","plus")}_{scenario["name"]}'+('_readout' if readout else '')
    fullpath=OUT/f'{tag}_single.npz';scaledpath=OUT/f'{tag}_scaled.npz'
    for mode,path in [('single',fullpath),('scaled',scaledpath)]:
        if path.exists():continue
        rng=np.random.default_rng(np.random.SeedSequence([P['seed_base'],index,0 if mode=='single' else 1]))
        if mode=='single':
            groups=np.eye(4);scales=np.ones(4)
            n=rng.multinomial(P['budget'],[.125]*4+[.5],size=P['replicates'])[:,:4]
        else:
            scales=np.array(P['acquisitions']['scaled']['scales']);groups=np.full((4,4),.25)
            signals=(P['budget']*np.array(P['acquisitions']['scaled']['weights'])).astype(int)
            assert signals.sum()==P['budget'];n=rng.binomial(signals,.5,size=(P['replicates'],4))
        q=readout+(1-2*readout)*probabilities([a,b,r],paths,groups,scales)
        counts=np.stack([rng.binomial(n,q),n],axis=-1)
        save(path,counts=counts,groups=groups,scales=scales,paths=paths,circuit=circuit,
             scenario=scenario['name'],pA=a,pB=b,r=r,readout=readout,budget=P['budget'],
             target=r/4,effective_target=readout+(1-2*readout)*r/4,true_probabilities=q)
    with np.load(fullpath) as z:counts=z['counts']
    common=dict(circuit=circuit,scenario=scenario['name'],target=r/4,
                effective_target=readout+(1-2*readout)*r/4,readout=readout)
    for method in ['full','basis','subset','scaled']:
        if method=='subset' and circuit!='PRIMARY':continue
        path=OUT/f'{tag}_{method}_fit.npz'
        if path.exists():continue
        if method=='subset':
            e,n=counts[:,0,0],counts[:,0,1]
            fit=dict(estimates=np.divide(e,n,out=np.full(len(n),np.nan),where=n>0),
                     intervals=np.column_stack(wilson(e,n)),empty_count=n==0)
        else:
            if method=='full':c=counts;groups=np.eye(4);scales=np.ones(4)
            elif method=='basis':
                c=counts.reshape(len(counts),2,2,2).sum(axis=2)
                groups=np.array([[.5,.5,0,0],[0,0,.5,.5]]);scales=np.ones(2)
            else:
                with np.load(scaledpath) as z:c=z['counts'];groups=z['groups'];scales=z['scales']
            if circuit=='B+' and method!='scaled':fit=nonidentified(c)
            else:fit=summarize_fit(c,paths,groups,scales,method=='basis')
        save(path,method=method,**common,**fit)
        print(tag,method,flush=True)


# Reanalyse the fixed matched-range cohort; reuse counts without generating new signals.
def retained():
    folder=HERE.parent/'diagnostic_validation/results'
    for scenario in ['equal','off_grid','alice_low','bob_low','curvature']:
        path=OUT/f'retained_{scenario}_profile.npz'
        if path.exists():continue
        with np.load(folder/f'holdout_{scenario}_independent_devices.npz') as z:
            counts=z['counts'];scales=z['scales'];target=float(z['target'])
            old_estimates=z['estimates'];old_se=z['standard_errors'];old_pars=z['parameters']
            old_cov=z['covariance']
        paths=np.asarray(P['circuits']['PRIMARY'],dtype=np.int64);groups=np.full((4,4),.25)
        fit=summarize_fit(counts,paths,groups,scales)
        save(path,scenario=scenario,method='retained_profile',counts=counts,scales=scales,
             paths=paths,groups=groups,target=target,effective_target=target,readout=0.,
             old_estimates=old_estimates,old_standard_errors=old_se,old_parameters=old_pars,
             old_parameter_standard_errors=np.sqrt(np.diagonal(old_cov,axis1=1,axis2=2)),**fit)
        print('retained',scenario,flush=True)


# Check the protocol/source guard before generating or resuming a component.
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['new','retained','all'],default='all')
    parser.add_argument('--cell',type=int)
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    guard={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'protocol.json',HERE/'likelihood.py',HERE/'experiment.py']}
    gp=OUT/'source_guard.json'
    if gp.exists():assert json.loads(gp.read_text())==guard,'Source changed; inspect before reusing acquisitions.'
    else:gp.write_text(json.dumps(guard,indent=2)+'\n')
    cells=[(c,s,0.) for c in P['circuits'] for s in P['scenarios']]
    cells += [('PRIMARY',s,P['readout_control']['probability']) for s in P['scenarios']]
    cells += [('PRIMARY',P['boundary_control'],0.)]
    if args.stage in ['new','all']:
        for i,(c,s,r) in enumerate(cells):
            if args.cell is None or i==args.cell:run_cell(i,c,s,r)
    if args.stage in ['retained','all']:retained()


if __name__=='__main__':main()
