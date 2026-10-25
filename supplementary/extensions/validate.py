"""Independent references, exact record audit and optional complete decoder rerun."""
import argparse
import hashlib
import json
import numpy as np
from common import (OUT,P,LAMBDAS,frame_data,seed,reference_code,evaluate,
                    priors,toeplitz_hash,batch,code,entropy_benchmark,bp_layered)
from physics import Q,exact_density_Q,physical_transition,matched_transition
from provenance import atomic_json
from inventory import check_inventory, record_fingerprint


def dense_reference(H,syndrome,prior,sweeps):
    """NumPy layered sum-product reference, with no accelerated decoder calls."""
    llr=np.log((1-np.clip(prior,1e-6,.499))/np.clip(prior,1e-6,.499))
    post=llr.copy();messages=np.zeros_like(H,dtype=float)
    for iteration in range(1,sweeps+1):
        for row in range(len(H)):
            ix=np.flatnonzero(H[row]);ext=post[ix]-messages[row,ix]
            ts=np.tanh(np.clip(ext/2,-19,19))
            ts=np.where(np.abs(ts)<1e-12,np.where(ts>=0,1e-12,-1e-12),ts)
            values=np.clip(np.prod(ts)/ts,-.999999999,.999999999)
            messages[row,ix]=(1-2*int(syndrome[row]))*2*np.arctanh(values)
            post[ix]=ext+messages[row,ix]
        estimate=(post<0).astype(np.uint8)
        if np.array_equal((H@estimate)%2,syndrome):
            return estimate,True,iteration
    return estimate,False,sweeps


def references():
    rng=np.random.default_rng(361807);bp_cases=hash_cases=0
    for n,m in [(32,17),(61,35),(100,60)]:
        H=np.zeros((m,n),np.uint8)
        for row in H:row[rng.choice(n,6,replace=False)]=1
        rows,cols=np.nonzero(H);ptr=np.r_[0,np.cumsum(H.sum(1))].astype(np.int64)
        for q in [.005,.06,.15]:
            for _ in range(4):
                prior=rng.uniform(q/2,1.5*q,n);error=(rng.random(n)<prior).astype(np.uint8)
                syn=(H@error)%2
                got=bp_layered(ptr,cols,m,n,syn,prior,100)
                expected=dense_reference(H,syn,prior,100)
                assert got[1]==expected[1]
                # Failed trajectories can be chaotic under floating-point rounding.
                if got[1]:
                    assert np.array_equal(got[0],expected[0]) and got[2]==expected[2]
                bp_cases+=1
        seedbits=rng.integers(0,2,n+63,dtype=np.uint8)
        T=np.array([[seedbits[n-1-j+k] for j in range(n)] for k in range(64)],np.uint8)
        for _ in range(10):
            error=rng.integers(0,2,n,dtype=np.uint8)
            bits=(T@error)%2;value=sum(int(x)<<k for k,x in enumerate(bits))
            assert int(toeplitz_hash(error,seedbits))==value;hash_cases+=1
    maximum=0.;density_cases=0
    for case in ['legitimate','all_scaled']:
        for f in [0,.5,1,2,8]:
            for pE in [0,.02,.11]:
                analytic=float(Q(f,.015,.02,pE,.4,case))
                error=abs(analytic-exact_density_Q(f,.015,.02,pE,.4,case))
                assert error<1e-12;maximum=max(maximum,error);density_cases+=1
    channels={}
    for model in ['homogeneous','gate_dependent','pauli','thermal','readout']:
        t=physical_transition(model);mt,scale=matched_transition(model)
        assert np.all((t>=-1e-14)&(t<=1)) and abs(mt.mean()-.06)<1e-11
        if model=='homogeneous':assert np.ptp(t)<1e-12
        if model=='pauli':assert np.allclose(t,[[.02,.02],[.08,.08]],atol=1e-12)
        channels[model]=dict(fixed=t.tolist(),matched=mt.tolist(),scale=scale,
                             entropy=entropy_benchmark(mt))
    # Conditional-entropy implementation against an independent joint distribution.
    for _ in range(20):
        t=rng.uniform(0,.3,(2,2));expected=0.
        for b in [0,1]:
            joint=np.array([[1-t[b,0],t[b,0]],[t[b,1],1-t[b,1]]])/2
            marginal=joint.sum(0)
            expected-=sum(joint[x,y]*np.log2(joint[x,y]/marginal[y])
                          for x in [0,1] for y in [0,1])/2
        assert abs(expected-np.mean(entropy_benchmark(t)['conditional_entropy']))<1e-12
    return dict(independent_BP_cases=bp_cases,independent_Toeplitz_cases=hash_cases,
                separated_device_density_cases=density_cases,maximum_probability_error=maximum,
                physical_channels=channels,independent_entropy_cases=20)


def records(rerun=False):
    selections=json.loads((OUT/'reference_selection.json').read_text())
    rulepath=OUT/'frozen_rule.json';rule=json.loads(rulepath.read_text()) if rulepath.exists() else None
    audit=[];undetected=0;frames=outputs=arrays=0
    for stage in ['pilot','confirm']:
        methods=[0.,.25,.5,.75,1.,'shrink','oracle_pool','oracle_basis'] if stage=='pilot' else ['pooled','separate','shrink','selected','oracle_pool','oracle_basis']
        for path in sorted(OUT.glob(f'{stage}_*_evaluation.npz')):
            d=dict(np.load(path.with_name(path.name.replace('_evaluation','_frames')),allow_pickle=False))
            r=dict(np.load(path,allow_pickle=False));s=int(d['s']);n=int(d['n'])
            names=path.stem.split('_');qi=int(names[1][1:]);ai=int(names[2][1:]);si=int(names[3][1:])
            F=P['pilot']['frames'] if stage=='pilot' else P['confirmation']['all_75_cells']
            expected=frame_data(P['pilot']['qber'][qi],P['pilot']['asymmetry'][ai],P['pilot']['sample'][si],n,F,seed(stage,qi,ai,si))
            assert all(np.array_equal(d[k],expected[k]) for k in expected),path
            assert np.all(d['counts'].sum((1,2,3))==s)
            assert np.array_equal(np.all(r['estimates']==d['errors'][None,:,s:],axis=2),r['correct'])
            pooled=priors(d,'pooled');separate=priors(d,'separate')
            assert np.allclose(pooled,d['pool'][:,None])
            # A coherent contrast exactly preserves the sample-weighted centre.
            ss=d['counts'].sum((2,3))
            zpri=d['pool'][:,None]+d['rz']-d['raw'][:,None]
            assert np.allclose((zpri*ss).sum(1)/s,d['pool'])
            c=reference_code(float(d['q']),11 if stage=='pilot' else 29,int(r['m']))
            ptr,col,m,_,_=c
            target=np.bitwise_xor.reduceat(d['errors'][:,s:][:,col],ptr[:-1],axis=1)
            syndrome=np.bitwise_xor.reduceat(r['estimates'][:,:,col],ptr[:-1],axis=2)
            matched=np.all(syndrome==target[None],axis=2)
            accepted=matched.copy()
            for method,f in np.argwhere(matched & ~r['correct']):
                delta=r['estimates'][method,f]^d['errors'][f,s:]
                nonzero=np.flatnonzero(delta)
                bits=np.zeros(64,np.uint8)
                for j in nonzero:bits^=d['hash_seeds'][f,n-1-j:n-1-j+64]
                accepted[method,f]=not bits.any()
            assert np.array_equal(accepted,r['accepted'])
            undetected+=int(np.sum(accepted & ~r['correct']))
            if rerun:
                fresh,_=evaluate(c,expected,methods,rule)
                for k in r:assert np.array_equal(fresh[k],r[k]),(path,k)
                arrays+=len(r)
            if stage=='pilot' and rule is not None:
                assert hashlib.sha256(path.read_bytes()).hexdigest()==rule['pilot_sha256'][path.name]
            frames+=F;outputs+=F*len(methods)
            audit.append(dict(file=path.name,frames=F,methods=len(methods),rerun=rerun))
    return dict(cells=len(audit),independent_input_frames=frames,decoder_outputs=outputs,
                undetected_errors=undetected,exactly_reproduced_arrays=arrays,audit=audit)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rerun',action='store_true')
    args=parser.parse_args()
    inventory=check_inventory()
    report=dict(status='passed',inventory=inventory,references=references(),records=records(args.rerun),
                record_fingerprint=record_fingerprint())
    atomic_json(OUT/('validation_rerun.json' if args.rerun else 'validation.json'),report)
    print({k:v for k,v in report['records'].items() if k!='audit'},flush=True)


if __name__=='__main__':main()
