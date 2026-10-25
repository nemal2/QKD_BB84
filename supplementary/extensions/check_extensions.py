"""Audit sparse records, observable policy inputs, diagnostics and regeneration."""
import argparse
import hashlib
import json
import time
import numpy as np
from scipy.optimize import least_squares
from common import OUT,P,frame_data,priors,selected_lambda,code,evaluate
from pipelines import features,predict,pipeline,fit_policy
from diagnostics import simulate,gate_fit,response,observations
from provenance import atomic_json
from inventory import check_inventory, record_fingerprint
from audit_fixed_reference import fixed_reference_audit


def source_guards():
    checked=0
    for group in ['asymmetry','pipelines','diagnostics']:
        guard=json.loads((OUT/f'{group}_source_guard.json').read_text())
        for name,digest in guard.items():
            assert hashlib.sha256((OUT.parent/name).read_bytes()).hexdigest()==digest,(group,name)
            checked+=1
    return dict(status='passed',guarded_inputs=checked)


def code_audit():
    audit=[]
    for path in sorted((OUT/'codes').glob('*.npz')):
        z=np.load(path,allow_pickle=False);ptr=z['ptr'];cols=z['cols'];n=int(z['n']);m=int(z['m'])
        assert ptr.shape==(m+1,) and ptr[0]==0 and ptr[-1]==len(cols)
        assert np.all(np.diff(ptr)>=2) and np.all((cols>=0)&(cols<n))
        degree=np.bincount(cols,minlength=n)
        assert np.array_equal(degree,z['column_degrees'])
        for lo,hi in zip(ptr[:-1],ptr[1:]):assert np.all(np.diff(cols[lo:hi])>0)
        family=str(z['family']);seed=int(z['matrix_seed']);beta=float(z['beta'])
        # Independent construction checks do not assume algebraic full row rank.
        if family=='gallager':assert np.all(degree==3) and m%3==0
        audit.append(dict(file=path.name,n=n,m=m,edges=len(cols),minimum_column=int(degree.min()),
                          maximum_column=int(degree.max()),minimum_row=int(np.diff(ptr).min()),
                          leakage='All m transmitted rows charged, even if dependent.'))
    return audit


def observable_inputs():
    rule=json.loads((OUT/'frozen_rule.json').read_text());d=frame_data(.06,.5,200,1000,20,514818)
    other={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in d.items()}
    other['q']=.25;other['a']=0.;other['transitions']=np.full((2,2),.33)
    other['errors'][:,200:]^=1;other['bits'][:,200:]^=1 # Keep Bob's local word identical.
    assert np.array_equal(features(d,.5),features(other,.5))
    assert np.array_equal(selected_lambda(d,rule),selected_lambda(other,rule))
    for method in ['pooled','separate','shrink','selected','transition']:
        assert np.array_equal(priors(d,method,rule),priors(other,method,rule))
    return dict(status='passed',changed='True q,a,transition law and Alice/error block; Bob word preserved.',
                methods=['pooled','separate','shrink','selected','transition'])


def policy_audit():
    frozen=json.loads((OUT/'frozen_pipeline_policies.json').read_text())
    audit=[]
    # Rebuild in the original declared ordering; this also rules out a policy
    # file having been silently edited after its independent development fit.
    for family in ['gallager','irregular']:
        xx=[];yy=[[],[]]
        for n in P['codes']['lengths']:
            for k in P['codes']['development_seeds']:
                for qi in range(3):
                    for ai in range(2):
                        for si in range(3):
                            stem=f'develop_{family}_n{n}_k{k}_q{qi}_a{ai}_s{si}'
                            d=dict(np.load(OUT/f'{stem}_frames.npz',allow_pickle=False))
                            paths=sorted(OUT.glob(f'{stem}_m*_evaluation.npz'),
                                         key=lambda p:int(p.stem.split('_m')[-1].split('_')[0]))
                            for path in paths:
                                r=np.load(path,allow_pickle=False);xx.append(features(d,int(r['m'])/n))
                                for j in range(2):yy[j].append(~r['correct'][j])
        x=np.concatenate(xx)
        for j,method in enumerate(['pooled','selected']):
            rebuilt=fit_policy(x,np.concatenate(yy[j]))
            for key in ['mean','scale','weights','development_rows','failures']:
                assert np.array_equal(rebuilt[key],frozen['policies'][family][method][key]),(family,method,key)
            audit.append(dict(family=family,method=method,rows=len(x),status='exact fit reproduced'))
    return audit


def audit_result(path,d,r,m,n,code_name):
    F=len(d['pool']);s=int(d['s']);true=d['errors'][:,s:]
    est=np.unpackbits(r['estimates_packed'],axis=-1,count=n,bitorder='little')
    if est.ndim==2:est=est[None]
    correct=np.all(est==true[None],axis=2)
    flags=r['correct'][None] if r['correct'].ndim==1 else r['correct']
    assert np.array_equal(correct,flags),path
    accepted=r['accepted'][None] if r['accepted'].ndim==1 else r['accepted']
    if np.ndim(m)==0:groups=[(int(m),np.ones(F,bool),code_name)]
    else:
        groups=[]
        family=path.name.split('_')[1]
        matrix_seed=int(next(x[1:] for x in path.name.split('_') if x.startswith('k') and x[1:].isdigit()))
        for rows in np.unique(m):
            name=f'{family}_n{n}_m{int(rows)}_seed{matrix_seed}.npz'
            groups.append((int(rows),m==rows,name))
    allaccepted=np.zeros_like(correct)
    for rows,mask,name in groups:
        z=np.load(OUT/'codes'/name,allow_pickle=False);ptr=z['ptr'];col=z['cols']
        target=np.bitwise_xor.reduceat(true[mask][:,col],ptr[:-1],axis=1)
        match=np.all(np.bitwise_xor.reduceat(est[:,mask][:,:,col],ptr[:-1],axis=2)==target[None],axis=2)
        for method,part in np.argwhere(match & ~correct[:,mask]):
            f=np.flatnonzero(mask)[part];delta=est[method,f]^true[f]
            bits=np.zeros(64,np.uint8)
            for j in np.flatnonzero(delta):bits^=d['hash_seeds'][f,n-1-j:n-1-j+64]
            match[method,part]=not bits.any()
        allaccepted[:,mask]=match
    assert np.array_equal(allaccepted,accepted),path
    return int(correct.size),int(np.sum(accepted & ~correct))


def records(rerun=False):
    rule=json.loads((OUT/'frozen_rule.json').read_text())
    policyfile=OUT/'frozen_pipeline_policies.json'
    policies=json.loads(policyfile.read_text()) if policyfile.exists() else None
    frame_sets={};outputs=undetected=rerun_arrays=0;audited=[]
    checkpoint=OUT/'holdout_rerun_checks.json'
    completed=json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
    for pattern in ['develop_*_frames.npz','transfer_*_frames.npz','fixedtotal_*_frames.npz','physical_*_frames.npz']:
        for path in sorted(OUT.glob(pattern)):
            d=dict(np.load(path,allow_pickle=False));n=int(d['n']);s=int(d['s']);F=len(d['pool'])
            fresh=frame_data(float(d['q']),float(d['a']),s,n,F,int(d['seed']),d['transitions'])
            assert all(np.array_equal(d[k],fresh[k]) for k in fresh),path
            fingerprint=hashlib.sha256(d['bits'].tobytes()+d['bases'].tobytes()+d['errors'].tobytes()).hexdigest()
            frame_sets[fingerprint]=F
            stem=path.name.removesuffix('_frames.npz')
            paths=sorted(OUT.glob(f'{stem}*evaluation.npz'))+sorted(OUT.glob(f'{stem}_pipeline_*.npz'))
            for epath in paths:
                r=dict(np.load(epath,allow_pickle=False));ispipe='_pipeline_' in epath.name
                name=str(r['code']) if not ispipe else None
                m=r['m'];total,missed=audit_result(epath,d,r,m,n,name)
                outputs+=total;undetected+=missed
                # Full rerun of all holdout outputs; fixed prechosen development
                # records at each length/family additionally check generator/decoder reuse.
                do_rerun=rerun and (not path.name.startswith('develop_') or '_k101_q1_a0_s1_' in path.name)
                digest=hashlib.sha256(epath.read_bytes()).hexdigest()
                if do_rerun and completed.get(epath.name)==digest:
                    rerun_arrays+=5;do_rerun=False
                if do_rerun:
                    family=path.name.split('_')[1]
                    k=int(next(x[1:] for x in path.name.split('_') if x.startswith('k') and x[1:].isdigit()))
                    if ispipe:
                        method=epath.stem.split('_pipeline_')[1]
                        rr,_=pipeline(fresh,family,k,method,policies['policies'],rule)
                    else:
                        z=np.load(OUT/'codes'/name,allow_pickle=False)
                        c=(z['ptr'],z['cols'],int(z['m']),n,name)
                        rr,_=evaluate(c,fresh,r['methods'].tolist(),rule)
                        rr['estimates_packed']=np.packbits(rr.pop('estimates'),axis=-1,bitorder='little')
                    for key in ['correct','accepted','iterations','m','estimates_packed']:
                        assert np.array_equal(rr[key],r[key]),(epath,key)
                        rerun_arrays+=1
                    completed[epath.name]=digest
                    atomic_json(checkpoint,completed)
                audited.append(epath.name)
    return dict(unique_input_frames=sum(frame_sets.values()),unique_frame_batches=len(frame_sets),
                decoder_outputs=outputs,undetected_errors=undetected,files=len(audited),
                exactly_reproduced_arrays=rerun_arrays)


def diagnostic_audit(rerun=False):
    arrays=records=fit_checks=0
    for pattern in ['diagnostic_p*.npz','devicecase_*.npz','diagnostic_stress_*.npz']:
        for path in sorted(OUT.glob(pattern)):
            d=dict(np.load(path,allow_pickle=False));budget=int(d['budget'])
            assert d['signals_four'].sum()==budget and d['signals_two'].sum()==budget
            assert d['signals_calibration']+d['signals_attacked']==budget
            for name in ['counts4','counts2','calibration','signal']:
                c=d[name];assert np.all(c[...,0]>=0) and np.all(c[...,0]<=c[...,1])
            assert np.all(np.isfinite(d['estimates'])) and np.all(np.isfinite(d['standard_errors']))
            if rerun:
                fresh=simulate(float(d['p']),float(d['r']),budget,int(d['seed']),float(d['pE']),str(d['case']),str(d['stress']))
                for key in d:assert np.array_equal(fresh[key],d[key]),(path,key)
                arrays+=len(d)
            if (path.name in ['diagnostic_p0_r0_b0.npz','diagnostic_p3_r3_b2.npz','devicecase_1_p2_r2.npz','devicecase_2_p2_r2.npz']):
                for key,fkey in [('counts4','scales_four'),('counts2','scales_two')]:
                    f=d[fkey];got=gate_fit(d[key],f,float(d['pE']),str(d['case']))
                    for i in range(10):
                        y,v=observations(d[key][i]);w=1/np.sqrt(v)
                        fun=lambda z:(response(f,z[0],z[1],float(d['pE']),str(d['case']))-y)*w
                        # Independently initialized SciPy solver checks the batch fit optimum.
                        ref=least_squares(fun,[.025,.1],bounds=([1e-8,-.5],[min(.3,.999/max(f)),1.5]),
                                          ftol=1e-11,xtol=1e-11,gtol=1e-11)
                        assert np.linalg.norm(fun([got[2][i],got[3][i]]))<=np.linalg.norm(ref.fun)+1e-5,(path,key,i,got[2][i],got[3][i],ref.x)
                        fit_checks+=1
            records+=len(d['estimates'])
    return dict(acquisition_replicates=records,exactly_reproduced_arrays=arrays,independent_Scipy_fits=fit_checks)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--rerun',action='store_true')
    args=parser.parse_args();inventory=check_inventory()
    report=dict(status='passed',inventory=inventory,sources=source_guards(),codes=code_audit(),policies=policy_audit(),
        fixed_reference=fixed_reference_audit(OUT,P),observable_inputs=observable_inputs(),
        records=records(args.rerun),diagnostics=diagnostic_audit(args.rerun),record_fingerprint=record_fingerprint())
    atomic_json(OUT/('extension_validation_rerun.json' if args.rerun else 'extension_validation.json'),report)
    print({k:v for k,v in report.items() if k!='codes'},flush=True)


if __name__=='__main__':main()
