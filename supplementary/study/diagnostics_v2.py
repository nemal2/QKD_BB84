"""Matched-budget diagnostics, design selection, drift and native-law comparator.

The Fiorini interception-density formula is transcribed from
francesco-fiorini/BB84-QuantumSimulator, commit 9ce07ab3425bf3e98a7e352a3e97b6bafa493edf,
bb84/bb84_algorithm.py. Its MIT notice accompanies this implementation.
We report its disturbance estimate (interception density / 4), unclipped.
"""
import json
import itertools
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from reconciliation_v2 import P, OUT, qk,guard


def q0(p):
    return np.mean(qk(np.asarray(p)[...,None],np.arange(4)),axis=-1)


def qi(p):
    p=np.asarray(p);same=0.
    for counts in [[0,1],[2,3]]:
        for bit in [0,1]:
            f=qk(p,counts[bit]);a=qk(p,counts[bit]);b=qk(p,counts[1-bit])
            same+=(1-f)*a+f*(1-b)
    return .25+same/8


def qt(p,r):return (1-r)*q0(p)+r*qi(p)


def fisher(scales,n,p,r):
    scales=np.asarray(scales);q=qt(p*scales,r);eps=1e-6
    dp=(qt((p+eps)*scales,r)-qt((p-eps)*scales,r))/(2*eps)
    dr=qi(p*scales)-q0(p*scales);J=np.column_stack([dp,dr])
    F=J.T@((np.asarray(n)/2/(q*(1-q)))[:,None]*J)
    cov=np.linalg.inv(F)
    return float(cov[1,1]/16),float(cov[0,1]/np.sqrt(cov[0,0]*cov[1,1]))


def optimize_design():
    path=OUT/'diagnostic_design.json'
    if path.exists():return json.loads(path.read_text())
    grid=list(itertools.product([.02,.03,.04],[0,.1,.2,.3,.4]));designs=[]
    for high,share in itertools.product([2.5,4,6,8],[.5,.6,.7,.8,.9]):
        ns=[round(8000*share),round(8000*(1-share))]
        v=[fisher([1,high],ns,p,r)[0] for p,r in grid]
        designs.append(dict(scales=[1,high],signals=ns,max_variance=max(v)))
    best=min(designs,key=lambda x:x['max_variance'])
    calibration=[]
    for share in [.2,.3,.4,.5,.6,.7,.8]:
        nc=round(8000*share);ns=8000-nc;vs=[]
        for p,r in grid:
            e=q0(p);q=qt(p,r);eps=1e-6
            ep=(q0(p+eps)-q0(p-eps))/(2*eps)
            d=qi(p)-e;dp=(qt(p+eps,r)-qt(p-eps,r))/(2*eps)
            vs.append((q*(1-q)/(ns/2)+(dp/ep)**2*e*(1-e)/(nc/2))/(16*d*d))
        calibration.append(dict(calibration=nc,signal=ns,max_variance=max(vs)))
    result=dict(two_point=best,calibration=min(calibration,key=lambda x:x['max_variance']),
                two_point_candidates=designs,calibration_candidates=calibration,
                design_grid=grid,method='minimize worst Fisher/delta variance before Monte Carlo',
                four_point_fisher={str(r):fisher([1,1.5,2,2.5],[2000]*4,.03,r) for r in [0,.1,.2,.3,.4]},
                two_point_fisher={str(r):fisher(best['scales'],best['signals'],.03,r) for r in [0,.1,.2,.3,.4]})
    path.write_text(json.dumps(result,indent=2)+'\n');return result


def draw(rng,signals,q,size):
    n=rng.binomial(signals,.5,size=size)
    return np.column_stack([rng.binomial(n,q),n])


def fit(counts,scales):
    e,n=counts.T;y=e/n;s=(e+.5)/(n+1);W=np.sqrt(n/(s*(1-s)))
    X=np.column_stack([np.ones(len(scales)),scales])
    linear=np.linalg.lstsq(X*W[:,None],y*W,rcond=None)[0][0]
    pars=least_squares(lambda z:(qt(z[0]*scales,z[1])-y)*W,
                       [.03,np.clip(4*linear,-.4,1.4)],
                       bounds=([1e-8,-.5],[min(.3,.999/max(scales)),1.5]),
                       ftol=1e-10,xtol=1e-10,gtol=1e-10).x
    return linear,pars[1]/4,pars[0]


def calibrated(cal,signal):
    hw=cal[:,0]/cal[:,1];q=signal[:,0]/signal[:,1]
    # Monotone cubic inversion with Newton refinement; no fitted constants.
    phat=np.minimum(1.,hw/.75)
    for _ in range(10):
        deriv=(1+2*(1-phat)+3*(1-phat)**2)/8
        phat=np.clip(phat-(q0(phat)-hw)/deriv,0,1)
    bsc=(q-hw)/(1-2*hw)
    gate=(q-hw)/(qi(phat)-hw)/4
    fiorini=(q-hw)/(.25-hw*hw)/4
    return np.column_stack([bsc,gate,fiorini])


def main():
    guard()
    cfg=P['diagnostic'];design=optimize_design();R=cfg['replicates'];p=cfg['p_signal']
    for ri,r in enumerate(cfg['p_eve']):
        path=OUT/f'diag_{round(100*r):02d}.npz'
        if path.exists():continue
        rng=np.random.default_rng(P['seed_bases']['diagnostics']+ri*10000)
        s4=np.array([1,1.5,2,2.5]);s2=np.array(design['two_point']['scales'])
        counts4=np.stack([draw(rng,2000,qt(p*s,r),R) for s in s4],axis=1)
        counts2=np.stack([draw(rng,n,qt(p*s,r),R) for s,n in zip(s2,design['two_point']['signals'])],axis=1)
        fit4=np.array([fit(c,s4) for c in counts4]);fit2=np.array([fit(c,s2) for c in counts2])
        nc=design['calibration']['calibration'];ns=design['calibration']['signal']
        signal=draw(rng,ns,qt(p,r),R)
        offsets=np.array(cfg['calibration_relative_offset'])
        calibration=np.stack([draw(rng,nc,q0(p*(1+d)),R) for d in offsets],axis=1)
        estimates=np.stack([calibrated(calibration[:,j],signal) for j in range(len(offsets))],axis=1)
        # Closest-method control under its own readout/channel law, same budget.
        native_signal=draw(rng,ns,q0(p)+r*(.25-q0(p)**2),R)
        native_estimates=calibrated(calibration[:,list(offsets).index(0)],native_signal)
        np.savez_compressed(path,counts4=counts4,counts2=counts2,fit4=fit4,fit2=fit2,
                            signal=signal,calibration=calibration,calibrated=estimates,
                            offsets=offsets,native_signal=native_signal,native_estimates=native_estimates,p_eve=r)
        print(path.name,'MAE pp',100*np.mean(abs(fit4[:,:2]-r/4),axis=0),
              100*np.mean(abs(fit2[:,1]-r/4)),flush=True)
    # An unscaled receiver channel remains in the zero-gate-noise intercept.
    for ci,c in enumerate(cfg['unscaled_channel_control_qber']):
        for ri,r in enumerate([0,.2,.4]):
            path=OUT/f'diag_channel_{round(100*c):02d}_{round(100*r):02d}.npz'
            if path.exists():continue
            rng=np.random.default_rng(9800000+ci*10000+ri*1000)
            s=np.array(design['two_point']['scales'])
            counts=np.stack([draw(rng,n,c+(1-2*c)*qt(p*f,r),R) for f,n in zip(s,design['two_point']['signals'])],axis=1)
            estimates=np.array([fit(x,s) for x in counts])
            np.savez_compressed(path,counts=counts,estimates=estimates,channel=c,p_eve=r,
                                true_intercept=c+(1-2*c)*r/4)
    print('diagnostics complete',flush=True)


if __name__=='__main__':main()
