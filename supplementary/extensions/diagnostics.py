"""Separated-device diagnostics, presimulation design and robustness controls.

Fiorini's estimator is attributed and licensed in ../study/diagnostics_v2.py
and ../study/third_party/LICENSE-Fiorini.txt. Its full detection framework is
not evaluated here. Every interval below concerns one acquisition replicate.
"""
import json
import numpy as np
from scipy.optimize import root
from common import OUT,P,seed
from physics import Q,q_noeve,q_intercept
from provenance import atomic_json,atomic_npz,guarded_run

METHODS=['linear_four','gate_four','gate_two','calibrated_BSC','calibrated_gate','Fiorini']


def response(f,p,r,e=0.,case='legitimate'):
    eve=p if case=='all_scaled' else e
    return Q(f,p,p,eve,r,case)


def target(r,e=0.,case='legitimate'):
    return np.asarray(r)*(float(q_intercept(0.,0.,e)) if case!='all_scaled' else .25)


def fisher(scales,signals,p,r):
    f=np.asarray(scales);q=response(f,p,r);eps=1e-6
    dp=(response(f,p+eps,r)-response(f,p-eps,r))/(2*eps)
    dr=(response(f,p,r+eps)-response(f,p,r-eps))/(2*eps)
    J=np.column_stack([dp,dr]);information=J.T@((np.asarray(signals)/2/(q*(1-q)))[:,None]*J)
    return float(np.linalg.inv(information)[1,1]/16)


def design():
    path=OUT/'frozen_diagnostic_design.json'
    if path.exists():return json.loads(path.read_text())
    grid=[(p,r) for p in P['diagnostics']['p_legitimate'] for r in P['diagnostics']['attack']]
    candidates=[]
    for high in [2.5,4,6,8]:
        for share in [.5,.6,.7,.8,.9]:
            variance=[fisher([1,high],[8000*share,8000*(1-share)],p,r) for p,r in grid]
            candidates.append(dict(scales=[1,high],share=share,worst_variance=max(variance)))
    cal=[]
    for share in [.2,.3,.4,.5,.6,.7,.8]:
        variances=[]
        for p,r in grid:
            eps=1e-6;h=float(q_noeve(p,p));q=float(response(1,p,r))
            dh=(q_noeve(p+eps,p+eps)-q_noeve(p-eps,p-eps))/(2*eps)
            dp=(response(1,p+eps,r)-response(1,p-eps,r))/(2*eps)
            d=float(q_intercept(p,p,0)-h)
            variances.append(float((q*(1-q)/(4000*(1-share))+(dp/dh)**2*h*(1-h)/(4000*share))/(16*d*d)))
        cal.append(dict(calibration_share=share,worst_variance=max(variances)))
    result=dict(two=min(candidates,key=lambda x:x['worst_variance']),
                calibration=min(cal,key=lambda x:x['worst_variance']),
                candidates=candidates,calibration_candidates=cal,grid=grid,
                selection='Minimize worst Fisher/delta variance on declared design grid before any diagnostic simulation.',
                methods=METHODS,confidence='Unclipped signed point estimates; plug-in 95% delta/Wald intervals.',
                noisy_Eve_controls=dict(p_eve=.03,knowledge='known to model-based estimators; unknown-Eve identifiability separately assessed'),
                robustness='Main48 cells use ideal Eve. Noisy and all-scaled cases use all16(p,r) cells at budget8000. Stress uses declared(.03,.2,8000) cell.')
    atomic_json(path,result);return result


def counts(rng,signals,q,R=500):
    n=rng.binomial(int(signals),.5,R)
    return np.column_stack([rng.binomial(n,q),n])


def nested_counts(rng,signals,q,R=500):
    """Common initial transmitted signals couple base batches without budget reuse."""
    signals=np.asarray(signals,int);result={};running=np.zeros((R,2),np.int64);last=0
    for size in np.unique(signals):
        running+=counts(rng,int(size-last),q,R);result[int(size)]=running.copy();last=size
    return [result[int(size)] for size in signals]


def observations(c):
    e=c[...,0];n=c[...,1];y=e/np.maximum(n,1);s=(e+.5)/(n+1)
    return y,s*(1-s)/np.maximum(n,1)


def linear_fit(c,f):
    y,var=observations(c);w=1/var;f=np.asarray(f)[None]
    A=w.sum(1);B=(w*f).sum(1);C=(w*f*f).sum(1);det=A*C-B*B
    coefficients=w*(C[:,None]-B[:,None]*f)/det[:,None]
    return (coefficients*y).sum(1),np.sqrt((coefficients**2*var).sum(1))


def gate_fit(c,f,e=0.,case='legitimate'):
    """Batched two-parameter WLS, checked against independent SciPy fits."""
    y,var=observations(c);w=1/var;f=np.asarray(f)[None]
    linear,_=linear_fit(c,f.ravel());p=np.full(len(c),.025);r=np.clip(4*linear,-.4,1.4)
    pmax=min(.3,.999/float(f.max()));eps=1e-6
    def predicted(pp,rr):return response(f,pp[:,None],rr[:,None],e,case)
    def optimal_r(pp):
        base=predicted(pp,np.zeros(len(pp)))
        slope=predicted(pp,np.ones(len(pp)))-base
        return np.clip(np.sum(w*slope*(y-base),axis=1)/np.sum(w*slope*slope,axis=1),-.5,1.5)
    r=optimal_r(p)
    for iteration in range(70):
        pred=predicted(p,r);res=y-pred
        j0=(predicted(p+eps,r)-predicted(p-eps,r))/(2*eps)
        j1=(predicted(p,r+eps)-predicted(p,r-eps))/(2*eps)
        A=(w*j0*j0).sum(1);B=(w*j0*j1).sum(1);C=(w*j1*j1).sum(1)
        det=np.maximum(A*C-B*B,1e-20);g0=(w*j0*res).sum(1);g1=(w*j1*res).sum(1)
        dp=(C*g0-B*g1)/det
        # Conditional response is exactly affine in r. Reoptimizing r at every
        # trial p also handles p's constrained boundary, where a projected
        # unconstrained joint step can otherwise leave r far from its optimum.
        active=(r<=-.5+1e-12)|(r>=1.5-1e-12)
        dp[active]=g0[active]/np.maximum(A[active],1e-20)
        before=(w*res*res).sum(1);step=np.ones(len(c))
        for _ in range(12):
            pn=np.clip(p+step*dp,1e-8,pmax);rn=optimal_r(pn)
            after=(w*(y-predicted(pn,rn))**2).sum(1)
            bad=after>before+1e-10
            if not bad.any():break
            step[bad]*=.5
        if np.max(abs(pn-p))+np.max(abs(rn-r))<1e-10:
            p,r=pn,rn;break
        p,r=pn,rn
    j0=(predicted(p+eps,r)-predicted(p-eps,r))/(2*eps)
    j1=(predicted(p,r+eps)-predicted(p,r-eps))/(2*eps)
    A=(w*j0*j0).sum(1);B=(w*j0*j1).sum(1);C=(w*j1*j1).sum(1)
    det=np.maximum(A*C-B*B,1e-20)
    coefficient=float(target(1.,e,case))
    se=coefficient*np.sqrt(A/det)
    return coefficient*r,se,p,r


def invert_calibration(h):
    p=np.clip(np.asarray(h)/.75,0,1)
    for _ in range(16):
        derivative=(1+2*(1-p)+3*(1-p)**2)/8
        p=np.clip(p-(q_noeve(p,p)-h)/derivative,0,1)
    return p


def inversion(h,q,e=0.,case='legitimate'):
    p=invert_calibration(h);eve=p if case=='all_scaled' else e
    denominator=q_intercept(p,p,eve)-q_noeve(p,p)
    coefficient=float(target(1.,e,case))
    # Fiorini's interception-density estimate is (q-h)/(.25-h*h).
    # Map it to the common residual-disturbance target using the same known
    # zero-legitimate-noise coefficient as the model-based comparator. With
    # ideal Eve this is precisely the original density/4 disturbance formula.
    return np.column_stack([(q-h)/(1-2*h),coefficient*(q-h)/denominator,
                            coefficient*(q-h)/(.25-h*h)])


def calibrated(cal,signal,e=0.,case='legitimate'):
    h,vh=observations(cal);q,vq=observations(signal);eps=1e-6
    values=inversion(h,q,e,case)
    dq=(inversion(h,q+eps,e,case)-inversion(h,q-eps,e,case))/(2*eps)
    lower=np.maximum(h-eps,0);upper=h+eps
    dh=(inversion(upper,q,e,case)-inversion(lower,q,e,case))/(upper-lower)[:,None]
    se=np.sqrt(dq*dq*vq[:,None]+dh*dh*vh[:,None])
    return values,se


def simulate(p,r,budget,rngseed,e=0.,case='legitimate',stress='nominal'):
    cfg=design();rng=np.random.default_rng(rngseed);R=P['diagnostics']['replicates']
    f4=np.array([1.,1.5,2.,2.5]);f2=np.array(cfg['two']['scales'],float)
    n4=np.full(4,budget//4,int);n4[-1]+=budget-n4.sum()
    n2=np.array([round(budget*cfg['two']['share']),0]);n2[1]=budget-n2[0]
    nc=round(budget*cfg['calibration']['calibration_share']);ns=budget-nc
    channel=0.;offset=0.;epsilon=0.
    if stress.startswith('calibration_'):offset=float(stress.split('_')[1])
    if stress.startswith('curvature_'):epsilon=float(stress.split('_')[1])
    if stress.startswith('unscaled_'):channel=float(stress.split('_')[1])
    def law(f,time=0.):
        legitimate=p*(1+.1*time) if stress=='noise_ramp' else p
        attack=r*(1+.5*time) if stress=='attack_ramp' else r
        value=response(f+epsilon*(f-1)**2,legitimate,attack,e,case)
        return channel+(1-2*channel)*value
    if stress in ['noise_ramp','attack_ramp']:
        c4=np.stack([counts(rng,n,law(f,t),R) for f,n,t in zip(f4,n4,np.linspace(-.5,.5,4))],axis=1)
        c2=np.stack([counts(rng,n,law(f,t),R) for f,n,t in zip(f2,n2,[-.5,.5])],axis=1)
        signal=counts(rng,ns,law(1,.5),R)
    else:
        base4,base2,signal=nested_counts(rng,[n4[0],n2[0],ns],law(1),R)
        c4=np.stack([base4]+[counts(rng,n4[j],law(f),R) for j,f in enumerate(f4[1:],1)],axis=1)
        c2=np.stack([base2,counts(rng,n2[1],law(f2[1]),R)],axis=1)
    pcal=p*(1+offset)*(.95 if stress=='noise_ramp' else 1.)
    qcal=channel+(1-2*channel)*q_noeve(pcal,pcal)
    cal=counts(rng,nc,qcal,R)
    lin,slin=linear_fit(c4,f4);g4,s4,pp4,rr4=gate_fit(c4,f4,e,case)
    g2,s2,pp2,rr2=gate_fit(c2,f2,e,case);cv,cs=calibrated(cal,signal,e,case)
    estimates=np.column_stack([lin,g4,g2,cv]);se=np.column_stack([slin,s4,s2,cs])
    truth=channel+(1-2*channel)*float(target(r,e,case))
    return dict(counts4=c4,counts2=c2,calibration=cal,signal=signal,
                estimates=estimates,standard_errors=se,parameters_four=np.column_stack([pp4,rr4]),
                parameters_two=np.column_stack([pp2,rr2]),methods=np.array(METHODS),
                target=truth,ideal_attack=r/4,p=p,r=r,pE=e,case=case,stress=stress,
                budget=budget,scales_four=f4,scales_two=f2,signals_four=n4,signals_two=n2,
                signals_calibration=nc,signals_attacked=ns,seed=rngseed)


def identifiability():
    f=np.array([1.,8.]);truth=response(f,.03,.2,.03);alternatives=[]
    for eve in [0.,.015,.03,.06,.1]:
        solution=root(lambda z:response(f,z[0],z[1],eve)-truth,[.03,.2])
        if solution.success and 0<solution.x[0]<.125 and 0<solution.x[1]<1:
            p,r=solution.x
            alternatives.append(dict(p=float(p),r=float(r),pE=eve,
                probabilities=response(f,p,r,eve).tolist(),zero_legitimate_intercept=float(target(r,eve))))
    eps=1e-6;pars=np.array([.03,.2,.03]);J=[]
    for j in range(3):
        delta=np.eye(3)[j]*eps
        plus=pars+delta;minus=pars-delta
        J.append((response(f,plus[0],plus[1],plus[2])-response(f,minus[0],minus[1],minus[2]))/(2*eps))
    result=dict(settings=f.tolist(),jacobian_rank=int(np.linalg.matrix_rank(np.array(J).T)),
                unknown_parameters=3,equivalent_two_setting_solutions=alternatives,
                scope='Two aggregate settings cannot identify unknown legitimate noise, attack fraction and fixed Eve noise simultaneously. Known-Eve fits impose an extra assumption; four-setting identifiability is not ruled out.')
    atomic_json(OUT/'diagnostic_identifiability.json',result)


def main():
    with guarded_run('diagnostics',['protocol.json','common.py','physics.py','diagnostics.py']):
        design();cfg=P['diagnostics']
        for pi,p in enumerate(cfg['p_legitimate']):
            for ri,r in enumerate(cfg['attack']):
                for bi,budget in enumerate(cfg['budgets']):
                    path=OUT/f'diagnostic_p{pi}_r{ri}_b{bi}.npz'
                    if path.exists():continue
                    atomic_npz(path,**simulate(p,r,budget,seed('diagnostic',0,pi,ri,bi)))
                    print(path.name,flush=True)
        for ci,(case,e) in enumerate([('legitimate',.03),('all_scaled',0.)],1):
            for pi,p in enumerate(cfg['p_legitimate']):
                for ri,r in enumerate(cfg['attack']):
                    path=OUT/f'devicecase_{ci}_p{pi}_r{ri}.npz'
                    if path.exists():continue
                    atomic_npz(path,**simulate(p,r,8000,seed('diagnostic',ci,pi,ri),e,case))
                    print(path.name,flush=True)
        stresses=['calibration_-0.2','calibration_-0.1','calibration_0.1','calibration_0.2',
                  'curvature_-0.02','curvature_0.02','unscaled_0.01','unscaled_0.03','noise_ramp','attack_ramp']
        for si,stress in enumerate(stresses):
            path=OUT/f'diagnostic_stress_{stress}.npz'
            if path.exists():continue
            atomic_npz(path,**simulate(.03,.2,8000,seed('diagnostic',3,si),stress=stress))
            print(path.name,flush=True)
        identifiability()


if __name__=='__main__':main()
