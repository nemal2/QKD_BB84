"""Independent checks of response laws, proposition identities, fits, and intervals."""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.optimize import differential_evolution,minimize,minimize_scalar
from scipy.special import xlogy
from likelihood import PMAX,CUTOFF
from experiment import P,cells
HERE=Path(__file__).resolve().parent;OUT=HERE/'results'
I=np.eye(2);X=np.array([[0.,1.],[1.,0.]]);H=np.array([[1.,1.],[1.,-1.]])/np.sqrt(2)
G={'i':I,'x':X,'h':H}
PREP={'PRIMARY':[[],['x'],['h'],['x','h']],
      'A':[['i'],['i','x'],['i','h'],['i','x','h']],
      'B+':[['i','i'],['i','x'],['i','h'],['x','h']]}


def gate(rho,g,p):return (1-p)*(G[g]@rho@G[g].T)+p*I/2


def density(circuit,a,b,r,f,c=0.):
    rates=[]
    for k in range(4):
        basis=k//2;bit=k%2;rho=np.array([[1.,0.],[0.,0.]])
        for g in PREP[circuit][k]:rho=gate(rho,g,f*a)
        def receive(state):
            if basis:state=gate(state,'h',f*b)
            elif circuit=='B+':state=gate(state,'i',f*b)
            return float(state[1-bit,1-bit])
        q0=receive(rho);qi=0.
        for eb in [0,1]:
            state=gate(rho,'h',0) if eb else rho
            for outcome in [0,1]:
                resend=np.zeros((2,2));resend[outcome,outcome]=1
                if eb:resend=gate(resend,'h',0)
                qi+=.5*state[outcome,outcome]*receive(resend)
        q=(1-r)*q0+r*qi;rates.append(c+(1-2*c)*q)
    return np.array(rates)


def reference_nll(parameters,paths,groups,scales,counts):
    a,b,r=parameters
    # Direct products of per-path contractions, independent of the linear-B fit.
    q=[]
    for f,weights in zip(scales,groups):
        paths_q=[.5-(.5-r/4)*(1-f*a)**int(ka)*(1-f*b)**int(kb) for ka,kb in paths]
        q.append(np.dot(weights,paths_q))
    q=np.clip(q,1e-14,1-1e-14);e,n=counts[:,0],counts[:,1]
    return float(-np.sum(xlogy(e,q)+xlogy(n-e,1-q)))


def reference_endpoint(fun,seed,bounds=((0,PMAX),(0,PMAX))):
    # Evolution alone can miss a better boundary mode; inspect every edge explicitly.
    best=min(differential_evolution(fun,bounds,seed=seed+k,
        popsize=15,tol=1e-11,maxiter=450,polish=True).fun for k in [0,31])
    for coordinate in [0,1]:
        for fixed in bounds[coordinate]:
            scalar=lambda t:fun([fixed,t] if coordinate==0 else [t,fixed])
            grid=np.linspace(*bounds[1-coordinate],65);values=[scalar(t) for t in grid]
            best=min(best,*values)
            for i in range(1,64):
                if values[i]<=values[i-1] and values[i]<=values[i+1]:
                    best=min(best,minimize_scalar(scalar,bounds=(grid[i-1],grid[i+1]),
                        method='bounded',options={'xatol':1e-13}).fun)
    return best


def proposition_checks():
    rng=np.random.default_rng(57412007);error_inverse=0.;error_intercept=0.;error_injection=0.
    # Include noise/interception boundaries as well as interior parameters.
    parameters=[(a,b,r) for a in [0.,.03,PMAX] for b in [0.,.05,PMAX] for r in [0.,1.]]
    parameters += [(*rng.uniform(0,PMAX,2),rng.uniform(0,1)) for _ in range(1000)]
    for a,b,r in parameters:
        for c in [0.,.1]:
            d=(1-2*c)*(1-r/2);u=1-a;v=1-b
            for circuit in ['PRIMARY','A']:
                s=1-2*density(circuit,a,b,r,1.,c)
                if circuit=='PRIMARY':dd=s[0];uu=s[1]/dd;vv=s[2]/(dd*uu)
                else:uu=s[1]/s[0];dd=s[0]/uu;vv=s[2]/(dd*uu*uu)
                discrepancy=max(abs(dd-d),abs(uu-u),abs(vv-v))
                error_inverse=max(error_inverse,discrepancy);assert discrepancy<2e-12
            scales=np.array(P['scales']);q=np.array([density('B+',a,b,r,f,c).mean() for f in scales])
            weights=np.array([np.prod([-scales[k]/(scales[j]-scales[k]) for k in range(4) if k!=j]) for j in range(4)])
            err=abs(weights@q-(1-d)/2);error_intercept=max(error_intercept,err);assert err<3e-12
        for p in [a,b]:
            for f in P['scales']:
                eta=(f-1)*p/(1-p)
                err=abs(p+eta-p*eta-f*p);error_injection=max(error_injection,err)
                assert 0<=eta<=1 and err<1e-14
    a,b,r=.03,.03,.2;s=(1-r/2)*(1-a)**2*(1-b)
    a2=.02;r2=2*(1-s/((1-a2)**2*(1-b)))
    assert 0<r2<1 and abs(r2-r)>.01
    assert np.max(abs(density('B+',a,b,r,1.)-density('B+',a2,b,r2,1.)))<1e-14
    # Equal residual with unequal interception and readout, at every scale/path.
    ec=.059;c2=.02;r2=4*(ec-c2)/(1-2*c2)
    for circuit in P['circuits']:
        for f in [0.,1.,2.3,4.]:
            assert np.max(abs(density(circuit,.03,.05,.2,f,.01)-density(circuit,.03,.05,r2,f,c2)))<1e-14
    return dict(parameter_cases=len(parameters),inverse_checks=4*len(parameters),
        cubic_intercept_checks=2*len(parameters),maximum_inverse_error=error_inverse,
        maximum_intercept_error=error_intercept,maximum_injection_composition_error=error_injection,
        Bplus_distinct_interception_counterexample=True,unknown_readout_counterexample=True)


def main():
    for name,digest in json.loads((OUT/'source_guard.json').read_text()).items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,name
    report={'status':'passed','proposition':proposition_checks()}
    density_cases=0;max_density=0.;fit_checks=0;max_excess=0.;endpoint_checks=0;max_endpoint=0.;arrays=0
    for index,(circuit,budget,r,a,b) in enumerate(cells()):
        for mode in ['full','scaled']:
            with np.load(OUT/f'cell_{index:03d}_{mode}_counts.npz') as z:d={k:z[k] for k in z.files}
            count=d['counts'];assert count.shape==(P['replicates'],4,2)
            assert np.all(count[:,:,0]>=0) and np.all(count[:,:,0]<=count[:,:,1])
            if mode=='full':assert np.all(count[:,:,1].sum(axis=1)<=budget)
            else:assert np.all(count[:,:,1]<=d['transmitted']) and d['transmitted'].sum()==budget
            # Check the exact generating law against independent density propagation.
            expected=np.array([np.dot(weights,density(circuit,a,b,r,f)) for f,weights in zip(d['scales'],d['groups'])])
            error=float(np.max(abs(expected-d['true_probabilities'])));max_density=max(max_density,error)
            assert error<1e-12;density_cases+=4;arrays+=len(d)
            # Recreate every acquisition from its declared stream, without using stored counts.
            rng=np.random.default_rng(np.random.SeedSequence([P['seed_base'],index,int(mode=='scaled')]))
            if mode=='full':n=rng.multinomial(budget,[.125]*4+[.5],size=P['replicates'])[:,:4]
            else:n=rng.binomial(d['transmitted'],.5,size=(P['replicates'],4))
            e=rng.binomial(n,d['true_probabilities'])
            assert np.array_equal(np.stack([e,n],axis=-1),count)
            if circuit=='B+' and mode=='full':continue
            with np.load(OUT/f'cell_{index:03d}_{mode}_fit.npz') as z:fit={k:z[k] for k in z.files}
            assert np.all(np.isfinite(fit['parameters'])) and np.all(fit['parameters']>=0)
            assert np.all(fit['parameters'][:,:2]<=PMAX) and np.all(fit['parameters'][:,2]<=1)
            assert np.all(fit['intervals'][:,0]<=fit['estimates']+1e-8)
            assert np.all(fit['intervals'][:,1]>=fit['estimates']-1e-8)
            arrays+=len(fit)
            indices=sorted({index%P['replicates'],int(np.argmax(np.diff(fit['intervals'],axis=1)[:,0]))})
            for i in indices:
                fun=lambda x:reference_nll(x,d['paths'],d['groups'],d['scales'],count[i])
                opt=differential_evolution(fun,[(0,PMAX),(0,PMAX),(0,1)],seed=57418000+index*10+i,
                    popsize=10,tol=2e-10,maxiter=400,polish=True)
                candidates=[opt.fun]
                if fit['parameters'][i,2]<1e-6:
                    boundary=lambda x:fun([*x,0.])
                    candidates.append(reference_endpoint(boundary,57419000+index+i))
                value=fun(fit['parameters'][i]);excess=value-min(candidates)
                max_excess=max(max_excess,excess);assert excess<3e-6,(index,mode,i,excess)
                assert abs(value-fit['nll'][i])<1e-8
                fit_checks+=1
            # Endpoint checks cover both identified circuits and B+, all budgets and r levels.
            if a==.03 and b==.03:
                i=index%P['replicates'];fun=lambda x:reference_nll(x,d['paths'],d['groups'],d['scales'],count[i])
                base=fun(fit['parameters'][i])
                for endpoint in 4*fit['intervals'][i]:
                    if endpoint<1e-7 or endpoint>1-1e-7:continue
                    fixed=lambda x:fun([*x,endpoint]);ref=reference_endpoint(fixed,57420000+index)
                    error=abs(2*(ref-base)-CUTOFF);max_endpoint=max(max_endpoint,error)
                    assert error<5e-5,(index,mode,i,endpoint,error);endpoint_checks+=1
        print('Validated cell',index+1,flush=True)
    report.update(cells=len(cells()),replicate_blocks=len(cells())*P['replicates'],
        acquisition_streams_recreated=2*len(cells()),density_cases=density_cases,
        maximum_density_error=max_density,independent_optimizer_checks=fit_checks,
        maximum_optimizer_objective_excess=max_excess,independent_profile_endpoint_checks=endpoint_checks,
        maximum_profile_endpoint_deviance_error=max_endpoint,retained_arrays=arrays)
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()

