"""Independent density-state laws, likelihood fits, interval endpoints and budgets."""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.optimize import differential_evolution,minimize,minimize_scalar
from scipy.special import xlogy
from likelihood import probabilities,profile,information,PMAX,CUTOFF

HERE=Path(__file__).resolve().parent;OUT=HERE/'results'
P=json.loads((HERE/'protocol.json').read_text())
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


def main():
    guard=json.loads((OUT/'source_guard.json').read_text())
    for name,digest in guard.items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest
    max_density=0.;density_cases=0;budgets=0
    for path in sorted(OUT.glob('*_single.npz'))+sorted(OUT.glob('*_scaled.npz')):
        with np.load(path) as z:d={k:z[k] for k in z.files}
        counts=d['counts'];assert counts.shape==(500,4,2)
        assert np.all((counts[:,:,0]>=0)&(counts[:,:,0]<=counts[:,:,1]))
        if path.stem.endswith('_single'):assert np.all(counts[:,:,1].sum(axis=1)<=8000)
        else:assert np.all(counts[:,:,1]<=np.array([5600,400,400,1600]))
        budgets+=1
        for f in [0.,1.,2.3,4.]:
            actual=density(str(d['circuit']),float(d['pA']),float(d['pB']),float(d['r']),f,float(d['readout']))
            expected=float(d['readout'])+(1-2*float(d['readout']))*probabilities([d['pA'],d['pB'],d['r']],d['paths'],np.eye(4),np.full(4,f))
            max_density=max(max_density,float(np.max(abs(actual-expected))));density_cases+=1
            assert np.max(abs(actual-expected))<1e-12,(path,f,actual,expected)
    rng=np.random.default_rng(190014);inverse_error=0.
    for _ in range(1000):
        a,b=rng.uniform(.001,.15,2);r=rng.uniform(0,.9)
        for circuit in ['PRIMARY','A']:
            paths=np.asarray(P['circuits'][circuit]);q=probabilities([a,b,r],paths,np.eye(4),np.ones(4));s=1-2*q
            if circuit=='PRIMARY':d=s[0];u=s[1]/d;v=s[2]/(d*u)
            else:u=s[1]/s[0];d=s[0]/u;v=s[2]/(d*u*u)
            error=np.max(abs(np.array([1-u,1-v,2*(1-d)])-[a,b,r]));inverse_error=max(inverse_error,error)
            assert error<1e-12
    checks=0;endpoint_checks=0;maximum_excess=0.;max_endpoint_error=0.;jac_error=0.
    fitpaths=sorted(OUT.glob('*_full_fit.npz'))+sorted(OUT.glob('*_scaled_fit.npz'))+sorted(OUT.glob('retained_*_profile.npz'))
    for path in fitpaths:
        with np.load(path) as z:d={k:z[k] for k in z.files}
        if 'estimates' not in d:continue
        if path.name.startswith('retained_'):raw=d
        else:
            name=path.name.replace('_full_fit','_single').replace('_scaled_fit','_scaled')
            with np.load(OUT/name) as z:raw={k:z[k] for k in z.files}
        counts=raw['counts'];paths=raw['paths'];groups=raw['groups'];scales=raw['scales']
        indices=sorted({0,99,int(np.argmax(d['condition'])),int(np.argmin(d['parameters'][:,0]))})
        for i in indices:
            fun=lambda x:reference_nll(x,paths,groups,scales,counts[i])
            opt=differential_evolution(fun,[(0,PMAX),(0,PMAX),(0,1)],seed=1009+i,
                popsize=10,tol=1e-10,maxiter=450,polish=True)
            actual=fun(d['parameters'][i]);excess=actual-opt.fun
            maximum_excess=max(maximum_excess,excess);assert excess<2e-6,(path.name,i,excess)
            checks+=1
            for endpoint in d['intervals'][i]*4:
                if endpoint<1e-7 or endpoint>1-1e-7:continue
                endpoint_fun=lambda x:fun([*x,endpoint])
                ref=reference_endpoint(endpoint_fun,1901+i)
                error=abs(2*(ref-actual)-CUTOFF)
                max_endpoint_error=max(max_endpoint_error,error)
                assert error<5e-5,(path.name,i,endpoint,error)
                endpoint_checks+=1
            pars=d['parameters'][i]
            if np.all(pars[:2]>1e-5) and np.all(pars[:2]<PMAX-1e-5):
                jac=information(pars,paths,groups,scales,counts[i])[3];eps=1e-6
                numeric=np.column_stack([(probabilities(pars+np.eye(3)[j]*eps,paths,groups,scales)-probabilities(pars-np.eye(3)[j]*eps,paths,groups,scales))/(2*eps) for j in range(3)])
                jac_error=max(jac_error,float(np.max(abs(jac-numeric))));assert np.allclose(jac,numeric,rtol=1e-6,atol=1e-9)
    nuisance_checks=0;max_nuisance_error=0.
    for path in sorted(OUT.glob('retained_*_nuisance.npz')):
        with np.load(path) as z:intervals=z['intervals']
        with np.load(path.with_name(path.name.replace('_nuisance','_profile'))) as z:
            counts=z['counts'];paths=z['paths'];groups=z['groups'];scales=z['scales'];nll=z['nll']
        for i in [0,99]:
            for coordinate in [0,1]:
                for x in intervals[i,coordinate]:
                    if x<1e-7 or x>PMAX-1e-7:continue
                    def fun(free):
                        pars=[x,free[0],free[1]] if coordinate==0 else [free[0],x,free[1]]
                        return reference_nll(pars,paths,groups,scales,counts[i])
                    value=reference_endpoint(fun,712+i,((0,PMAX),(0,1)))
                    error=abs(2*(value-nll[i])-CUTOFF);max_nuisance_error=max(max_nuisance_error,error)
                    assert error<5e-5,(path.name,i,coordinate,error)
                    nuisance_checks+=1
    report={'status':'passed','new_acquisition_records':budgets,'new_replicate_blocks':6500,'density_cases':density_cases,
            'maximum_density_difference':max_density,'analytic_inverse_cases':2000,'maximum_inverse_error':inverse_error,
            'independent_global_optimizer_checks':checks,'maximum_optimizer_objective_excess':maximum_excess,
            'independent_profile_endpoint_checks':endpoint_checks,'maximum_endpoint_deviance_error':max_endpoint_error,
            'maximum_jacobian_difference':jac_error,'independent_device_profile_endpoints':nuisance_checks,
            'maximum_device_endpoint_deviance_error':max_nuisance_error}
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
