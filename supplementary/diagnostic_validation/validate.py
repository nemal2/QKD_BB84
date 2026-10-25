"""Independent laws, optimizer references, budgets, and retained-record checks."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from scipy.optimize import minimize_scalar,least_squares
from models import unequal_response,unequal_jacobian,basis_response
from experiment import P,OUT,configurations,allocate

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'extensions'))
from physics import exact_density_Q
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'finalization'))
from channel_information import density_basis_response


def reference_two(y,variance,scales=None):
    """Independent scalar profile with coarse search and bounded refinement."""
    weight=1/variance;pmax=.3 if scales is None else min(.3,.999/max(scales))
    def qnoise(p):
        if scales is None:return np.array([p/4,(5*p-4*p*p+p**3)/4])
        x=np.asarray(scales)*p
        return (6*x-4*x*x+x**3)/8
    def profile(p,full=False):
        noise=qnoise(p);slope=.25-noise/2
        r=np.clip(np.dot(weight*slope,y-noise)/np.dot(weight,slope*slope),-.5,1.5)
        value=float(np.dot(weight,(y-noise-r*slope)**2))
        return (value,r) if full else value
    grid=np.linspace(1e-8,pmax,65);values=np.array([profile(p) for p in grid]);candidates=[grid[0],grid[-1]]
    for lo,hi in [(grid[0],grid[1]),(grid[-2],grid[-1])]:
        candidates.append(minimize_scalar(profile,bounds=(lo,hi),method='bounded',options={'xatol':1e-13}).x)
    for j in range(1,len(grid)-1):
        if values[j]<=values[j-1] and values[j]<=values[j+1]:
            candidates.append(minimize_scalar(profile,bounds=(grid[j-1],grid[j+1]),method='bounded',options={'xatol':1e-13}).x)
    p=min(candidates,key=profile);value,r=profile(p,True)
    return value,p,r


def main():
    source_guard=json.loads((OUT/'source_guard.json').read_text())
    for name,digest in source_guard.items():
        assert hashlib.sha256((OUT.parent.parent/name).read_bytes()).hexdigest()==digest,name
    designs=configurations()['methods'];density_cases=0;optimizer_checks=0;three_checks=0
    max_excess=0.;max_two_intercept_difference=0.;max_density_difference=0.;files=0
    for stage in ['nominal','off_grid']:
        for p in P[stage]['p']:
            for r in P[stage]['r']:
                b=basis_response(p,r);bb=density_basis_response(p,r)
                error=float(np.max(abs(b-bb)));assert error<1e-12;max_density_difference=max(max_density_difference,error)
                for f in [0.,1.,2.3,2.5,4.,8.]:
                    error=abs(float(unequal_response(f,p,p,r))-exact_density_Q(f,p,p,0.,r))
                    assert error<1e-12;max_density_difference=max(max_density_difference,error);density_cases+=1
    for path in sorted(OUT.glob('*.npz')):
        files+=1
        with np.load(path,allow_pickle=False) as z:d={k:z[k] for k in z.files}
        assert len(d['estimates'])==P['replicates']
        assert np.isfinite(d['estimates']).all() and np.isfinite(d['standard_errors']).all()
        method=str(d['method'])
        if method=='calibrated':
            budget=int(d['budget']);nc=round(budget*P['design']['calibration_share'])
            for c,n in [(d['calibration'],nc),(d['signal'],budget-nc)]:
                assert ((0<=c[:,0])&(c[:,0]<=c[:,1])&(c[:,1]<=n)).all()
            continue
        c=d['counts'];assert np.all((c[...,0]>=0)&(c[...,0]<=c[...,1]))
        budget=int(d['training_budget'] if 'training_budget' in d else d['budget'])
        if method=='basis_single':
            assert np.all(c[:,:,1].sum(axis=1)<=budget)
            assert d['moment_valid'].all()
        else:
            design=designs['aggregate_four_cap4' if method=='independent_devices' else method]
            assert np.array_equal(d['scales'],design['scales'])
            signals=allocate(budget,design['weights']);assert np.all(c[:,:,1]<=signals)
        y=c[:,:,0]/c[:,:,1];smooth=(c[:,:,0]+.5)/(c[:,:,1]+1);variance=smooth*(1-smooth)/c[:,:,1]
        indices=set([0,99,199,299,499,int(np.argmin(d['parameters'][:,0])),int(np.argmax(d['parameters'][:,0]))])
        if method=='independent_devices':
            original=path.with_name(path.name.replace('independent_devices','aggregate_four_cap4'))
            with np.load(original,allow_pickle=False) as z:assert np.array_equal(c,z['counts']) and np.array_equal(d['validation'],z['validation'])
            for i in sorted(indices):
                f=d['scales'];pmax=min(.3,.999/max(f));roots=np.sqrt(variance[i])
                def fun(x):
                    # Explicit path-average reference, independent of polynomial evaluation.
                    a,b,r=x;aa=1-f*a;bb=1-f*b
                    q=((1-aa)+(1-aa*bb)+(1-aa*aa*bb))/8
                    return (r/4+(1-r/2)*q-y[i])/roots
                best=np.inf
                for start in [[.01,.05,.2],[.05,.01,.2],[.03,.03,.2]]:
                    fit=least_squares(fun,start,bounds=([1e-8,1e-8,-.5],[pmax,pmax,1.5]),
                                      xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=2000,x_scale='jac')
                    best=min(best,float(np.dot(fit.fun,fit.fun)))
                actual=float(np.dot(fun(d['parameters'][i]),fun(d['parameters'][i])))
                excess=actual-best;max_excess=max(max_excess,excess)
                assert excess<1e-6,(path.name,i,actual,best,d['parameters'][i])
                jac=unequal_jacobian(f,*d['parameters'][i]);eps=1e-6;x=d['parameters'][i]
                numeric=np.column_stack([(unequal_response(f,*(x+np.eye(3)[j]*eps))-unequal_response(f,*(x-np.eye(3)[j]*eps)))/(2*eps) for j in range(3)])
                assert np.allclose(jac,numeric,rtol=2e-6,atol=1e-10)
                three_checks+=1
        else:
            f=None if method=='basis_single' else d['scales']
            for i in sorted(indices):
                reference,p,r=reference_two(y[i],variance[i],f)
                pp,rr=d['parameters'][i]
                predicted=basis_response(pp,rr) if f is None else unequal_response(f,pp,pp,rr)
                actual=float(np.sum((predicted-y[i])**2/variance[i]));excess=actual-reference
                max_excess=max(max_excess,excess)
                assert excess<1e-6,(path.name,i,actual,reference,pp,p)
                max_two_intercept_difference=max(max_two_intercept_difference,abs(rr-r)/4)
                optimizer_checks+=1
        if 'validation' in d:
            a,b,r,epsilon=[float(d[key]) for key in ['pA','pB','r','curvature']]
            f=float(d['validation_scale']);effective=f+epsilon*f*(f-1)
            exact=exact_density_Q(effective,a,b,0.,r)
            assert abs(exact-float(d['true_validation_qber']))<1e-12
            density_cases+=1
            validation=d['validation'];assert np.all((validation[:,0]>=0)&(validation[:,0]<=validation[:,1])&(validation[:,1]<=int(d['validation_budget'])))
    assert files==476,files
    report=dict(status='passed',records_checked=files,density_cases=density_cases,
                independent_two_parameter_fits=optimizer_checks,independent_three_parameter_fits=three_checks,
                maximum_density_difference=max_density_difference,maximum_reference_objective_excess=max_excess,
                maximum_two_parameter_reference_intercept_difference=max_two_intercept_difference,
                scope='All retained acquisition budgets and counts; density-state laws; fixed and extremal sample fits against independent optimizer references. No decoder experiments in this validation.')
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
