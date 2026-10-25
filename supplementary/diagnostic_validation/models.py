"""Diagnostic response laws and predeclared weighted least-squares estimators."""
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import minimize_scalar

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'extensions'))
from diagnostics import observations,gate_fit,calibrated


# Analytic path laws and their derivatives for the declared observation maps.
def basis_noise(p):
    p=np.asarray(p)
    return np.stack([p/4,(5*p-4*p*p+p**3)/4],axis=-1)


def basis_response(p,r):
    return np.asarray(r)[...,None]/4+(1-np.asarray(r)[...,None]/2)*basis_noise(p)


def unequal_response(f,a,b,r):
    f=np.asarray(f)
    q=((4*a+2*b)*f-(a*a+3*a*b)*f*f+a*a*b*f**3)/8
    return r/4+(1-r/2)*q


def unequal_jacobian(f,a,b,r):
    f=np.asarray(f)
    q=((4*a+2*b)*f-(a*a+3*a*b)*f*f+a*a*b*f**3)/8
    da=(4*f-(2*a+3*b)*f*f+2*a*b*f**3)/8
    db=(2*f-3*a*f*f+a*a*f**3)/8
    return np.stack([(1-r/2)*da,(1-r/2)*db,.25-q/2],axis=-1)


def equal_jacobian(p,r,scales=None):
    p=np.asarray(p);r=np.asarray(r)
    if scales is None:
        q=basis_noise(p)
        dp=np.stack([np.full_like(p,.25),(5-8*p+3*p*p)/4],axis=-1)
        return np.stack([(1-r[...,None]/2)*dp,.25-q/2],axis=-1)
    j=unequal_jacobian(np.asarray(scales)[None,:],p[:,None],p[:,None],r[:,None])
    return np.stack([j[...,0]+j[...,1],j[...,2]],axis=-1)


# Local information and covariance calculations used by the WLS diagnostic estimators.
def covariance(jac,variance):
    information=np.einsum('nki,nk,nkj->nij',jac,1/variance,jac)
    inverse=np.linalg.inv(information)
    assert np.all(np.isfinite(inverse)) and np.all(np.diagonal(inverse,axis1=1,axis2=2)>0)
    return inverse


# Constrained operating-data fit using the two disclosed basis rates.
def fit_basis(counts):
    y,var=observations(counts);w=1/var
    contrast=(y[:,1]-y[:,0])/(1-2*y[:,0])
    discriminant=1-2*contrast
    raw_p=1-np.sqrt(np.maximum(discriminant,0))
    candidates=np.stack([np.zeros(len(y))+1e-8,np.clip(raw_p,1e-8,.3),np.full(len(y),.3)])
    candidates_r=[];losses=[]
    for p in candidates:
        q=basis_noise(p);slope=.25-q/2
        r=np.clip(np.sum(w*slope*(y-q),axis=1)/np.sum(w*slope*slope,axis=1),-.5,1.5)
        candidates_r.append(r);losses.append(np.sum(w*(y-basis_response(p,r))**2,axis=1))
    ix=np.argmin(losses,axis=0);cols=np.arange(len(y))
    p=candidates[ix,cols];r=np.array(candidates_r)[ix,cols]
    # Bound-active r is uncommon, but requires profiling rather than projecting.
    active=(r<=-.5+1e-10)|(r>=1.5-1e-10)|(discriminant<0)
    for i in np.where(active)[0]:
        def objective(pp):
            q=basis_noise(pp);slope=.25-q/2
            rr=np.clip(np.sum(w[i]*slope*(y[i]-q))/np.sum(w[i]*slope*slope),-.5,1.5)
            return float(np.sum(w[i]*(y[i]-basis_response(pp,rr))**2)),rr
        fitted=minimize_scalar(lambda pp:objective(pp)[0],bounds=(1e-8,.3),method='bounded',options={'xatol':1e-13})
        best=min([1e-8,.3,fitted.x],key=lambda pp:objective(pp)[0]);p[i]=best;r[i]=objective(best)[1]
    cov=covariance(equal_jacobian(p,r),var)
    raw_r=(4*y[:,0]-raw_p)/(1-raw_p/2)
    valid=discriminant>1e-12
    raw_cov=np.full((len(y),2,2),np.nan)
    raw_cov[valid]=covariance(equal_jacobian(raw_p[valid],raw_r[valid]),var[valid])
    return dict(estimates=r/4,standard_errors=np.sqrt(cov[:,1,1])/4,parameters=np.column_stack([p,r]),covariance=cov,
        moment_estimates=raw_r/4,moment_standard_errors=np.sqrt(raw_cov[:,1,1])/4,moment_valid=valid)


# Equal-device fit for the declared aggregate scaling design.
def fit_aggregate(counts,scales):
    estimates,se,p,r=gate_fit(counts,scales)
    _,var=observations(counts);cov=covariance(equal_jacobian(p,r,scales),var)
    assert np.allclose(se,np.sqrt(cov[:,1,1])/4,rtol=1e-5,atol=1e-9)
    return dict(estimates=estimates,standard_errors=se,parameters=np.column_stack([p,r]),covariance=cov)


# Separate-device fit with physical parameter bounds and boundary candidates.
def fit_unequal(counts,scales):
    """Profile Alice noise; the other parameters form a constrained quadratic fit.

    d=1-r/2 and k=d*pB lie in a trapezoid. For each pA, evaluate the
    unconstrained quadratic solution and all four edges exactly. Search every
    local minimum on a fixed 129-point pA grid, including both endpoints.
    """
    y,var=observations(counts);f=np.asarray(scales);pmax=min(.3,.999/f.max())
    vertices=np.array([[.25,0],[1.25,0],[1.25,1.25*pmax],[.25,.25*pmax]])
    parameters=[];losses=[]
    for yy,vv in zip(y,var):
        weight=1/vv;target=yy-.5
        def profile(a,full=False):
            g=(4*a*f-a*a*f*f)/8
            h=(2*f-3*a*f*f+a*a*f**3)/8
            X=np.column_stack([g-.5,h]);A=X.T@(weight[:,None]*X);b=X.T@(weight*target)
            unconstrained=np.linalg.solve(A,b)
            candidates=[]
            d,k=unconstrained
            if .25<=d<=1.25 and 0<=k<=pmax*d:candidates.append(unconstrained)
            for vertex,next_vertex in zip(vertices,np.roll(vertices,-1,axis=0)):
                direction=next_vertex-vertex
                step=np.clip(direction@(b-A@vertex)/(direction@A@direction),0,1)
                candidates.append(vertex+step*direction)
            values=[float(np.sum(weight*(target-X@v)**2)) for v in candidates]
            ix=int(np.argmin(values));d,k=candidates[ix]
            return (values[ix],np.array([a,k/d,2*(1-d)])) if full else values[ix]
        grid=np.linspace(1e-8,pmax,129);values=np.array([profile(a) for a in grid])
        candidates=[grid[0],grid[-1]]
        for lo,hi in [(grid[0],grid[1]),(grid[-2],grid[-1])]:
            candidates.append(minimize_scalar(profile,bounds=(lo,hi),method='bounded',options={'xatol':1e-12}).x)
        for i in range(1,len(grid)-1):
            if values[i]<=values[i-1] and values[i]<=values[i+1]:
                result=minimize_scalar(profile,bounds=(grid[i-1],grid[i+1]),method='bounded',options={'xatol':1e-12})
                candidates.append(result.x)
        best=min(candidates,key=profile);loss,pars=profile(best,True)
        parameters.append(pars);losses.append(loss)
    pars=np.array(parameters)
    jac=np.array([unequal_jacobian(f,*v) for v in pars]);cov=covariance(jac,var)
    return dict(estimates=pars[:,2]/4,standard_errors=np.sqrt(cov[:,2,2])/4,
                parameters=pars,covariance=cov,weighted_loss=np.array(losses))
