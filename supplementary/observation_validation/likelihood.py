"""Binomial observation models and nuisance-profile likelihood intervals.

The four path exponents are supplied as data, not inferred from outcomes.
Bob has at most one gate per path, making his fixed-target likelihood convex.
"""
import numpy as np
from numba import njit, prange

PMAX = .24975
CUTOFF = 3.841458820694124


# Path probabilities are affine in Bob noise when Alice noise and interception are fixed.
@njit(cache=True)
def coefficients(a, r, paths, groups, scales):
    aa = np.empty(len(scales)); bb = np.empty(len(scales))
    for j in range(len(scales)):
        u = 1-scales[j]*a
        total = 0.; receiver = 0.
        for k in range(4):
            term = groups[j,k]*u**paths[k,0]
            total += term
            receiver += term*paths[k,1]
        aa[j] = .5-.5*(1-r/2)*total
        bb[j] = .5*(1-r/2)*scales[j]*receiver
    return aa, bb


@njit(cache=True)
def objective(b, aa, bb, counts):
    value = 0.; gradient = 0.; hessian = 0.
    for j in range(len(aa)):
        q = min(1-1e-14,max(1e-14,aa[j]+bb[j]*b))
        e,n = counts[j,0],counts[j,1]
        value -= e*np.log(q)+(n-e)*np.log1p(-q)
        gradient += bb[j]*((n-e)/(1-q)-e/q)
        hessian += bb[j]**2*((n-e)/(1-q)**2+e/q**2)
    return value,gradient,hessian


# Minimize the convex one-dimensional Bob-noise likelihood inside the physical box.
@njit(cache=True)
def profile_b(a,r,paths,groups,scales,counts):
    aa,bb=coefficients(a,r,paths,groups,scales)
    low=objective(0.,aa,bb,counts)
    if low[1]>=0:return low[0],0.
    high=objective(PMAX,aa,bb,counts)
    if high[1]<=0:return high[0],PMAX
    lo=0.;hi=PMAX;b=PMAX/2
    for _ in range(60):
        value,g,h=objective(b,aa,bb,counts)
        if g<0:lo=b
        else:hi=b
        next_b=b-g/h if h>0 else (lo+hi)/2
        if next_b<=lo or next_b>=hi:next_b=(lo+hi)/2
        if abs(next_b-b)<1e-12:return objective(next_b,aa,bb,counts)[0],next_b
        b=next_b
    return objective(b,aa,bb,counts)[0],b


@njit(cache=True)
def fixed_a(a,r,paths,groups,scales,counts,equal):
    if equal:
        aa,bb=coefficients(a,r,paths,groups,scales)
        return objective(a,aa,bb,counts)[0],a
    return profile_b(a,r,paths,groups,scales,counts)


# Refine each bracketed Alice-noise candidate, including boundary intervals.
@njit(cache=True)
def refine_a(lo,hi,r,paths,groups,scales,counts,equal):
    ratio=(np.sqrt(5.)-1)/2
    c=hi-ratio*(hi-lo);d=lo+ratio*(hi-lo)
    fc=fixed_a(c,r,paths,groups,scales,counts,equal)[0]
    fd=fixed_a(d,r,paths,groups,scales,counts,equal)[0]
    for _ in range(38):
        if fc<fd:
            hi=d;d=c;fd=fc;c=hi-ratio*(hi-lo)
            fc=fixed_a(c,r,paths,groups,scales,counts,equal)[0]
        else:
            lo=c;c=d;fc=fd;d=lo+ratio*(hi-lo)
            fd=fixed_a(d,r,paths,groups,scales,counts,equal)[0]
    a=c if fc<fd else d
    value,b=fixed_a(a,r,paths,groups,scales,counts,equal)
    return value,a,b


# Profile both nuisance parameters for a fixed interception fraction.
@njit(cache=True)
def profile(r,paths,groups,scales,counts,equal=False,grid_size=65):
    grid=np.linspace(0.,PMAX,grid_size)
    values=np.empty(grid_size);bs=np.empty(grid_size)
    for i in range(grid_size):values[i],bs[i]=fixed_a(grid[i],r,paths,groups,scales,counts,equal)
    ix=np.argmin(values);best=(values[ix],grid[ix],bs[ix])
    for i in range(grid_size):
        if i==0:lo=grid[0];hi=grid[1]
        elif i==grid_size-1:lo=grid[-2];hi=grid[-1]
        elif values[i]<=values[i-1] and values[i]<=values[i+1]:lo=grid[i-1];hi=grid[i+1]
        else:continue
        candidate=refine_a(lo,hi,r,paths,groups,scales,counts,equal)
        if candidate[0]<best[0]:best=candidate
    return best


@njit(cache=True)
def refine_r(lo,hi,paths,groups,scales,counts,equal):
    ratio=(np.sqrt(5.)-1)/2
    c=hi-ratio*(hi-lo);d=lo+ratio*(hi-lo)
    fc=profile(c,paths,groups,scales,counts,equal)[0]
    fd=profile(d,paths,groups,scales,counts,equal)[0]
    for _ in range(30):
        if fc<fd:
            hi=d;d=c;fd=fc;c=hi-ratio*(hi-lo)
            fc=profile(c,paths,groups,scales,counts,equal)[0]
        else:
            lo=c;c=d;fc=fd;d=lo+ratio*(hi-lo)
            fd=profile(d,paths,groups,scales,counts,equal)[0]
    return c if fc<fd else d


# Find the joint optimum, then solve the likelihood-ratio endpoints for the target.
@njit(cache=True)
def fit_one(counts,paths,groups,scales,equal=False):
    grid=np.linspace(0.,1.,41)
    values=np.empty(41)
    for i in range(41):values[i]=profile(grid[i],paths,groups,scales,counts,equal)[0]
    ix=np.argmin(values);r=grid[ix];best=values[ix]
    for i in range(41):
        if i==0:lo=grid[0];hi=grid[1]
        elif i==40:lo=grid[-2];hi=grid[-1]
        elif values[i]<=values[i-1] and values[i]<=values[i+1] and (values[i]<values[i-1]-1e-9 or values[i]<values[i+1]-1e-9):lo=grid[i-1];hi=grid[i+1]
        else:continue
        rr=refine_r(lo,hi,paths,groups,scales,counts,equal)
        v=profile(rr,paths,groups,scales,counts,equal)[0]
        if v<best:best=v;r=rr
    _,a,b=profile(r,paths,groups,scales,counts,equal)
    # Report the hull if a likelihood set has disconnected components.
    accepted=values<=best+CUTOFF/2
    low=r;high=r
    for i in range(41):
        if accepted[i]:low=min(low,grid[i]);high=max(high,grid[i])
    if low>0:
        lo=max(0.,low-.025);hi=low
        for _ in range(28):
            mid=(lo+hi)/2;v=profile(mid,paths,groups,scales,counts,equal)[0]
            if v>best+CUTOFF/2:lo=mid
            else:hi=mid
        low=(lo+hi)/2
    if high<1:
        lo=high;hi=min(1.,high+.025)
        for _ in range(28):
            mid=(lo+hi)/2;v=profile(mid,paths,groups,scales,counts,equal)[0]
            if v>best+CUTOFF/2:hi=mid
            else:lo=mid
        high=(lo+hi)/2
    return np.array([a,b,r,best,low/4,high/4])


# Fit replicate blocks independently; parallelism does not change their random streams.
@njit(cache=True,parallel=True)
def fit_batch(counts,paths,groups,scales,equal=False):
    result=np.empty((len(counts),6))
    for i in prange(len(counts)):result[i]=fit_one(counts[i],paths,groups,scales,equal)
    return result


# Evaluate model probabilities and local information for independent diagnostics.
def probabilities(parameters,paths,groups,scales):
    a,b,r=parameters
    aa,bb=coefficients(a,r,paths,groups,scales)
    return aa+bb*b


def information(parameters,paths,groups,scales,counts,equal=False):
    a,b,r=parameters;f=np.asarray(scales)
    u=1-f[:,None]*a;v=1-f[:,None]*b
    k=paths[:,0][None,:];l=paths[:,1][None,:]
    terms=u**k*v**l
    jac=np.column_stack([.5*(1-r/2)*np.sum(groups*k*f[:,None]*u**np.maximum(k-1,0)*v**l,axis=1),
                         .5*(1-r/2)*np.sum(groups*l*f[:,None]*u**k,axis=1),
                         .25*np.sum(groups*terms,axis=1)])
    q=np.clip(probabilities(parameters,paths,groups,scales),1e-12,1-1e-12)
    design=np.column_stack([jac[:,0]+jac[:,1],jac[:,2]]) if equal else jac
    matrix=design.T@((counts[:,1]/(q*(1-q)))[:,None]*design)
    vals,vecs=np.linalg.eigh(matrix);keep=vals>vals[-1]*1e-12
    inverse=(vecs[:,keep]/vals[keep])@vecs[:,keep].T
    projection=vecs[:,keep]@vecs[:,keep].T
    se=np.sqrt(np.diag(inverse))
    for j in range(len(se)):
        if np.linalg.norm(projection[:,j]-np.eye(len(se))[:,j])>1e-6:se[j]=np.inf
    if equal:se=np.array([se[0],se[0],se[1]])
    condition=vals[-1]/vals[0] if vals[0]>0 else np.inf
    return condition,int(keep.sum()),se,jac
