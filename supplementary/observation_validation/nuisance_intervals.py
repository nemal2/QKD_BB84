"""Profile each device parameter separately on the retained four-setting counts."""
from pathlib import Path
import numpy as np
from numba import njit,prange
from likelihood import coefficients,objective,PMAX,CUTOFF


@njit(cache=True)
def fixed_devices(a,b,paths,groups,scales,counts):
    aa,bb=coefficients(a,0.,paths,groups,scales)
    q=aa+bb*b;slope=.25-.5*q
    low=objective(0.,q,slope,counts)
    if low[1]>=0:return low[0]
    high=objective(1.,q,slope,counts)
    if high[1]<=0:return high[0]
    lo=0.;hi=1.;r=.2
    for _ in range(60):
        value,g,h=objective(r,q,slope,counts)
        if g<0:lo=r
        else:hi=r
        rr=r-g/h if h>0 else (lo+hi)/2
        if rr<=lo or rr>=hi:rr=(lo+hi)/2
        if abs(rr-r)<1e-12:return objective(rr,q,slope,counts)[0]
        r=rr
    return objective(r,q,slope,counts)[0]


@njit(cache=True)
def free_device(t,x,coordinate,paths,groups,scales,counts):
    a,b=(x,t) if coordinate==0 else (t,x)
    return fixed_devices(a,b,paths,groups,scales,counts)


@njit(cache=True)
def profile_device(x,coordinate,paths,groups,scales,counts):
    grid=np.linspace(0.,PMAX,65);values=np.empty(65)
    for i in range(65):values[i]=free_device(grid[i],x,coordinate,paths,groups,scales,counts)
    best=np.min(values);ratio=(np.sqrt(5.)-1)/2
    for i in range(65):
        if i==0:lo=grid[0];hi=grid[1]
        elif i==64:lo=grid[-2];hi=grid[-1]
        elif values[i]<=values[i-1] and values[i]<=values[i+1]:lo=grid[i-1];hi=grid[i+1]
        else:continue
        c=hi-ratio*(hi-lo);d=lo+ratio*(hi-lo)
        fc=free_device(c,x,coordinate,paths,groups,scales,counts)
        fd=free_device(d,x,coordinate,paths,groups,scales,counts)
        for _ in range(38):
            if fc<fd:
                hi=d;d=c;fd=fc;c=hi-ratio*(hi-lo)
                fc=free_device(c,x,coordinate,paths,groups,scales,counts)
            else:
                lo=c;c=d;fc=fd;d=lo+ratio*(hi-lo)
                fd=free_device(d,x,coordinate,paths,groups,scales,counts)
        best=min(best,fc,fd)
    return best


@njit(cache=True,parallel=True)
def intervals(counts,parameters,nll,paths,groups,scales):
    result=np.empty((len(counts),2,2));grid=np.linspace(0.,PMAX,41)
    for i in prange(len(counts)):
        for coordinate in range(2):
            low=parameters[i,coordinate];high=low;limit=nll[i]+CUTOFF/2
            for x in grid:
                if profile_device(x,coordinate,paths,groups,scales,counts[i])<=limit:
                    low=min(low,x);high=max(high,x)
            if low>0:
                lo=max(0.,low-PMAX/40);hi=low
                for _ in range(28):
                    mid=(lo+hi)/2
                    if profile_device(mid,coordinate,paths,groups,scales,counts[i])>limit:lo=mid
                    else:hi=mid
                low=(lo+hi)/2
            if high<PMAX:
                lo=high;hi=min(PMAX,high+PMAX/40)
                for _ in range(28):
                    mid=(lo+hi)/2
                    if profile_device(mid,coordinate,paths,groups,scales,counts[i])>limit:hi=mid
                    else:lo=mid
                high=(lo+hi)/2
            result[i,coordinate,0]=low;result[i,coordinate,1]=high
    return result


def main():
    root=Path(__file__).resolve().parent/'results'
    for path in sorted(root.glob('retained_*_profile.npz')):
        out=path.with_name(path.stem.replace('_profile','_nuisance')+'.npz')
        if out.exists():continue
        with np.load(path) as z:
            result=intervals(z['counts'],z['parameters'],z['nll'],z['paths'],z['groups'],z['scales'])
            np.savez_compressed(out,intervals=result,scenario=z['scenario'])
        print(path.name,100*np.median(np.diff(result,axis=2),axis=0).ravel(),flush=True)


if __name__=='__main__':main()
