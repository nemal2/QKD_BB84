"""Sparse codes and paired channel simulation for the extension protocol."""
from pathlib import Path
import json,sys,time,hashlib
import numpy as np
from numba import njit,prange,set_num_threads,get_num_threads
HERE=Path(__file__).resolve().parent
OUT=HERE/'results';OUT.mkdir(exist_ok=True)
P=json.loads((HERE/'protocol.json').read_text())
sys.path.insert(0,str(HERE.parent/'study'))
from reconciliation_v2 import bp_layered,h2,build_library,graph
set_num_threads(min(6,get_num_threads()))
LAMBDAS=np.array([0,.25,.5,.75,1.])

# Edge-perspective variable coefficients from Table I of arXiv:0901.2140.
# Exponents are degree-1; normalize the printed rounded coefficients before
# converting edge fractions lambda_d into node fractions proportional to lambda_d/d.
PROFILES={
 .15:([2,3,4,5,6,7,9,21],[.04528,.20537,.05878,.094274,.08454,.01176,.05137,.50015]),
 .20:([2,3,6,7,8,17,26],[.09420,.18088,.11972,.08550,.09816,.07194,.34960]),
 .25:([2,3,4,5,6,7,28,31],[.10805,.09511,.01449,.13764,.10667,.05288,.01107,.47408]),
 .30:([2,3,6,9,12,25,46,62,65,73],[.05343,.29406,.00896,.15571,.12189,.19872,.09572,.02741,.04056,.00354]),
 .35:([2,3,4,5,9,13,15,21,51],[.10451,.15652,.08057,.00056,.12151,.10485,.10719,.00771,.31656]),
 .40:([2,3,8,9,26,27,46,71],[.11040,.20804,.14163,.14858,.14438,.08909,.00748,.15038]),
 .45:([2,3,6,15,16,18,19,31],[.16880,.20994,.18095,.03846,.02635,.23454,.05815,.08280]),
 .50:([2,3,4,5,6,8,9,10,11,14,15,17,47,49,55,56,57,58,59,66],[.14438,.19026,.01836,.00233,.04697,.053943,.05590,.01290,.00162,.06159,.13115,.01481,.00879,.00650,.00210,.00099,.11178,.06238,.05094,.02230])}


def seed(stage,*coordinates):
    # SeedSequence coordinates avoid arithmetic seed collisions across designs.
    return int(np.random.SeedSequence([P['seed_bases'][stage],*map(int,coordinates)]).generate_state(1)[0])


@njit(cache=True)
def peg_edges(degrees,m,rngseed):
    np.random.seed(rngseed);n=len(degrees);edges=int(degrees.sum())
    maxcheck=max(128,int(edges/m)+80);cv=np.full((m,maxcheck),-1,np.int64);load=np.zeros(m,np.int64)
    chosen=np.full(int(degrees.max()),-1,np.int64);mark=np.zeros(n,np.int64)
    row=np.empty(edges,np.int64);col=np.empty(edges,np.int64);t=0
    order=np.argsort(degrees)
    for pos in range(n):
        v=order[pos];degree=degrees[v]
        for e in range(degree):
            if e:
                cprev=chosen[e-1]
                for j in range(load[cprev]):mark[cv[cprev,j]]=pos+1
            best=-1;bestcycle=1000000;bestload=1000000
            for attempt in range(96):
                c=np.random.randint(m);duplicate=False
                for j in range(e):
                    if chosen[j]==c:duplicate=True
                if duplicate or load[c]>=maxcheck:continue
                cycles=0
                for j in range(load[c]):
                    if mark[cv[c,j]]==pos+1:cycles+=1
                if cycles<bestcycle or (cycles==bestcycle and load[c]<bestload):
                    best=c;bestcycle=cycles;bestload=load[c]
            if best<0:
                for c in range(m):
                    duplicate=False
                    for j in range(e):
                        if chosen[j]==c:duplicate=True
                    if not duplicate and load[c]<bestload:best=c;bestload=load[c]
            assert best>=0 and load[best]<maxcheck
            chosen[e]=best;cv[best,load[best]]=v;load[best]+=1;row[t]=best;col[t]=v;t+=1
    return row,col


def code(family,n,beta,matrix_seed):
    m=3*round(beta*n/3) if family=='gallager' else round(beta*n)
    path=OUT/'codes'/f'{family}_n{n}_m{m}_seed{matrix_seed}.npz';path.parent.mkdir(exist_ok=True)
    if path.exists():
        z=np.load(path);return z['ptr'],z['cols'],m,n,path.name
    rng=np.random.default_rng(np.random.SeedSequence([matrix_seed,n,m,0 if family=='gallager' else 1]))
    if family=='gallager':
        row=[];col=[];checks=m//3
        for layer in range(3):
            permutation=rng.permutation(n)
            row.extend((np.arange(n)*checks//n+layer*checks).tolist());col.extend(permutation.tolist())
        row=np.array(row);col=np.array(col);degrees=np.full(n,3)
    else:
        key=min(PROFILES,key=lambda b:abs(b-min(beta,.5)))
        deg=np.array(PROFILES[key][0]);edge=np.array(PROFILES[key][1]);node=edge/deg;node/=node.sum()
        counts=np.floor(node*n).astype(int);left=n-counts.sum()
        counts[np.argsort(node*n-counts)[-left:]]+=1 if left else 0
        degrees=np.repeat(deg,counts);rng.shuffle(degrees)
        row,col=peg_edges(degrees,m,int(rng.integers(0,2**31-1)))
    ix=np.lexsort((col,row));row=row[ix];col=col[ix]
    assert not np.any((row[1:]==row[:-1])&(col[1:]==col[:-1]))
    ptr=np.r_[0,np.cumsum(np.bincount(row,minlength=m))].astype(np.int64)
    assert np.min(np.diff(ptr))>=2
    np.savez_compressed(path,ptr=ptr,cols=col.astype(np.int64),n=n,m=m,column_degrees=degrees,
                        family=family,matrix_seed=matrix_seed,beta=beta)
    return ptr,col.astype(np.int64),m,n,path.name


def reference_code(q,matrix_seed=11,m=None):
    codes=build_library(matrix_seed)
    if m is None:m={.02:275,.06:500,.10:650}[q]
    H=codes[m];ptr,col,_,_=graph(H)
    return ptr,col,m,1000,f'nested_{matrix_seed}_{m}'


@njit(cache=True)
def toeplitz_hash(error,seedbits):
    n=len(error);v=np.uint64(0)
    for i in range(n):
        if error[i]:
            column=np.uint64(0)
            for k in range(64):
                if seedbits[n-1-i+k]:column|=np.uint64(1)<<np.uint64(k)
            v^=column
    return v


@njit(cache=True,parallel=True)
def batch(ptr,cols,m,n,errors,priors,hashseeds,sweeps=100):
    F=len(errors);correct=np.zeros(F,np.bool_);accepted=np.zeros(F,np.bool_);iters=np.zeros(F,np.int64)
    estimates=np.zeros_like(errors)
    for i in prange(F):
        syndrome=np.zeros(m,np.uint8)
        for r in range(m):
            for j in range(ptr[r],ptr[r+1]):syndrome[r]^=errors[i,cols[j]]
        eh,converged,it,post=bp_layered(ptr,cols,m,n,syndrome,priors[i],sweeps)
        estimates[i]=eh;correct[i]=np.all(eh==errors[i]);iters[i]=it
        accepted[i]=converged and toeplitz_hash(eh^errors[i],hashseeds[i])==0
    return correct,accepted,iters,estimates


def frame_data(q,a,s,n,F,rngseed,transitions=None):
    rng=np.random.default_rng(rngseed)
    bits=rng.integers(0,2,(F,n+s),dtype=np.uint8);bases=rng.integers(0,2,(F,n+s),dtype=np.uint8)
    if transitions is None:transitions=np.array([[q*(1-a)]*2,[q*(1+a)]*2])
    transitions=np.asarray(transitions)
    errors=(rng.random((F,n+s))<transitions[bases,bits]).astype(np.uint8)
    # IID positions allow the first S positions to be the random sacrificed set;
    # no decoded position is disclosed. This has the same law as random sampling.
    sample=errors[:,:s];sb=bases[:,:s];sx=bits[:,:s];sy=sx^sample
    counts=np.zeros((F,2,2,2),np.int64)
    for z in [0,1]:
        for x in [0,1]:
            for y in [0,1]:counts[:,z,x,y]=np.sum((sb==z)&(sx==x)&(sy==y),axis=1)
    ss=counts.sum((2,3));ee=counts[:,:,0,1]+counts[:,:,1,0]
    pool=(ee.sum(1)+.5)/(s+1);raw=ee.sum(1)/s
    rz=np.divide(ee,ss,out=np.tile(raw[:,None],(1,2)),where=ss>0)
    diff=rz[:,1]-rz[:,0];v=pool*(1-pool)*np.sum(1/np.maximum(ss,1),axis=1)
    w=np.maximum(0,1-v/np.maximum(diff*diff,1e-30));w[np.any(ss==0,axis=1)]=0
    seedbits=rng.integers(0,2,(F,n+63),dtype=np.uint8)
    return dict(bits=bits,bases=bases,errors=errors,counts=counts,pool=pool,raw=raw,rz=rz,
                diff=diff,variance=v,weight=w,hash_seeds=seedbits,s=s,n=n,q=q,a=a,
                transitions=transitions,seed=rngseed)


def selected_lambda(data,rule):
    i=np.searchsorted([100],data['s'],side='left')
    j=np.searchsorted([.04,.08],data['pool'],side='right')
    k=np.searchsorted([1,4,9],data['diff']**2/np.maximum(data['variance'],1e-30),side='right')
    return np.array(rule['lambda'])[i,j,k]


def priors(data,kind,rule=None):
    F,n=data['errors'].shape[0],data['n'];s=data['s'];bb=data['bases'][:,s:]
    pool=data['pool'][:,None];contrast=np.take_along_axis(data['rz']-data['raw'][:,None],bb,axis=1)
    if isinstance(kind,(float,int)):w=float(kind)
    elif kind=='pooled':w=0.
    elif kind=='separate':w=1.
    elif kind=='shrink':w=data['weight'][:,None]
    elif kind=='selected':w=selected_lambda(data,rule)[:,None]
    elif kind=='oracle_pool':return np.full((F,n),np.mean(data['transitions']))
    elif kind=='oracle_basis':return data['transitions'].mean(1)[bb]
    elif kind=='transition':
        c=data['counts'];nr=c.sum(3)
        theta01=(c[:,:,0,1]+.5)/(nr[:,:,0]+1);theta10=(c[:,:,1,0]+.5)/(nr[:,:,1]+1)
        bob=data['bits'][:,s:]^data['errors'][:,s:]
        p0=theta10/(1-theta01+theta10);p1=theta01/(1-theta10+theta01)
        return np.where(bob,np.take_along_axis(p1,bb,axis=1),np.take_along_axis(p0,bb,axis=1))
    else:raise ValueError(kind)
    return np.broadcast_to(pool+w*contrast,(F,n)).copy()


def evaluate(c,data,methods,rule=None,sweeps=100,retain=True):
    ptr,col,m,n,name=c;results=[];timings=[]
    for kind in methods:
        prior=priors(data,kind,rule);start=time.perf_counter()
        result=batch(ptr,col,m,n,data['errors'][:,data['s']:],prior,data['hash_seeds'],sweeps)
        timings.append(time.perf_counter()-start);results.append(result)
    return dict(correct=np.array([r[0] for r in results]),accepted=np.array([r[1] for r in results]),
                iterations=np.array([r[2] for r in results]),
                estimates=np.array([r[3] for r in results]) if retain else np.empty((0,),np.uint8),
                m=m,n=n,code=name),timings


def save_data(path,data):np.savez_compressed(path,**data)


def entropy_benchmark(t):
    t=np.asarray(t);q=t.mean(1)
    def conditional(row):return 1+.5*(h2(row[0])+h2(row[1]))-h2(.5*(row[0]+1-row[1]))
    hc=np.array([conditional(row) for row in t]);pooled=conditional(t.mean(0))
    return dict(qbar=float(t.mean()),basis_qber=q.tolist(),conditional_entropy=hc.tolist(),
                pooled_conditional_entropy=float(pooled),G=float(pooled-hc.mean()),
                BSC_Jensen_gap=float(h2(q.mean())-np.mean(h2(q))))
