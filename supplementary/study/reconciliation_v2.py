"""Controlled code/estimator evaluation for the internal research protocol.

Every code has column weight three. Splitting checks makes the code family
nested: each old parity equation is the XOR of its replacement equations.
Layered BP and ordered-statistics postprocessing use identical budgets for
pooled and class-dependent priors. Public verification triggers OSD retries.
"""
from __future__ import annotations
import json
import hashlib
import math
from pathlib import Path
import time
import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads
from scipy.optimize import brentq

HERE=Path(__file__).resolve().parent
OUT=HERE/'results';OUT.mkdir(exist_ok=True)
P=json.loads((HERE/'protocol.json').read_text())
COUNTS={'PRIMARY':np.array([[0,1],[2,3]]),'A':np.array([[1,2],[3,4]]),
        'B':np.array([[2,2],[3,3]]),'B+':np.array([[3,3],[3,3]])}


def guard():
    digest={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
            for name in ['protocol.json','reconciliation_v2.py','diagnostics_v2.py']}
    path=OUT/'input_guard.json'
    if path.exists() and json.loads(path.read_text())!=digest:
        raise RuntimeError('Protocol or simulation changed: use a fresh result directory.')
    path.write_text(json.dumps(digest,indent=2)+'\n')


def h2(p):
    p=np.clip(p,1e-15,1-1e-15)
    return -p*np.log2(p)-(1-p)*np.log2(1-p)


def qk(p,k):return (1-(1-p)**k)/2


def physical_parameter(scheme,total,channel=0):
    gate=(total-channel)/(1-2*channel)
    if abs(gate)<1e-14:return 0.
    return brentq(lambda x:np.mean(qk(x,COUNTS[scheme]))-gate,0,1)


def peg_base(n=1000,m=150,seed=11):
    """PEG with degree balancing and breadth-first cycle avoidance."""
    rng=np.random.default_rng(seed);vc=[set() for _ in range(n)];cv=[set() for _ in range(m)]
    cap=math.ceil(3*n/m)+1
    for v in rng.permutation(n):
        for edge in range(3):
            allowed=[c for c in range(m) if c not in vc[v] and len(cv[c])<cap]
            if not allowed:allowed=[c for c in range(m) if c not in vc[v]]
            if edge:
                seen_v={int(v)};seen_c=set();front={int(v)};distance=np.full(m,n+1,int);depth=0
                while front:
                    newc=set().union(*(vc[x] for x in front))-seen_c
                    if not newc:break
                    for c in newc:distance[c]=depth
                    seen_c|=newc
                    newv=set().union(*(cv[c] for c in newc))-seen_v
                    seen_v|=newv;front=newv;depth+=1
                far=max(distance[c] for c in allowed)
                allowed=[c for c in allowed if distance[c]==far]
            degree=min(len(cv[c]) for c in allowed)
            candidates=[c for c in allowed if len(cv[c])==degree]
            c=int(rng.choice(candidates));vc[v].add(c);cv[c].add(int(v))
    assert all(len(x)==3 for x in vc) and min(map(len,cv))>=2
    return cv


def build_library(seed=11):
    path=OUT/f'codes_seed{seed}.npz'
    if path.exists():return {int(k[1:]):v for k,v in np.load(path).items()}
    rows=peg_base(seed=seed);rng=np.random.default_rng(seed+801)
    codes={}
    for target in range(150,901,25):
        while len(rows)<target:
            j=max(range(len(rows)),key=lambda i:len(rows[i]))
            assert len(rows[j])>=4
            values=rng.permutation(sorted(rows[j]));cut=len(values)//2
            rows[j]=set(values[:cut].tolist());rows.append(set(values[cut:].tolist()))
        H=np.zeros((target,1000),np.uint8)
        for i,row in enumerate(rows):H[i,list(row)]=1
        assert np.all(H.sum(axis=0)==3) and np.min(H.sum(axis=1))>=2
        codes[target]=H
    np.savez_compressed(path,**{f'm{k}':v for k,v in codes.items()})
    return codes


def graph(H):
    rows,cols=np.nonzero(H)
    ptr=np.r_[0,np.cumsum(H.sum(axis=1))].astype(np.int64)
    return ptr,cols.astype(np.int64),H.shape[0],H.shape[1]


@njit(cache=True)
def bp_layered(ptr,cols,m,n,syndrome,prior,max_iter=300):
    pp=np.minimum(.499,np.maximum(1e-6,prior))
    llr=np.log((1-pp)/pp);post=llr.copy();messages=np.zeros(len(cols))
    eh=np.zeros(n,np.uint8)
    for it in range(1,max_iter+1):
        for row in range(m):
            lo,hi=ptr[row],ptr[row+1];length=hi-lo
            ts=np.empty(length);ext=np.empty(length);prod=1.
            for j in range(length):
                e=lo+j;ext[j]=post[cols[e]]-messages[e]
                val=np.tanh(min(19.,max(-19.,ext[j]/2)))
                if abs(val)<1e-12:val=1e-12 if val+1e-30>=0 else -1e-12
                ts[j]=val;prod*=val
            sign=-1. if syndrome[row] else 1.
            for j in range(length):
                e=lo+j;val=min(.999999999,max(-.999999999,prod/ts[j]))
                messages[e]=sign*2*np.arctanh(val)
                post[cols[e]]=ext[j]+messages[e]
        for i in range(n):eh[i]=post[i]<0
        match=True
        for row in range(m):
            value=0
            for e in range(ptr[row],ptr[row+1]):value^=eh[cols[e]]
            if value!=syndrome[row]:match=False;break
        if match:return eh,True,it,post
    return eh,False,max_iter,post


@njit(cache=True)
def osd_one(H,syndrome,posterior,prior):
    """Full order-one OSD in a least-reliable parity basis, channel-LLR score."""
    m,n=H.shape;words=(n+1+63)//64
    A=np.zeros((m,words),np.uint64)
    hard=(posterior<0).astype(np.uint8)
    for row in range(m):
        residual=syndrome[row]
        for col in range(n):
            if H[row,col]:
                A[row,col//64]|=np.uint64(1)<<np.uint64(col%64)
                residual^=hard[col]
        if residual:A[row,n//64]|=np.uint64(1)<<np.uint64(n%64)
    pivots=np.empty(m,np.int64);is_pivot=np.zeros(n,np.bool_);rank=0
    order=np.argsort(np.abs(posterior))
    for col in order:
        pivot=-1
        for row in range(rank,m):
            if (A[row,col//64]>>np.uint64(col%64))&np.uint64(1):pivot=row;break
        if pivot<0:continue
        temp=A[rank].copy();A[rank]=A[pivot];A[pivot]=temp
        for row in range(m):
            if row!=rank and ((A[row,col//64]>>np.uint64(col%64))&np.uint64(1)):
                for w in range(words):A[row,w]^=A[rank,w]
        pivots[rank]=col;is_pivot[col]=True;rank+=1
        if rank==m:break
    base=hard.copy()
    for row in range(rank):
        if (A[row,n//64]>>np.uint64(n%64))&np.uint64(1):base[pivots[row]]^=1
    pp=np.minimum(.499,np.maximum(1e-6,prior));llr=np.log((1-pp)/pp)
    score=0.
    for col in range(n):score+=llr[col]*base[col]
    best=base.copy();best_score=score
    for col in range(n):
        if is_pivot[col]:continue
        candidate_score=score+llr[col]*(1-2*int(base[col]))
        for row in range(rank):
            if (A[row,col//64]>>np.uint64(col%64))&np.uint64(1):
                j=pivots[row];candidate_score+=llr[j]*(1-2*int(base[j]))
        if candidate_score<best_score-1e-12:
            best_score=candidate_score;best=base.copy();best[col]^=1
            for row in range(rank):
                if (A[row,col//64]>>np.uint64(col%64))&np.uint64(1):best[pivots[row]]^=1
    return best,rank


@njit(cache=True)
def hash_delta(error,columns):
    value=np.uint64(0)
    for j in range(len(error)):
        if error[j]:value^=columns[j]
    return value


@njit(cache=True,parallel=True)
def decode_batch(H,ptr,cols,errors,priors,hash_columns,strong=False,max_iter=300):
    frames,n=errors.shape;m=H.shape[0]
    correct=np.zeros(frames,np.bool_);bp_correct=np.zeros(frames,np.bool_)
    iterations=np.zeros(frames,np.int64);retried=np.zeros(frames,np.bool_)
    estimates=np.zeros_like(errors);collision=np.zeros(frames,np.bool_)
    for i in prange(frames):
        syndrome=np.zeros(m,np.uint8)
        for row in range(m):
            for e in range(ptr[row],ptr[row+1]):syndrome[row]^=errors[i,cols[e]]
        eh,conv,it,post=bp_layered(ptr,cols,m,n,syndrome,priors[i],max_iter)
        bp_correct[i]=np.all(eh==errors[i]);iterations[i]=it
        # This is exactly Alice/Bob's public Toeplitz verification difference.
        # Exact correctness is never used to decide whether to invoke OSD.
        mismatch=hash_delta(eh^errors[i],hash_columns[i])!=0
        if strong and (not conv or mismatch):
            eh,rank=osd_one(H,syndrome,post,priors[i]);retried[i]=True
        estimates[i]=eh;correct[i]=np.all(eh==errors[i])
        collision[i]=(hash_delta(eh^errors[i],hash_columns[i])==0) and not correct[i]
    return correct,bp_correct,iterations,retried,estimates,collision


def make_frames(scheme,q,frames,seed,channel=0):
    rng=np.random.default_rng(seed);p=physical_parameter(scheme,q,channel)
    bits=rng.integers(0,2,(frames,1200),dtype=np.uint8)
    bases=rng.integers(0,2,(frames,1200),dtype=np.uint8)
    probabilities=channel+(1-2*channel)*qk(p,COUNTS[scheme][bases,bits])
    errors=(rng.random((frames,1200))<probabilities).astype(np.uint8)
    samples=np.empty((frames,200),np.int64);kept=np.empty((frames,1000),np.int64)
    priors=np.empty((4,frames,1000));weights=np.empty(frames)
    counts=np.empty((frames,4),np.int64);hash_seeds=rng.integers(0,2,(frames,1063),dtype=np.uint8)
    hashes=np.empty((frames,1000),np.uint64)
    for i in range(frames):
        samples[i]=np.sort(rng.choice(1200,200,replace=False));mask=np.ones(1200,bool);mask[samples[i]]=False
        kept[i]=np.flatnonzero(mask);se=errors[i,samples[i]];sb=bases[i,samples[i]]
        er=int(se[sb==0].sum());sr=int((sb==0).sum());ed=int(se[sb==1].sum());sd=200-sr
        pool=(er+ed+.5)/201;raw=(er+ed)/200
        rr=er/sr if sr else raw;rd=ed/sd if sd else raw;d=rd-rr
        variance=pool*(1-pool)*(1/sr+1/sd) if sr and sd else 1.
        w=max(0.,1-variance/(d*d)) if d else 0.;weights[i]=w
        bb=bases[i,mask]
        priors[0,i]=pool
        priors[1,i]=np.where(bb==0,(er+.5)/(sr+1),(ed+.5)/(sd+1))
        priors[2,i]=pool+np.where(bb==0,rr-raw,rd-raw)
        priors[3,i]=pool+w*np.where(bb==0,rr-raw,rd-raw)
        counts[i]=[er,sr,ed,sd]
        value=sum(int(hash_seeds[i,999+j])<<j for j in range(64));hashes[i,0]=value
        for j in range(1,1000):
            value=((value<<1)&((1<<64)-1))|int(hash_seeds[i,999-j]);hashes[i,j]=value
    block_errors=np.take_along_axis(errors,kept,axis=1)
    return dict(bits=bits,bases=bases,errors=errors,sample_indices=samples,kept_indices=kept,
                sample_counts=counts,shrink_weights=weights,priors=priors,
                hash_seeds=hash_seeds,hash_columns=hashes,block_errors=block_errors,
                physical_p=p,channel_qber=channel,target_qber=q,seed=seed,scheme=scheme)


def save_frames(path,data):
    # Hash columns and decoded-block errors are deterministic from retained
    # Toeplitz seeds / frame arrays. Avoid duplicating dense derived arrays.
    np.savez_compressed(path,**{k:v for k,v in data.items() if k not in ['hash_columns','block_errors']})


def wilson_upper(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    return (p+z*z/(2*n)+z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/den


def evaluate(H,data,which=(0,3),strong=False,patterns=False):
    ptr,cols,m,n=graph(H);out=[]
    for method in which:
        r=decode_batch(H,ptr,cols,data['block_errors'],data['priors'][method],data['hash_columns'],strong)
        out.append(dict(correct=r[0],bp_correct=r[1],iterations=r[2],retried=r[3],
                        estimated_errors=r[4] if patterns else np.empty((0,0),np.uint8),collisions=r[5]))
    return out


def calibration_cell(scheme,q,seed,codes,tag,channel=0):
    selection_path=OUT/f'{tag}_selection.json'
    if selection_path.exists():return json.loads(selection_path.read_text())
    data=make_frames(scheme,q,P['calibration_frames'],seed,channel)
    save_frames(OUT/f'{tag}_frames.npz',data)
    key=min(P['calibration_grid'],key=lambda x:abs(float(x)-q))
    grid=[round(r*1000) for r in P['calibration_grid'][key]]
    if q not in P['qber_targets']:
        grid=list(range(max(150,int(1000*h2(q)*1.02)//25*25),901,25))
    results={};start=time.perf_counter()
    for m in grid:
        res=evaluate(codes[m],data)
        results[m]=res
    for j in range(2):
        if not any(wilson_upper(int((~results[m][j]['correct']).sum()),len(data['bits']))<=.01 for m in grid):
            for m in range(max(grid)+25,901,25):
                if m not in results:results[m]=evaluate(codes[m],data);grid.append(m)
    selected=[]
    for j in range(2):
        eligible=[m for m in sorted(grid) if all(wilson_upper(int((~results[k][j]['correct']).sum()),len(data['bits']))<=.01 for k in grid if k>=m)]
        selected.append(min(eligible) if eligible else None)
    arrays={'m_values':np.array(sorted(grid)),
            'correct':np.array([[results[m][j]['correct'] for j in range(2)] for m in sorted(grid)]),
            'iterations':np.array([[results[m][j]['iterations'] for j in range(2)] for m in sorted(grid)])}
    np.savez_compressed(OUT/f'{tag}_calibration.npz',**arrays)
    summary=dict(scheme=scheme,qber=q,channel_qber=channel,selected_m=selected,
                 failures={str(m):[int((~results[m][j]['correct']).sum()) for j in range(2)] for m in sorted(grid)})
    selection_path.write_text(json.dumps(summary,indent=2)+'\n')
    print(tag,'selected',selected,f'{time.perf_counter()-start:.1f}s',flush=True)
    return summary


def holdout_cell(scheme,q,seed,codes,selection,tag,channel=0,frames=2000):
    path=OUT/f'{tag}_evaluation.npz'
    if path.exists():return
    data=make_frames(scheme,q,frames,seed,channel);save_frames(OUT/f'{tag}_frames.npz',data)
    choices=sorted({int(m) for m in selection['selected_m'] if m is not None})
    if not choices:raise RuntimeError(f'No code met the calibration target for {tag}; do not retune on holdout.')
    correct=[];bp_correct=[];iterations=[];retried=[];patterns=[];collisions=[]
    for m in choices:
        res=evaluate(codes[m],data,which=(0,1,2,3),strong=True,patterns=True)
        correct.append([r['correct'] for r in res]);bp_correct.append([r['bp_correct'] for r in res])
        iterations.append([r['iterations'] for r in res]);retried.append([r['retried'] for r in res])
        patterns.append([r['estimated_errors'] for r in res]);collisions.append([r['collisions'] for r in res])
    np.savez_compressed(path,m_values=choices,correct=correct,bp_correct=bp_correct,iterations=iterations,
                        retried=retried,estimated_errors=patterns,collisions=collisions)
    print(tag,'BP fails',np.sum(~np.array(bp_correct),axis=2).tolist(),
          'BP+OSD fails',np.sum(~np.array(correct),axis=2).tolist(),flush=True)


def main():
    guard()
    set_num_threads(min(6,get_num_threads()));codes=build_library(P['matrix_seed'])
    selections={}
    for si,scheme in enumerate(P['schemes']):
        for qi,q in enumerate(P['qber_targets']):
            tag=f'cal_{scheme.replace("+","plus")}_{int(q*100):02d}'
            selection=calibration_cell(scheme,q,P['seed_bases']['calibration']+si*10000+qi*1000,codes,tag)
            selections[(scheme,q)]=selection
    (OUT/'selections.json').write_text(json.dumps({f'{s}_{q}':v for (s,q),v in selections.items()},indent=2)+'\n')
    # All operating choices are saved before opening the independent holdout.
    for si,scheme in enumerate(P['schemes']):
        for qi,q in enumerate(P['qber_targets']):
            holdout_cell(scheme,q,P['seed_bases']['holdout']+si*10000+qi*1000,codes,selections[(scheme,q)],
                         f'main_{scheme.replace("+","plus")}_{int(q*100):02d}')
    for si,scheme in enumerate(P['curves']['schemes']):
        for qi,q in enumerate(P['curves']['qber_grid']):
            tag=f'curve_{scheme.replace("+","plus")}_{int(round(q*100)):02d}'
            if (OUT/f'{tag}.npz').exists():continue
            data=make_frames(scheme,q,P['curves']['frames'],P['seed_bases']['curves']+si*10000+qi*100)
            save_frames(OUT/f'{tag}_frames.npz',data)
            correct=[]
            for rate in P['curves']['row_rates']:
                correct.append([r['correct'] for r in evaluate(codes[round(1000*rate)],data)])
            np.savez_compressed(OUT/f'{tag}.npz',correct=correct,row_rates=P['curves']['row_rates'])
            print(tag,flush=True)
    for si,scheme in enumerate(P['channel_control']['schemes']):
        for ci,channel in enumerate(P['channel_control']['channel_qber']):
            q=.10;tag=f'chcal_{scheme}_{int(round(channel*1000)):03d}'
            selection=calibration_cell(scheme,q,P['seed_bases']['channel_calibration']+si*10000+ci*1000,codes,tag,channel)
            holdout_cell(scheme,q,P['seed_bases']['channel_holdout']+si*10000+ci*1000,codes,selection,
                         f'channel_{scheme}_{int(round(channel*1000)):03d}',channel)
    for code_seed in P['replication_matrix_seeds']:
        library=build_library(code_seed)
        for si,scheme in enumerate(['PRIMARY','A','B+']):
            holdout_cell(scheme,.10,P['seed_bases']['replication']+code_seed*10000+si*1000,
                         library,selections[(scheme,.10)],f'rep_{code_seed}_{scheme.replace("+","plus")}',frames=1000)


if __name__=='__main__':main()
