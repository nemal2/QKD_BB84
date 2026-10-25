"""Independent invariants: matrix nesting, OSD reference, public hashes and circuits."""
import json
from pathlib import Path
import sys
import numpy as np
from reconciliation_v2 import OUT,build_library,make_frames,osd_one,hash_delta,qk,COUNTS


def row_integers(H):return [int.from_bytes(np.packbits(r,bitorder='little').tobytes(),'little') for r in H]


def basis_of(rows):
    basis={}
    for value in rows:
        while value:
            j=value.bit_length()-1
            if j in basis:value^=basis[j]
            else:basis[j]=value;break
    return basis


def contains(basis,value):
    while value:
        j=value.bit_length()-1
        if j not in basis:return False
        value^=basis[j]
    return True


def osd_reference(H,syn,posterior,prior):
    m,n=H.shape;hard=(posterior<0).astype(np.uint8)
    A=np.column_stack([H.copy(),(syn+H@hard)%2]);pivots=[];rank=0
    for c in np.argsort(abs(posterior)):
        ix=np.flatnonzero(A[rank:,c])
        if not len(ix):continue
        j=rank+ix[0];A[[rank,j]]=A[[j,rank]]
        for j in range(m):
            if j!=rank and A[j,c]:A[j]^=A[rank]
        pivots.append(c);rank+=1
        if rank==m:break
    base=hard.copy();base[pivots]^=A[:rank,-1];candidates=[base]
    for c in sorted(set(range(n))-set(pivots)):
        b=base.copy();b[c]^=1;b[pivots]^=A[:rank,c];candidates.append(b)
    llr=np.log((1-prior)/prior);scores=np.array([llr@b for b in candidates])
    return candidates[int(np.argmin(scores))],rank


def main():
    rng=np.random.default_rng(12815);audit=[]
    for seed in [11,29,47]:
        codes=build_library(seed);old=None
        for m,H in codes.items():
            rows=row_integers(H);basis=basis_of(rows)
            assert np.all(H.sum(0)==3) and H.sum(1).min()>=2
            if old is not None:assert all(contains(basis,r) for r in old)
            old=rows;audit.append(dict(seed=seed,m=m,rank=len(basis),minimum_check_degree=int(H.sum(1).min()),maximum_check_degree=int(H.sum(1).max())))
    tests=0
    for n in [12,30,70]:
        for _ in range(30):
            m=n//2;H=rng.integers(0,2,(m,n),dtype=np.uint8);e=rng.integers(0,2,n,dtype=np.uint8)
            syn=(H@e)%2;post=rng.normal(size=n);prior=rng.uniform(.005,.4,n)
            a,ra=osd_one(H,syn,post,prior);b,rb=osd_reference(H,syn,post,prior)
            assert ra==rb and np.array_equal(a,b) and np.array_equal((H@a)%2,syn);tests+=1
    d=make_frames('A',.1,20,128155)
    for i in range(20):
        seed=d['hash_seeds'][i];T=np.array([[seed[999-j+k] for j in range(1000)] for k in range(64)],np.uint8)
        hashed=(T@d['block_errors'][i])%2
        expected=sum(int(x)<<k for k,x in enumerate(hashed))
        assert int(hash_delta(d['block_errors'][i],d['hash_columns'][i]))==expected
        er,sr,ed,sd=d['sample_counts'][i];raw=(er+ed)/200;pool=(er+ed+.5)/201
        for method,w in [(2,1.),(3,d['shrink_weights'][i])]:
            pr=pool+w*(er/sr-raw);pd=pool+w*(ed/sd-raw)
            assert abs((sr*pr+sd*pd)/200-pool)<1e-14
    # Validate the physical probabilities against actual density matrices, with
    # the additional isotropic link inserted independently of the noisy gates.
    from circuits_v2 import circuit,probabilities,export
    from diagnostics_v2 import qi
    from qiskit_aer.noise import depolarizing_error
    from qiskit_aer import AerSimulator
    from qiskit_aer.utils import insert_noise
    from bb84_noise import QuantumChannel,NoiseModelType
    circuit_checks=[]
    for scheme in COUNTS:
        for b in [0,1]:
            for bit in [0,1]:
                c=circuit(bit,b,b,scheme)
                for p in [.01,.03,.1]:
                    for channel in [0,.025,.1]:
                        actual=probabilities(c,p,channel)[1-bit]
                        expected=channel+(1-2*channel)*qk(p,int(COUNTS[scheme][b,bit]))
                        assert abs(actual-expected)<1e-11
                        circuit_checks.append(float(abs(actual-expected)))
    attacked=[]
    for p in [.01,.03,.075,.24]:
        value=0.
        for b in [0,1]:
            for x in [0,1]:
                for eb in [0,1]:
                    pe=probabilities(circuit(x,b,eb),p)
                    for ev in [0,1]:value+=pe[ev]*probabilities(circuit(ev,eb,b),p)[1-x]/8
        assert abs(value-qi(p))<1e-11;attacked.append(dict(p=p,exact=float(value),analytic=float(qi(p))))
    export()
    report=dict(status='passed',matrix_audit=audit,independent_OSD_reference_cases=tests,
                independent_Toeplitz_checks=20,coherent_prior_checks=40,
                density_matrix_checks=len(circuit_checks),maximum_probability_error=max(circuit_checks),two_hop_checks=attacked)
    (OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k!='matrix_audit'})


if __name__=='__main__':main()
