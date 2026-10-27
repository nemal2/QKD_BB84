"""Check methodological descriptions against immutable records and response laws."""
from pathlib import Path
import itertools
import json
import numpy as np
from scipy.optimize import root

HERE=Path(__file__).resolve().parent
SUP=HERE.parent


def analyse():
    """Reconstruct boundary, pooling, finite-design and path-identification checks.

    The checks use retained records and exact response laws. They generate no new
    observations and leave every retained fit intact; mismatches fail explicitly.
    """
    # Compare nuisance and target boundary flags with the stored fits and intervals.
    rows=[]
    for path in sorted((SUP/'observation_validation/results').glob('retained_*_profile.npz')):
        with np.load(path) as z:
            pars=z['parameters']; interval=z['intervals']
            low=pars[:,:2]<1e-6; high=pars[:,:2]>.24975-1e-6
            assert np.array_equal(z['boundary'],np.any(low|high,axis=1))
            rows.append(dict(scenario=str(z['scenario']),records=len(pars),
                lower_noise_hits=low.sum(0).tolist(),upper_noise_hits=high.sum(0).tolist(),
                target_estimate_boundary_hits=int(z['target_boundary'].sum()),
                target_interval_boundary_hits=int(np.any((interval<=1e-8)|(interval>=.25-1e-8),axis=1).sum()),
                r_range=[float(pars[:,2].min()),float(pars[:,2].max())],
                interval_endpoint_range=[float(interval.min()),float(interval.max())]))
    assert sum(r['records'] for r in rows)==2500
    assert all(r['upper_noise_hits']==[0,0] and r['target_estimate_boundary_hits']==0
               and r['target_interval_boundary_hits']==0 for r in rows)
    # Recover the pilot score using paired frame yields and the recorded bin inputs.
    folder=SUP/'extensions/results';bins={}
    for path in sorted(folder.glob('pilot_*_frames.npz')):
        with np.load(path) as d,np.load(path.with_name(path.name.replace('_frames','_evaluation'))) as z:
            i=int(int(d['s'])>100);jj=np.searchsorted([.04,.08],d['pool'],side='right')
            kk=np.searchsorted([1,4,9],d['diff']**2/np.maximum(d['variance'],1e-30),side='right')
            yields=z['correct'][:5].T.astype(float)*(1000-int(z['m'])-64)/(1000+int(d['s']))
            for j,k,y in zip(jj,kk,yields):bins.setdefault((i,int(j),int(k)),[]).append(y-y[0])
    rule=json.loads((folder/'frozen_rule.json').read_text());weights=[0.,.25,.5,.75,1.]
    for entry in rule['audit']:
        differences=np.asarray(bins.get(tuple(entry['bin']),[]));n=len(differences)
        score=np.zeros(5) if n<=1 else differences.mean(0)-1.96*differences.std(0,ddof=1)/np.sqrt(n)
        assert n==entry['frames']
        assert np.allclose(score,entry['lower_bounds'],rtol=0,atol=1e-14)
        assert weights[int(np.argmax(score))]==entry['chosen']
    # Reconstruct the finite Fisher search independently of its cached selection.
    folder=SUP/'diagnostic_validation'
    protocol=json.loads((folder/'protocol.json').read_text())
    retained=json.loads((folder/'results/design.json').read_text())['methods'];design_report=[]
    for cap in protocol['scaling_caps']:
        for settings in [2,4]:
            units=[v for v in itertools.product(range(1,20),repeat=settings-1) if sum(v)<20]
            w=np.asarray([(*v,20-sum(v)) for v in units])/20
            interior=[1+j*(cap-1)/6 for j in range(1,6)]
            designs=[[1.,cap]] if settings==2 else [[1.,*middle,cap] for middle in itertools.combinations(interior,2)]
            best=None
            for scales in designs:
                f=np.asarray(scales);worst=np.zeros(len(w))
                for p,r in itertools.product(protocol['nominal']['p'],protocol['nominal']['r']):
                    x=f*p;q0=(6*x-4*x*x+x*x*x)/8
                    q=r/4+(1-r/2)*q0
                    dp=(1-r/2)*(6-8*x+3*x*x)*f/8;dr=.25-q0/2
                    effective=4000*w/(q*(1-q))
                    A=effective@dp**2;B=effective@(dp*dr);C=effective@dr**2
                    worst=np.maximum(worst,A/(A*C-B*B)/16)
                ix=int(np.argmin(worst));candidate=(float(worst[ix]),scales,w[ix])
                if best is None or candidate[0]<best[0]-1e-15:best=candidate
            name=f'aggregate_{"two" if settings==2 else "four"}_cap{cap:g}'
            expected=retained[name]
            assert np.allclose(best[1],expected['scales'],rtol=0,atol=2e-15)
            assert np.array_equal(best[2],expected['weights'])
            assert np.isclose(best[0],expected['worst_variance'],rtol=1e-10,atol=1e-14)
            design_report.append(dict(method=name,location_sets=len(designs),allocations_per_set=len(w)))
    # Distinct monomials alone are insufficient: augmented exponent rank matters.
    assert np.linalg.matrix_rank([[1,0,0],[1,1,1],[1,2,2]])==2
    path_report=[]
    for name,paths,expected_rank,expected_target in [
        ('Standard',[(0,0),(1,0),(1,1),(2,1)],3,True),
        ('A',[(1,0),(2,0),(2,1),(3,1)],3,True),
        ('B',[(2,0),(2,0),(2,1),(2,1)],2,False),
        ('B+',[(2,1)]*4,1,False),
        ('Residual-only example',[(0,0),(1,0)],2,True),
        ('Distinct dependent monomials',[(0,0),(1,1),(2,2)],2,True)]:
        matrix=np.array([[1,*p] for p in paths],dtype=float)
        rank=int(np.linalg.matrix_rank(matrix))
        target=bool(np.linalg.matrix_rank(np.vstack([matrix,[1,0,0]]))==rank)
        assert (rank,target)==(expected_rank,expected_target)
        # The log-linear observation equals the directly multiplied contraction.
        eta=np.log([.9,.97,.95]);s=np.exp(matrix@eta)
        assert np.allclose(s,[.9*.97**a*.95**b for a,b in paths],rtol=0,atol=2e-16)
        if target:
            weights=np.linalg.lstsq(matrix.T,[1.,0.,0.],rcond=None)[0]
            assert abs(np.prod(s**weights)-.9)<2e-15
        else:
            _,singular,vh=np.linalg.svd(matrix,full_matrices=True)
            direction=next(v for v in vh[rank:] if abs(v[0])>1e-8)
            altered=eta+.001*direction
            assert np.allclose(np.exp(matrix@altered),s,rtol=0,atol=2e-15)
            assert abs(np.exp(altered[0])-.9)>1e-6
        path_report.append(dict(circuit=name,paths=[list(p) for p in paths],rank=rank,residual_identified=target))
    # Three positive settings also need not identify the restricted B+ model globally.
    f=np.array([1.,3.,4.]);truth=np.array([.03,.05,.2])
    contraction=lambda p:(1-p[2]/2)*(1-f*p[0])**2*(1-f*p[1])
    fitted=root(lambda p:contraction(p)-contraction(truth),[.045,.02,.2],tol=1e-11)
    other=fitted.x;error=float(np.max(np.abs(contraction(other)-contraction(truth))))
    assert fitted.success and error<1e-12
    assert np.all(other[:2]>0) and np.all(other[:2]<.25) and 0<other[2]<1
    assert abs(other[2]-truth[2])>1e-5
    return dict(status='passed',retained_boundary_records=rows,pooling_bins_reconstructed=24,
        pooling_frames_reconstructed=sum(len(v) for v in bins.values()),matched_range_designs=design_report,
        rank_condition_checked=True,path_identification=path_report,three_setting_counterexample=dict(scales=f.tolist(),
            parameters=truth.tolist(),alternative_parameters=other.tolist(),
            maximum_contraction_discrepancy=error,residuals=[float(truth[2]/4),float(other[2]/4)]),
        scope='Retained-record and algebraic checks; no new stochastic experiments or relaxed-bound fitting.')


if __name__=='__main__':
    result=analyse();(HERE/'method-checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
