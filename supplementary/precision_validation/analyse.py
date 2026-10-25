"""Report all cells and Monte Carlo uncertainty for independent MAE contrasts."""
from pathlib import Path
import csv,json
import numpy as np
from scipy.stats import t
from experiment import P,cells

HERE=Path(__file__).resolve().parent;OUT=HERE/'results'

def wilson(k,n):
    z=1.959963984540054;p=k/n;d=1+z*z/n
    mid=(p+z*z/(2*n))/d;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [100*max(0.,mid-half),100*min(1.,mid+half)]

def contrast(x,y):
    n=len(x);m=len(y);vx=x.var(ddof=1)/n;vy=y.var(ddof=1)/m
    se=np.sqrt(vx+vy);df=(vx+vy)**2/(vx*vx/(n-1)+vy*vy/(m-1))
    mean=float(x.mean()-y.mean());half=t.ppf(.975,df)*se
    adjusted=t.ppf(1-.05/(2*54),df)*se
    return dict(delta_MAE_pp=mean,se_pp=float(se),df=float(df),
        mc95_pp=[float(mean-half),float(mean+half)],
        simultaneous95_pp=[float(mean-adjusted),float(mean+adjusted)])

def main():
    rows=[];comparisons=[]
    for index,(circuit,budget,r,a,b) in enumerate(cells()):
        errors={}
        common=dict(cell=index,circuit=circuit,budget=budget,r=r,pA=a,pB=b,replicates=P['replicates'])
        for method in ['full','scaled']:
            if circuit=='B+' and method=='full':continue
            with np.load(OUT/f'cell_{index:03d}_{method}_fit.npz') as z:
                signed=(z['estimates']-r/4)*100;absolute=abs(signed)
                covered=(z['intervals'][:,0]<=r/4)&(z['intervals'][:,1]>=r/4)
                widths=np.diff(z['intervals'],axis=1)[:,0]*100
                row=dict(**common,method=method,MAE_pp=float(absolute.mean()),
                    MAE_se_pp=float(absolute.std(ddof=1)/np.sqrt(len(absolute))),bias_pp=float(signed.mean()),
                    coverage_pct=float(100*covered.mean()),coverage_mc95_pct=wilson(int(covered.sum()),len(covered)),
                    width_median_pp=float(np.median(widths)),width_p95_pp=float(np.quantile(widths,.95)),
                    target_boundary_pct=float(100*z['target_boundary'].mean()),
                    noise_boundary_pct=float(100*z['noise_boundary'].mean()))
                errors[method]=absolute;rows.append(row)
        if circuit!='B+':comparisons.append(dict(**common,**contrast(errors['full'],errors['scaled'])))
    family=[]
    for circuit in ['PRIMARY','A']:
        subset=[c for c in comparisons if c['circuit']==circuit]
        family.append(dict(circuit=circuit,cells=len(subset),positive=sum(c['delta_MAE_pp']>0 for c in subset),
            negative=sum(c['delta_MAE_pp']<0 for c in subset),
            pointwise_scaling_better=sum(c['mc95_pp'][0]>0 for c in subset),
            pointwise_full_better=sum(c['mc95_pp'][1]<0 for c in subset),
            simultaneous_scaling_better=sum(c['simultaneous95_pp'][0]>0 for c in subset),
            simultaneous_full_better=sum(c['simultaneous95_pp'][1]<0 for c in subset),
            min_delta_pp=min(c['delta_MAE_pp'] for c in subset),max_delta_pp=max(c['delta_MAE_pp'] for c in subset)))
    report=dict(study=P['study'],replicate_blocks=len(cells())*P['replicates'],
        acquisitions=2*len(cells())*P['replicates'],cells=len(cells()),method_cells=len(rows),
        comparison_cells=len(comparisons),outcomes=rows,comparisons=comparisons,circuit_summary=family)
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    for name,records in [('outcomes',rows),('contrasts',comparisons)]:
        with (OUT/(name+'.csv')).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    print(json.dumps({k:v for k,v in report.items() if k not in ['outcomes','comparisons']},indent=2))

if __name__=='__main__':main()
