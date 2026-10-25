"""One canonical analysis for all tables and plotted data; no hand-entered results."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from reconciliation_v2 import OUT,P,h2,COUNTS,qk

RNG=np.random.default_rng(P['statistics']['seed'])


def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n;c=(p+z*z/(2*n))/den
    d=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(max(0,c-d)),float(min(1,c+d))]


def paired(a,b,ya=1.,yb=1.):
    a=np.asarray(a,bool);b=np.asarray(b,bool);n=len(a)
    ct=np.bincount(a.astype(int)*2+b.astype(int),minlength=4)
    boot=RNG.multinomial(n,ct/n,size=10000)/n
    ca=boot[:,2]+boot[:,3];cb=boot[:,1]+boot[:,3]
    interval=np.quantile(yb*cb-ya*ca,[.025,.975])
    p=binomtest(int(ct[1]),int(ct[1]+ct[2]),.5).pvalue if ct[1]+ct[2] else 1.
    fa=int((~a).sum());fb=int((~b).sum())
    ratio=fb/fa if fa else None
    valid=(1-ca)>0
    rci=np.quantile((1-cb[valid])/(1-ca[valid]),[.025,.975]).tolist() if fa>=5 and valid.sum()>9500 else None
    return dict(frames=n,failures_pooled=fa,failures_aware=fb,fer_pooled=fa/n,fer_aware=fb/n,
                fer_pooled_ci=wilson(fa,n),fer_aware_ci=wilson(fb,n),rescued=int(ct[1]),harmed=int(ct[2]),
                ratio=ratio,ratio_ci=rci,p_exact=float(p),delta_yield=float(yb*b.mean()-ya*a.mean()),
                delta_yield_ci=interval.tolist())


def holm(rows,key='p_exact'):
    ix=sorted(range(len(rows)),key=lambda i:rows[i][key]);running=0
    for j,i in enumerate(ix):
        running=max(running,min(1.,rows[i][key]*(len(rows)-j)));rows[i]['p_holm']=running


def write_csv(name,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)


def cell(scheme,q,tag,selection,code_seed=11):
    ev=np.load(OUT/f'{tag}_evaluation.npz');frames=np.load(OUT/f'{tag}_frames.npz')
    choices=ev['m_values'].tolist();mu,ma=selection['selected_m'];iu=choices.index(mu);ia=choices.index(ma)
    true=np.take_along_axis(frames['errors'],frames['kept_indices'],axis=1)
    measured=float(true.mean());base=dict(scheme=scheme,target_qber=q,channel_qber=float(frames['channel_qber']),
             measured_qber=measured,m_pooled=mu,m_aware=ma,code_seed=code_seed,
             f_syn_pooled=mu/(1000*h2(measured)),f_syn_aware=ma/(1000*h2(measured)),
             f_ec_pooled=(mu+64)/(1000*h2(measured)),f_ec_aware=(ma+64)/(1000*h2(measured)))
    rows=[]
    for label,key in [('BP','bp_correct'),('BP+OSD','correct')]:
        c=ev[key];yu=(1000-mu-64)/1200;ya=(1000-ma-64)/1200
        rows.append(dict(base,decoder=label,comparison='operating',**paired(c[iu,0],c[ia,3],yu,ya)))
        rows.append(dict(base,decoder=label,comparison='same_aware_code',**paired(c[ia,0],c[ia,3])))
        rows.append(dict(base,decoder=label,comparison='same_pooled_code',**paired(c[iu,0],c[iu,3])))
    ablations=[]
    for j,label in enumerate(['pooled','separate_Jeffreys','coherent_unshrunk','coherent_shrink']):
        for decoder,key in [('BP','bp_correct'),('BP+OSD','correct')]:
            ablations.append(dict(base,prior=label,decoder=decoder,m=ma,
                                  failures=int((~ev[key][ia,j]).sum()),frames=len(true),
                                  **{k:v for k,v in paired(ev[key][ia,0],ev[key][ia,j]).items() if k not in ['frames']}))
    return rows,ablations


def bootstrap_means(values):
    # Stratify over attack fraction, retaining method pairing in every replicate.
    v=np.asarray(values);b=np.empty((10000,v.shape[-1]))
    for start in range(0,10000,100):
        idx=RNG.integers(v.shape[1],size=(v.shape[0],100,v.shape[1]))
        for j in range(100):b[start+j]=np.mean(np.take_along_axis(v,idx[:,j,:,None],axis=1),axis=(0,1))
    return np.quantile(b,[.025,.975],axis=0),b


def main():
    selections=json.loads((OUT/'selections.json').read_text());mainrows=[];ab=[];channel=[];rep=[]
    for scheme in P['schemes']:
        for q in P['qber_targets']:
            r,a=cell(scheme,q,f'main_{scheme.replace("+","plus")}_{round(q*100):02d}',selections[f'{scheme}_{q}'])
            mainrows+=r;ab+=a
    for label in ['BP','BP+OSD']:
        for comp in ['operating','same_aware_code','same_pooled_code']:
            holm([r for r in mainrows if r['decoder']==label and r['comparison']==comp])
    for scheme in P['channel_control']['schemes']:
        for c in P['channel_control']['channel_qber']:
            sel=json.loads((OUT/f'chcal_{scheme}_{round(1000*c):03d}_selection.json').read_text())
            r,_=cell(scheme,.1,f'channel_{scheme}_{round(1000*c):03d}',sel);channel+=r
    for seed in P['replication_matrix_seeds']:
        for scheme in ['PRIMARY','A','B+']:
            r,_=cell(scheme,.1,f'rep_{seed}_{scheme.replace("+","plus")}',selections[f'{scheme}_0.1'],seed);rep+=r
    write_csv('main_results.csv',mainrows);write_csv('prior_ablation.csv',ab)
    write_csv('channel_results.csv',channel);write_csv('replication_results.csv',rep)
    ds=[np.load(OUT/f'diag_{round(100*r):02d}.npz') for r in P['diagnostic']['p_eve']]
    names=['linear_four','gate_four','gate_two','calibrated_BSC','calibrated_gate','Fiorini']
    estimates=[];per=[]
    for r,d in zip(P['diagnostic']['p_eve'],ds):
        zero=list(d['offsets']).index(0)
        est=np.column_stack([d['fit4'][:,:2],d['fit2'][:,1],d['calibrated'][:,zero]])
        errors=est-r/4;estimates.append(errors)
        for j,name in enumerate(names):
            per.append(dict(p_eve=r,method=name,mae_pp=float(100*np.mean(abs(errors[:,j]))),
                            bias_pp=float(100*errors[:,j].mean()),sd_pp=float(100*errors[:,j].std(ddof=1))))
    errors=np.array(estimates);ci,boot=bootstrap_means(100*abs(errors));diagnostic=[]
    for j,name in enumerate(names):
        diagnostic.append(dict(method=name,mae_pp=float(np.mean(100*abs(errors[:,:,j]))),
                               mae_ci_pp=ci[:,j].tolist(),bias_pp=float(errors[:,:,j].mean()*100),
                               sd_pp=float(np.sqrt(np.mean(np.var(errors[:,:,j],axis=1,ddof=1)))*100)))
    difference=boot[:,2]-boot[:,4]
    drift=[]
    for j,delta in enumerate(ds[0]['offsets']):
        values=np.array([100*abs(d['calibrated'][:,j]-r/4) for r,d in zip(P['diagnostic']['p_eve'],ds)])
        dci,_=bootstrap_means(values)
        for k,name in enumerate(names[3:]):
            drift.append(dict(delta=float(delta),method=name,mae_pp=float(values[:,:,k].mean()),ci_pp=dci[:,k].tolist()))
    native=[]
    for r,d in zip(P['diagnostic']['p_eve'],ds):
        e=d['native_estimates'][:,2]-r/4
        native.append(dict(p_eve=r,method='Fiorini_native',mae_pp=float(100*abs(e).mean()),bias_pp=float(100*e.mean())))
    fixed=[]
    for scheme in ['PRIMARY','A']:
        for r in mainrows:
            if r['scheme']==scheme and r['target_qber']==.1 and r['comparison']=='same_aware_code':
                fixed.append(dict(r,m=r['m_aware']))
        for c in [.025,.05,.075,.10]:
            ev=np.load(OUT/f'chfixed_{scheme}_{round(c*1000):03d}_evaluation.npz')
            for decoder,key in [('BP','bp_correct'),('BP+OSD','correct')]:
                fixed.append(dict(scheme=scheme,channel_qber=c,decoder=decoder,m=int(ev['m_values'][0]),
                                  **paired(ev[key][0,0],ev[key][0,1])))
    write_csv('channel_fixed_results.csv',fixed)
    summary=dict(main=mainrows,ablation=ab,channel=channel,channel_fixed=fixed,replication=rep,diagnostic=diagnostic,
                 diagnostic_by_attack=per,drift=drift,native_comparator=native,
                 two_point_minus_calibrated_mae_ci_pp=np.quantile(difference,[.025,.975]).tolist(),
                 protocol=json.loads((Path(__file__).parent/'protocol.json').read_text()))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    write_csv('diagnostic_results.csv',diagnostic);write_csv('diagnostic_by_attack.csv',per);write_csv('drift_results.csv',drift)
    print('canonical analysis complete',flush=True)


if __name__=='__main__':main()
