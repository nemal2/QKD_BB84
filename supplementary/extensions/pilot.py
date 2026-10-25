"""Independent pilot, sample-only partial-pooling policy, then 75-cell confirmation."""
import argparse,json,hashlib,time
import numpy as np
from common import *
from provenance import guarded_run, atomic_json, atomic_npz


def calibrate():
    path=OUT/'reference_selection.json'
    if path.exists():return json.loads(path.read_text())
    result={}
    for qi,q in enumerate(P['pilot']['qber']):
        data=frame_data(q,0,200,1000,150,seed('calibrate',qi))
        grid={.02:range(225,351,25),.06:range(425,576,25),.1:range(575,751,25)}[q]
        failures={}
        for m in grid:
            r,_=evaluate(reference_code(q,11,m),data,['pooled'],retain=False)
            failures[m]=int((~r['correct'][0]).sum())
        selected=min(grid,key=lambda m:(abs(failures[m]/150-.1),m))
        result[str(q)]=dict(m=selected,failures=failures,frames=150)
        save_data(OUT/f'reference_cal_{qi}.npz',data)
        print('reference',q,result[str(q)],flush=True)
    atomic_json(path,result);return result


def train():
    selections=calibrate()
    for qi,q in enumerate(P['pilot']['qber']):
        for ai,a in enumerate(P['pilot']['asymmetry']):
            for si,s in enumerate(P['pilot']['sample']):
                tag=f'pilot_q{qi}_a{ai}_s{si}';path=OUT/f'{tag}_evaluation.npz'
                if path.exists():continue
                data=frame_data(q,a,s,1000,200,seed('pilot',qi,ai,si))
                save_data(OUT/f'{tag}_frames.npz',data)
                r,t=evaluate(reference_code(q,11,selections[str(q)]['m']),data,[0.,.25,.5,.75,1.,'shrink','oracle_pool','oracle_basis'])
                atomic_npz(path,**r)
                (OUT/f'{tag}_timing.json').write_text(json.dumps(t))
                print(tag,'fails',(~r['correct']).sum(1).tolist(),flush=True)
    rulepath=OUT/'frozen_rule.json'
    if rulepath.exists():return json.loads(rulepath.read_text())
    values=np.zeros((2,3,4));audit=[];entries=[]
    for path in sorted(OUT.glob('pilot_*_frames.npz')):
        d=dict(np.load(path));r=np.load(path.with_name(path.name.replace('_frames','_evaluation')))
        ii=np.searchsorted([100],int(d['s']),side='left');jj=np.searchsorted([.04,.08],d['pool'],side='right')
        kk=np.searchsorted([1,4,9],d['diff']**2/np.maximum(d['variance'],1e-30),side='right')
        for f in range(len(d['pool'])):entries.append((ii,int(jj[f]),int(kk[f]),r['correct'][:5,f].astype(float)*(1000-int(r['m'])-64)/(1000+int(d['s']))))
    for i in range(2):
        for j in range(3):
            for k in range(4):
                v=np.array([e[3] for e in entries if e[:3]==(i,j,k)])
                gain=np.zeros(5);lower=np.zeros(5)
                if len(v)>1:
                    diff=v-v[:,[0]];gain=diff.mean(0);lower=gain-1.96*diff.std(0,ddof=1)/np.sqrt(len(v))
                choice=int(np.argmax(lower));values[i,j,k]=LAMBDAS[choice]
                audit.append(dict(bin=[i,j,k],frames=len(v),mean_gains=gain.tolist(),lower_bounds=lower.tolist(),chosen=float(values[i,j,k])))
    rule=dict(lambda_grid=LAMBDAS.tolist(),audit=audit,inputs=P['automatic_rule']['inputs'],
              selection=P['automatic_rule']['selection'],pilot_files=len(list(OUT.glob('pilot_*_evaluation.npz'))))
    rule['lambda']=values.tolist()
    rule['pilot_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(OUT.glob('pilot_*_evaluation.npz'))}
    atomic_json(rulepath,rule)
    plan=[]
    for c in P['transfer']['conditions']:
        ai=P['pilot']['asymmetry'].index(c['a']);si=P['pilot']['sample'].index(c['s'])
        d=dict(np.load(OUT/f'pilot_q1_a{ai}_s{si}_frames.npz'));r=np.load(OUT/f'pilot_q1_a{ai}_s{si}_evaluation.npz')
        ix=np.searchsorted(LAMBDAS,selected_lambda(d,rule));F=len(ix)
        delta=(r['correct'][ix,np.arange(F)].astype(float)-r['correct'][0])*((1000-int(r['m'])-64)/(1000+c['s']))
        variance=max(float(np.var(delta,ddof=1)),3/F*((1000-int(r['m'])-64)/(1000+c['s']))**2)
        count=int(np.clip(np.ceil(7.84*variance/(.005**2*5)/100)*100,500,2000))
        plan.append(dict(c,frames_per_matrix=count,pilot_variance=variance,power_scope='conditional Monte Carlo approximation; five-matrix variation assessed separately'))
    atomic_json(OUT/'frozen_confirmation_plan.json',plan)
    print('rule frozen',values.tolist(),'transfer',plan,flush=True);return rule


def confirm():
    rule=train();selections=calibrate()
    for qi,q in enumerate(P['pilot']['qber']):
        for ai,a in enumerate(P['pilot']['asymmetry']):
            for si,s in enumerate(P['pilot']['sample']):
                tag=f'confirm_q{qi}_a{ai}_s{si}';path=OUT/f'{tag}_evaluation.npz'
                if path.exists():continue
                data=frame_data(q,a,s,1000,500,seed('confirm',qi,ai,si));save_data(OUT/f'{tag}_frames.npz',data)
                r,t=evaluate(reference_code(q,29,selections[str(q)]['m']),data,['pooled','separate','shrink','selected','oracle_pool','oracle_basis'],rule)
                atomic_npz(path,**r);(OUT/f'{tag}_timing.json').write_text(json.dumps(t))
                print(tag,'fails',(~r['correct']).sum(1).tolist(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['pilot','confirm'],default='confirm')
    args=parser.parse_args()
    with guarded_run('asymmetry', ['protocol.json','common.py','pilot.py',
                                  '../study/reconciliation_v2.py']):
        train() if args.stage=='pilot' else confirm()
