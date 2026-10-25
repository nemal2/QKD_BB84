"""Fresh budget/interception acquisitions using the unchanged profile fitter."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
from likelihood import probabilities,fit_batch,PMAX

HERE=Path(__file__).resolve().parent
P=json.loads((HERE/'protocol.json').read_text())
OUT=HERE/'results'

# Enumerate the full protocol grid in a fixed order so cell indices and seeds are stable.
def cells():
    return [(c,t,r,a,b) for c in P['circuits'] for t in P['budgets']
            for r in P['interception'] for a,b in P['devices']]

def save(path,**data):
    temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,**data);temp.replace(path)

# Generate independent full/scaled counts and retain physical-likelihood fits once per cell.
def run(index,cell):
    circuit,budget,r,a,b=cell
    paths=np.asarray(P['circuits'][circuit],dtype=np.int64)
    stem=f'cell_{index:03d}'
    for mode in ['full','scaled']:
        raw=OUT/f'{stem}_{mode}_counts.npz'
        if not raw.exists():
            rng=np.random.default_rng(np.random.SeedSequence([P['seed_base'],index,int(mode=='scaled')]))
            if mode=='full':
                groups=np.eye(4);scales=np.ones(4)
                transmitted=np.array([budget]);n=rng.multinomial(budget,[.125]*4+[.5],size=P['replicates'])[:,:4]
            else:
                groups=np.full((4,4),.25);scales=np.array(P['scales'])
                transmitted=(budget*np.array(P['weights'])).astype(int)
                assert transmitted.sum()==budget
                n=rng.binomial(transmitted,.5,size=(P['replicates'],4))
            q=probabilities([a,b,r],paths,groups,scales)
            counts=np.stack([rng.binomial(n,q),n],axis=-1)
            save(raw,counts=counts,groups=groups,scales=scales,paths=paths,
                 circuit=circuit,budget=budget,r=r,pA=a,pB=b,target=r/4,
                 transmitted=transmitted,true_probabilities=q,cell=index,method=mode)
        fitpath=OUT/f'{stem}_{mode}_fit.npz'
        if fitpath.exists() or (circuit=='B+' and mode=='full'):continue
        with np.load(raw) as z:
            fit=fit_batch(z['counts'],z['paths'],z['groups'],z['scales'])
        save(fitpath,parameters=fit[:,:3],nll=fit[:,3],estimates=fit[:,2]/4,
             intervals=fit[:,4:],noise_boundary=np.any((fit[:,:2]<1e-6)|(fit[:,:2]>PMAX-1e-6),axis=1),
             target_boundary=(fit[:,2]<1e-6)|(fit[:,2]>1-1e-6))
    print(f'{index+1:02d}/{len(cells())} {circuit} T={budget} r={r} devices={a},{b}',flush=True)

# Protect checkpoint reuse with a source guard, then execute the requested grid cells.
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cell',type=int);args=p.parse_args()
    OUT.mkdir(exist_ok=True)
    guard={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ['protocol.json','likelihood.py','experiment.py']}
    path=OUT/'source_guard.json'
    if path.exists():assert json.loads(path.read_text())==guard,'Protocol or generator changed.'
    else:path.write_text(json.dumps(guard,indent=2)+'\n')
    for index,cell in enumerate(cells()):
        if args.cell is None or args.cell==index:run(index,cell)

if __name__=='__main__':main()
