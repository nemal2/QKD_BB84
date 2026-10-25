"""Verify or reproduce the independent budget/interception precision study."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys
import numpy as np

SUP=Path(__file__).resolve().parent;STUDY=SUP/'precision_validation'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify-only',action='store_true')
    p.add_argument('--mode',choices=['analysis','full'],default='analysis')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    for name,digest in json.loads((STUDY/'SHA256SUMS.json').read_text()).items():
        file=(STUDY/name).resolve();assert file.is_relative_to(STUDY.resolve())
        assert hashlib.sha256(file.read_bytes()).hexdigest()==digest,name
    if args.verify_only:print('Verified budget/interception precision study.');return
    if args.output is None:raise ValueError('--output is required.')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    target=out/'supplementary/precision_validation';target.mkdir(parents=True)
    for file in STUDY.iterdir():
        if file.is_file() and file.suffix in ['.py','.json','.md'] and file.name!='SHA256SUMS.json':shutil.copy2(file,target/file.name)
    if args.mode=='analysis':shutil.copytree(STUDY/'results',target/'results')
    env=dict(os.environ);env['NUMBA_NUM_THREADS']='4'
    scripts=(['experiment.py'] if args.mode=='full' else [])+['analyse.py','validate.py']
    for script in scripts:
        with (out/(script+'.log')).open('w') as log:
            subprocess.run([sys.executable,str(target/script)],cwd=target,env=env,
                           stdout=log,stderr=subprocess.STDOUT,check=True)
    count=0
    for file in sorted((STUDY/'results').glob('*.npz')):
        with np.load(file) as a,np.load(target/'results'/file.name) as b:
            assert a.files==b.files,file.name
            for key in a.files:
                assert np.array_equal(a[key],b[key],equal_nan=True) if a[key].dtype.kind in 'fc' else np.array_equal(a[key],b[key]),(file.name,key)
                count+=1
    for name in ['summary.json','outcomes.csv','contrasts.csv','verification.json']:
        assert (STUDY/'results'/name).read_bytes()==(target/'results'/name).read_bytes(),name
    report=dict(status='passed',mode=args.mode,retained_arrays_matched=count,
                stochastic_acquisitions_rerun=args.mode=='full',summary_files_matched=4)
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
