"""Verify or reproduce the focused full-observation and cross-circuit study."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys
import numpy as np

SUP=Path(__file__).resolve().parent
STUDY=SUP/'observation_validation'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify-only',action='store_true')
    p.add_argument('--mode',choices=['analysis','full'],default='analysis')
    p.add_argument('--output',type=Path)
    p.add_argument('--working-copy',action='store_true')
    args=p.parse_args()
    manifest=STUDY/'SHA256SUMS.json'
    if manifest.exists() and not args.working_copy:
        for name,digest in json.loads(manifest.read_text()).items():
            file=(STUDY/name).resolve();assert file.is_relative_to(STUDY.resolve())
            assert hashlib.sha256(file.read_bytes()).hexdigest()==digest,name
    elif not args.working_copy:raise RuntimeError('Missing frozen study manifest.')
    if args.verify_only:
        print('Verified full-observation study files.');return
    if args.output is None:raise ValueError('--output is required.')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    target=out/'supplementary/observation_validation';target.mkdir(parents=True)
    for file in STUDY.iterdir():
        if file.is_file() and file.suffix in ['.py','.json','.md'] and file.name!='SHA256SUMS.json':
            shutil.copy2(file,target/file.name)
    previous=out/'supplementary/diagnostic_validation';previous.mkdir()
    shutil.copytree(SUP/'diagnostic_validation/results',previous/'results')
    if args.mode=='analysis':shutil.copytree(STUDY/'results',target/'results')
    environment=dict(os.environ);environment['NUMBA_NUM_THREADS']='4'
    scripts=['experiment.py','nuisance_intervals.py'] if args.mode=='full' else []
    scripts+=['analyse.py','validate.py']
    for script in scripts:
        with (out/(script+'.log')).open('w') as log:
            subprocess.run([sys.executable,str(target/script)],cwd=target,env=environment,
                           stdout=log,stderr=subprocess.STDOUT,check=True)
    count=0
    for path in sorted((STUDY/'results').glob('*.npz')):
        with np.load(path) as left,np.load(target/'results'/path.name) as right:
            assert left.files==right.files,path.name
            for name in left.files:
                if left[name].dtype.kind in 'fc':equal=np.array_equal(left[name],right[name],equal_nan=True)
                else:equal=np.array_equal(left[name],right[name])
                assert equal,(path.name,name);count+=1
    for name in ['summary.json','summary.csv','verification.json']:
        assert (STUDY/'results'/name).read_bytes()==(target/'results'/name).read_bytes(),name
    report={'status':'passed','mode':args.mode,'retained_arrays_matched':count,
            'stochastic_acquisitions_rerun':args.mode=='full','summary_files_matched':3}
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
