"""Reproduce the basis-resolved and matched-range diagnostic validation."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import numpy as np

SUP=Path(__file__).resolve().parent
STUDY=SUP/'diagnostic_validation'


def verify():
    manifest=json.loads((STUDY/'SHA256SUMS.json').read_text())
    for name,digest in manifest.items():
        path=(STUDY/name).resolve()
        assert path.is_relative_to(STUDY.resolve()) and path.is_file(),name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
    print(f'Verified {len(manifest)} diagnostic-validation files.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify-only',action='store_true')
    p.add_argument('--mode',choices=['analysis','full'],default='analysis');p.add_argument('--output',type=Path)
    args=p.parse_args();verify()
    if args.verify_only:return
    if args.output is None:raise ValueError('Choose a fresh --output directory.')
    destination=args.output.resolve();destination.mkdir(parents=True,exist_ok=False)
    target=destination/'supplementary/diagnostic_validation';target.mkdir(parents=True)
    for name in ['models.py','experiment.py','analyse.py','validate.py','protocol.json','README.md']:
        shutil.copy2(STUDY/name,target/name)
    dependencies={'extensions':['common.py','physics.py','diagnostics.py','protocol.json','provenance.py'],
                  'study':['reconciliation_v2.py','protocol.json'],
                  'finalization':['channel_information.py']}
    for directory,names in dependencies.items():
        folder=destination/'supplementary'/directory;folder.mkdir(parents=True)
        for name in names:shutil.copy2(SUP/directory/name,folder/name)
    results=target/'results';results.mkdir()
    if args.mode=='analysis':
        for source in (STUDY/'results').iterdir():
            if source.is_file():shutil.copy2(source,results/source.name)
    else:
        shutil.copy2(STUDY/'results/design.json',results/'design.json')
        with (destination/'experiment.log').open('w') as log:
            subprocess.run([sys.executable,str(target/'experiment.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
    for name in ['analyse.py','validate.py']:
        with (destination/(name+'.log')).open('w') as log:
            subprocess.run([sys.executable,str(target/name)],stdout=log,stderr=subprocess.STDOUT,check=True)
    for name in ['summary.json','cell-results.csv','held-out-results.csv','verification.json']:
        assert (results/name).read_bytes()==(STUDY/'results'/name).read_bytes(),name
    arrays=0
    if args.mode=='full':
        for source in (STUDY/'results').glob('*.npz'):
            with np.load(source,allow_pickle=False) as a,np.load(results/source.name,allow_pickle=False) as b:
                assert set(a.files)==set(b.files),source.name
                for key in a.files:
                    assert np.array_equal(a[key],b[key],equal_nan=True) if a[key].dtype.kind in 'fc' else np.array_equal(a[key],b[key]),(source.name,key)
                    arrays+=1
    report=dict(status='passed',mode=args.mode,stochastic_experiments_rerun=args.mode=='full',
                retained_arrays_matched=arrays,summary_files_matched=4)
    (destination/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
