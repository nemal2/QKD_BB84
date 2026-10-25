"""Verify or reproduce the extension study in a separate local workspace."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

SUPPLEMENT=Path(__file__).resolve().parent
EXT=SUPPLEMENT/'extensions'


def verify():
    manifest=json.loads((EXT/'SHA256SUMS.json').read_text())
    for name,digest in manifest.items():
        path=(EXT/name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(SUPPLEMENT.parent.resolve()):
            raise RuntimeError(f'Extension manifest path escapes its package: {name}')
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f'Missing or changed extension input: {name}')
    print(f'Verified {len(manifest)} extension inputs.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['analysis','full'],default='analysis')
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--rerun-decoders',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();verify()
    if args.verify_only:return
    if sys.version_info[:2]!=(3,13):raise RuntimeError('Use Python3.13 with supplementary/requirements.txt.')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination=args.output.resolve() if args.output else SUPPLEMENT.parent/'work'/f'extensions-{args.mode}-{stamp}'
    destination.mkdir(parents=True,exist_ok=False)
    ignore=shutil.ignore_patterns('results','__pycache__','*.pyc','*.nbc','*.nbi','.matplotlib')
    for name in ['extensions','study']:
        shutil.copytree(SUPPLEMENT/name,destination/'supplementary'/name,ignore=ignore)
    fresh=destination/'supplementary/extensions';results=fresh/'results'
    if args.mode=='analysis':
        shutil.copytree(EXT/'results',results,ignore=shutil.ignore_patterns('.*.lock'))
        # A fresh report must describe this run, not inherit a previous audit.
        for name in ['validation.json','validation_rerun.json','extension_validation.json',
                     'extension_validation_rerun.json','holdout_rerun_checks.json',
                     'diagnostic_reference_audit.json','analysis_provenance.json']:
            (results/name).unlink(missing_ok=True)
        # Reference matrices are frozen baseline inputs, kept outside extension results.
        target=destination/'supplementary/study/results';target.mkdir()
        for k in [11,29]:shutil.copy2(SUPPLEMENT/f'study/results/codes_seed{k}.npz',target)
    def step(script,*options):
        print(f'Running {script} ...',flush=True)
        with (destination/'reproduction.log').open('a') as handle:
            subprocess.run([sys.executable,str(fresh/script),*options],cwd=destination,
                           stdout=handle,stderr=subprocess.STDOUT,check=True)
    if args.mode=='full':
        step('pilot.py');step('pipelines.py');step('diagnostics.py')
    step('inventory.py')
    rerun=['--rerun'] if args.rerun_decoders else []
    step('validate.py',*rerun);step('check_extensions.py',*rerun)
    step('audit_diagnostic_reference.py')
    step('analyse.py');step('analyse_extensions.py')
    if args.mode=='full':
        import numpy as np
        expected=sorted(p.relative_to(EXT/'results') for p in (EXT/'results').rglob('*.npz'))
        actual=sorted(p.relative_to(results) for p in results.rglob('*.npz'))
        if expected!=actual:raise RuntimeError('Regenerated result file set differs.')
        count=0
        for name in expected:
            a=np.load(EXT/'results'/name,allow_pickle=False);b=np.load(results/name,allow_pickle=False)
            if set(a.files)!=set(b.files):raise RuntimeError(f'Array schema differs: {name}')
            for key in a.files:
                if not np.array_equal(a[key],b[key]):raise RuntimeError(f'Array differs: {name}/{key}')
                count+=1
        (destination/'numeric_comparison.json').write_text(json.dumps(dict(files=len(expected),arrays_exactly_equal=count),indent=2)+'\n')
    step('report.py')
    print(f'Completed extension reproduction: {destination}',flush=True)


if __name__=='__main__':main()
