"""Reproduce the study and manuscript in an isolated workspace.

python supplementary/reproduce.py --mode analysis --compile
python supplementary/reproduce.py --mode full --compile
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

SUPPLEMENT=Path(__file__).resolve().parent


def sha256(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inputs():
    manifest=json.loads((SUPPLEMENT/'SHA256SUMS.json').read_text())
    for name,digest in manifest.items():
        p=SUPPLEMENT/name
        if not p.is_file() or sha256(p)!=digest:raise RuntimeError(f'Missing or changed input: {name}')
    print(f'Verified {len(manifest)} supplementary files.',flush=True)
    return len(manifest)


def verify_environment():
    if sys.version_info[:2]!=(3,13):raise RuntimeError('Use Python 3.13; see README.md.')
    versions=json.loads((SUPPLEMENT/'provenance/dependency-versions.json').read_text())['versions']
    mismatches=[]
    for name,expected in versions.items():
        try:found=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:found='missing'
        if found!=expected:mismatches.append(f'{name}: expected {expected}; found {found}')
    if mismatches:raise RuntimeError('Install requirements.txt:\n'+'\n'.join(mismatches))
    return versions


def prepare(destination,mode):
    destination.mkdir(parents=True,exist_ok=False);sup=destination/'supplementary'
    for name in ['study','upstream','manuscript_support']:
        shutil.copytree(SUPPLEMENT/name,sup/name,ignore=shutil.ignore_patterns('results','__pycache__','.matplotlib','*.pyc','*.nbc','*.nbi'))
    overleaf=destination/'overleaf';(overleaf/'images').mkdir(parents=True)
    for name in ['references.bib','acmart.cls','ACM-Reference-Format.bst']:
        shutil.copy2(SUPPLEMENT/'manuscript_support'/name,overleaf/name)
    if mode=='analysis':
        target=sup/'study/results';target.mkdir()
        for p in (SUPPLEMENT/'study/results').iterdir():
            if p.is_file() and p.suffix in ['.npz','.json']:shutil.copy2(p,target/p.name)
    return sup/'study'


def step(destination,script,args=()):
    print(f'Running {script.name} ...',flush=True)
    with (destination/'reproduction.log').open('a') as log:
        subprocess.run([sys.executable,str(script),*map(str,args)],cwd=destination,
                       stdout=log,stderr=subprocess.STDOUT,check=True)


def compare_arrays(fresh,reference):
    # Called in a subprocess with the scientific environment already checked.
    import numpy as np
    names=sorted(p.name for p in reference.glob('*.npz'))
    assert names==sorted(p.name for p in fresh.glob('*.npz')),'Result files differ'
    arrays=0
    for name in names:
        a=np.load(fresh/name,allow_pickle=False);b=np.load(reference/name,allow_pickle=False)
        assert set(a.files)==set(b.files),name
        for key in a.files:
            assert np.array_equal(a[key],b[key]),(name,key)
            arrays+=1
    return dict(npz_files=len(names),arrays_exactly_equal=arrays)


def compile_paper(destination):
    output=destination/'output';output.mkdir()
    if shutil.which('tectonic'):cmd=['tectonic','-X','compile','main.tex','--outdir','../output','--keep-logs']
    elif shutil.which('latexmk'):cmd=['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error','-outdir=../output','main.tex']
    else:raise RuntimeError('Install Tectonic or latexmk to use --compile.')
    with (destination/'reproduction.log').open('a') as log:
        subprocess.run(cmd,cwd=destination/'overleaf',stdout=log,stderr=subprocess.STDOUT,check=True)
    assert (output/'main.pdf').is_file()
    return 'output/main.pdf'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['full','analysis'],default='full')
    parser.add_argument('--compile',action='store_true');parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--output',type=Path);args=parser.parse_args()
    checked=verify_inputs()
    if args.verify_only:return
    versions=verify_environment()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination=args.output.resolve() if args.output else SUPPLEMENT.parent/'work'/f'{args.mode}-{stamp}'
    study=prepare(destination,args.mode)
    try:
        if args.mode=='full':
            step(destination,study/'reconciliation_v2.py');step(destination,study/'diagnostics_v2.py')
            step(destination,study/'channel_fixed.py')
        for name in ['validate_v2.py','check_v2.py','analyse_v2.py']:
            step(destination,study/name)
        arrays=compare_arrays(study/'results',SUPPLEMENT/'study/results')
        assert (study/'results/summary.json').read_bytes()==(SUPPLEMENT/'study/results/summary.json').read_bytes(),'Analysis differs'
        for name in ['plot_v2.py','write_v2.py']:step(destination,study/name)
        expected=json.loads((SUPPLEMENT/'provenance/expected-manuscript.json').read_text())
        for name,digest in expected.items():assert sha256(destination/'overleaf'/name)==digest,name
        expected_text=json.loads((SUPPLEMENT/'provenance/expected-figure-text.json').read_text())
        if shutil.which('pdftotext'):
            for name,value in expected_text.items():
                actual=subprocess.check_output(['pdftotext','-layout',str(destination/'overleaf/images'/name),'-'],text=True)
                assert actual==value,name
        report=dict(status='passed',mode=args.mode,input_files=checked,dependency_versions=versions,
                    array_comparison=arrays,canonical_summary_identical=True,
                    manuscript_source_identical=True,figure_text_checked=bool(shutil.which('pdftotext')))
        if args.compile:report['compiled_pdf']=compile_paper(destination)
        (destination/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
        print(f'Reproduction passed: {destination}',flush=True)
    except Exception:
        print(f'Inspect {destination / "reproduction.log"}',file=sys.stderr);raise


if __name__=='__main__':main()
