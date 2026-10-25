"""Reproduce the integrated final manuscript without overwriting working sources.

python supplementary/reproduce_final.py --mode analysis --compile
Use --mode build for checked tables/plots only; analysis reruns all five study analyses.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys

SUP=Path(__file__).resolve().parent
ROOT=SUP.parent
sys.path.insert(0,str(SUP/'finalization'))
import build_artifacts
import audit_manuscript


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['build','analysis'],default='analysis')
    p.add_argument('--compile',action='store_true')
    p.add_argument('--working-copy',action='store_true',
                   help='Allow editorial changes to final sources; all five scientific manifests are still verified.')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    # The original manifests remain untouched; the final manifest records only the integration.
    for script in ['reproduce.py','reproduce_extensions.py','reproduce_diagnostic_validation.py','reproduce_observation_validation.py','reproduce_precision_validation.py']:
        subprocess.run([sys.executable,str(SUP/script),'--verify-only'],check=True)
    manifest_path=SUP/'finalization/SHA256SUMS.json'
    if manifest_path.exists() and not args.working_copy:
        for name,digest in json.loads(manifest_path.read_text()).items():
            source=(ROOT/name).resolve()
            if not source.is_relative_to(ROOT.resolve()) or not source.is_file() or sha(source)!=digest:
                raise RuntimeError(f'Changed final manuscript input: {name}')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination=args.output.resolve() if args.output else ROOT/'work'/f'final-{args.mode}-{stamp}'
    destination.mkdir(parents=True,exist_ok=False)
    baseline=None
    if args.mode=='analysis':
        for script,child in [('reproduce.py','circuit'),('reproduce_extensions.py','extension')]:
            with (destination/(child+'.log')).open('w') as log:
                subprocess.run([sys.executable,str(SUP/script),'--mode','analysis','--output',str(destination/child)],
                               cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        baseline=destination/'circuit/overleaf'
        for name in ['asymmetry_results.csv','asymmetry_summary.json','transfer_summary.json',
                     'controls_summary.json','diagnostic_summary.json','diagnostic_results.csv',
                     'transfer_results.csv','pipeline_results.csv','physical_results.csv','fixed_total_results.csv']:
            assert (destination/'extension/supplementary/extensions/results'/name).read_bytes()==(SUP/'extensions/results'/name).read_bytes(),name
        with (destination/'diagnostic-validation.log').open('w') as log:
            subprocess.run([sys.executable,str(SUP/'reproduce_diagnostic_validation.py'),
                            '--mode','analysis','--output',str(destination/'diagnostic-validation')],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        with (destination/'observation-validation.log').open('w') as log:
            subprocess.run([sys.executable,str(SUP/'reproduce_observation_validation.py'),
                            '--mode','analysis','--output',str(destination/'observation-validation')],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        with (destination/'precision-validation.log').open('w') as log:
            subprocess.run([sys.executable,str(SUP/'reproduce_precision_validation.py'),
                            '--mode','analysis','--output',str(destination/'precision-validation')],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
    source=destination/'overleaf';(source/'images').mkdir(parents=True)
    for name in ['main.tex','appendix.tex','references.bib','acmart.cls','ACM-Reference-Format.bst']:
        shutil.copy2(ROOT/'overleaf'/name,source/name)
    text=(source/'main.tex').read_text()+(source/'appendix.tex').read_text()
    figures=re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',text)
    for name in figures:
        shutil.copy2(ROOT/'overleaf'/name,source/name)
    if baseline:
        for name in ['complete_circuits.pdf','gate_exposure.pdf','fixed_code_curves.pdf']:
            shutil.copy2(baseline/'images'/name,source/'images'/name)
    build_artifacts.fill(source);build_artifacts.plots(source)
    report=audit_manuscript.audit(source,baseline)
    report.update(mode=args.mode,analysis_regenerated=args.mode=='analysis',
                  raw_experiments_rerun=False,working_copy=args.working_copy,
                  final_manifest_verified=manifest_path.exists() and not args.working_copy)
    for name in ['main.tex','appendix.tex','references.bib']:
        assert (source/name).read_bytes()==(ROOT/'overleaf'/name).read_bytes(),name
    report['manuscript_source_identical']=True
    if args.compile:
        out=destination/'output/pdf';out.mkdir(parents=True)
        if shutil.which('tectonic'):
            command=['tectonic','-X','compile','main.tex','--outdir',str(out),'--keep-logs','--keep-intermediates']
        elif shutil.which('latexmk'):
            command=['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error','-outdir='+str(out),'main.tex']
        else:raise RuntimeError('Install Tectonic or latexmk to compile.')
        with (destination/'compile.log').open('w') as log:
            subprocess.run(command,cwd=source,stdout=log,stderr=subprocess.STDOUT,check=True)
        shutil.copy2(out/'main.pdf',out/'manuscript.pdf')
        report['compiled_pdf']='output/pdf/manuscript.pdf'
    (destination/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Final manuscript reproduction passed: {destination}')


if __name__=='__main__':main()
