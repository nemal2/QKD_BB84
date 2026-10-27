"""Check generated tables, inherited circuit tables, references, and headline claims."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_artifacts as build
import channel_information
import review_analysis
import method_checks


def select(rows, **filters):
    matches = [r for r in rows if all(r.get(k) == v for k, v in filters.items())]
    if len(matches) != 1:
        raise AssertionError((filters, len(matches)))
    return matches[0]


def physical_coincidence():
    """Recompute the equal table summaries from distinct per-frame records."""
    import numpy as np
    from scipy.stats import t
    folder = build.SUP/'extensions/results'
    with (folder/'physical_results.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    controls = build.load('controls_summary.json')
    outcomes, report = {}, {}
    for model in ['gate_dependent', 'pauli']:
        values, nets, outcomes[model] = [], [], []
        for seed in [211,223,227,229,233]:
            with np.load(folder/f'physical_gallager_k{seed}_{model}_matched_evaluation.npz',
                         allow_pickle=False) as data:
                names = list(data['methods'])
                base = data['correct'][names.index('pooled')]
                aware = data['correct'][names.index('transition')]
                rescued = int(np.sum(~base & aware))
                harmed = int(np.sum(base & ~aware))
                value = (rescued-harmed)/len(base)*(int(data['n'])-int(data['m'])-64)/1200
                row = select(rows, family='gallager',matrix_seed=str(seed),model=model,
                             setting='matched',method='transition')
                assert (rescued,harmed)==(int(row['rescued']),int(row['harmed']))
                assert np.isclose(value,float(row['delta_yield']),rtol=0,atol=1e-14)
                values.append(value);nets.append(rescued-harmed)
                outcomes[model].append(np.stack([base,aware]))
        mean = float(np.mean(values))
        half = float(t.ppf(.975,4)*np.std(values,ddof=1)/np.sqrt(5))
        summary = select(controls,experiment='physical',family='gallager',model=model,
                         setting='matched',method='transition')
        assert np.allclose([mean,mean-half,mean+half],
                           [summary['mean'],*summary['ci']],rtol=0,atol=1e-14)
        report[model] = dict(net_rescues=nets,mean_pp=100*mean,
                             ci_pp=[100*(mean-half),100*(mean+half)])
    assert report['gate_dependent']['net_rescues']==[6,1,1,3,1]
    assert report['pauli']['net_rescues']==[6,3,1,1,1]
    assert all(not np.array_equal(x,y) for x,y in zip(*outcomes.values()))
    return report


def audit(source, baseline=None):
    build.fill(source,check=True)
    main=(source/'main.tex').read_text();appendix=(source/'appendix.tex').read_text()
    text=main+'\n'+appendix;plain=re.sub(r'\s+',' ',text)
    labels=re.findall(r'\\label\{([^}]+)\}',text)
    assert len(labels)==len(set(labels))
    refs=re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',text)
    assert set(refs)<=set(labels),set(refs)-set(labels)
    keys=re.findall(r'@\w+\{([^,]+),',(source/'references.bib').read_text())
    assert len(keys)==len(set(keys))
    cited={k.strip() for x in re.findall(r'\\cite\{([^}]+)\}',text) for k in x.split(',')}
    assert cited<=set(keys),cited-set(keys)
    figures=re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',text)
    assert len(figures)==9 and all((source/f).is_file() for f in figures)
    assert text.count(r'\Description{')==len(figures)
    assert main.count(r'\begin{table}')==4 and appendix.count(r'\begin{table}')==22
    assert len(re.findall(r'% BEGIN GENERATED ',text))==19
    assert not re.search(r'@@[A-Z_]+@@|TODO|TBD|XXXXXXX',text)
    assert not re.search(r'(?<!\\)\b(?:ef|abel)\{(?:tab|eq|fig|sec|app):',text)
    assert not any(ord(c)<32 and c not in '\n\t' for c in text)
    assert not re.search(r'\bextension\b|\bhistorical\b|added after',text,re.I)
    assert all((source/f).read_text().startswith('% ') for f in ['main.tex','appendix.tex'])
    abstract=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',main,re.S).group(1)
    assert len(abstract.split())<200
    assert r'\documentclass[manuscript,screen,review,anonymous]{acmart}' in main
    assert not re.search(r'\\(?:author|email|affiliation|orcid)\{', main)
    assert r'\acmJournal{TQC}' in main and r'\acmDOI{}' in main
    assert 'interior conditions' in main and 'Physical bounds' in main
    assert 'Three values do not' in main and 'determine a general cubic' in main
    assert r'\label{eq:poolingscore}' in appendix
    assert 'exact ties choose the smallest weight' in plain
    assert '969 for each of the ten' in plain
    assert 'remains usable' not in main
    assert 'post hoc fixed-code sensitivity analysis' in appendix
    assert "QBER, asymmetry, sample-size grid and syndrome lengths are repeated" in plain
    assert 'One development' in main and 'couple sample' in main
    assert r'\label{eq:subset}' in main and r'\label{eq:basisinverse}' in main
    assert 'Different pairs $(c,r)$ can give the' in main
    assert 'row space of $M$' in main and r'\label{cor:circuits}' in main
    # Preserve every experiment while placing full recovery methods with their records.
    required_labels = ['tab:observations','fig:overview','tab:assumptions','prop:identification',
        'fig:precisiongrid','sec:precisiongrid','tab:crosscircuit','tab:recoveryoverview',
        'fig:circuits','fig:ambiguity','fig:transfer','eq:separatedresponse','sec:limits']
    assert all(r'\label{'+label+'}' in main for label in required_labels)
    supporting_labels = ['tab:information','tab:matched','tab:asymmetry','fig:asymmetry',
        'eq:yielddecomposition','eq:channel','app:reconmethods','app:confirmation',
        'tab:cohorts','tab:validation','tab:precisionprimary','tab:precisiona','tab:precisionbplus',
        'tab:operating','tab:ablation','tab:replication','tab:channel','tab:rule','tab:yieldaccounting',
        'tab:absolutetransfer','tab:transferall','tab:sample','tab:physical','tab:holdout',
        'tab:profiles','tab:diagnostic','tab:stress','fig:exposure','fig:curves']
    assert all(r'\label{'+label+'}' in appendix for label in supporting_labels)
    manifests={}
    for folder in [build.SUP,build.SUP/'extensions',build.SUP/'diagnostic_validation',build.SUP/'observation_validation',build.SUP/'precision_validation']:
        path=folder/'SHA256SUMS.json'
        if not path.exists():raise AssertionError(f'Missing scientific manifest: {path}')
        items=json.loads(path.read_text())
        for name,digest in items.items():
            file=(folder/name).resolve();assert file.is_relative_to(build.ROOT.resolve())
            assert hashlib.sha256(file.read_bytes()).hexdigest()==digest,name
        manifests[str(folder.relative_to(build.ROOT))]=len(items)
    claims=[]
    def record(name,evidence,rendered):
        assert re.sub(r'\s+',' ',rendered) in plain,(name,rendered)
        claims.append(dict(claim=name,evidence=evidence,manuscript_text=rendered))
    obs=json.loads((build.SUP/'observation_validation/results/summary.json').read_text())
    verify=json.loads((build.SUP/'observation_validation/results/verification.json').read_text())
    assert verify['status']=='passed' and verify['new_replicate_blocks']==6500
    assert len(obs['cross_circuit'])==46 and len(obs['retained_profiles'])==5
    assert obs['inverse_validity']=={'valid':31000}
    for circuit,method in [('PRIMARY','full'),('PRIMARY','scaled'),('A','full'),('A','scaled')]:
        row=select(obs['cross_circuit'],circuit=circuit,scenario='equal',method=method,readout=0.)
        record(f'{circuit} {method} equal-device MAE','observation_validation/results/summary.json',f"{row['MAE_pp']:.3f}")
    for field in ['width_median_pp','coverage_pct']:
        vals=[r[field] for r in obs['retained_profiles']];fmt='.2f' if field=='width_median_pp' else '.1f'
        record('Retained profile '+field,'observation_validation/results/summary.json',f'{min(vals):{fmt}}--{max(vals):{fmt}}')
    for device in ['pA','pB']:
        vals=[r['device_profiles'][device]['width_median_pp'] for r in obs['retained_profiles']]
        record(device+' profile width range','observation_validation/results/summary.json',f'{min(vals):.2f}--{max(vals):.2f}')
    olddiag=json.loads((build.SUP/'diagnostic_validation/results/summary.json').read_text())
    for method in ['basis_single','basis_inverse','aggregate_two_cap2.5','aggregate_two_cap8','calibrated']:
        row=select(olddiag['grid_aggregate'],stage='nominal',budget=None,method=method)
        record('Range comparison '+method,'diagnostic_validation/results/summary.json',f"{row['MAE_pp']:.3f}")
    precision=json.loads((build.SUP/'precision_validation/results/summary.json').read_text())
    precision_verify=json.loads((build.SUP/'precision_validation/results/verification.json').read_text())
    assert precision_verify['status']=='passed' and precision_verify['retained_arrays']==3078
    assert precision['replicate_blocks']==40500 and precision['acquisitions']==81000
    assert len(precision['outcomes'])==135 and len(precision['comparisons'])==54
    assert (build.SUP/'precision_validation/likelihood.py').read_bytes()==(build.SUP/'observation_validation/likelihood.py').read_bytes()
    methods=method_checks.analyse()
    assert methods==json.loads((build.HERE/'method-checks.json').read_text())
    assert methods['pooling_frames_reconstructed']==15000 and methods['pooling_bins_reconstructed']==24
    lower=[sum(r['lower_noise_hits'][i] for r in methods['retained_boundary_records']) for i in range(2)]
    assert lower==[1241,1]
    record('Lower nuisance boundary count','method-checks.json','1241 reach the lower Alice-noise bound and one reaches the lower Bob-noise bound;')
    widths=[r['width_median_pp'] for r in precision['outcomes'] if r['circuit']=='B+' and r['r']==.02 and r['budget']==2000]
    record('Weak-interception B+ minimum median width','precision_validation/results/summary.json',f'{min(widths):.2f}')
    record('Weak-interception B+ maximum median width','precision_validation/results/summary.json',f'{max(widths):.2f}')
    ps=select(precision['circuit_summary'],circuit='PRIMARY')
    aa=select(precision['circuit_summary'],circuit='A')
    assert (ps['negative'],ps['simultaneous_full_better'])==(27,26)
    assert (aa['positive'],aa['simultaneous_scaling_better'],aa['simultaneous_full_better'])==(25,19,2)
    record('PRIMARY grid direction','precision_validation/results/summary.json','Full data have lower MAE in all 27 Standard comparisons, with 26 simultaneous')
    record('A grid direction','precision_validation/results/summary.json','A favors scaling in 25 of 27 point comparisons')
    for cell in [28,47]:
        row=select(precision['comparisons'],cell=cell)
        assert row['simultaneous95_pp'][1]<0
        record('A weak-interception reversal '+str(cell),'precision_validation/results/summary.json',f"{row['delta_MAE_pp']:.3f}")
        record('A reversal interval '+str(cell),'precision_validation/results/summary.json',f"[{row['mc95_pp'][0]:+.3f}, {row['mc95_pp'][1]:+.3f}]")
    study=json.loads((build.SUP/'study/results/summary.json').read_text())
    a=select(study['main'],scheme='A',target_qber=.1,decoder='BP+OSD',comparison='same_aware_code')
    assert (a['failures_pooled'],a['failures_aware'],a['frames'])==(28,13,2000)
    record('Matched-code failures','study/results/summary.json','failures from 28 to 13 out of 2000')
    record('Matched-code ratio','study/results/summary.json',f"{a['ratio']:.3f} [{a['ratio_ci'][0]:.2f}, {a['ratio_ci'][1]:.2f}]")
    for decoder in ['BP','BP+OSD']:
        row=select(study['main'],scheme='A',target_qber=.1,decoder=decoder,comparison='same_pooled_code')
        assert (row['m_pooled'],row['failures_pooled'],row['failures_aware'])==(700,0,0)
    operating=select(study['main'],scheme='A',target_qber=.1,decoder='BP',comparison='operating')
    assert (operating['failures_aware'],operating['failures_pooled'])==(19,0)
    record('Operating upper FER bound','study/results/summary.json',f"Wilson interval [{100*operating['fer_aware_ci'][0]:.2f}\\%,{100*operating['fer_aware_ci'][1]:.2f}\\%]")
    selected=select(build.load('asymmetry_summary.json')['method_summaries'],method='selected')
    record('Corrected confirmation gains','extensions/results/asymmetry_summary.json',f"{selected['significant_gains']} Holm-supported gains")
    assert selected['significant_harms']==0
    record('Confirmation mean','extensions/results/asymmetry_summary.json',f"mean gain is {100*selected['mean_delta_yield']:.3f} pp")
    record('Confirmation harm','extensions/results/asymmetry_summary.json',f"{100*selected['worst_cell']['delta_yield']:.3f} pp")
    transfer=build.load('transfer_summary.json')
    for family in ['gallager','irregular']:
        row=select(transfer,family=family,n=16000,condition=2,method='adaptive_pipeline')
        record(family+' pipeline gain','extensions/results/transfer_summary.json',f"{100*row['mean']:+.2f} [{100*row['ci'][0]:+.2f}, {100*row['ci'][1]:+.2f}]")
    information=channel_information.analyse()
    assert information==json.loads((build.HERE/'channel-information.json').read_text())
    review=review_analysis.analyse()
    assert review==json.loads((build.HERE/'review-analysis.json').read_text())
    assert all(r['upper_bits']==r['lower_bits']==0 for cohort in review['clipping'].values() for r in cohort.values())
    for row in information['pipeline']:
        for key in ['disclosure_pp','recovery_pp']:record(row['family']+' '+key,'channel-information.json',f"{row[key]:+.3f}")
    coincidence=physical_coincidence();inherited=[]
    if baseline:
        old=(baseline/'main.tex').read_text()+'\n'+(baseline/'appendix.tex').read_text()
        def rows(document,label):
            table=next(t for t in re.findall(r'\\begin\{table\}.*?\\end\{table\}',document,re.S) if r'\label{'+label+'}' in t)
            return table.split(r'\midrule')[1].split(r'\bottomrule')[0].strip()
        for label in ['tab:matched','tab:operating','tab:ablation','tab:replication','tab:channel']:
            assert rows(text,label)==rows(old,label).replace('PRIMARY','Standard'),label;inherited.append(label)
    return dict(status='passed',abstract_words=len(abstract.split()),anonymous=True,figure_count=9,table_count=26,
        main_tables=4,appendix_tables=22,generated_tables_checked=19,citation_count=len(cited),unresolved_references=0,
        comprehensive_main_labels=required_labels,comprehensive_supporting_labels=supporting_labels,
        anonymous_front_matter=True,scientific_manifests_verified=manifests,
        channel_information_analysis_verified=True,retained_record_reporting_verified=True,
        observation_validation_verified=verify,precision_validation_verified=precision_verify,method_documentation_verified=methods,new_reconciliation_experiments=False,
        physical_coincidence_checked_from_records=coincidence,inherited_circuit_tables_checked=inherited,
        headline_claims=claims,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [source/'main.tex',source/'appendix.tex',source/'references.bib']})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,default=build.ROOT/'overleaf')
    p.add_argument('--baseline-source',type=Path)
    p.add_argument('--output',type=Path)
    args=p.parse_args();result=audit(args.source_dir,args.baseline_source)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['headline_claims','source_sha256']},indent=2))


if __name__=='__main__':main()
