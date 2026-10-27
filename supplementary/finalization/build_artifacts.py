"""Regenerate final-manuscript tables and plots from the frozen numerical analyses.

Run from anywhere: python supplementary/finalization/build_artifacts.py
Use --check to check tables without changing manuscript sources.
"""
from pathlib import Path
import argparse
import csv
import json
import os
import re

HERE = Path(__file__).resolve().parent
SUP = HERE.parent
ROOT = SUP.parent


def load(name):
    return json.loads((SUP / 'extensions/results' / name).read_text())


def ci(row):
    return f"{100*row['mean']:+.2f} [{100*row['ci'][0]:+.2f}, {100*row['ci'][1]:+.2f}]"


def table(label, caption, columns, header, rows):
    return '\n'.join([
        r'\begin{table}[!htbp]', r'\centering', r'\small\setlength{\tabcolsep}{4pt}',
        r'\caption{' + caption + '}', r'\label{tab:' + label + '}',
        r'\begin{tabular}{@{}' + columns + '@{}}', r'\toprule',
        header + r' \\', r'\midrule',
        *[' & '.join(map(str, r)) + r' \\' for r in rows],
        r'\bottomrule', r'\end{tabular}', r'\end{table}'
    ])


def tables():
    asym = load('asymmetry_summary.json')['method_summaries']
    names = {'separate': 'Coherent separate', 'shrink': 'Positive-part shrinkage',
             'selected': 'Selected pooling', 'oracle_pool': 'Oracle pooled',
             'oracle_basis': 'Oracle basis'}
    result = {}
    result['ASYMMETRY_TABLE'] = table('asymmetry',
        r'Confirmation against estimated pooling. Supported gains and harms pass a two-sided '
        r'exact McNemar test with Holm correction across 75 cells separately for each method. '
        r'All cells have 500 paired frames. Means weight cells equally; the equal-basis '
        r'column uses the 15 cells with $a=0$. Yield differences are in percentage points.',
        'lrrrr', r'Prior & Gains /75 & Harms /75 & Mean $\Delta Y$ & $a=0$ mean',
        [(names[r['method']], r['significant_gains'], r['significant_harms'],
          f"{100*r['mean_delta_yield']:+.3f}", f"{100*r['symmetric_mean_delta_yield']:+.3f}")
         for r in asym])
    controls = load('controls_summary.json')
    rows = []
    for r in controls:
        if r['experiment'] != 'fixed_total':
            continue
        rows.append((r['family'].capitalize(), r['sample'], 2000-r['sample'],
                     f"{100*r['pooled_yield']['mean']:.2f}",
                     f"{100*r['selected_yield']['mean']:.2f}", ci(r['delta'])))
    result['SAMPLE_TABLE'] = table('sample',
        r'Fixed total of 2000 sifted bits, at $Q=6\%$ and $a=0.5$. '
        r'Yields are percentages; differences are percentage points with exploratory '
        r'95\% intervals across five matrix libraries, each with 500 paired frames. '
        r'Policies are transferred without retuning on these outcomes.',
        'lrrrrl', r'Family & $S$ & $n$ & Pooled $Y$ & Selected $Y$ & $\Delta Y$ [95\% CI]', rows)
    rows = []
    modelnames = {'homogeneous': 'Homogeneous', 'gate_dependent': 'Gate-dependent',
                  'pauli': 'Pauli link', 'thermal': 'Thermal', 'readout': 'Readout'}
    for model in modelnames:
        for family in ['gallager', 'irregular']:
            take = lambda method: next(r for r in controls if r['experiment'] == 'physical'
                and r['model'] == model and r['family'] == family
                and r['setting'] == 'matched' and r['method'] == method)
            rows.append((modelnames[model], family.capitalize(), ci(take('selected')), ci(take('transition'))))
    result['PHYSICAL_TABLE'] = table('physical',
        r'B+ physical controls matched to 6\% overall QBER. Paired yield differences from '
        r'estimated pooling are in percentage points, with exploratory 95\% $t_4$ intervals '
        r'across five libraries (500 frames each). Both compared priors use the same fixed '
        r'code and disclosure as pooling. Selected uses basis; transition-aware also uses '
        r'Bob\textquotesingle s local bit.',
        'llll', r'Mechanism & Family & Selected [95\% CI] & Transition-aware [95\% CI]', rows)
    diagnostic = load('diagnostic_summary.json')
    methods = ['linear_four','gate_four','gate_two','calibrated_gate','Fiorini']
    methodnames = ['Linear, four settings','Gate, four settings','Gate, two settings',
                   'Calibrated inversion (BSC/gate)','Fiorini formula']
    rows = []
    for method, name in zip(methods, methodnames):
        rr = [r for r in diagnostic if r['experiment'] == 'grid' and r['method'] == method]
        assert len(rr) == 48
        rows.append((name, f"{100*sum(r['MAE'] for r in rr)/48:.3f}",
                     sum(bool(r['reliable']) for r in rr)))
    result['DIAGNOSTIC_TABLE'] = table('diagnostic',
        r'Nominal diagnostic grid, with equal weighting over 48 cells and 500 replicates '
        r'per cell. The descriptive criteria are MAE $\leq1$ pp and empirical coverage '
        r'of individual nominal 95\% intervals $\geq90\%$. Calibrated BSC and gate '
        r'inversions are algebraically identical in this ideal-Eve grid and share one row.',
        'lrr', r'Estimator & Mean MAE (pp) & Cells meeting both criteria /48', rows)
    labels = {'attack_ramp': 'Attack ramp', 'calibration_-0.1': r'Calibration $-10\%$',
              'calibration_-0.2': r'Calibration $-20\%$', 'calibration_0.1': r'Calibration $+10\%$',
              'calibration_0.2': r'Calibration $+20\%$', 'curvature_-0.02': r'Curvature $-0.02$',
              'curvature_0.02': r'Curvature $+0.02$', 'noise_ramp': 'Noise ramp',
              'unscaled_0.01': r'Unscaled $c=1\%$', 'unscaled_0.03': r'Unscaled $c=3\%$'}
    rows = []
    for stress, name in labels.items():
        row = [name]
        for method in ['gate_two', 'calibrated_gate']:
            r = next(r for r in diagnostic if r['stress'] == stress and r['method'] == method)
            row += [f"{100*r['MAE']:.3f}", f"{100*r['bias']:+.3f}", f"{100*r['coverage']:.1f}"]
        rows.append(row)
    result['STRESS_TABLE'] = table('stress',
        r'All ten diagnostic stress cells at $p=0.03,r=0.2,T=8000$, with 500 '
        r'replicates each. MAE and bias are in percentage points; coverage is a percentage. '
        r'For unscaled noise, the target is $e_c$ in Eq.~\eqref{eq:residual}. '
        r'Attack-ramp coverage uses the nominal reference despite the absence of a stationary target.',
        'lrrrrrr', r'& \multicolumn{3}{c}{Two-setting gate} & \multicolumn{3}{c}{Calibrated gate} \\'
        '\n' + r'Stress & MAE & Bias & Cover. & MAE & Bias & Cover.', rows)
    rule = load('frozen_rule.json')['lambda']
    rows = []
    for i, ss in enumerate([r'$S\leq100$', r'$S>100$']):
        for j, qq in enumerate([r'$\widehat p<0.04$', r'$0.04\leq\widehat p<0.08$', r'$\widehat p\geq0.08$']):
            rows.append((ss, qq, *[f'{v:g}' for v in rule[i][j]]))
    result['RULE_TABLE'] = table('rule',
        r'Frozen partial-pooling weights $\lambda$ selected using pilot data only. '
        r'Columns give the squared sample contrast $d^2/v$.', 'llrrrr',
        r'Sample & Pooled estimate & $<1$ & $[1,4)$ & $[4,9)$ & $\geq9$', rows)
    trans = load('transfer_summary.json')
    rows = []
    for family in ['gallager','irregular']:
        for n in [1000,4000,16000]:
            for condition, (a,s) in enumerate([(0,50),(.25,200),(.75,1000)]):
                take = lambda method: next(r for r in trans if r['family'] == family and
                    r['n'] == n and r['condition'] == condition and r['method'] == method)
                rows.append((family.capitalize(), n, f'{a:g}', s,
                             ci(take('selected')), ci(take('adaptive_pipeline'))))
    result['TRANSFER_TABLE'] = table('transferall',
        r'All transfer conditions at 6\% QBER. Paired yield differences in percentage '
        r'points, with exploratory 95\% $t_4$ intervals across five matrix libraries. '
        r'Fixed-code comparisons hold code and disclosure constant; pipeline comparisons '
        r'include the separately trained rate selectors.', 'lrrrll',
        r'Family & $n$ & $a$ & $S$ & Fixed code [95\% CI] & Pipeline [95\% CI]', rows).replace(r'\begin{table}[!htbp]', r'\begin{table}[htbp]')
    information=json.loads((HERE/'channel-information.json').read_text())
    rows=[(modelnames[r['model']],f"{r['H_A_given_B']:.5f}",f"{r['H_A_given_BZ']:.5f}",
           f"{r['G']:.5f}",f"{r['bit_prior_logloss_gain']:.5f}")
          for r in information['physical'] if r['setting']=='matched']
    result['INFORMATION_TABLE']=table('information',
        r'Channel information for the five B+ controls matched to $Q=6\%$. Values are '
        r'bits per retained bit, computed from the stored transition probabilities with '
        r'uniform input bits and bases. $G=I(A;Z\mid B)$ is the basis benefit relative to '
        r'a decoder already using $B$; $D_B=I(U;B\mid Z)$ is the extra ideal error-prior '
        r'log-loss reduction from Bob\textquotesingle s bit after conditioning on basis. '
        r'These are distinct quantities and are not added together.',
        'lrrrr',r'Mechanism & $H(A\mid B)$ & $H(A\mid B,Z)$ & $G$ & $D_B$',rows)
    rows=[(r['family'].capitalize(),f"{r['pooled_row_fraction']:.5f}",
           f"{r['selected_row_fraction']:.5f}",f"{r['disclosure_pp']:+.3f}",
           f"{r['recovery_pp']:+.3f}",f"{r['delta_pp']:+.3f}") for r in information['pipeline']]
    result['YIELD_ACCOUNTING_TABLE']=table('yieldaccounting',
        r'Accounting for strong-asymmetry transfer at $n=16000,S=1000,Q=6\%,a=0.75$. '
        r'Means use the same 2500 frames per family as the transfer comparison. '
        r'$D$ is the yield change due to syndrome length and $R$ the change in '
        r'failure-related loss, so $\Delta Y=D+R$. The pooled and conditional entropy '
        r'benchmarks are 0.32744 and 0.29850 bits per retained bit. '
        r'$D$, $R$, and $\Delta Y$ are percentage points.',
        'lrrrrr',r'Family & $\overline m_{\rm P}/n$ & $\overline m_{\rm S}/n$ & $D$ & $R$ & $\Delta Y$',rows)
    result['COHORT_TABLE']=table('cohorts',
        r'Experimental map. Frame counts denote distinct inputs before reuse across '
        r'priors, codes, or decoders. A diagnostic replicate block evaluates several '
        r'fully budgeted acquisition strategies and produces no key. The 450 symmetric '
        r'reference-calibration frames are additional. Detailed grids and reuse rules '
        r'are given in the methods and supplement. Target profiles additionally reuse '
        r'2500 held-out-validation fitting records, without new acquisitions.',
        'lllr',r'Cohort & Resource & Outcome & Inputs',[
        ('Circuit',r'$n=1000,S=200$','FER and yield','68,000'),
        ('Pooling/rate development','Varied $n,S$','Frozen rules','25,800'),
        ('Pooling confirmation',r'$n=1000$, varied $S$','Fixed-code yield','37,500'),
        ('Code transfer',r'$n=1000,4000,16000$','Prior/policy yield','22,500'),
        ('Sample allocation',r'$n+S=2000$','Total yield','12,500'),
        ('Physical controls',r'$n=1000,S=200$','Conditional priors','25,000'),
        ('Diagnostic reference/stress','Varied $T$','Residual and coverage','45,000'),
        ('Basis/range validation','Varied $T$; held-out setting','Residual and prediction','31,000'),
        ('Full-data/circuit validation',r'$T=8000$ per strategy','Residual and coverage','6,500'),
        ('Budget/interception validation','Varied $T,r$','Precision contrasts','40,500')])
    review=json.loads((HERE/'review-analysis.json').read_text())
    rows=[]
    for r in review['absolute_transfer']:
        if r['n']==16000 and r['asymmetry'] in [0,.75] and r['method'].startswith('pipeline'):
            rows.append((r['family'].capitalize(),f"{r['asymmetry']:g}",
                'P' if r['method']=='pipeline_pooled' else 'S',f"{r['failures']}: {100*r['FER']:.2f}",
                f"{100*r['yield_mean']:.2f}",f"{r['row_fraction']:.5f}",f"{r['f_EC']:.3f}",f"{r['batch_ms_per_frame']:.2f}"))
    result['ABSOLUTE_TRANSFER_TABLE']=table('absolutetransfer',
        r'Absolute pipeline outcomes at $n=16000,Q=6\%$. Each row pools 2500 frames '
        r'across five libraries. P/S denote pooled/selected priors and their own learned '
        r'selectors; $S=50$ at $a=0$ and $S=1000$ at $a=0.75$. FER entries give failure '
        r'counts and percentages; $Y$ is a percentage. Timing is retained parallel-batch '
        r'wall time divided by frames, in milliseconds; it is not single-frame latency '
        r'or a controlled hardware benchmark. Full absolute outcomes and uncertainty '
        r'for every condition accompany the supplement.',
        'lrrlrrrr',r'Family & $a$ & Prior & FER & $Y$ & $\overline m/n$ & $f_{\rm EC}$ & ms/frame',rows)
    validation=json.loads((SUP/'diagnostic_validation/results/summary.json').read_text())
    method_names={'basis_single':'Basis, constrained fit','basis_inverse':'Basis, direct inverse',
        'aggregate_two_cap2.5':r'Two settings, $f_{\max}=2.5$',
        'aggregate_four_cap2.5':r'Four settings, $f_{\max}=2.5$',
        'aggregate_two_cap4':r'Two settings, $f_{\max}=4$',
        'aggregate_four_cap4':r'Four settings, $f_{\max}=4$',
        'aggregate_two_cap8':r'Two settings, $f_{\max}=8$',
        'aggregate_four_cap8':r'Four settings, $f_{\max}=8$',
        'calibrated':'Calibrated inversion'}
    rows=[]
    for method,name in method_names.items():
        take=lambda stage:next(r for r in validation['grid_aggregate'] if r['stage']==stage and r['budget'] is None and r['method']==method)
        a,b=take('nominal'),take('off_grid')
        fmt=lambda r:f"{r['MAE_pp']:.3f} [{r['MAE_MC_CI_pp'][0]:.3f}, {r['MAE_MC_CI_pp'][1]:.3f}]"
        rows.append((name,fmt(a),fmt(b),f"{100*a['average_coverage']:.1f}/{100*b['average_coverage']:.1f}"))
    result['VALIDATION_TABLE']=table('validation',
        r'Diagnostic validation, with 500 replicates per cell. Nominal means weight 48 '
        r'cells equally; interpolation means weight nine interior cells at $T=8000$. MAE and its '
        r'95\% Monte Carlo (MC) interval are in percentage points (pp). Coverage is the average '
        r'percentage of individual nominal 95\% intervals covering the target, shown '
        r'as nominal/interpolation. The two basis estimators reuse the same counts. '
        r'Intervals condition on the frozen grids and do not establish equivalence.',
        'lllr',r'Estimation/acquisition & Nominal MAE [MC CI] & Interpolation MAE [MC CI] & Cover. (\%)',rows)
    rows=[]
    names={'basis_single':'Basis','aggregate_two_cap4':'Two settings',
           'aggregate_four_cap4':'Four, equal noise','independent_devices':'Four, separate noise'}
    scenarios={'equal':'Equal noise','off_grid':'Interior','alice_low':'Alice lower','bob_low':'Bob lower','curvature':'Curvature'}
    for r in validation['held_out']:
        rows.append((scenarios[r['scenario']],names[r['method']],f"{r['MAE_pp']:.3f}",
            f"{100*r['coverage']:.1f}",f"{r['mean_interval_width_pp']:.2f}",f"{100*r['prediction_coverage']:.1f}",f"{100*r['noise_bound_fraction']:.1f}"))
    result['HOLDOUT_TABLE']=table('holdout',
        r'Signed-WLS fits and local-Wald intervals in held-out-setting validation: '
        r'8000 fitting plus 2000 validation signals, '
        r'500 replicates per scenario. Scaling is capped at 4; validation uses the '
        r'unfitted setting $f=2.3$. Residual MAE and mean interval width are in pp; '
        r'coverage and the percentage of fits at a device-noise boundary are percentages. '
        r'Four-setting model variants reuse identical acquisitions; target-specific '
        r'profile intervals on the same records appear in Table~\ref{tab:profiles}. The supplement '
        r'includes Wilson Monte Carlo intervals and complete fitted records.',
        'llrrrrr',r'Scenario & Fit & MAE & Residual cover. & Width & Prediction cover. & Bound',rows)
    observation=json.loads((SUP/'observation_validation/results/summary.json').read_text())
    rows=[]
    for circuit in ['PRIMARY','A','B+']:
        for scenario,pair in [('equal',r'$(.03,.03)$'),('alice_low',r'$(.01,.05)$'),('bob_low',r'$(.05,.01)$')]:
            row=[circuit,pair]
            for method in ['subset','basis','full','scaled']:
                rr=[r for r in observation['cross_circuit'] if r['circuit']==circuit and
                    r['scenario']==scenario and r['method']==method and r['readout']==0]
                if not rr:row.extend(['--','--'])
                elif not rr[0]['point_identified']:
                    row.extend([r'\shortstack{Not\\identified}',f"{rr[0]['coverage_pct']:.1f}"])
                else:row.extend([f"{rr[0]['MAE_pp']:.3f}",f"{rr[0]['coverage_pct']:.1f}"])
            rows.append(row)
    result['CROSS_CIRCUIT_TABLE']=table('crosscircuit',
        r'Cross-circuit diagnostics: 8000 transmitted signals, $r=0.2$, and 500 '
        r'replicates per cell. Each estimator has separate columns for residual MAE '
        r'in percentage points (pp) and interval coverage (\%). '
        r'The first three estimators share one operating acquisition; scaling uses '
        r'an independent, equally budgeted four-setting acquisition. Basis assumes '
        r'equal device noise; full and scaling fit the devices independently. '
        r'For nonidentified operating fits, coverage concerns the residual set; '
        r'no point-estimator MAE is reported. '
        r'Subset applies only to PRIMARY. Coverage has Monte Carlo uncertainty; '
        r'cell-level intervals accompany the records.',
        'llrrrrlrrr',r'& & \multicolumn{2}{c}{Subset} & \multicolumn{2}{c}{Basis} & '
        r'\multicolumn{2}{c}{Full} & \multicolumn{2}{c}{Scaling} \\'+'\n'+
        r'Circuit & $(p_A,p_B)$ & MAE & Cover. & MAE & Cover. & MAE & Cover. & MAE & Cover.',rows
        ).replace(r'\small\setlength{\tabcolsep}{4pt}',r'\small\setlength{\tabcolsep}{3pt}')
    rows=[]
    names={'equal':'Equal','off_grid':'Interior','alice_low':'Alice lower','bob_low':'Bob lower','curvature':'Curved'}
    import math
    for scenario in names:
        r=next(x for x in observation['retained_profiles'] if x['scenario']==scenario)
        old=r['WLS_Wald'];devices=r['device_profiles']
        rows.append((names[scenario],f"{old['width_median_pp']:.2f}/{old['width_p95_pp']:.1f}",
            f"{r['width_median_pp']:.2f}/{r['width_p95_pp']:.2f}",f"{r['coverage_pct']:.1f}",
            f"{devices['pA']['width_median_pp']:.2f}/{devices['pB']['width_median_pp']:.2f}",
            f"{math.log10(r['condition_median']):.2f}",f"{r['noise_boundary_pct']:.1f}"))
    result['PROFILE_TABLE']=table('profiles',
        r'Target versus nuisance uncertainty on the same 500 four-setting records '
        r'per scenario. Residual widths give median/95th percentile in pp for signed-WLS '
        r'Wald and physical binomial profile intervals; coverage concerns the latter. '
        r'Device columns give median profile widths in pp for $p_A/p_B$. '
        r'$\log_{10}\kappa$ is the log of the median information-matrix condition '
        r'number in $(p_A,p_B,r)$ coordinates; boundary is the percentage of '
        r'likelihood fits at a device-noise boundary. The loss and bounds change '
        r'as well as interval construction.',
        'lllllll',r'Scenario & Wald width & Profile width & Cover. & Device widths & $\log_{10}\kappa$ & Bound (\%)',rows)
    precision=json.loads((SUP/'precision_validation/results/summary.json').read_text())
    for circuit,key,label in [('PRIMARY','PRECISION_PRIMARY_TABLE','precisionprimary'),
                              ('A','PRECISION_A_TABLE','precisiona')]:
        rows=[]
        for c in precision['comparisons']:
            if c['circuit']!=circuit:continue
            rr={r['method']:r for r in precision['outcomes'] if r['cell']==c['cell']}
            interval=lambda field:'['+', '.join(f'{v:+.3f}' for v in c[field])+']'
            rows.append((c['budget'],f"{c['r']:.2f}",f"$({c['pA']:.2f},{c['pB']:.2f})$",
                f"{rr['full']['MAE_pp']:.3f}",f"{rr['scaled']['MAE_pp']:.3f}",
                f"{c['delta_MAE_pp']:+.3f} "+interval('mc95_pp'),interval('simultaneous95_pp')))
        result[key]=table(label,
            ('Scheme A' if circuit=='A' else circuit)+r' precision grid, with 500 fresh replicates per cell and strategy. '
            r'All errors and differences are in pp. $\Delta$ is full-data minus scaling MAE; '
            r'positive favors scaling. Pointwise 95\% Monte Carlo intervals use independent '
            r'acquisition variances; simultaneous intervals apply Bonferroni correction '
            r'across all 54 PRIMARY/A comparisons. The design and fitter are fixed.',
            'rrlrrll',r'$T$ & $r$ & $(p_A,p_B)$ & Full & Scaling & $\Delta$ [pointwise] & Simultaneous',rows)
    rows=[]
    for r in precision['outcomes']:
        if r['circuit']!='B+':continue
        rows.append((r['budget'],f"{r['r']:.2f}",f"$({r['pA']:.2f},{r['pB']:.2f})$",
            f"{r['MAE_pp']:.3f}",f"{r['coverage_pct']:.1f}",
            f"{r['width_median_pp']:.2f}/{r['width_p95_pp']:.2f}"))
    result['PRECISION_BPLUS_TABLE']=table('precisionbplus',
        r'B+ scaling precision, with 500 fresh replicates per cell. MAE and profile '
        r'widths are in pp; coverage is a percentage. Width entries give median/95th '
        r'percentile. No operating point-estimator comparison is reported because '
        r'the residual is not identified at a single setting. Coverage has Monte '
        r'Carlo uncertainty; Wilson intervals accompany the complete records.',
        'rrlrrl',r'$T$ & $r$ & $(p_A,p_B)$ & MAE & Coverage & Profile width',rows)
    # The compact main-text synthesis is calculated from the same complete records.
    study=json.loads((SUP/'study/results/summary.json').read_text())
    matched=next(r for r in study['main'] if r['scheme']=='A' and r['target_qber']==.1
        and r['decoder']=='BP+OSD' and r['comparison']=='same_aware_code')
    selected=next(r for r in asym if r['method']=='selected')
    rows=[('A, matched code',
           f"{matched['failures_pooled']} $\\to$ {matched['failures_aware']} failures /2000",
           r'$m=650$; both zero at $m=700$'),
          ('Frozen pooling',f"{selected['significant_gains']} gains; {selected['significant_harms']} harms /75",
           r'Holm-supported; repeated grid')]
    for family in ['gallager','irregular']:
        take=lambda method:next(r for r in trans if r['family']==family and r['n']==16000
            and r['condition']==2 and r['method']==method)
        rows.extend([(family.capitalize()+', fixed',ci(take('selected')),r'Common code and disclosure'),
                     (family.capitalize()+', policy',ci(take('adaptive_pipeline')),r'Prior and rate selector together')])
    result['RECOVERY_OVERVIEW_TABLE']=table('recoveryoverview',
        r'Recovery evidence at three levels. A uses matched-code BP+OSD at 10\% QBER. '
        r'Frozen pooling uses 75 confirmation conditions, 500 paired frames each, with '
        r'Holm correction across those conditions. Long-block transfer rows use '
        r'$n=16000,Q=6\%,a=0.75,S=1000$; entries are paired yield differences '
        r'in percentage points (pp), with exploratory 95\% $t_4$ intervals across '
        r'five unseen matrix libraries. The appendices retain all operating points.',
        'lll',r'Comparison & Result & Interpretation',rows)
    # Display names are independent of stable identifiers in the numerical records.
    return {key:value.replace('PRIMARY','Standard') for key,value in result.items()}


def fill(source, check=False):
    texts = {p: p.read_text() for p in [source/'main.tex', source/'appendix.tex']}
    for key, value in tables().items():
        start, end = '% BEGIN GENERATED '+key, '% END GENERATED '+key
        replacement = start+'\n'+value+'\n'+end
        hits = 0
        for p, text in texts.items():
            token = '@@'+key+'@@'
            if token in text:
                texts[p] = text.replace(token, replacement); hits += 1
            elif start in text:
                texts[p] = re.sub(re.escape(start)+r'.*?'+re.escape(end),
                                  lambda _: replacement, text, flags=re.S); hits += 1
        if hits == 0:
            continue
        if hits != 1:
            raise RuntimeError(f'{key}: expected exactly one table location, got {hits}')
    for p, text in texts.items():
        if re.search(r'@@[A-Z_]+@@', text):
            raise RuntimeError(f'Unfilled placeholder in {p}')
        if check:
            if p.read_text() != text:
                raise RuntimeError(f'Stale generated table in {p}')
        else:
            p.write_text(text)


def plots(source):
    import tempfile
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'bb84-final-mpl'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({'font.size':9, 'axes.titlesize':10, 'axes.labelsize':9,
        'xtick.labelsize':8, 'ytick.labelsize':8, 'legend.fontsize':8,
        'pdf.fonttype':42, 'ps.fonttype':42, 'axes.spines.top':False,
        'axes.spines.right':False, 'savefig.facecolor':'white'})
    output = source/'images'; output.mkdir(exist_ok=True)
    def save(fig, name):
        fig.savefig(output/(name+'.pdf'), bbox_inches='tight',
                    metadata={'Creator':'BB84 finalization: verified numerical records',
                              'CreationDate':None, 'ModDate':None})
        plt.close(fig)
    with (SUP/'extensions/results/asymmetry_results.csv').open() as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(3,3,figsize=(7.6,5.6),layout='constrained')
    for i, (method, title) in enumerate([('separate','Coherent separate'),('shrink','Shrinkage'),('selected','Selected pooling')]):
        for j,q in enumerate([.02,.06,.1]):
            ax = axes[i,j]; mat = np.zeros((5,5)); stars=[]
            for ai,a in enumerate([0,.1,.25,.5,.75]):
                for si,s in enumerate([50,100,200,500,1000]):
                    r=next(r for r in rows if r['method']==method and float(r['qber'])==q and
                           float(r['asymmetry'])==a and int(r['sample'])==s)
                    mat[ai,si]=100*float(r['delta_yield'])
                    if float(r['p_holm']) < .05: stars.append((si,ai))
            im=ax.imshow(mat,origin='lower',aspect='auto',cmap='RdBu',norm=TwoSlopeNorm(vmin=-8,vcenter=0,vmax=8))
            if stars: ax.scatter(*zip(*stars),s=12,c='black',marker='o',edgecolors='white',linewidths=.35)
            ax.set_xticks(range(5),[50,100,200,500,1000]);ax.set_yticks(range(5),[0,.1,.25,.5,.75])
            if i==0:ax.set_title(f'QBER {100*q:.0f}%')
            if j==0:ax.set_ylabel(title+'\nAsymmetry a')
            if i==2:ax.set_xlabel('Disclosed sample S')
    fig.colorbar(im,ax=axes,shrink=.78,label='Yield difference (pp)',pad=.02)
    save(fig,'asymmetry_map')
    trans=load('transfer_summary.json')
    fig, axes=plt.subplots(1,2,figsize=(7.6,5.1),sharey=True,layout='constrained')
    colors=['#64748b','#d18a23','#007d80'];markers=['o','s','D']
    labels=[];ticks=[]
    for fi,family in enumerate(['gallager','irregular']):
        for ni,n in enumerate([1000,4000,16000]):
            group=fi*3+ni;ticks.append(group*4+1);labels.append(f'{family.capitalize()}\nn = {n:,}')
            for cond in range(3):
                y=group*4+cond
                for ax,method in zip(axes,['selected','adaptive_pipeline']):
                    r=next(r for r in trans if r['family']==family and r['n']==n and
                           r['condition']==cond and r['method']==method)
                    value=100*r['mean'];lo,hi=np.array(r['ci'])*100
                    ax.errorbar(value,y,xerr=[[max(0,value-lo)],[max(0,hi-value)]],
                                fmt=markers[cond],color=colors[cond],markersize=4,capsize=2,lw=1)
    for ax,title in zip(axes,['Same code and disclosure','Separate rate policies']):
        ax.axvline(0,color='.45',lw=.8);ax.set_title(title);ax.set_xlabel('Yield difference (pp)')
        ax.set_yticks(ticks,labels);ax.grid(axis='x',alpha=.16)
        for boundary in [3,7,11,15,19]:ax.axhline(boundary,color='.9',lw=.7)
    axes[0].invert_yaxis();axes[0].set_xlim(-1.25,1.25);axes[1].set_xlim(-1.35,4.7)
    from matplotlib.lines import Line2D
    handles=[Line2D([0],[0],marker=markers[i],color=colors[i],lw=0,
        label=x) for i,x in enumerate(['a=0, S=50','a=0.25, S=200','a=0.75, S=1000'])]
    fig.legend(handles=handles,loc='outside lower center',ncol=3,frameon=False)
    save(fig,'transfer_comparison')
    diag=json.loads((SUP/'diagnostic_validation/results/summary.json').read_text())['grid_aggregate']
    fig, axes=plt.subplots(1,2,figsize=(7.6,3.2),layout='constrained')
    for ax,stage,title in zip(axes,['nominal','off_grid'],['Nominal grid','Interior interpolation, 8,000 signals']):
        take=lambda method:next(r for r in diag if r['stage']==stage and r['budget'] is None and r['method']==method)
        for settings,name,color,marker in [('two','Two settings','#007d80','o'),('four','Four settings','#72569b','s')]:
            rr=[take(f'aggregate_{settings}_cap{cap}') for cap in ['2.5','4','8']]
            means=np.array([r['MAE_pp'] for r in rr]);low=np.array([r['MAE_MC_CI_pp'][0] for r in rr]);high=np.array([r['MAE_MC_CI_pp'][1] for r in rr])
            ax.errorbar([2.5,4,8],means,yerr=[means-low,high-means],label=name,color=color,marker=marker,lw=1.5,ms=4,capsize=3)
        for method,name,color,style in [('basis_single','Basis fit','#222222','-'),
                ('basis_inverse','Basis inverse','#64748b',':'),('calibrated','Calibrated','#c17b18','--')]:
            ax.axhline(take(method)['MAE_pp'],label=name,color=color,ls=style,lw=1.3)
        ax.set_xticks([2.5,4,8]);ax.set_xlabel('Maximum noise scale');ax.grid(alpha=.16)
        ax.set_ylabel('Mean absolute error (pp)');ax.set_ylim(.39,.91);ax.set_title(title)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=5,frameon=False)
    save(fig,'diagnostic_comparison')
    from channel_information import ideal_response,basis_response
    information=json.loads((HERE/'channel-information.json').read_text())
    example=information['aggregate_ambiguity']
    fig,axes=plt.subplots(1,2,figsize=(7.6,2.9),layout='constrained')
    cases=[(example['attacked_p'],example['attack_fraction'],'#007d80',
            r'Device noise $p=0.015$, interception $r=0.10$'),
           (example['noise_only_p'],0.,'#b45a6a',
            f"Device noise $p={example['noise_only_p']:.3f}$, no interception")]
    for p,r,color,label in cases:
        axes[0].plot([0,1],100*basis_response(p,r),'o-',color=color,label=label,lw=1.7,ms=5)
        f=np.linspace(1,8,200);axes[1].plot(f,100*ideal_response(f,p,r),color=color,lw=1.7)
        f=np.linspace(0,1,80);axes[1].plot(f,100*ideal_response(f,p,r),color=color,ls='--',lw=1.4)
        axes[1].plot(0,25*r,'o',mfc='white',mec=color,ms=5,clip_on=False)
    axes[0].axhline(100*example['common_operating_qber'],color='.45',ls=':',lw=1)
    axes[0].set_xticks([0,1],['Rectilinear','Diagonal']);axes[0].set_xlim(-.15,1.15)
    axes[0].set_ylabel('Basis QBER (%)');axes[0].set_xlabel('Operating basis label')
    axes[1].axvline(1,color='.55',ls=':',lw=1)
    axes[1].plot(1,100*example['common_operating_qber'],'o',color='#222222',ms=4)
    axes[1].set_xlabel('Legitimate-device noise scale f');axes[1].set_ylabel('Aggregate QBER (%)')
    axes[1].set_xlim(0,8);axes[1].set_xticks([0,1,2,4,6,8])
    for ax in axes:ax.grid(alpha=.15);ax.set_ylim(bottom=0)
    axes[0].set_title('Basis labels already distinguish the states')
    axes[1].set_title('Scaling also separates their responses')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=1,frameon=False)
    save(fig,'aggregate_ambiguity')
    precision=json.loads((SUP/'precision_validation/results/summary.json').read_text())
    fig,axes=plt.subplots(2,3,figsize=(7.6,4.5),sharex=True,sharey=True,layout='constrained')
    colors=['#b45a6a','#007d80','#72569b'];markers=['o','s','D']
    for i,circuit in enumerate(['PRIMARY','A']):
        for j,(pA,pB) in enumerate([(.03,.03),(.01,.05),(.05,.01)]):
            ax=axes[i,j]
            for k,r in enumerate([.02,.1,.2]):
                rows=sorted([c for c in precision['comparisons'] if c['circuit']==circuit
                    and c['pA']==pA and c['pB']==pB and c['r']==r],key=lambda c:c['budget'])
                xs=np.arange(3)+(k-1)*.09
                ys=np.array([c['delta_MAE_pp'] for c in rows])
                ax.plot(xs,ys,color=colors[k],lw=.8,alpha=.75)
                for x,y,c in zip(xs,ys,rows):
                    lo,hi=c['mc95_pp'];slo,shi=c['simultaneous95_pp']
                    ax.errorbar(x,y,yerr=[[y-lo],[hi-y]],fmt=markers[k],color=colors[k],
                        mfc=colors[k] if slo>0 or shi<0 else 'white',ms=4,capsize=2,lw=.9)
            ax.axhline(0,color='.4',lw=.8,ls='--');ax.grid(axis='y',alpha=.15)
            ax.set_xticks(range(3),['2,000','8,000','32,000']);ax.set_xlim(-.24,2.24)
            if i==0:ax.set_title(f'$(p_A,p_B)=({pA:.2f},{pB:.2f})$')
            else:ax.set_xlabel('Transmitted signals T')
            if j==0:
                ax.set_ylabel(('Standard' if circuit=='PRIMARY' else 'A')+'\nFull − scaling MAE (pp)')
            if j==2:
                ax.text(.97,.95,'↑ favors scaling',transform=ax.transAxes,ha='right',va='top',fontsize=8)
                ax.text(.97,.05,'↓ favors full',transform=ax.transAxes,ha='right',va='bottom',fontsize=8)
    handles=[Line2D([0],[0],marker=markers[k],color=colors[k],lw=.8,
        label=f'Interception r={r:.2f}') for k,r in enumerate([.02,.1,.2])]
    fig.legend(handles=handles,loc='outside lower center',ncol=3,frameon=False)
    save(fig,'precision_grid')

    # A visual guide links the questions without implying a joint experimental pipeline.
    from matplotlib.patches import FancyBboxPatch
    fig,ax=plt.subplots(figsize=(7.6,6.1));ax.set_axis_off();ax.set_xlim(0,1);ax.set_ylim(0,1)
    def box(x,y,w,h,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.008,rounding_size=.012',
            facecolor=color,edgecolor='#cbd5e1',lw=.8))
    box(.01,.86,.98,.12,'#eaf1f5')
    ax.text(.5,.935,'Disclosed BB84 observations',ha='center',va='center',fontsize=15,weight='bold')
    ax.text(.5,.89,'Bits, basis labels and outcomes contain more than aggregate QBER.',ha='center',fontsize=10)
    cards=[('1  IDENTIFY','#e6f2f0','What do the observations\ndetermine?',
        'Gate-path counts\nExact response laws\nRank and row-space tests',
        'Standard and A identify\nthe residual at one setting.\nB and B+ do not.',
        'Additional observations\ncan restore identification.'),
        ('2  ESTIMATE','#f0edf7','How precisely can it\nbe learned?',
        'Matched scaling ranges\nFull operating observations\n81-condition precision grid',
        'Full data favor Standard.\nScaling usually favors A.\nBudget and attack level matter.',
        'Compare complete acquisitions\nat the same signal budget.'),
        ('3  USE','#faf1e3','Does estimated structure\nimprove recovery?',
        'Circuit and pooling controls\nNew matrix libraries\nFixed-code vs. rate policies',
        'Pooling limits estimation loss.\nLong-block gains arise mainly\nthrough rate selection.',
        'Charge estimation, disclosure\nand frame-loss costs.')]
    for i,(title,color,question,design,finding,lesson) in enumerate(cards):
        x=.01+i*.335;box(x,.15,.31,.65,color);cx=x+.155
        ax.text(cx,.755,title,ha='center',fontsize=11,weight='bold')
        ax.text(cx,.685,question,ha='center',va='center',fontsize=10,weight='bold',linespacing=1.4)
        ax.text(cx,.57,design,ha='center',va='center',fontsize=9.5,linespacing=1.55)
        ax.plot([x+.025,x+.285],[.48,.48],color='#cbd5e1',lw=.8)
        ax.text(cx,.405,finding,ha='center',va='center',fontsize=9.5,linespacing=1.55)
        ax.text(cx,.245,lesson,ha='center',va='center',fontsize=9.5,weight='bold',linespacing=1.5)
    ax.text(.5,.075,'Retain the labels  →  establish identification  →  measure precision  →  test utility',
        ha='center',fontsize=10,weight='bold')
    fig.subplots_adjust(left=.01,right=.99,bottom=.01,top=.99);save(fig,'study_overview')
    # Recreate the inherited circuit figures with reader-facing display names.
    import importlib.util
    import sys
    from matplotlib.text import Text
    sys.path.insert(0,str(SUP/'study'))
    spec=importlib.util.spec_from_file_location('inherited_circuit_plots',SUP/'study/plot_v2.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def circuit_save(fig,name):
        for item in fig.findobj(match=Text):
            item.set_text(item.get_text().replace('PRIMARY','Standard'))
        save(fig,Path(name).stem)
    module.save=circuit_save
    module.circuits();module.exposure();module.curves()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path,default=ROOT/'overleaf')
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args();fill(args.source_dir,args.check)
    if not args.check:plots(args.source_dir)
    print('Verified generated tables.' if args.check else 'Generated active manuscript tables and figures.')


if __name__=='__main__':main()
