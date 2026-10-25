"""Write a reviewable research report directly from canonical extension results."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from common import OUT,P
from inventory import check_inventory, record_fingerprint


def load(name):return json.loads((OUT/name).read_text())
def interval(value):return f'{100*value["mean"]:+.2f} [{100*value["ci"][0]:+.2f}, {100*value["ci"][1]:+.2f}]'
def percent(value):return f'{100*value:.2f}'


def validated(name):
    """Use the rerun audit when available, otherwise describe the record audit."""
    path=OUT/f'{name}_rerun.json'
    if not path.is_file():path=OUT/f'{name}.json'
    audit=json.loads(path.read_text())
    if audit.get('status')!='passed':raise RuntimeError(f'Audit has not passed: {path}')
    return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,help='Report path; default notes/extension-report.md.')
    args=parser.parse_args()
    inventory=check_inventory()
    asym=load('asymmetry_summary.json');transfer=load('transfer_summary.json')
    controls=load('controls_summary.json');diag=load('diagnostic_summary.json')
    validation=validated('extension_validation');initial=validated('validation')
    fingerprint=record_fingerprint()
    provenance=load('analysis_provenance.json')
    for evidence in [validation,initial,asym,provenance]:
        if evidence.get('record_fingerprint')!=fingerprint:
            raise RuntimeError('Analysis or audit is stale; regenerate it against the current records.')
    if len(transfer)!=90 or len(controls)!=90 or len(asym['method_summaries'])!=5:
        raise RuntimeError('Summary tables do not cover the declared experiment.')
    if initial['records']['cells']!=150 or validation['records']['files']!=1262:
        raise RuntimeError('Record audits do not cover the complete experiment.')
    ident=load('diagnostic_identifiability.json');design=load('frozen_diagnostic_design.json')
    diagnostic_reference=load('diagnostic_reference_audit.json')
    if diagnostic_reference.get('status')!='passed' or diagnostic_reference['diagnostic_cells']!=90:
        raise RuntimeError('Independent diagnostic reference audit is incomplete.')
    totalframes=initial['records']['independent_input_frames']+validation['records']['unique_input_frames']
    outputs=initial['records']['decoder_outputs']+validation['records']['decoder_outputs']
    diagnostic_replicates=validation['diagnostics']['acquisition_replicates']
    if diagnostic_replicates!=inventory['diagnostic_acquisition_replicates']:
        raise RuntimeError('Diagnostic audit does not cover the complete inventory.')
    if len(diag)!=inventory['diagnostic_cells']*6:
        raise RuntimeError('Diagnostic summary does not cover every estimator/cell.')
    lines=[
        '# Basis-aware reconciliation: extension study',
        '',
        'This report summarises the pooling, transfer, physical-control and diagnostic studies.',
        '',
        f'The extension evaluates **{totalframes:,} independent input frames** and '
        f'**{outputs:,} decoder outputs**, plus {diagnostic_replicates:,} diagnostic acquisition replicates. '
        'Matching frame records across code families are counted once. The '
        f'{inventory["reference_calibration_frames"]:,} separate '
        'reference-calibration frames are additional. Every prespecified condition is retained, '
        'including degradation and zero-failure cells.',
        '',
        '## The basis-asymmetry result',
        '',
        'The binary-channel experiment varies overall QBER (2%, 6%, 10%), basis asymmetry '
        '(0, .1, .25, .5, .75) and sacrificed sample (50, 100, 200, 500, 1000), while '
        'decoding 1000 bits. Its 15,000 pilot frames train a sample-only partial-pooling '
        'rule. The rule and replication counts are frozen before 37,500 confirmation '
        'frames are generated on unseen matrix seed29. Neither true asymmetry, true '
        'probabilities nor decoded-block truth selects its pooling weight.',
        'All extension decoders use layered sum-product BP with 100 sweeps and the same '
        'clipping and stopping rules. The stronger OSD retry evaluated in the frozen '
        'paper is not included in these new runs, so transfer to that decoder is not established.',
        '',
        '| Prior | Supported gains /75 | Supported harms /75 | Mean yield change (pp) | Equal-basis mean change (pp) |',
        '| --- | ---: | ---: | ---: | ---: |']
    for r in asym['method_summaries']:
        lines.append(f'| {r["method"]} | {r["significant_gains"]} | {r["significant_harms"]} | {100*r["mean_delta_yield"]:+.3f} | {100*r["symmetric_mean_delta_yield"]:+.3f} |')
    lines += ['',
        'Supported means a two-sided exact paired McNemar test with Holm correction over '
        '75 cells separately for each comparator. Zero supported harms does not establish '
        'zero harm. Grid averages weight the design cells equally; they are not a deployment '
        'average. Wilson FER intervals and paired-bootstrap yield intervals are conditional '
        'on the fixed matrix in each QBER cell.',
        '',
        'The information opportunity is G = H(A|B) - H(A|B,Z). For basis-conditioned BSCs '
        'this reduces to the binary-entropy Jensen gap. The physical bit-asymmetric controls '
        'use the actual transition probabilities. Neither nG nor population signal-to-noise '
        'appears in the deployed rule. Their scatter plots are descriptive evidence, not '
        'a universal finite-length performance law.',
        '',
        '[All 450 method/cell rows](../supplementary/extensions/results/asymmetry_results.csv) '
        'and [the benefit/harm heatmaps](../supplementary/extensions/results/figures/asymmetry_heatmaps.png).',
        '',
        '## Transfer to unseen matrices and longer blocks',
        '',
        'Both code families use development library seeds101,103 and unseen test seeds '
        '211,223,227,229,233. The two pipelines receive the same code pool, sample features, '
        'ridge-logistic training procedure and development budget. Each chooses its row '
        'count from observable samples before decoding a test frame. A separate fixed-code '
        'comparison holds matrix and disclosure identical. Pilot variance selected 500 '
        'frames per matrix in each transfer condition.',
        '',
        '| Family | n | Fixed code: a=0,s=50 | Fixed code: a=.25,s=200 | Fixed code: a=.75,s=1000 |',
        '| --- | ---: | --- | --- | --- |']
    for family in ['gallager','irregular']:
        for n in P['codes']['lengths']:
            def tr(ci,method):return next(r for r in transfer if r['family']==family and r['n']==n and r['condition']==ci and r['method']==method)
            cells=[interval(tr(ci,'selected')) for ci in range(3)]
            lines.append(f'| {family} | {n} | '+ ' | '.join(cells)+' |')
    lines += ['', '| Family | n | Pipeline: a=0,s=50 | Pipeline: a=.25,s=200 | Pipeline: a=.75,s=1000 |',
              '| --- | ---: | --- | --- | --- |']
    for family in ['gallager','irregular']:
        for n in P['codes']['lengths']:
            rr=[r for r in transfer if r['family']==family and r['n']==n and r['method']=='adaptive_pipeline']
            lines.append(f'| {family} | {n} | '+' | '.join(interval(next(r for r in rr if r['condition']==ci)) for ci in range(3))+' |')
    fixed_rule=[r for r in transfer if r['method']=='selected']
    lines += ['', f'Across the 18 fixed-code transfer conditions, {sum(r["ci"][0]>0 for r in fixed_rule)} '
              f'matrix intervals lie above zero and {sum(r["ci"][1]<0 for r in fixed_rule)} lie below zero. '
              + ('These exploratory intervals include evidence of degradation after transfer. '
                 if any(r['ci'][1]<0 for r in fixed_rule) else
                 'These exploratory intervals do not establish an absence of transfer harms. ')
              + 'The confirmation-grid finding of no Holm-supported harms is conditional '
              'on its original matrices and does not guarantee harmlessness on other codes.']
    lines += ['',
        'Entries are paired yield changes in percentage points, with 95% t intervals '
        'across five matrix-library means (four degrees of freedom). Within-matrix paired '
        'uncertainty is supplied separately. These transfer intervals are exploratory and '
        'are not familywise-corrected. The independent matrix library is the unit for '
        'generalization; frames sharing a matrix are not counted as independent codes. '
        'Zero-discordance bootstrap intervals or identical zero matrix gains can collapse '
        'at zero; this does not rule out rare harms. Wilson FER bounds remain available.',
        '',
        'The irregular construction uses the published variable-degree fractions from '
        '[Elkouss et al., Table I](https://arxiv.org/abs/0901.2140), converted from edge '
        'to node fractions. Its bounded local-cycle search and degree-balanced checks '
        'do not reproduce the published check distribution or optimized ensemble. '
        'Published asymptotic thresholds and state-of-the-art efficiency are not claimed. '
        'Row fractions above .50 reuse the .50 variable-degree profile. '
        'The Gallager family uses three balanced partitions and exact row counts. '
        'All transmitted rows are charged even if algebraically dependent.',
        'The rate model predicts across the full code pool after development on '
        'entropy-margin candidate subsets; some feature/rate combinations involve '
        'model extrapolation, and globally optimal code selection is not established.',
        'Each fixed reference row count maximizes pooled-decoder mean yield on the '
        'independent 6%-QBER development cohort, pooling asymmetries 0 and .5 and '
        'sample sizes 50, 200 and 1000. The physical controls use that same reference; '
        'this calibration is not restricted to symmetric channels.',
        '',
        '[All fixed-code comparisons](../supplementary/extensions/results/transfer_results.csv), '
        '[all pipeline comparisons](../supplementary/extensions/results/pipeline_results.csv), '
        'and [cross-code plots](../supplementary/extensions/results/figures/code_transfer.png).',
        '',
        '## Sampling with a fixed total of 2000 sifted bits',
        '',
        '| Family | Sample | Decoded bits | Pooled yield (%) | Selected-prior yield (%) | Paired difference (pp), matrix interval |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    for r in controls:
        if r['experiment']=='fixed_total':
            lines.append(f'| {r["family"]} | {r["sample"]} | {2000-r["sample"]} | {percent(r["pooled_yield"]["mean"])} | {percent(r["selected_yield"]["mean"])} | {interval(r["delta"])} |')
    lines += ['',
        'Matrices are built at the exact retained length; there is no silent padding. '
        'The policies are transferred from independent development data without retuning '
        'on these sample-allocation outcomes. More accurate priors can coexist with '
        'lower total yield. All yields include sacrificed samples, syndrome rows and '
        '64 public verification bits; none is a composable secure-key rate.',
        'Among the five tested allocations, both selected-prior pipelines have their '
        'highest observed mean yield at s=200. This descriptive comparison does not '
        'establish an optimal sample size outside the evaluated grid.',
        '',
        '## Distinct physical sources of reliability',
        '',
        'The B+ circuit is evaluated with homogeneous or gate-dependent depolarization, '
        'a Pauli link, thermal relaxation and asymmetric readout. Fixed-parameter and '
        'matched-6%-QBER comparisons are both retained. The estimated transition-aware '
        'decoder conditions on Bob\'s local bit as well as basis, with Jeffreys-smoothed '
        'sample transition counts. Its input uses no undisclosed Alice bit.',
        '',
        '| Family | Model | Setting | Separate (pp) | Shrink (pp) | Frozen rule (pp) | Transition-aware (pp) |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for family in ['gallager','irregular']:
        for model in ['homogeneous','gate_dependent','pauli','thermal','readout']:
            for setting in ['fixed','matched']:
                rr=[r for r in controls if r['experiment']=='physical' and r['family']==family and r['model']==model and r['setting']==setting]
                lines.append(f'| {family} | {model} | {setting} | '+' | '.join(interval(next(r for r in rr if r['method']==m)) for m in ['separate','shrink','selected','transition'])+' |')
    lines += ['',
        'Every entry is a paired yield difference with a five-matrix t interval. '
        'These intervals are exploratory and are not corrected across physical conditions. '
        'Bit-asymmetric mechanisms need not benefit a basis-only estimate. Physical '
        'parameters, durations and conventions are specified in the extension README; '
        'T2=2500ns is below 2T1=3600ns. These are controlled simulations, not hardware fits.',
        '',
        '[All physical rows](../supplementary/extensions/results/physical_results.csv) '
        'and [all sample-allocation rows](../supplementary/extensions/results/fixed_total_results.csv).',
        '',
        '## Diagnostic targets and robustness',
        '',
        'The diagnostic model separates legitimate preparation and receiver noise from '
        'Eve\'s operations. Scaling legitimate noise to zero leaves residual disturbance '
        'r*q_intercept(0,0,pE). This equals r/4 with ideal Eve; fixed noisy Eve generally '
        'changes it. The historical all-scaled model remains a separate special case. '
        'Model-based fits in the fixed noisy-Eve controls assume pE is known.',
        'The underlying response permits distinct preparation and receiver noise. '
        'These acquisition simulations and fitted models impose pA=pB=p, so the '
        'Monte Carlo results do not validate inference with independently unknown '
        'legitimate-device noise parameters. Point estimates retain their sign; the '
        'model fits still have declared computational bounds on p and r.',
        '',
        f'The nominal grid has 48 cells: four noise levels, four attack fractions and '
        f'three transmitted-signal budgets. Acquisition design was selected from Fisher '
        f'and delta calculations before diagnostic simulation. The two-setting design '
        f'uses scales {design["two"]["scales"]} and {100*design["two"]["share"]:.0f}% of '
        f'signals at scale1; calibration uses {100*design["calibration"]["calibration_share"]:.0f}% '
        'of its total signals in an independent Eve-free acquisition. All diagnostic '
        'sifted bits are disclosed. Each method consumes its full declared signal budget.',
        '',
        '| Estimator | Grid mean MAE (pp) | Cells meeting both thresholds /48 |',
        '| --- | ---: | ---: |']
    for method in ['linear_four','gate_four','gate_two','calibrated_BSC','calibrated_gate','Fiorini']:
        rr=[r for r in diag if r['experiment']=='grid' and r['method']==method]
        lines.append(f'| {method} | {100*np.mean([r["MAE"] for r in rr]):.3f} | {sum(r["reliable"] for r in rr)} |')
    lines += ['',
        'The descriptive thresholds are MAE<=1 percentage point and empirical coverage '
        'of individual nominal95% intervals>=90%. Coverage intervals, RMSE, signed bias '
        'and interval width are reported for every cell; the table is not a detection '
        'power claim. Methods are paired through common initial base acquisitions when '
        'the acquisition law is shared. Finite-sample interval failure remains visible.',
        '',
        'For ideal Eve and this isotropic legitimate-noise model, '
        'q_intercept(pA,pB,0)=1/4+q_noeve(pA,pB)/2. Consequently the calibrated-BSC and '
        'calibrated-gate inversions are algebraically identical in the nominal grid. '
        'Their matching results are not independent evidence from two different methods. '
        'The Fiorini interception-density formula is evaluated on the same calibration/attacked counts '
        'and mapped to residual disturbance with the known zero-noise attack coefficient '
        '(density/4 with ideal Eve). This tests its estimator rather than its full threshold framework.',
        '',
        'The 10 stress cells use p=.03, r=.2 and 8000 transmitted signals:',
        '',
        '| Stress | Two-setting MAE (pp) | Two-setting bias (pp) | Coverage | Calibrated-gate MAE (pp) | Calibrated-gate bias (pp) | Coverage |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for stress in sorted(set(r['stress'] for r in diag if r['experiment']=='stress')):
        rr=[r for r in diag if r['experiment']=='stress' and r['stress']==stress]
        a=next(r for r in rr if r['method']=='gate_two');b=next(r for r in rr if r['method']=='calibrated_gate')
        lines.append(f'| {stress} | {100*a["MAE"]:.3f} | {100*a["bias"]:+.3f} | {100*a["coverage"]:.1f}% | {100*b["MAE"]:.3f} | {100*b["bias"]:+.3f} | {100*b["coverage"]:.1f}% |')
    alternatives=ident['equivalent_two_setting_solutions']
    lines += ['',
        'The unstable-attack ramp is a failure-mode test. There is no single stationary '
        'intercept in that experiment; coverage is measured against the nominal reference. '
        'Calibration drift, scaling curvature and unscaled disturbance affect different '
        'estimands and assumptions and should not be combined into a universal ranking.',
        '',
        'With unknown fixed Eve noise, two settings provide only a rank2 Jacobian for '
        'three unknown parameters. Equivalent noiseless observations were constructed '
        f'with attack fractions from {min(r["r"] for r in alternatives):.4f} to '
        f'{max(r["r"] for r in alternatives):.4f}. The corresponding residual-intercept '
        f'range in this example is only {100*(max(r["zero_legitimate_intercept"] for r in alternatives)-min(r["zero_legitimate_intercept"] for r in alternatives)):.4f}pp; '
        'the example demonstrates parameter non-identification, not a large practical '
        'residual-QBER ambiguity. It does not establish non-identification with four settings.',
        '',
        f'[All {len(diag)} diagnostic rows](../supplementary/extensions/results/diagnostic_results.csv), '
        '[reliability maps](../supplementary/extensions/results/figures/diagnostic_reliability.png) '
        'and [explicit equivalent device models](../supplementary/extensions/results/diagnostic_identifiability.json).',
        '',
        '## Verification and reproducibility',
        '',
        f'- The protocol-derived inventory contains every required record: {inventory["record_files"]:,} NPZ files, with prescribed frame/replicate counts.',
        (f'- Every pilot/confirmation decode was rerun: {initial["records"]["exactly_reproduced_arrays"]:,} arrays match exactly.'
         if initial['records']['exactly_reproduced_arrays'] else
         '- Pilot/confirmation records and outcomes passed the audit; this audit did not repeat their decodes.'),
        (f'- Every transfer, fixed-total and physical holdout decode was rerun, along with preselected development cases: {validation["records"]["exactly_reproduced_arrays"]:,} arrays match exactly.'
         if validation['records']['exactly_reproduced_arrays'] else
         '- Transfer, fixed-total and physical records and outcomes passed the audit; this audit did not repeat their decodes.'),
        (f'- All {diagnostic_replicates:,} diagnostic replicates were regenerated: {validation["diagnostics"]["exactly_reproduced_arrays"]:,} arrays match exactly; '
         if validation['diagnostics']['exactly_reproduced_arrays'] else
         f'- All {diagnostic_replicates:,} diagnostic replicates passed record checks; ')
        + f'{validation["diagnostics"]["independent_Scipy_fits"]} fits were independently checked with SciPy.',
        f'- An additional audit checked {diagnostic_reference["independent_Scipy_fits"]:,} selected retained fits '
        f'across all {diagnostic_reference["diagnostic_cells"]} diagnostic cells and '
        f'{diagnostic_reference["extra_density_cases"]} extra density-state references; all passed. '
        'Its fit selection and source/input hashes are retained in diagnostic_reference_audit.json.',
        f'- All {len(validation["codes"])} sparse matrices passed CSR, degree and uniqueness checks. Correctness and public-verification flags were recomputed; {initial["records"]["undetected_errors"]+validation["records"]["undetected_errors"]} undetected errors were observed.',
        '- Independent references cover layered BP, Toeplitz hashing, conditional entropy and separated-device density matrices. An observable-input perturbation test changes true channel parameters and undisclosed Alice/error bits while preserving Bob\'s local word; all deployed prior inputs and selection features remain unchanged.',
        '- A boundary-optimizer error found during independent checking was corrected by profiling the affine attack parameter at every candidate noise value. All 360 raw diagnostic acquisition arrays stayed identical. Retired estimates are archived outside the distributed extension.',
        '- The full-from-scratch command also regenerates every development decode; the optional rerun audit independently repeats all holdout decodes and preselected development cases.',
        '',
        'Run `python supplementary/reproduce_extensions.py --mode analysis` from the '
        'repository root with the pinned Python3.13 environment. Add `--rerun-decoders` '
        'to repeat the decoder audit; `--mode full` reconstructs all development and '
        'evaluation data in a separate workspace. The portable extension archive includes '
        'code, records, matrices, frozen policies, data schemas, dependencies and hashes.',
        '',
        'The defensible contribution is a tested sample-only rule with an explicit '
        'benefit/harm boundary, plus diagnostic scope under separately modeled devices. '
        'It does not establish universally positive gains, hardware performance, an '
        'optimized-code threshold replication or a composable security improvement.',
        '']
    full_comparison=OUT.parents[2]/'numeric_comparison.json'
    if full_comparison.is_file():
        full=json.loads(full_comparison.read_text())
        lines += [f'The separate full reconstruction compared {full["files"]:,} NPZ files '
                  f'and reproduced all {full["arrays_exactly_equal"]:,} arrays exactly.', '']
    path=args.output or OUT.parents[2]/'notes/extension-report.md'
    path.parent.mkdir(parents=True,exist_ok=True)
    content='\n'.join(lines)
    relative=os.path.relpath(OUT.parents[1],path.resolve().parent)
    content=content.replace('](../supplementary/',f']({relative}/')
    path.write_text(content)
    print(path,flush=True)


if __name__=='__main__':main()
