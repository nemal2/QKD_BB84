"""Reject incomplete extension cohorts before auditing, analyzing or reporting them.

The inventory is derived from the frozen protocol, not from the files that happen
to exist. This module does not run experiments or modify their checkpoints.
"""
import json
import hashlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
FAMILIES = ('gallager', 'irregular')
MODELS = ('homogeneous', 'gate_dependent', 'pauli', 'thermal', 'readout')
STRESSES = ('calibration_-0.2', 'calibration_-0.1', 'calibration_0.1',
            'calibration_0.2', 'curvature_-0.02', 'curvature_0.02',
            'unscaled_0.01', 'unscaled_0.03', 'noise_ramp', 'attack_ramp')


def expected_records(protocol, result_dir):
    """Map every prescribed NPZ filename to its expected frame/replicate count."""
    p = protocol
    records = {}

    def add(stem, suffixes, frames):
        for suffix in suffixes:
            records[f'{stem}{suffix}.npz'] = frames

    for qi, _ in enumerate(p['pilot']['qber']):
        add(f'reference_cal_{qi}', [''], 150)
        for ai, _ in enumerate(p['pilot']['asymmetry']):
            for si, _ in enumerate(p['pilot']['sample']):
                for stage, frames in [('pilot', p['pilot']['frames']),
                                      ('confirm', p['confirmation']['all_75_cells'])]:
                    add(f'{stage}_q{qi}_a{ai}_s{si}', ['_frames', '_evaluation'], frames)

    rates = np.asarray(p['pipeline_development']['rate_pool'])
    plan = json.loads((result_dir / 'frozen_confirmation_plan.json').read_text())
    if len(plan) != len(p['transfer']['conditions']):
        raise RuntimeError('Frozen confirmation plan has an unexpected condition count.')
    for actual, intended in zip(plan, p['transfer']['conditions']):
        if any(actual[key] != intended[key] for key in ['a', 's']):
            raise RuntimeError('Frozen confirmation plan differs from the declared transfer grid.')
    for family in FAMILIES:
        for n in p['codes']['lengths']:
            for k in p['codes']['development_seeds']:
                for qi, q in enumerate(p['pipeline_development']['qber']):
                    entropy = -q * np.log2(q) - (1-q) * np.log2(1-q)
                    candidates = np.unique([rates[np.argmin(abs(rates-alpha*entropy))]
                                            for alpha in p['pipeline_development']['candidate_margins']])
                    for ai, _ in enumerate(p['pipeline_development']['asymmetry']):
                        for si, _ in enumerate(p['pipeline_development']['sample']):
                            stem = f'develop_{family}_n{n}_k{k}_q{qi}_a{ai}_s{si}'
                            frames = p['pipeline_development']['frames_per_condition_and_code_seed']
                            add(stem, ['_frames'], frames)
                            for beta in candidates:
                                m = 3*round(beta*n/3) if family == 'gallager' else round(beta*n)
                                add(stem, [f'_m{m}_evaluation'], frames)
            for k in p['codes']['test_seeds']:
                for ci, condition in enumerate(plan):
                    add(f'transfer_{family}_n{n}_k{k}_c{ci}',
                        ['_frames', '_evaluation', '_pipeline_pooled', '_pipeline_selected'],
                        condition['frames_per_matrix'])
        for k in p['codes']['test_seeds']:
            for s in p['fixed_total']['sample']:
                add(f'fixedtotal_{family}_k{k}_s{s}',
                    ['_frames', '_pipeline_pooled', '_pipeline_selected'],
                    p['fixed_total']['frames_per_matrix'])
            for model in MODELS:
                for setting in ['fixed', 'matched']:
                    add(f'physical_{family}_k{k}_{model}_{setting}',
                        ['_frames', '_evaluation'], p['physical']['frames_per_matrix'])
    diagnostic = p['diagnostics']
    for pi, _ in enumerate(diagnostic['p_legitimate']):
        for ri, _ in enumerate(diagnostic['attack']):
            for bi, _ in enumerate(diagnostic['budgets']):
                add(f'diagnostic_p{pi}_r{ri}_b{bi}', [''], diagnostic['replicates'])
            for ci in [1, 2]:
                add(f'devicecase_{ci}_p{pi}_r{ri}', [''], diagnostic['replicates'])
    for stress in STRESSES:
        add(f'diagnostic_stress_{stress}', [''], diagnostic['replicates'])
    return records


def check_inventory(result_dir=None, protocol=None, check_counts=True):
    result_dir = HERE/'results' if result_dir is None else Path(result_dir)
    protocol = json.loads((HERE/'protocol.json').read_text()) if protocol is None else protocol
    required = ['reference_selection.json', 'frozen_rule.json', 'frozen_confirmation_plan.json',
                'frozen_pipeline_policies.json', 'frozen_diagnostic_design.json',
                'diagnostic_identifiability.json', 'asymmetry_source_guard.json',
                'pipelines_source_guard.json', 'diagnostics_source_guard.json']
    missing_metadata = [name for name in required if not (result_dir/name).is_file()]
    if missing_metadata:
        raise RuntimeError(f'Incomplete extension metadata: {missing_metadata}')
    expected = expected_records(protocol, result_dir)
    actual = {path.name for path in result_dir.glob('*.npz')}
    missing = sorted(expected.keys()-actual)
    unexpected = sorted(actual-expected.keys())
    if missing or unexpected:
        raise RuntimeError(f'Incomplete extension inventory: {len(missing)} missing, '
                           f'{len(unexpected)} unexpected NPZ files. '
                           f'Missing examples: {missing[:12]}; unexpected: {unexpected[:12]}')
    if check_counts:
        for name, expected_count in expected.items():
            with np.load(result_dir/name, allow_pickle=False) as record:
                key = 'pool' if 'pool' in record.files else 'correct' if 'correct' in record.files else 'estimates'
                count = record[key].shape[-1] if key == 'correct' else len(record[key])
                if count != expected_count:
                    raise RuntimeError(f'Wrong frame/replicate count in {name}: {count} != {expected_count}')
    diagnostic_files = [name for name in expected if name.startswith(('diagnostic_', 'devicecase_'))]
    return dict(status='complete',record_files=len(expected),
                diagnostic_cells=len(diagnostic_files),
                diagnostic_acquisition_replicates=sum(expected[name] for name in diagnostic_files),
                reference_calibration_frames=sum(count for name,count in expected.items()
                                                 if name.startswith('reference_cal_')))


def record_fingerprint(result_dir=None):
    """Bind a completed audit to every retained numeric record and matrix."""
    result_dir = HERE/'results' if result_dir is None else Path(result_dir)
    digest = hashlib.sha256()
    for path in sorted(result_dir.rglob('*.npz')):
        digest.update(path.relative_to(result_dir).as_posix().encode()+b'\0')
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


if __name__ == '__main__':
    print(json.dumps(check_inventory(), indent=2))
