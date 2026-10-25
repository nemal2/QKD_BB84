"""Independently audit retained diagnostic fits without regenerating acquisitions.

Run from any directory with the extension's Python dependencies installed. The
audit checks selected replicates in every diagnostic cell, including up to three
low-noise boundary fits per design. It does not claim to refit every replicate.
SciPy uses the declared response law but an independent optimizer/initialization;
the additional density-state checks independently test that response law.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from common import OUT
from diagnostics import observations, response
from physics import Q, exact_density_Q
from provenance import atomic_json


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    patterns = {
        'diagnostic_p*.npz': 48,
        'devicecase_*.npz': 32,
        'diagnostic_stress_*.npz': 10,
    }
    files = []
    for pattern, expected in patterns.items():
        matches = sorted(OUT.glob(pattern))
        assert len(matches) == expected, (pattern, len(matches), expected)
        files.extend(matches)

    total_checks = total_boundary = 0
    maximum_excess = 0.0
    rows = []
    for path in files:
        with np.load(path, allow_pickle=False) as record:
            d = dict(record)
        assert d['estimates'].shape == (500, 6), path
        row = {'file': path.name, 'sha256': sha256(path), 'designs': {}}
        for counts_key, scales_key, parameters_key in [
            ('counts4', 'scales_four', 'parameters_four'),
            ('counts2', 'scales_two', 'parameters_two'),
        ]:
            f = d[scales_key]
            stored = d[parameters_key]
            lower = np.array([1e-8, -0.5])
            upper = np.array([min(0.3, 0.999 / max(f)), 1.5])
            at_boundary = np.any(
                np.isclose(stored, lower, atol=1e-9)
                | np.isclose(stored, upper, atol=1e-9), axis=1,
            )
            indices = sorted(set(
                [0, 73, 199, 337, 499]
                + np.flatnonzero(stored[:, 0] < 1e-7)[:3].tolist()
            ))
            excesses = []
            for index in indices:
                y, variance = observations(d[counts_key][index])
                weights = 1 / np.sqrt(variance)

                def residual(parameters):
                    predicted = response(
                        f, parameters[0], parameters[1],
                        float(d['pE']), str(d['case']),
                    )
                    return (predicted - y) * weights

                fit = least_squares(
                    residual, [0.03, 0.2], bounds=(lower, upper),
                    ftol=1e-11, xtol=1e-11, gtol=1e-11,
                )
                assert fit.success, (path, counts_key, index, fit.message)
                excess = float(
                    np.linalg.norm(residual(stored[index]))
                    - np.linalg.norm(fit.fun)
                )
                assert excess <= 1e-5, (path, counts_key, index, excess)
                excesses.append(excess)
            maximum_excess = max(maximum_excess, max(excesses))
            total_checks += len(indices)
            total_boundary += int(at_boundary.sum())
            row['designs'][counts_key] = {
                'checked_indices': indices,
                'fit_checks': len(indices),
                'boundary_fits_in_all_500_replicates': int(at_boundary.sum()),
                'maximum_objective_norm_excess': max(excesses),
            }
        rows.append(row)

    density_rows = []
    for p_a, p_b, p_e, attack, scale in [
        (0.005, 0.06, 0.1, 0.4, 8.0),
        (0.06, 0.005, 0.03, 0.2, 8.0),
        (0.06, 0.06, 0.03, 0.4, 8.0),
        (0.001, 0.06, 0.1, 0.7, 0.0),
        (0.06, 0.06, 0.06, 0.4, 8.0),
    ]:
        for case in ['legitimate', 'all_scaled']:
            analytic = float(Q(scale, p_a, p_b, p_e, attack, case))
            density = exact_density_Q(scale, p_a, p_b, p_e, attack, case)
            error = abs(analytic - density)
            assert error < 1e-12, (p_a, p_b, p_e, attack, scale, case)
            density_rows.append({
                'pA': p_a, 'pB': p_b, 'pE': p_e, 'attack': attack,
                'scale': scale, 'case': case, 'analytic': analytic,
                'density_reference': density, 'absolute_error': error,
            })

    directory = Path(__file__).resolve().parent
    return {
        'status': 'passed',
        'scope': (
            'Selected stored fits in every diagnostic cell, with an independent '
            'SciPy optimizer and initialization; no acquisitions regenerated. '
            'Extra density-state references include unequal legitimate-device '
            'noise and high/zero noise scaling. The Monte Carlo experiment '
            'itself sets pA=pB.'
        ),
        'sources': {name: sha256(directory / name) for name in [
            'audit_diagnostic_reference.py', 'diagnostics.py', 'physics.py',
        ]},
        'diagnostic_cells': len(files),
        'acquisition_replicates': 500 * len(files),
        'independent_Scipy_fits': total_checks,
        'maximum_objective_norm_excess': maximum_excess,
        'objective_norm_tolerance': 1e-5,
        'boundary_fits_in_all_retained_gate_designs': total_boundary,
        'extra_density_cases': len(density_rows),
        'maximum_density_probability_error': max(
            row['absolute_error'] for row in density_rows
        ),
        'fit_cells': rows,
        'density_checks': density_rows,
    }


def main():
    result = audit()
    atomic_json(OUT / 'diagnostic_reference_audit.json', result)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ['sources', 'fit_cells', 'density_checks']}, indent=2))


if __name__ == '__main__':
    main()
