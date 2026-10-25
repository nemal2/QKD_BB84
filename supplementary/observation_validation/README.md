# Observation design and circuit validation

This focused simulation compares reductions of fully disclosed BB84 observations
against a separately budgeted noise-scaling acquisition. It does not generate
reconciliation frames or train new rate selectors.

`protocol.json` specifies the acquisition design, targets, parameter bounds, and
numerical methods. There are 13 cells with 500 replicate blocks each: nine
circuit/device pairs, three symmetric-readout controls, and a zero-interception
control. Each block has two independent acquisitions of 8000 transmitted signals.
Subset, basis-average, and full-transition estimators share the operating counts.
Balanced B+ is structurally nonidentified at a single setting, so no arbitrary
point-estimator MAE is reported there.

The study also reanalyses 2500 four-setting records from `diagnostic_validation`
using physical binomial likelihood. Residual and individual device parameters
have separate profile intervals. The nominal likelihood-ratio cutoff is
3.841458820694124; observed coverage is reported, including boundary strata,
rather than treating asymptotic calibration as exact.

- `likelihood.py`: path probabilities, binomial fits, residual profiles and information.
- `experiment.py`: seeded acquisitions and retained-cohort fits.
- `nuisance_intervals.py`: individual device-parameter profiles on retained counts.
- `analyse.py`: accuracy, coverage, width quantiles, conditioning, and inverse guards.
- `validate.py`: independent density propagation, algebraic inverses, optimizer and
  boundary checks, and profile-endpoint verification.
- `results/summary.json` and `summary.csv`: all reported outcomes. Null diagnostic
  values mean nonfinite or nonestimable quantities; nonidentified B+ rows have no MAE.

From the enclosing repository root:

```sh
python supplementary/reproduce_observation_validation.py --verify-only
python supplementary/reproduce_observation_validation.py --mode full --output work/observations-full
```

The wrapper creates a separate workspace and compares every retained array and
summary. The component checksum manifest verifies the distributed inputs.
