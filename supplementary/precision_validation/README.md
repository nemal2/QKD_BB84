# Budget and interception precision study

This independent, within-model study crosses PRIMARY, Scheme A, and B+ with
three transmitted budgets (2000, 8000, 32000), three interception fractions
(0.02, 0.10, 0.20), and three device pairs ((0.03,0.03), (0.01,0.05),
(0.05,0.01)). All 81 cells contain 500 fresh replicate blocks. Each block uses
two independent acquisitions at the full stated budget: full operating
transitions and four-setting aggregate scaling. B+ operating records are
retained, but no unidentified point-estimator MAE is reported.

The design, physical parameter bounds, and fitter are fixed before generation.
`likelihood.py` is byte-identical to the frozen `observation_validation` fitter.
The scale locations (1,3,3.5,4) and shares (0.70,0.05,0.05,0.20) are unchanged.
No new reconciliation experiments or acquisition-design optimization are performed.

`results/contrasts.csv` reports MAE(full) minus MAE(scaling); positive values
favor scaling. Pointwise 95% Monte Carlo intervals use independent-acquisition
absolute-error variances and Welch t critical values. Bonferroni intervals cover
the family of 54 identified comparisons simultaneously at nominal 95% confidence.
These approximate Monte Carlo intervals concern expected errors within the fixed
design, not practical equivalence, hardware uncertainty, or individual-target
coverage. `results/outcomes.csv` retains every MAE, bias, profile coverage, width,
and boundary fraction, including the 27 B+ scaling cells.

- `protocol.json` and `results/source_guard.json`: fixed design and generation guard.
- `experiment.py`: independent, seeded acquisitions and unchanged likelihood fits.
- `analyse.py`: complete cell outcomes and uncertainty for MAE differences.
- `validate.py`: independent density propagation; numerical checks of the proposition,
  readout nonidentification, and depolarizing injection identity; acquisition-stream
  reconstruction; independent fit and profile-endpoint checks.
- `results/verification.json`: exact scope and discrepancies of those checks.

From the enclosing repository, run:

```sh
python supplementary/reproduce_precision_validation.py --verify-only
python supplementary/reproduce_precision_validation.py --mode full --output work/precision-full
```

The reproduction wrapper builds a separate workspace, compares every retained
array and all analysis/verification reports, and leaves the retained study intact.
The component checksum manifest verifies the distributed inputs.
