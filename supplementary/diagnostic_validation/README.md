# Basis-resolved diagnostics and acquisition range

This simulation study compares information available in one basis-resolved
operating acquisition with additional aggregate noise-scaling acquisitions and
an Eve-free calibration reference. `protocol.json` fixes the cohorts, budgets,
candidate designs, estimator domains, intervals, and validation settings.

The nominal comparison uses 48 cells (four device-noise values, four interception
fractions, three budgets), with 500 replicates per cell. Nine off-grid cells use
new noise/interception values at 8000 transmitted signals. Each replicate block
contains eight independently sampled, fully budgeted acquisition strategies.
The analytic basis inverse is a sensitivity estimator on the same basis counts.
There are 28,500 grid replicate blocks and 228,000 budgeted acquisitions.

For each maximum scale 2.5, 4, or 8, the two-setting candidate uses both endpoints.
Four-setting candidates add two of five equally spaced interior points. Signal
shares are positive multiples of 5%; every setting receives at least 5%. The
minimax Fisher criterion uses the nominal 16 noise/interception pairs before
sampling. This is a finite candidate optimization; four settings may collapse to
two if zero allocations are allowed. `results/design.json` stores each selected
design and the best allocation for every candidate set of locations.

Five held-out scenarios use 8000 training and 2000 validation signals per strategy,
at validation scale 2.3. This setting is never fitted. There are 2500 independent
replicate blocks, each with three training acquisition strategies. Each pays the
full validation budget even though matched validation records are reused across
strategies. The four-setting record is fitted under both equal-device noise and
independently unknown Alice/Bob noise. Curvature preserves scales zero and one:
`f_effective = f + epsilon*f*(f-1)`. It differs from the acquisition-offset control
in the separate reference/stress study.

All point estimates and local-Wald intervals are retained, including signed
intercepts, boundary fits, poor coverage, and large intervals. Fixed-grid mean
error intervals describe Monte Carlo sampling only. Average coverage does not
replace per-cell coverage, and no equivalence margin was prespecified.

`results/summary.json` and the two CSV files report every cell. The NPZ files
contain counts, fitted parameters, covariance matrices, budgets, seeds, and
targets. `results/verification.json` records checks of all budgets and retained
counts, density-state reference laws, and independent optimizer comparisons.

From the package root:

```sh
python supplementary/reproduce_diagnostic_validation.py --verify-only
python supplementary/reproduce_diagnostic_validation.py --mode analysis --output work/diagnostic-analysis
python supplementary/reproduce_diagnostic_validation.py --mode full --output work/diagnostic-full
```

Use a fresh output directory. Analysis mode refits the independent references
and regenerates summaries. Full mode also regenerates every acquisition and
fitted result, comparing every retained array exactly. Dependencies are pinned
in `supplementary/requirements.txt`. The original circuit and other diagnostic
records are unchanged. No reconciliation codes or policies are trained here.
