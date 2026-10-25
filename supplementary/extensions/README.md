# Basis-asymmetry and separated-device research extension

These experiments provide pooling, transfer, physical-control and diagnostic evidence
for the integrated manuscript in `../../overleaf/`. The reproduction wrapper
generates a detailed numerical report in its isolated output directory.

Use Python 3.13 and the repository's `supplementary/requirements.txt`. From the
repository root:

```sh
python supplementary/reproduce_extensions.py --mode analysis
python supplementary/reproduce_extensions.py --mode analysis --rerun-decoders
python supplementary/reproduce_extensions.py --mode full
```

The first command audits and analyzes supplied records in a separate workspace.
The second additionally repeats every asymmetry and transfer/control holdout
decode, plus preselected development decodes and all diagnostics. The last
command rebuilds all experiments and compares every saved NPZ array exactly.
The full run can take substantially longer than analysis. Nothing is published.

## Experiments and independence

- `pilot.py`: 75 independent development cells of 200 frames, a frozen
  sample-only pooling rule, and 75 holdout cells of 500 frames on unseen matrix
  seed 29. The pre-existing matrix choices were calibrated on separate symmetric
  frames, never on confirmation outcomes.
- `pipelines.py`: two development library seeds, three block lengths and equal
  ridge-logistic training procedures for pooled and selected-prior pipelines.
  Policies are saved before five unseen library seeds are decoded. It also runs
  fixed-total 2000-bit sample allocations and five physical-channel controls.
- `diagnostics.py`: 48 nominal grid cells, 32 separated-device controls and 10
  stress cells, each with 500 acquisition replicates and six estimators. Signal
  budgets include calibration or every noise setting. The base attacked batches
  are coupled through shared initial signals when the acquisition laws agree.
- `validate.py`, `check_extensions.py`: independent decoder/hash/entropy and
  density references, exact frame regeneration, every correctness/verification
  flag, sparse invariants, observable-input checks and independent SciPy fits.
- `inventory.py` rejects missing cells and wrong replicate counts before final
  validation or analysis. Audits and summaries record a fingerprint of all
  numeric records and matrices; the report rejects stale evidence.
- `audit_fixed_reference.py` independently reconstructs every fixed-code
  development choice. `audit_diagnostic_reference.py` checks selected fits
  throughout all 90 diagnostic cells against a separate SciPy optimizer.
- `analyse.py`, `analyse_extensions.py`: one source for tables, uncertainty and
  plots. Matrix intervals use the five library seeds as the unit; within-matrix
  frame intervals are reported separately.

## Interpretation

The binary-channel grid varies basis asymmetry independently of QBER. It is not
a circuit experiment. Its information benchmark is the conditional-entropy gap;
physical bit-asymmetric channels use actual H(A|B,Z), rather than a BSC proxy.
Yield charges the sample, every transmitted syndrome row and 64 verification
bits. It is reconciliation yield, not a composable secure-key rate.

The irregular realization uses the variable-degree fractions in [Elkouss et al.,
Table I](https://arxiv.org/abs/0901.2140), converted from edge to node fractions.
Its bounded local-cycle search and degree-balanced check construction differ
from the published ensemble. Published thresholds are not claimed. The Gallager
construction uses three balanced partitions, with exact row counts and no
padding. All rows are charged even when algebraically dependent.
For row fractions above .50, the irregular construction reuses the .50
variable-degree profile. Both policies predict across the full row-fraction
pool after training on entropy-margin candidate subsets; some combinations
therefore involve model extrapolation.

The fixed-reference choices maximize pooled-decoder development yield at 6%
QBER, using both asymmetries 0 and .5 and all three development sample sizes.
The implementation, retained records, code comments and independent audit all
use this mixture.

Physical B+ preparation uses two noisy Alice operations and one noisy Bob
operation in either basis. Gate-dependent depolarization uses pX=.02, pH=.05,
pIdle=.005. The Pauli link uses(px,py,pz)=(.015,.005,.075). Thermal relaxation
uses T1=1800ns, T2=2500ns and durations X 40 ns, H 60 ns, idle 50 ns. Readout uses
0->1=.01 and1->0=.11. A common scale matches overall QBER to.06 in the matched
controls; fixed-parameter controls retain scale1.

Diagnostic targets remove controllable legitimate-device noise. Fixed Eve noise
remains at the intercept; the target equals ideal r/4 only in the specified
ideal-Eve or all-scaled cases. Fixed noisy-Eve model fits assume its noise is
known. Two settings do not jointly identify three unknown device/attack
parameters. The unstable-attack stress test has no single stationary intercept;
its reported coverage is against the nominal reference target only.
The general device law permits unequal pA and pB, while the Monte Carlo grid
and fitted models use pA=pB=p. Gate estimates remain signed, with no truncation
at zero, but fitting constrains p to [1e-8,min(.3,.999/max(f))] and r to [-.5,1.5].
The Fiorini interception-density estimate is mapped to the shared residual
target with the same known zero-noise coefficient; this is density/4 for ideal
Eve. It is not scored against a different ideal-attack target in noisy-Eve cells.

## Record schema

All NPZ files use numeric or string arrays and open with `allow_pickle=False`.
`*_frames.npz` contains Alice bits, basis labels, errors, public sample
transition counts and estimates, independent Toeplitz seeds, the transition law
and the generator seed. The first s IID positions are sacrificed; their
complement of n positions is decoded. This has the law of a random partition
of exchangeable inputs. Across code families, matching frame seeds deliberately
produce identical inputs; they must not be double-counted as independent frames.

Asymmetry evaluations store full error estimates. Other evaluations store
`estimates_packed`, with little-endian packing along the last axis; unpack using
`np.unpackbits(..., axis=-1, count=n, bitorder='little')`. `correct`, `accepted`
and `iterations` share the method/frame axes. Fixed-code files give scalar m,n
and code identity; pipeline files give per-frame m. Code files contain CSR row
pointers, variable indices, degrees, length, row count and construction seed.

Diagnostic records contain(error,sifted-count) pairs for every acquisition,
point estimates, standard errors, model parameters, signal budgets, target,
device case, stress and seed. Individual 95% intervals are estimate +/-1.959964SE,
without clipping their endpoints. Their empirical coverage is measured, not
assumed. Reliability thresholds(MAE<=1pp,coverage>=90%) are descriptive.

Timing files contain wall-clock observations separately from science arrays.
The first decoder call may include cache loading; timing is not a hardware
throughput claim. Source guards reject resuming after a scientific change, and
atomic writes plus advisory locks protect interrupted jobs. All 360 diagnostic acquisition arrays are retained with the fitted estimates.

Portable reproduction creates a fresh report under its output `notes/`
directory. An analysis-only report describes record audits; it does not inherit
decoder-rerun claims from the supplied package. Add `--rerun-decoders` to obtain
those checks in the new workspace, or use `--mode full` to regenerate all data
and compare every numeric array.
