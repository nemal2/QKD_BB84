# Code map

All project-specific Python modules used by the five studies and current manuscript
are included below. The numerical generators and analysis kernels retain their
verified algorithms. The directory layout is stable because independent wrappers
copy each component into an isolated workspace.

## Execution order

Protocols and seed definitions -> generation -> independent validation -> summaries
and uncertainty -> integrated tables/figures -> manuscript audit and compilation.
The `reproduce_*.py` wrappers implement that order and compare the generated outputs.

## Python modules

| Module | Purpose |
| --- | --- |
| `supplementary/diagnostic_validation/analyse.py` | Summaries and Monte Carlo uncertainty for the frozen diagnostic validation. |
| `supplementary/diagnostic_validation/experiment.py` | Frozen equal-budget diagnostic comparisons and independent response validation. |
| `supplementary/diagnostic_validation/models.py` | Diagnostic response laws and predeclared weighted least-squares estimators. |
| `supplementary/diagnostic_validation/validate.py` | Independent laws, optimizer references, budgets, and retained-record checks. |
| `supplementary/extensions/analyse.py` | Canonical paired analysis and scientific plots for every asymmetry test cell. |
| `supplementary/extensions/analyse_extensions.py` | Cross-matrix uncertainty, sample costs, physical controls and diagnostics. |
| `supplementary/extensions/audit_diagnostic_reference.py` | Independently audit retained diagnostic fits without regenerating acquisitions. |
| `supplementary/extensions/audit_fixed_reference.py` | Independently reconstruct the fixed-code choices from development outputs. |
| `supplementary/extensions/check_extensions.py` | Audit sparse records, observable policy inputs, diagnostics and regeneration. |
| `supplementary/extensions/common.py` | Sparse codes and paired channel simulation for the extension protocol. |
| `supplementary/extensions/diagnostics.py` | Separated-device diagnostics, presimulation design and robustness controls. |
| `supplementary/extensions/inventory.py` | Reject incomplete extension cohorts before auditing, analyzing or reporting them. |
| `supplementary/extensions/physics.py` | Separated-device BB84 probabilities and distinct physical channel mechanisms. |
| `supplementary/extensions/pilot.py` | Independent pilot, sample-only partial-pooling policy, then 75-cell confirmation. |
| `supplementary/extensions/pipelines.py` | Equal-budget development policies, independent code transfer and sample costs. |
| `supplementary/extensions/provenance.py` | Atomic checkpoints, source guards and automatically released writer locks. |
| `supplementary/extensions/report.py` | Write a reviewable research report directly from canonical extension results. |
| `supplementary/extensions/validate.py` | Independent references, exact record audit and optional complete decoder rerun. |
| `supplementary/finalization/audit_manuscript.py` | Check generated tables, inherited circuit tables, references, and headline claims. |
| `supplementary/finalization/build_artifacts.py` | Regenerate final-manuscript tables and plots from the frozen numerical analyses. |
| `supplementary/finalization/channel_information.py` | Descriptive channel-information analysis of unchanged, retained experiments. |
| `supplementary/finalization/method_checks.py` | Check methodological descriptions against immutable records and response laws. |
| `supplementary/finalization/review_analysis.py` | Additional reporting from retained records; no new stochastic experiments. |
| `supplementary/observation_validation/analyse.py` | Summarize target accuracy, empirical interval coverage, and weak identification. |
| `supplementary/observation_validation/experiment.py` | Focused cross-circuit acquisitions and retained-cohort likelihood analysis. |
| `supplementary/observation_validation/likelihood.py` | Binomial observation models and nuisance-profile likelihood intervals. |
| `supplementary/observation_validation/nuisance_intervals.py` | Profile each device parameter separately on the retained four-setting counts. |
| `supplementary/observation_validation/validate.py` | Independent density-state laws, likelihood fits, interval endpoints and budgets. |
| `supplementary/precision_validation/analyse.py` | Report all cells and Monte Carlo uncertainty for independent MAE contrasts. |
| `supplementary/precision_validation/experiment.py` | Fresh budget/interception acquisitions using the unchanged profile fitter. |
| `supplementary/precision_validation/likelihood.py` | Binomial observation models and nuisance-profile likelihood intervals. |
| `supplementary/precision_validation/validate.py` | Independent checks of response laws, proposition identities, fits, and intervals. |
| `supplementary/reproduce.py` | Reproduce the study and manuscript in an isolated workspace. |
| `supplementary/reproduce_diagnostic_validation.py` | Reproduce the basis-resolved and matched-range diagnostic validation. |
| `supplementary/reproduce_extensions.py` | Verify or reproduce the extension study in a separate local workspace. |
| `supplementary/reproduce_final.py` | Reproduce the integrated final manuscript without overwriting working sources. |
| `supplementary/reproduce_observation_validation.py` | Verify or reproduce the focused full-observation and cross-circuit study. |
| `supplementary/reproduce_precision_validation.py` | Verify or reproduce the independent budget/interception precision study. |
| `supplementary/study/analyse_v2.py` | One canonical analysis for all tables and plotted data; no hand-entered results. |
| `supplementary/study/channel_fixed.py` | Fixed-disclosure channel sensitivity; no rate tuning on channel outcomes. |
| `supplementary/study/check_v2.py` | Audit every retained decoded pattern and the calibration/holdout separation. |
| `supplementary/study/circuits_v2.py` | Actual source circuits and exact probability calculations (no shot fitting). |
| `supplementary/study/diagnostics_v2.py` | Matched-budget diagnostics, design selection, drift and native-law comparator. |
| `supplementary/study/plot_v2.py` | Consistent vector figures from canonical records and actual source circuits. |
| `supplementary/study/reconciliation_v2.py` | Controlled code/estimator evaluation for the internal research protocol. |
| `supplementary/study/validate_v2.py` | Independent invariants: matrix nesting, OSD reference, public hashes and circuits. |
| `supplementary/study/write_v2.py` | Fill evidence tables into the manuscript; keep the Overleaf paper in main.tex. |
| `supplementary/upstream/bb84_config.py` | bb84_config.py ============== Configuration dataclasses for the BB84 QKD simulator. |
| `supplementary/upstream/bb84_core.py` | bb84_core.py ============ Quantum parties and classical post-processing for BB84 QKD. |
| `supplementary/upstream/bb84_gate_schemes.py` | bb84_gate_schemes.py ===================== Control-experiment gate-exposure schemes for the caveat in Sec. VII-C/D of the report: "the observed error correlation occurred under an idealized noise model where... |
| `supplementary/upstream/bb84_noise.py` | bb84_noise.py ============= Phase 3: physically-motivated quantum channel noise models for the BB84 QKD simulator, derived from the Lindblad master equation governing open quantum systems (see report Sec. 2.... |
| `supplementary/upstream/reconciliation.py` | reconciliation.py ================= Information-reconciliation experiment library for the LDPC-vs-GRAND comparison on BB84 sifted keys (see error_analysis.ipynb). |

## Latest methodological checks

`finalization/method_checks.py` independently reconstructs the paired pilot score
and all 24 selected weights, enumerates six matched-range Fisher designs, audits
2500 profile fits, checks the augmented path-count rank and provides a numerical
three-setting counterexample. `audit_manuscript.py` recomputes that report and
compares the quantities and wording used in the current paper.

`channel_information.py` computes conditional-entropy benchmarks and disclosure/
recovery accounting. `review_analysis.py` checks clipping and retained diagnostic
comparisons. Neither generates new experiments.

## Shared implementation

The two `likelihood.py` copies are intentionally identical, making each validation
component independently runnable with the exact tested fitter. The manuscript
audit checks their equality. Matched frame records likewise share content across
paired code-family comparisons; they must not be treated as independent data.
