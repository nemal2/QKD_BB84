# Data, units and shared observations

Every NPZ archive opens with `numpy.load(path, allow_pickle=False)`. No pickle
loading is needed. The files retain observed counts/frames, matrices, fitted
estimates and explicit generator parameters; they are inputs to reproducible
analysis, not a promise of hardware performance.

| Study | Observation unit | Main outputs |
| --- | --- | --- |
| Circuit (`study`) | Sifted frame with disjoint disclosed and decoded positions | Paired decoding flags, explicit matrices, operating selections, diagnostic counts |
| Pooling/transfer (`extensions`) | Pilot/confirmation frame, library-specific transfer frame, or diagnostic acquisition | Frozen rules/policies, paired outcomes, full control/stress summaries |
| Matched range (`diagnostic_validation`) | Replicate block of independently budgeted acquisition strategies | Counts, WLS fits/covariances, range designs, held-out prediction |
| Full observations (`observation_validation`) | Operating/scaled block; retained four-setting cohort | Physical likelihood fits, residual/device profiles and coverage |
| Precision (`precision_validation`) | Cell/block with separately budgeted full/scaled acquisitions | Counts, fits, all 135 method/cell outcomes and 54 accuracy contrasts |

An error estimate or yield is usually stored as a fraction. Reported differences
labelled `pp` are 100 times a fractional difference. Coverage/FER columns labelled
percent use percentages. Read each CSV header and the associated analysis code;
there is no implicit conversion applied to all stored arrays.

Protocols distinguish training, code calibration, same-grid confirmation and
transfer. Reused records must not be counted as fresh observations. In particular,
identical frame archives across Gallager and irregular constructions deliberately
use shared inputs, and several estimators share the same operating counts. The
record inventory counts those inputs once. These files remain under their expected
names so independent reproduction and source guards continue to work.

Timing JSON files record machine-dependent wall-clock observations separately from
science arrays. They are not used to establish numerical equality across machines.
All failure, boundary and zero-gain cells remain available. A JSON null in a profile
summary denotes a nonfinite/nonestimable diagnostic, not a silently dropped cell.

The complete circuit schemas are in `supplementary/README.md`; the four other study
READMEs describe their schemas. The five `protocol.json` files are the authoritative
source of seeds, budgets, sample sizes, scales, bounds and inference families.

All 3893 numerical NPZ/CSV/QASM files are byte-identical to the verified study
records. Repository-level checksums and source guards reflect the distributed
comments, anonymous metadata and documentation. The executable syntax of numerical
kernels is unchanged; the incorrect symmetric-only pipeline comment now describes
the actual mixed-asymmetry development design.
