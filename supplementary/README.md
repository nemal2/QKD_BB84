# Computational studies

Use the repository-level [reproduction guide](../docs/REPRODUCING.md) for the
complete current manuscript. `reproduce_final.py` integrates all five studies;
`reproduce.py` reproduces only the circuit study and its internal table fixture.

All experiments use the pinned environment in `requirements.txt`, independent
seeded cohorts, explicit signal budgets and retained observations. Each study's
README and `protocol.json` describe its own design and array semantics. Current
results are under each study's `results/`; obsolete cohorts are not distributed.

The source modules in `upstream/` provide complete gate paths. Current decoding
is implemented in `study/reconciliation_v2.py` and `extensions/common.py`.
The anonymous `manuscript_support/` files are needed only for the circuit
regression fixture. `provenance/` contains dependency versions, expected output
hashes and the third-party formula source. All reference notices are retained.

## Experimental cohorts

- Main calibration: 12 encoding/QBER cells × 500 frames = 6000.
- Main holdout: the same 12 cells × 2000 independent frames = 24,000.
- Fixed-code curves: 2 encodings × 12 QBERs × 500 frames = 12,000. Three matrices and two prior rules reuse each cohort.
- Channel controls: 8 cells × 500 calibration + 2000 holdout frames = 20,000. The fixed-disclosure sensitivity analysis reuses the holdout frames.
- Matrix replication: 2 further seeds × 3 encodings × 1000 fresh frames = 6000.
- Total: 68,000 independently generated 1200-bit sifted frames, with disjoint 200-bit samples and 1000-bit decoded blocks.
- Diagnostics: 1000 replicates at each of five interception fractions, with 8000 transmitted signals per method/replicate. Separate native-law and unscaled-channel controls retain their own counts.

Main operating matrices are selected using calibration only, saved before the
holdout is generated. The target is a per-code Wilson upper FER bound of 1%,
with a passing tail across higher tested row counts. Holdout intervals separately
assess that policy. Observed FER below 1% does not certify every upper bound below
1%. The same-rate table uses the shrinkage-selected matrix for both methods;
the operating table compares separately calibrated row counts.

The positive-part shrinkage rule is a plug-in heuristic. Separate Jeffreys,
coherent unshrunk, and coherent shrinkage ablations distinguish pseudocount
centring from class-contrast fitting. Equal-exposure and pure-channel controls
retain the observed harms as well as gains. BP+OSD uses the same budget for both
priors and retries on nonconvergence or public Toeplitz verification failure.
Exact simulator correctness never triggers a decoder action.

## Array schemas

All NPZ archives contain ordinary NumPy arrays, readable with
`numpy.load(path, allow_pickle=False)`.

`*_frames.npz` retains `bits`, `bases`, `errors` (F×1200),
`sample_indices` (F×200), `kept_indices` (F×1000), `sample_counts`
(F×4: rect errors/count, diagonal errors/count), `priors` (4×F×1000),
`shrink_weights`, `hash_seeds` (F×1063), and generator parameters.
Bob's bit is Alice's bit XOR the error. Decoded-block errors are obtained by
gathering `errors` at `kept_indices`. Hash columns are deterministic Toeplitz
columns reconstructed from `hash_seeds`; these dense derived arrays are not
stored twice. Prior order is pooled, separate Jeffreys, coherent unshrunk,
centred shrinkage.

`*_evaluation.npz` retains selected `m_values` and arrays indexed
(matrix, prior, frame): `correct` (BP+OSD exact recovery), `bp_correct`,
`iterations`, `retried`, `collisions`; `estimated_errors` additionally has a
1000-bit axis and retains the final strong-decoder error estimate. Fixed-channel
files have two priors, pooled and centred shrinkage. Their input frames are the
corresponding `channel_*_frames.npz` cohort, not newly duplicated records.

`*_calibration.npz` has `m_values`, paired `correct`, and BP `iterations` for
all tested matrices. `*_selection.json` retains every count and selected row
count. `selections.json` freezes all main choices before main holdout evaluation.
`codes_seed*.npz` contains every explicit binary parity-check matrix, keyed by
row count. `curve_*.npz` contains paired outcomes for three fixed matrices.

`diag_??.npz` contains error-count/sifted-denominator pairs for both noise-setting
designs, every stale calibration, the signal run, and the native-law control.
`fit4` and `fit2` columns are linear intercept, gate-model disturbance, and fitted
base gate noise. `calibrated` is indexed (replicate, drift offset, method), with
methods BSC, calibrated gate, and Fiorini. All estimates are unclipped.
`diag_channel_*.npz` retains the unscaled receiver-channel control.
`diagnostic_design.json` stores all candidate designs and Fisher calculations,
selected before Monte Carlo performance evaluation.

The Fiorini implementation transcribes the interception-density formula from
`francesco-fiorini/BB84-QuantumSimulator` at commit
`9ce07ab3425bf3e98a7e352a3e97b6bafa493edf`, file `bb84/bb84_algorithm.py`.
It estimates the common disturbance target by dividing interception density by
four. This compares the formula, not the full threshold-detection application.
Its corresponding algebraic-law control checks the transcription separately
from the different two-hop gate model. Provenance and the original MIT notice
are included.

## Validation and interpretation

`validation.json` records full row ranks and nesting of all 93 matrices, 90
independent OSD reference cases, independent Toeplitz and coherent-prior checks,
144 exact circuit/channel probability checks, and four two-hop checks through
the maximum simulated gate noise. `record_integrity.json` checks all 68,000
frame partitions, retained sample statistics, 296,000 final method/frame decoding
patterns, public-hash comparisons, and all 20 calibration selections.

The fresh full simulation is compared array by array, including counts, patterns,
priors, matrices, and fitted estimates. Reanalysis must reproduce the canonical
summary and generated TeX exactly in the recorded environment; figure text is
checked when `pdftotext` is available. PDF bytes are not compared because PDF
creation metadata varies. Repository verification evidence is in `../docs/verification.json`.

FER intervals are Wilson intervals restricted to [0,1] at numerical boundaries.
Paired exact tests and Holm families are specified in the protocol. Operating
yield intervals condition on frozen calibration choices. Diagnostic intervals
stratify by attack fraction and preserve within-replicate method pairing.
The fixed-disclosure channel probe is a separately labelled sensitivity analysis,
not part of the frozen main confirmation family.

The paper concerns a controlled homogeneous gate-noise model. The code family
is conservative and is not an optimized practical QKD code. Reconciliation yield
includes sample and 64-bit verification costs, but precedes privacy amplification
and authentication. Disturbance estimation under a fixed attack model is distinct
from a composable security bound. These scopes are part of the scientific result.
