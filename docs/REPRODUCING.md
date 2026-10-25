# Reproduction workflow

## Environment and first check

From the repository root, run `python3 tools/verify.py`. This checks every release
file against `checksums.json`, rejects unsafe manifest paths, parses Python/JSON,
and checks that the current manuscript has anonymous front matter. It does not
execute the scientific modules. Install Python 3.13 and the exact dependency
closure in `supplementary/requirements.txt`; the bundled wheel is local.

The recorded platform is macOS/Apple Silicon. The checkpoint lock uses the POSIX
`fcntl` module, so use a POSIX environment (macOS, Linux or WSL). Cross-platform
floating-point bit identity is not assumed. The wrappers report a mismatch rather
than silently replacing the supplied reference arrays. The exact dependency check
is separate from that scientific equality check.

Activate the environment before the following commands. `work/` is ignored by Git.
Each output directory must be new; failed runs retain their logs for inspection.

## Build the current manuscript

```sh
python supplementary/reproduce_final.py --mode build --compile --output work/manuscript-build
```

This verifies all five component manifests, fills 18 generated tables, redraws
figures, checks the complete 24-table/8-figure manuscript and compiles the source.
The result is `work/manuscript-build/output/pdf/manuscript.pdf` and its
`verification.json`. Without `--compile`, tables, figures and source still build.
The latest `method_checks.py` is run by the manuscript audit, not just copied from
a prior report. Use `--working-copy` only for intentional manuscript edits; it does
not bypass the scientific component manifests.

## Reanalyse all retained observations

```sh
python supplementary/reproduce_final.py --mode analysis --compile --output work/all-analysis
```

This runs the five study analysis wrappers, independent numerical checks, canonical
summary comparisons, figure generation and manuscript validation. The circuit
component regenerates its anonymous table fixture; the final paper remains the
source under `overleaf/`. The analysis does not claim to repeat every stochastic
acquisition or every large decoder run. Logs are kept per component and step.

To additionally rerun the selected and holdout decoder checks in the pooling and
transfer study:

```sh
python supplementary/reproduce_extensions.py --mode analysis --rerun-decoders --output work/decoder-checks
```

## Regenerate the experiments

```sh
python supplementary/reproduce.py --mode full --output work/circuit-full
python supplementary/reproduce_extensions.py --mode full --output work/pooling-transfer-full
python supplementary/reproduce_diagnostic_validation.py --mode full --output work/range-full
python supplementary/reproduce_observation_validation.py --mode full --output work/observation-full
python supplementary/reproduce_precision_validation.py --mode full --output work/precision-full
```

Each command has its own fixed seeds/protocol and compares generated records with
retained reference arrays. These runs can be substantially more expensive than
analysis, especially the long-block decoder work. No universal runtime estimate
is claimed. Run them sequentially unless sufficient compute and memory are available.
The observation study's retained-profile analysis deliberately reads the supplied
matched-range records; rebuilding one component does not silently substitute the
output of a different run. All targets and budgets are specified in the protocols.

## Focused checks

```sh
python supplementary/finalization/method_checks.py
python supplementary/finalization/channel_information.py
python supplementary/finalization/review_analysis.py
```

These regenerate their adjacent JSON/CSV summaries from supplied records. They
cover the pooling score and tie handling, Fisher design enumeration, nuisance and
target boundaries, the rank criterion and a three-setting counterexample, channel
information, yield accounting and estimator clipping. In the recorded environment
the results are identical to the distributed summaries.

## Interpreting a failure

- A checksum mismatch means an input changed or is missing. Restore the released
  file before interpreting downstream results.
- A source-guard failure means a generation directory contains results from a
  different code/protocol version. Use a fresh output directory.
- An array or summary mismatch is reported with its file/key. Check the pinned
  environment and numerical platform before changing tolerances.
- A TeX-only failure does not imply an experimental failure. Inspect `compile.log`
  and `output/pdf/main.log`; compilation may need TeX packages on first use.
- The study-only `reproduce.py` builds the circuit regression fixture. Use
  `reproduce_final.py` for the complete current manuscript.

## Repository handoff

The standalone local repository uses branch `reproducibility-clean`, with no remote
configured and no imported private history. Its initial commit uses a generic
artifact identity. After reviewing the files, a remote can be configured and this
branch pushed by name; do not force-push or replace an existing branch. A branch
in an existing repository does not conceal that repository's other history.
The release ZIP contains the working files, not the Git database.

`python tools/package.py --output ../bb84-reproducibility.zip` verifies the package
and creates a fresh archive from the checksum manifest. It never uploads anything
and refuses to overwrite an existing archive.
