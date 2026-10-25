# BB84 diagnostics and reconciliation: reproducible research

Code, fixed protocols, retained observations and anonymised manuscript sources for
**Beyond Aggregate QBER in Gate-Based BB84: Noise-Scaling Diagnostics and
Basis-Aware Reconciliation**.

This repository covers all five studies and the final information, pooling-score,
finite-design-search, boundary and identification checks. Start with the complete
manuscript in `overleaf/main.tex`; the individual study templates are regression
fixtures for their component tables, not alternative current manuscripts.

## Quick start

Run commands from this repository root. The integrity check uses only Python's
standard library:

```sh
python3 tools/verify.py
```

The recorded scientific environment is CPython 3.13.2 on macOS/Apple Silicon.
Use Python 3.13 and the pinned dependencies:

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r supplementary/requirements.txt
.venv/bin/python supplementary/reproduce_final.py --mode build --compile
```

`build` verifies the component manifests, recalculates the final checks and tables,
and generates figures from retained results. It does not rerun the simulations.
`--compile` additionally requires Tectonic or latexmk and a working TeX installation;
omit it to build the source and figures only. Initial dependency/TeX installation
may need network access. The simulations and analyses use local inputs thereafter.
Every run writes to a new `work/` directory; supplied observations remain intact.

## Choose the depth of reproduction

| Goal | Entry point | What runs |
| --- | --- | --- |
| File integrity | `python3 tools/verify.py` | All distributed file checksums and structural checks; no scientific dependencies |
| Manuscript rebuild | `python supplementary/reproduce_final.py --mode build --compile` | All five input manifests, tables, figures, claim checks and TeX |
| Full retained-data analysis | `python supplementary/reproduce_final.py --mode analysis --compile` | Analyses and independent checks for all five studies, then the manuscript |
| Latest methodological checks | `python supplementary/finalization/method_checks.py` | All 24 pooling choices, six finite Fisher designs, 2500 retained profiles, rank and cubic checks |
| Fresh simulations | Individual `reproduce_*.py --mode full` commands | New seeded observations/fits/decodes, compared with retained arrays |

Use the activated environment's `python` in the last four rows. Exact commands,
outputs and the distinction between analysis and simulation are in
[docs/REPRODUCING.md](docs/REPRODUCING.md).

## Repository map

| Path | Contents |
| --- | --- |
| `overleaf/` | Current anonymous ACM manuscript, bibliography and eight vector figures |
| `supplementary/study/` | Circuit paths, nested codes, BP/OSD, diagnostics and 68,000-frame circuit study |
| `supplementary/extensions/` | Frozen pooling, confirmation, rate-policy transfer, physical mechanisms and stress controls |
| `supplementary/diagnostic_validation/` | Basis baseline, matched scaling ranges and held-out validation |
| `supplementary/observation_validation/` | Full-transition comparisons and target/device profile likelihood |
| `supplementary/precision_validation/` | 81-cell budget/interception grid, all contrasts and independent checks |
| `supplementary/finalization/` | Integrated tables/figures, information accounting and latest methodological checks |
| `supplementary/upstream/` | Source circuit modules and their required licence |
| `supplementary/manuscript_support/` | Anonymous circuit-study table-regression template and TeX support |
| `supplementary/provenance/`, `supplementary/vendor/` | Dependencies, expected component outputs and required notices |
| `docs/` | Reproduction workflow, code map, data semantics and verification evidence |
| `tools/` | Dependency-free repository verification and archive creation |

[docs/CODE_MAP.md](docs/CODE_MAP.md) explains the role of every Python module.
[docs/DATA.md](docs/DATA.md) describes shared inputs, array schemas and units.

## Scientific scope

The experiments concern specified BB84 circuits and characterized noise laws.
Diagnostic disturbance estimates and reconciliation yields are not secure-key
rates. Same-grid confirmation, transfer losses, weak-interception reversals and
rate-policy contributions are retained. The distributed experimental parameters, all 3858 NPZ
archives and every retained numerical CSV/QASM file match the verified research
records; [docs/data-preservation.json](docs/data-preservation.json) records the check.

Caches, private review notes, older release archives, obsolete cohorts, duplicate
workspace copies and Git history are excluded. Matched frame files are deliberately
retained under both experiment labels: they represent shared inputs, not independent
observations. Git stores identical content once. The identical profile fitter in
two studies is retained so each isolated reproduction uses its verified kernel.

The repository edition omits author details and personal repository links.
Published scientific citations and required copyright/licence notices remain;
the upstream MIT notice includes an institutional copyright holder. No claim of
complete institutional anonymity is made. See [docs/NOTICES.md](docs/NOTICES.md).
