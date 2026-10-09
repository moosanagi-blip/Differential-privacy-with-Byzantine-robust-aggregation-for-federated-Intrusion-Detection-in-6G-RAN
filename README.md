# DP FL IDS code and experimental results

Code and recorded evidence for an empirical evaluation of client-side DP-SGD
and a Multi-Krum variant selecting seven of ten clients. Prepared CICIoT2023
network features are augmented with seven synthetic radio features.
This repository does not demonstrate real 6G deployment or general robustness.

## Contents

- `code/`: implementation, tests, utility scripts and dependency requirements.
- `analysis_scripts/`: experiment drivers, offline audits and analysis tools.
- `protocols/`: boundary-extension and final instrumented-run protocols.
- `results/`: matrix-listed actual-run records, including numerical failures.
- `MANIFEST.json`: file sizes and SHA-256 hashes for the packaged files.

## Result scope

The corrected campaign has 102 conditions: 97 completed and five numerical
failures. The boundary extension has 36 conditions: 33 completed and three
numerical failures. The final instrumented batch has nine completed conditions.
The separately supplied seed-42 main record is reused in the corrected campaign;
it is not an extra independent replicate. Three seeds are the replicate units,
not rounds, clients, classes or the total number of condition records.

Failures are evidence of failed execution, not zero-valued final utility.
The evaluated prepared population has 23,366,212 records, with 18,692,957
training and 4,673,255 held-out records. Radio features are synthetic.
Privacy bounds are conditional on fixed prepared inputs and the stated release
scope; seeded research operation is not a production privacy deployment.

## Reproduction limitations

This is a curated source/evidence handoff, not a tested turnkey installation.
Data are not included. Obtain the dataset separately and follow the preparation
and protocol instructions. Drivers retain original workspace and data-path
assumptions; inspect and configure those paths before running them. Some analysis
scripts require excluded local diagnostic inputs and cannot run from this folder
alone. Do not launch every driver indiscriminately: some start long experiments.

`code/requirements.txt` lists dependency lower bounds. Its historical version
comments are not a lockfile for the corrected campaign; consult the saved records
and protocols for available execution provenance. No claim of fresh reproduction
inside this copied folder is made. Frozen research source was not changed during
packaging. No software or dataset licence is granted merely by including files;
the repository owner must choose an appropriate code licence and respect source
dataset terms before publishing.

## Exclusions and privacy

No manuscript, paper PDF, LaTeX, bibliography, reviewer letter or response is
included. Datasets, virtual environments, caches, logs, interrupted attempts,
unvalidated historical outputs, raw noise-replay files and detailed counterfactual
statistics are excluded. Actual-run diagnostic evidence is not independently
certified private merely because training used DP. Do not add private datasets,
credentials or excluded counterfactual records to this upload folder.

Assembled 10 October 2026. No GitHub upload was performed by the packaging step.
