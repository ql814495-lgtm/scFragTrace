#!/usr/bin/env bash
# Train the shipped bundle and write the demo's expected outputs (run from the repository root).
#   bash model/training/run_training.sh <labelled input dir> [threads]
# The labelled input dir holds the 539 Cristiano genomes in the format of model/docs/INPUTS.md (including
# delfi_features.npz) plus the columns cancer, true_class, train_too. These inputs are derived from a
# controlled-access cohort (EGAD00001005339); rebuild them with `scfragtrace extract`.
# One training process, about 2 GB RAM, roughly one hour on 4 CPU threads.
set -euo pipefail
IN=${1:?labelled input dir}; T=${2:-2}
b=cristiano_only
python model/training/train_model.py --inputs "$IN" --config model/training/configs/$b.json --out model/weights/$b --threads "$T"
python model/examples/run_demo.py --bundle $b --out model/examples/expected_output/$b
python model/examples/run_demo.py --bundle $b --mode full --out model/examples/expected_output/${b}_full
python -m pytest -q tests
