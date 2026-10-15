#!/bin/bash
# The Python environment in <root>/envs/v2: JAX 0.10.2 and Flax 0.12.3 (the versions jaxpi2 pins), SOAP, and jaxpi2
# from code/external (setup_external.sh), installed from a copy so that the clone stays untouched. Python 3.12,
# CUDA 13 (JAX_EXTRA=cuda12 for CUDA 12).
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
E=$ROOT/envs/v2
X=$ROOT/code/external
python3 -m venv "$E"
P=$E/bin/pip
$P install -q --upgrade pip
$P install -q "jax[${JAX_EXTRA:-cuda13}]==0.10.2" "flax==0.12.3" "optax==0.2.8" "orbax-checkpoint==0.12.6" \
    "soap-jax @ git+https://github.com/haydn-jones/SOAP_JAX@45145bcadb20864e821b87999c306e885bed7171" \
    "ml_collections==1.1.0" "absl-py==2.5.0" "numpy==2.3.5" "scipy==1.18.1" "matplotlib==3.11.2" "wandb==0.30.0" \
    "tabulate==0.10.0" "tqdm==4.70.1"
T=$(mktemp -d)
cp -r "$X/jaxpi2" "$T/"
$P install -q --no-deps "$T/jaxpi2"
rm -rf "$T"
echo "environment in $E"
