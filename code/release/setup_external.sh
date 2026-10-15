#!/bin/bash
# Clones jaxpi2 at the commit this work used into code/external/jaxpi2. Nothing in the clone is edited.
set -e
X="$(cd "$(dirname "$0")/.." && pwd)/external"
mkdir -p "$X"
[ -d "$X/jaxpi2" ] || git clone -q https://github.com/sifanexisted/jaxpi2.git "$X/jaxpi2"
git -C "$X/jaxpi2" checkout -q 77a5c1315a056388271822c35ad512a5a192b60d
echo "jaxpi2 at $(git -C "$X/jaxpi2" rev-parse --short HEAD)"
