#!/usr/bin/env bash
set -euo pipefail
root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ ! -x .venv/bin/python ]]; then
  echo 'Create .venv and install requirements-simulation.txt first.' >&2
  exit 1
fi
# ROS environment scripts can reference unset variables.
set +u
source /opt/ros/jazzy/setup.bash
set -u
exec .venv/bin/python scripts/run_localization.py --ros "$@"
