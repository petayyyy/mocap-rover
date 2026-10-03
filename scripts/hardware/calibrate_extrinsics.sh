#!/usr/bin/env bash
# Mutual poses of all six cameras with the 12x3 m ChArUco floor strip.
# Guide: docs/calibration_extrinsics.md
#
#   scripts/hardware/calibrate_extrinsics.sh start                 # station + page http://localhost:8090
#   scripts/hardware/calibrate_extrinsics.sh start --rate 10       # fixed 10 Hz instead of auto 20/10
#   scripts/hardware/calibrate_extrinsics.sh solve SESSION_DIR     # offline re-solve of a session
#   scripts/hardware/calibrate_extrinsics.sh demo                  # no hardware: synthetic nodes
#
# NODES="192.168.10.101:5600 ..." overrides the six default nodes; PORT the page port.
# Extra arguments go to scripts/calibration_station.py (see its --help), e.g.
# --session DIR to continue a session, --intrinsics DIR for K, D from elsewhere.
set -euo pipefail
root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root_dir"
py=.venv/bin/python
[[ -x $py ]] || { echo "no .venv: run scripts/hardware/setup_laptop.sh first" >&2; exit 1; }
NODES="${NODES:-192.168.10.101:5600 192.168.10.102:5600 192.168.10.103:5600 192.168.10.104:5600 192.168.10.105:5600 192.168.10.106:5600}"
PORT="${PORT:-8090}"
read -r -a nodes <<< "$NODES"
cmd="${1:-help}"; shift || true

case "$cmd" in
  start)
    exec "$py" scripts/calibration_station.py --nodes "${nodes[@]}" --port "$PORT" --tab extrinsic "$@"
    ;;
  demo)
    exec "$py" scripts/calibration_station.py --demo --port "$PORT" --tab extrinsic "$@"
    ;;
  solve)
    session="${1:?session directory, e.g. artifacts/calibration/20261003_120000}"; shift
    exec "$py" scripts/solve_extrinsics.py "$session" "$@"
    ;;
  *)
    sed -n 2,13p "$0"
    ;;
esac
