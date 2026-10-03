#!/usr/bin/env bash
# Fisheye intrinsics (K, D) of one camera with a hand-held ChArUco board;
# one set of ~35 captures gives three calibrations: 1640x1232, 820x616, 640x480.
# Guide: docs/calibration_intrinsics.md
#
#   scripts/hardware/calibrate_intrinsics.sh start camera_3          # page opens on that camera
#   scripts/hardware/calibrate_intrinsics.sh offline CAPTURES camera_3
#   scripts/hardware/calibrate_intrinsics.sh demo
#
# The board defaults to sverk-ros2 camera_calibration's A4 board (11x8 squares
# of 22 mm, 16 mm markers, DICT_4X4_50); another board:
#   ... start camera_3 --board-squares 9 6 --board-square-mm 30 --board-marker-mm 22 --board-dict DICT_5X5_100
# NODES / PORT as in calibrate_extrinsics.sh; extra arguments go to the station.
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
    camera="${1:-}"; [[ $# -gt 0 ]] && shift
    exec "$py" scripts/calibration_station.py --nodes "${nodes[@]}" --port "$PORT" --tab intrinsic \
      ${camera:+--camera "$camera"} "$@"
    ;;
  demo)
    exec "$py" scripts/calibration_station.py --demo --port "$PORT" --tab intrinsic --camera camera_1 "$@"
    ;;
  offline)
    captures="${1:?folder of 1640x1232 captures}"; camera="${2:?camera id, e.g. camera_3}"; shift 2
    exec "$py" scripts/calibrate_intrinsics.py "$captures" --camera "$camera" "$@"
    ;;
  *)
    sed -n 2,14p "$0"
    ;;
esac
