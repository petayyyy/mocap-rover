#!/usr/bin/env bash
# Run the system on the real stand from the laptop.  Camera nodes must already
# run camera_node (scripts/hardware/setup_cm4_node.sh).
#
#   scripts/hardware/run_hardware.sh status                    # ping, link speed, GPU
#   scripts/hardware/run_hardware.sh check [SECONDS]           # live channel: fps, drops, latency
#   scripts/hardware/run_hardware.sh background NAME [SECONDS] # empty arena recording
#   scripts/hardware/run_hardware.sh record NAME [SECONDS]     # recording with the rovers
#   scripts/hardware/run_hardware.sh track NAME [BACKGROUND]   # localization on a recording
#   scripts/hardware/run_hardware.sh sam2-bench                # SAM2 speed on this GPU
#
# Environment: NODES="192.168.10.101:5600 ..." (default: .101-.106),
# CAM_CONFIG (default config/mocap_arena_imx219/runtime_cameras.json; replace it
# with the calibration of the real stand), IFACE (laptop port for 'status'),
# DIVISOR (every N-th full frame is recorded, default 12 -> ~7 fps per camera).
set -euo pipefail
root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root_dir"
py=.venv/bin/python
[[ -x $py ]] || { echo "no .venv: run scripts/hardware/setup_laptop.sh first" >&2; exit 1; }
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

NODES="${NODES:-192.168.10.101:5600 192.168.10.102:5600 192.168.10.103:5600 192.168.10.104:5600 192.168.10.105:5600 192.168.10.106:5600}"
CAM_CONFIG="${CAM_CONFIG:-config/mocap_arena_imx219/runtime_cameras.json}"
DIVISOR="${DIVISOR:-12}"
read -r -a nodes <<< "$NODES"
cmd="${1:-help}"; shift || true

free_gb() { df -BG --output=avail artifacts 2>/dev/null | tail -1 | tr -dc 0-9; }
need_space() {  # a 60 s recording of six cameras at divisor 12 is ~5 GB
  local gb; gb="$(free_gb)"
  if (( gb < $1 )); then echo "only ${gb} GB free in artifacts/, need ${1} GB" >&2; exit 1; fi
}

case "$cmd" in
  status)
    for n in "${nodes[@]}"; do
      host="${n%%:*}"
      if ping -c1 -W1 "$host" >/dev/null; then echo "up   $n"; else echo "DOWN $n"; fi
    done
    [[ -n "${IFACE:-}" ]] && ethtool "$IFACE" | grep Speed
    nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
    echo "free in artifacts/: $(free_gb) GB"
    ;;
  check)
    secs="${1:-60}"
    # The production window plan: marker window 320 raw, opponent small stream.
    "$py" scripts/lan_camera_timing.py --nodes "${nodes[@]}" --seconds "$secs" \
      --window-size 320 --window-format y8 --small 640x480@30 --small-stagger
    ;;
  background|record)
    name="${1:?dataset name}"; secs="${2:-$([[ $cmd == background ]] && echo 20 || echo 60)}"
    out="artifacts/dataset_$name"
    [[ -e $out ]] && { echo "$out exists; datasets are never overwritten" >&2; exit 1; }
    need_space $(( secs / 12 + 2 ))
    "$py" scripts/record_lan_dataset.py --nodes "${nodes[@]}" --config "$CAM_CONFIG" \
      --output "$out" --seconds "$secs" --divisor "$DIVISOR"
    "$py" -c "import json,sys; m=json.load(open('$out/meta.json')); print(json.dumps(m.get('achieved', m), indent=1))"
    ;;
  track)
    name="${1:?dataset name}"; bg="${2:-}"
    # Calibration: the runtime_cameras.json copied into the recording.
    args=(--output "artifacts/track_$name" --no-lidar
          --detector-profile sweep --roi-detector-profile window)
    if [[ -n $bg ]]; then
      # Opponent needs an empty-arena recording.  A LAN recording has no truth,
      # so the operator rectangle that starts the opponent is not available:
      # the opponent track stays off until a live operator input exists.
      args+=(--camera-background "artifacts/dataset_$bg")
    else
      args+=(--no-opponent)
    fi
    "$py" scripts/replay_dataset.py "artifacts/dataset_$name" "${args[@]}" "$@"
    echo "result: artifacts/track_$name"
    ;;
  sam2-bench)
    "$py" scripts/bench_sam2.py --models tiny --dtypes bfloat16 --image-sizes 512 \
      --cameras 1 2 3 --seconds 20 "$@"
    ;;
  *)
    sed -n 2,16p "$0"
    ;;
esac
