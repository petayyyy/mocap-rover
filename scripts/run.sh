#!/usr/bin/env bash
set -euo pipefail
root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export GZ_SIM_RESOURCE_PATH="$root_dir/models:$root_dir${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
# Keep the lock through exec and until Gazebo's child processes have exited.
# Overlapping restarts can leave the GUI attached to a departing server.
runtime_dir="${XDG_RUNTIME_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}}/mocap-rover"
mkdir -p "$runtime_dir"
exec 9>"$runtime_dir/gazebo.lock"
if ! flock -n 9; then
  echo 'Арена уже запущена или ещё завершает работу. Закройте её и повторите запуск через несколько секунд.' >&2
  exit 1
fi
# VS Code installed via Snap exports GTK/GIO modules linked to Snap's glibc.
# Native Gazebo must load the host modules instead.
if [[ "${SNAP_NAME:-}" == code ]]; then
  unset GTK_PATH GTK_EXE_PREFIX GTK_IM_MODULE_FILE GIO_MODULE_DIR LOCPATH
  export XDG_DATA_DIRS="${XDG_DATA_DIRS_VSCODE_SNAP_ORIG:-/usr/local/share:/usr/share}"
  export XDG_CONFIG_DIRS="${XDG_CONFIG_DIRS_VSCODE_SNAP_ORIG:-/etc/xdg}"
fi
# After a driver change the running kernel can still use the previous version.
# A matching, locally extracted library set lets this process run until reboot.
# Do not use the workaround once the system driver works normally.
if [[ -r /proc/driver/nvidia/version ]] && command -v nvidia-smi >/dev/null && ! nvidia-smi >/dev/null 2>&1; then
  driver_version="$(sed -nE 's/^NVRM version:.* ([0-9]+\.[0-9]+\.[0-9]+) .*/\1/p' /proc/driver/nvidia/version)"
  driver_libs="${XDG_CACHE_HOME:-$HOME/.cache}/mocap-rover/nvidia-${driver_version}/usr/lib/x86_64-linux-gnu"
  if [[ -n "$driver_version" && -f "$driver_libs/libGLX_nvidia.so.$driver_version" ]]; then
    export LD_LIBRARY_PATH="$driver_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "Gazebo: using local NVIDIA $driver_version libraries (system driver mismatch)." >&2
  fi
fi
world_path="${MOCAP_WORLD:-$root_dir/worlds/mocap_arena.sdf}"
exec gz sim -r "$world_path" --gui-config "$root_dir/config/gui.config" "$@"
