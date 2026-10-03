#!/usr/bin/env bash
# Laptop (Ubuntu 24.04, x86_64, NVIDIA GPU): system packages, NVIDIA driver,
# Python venv with OpenCV / torch (CUDA) / SAM2, SAM2 weights, static address
# on the camera network.  Safe to run again: every step checks before it acts.
#
#   scripts/hardware/setup_laptop.sh                     # packages + venv + weights
#   scripts/hardware/setup_laptop.sh --driver            # also install nvidia-driver-580-open
#   scripts/hardware/setup_laptop.sh --iface enx00e04c   # also set 192.168.10.1/24 on that port
#   scripts/hardware/setup_laptop.sh --no-sam2           # skip torch / SAM2 / weights
set -euo pipefail
root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root_dir"

install_driver=0
iface=""
with_sam2=1
laptop_ip="192.168.10.1/24"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --driver) install_driver=1 ;;
    --iface) iface="$2"; shift ;;
    --laptop-ip) laptop_ip="$2"; shift ;;
    --no-sam2) with_sam2=0 ;;
    -h|--help) sed -n 2,10p "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

step() { printf '\n== %s\n' "$*"; }

step "System packages"
sudo apt-get update
sudo apt-get install -y git python3-venv python3-pip python3-dev build-essential \
  ffmpeg ethtool iputils-ping network-manager rsync curl pciutils

step "NVIDIA driver"
if nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
elif [[ $install_driver == 1 ]]; then
  # Ubuntu's packaged open module, branch 580 (docs/nvidia_driver_plan.md).
  # Do not mix branches; reboot before using the GPU.
  sudo apt-get install -y "linux-headers-$(uname -r)" nvidia-driver-580-open
  echo "Driver installed: REBOOT, then run this script again." >&2
  exit 0
else
  echo "nvidia-smi does not work. Run again with --driver, or install the driver yourself." >&2
  echo "Continuing: everything except SAM2 works on the CPU." >&2
fi

step "Python venv (.venv)"
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv --system-site-packages .venv
fi
py=.venv/bin/python
"$py" -m pip install --upgrade pip wheel
"$py" -m pip install numpy==1.26.4 opencv-python==4.10.0.84 simplejpeg scipy matplotlib pytest

if [[ $with_sam2 == 1 ]]; then
  step "torch (CUDA 12.4 wheels) and SAM2"
  if ! "$py" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    "$py" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
  fi
  if ! "$py" -c 'import sam2' 2>/dev/null; then
    # The CUDA extension of sam2 is optional (only mask post-processing).
    SAM2_BUILD_CUDA=0 "$py" -m pip install "git+https://github.com/facebookresearch/sam2.git"
  fi
  step "SAM2 weights -> models/sam2/ (not in git)"
  mkdir -p models/sam2
  for name in sam2.1_hiera_tiny.pt sam2.1_hiera_small.pt; do
    if [[ ! -s models/sam2/$name ]]; then
      curl -fL --retry 3 -o "models/sam2/$name.part" \
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/$name"
      mv "models/sam2/$name.part" "models/sam2/$name"
    fi
  done
fi

if [[ -n "$iface" ]]; then
  step "Static address $laptop_ip on $iface"
  speed="$(ethtool "$iface" 2>/dev/null | sed -n 's/.*Speed: //p')"
  echo "link speed: ${speed:-unknown} (must be 1000Mb/s)"
  con="mocap-cameras"
  if ! nmcli -t -f NAME con show | grep -qx "$con"; then
    sudo nmcli con add type ethernet ifname "$iface" con-name "$con" \
      ipv4.method manual ipv4.addresses "$laptop_ip" ipv6.method disabled
  else
    sudo nmcli con mod "$con" ifname "$iface" ipv4.method manual ipv4.addresses "$laptop_ip"
  fi
  sudo nmcli con up "$con"
fi

step "Check"
"$py" - <<'EOF'
import cv2, numpy, scipy
print("opencv", cv2.__version__, "| numpy", numpy.__version__, "| scipy", scipy.__version__)
print("aruco/apriltag:", hasattr(cv2.aruco, "generateImageMarker"))
try:
    import torch
    print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
          "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
    import sam2
    print("sam2 ok")
except ImportError as e:
    print("torch/sam2 not installed:", e)
EOF
"$py" -m pytest tests/test_lan_capture.py -q
echo
echo "Laptop ready. Next: docs/hardware_runbook.md, section 3 (camera nodes)."
