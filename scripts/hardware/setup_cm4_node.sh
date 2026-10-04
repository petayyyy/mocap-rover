#!/usr/bin/env bash
# One Raspberry Pi CM4/CM5 camera node (Raspberry Pi OS Lite 64-bit, Bookworm).
# Run ON the Pi, from a clone of the repository:
#
#   sudo scripts/hardware/setup_cm4_node.sh 1          # camera_1, 192.168.10.101, PTP master
#   sudo scripts/hardware/setup_cm4_node.sh 4          # camera_4, 192.168.10.104, PTP slave
#   sudo scripts/hardware/setup_cm4_node.sh 4 --cm5    # CM5: "isp" stream instead of "raw"
#   sudo scripts/hardware/setup_cm4_node.sh 4 --cpp    # the C++ node (one H.264 stream), the main scenario
#
# Installs packages, enables the IMX219 overlay, sets the static address,
# deploys pi_cam/ to /opt/mocap-rover, writes /etc/mocap-rover/node_config.json,
# and enables camera_node + ptp4l + phc2sys.  Reboot when it asks.
#
# --cpp additionally builds cpp/ on the node, installs /usr/local/bin/mocap_camd,
# writes /etc/mocap-rover/camd.json (kept if it exists) and enables
# mocap_camd instead of camera_node (the two cannot share the camera; the
# Python node stays installed for the calibration station:
# "sudo systemctl stop mocap_camd; sudo systemctl start camera_node").
set -euo pipefail
if [[ $EUID -ne 0 ]]; then echo "run with sudo" >&2; exit 1; fi
n="${1:?camera number 1..6}"; shift
[[ "$n" =~ ^[1-9]$ ]] || { echo "camera number must be 1..9" >&2; exit 2; }
board=cm4
cpp=0
iface=eth0
subnet=192.168.10
user_name="${SUDO_USER:-pi}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cm5) board=cm5 ;;
    --cpp) cpp=1 ;;
    --iface) iface="$2"; shift ;;
    --subnet) subnet="$2"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ip="$subnet.$((100 + n))"
need_reboot=0
step() { printf '\n== %s\n' "$*"; }

step "Packages"
apt-get update
apt-get install -y python3-picamera2 python3-numpy python3-simplejpeg linuxptp \
  ethtool python3-lgpio python3-opencv git rsync   # opencv: demosaic for calibration frames

step "Camera overlay (CM4-NANO-B does not auto-detect the IMX219)"
cfg=/boot/firmware/config.txt
if ! grep -q '^dtoverlay=imx219' "$cfg"; then
  sed -i 's/^camera_auto_detect=1/camera_auto_detect=0/' "$cfg"
  grep -q '^camera_auto_detect=0' "$cfg" || echo 'camera_auto_detect=0' >> "$cfg"
  echo 'dtoverlay=imx219,cam0' >> "$cfg"
  need_reboot=1
fi

step "Static address $ip/24 on $iface"
con="$(nmcli -t -f NAME,DEVICE con show | awk -F: -v d="$iface" '$2==d{print $1; exit}')"
con="${con:-Wired connection 1}"
nmcli con mod "$con" ipv4.method manual ipv4.addresses "$ip/24" ipv6.method disabled
nmcli con up "$con" || need_reboot=1

step "Camera node -> /opt/mocap-rover/pi_cam"
mkdir -p /opt/mocap-rover /etc/mocap-rover
rsync -a --delete --exclude __pycache__ "$repo/pi_cam/" /opt/mocap-rover/pi_cam/
conf=/etc/mocap-rover/node_config.json
if [[ ! -f $conf ]]; then
  stream=raw; buffers=4
  [[ $board == cm5 ]] && { stream=isp; buffers=6; }
  python3 - "$repo/pi_cam/node_config.example.json" "$conf" "$n" "$stream" "$buffers" <<'EOF'
import json, sys
src, dst, n, stream, buffers = sys.argv[1:]
c = json.load(open(src))
# CM4: the ISP delivers only every second frame, so read the raw Bayer stream.
c.update(camera_id=f"camera_{n}", stream=stream, buffer_count=int(buffers),
         exposure_us=800, analogue_gain=4.0, send_queue=3)
json.dump(c, open(dst, "w"), indent=2)
EOF
  echo "wrote $conf (edit stamp_correction_ns / line_time_ns after the LED probe)"
else
  echo "$conf exists, left unchanged"
fi
sed "s/^User=pi/User=$user_name/" "$repo/pi_cam/camera_node.service" > /etc/systemd/system/camera_node.service

step "PTP: camera_1 is the master, the others are slaves (never the laptop)"
ptpconf=/etc/linuxptp/mocap-ptp4l.conf
{
  echo "[global]"
  echo "tx_timestamp_timeout 50"      # CM4/CM5 MAC is slow; without it ptp4l goes FAULTY
  if [[ $n == 1 ]]; then echo "masterOnly 1"; echo "priority1 10"; else echo "slaveOnly 1"; fi
} > "$ptpconf"
cat > /etc/systemd/system/ptp4l.service <<EOF
[Unit]
Description=PTP (mocap-rover, $iface)
After=network-online.target
Wants=network-online.target
[Service]
ExecStart=/usr/sbin/ptp4l -i $iface -f $ptpconf -m
Restart=always
[Install]
WantedBy=multi-user.target
EOF
phc_args="-a -r"
[[ $n == 1 ]] && phc_args="-a -r -r"   # master: system clock drives the PHC
cat > /etc/systemd/system/phc2sys.service <<EOF
[Unit]
Description=PHC <-> system clock (mocap-rover)
After=ptp4l.service
Requires=ptp4l.service
[Service]
ExecStart=/usr/sbin/phc2sys $phc_args -m
Restart=always
[Install]
WantedBy=multi-user.target
EOF
# The node polls pmc as the service user.
echo "$user_name ALL=(root) NOPASSWD: /usr/sbin/pmc" > /etc/sudoers.d/mocap-pmc
chmod 440 /etc/sudoers.d/mocap-pmc
usermod -aG video "$user_name"

if [[ $cpp == 1 ]]; then
  step "C++ node mocap_camd (one H.264 stream per camera)"
  apt-get install -y cmake g++ pkg-config libopencv-dev nlohmann-json3-dev
  build="$repo/cpp/build-node"
  sudo -u "$user_name" cmake -S "$repo/cpp" -B "$build" -DMOCAP_BUILD_NODE=ON -DMOCAP_BUILD_LAPTOP=OFF
  sudo -u "$user_name" cmake --build "$build" -j4
  install -m 755 "$build/mocap_camd" /usr/local/bin/mocap_camd
  camd=/etc/mocap-rover/camd.json
  if [[ ! -f $camd ]]; then
    python3 - "$repo/cpp/node/camd.example.json" "$camd" "$n" <<'EOF'
import json, sys
src, dst, n = sys.argv[1:]
c = json.load(open(src))
c["camera_id"] = f"camera_{n}"
json.dump(c, open(dst, "w"), indent=2)
EOF
  fi
  chown "$user_name" "$camd"   # the web page's "save" writes it
  sed "s/^User=pi/User=$user_name/" "$repo/cpp/node/mocap_camd.service" > /etc/systemd/system/mocap_camd.service
fi

systemctl daemon-reload
systemctl enable ptp4l phc2sys
if [[ $cpp == 1 ]]; then
  systemctl disable camera_node 2>/dev/null || true
  systemctl stop camera_node 2>/dev/null || true
  systemctl enable mocap_camd
  node_service=mocap_camd
else
  systemctl enable camera_node
  node_service=camera_node
fi

step "Check"
ethtool -T "$iface" | grep -E 'PTP Hardware Clock|hardware-transmit' || true
if [[ $need_reboot == 1 ]]; then
  echo "REBOOT now (sudo reboot), then:"
else
  systemctl restart ptp4l phc2sys $node_service
  sleep 5
fi
if [[ $cpp == 1 ]]; then
cat <<EOF
  journalctl -u mocap_camd -n 20            # 'camera 1640x1232 SBGGR8 via unicam', 'web page on :8080'
  open http://$ip:8080                      # live picture, exposure, resolution, colour, bitrate
  sudo pmc -u -b 0 'GET PORT_DATA_SET'      # MASTER on camera_1, SLAVE elsewhere
node: camera_$n  $ip:5600 (stream), :8080 (web)  board $board, C++ node
EOF
else
cat <<EOF
  rpicam-hello --list-cameras               # imx219, mode 1640x1232
  journalctl -u camera_node -n 30           # 'Selected sensor format: 1640x1232-SBGGR8_1X8'
  sudo pmc -u -b 0 'GET PORT_DATA_SET'      # MASTER on camera_1, SLAVE elsewhere
node: camera_$n  $ip:5600  board $board
EOF
fi
