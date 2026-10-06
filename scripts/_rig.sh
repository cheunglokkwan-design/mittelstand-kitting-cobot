# Shared rig constants, sourced by the other scripts/*.sh — not run directly.
# Ports can shift on replug/reboot; re-confirm with `lerobot-find-port` if a
# script ever seems to be talking to the wrong arm.

FOLLOWER_PORT="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B7B014143-if00"
LEADER_PORT="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B7B015096-if00"
FOLLOWER_ID="${FOLLOWER_ID:-follower_arm}"
LEADER_ID="${LEADER_ID:-leader_arm}"
CAMERA_SERIAL="${CAMERA_SERIAL:-035422070365}"
WRIST_INDEX="${WRIST_INDEX:-6}"
CAMERAS="{ front: {type: intelrealsense, serial_number_or_name: '${CAMERA_SERIAL}', width: 640, height: 480, fps: 15}, wrist: {type: opencv, index_or_path: ${WRIST_INDEX}, width: 640, height: 480, fps: 30} }"

if [ -z "${HF_USER:-}" ]; then
  HF_USER="$(hf auth whoami --quiet 2>/dev/null)" || true
fi
if [ -z "$HF_USER" ]; then
  echo "error: not logged into Hugging Face and HF_USER is unset. Run 'hf auth login' or set HF_USER=<name>." >&2
  exit 1
fi
