#!/usr/bin/env bash
# Sanity-check the leader/follower link + camera feed. No recording.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/_rig.sh

lerobot-teleoperate \
  --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" --robot.id="$FOLLOWER_ID" \
  --teleop.type=so101_leader  --teleop.port="$LEADER_PORT"  --teleop.id="$LEADER_ID" \
  --robot.cameras="$CAMERAS" \
  --display_data=true
