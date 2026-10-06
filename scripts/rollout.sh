#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/_rig.sh

TASK="${1:?usage: scripts/rollout.sh <task-name> <policy-repo-id-or-checkpoint-path> [num_episodes]}"
POLICY="${2:?usage: scripts/rollout.sh <task-name> <policy-repo-id-or-checkpoint-path> [num_episodes]}"
NUM_EPISODES="${3:-10}"

lerobot-rollout \
    --robot.type=so101_follower \
    --robot.port="$FOLLOWER_PORT" \
    --robot.id="$FOLLOWER_ID" \
    --robot.cameras="$CAMERAS" \
    --policy.path="$POLICY" \
    --fps=15 \
    --inference.type=sync
