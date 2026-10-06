#!/usr/bin/env bash
# Record teleop demonstrations into a LeRobot dataset.
#
# Usage: scripts/record.sh <task-name> "<task description>" [num_episodes] [episode_time_s]
#
# Keys while recording: -> next episode (end it early — episode_time_s is just
# a cap), <- redo last episode, ESC finish. Keep episode_time_s tight to this
# task's actual pick-and-place duration: a long cap invites idle/held-still
# frames if you're slow on the key, which teaches the policy hesitation.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/_rig.sh

TASK="${1:?usage: scripts/record.sh <task-name> \"<task description>\" [num_episodes] [episode_time_s]}"
DESCRIPTION="${2:?usage: scripts/record.sh <task-name> \"<task description>\" [num_episodes] [episode_time_s]}"
NUM_EPISODES="${3:-50}"
EPISODE_TIME_S="${4:-15}"

lerobot-record \
    --dataset.fps=15 \
  --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" --robot.id="$FOLLOWER_ID" \
  --teleop.type=so101_leader  --teleop.port="$LEADER_PORT"  --teleop.id="$LEADER_ID" \
  --robot.cameras="$CAMERAS" \
  --dataset.repo_id="${HF_USER}/${TASK}" \
  --dataset.no_stamp=true \
  --resume=false \
  --dataset.root="$HOME/.cache/huggingface/lerobot/${HF_USER}/${TASK}" \
  --dataset.single_task="$DESCRIPTION" \
  --dataset.num_episodes="$NUM_EPISODES" \
  --dataset.episode_time_s="$EPISODE_TIME_S" \
  --dataset.reset_time_s=10 \
  --display_data=true
