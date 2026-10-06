#!/usr/bin/env bash
# Train an ACT policy on a dataset recorded by scripts/record.sh.
#
# Usage: scripts/train.sh <task-name> [extra lerobot-train args...]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/_rig.sh

TASK="${1:?usage: scripts/train.sh <task-name> [extra lerobot-train args...]}"
shift || true

lerobot-train \
  --dataset.repo_id="${HF_USER}/${TASK}" \
  --policy.type=act \
  --policy.device=cuda \
  --output_dir="outputs/train/act_${TASK}" \
  --job_name="act_${TASK}" \
  --batch_size=4 \
    --steps=25000 \
    --num_workers=2 \
  --policy.repo_id="${HF_USER}/act_${TASK}" \
  "$@"
