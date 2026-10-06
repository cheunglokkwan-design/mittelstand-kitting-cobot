# Mittelstand Micro-Kitting Cobot

NVIDIA GTC Berlin Golden Ticket Developer Contest entry. A desktop robot that
watches a workspace via an Intel RealSense D435i and sorts small
mechanical/electronic parts into labeled kitting trays with an SO-101 arm — a
**LeRobot ACT policy**, trained from teleoperated demonstrations, drives the
whole pick-and-place motion end-to-end (no hand-coded kinematics). Runs
on-device on a Jetson Orin Nano Super.

**Why:** Germany's Mittelstand manufacturing sector has a well-documented
skilled-labor shortage (~16.6% vacancy rate in electrical trades,
mechatronics, and industrial maintenance). Parts kitting is repetitive prep
work that pulls skilled workers away from higher-value tasks; automating it
targets a real, current bottleneck.

## Rig

| Component | Details |
|---|---|
| Arm | SO-101 follower + leader (teleop), Feetech STS3215 |
| Camera | Intel RealSense D435i, serial `035422070365` |
| Compute | Jetson Orin Nano Super |
| Follower port | `/dev/ttyACM0`, calibration id `follower_arm` |
| Leader port | `/dev/ttyACM1`, calibration id `leader_arm` |

Port assignment can shift on replug/reboot — if a script talks to the wrong
arm, re-confirm with `lerobot-find-port` before continuing.

## Setup

```bash
source ~/Projects/LokKwan/venv/bin/activate
hf auth login              # needed to push/pull datasets & policy checkpoints
```

Motor calibration for both arms already exists in
`~/.cache/huggingface/lerobot/calibration/`. Re-run only if a motor was
replaced or the arm stops tracking correctly:

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=follower_arm
lerobot-calibrate --teleop.type=so101_leader  --teleop.port=/dev/ttyACM1 --teleop.id=leader_arm
```

## Usage

There's no custom orchestration script — recording demonstrations and
running the trained policy both go straight through LeRobot's own CLI.

```bash
scripts/teleoperate.sh                       # sanity-check the leader/follower link + camera feed
scripts/record.sh screws_v1 "Pick up the screw and place it in the screws tray" 50 15
scripts/train.sh screws_v1
scripts/rollout.sh screws_v1 <hf_user>/act_screws_v1 10   # run the trained policy live
```

See each script for the underlying `lerobot-record` / `lerobot-train` /
`lerobot-rollout` invocation — they're thin wrappers pinning this rig's
ports/camera/ids, not a replacement for reading
[`AGENT_GUIDE.md`](../AGENT_GUIDE.md) in the LeRobot checkout one level up,
which has the full cheat sheet (data collection tips, policy choice, how long
to train, troubleshooting).

**Recommended path (see `AGENT_GUIDE.md` §5.5 "start small, then extend"):**
record ~50 clean episodes for **one** category/tray first, train, and check
the loop works end-to-end on hardware before recording the other three
categories.

## Results

_TODO: before/after manual-sort timing comparison, per-category success
rate, demo video link._

## Limitations

Single part per frame only, type-level classification (not size/thread),
parts below ~5mm are out of scope. See the PRD for the full non-goals list.

> **Note:** `PRD.md` still describes the earlier hand-coded IK / VLM-classification
> architecture and hasn't been updated for the ACT pivot yet.
