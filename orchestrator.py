import base64
import json
import os
import subprocess
import time
import cv2
import numpy as np
from openai import OpenAI
import pyrealsense2 as rs

client = OpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1"),
    api_key=os.environ.get("OPENAI_API_KEY", "ollama"),
)

CONNECTOR_POLICY = "outputs/train/act_kitting_connector_robust/checkpoints/050000/pretrained_model"
BOLT_POLICY = "outputs/train/act_kitting_bolt_robust/checkpoints/040000/pretrained_model"


def capture_snapshot():
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipe.start(cfg)

    # Let exposure settle
    for _ in range(15):
        frames = pipe.wait_for_frames()

    color_frame = frames.get_color_frame()
    img = np.asanyarray(color_frame.get_data())
    pipe.stop()

    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


def inspect_scene_with_qwen(base64_image):
    prompt = """
    Analyze this tabletop workspace for robotic kitting:
    - Brass connector belongs in the orange container.
    - Steel bolt belongs in the blue container.

    Determine which action must be taken next:
    - "SORT_CONNECTOR": If a brass connector is still on the mat needing sorting.
    - "SORT_BOLT": If a bolt is still on the mat needing sorting.
    - "IDLE": If all parts are already sorted into their respective trays, or no parts are left on the mat.

    Respond ONLY with valid JSON in this format:
    {"action": "SORT_CONNECTOR" | "SORT_BOLT" | "IDLE", "reason": "brief rationale"}
    """

    response = client.chat.completions.create(
        model=os.environ.get("VLM_MODEL", "qwen2.5vl:3b"),
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
        temperature=0.1,
    )

    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content)


def run_lerobot_policy(policy_path, task_name, duration=25):
    follower_port = os.environ.get("FOLLOWER_PORT")
    follower_id = os.environ.get("FOLLOWER_ID")
    cameras = os.environ.get("CAMERAS")

    if not follower_port or not cameras:
        raise ValueError("Missing hardware environment variables! Run 'source scripts/_rig.sh' first.")

    cmd = [
        "lerobot-rollout",
        "--strategy.type=base",
        f"--policy.path={policy_path}",
        "--robot.type=so101_follower",
        f"--robot.port={follower_port}",
        f"--robot.id={follower_id}",
        f"--robot.cameras={cameras}",
        f"--task={task_name}",
        f"--duration={duration}",
        "--display_data=false",
    ]
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    print("Starting Kitting Orchestrator Loop...")
    while True:
        print("\n--- Inspecting Workspace ---")
        img_b64 = capture_snapshot()
        decision = inspect_scene_with_qwen(img_b64)
        action = decision.get("action")
        reason = decision.get("reason")
        print(f"Qwen Assessment: {action} ({reason})")

        if action == "SORT_CONNECTOR":
            print("Dispatching Brass Connector ACT Policy...")
            run_lerobot_policy(
                CONNECTOR_POLICY,
                "Sort brass connector to orange container",
                duration=25,
            )
        elif action == "SORT_BOLT":
            print("Dispatching Bolt ACT Policy...")
            run_lerobot_policy(
                BOLT_POLICY,
                "Sort bolt to blue container",
                duration=25,
            )
        elif action == "IDLE":
            print("Workspace clear or all parts sorted. Standing by...")
            time.sleep(5)
            break
        else:
            print(f"Unknown action {action}, retrying...")
            time.sleep(2)


if __name__ == "__main__":
    main()
