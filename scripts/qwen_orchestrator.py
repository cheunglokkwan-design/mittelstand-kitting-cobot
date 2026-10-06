import os
import subprocess
import torch
import numpy as np
import pyrealsense2 as rs
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

def capture_overhead_scene(serial="035422070365"):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    
    print("[QWEN] Opening RealSense overhead camera...")
    pipeline.start(config)
    for _ in range(15):  # settle exposure
        frames = pipeline.wait_for_frames()
    
    color_frame = frames.get_color_frame()
    img = Image.fromarray(np.asanyarray(color_frame.get_data()))
    pipeline.stop()
    return img


def classify_target(image):
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    print(f"[QWEN] Running visual inference using {model_id}...")
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)

    prompt = (
        "Look at the green kitting mat. Which target part should be picked: "
        "a 'bolt' or a 'brass connector'? "
        "Answer with strictly either 'bolt' or 'brass connector'."
    )

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        output_text = processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )[0].strip().lower()

    # Free GPU memory before starting the ACT policy
    del model
    del processor
    torch.cuda.empty_cache()

    return output_text


def dispatch_rollout(decision):
    print(f"\n[QWEN DECISION] Detected target: '{decision}'")

    if "connector" in decision:
        task_name = "kitting_connector"
        policy_path = "outputs/train/act_kitting_connector/checkpoints/last/pretrained_model"
    else:
        task_name = "kitting_bolt"
        policy_path = "outputs/train/act_kitting_bolt/checkpoints/last/pretrained_model"

    print(f"[DISPATCH] Triggering rollout.sh with task='{task_name}', policy='{policy_path}'")
    
    # Executes your exact rollout.sh with the required positional arguments
    subprocess.run(["bash", "scripts/rollout.sh", task_name, policy_path, "1"])

if __name__ == "__main__":
    frame = capture_overhead_scene()
    decision = classify_target(frame)
    dispatch_rollout(decision)
