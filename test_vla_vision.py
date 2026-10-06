import os
os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts"

import re
import cv2
import numpy as np
from PIL import Image
import torch
import pyrealsense2 as rs
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 1. Initialize RealSense with Aligned Color and Depth Streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)

profile = pipeline.start(config)

# Setup alignment object to map depth frame onto color frame coordinates
align_to = rs.stream.color
align = rs.align(align_to)

print("[INFO] Warming up RealSense sensors...")
for _ in range(30):
    frames = pipeline.wait_for_frames()

aligned_frames = align.process(frames)
color_frame = aligned_frames.get_color_frame()
depth_frame = aligned_frames.get_depth_frame()

color_image = np.asanyarray(color_frame.get_data())
pipeline.stop()

# Retrieve camera intrinsic parameters for 3D deprojection
color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

annotated_frame = color_image.copy()
img_h, img_w, _ = color_image.shape

# Convert to RGB PIL image for VLM
image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(image_rgb)
pil_img.save("vla_input_test.jpg")
print("[INFO] Captured test frame to 'vla_input_test.jpg'")

# 2. Load model safely onto Jetson GPU
model_id = "Qwen/Qwen2-VL-2B-Instruct"
print(f"[INFO] Loading {model_id}...")

processor = AutoProcessor.from_pretrained(model_id)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 3. Visual Grounding Prompt
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image", 
                "image": pil_img,
                "resized_height": 480,
                "resized_width": 640
            },
            {
                "type": "text", 
                "text": "Locate the silver metal bolt on the mat. Output the bounding box in normalized coordinates 0-1000: bolt: [xmin, ymin, xmax, ymax]."
            },
        ],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to("cuda")

print("[INFO] Running visual grounding inference...")
with torch.no_grad():
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=32,
        do_sample=False
    )

generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
result_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

print(f"\n--- VLA Grounding Output ---\n{result_text.strip()}\n---------------------------")

# 4. Parse Cartesian coordinates and project to 3D Camera Space
nums = re.findall(r"([0-9]*\.?[0-9]+)", result_text)

if len(nums) >= 4:
    xmin, ymin, xmax, ymax = map(float, nums[:4])
    scale = 1.0 if max(xmin, ymin, xmax, ymax) <= 1.0 else 1000.0
    
    left = int((xmin / scale) * img_w)
    top = int((ymin / scale) * img_h)
    right = int((xmax / scale) * img_w)
    bottom = int((ymax / scale) * img_h)

    # 2D pixel center
    center_u = int((left + right) / 2)
    center_v = int((top + bottom) / 2)

    # Query metric depth (Z in meters) at the object center
    depth_in_meters = depth_frame.get_distance(center_u, center_v)

    # If the center pixel hit a reflection or hole, search a 5x5 neighborhood for median depth
    if depth_in_meters == 0:
        depth_crop = []
        for du in range(-2, 3):
            for dv in range(-2, 3):
                d = depth_frame.get_distance(center_u + du, center_v + dv)
                if d > 0:
                    depth_crop.append(d)
        depth_in_meters = np.median(depth_crop) if depth_crop else 0.0

    # Deproject 2D pixel + Depth into 3D Camera Frame coordinates (X, Y, Z in meters)
    point_3d = rs.rs2_deproject_pixel_to_point(color_intrinsics, [center_u, center_v], depth_in_meters)
    cam_x, cam_y, cam_z = point_3d

    print(f"\n=== 3D Spatial Position (Camera Frame) ===")
    print(f"X (Right)   : {cam_x * 1000:.1f} mm")
    print(f"Y (Down)    : {cam_y * 1000:.1f} mm")
    print(f"Z (Forward) : {cam_z * 1000:.1f} mm")
    print(f"==========================================\n")

    # Draw box, center marker, and metric coordinates
    cv2.rectangle(annotated_frame, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.circle(annotated_frame, (center_u, center_v), 4, (0, 0, 255), -1)
    
    coord_label = f"X:{cam_x*100:.1f}cm Y:{cam_y*100:.1f}cm Z:{cam_z*100:.1f}cm"
    cv2.putText(
        annotated_frame,
        coord_label,
        (left, max(top - 10, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 0),
        2
    )

output_filename = "vla_grounding_output.jpg"
cv2.imwrite(output_filename, annotated_frame)
print(f"[INFO] Annotated image saved to '{output_filename}'")

cv2.imshow("VLA Grounding Result (Press any key to close)", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()