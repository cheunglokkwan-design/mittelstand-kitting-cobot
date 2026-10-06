import torch
import pyrealsense2 as rs
import numpy as np
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

def test_qwen(serial="035422070365"):
    print("[1/3] Capturing snapshot from RealSense...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    pipeline.start(config)

    # Let auto-exposure settle
    for _ in range(25):
        frames = pipeline.wait_for_frames()

    img_data = np.asanyarray(frames.get_color_frame().get_data())
    pipeline.stop()

    image = Image.fromarray(img_data)
    image.save("qwen_test_view.jpg")
    print("      Frame captured and saved as 'qwen_test_view.jpg'.")

    print("[2/3] Loading Qwen2-VL-2B-Instruct...")
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)

    print("[3/3] Running perception prompt...")
    prompt = (
        "Look at the green cutting mat. There is a bolt and a brass connector. "
        "Describe where the bolt is and where the brass connector is. "
        "Which object is on the left, and which is on the right?"
    )

    conversation = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": prompt}]
    }]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=120)
        result = processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]

    print("\n--- QWEN PERCEPTION OUTPUT ---")
    print(result.strip())
    print("------------------------------\n")

if __name__ == "__main__":
    test_qwen()
