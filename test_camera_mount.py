import os
os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts"

import cv2
import numpy as np
import pyrealsense2 as rs

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

align = rs.align(rs.stream.color)
colorizer = rs.colorizer()

print("[INFO] Starting camera live preview. Press 'q' or 'ESC' to exit.")
pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(colorizer.colorize(depth_frame).get_data())

        # Draw workspace alignment crosshairs
        h, w, _ = color_img.shape
        cx, cy = w // 2, h // 2
        cv2.line(color_img, (cx, 0), (cx, h), (0, 255, 255), 1)
        cv2.line(color_img, (0, cy), (w, cy), (0, 255, 255), 1)

        # Center depth metric readout
        dist = depth_frame.get_distance(cx, cy)
        cv2.putText(
            color_img,
            f"Center Depth: {dist:.3f} m",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        # Combine side-by-side
        combined = np.hstack((color_img, depth_img))
        cv2.imshow("RealSense Mount Verification (RGB | Depth)", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()