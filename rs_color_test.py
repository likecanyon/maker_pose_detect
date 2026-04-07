import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

# Get color intrinsics
color_stream = profile.get_stream(rs.stream.color)
intr = color_stream.as_video_stream_profile().get_intrinsics()

print("fx fy ppx ppy =", intr.fx, intr.fy, intr.ppx, intr.ppy)
print("dist coeffs =", intr.coeffs)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        image = np.asanyarray(color_frame.get_data())
        cv2.imshow("RealSense Color", image)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()