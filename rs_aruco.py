import cv2
import numpy as np
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

color_stream = profile.get_stream(rs.stream.color)
intr = color_stream.as_video_stream_profile().get_intrinsics()

camera_matrix = np.array([
    [intr.fx, 0, intr.ppx],
    [0, intr.fy, intr.ppy],
    [0, 0, 1]
], dtype=np.float64)

dist_coeffs = np.array(intr.coeffs, dtype=np.float64)

print("camera_matrix:\n", camera_matrix)
print("dist_coeffs:", dist_coeffs)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, params)

marker_length = 0.10  # 10 cm

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        vis = image.copy()

        corners, ids, rejected = detector.detectMarkers(gray)

        detected_n = 0 if ids is None else len(ids)
        rejected_n = 0 if rejected is None else len(rejected)

        cv2.putText(vis, f"detected={detected_n} rejected={rejected_n}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, marker_length, camera_matrix, dist_coeffs
            )

            for i in range(len(ids)):
                marker_id = int(ids[i][0])

                cv2.drawFrameAxes(
                    vis,
                    camera_matrix,
                    dist_coeffs,
                    rvecs[i],
                    tvecs[i],
                    marker_length * 0.5
                )

                x, y, z = tvecs[i][0]
                print(f"ID={marker_id}  x={x:.3f} y={y:.3f} z={z:.3f}")

                p = corners[i][0][0]
                px, py = int(p[0]), int(p[1])

                cv2.putText(vis, f"id={marker_id}", (px, py - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(vis, f"x={x:.3f} y={y:.3f} z={z:.3f}",
                            (px, py - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("ArUco Pose Detection", vis)
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()