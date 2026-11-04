import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import cv2
import numpy as np

from estimation.ekf import RobotEKF
from main import move_robot, WHEEL_RADIUS, WHEEL_BASE, DT

Path("results").mkdir(exist_ok=True)

model = mujoco.MjModel.from_xml_path("models/turtlebot3.xml")
data = mujoco.MjData(model)

data.qpos[0] = 0.9
data.qpos[1] = 1.8
data.qpos[2] = 0.10
mujoco.mj_forward(model, data)

ekf = RobotEKF(
    initial_state=(0.9, 1.8, 0.0),
    wheel_base=WHEEL_BASE
)

renderer = mujoco.Renderer(model, width=640, height=480)

options = mujoco.MjvOption()
options.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)
detector = cv2.aruco.ArucoDetector(dictionary)

width, height = 640, 480
fovy = np.deg2rad(70)
fy = (height / 2) / np.tan(fovy / 2)
fx = fy

K = np.array([
    [fx, 0, width / 2],
    [0, fy, height / 2],
    [0, 0, 1]
], dtype=np.float64)

dist = np.zeros(5)
marker_size = 0.20
half = marker_size / 2

object_points = np.array([
    [-half, half, 0],
    [half, half, 0],
    [half, -half, 0],
    [-half, -half, 0]
], dtype=np.float32)

landmark_xy = np.array([1.9, 1.8])

previous_wheels = data.qpos[7:9].copy()
detections = 0

for step in range(100):
    move_robot(model, data, 1.0, 1.0, DT)

    wheels = data.qpos[7:9].copy()
    dl, dr = (wheels - previous_wheels) * WHEEL_RADIUS
    previous_wheels = wheels

    ekf.predict(dl, dr)

    if step % 10 != 0:
        continue

    renderer.update_scene(
        data,
        camera="robot_camera",
        scene_option=options
    )
    image = renderer.render()

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or 0 not in ids.flatten():
        continue

    index = list(ids.flatten()).index(0)
    image_points = corners[index][0].astype(np.float32)

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        K,
        dist,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )

    if not success:
        continue

    tvec = tvec.flatten()

    forward = 0.10 + tvec[2]
    left = -tvec[0]

    measured_range = np.hypot(forward, left)
    measured_bearing = np.arctan2(left, forward)

    ekf.update_landmark(
        measured_range,
        measured_bearing,
        landmark_xy
    )

    detections += 1

    print(
        f"Step {step}: "
        f"range={measured_range:.3f}m, "
        f"bearing={np.degrees(measured_bearing):.1f}deg"
    )

renderer.close()

true_xy = data.qpos[:2]
estimated_xy = ekf.x[:2]

error = np.linalg.norm(true_xy - estimated_xy)

print("\nCAMERA + EKF RESULTS")
print("Marker detections:", detections)
print("True position:", np.round(true_xy, 3))
print("EKF position:", np.round(estimated_xy, 3))
print(f"Final position error: {error:.4f} m")
