import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import cv2
import numpy as np
import matplotlib.pyplot as plt

from main import move_robot, WHEEL_RADIUS, WHEEL_BASE, DT
from estimation.ekf import RobotEKF
from sensors.lidar import read_lidar

Path("results").mkdir(exist_ok=True)

model = mujoco.MjModel.from_xml_path("models/turtlebot3.xml")
data = mujoco.MjData(model)

# Start near the station, as in the successful navigation test.
data.qpos[0] = 0.9
data.qpos[1] = 1.8
data.qpos[2] = 0.10

# Face approximately towards the marker.
theta = 0.0
data.qpos[3:7] = [np.cos(theta/2), 0, 0, np.sin(theta/2)]

mujoco.mj_forward(model, data)

ekf = RobotEKF(
    initial_state=(0.9, 1.8, theta),
    wheel_base=WHEEL_BASE
)

renderer = mujoco.Renderer(model, width=640, height=480)

options = mujoco.MjvOption()
options.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)
detector = cv2.aruco.ArucoDetector(dictionary)

height, width = 480, 640
fovy = np.deg2rad(70)
f = (height / 2) / np.tan(fovy / 2)

K = np.array([
    [f, 0, width/2],
    [0, f, height/2],
    [0, 0, 1]
], dtype=np.float64)

half = 0.10
object_points = np.array([
    [-half, half, 0],
    [half, half, 0],
    [half, -half, 0],
    [-half, -half, 0]
], dtype=np.float32)

landmark = np.array([1.9, 1.8])
target_range = 0.55

previous_wheels = data.qpos[7:9].copy()
trajectory = []
detections = 0
docked = False
status = "TIMEOUT"
last_valid_measurement = None
missed_frames = 0

for step in range(1200):
    renderer.update_scene(
        data,
        camera="robot_camera",
        scene_option=options
    )
    image = renderer.render()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    corners, ids, _ = detector.detectMarkers(gray)

    measurement = None

    if ids is not None and 0 in ids.flatten():
        index = list(ids.flatten()).index(0)
        points = corners[index][0]

        pixel_width = np.linalg.norm(
            points[1] - points[0]
        )

        centre_x = np.mean(points[:, 0])

        if pixel_width > 5:
            # Pinhole camera distance estimate.
            camera_distance = f * 0.20 / pixel_width

            # Camera is mounted 0.10 m ahead of robot centre.
            forward = camera_distance + 0.10

            # Bearing from marker centre in image.
            measured_bearing = -np.arctan2(
                centre_x - width / 2,
                f
            )

            measured_range = forward / max(
                np.cos(measured_bearing), 0.1
            )

            if (
                np.isfinite(measured_range)
                and np.isfinite(measured_bearing)
            ):
                measurement = (
                    measured_range,
                    measured_bearing
                )

    if measurement is None:
        missed_frames += 1

        if missed_frames > 10 or last_valid_measurement is None:
            status = "CAMERA TRACKING LOST"
            print("Camera tracking lost. Stopping safely.")
            break

        # Hold position while waiting for a fresh measurement.
        continue

    missed_frames = 0
    last_valid_measurement = measurement
    measured_range, measured_bearing = measurement

    ekf.update_landmark(
        measured_range,
        measured_bearing,
        landmark
    )

    if not np.all(np.isfinite(ekf.x)):
        status = "INVALID EKF STATE"
        break

    detections += 1

    if measured_range <= target_range + 0.02:
        docked = True
        status = "DOCKED"
        break

    # Slow proportional camera-guided control.
    v = np.clip(
        0.5 * (measured_range - target_range),
        0.0,
        0.08
    )

    omega = np.clip(
        1.5 * measured_bearing,
        -0.5,
        0.5
    )

    if abs(measured_bearing) > np.deg2rad(15):
        v = 0.0

    lidar = read_lidar(model, data)

    if np.min(lidar) < 0.20:
        status = "LIDAR SAFETY STOP"
        print("LiDAR safety stop activated.")
        break

    wl = (v - omega * WHEEL_BASE/2) / WHEEL_RADIUS
    wr = (v + omega * WHEEL_BASE/2) / WHEEL_RADIUS

    if not np.all(np.isfinite([wl, wr])):
        status = "INVALID WHEEL COMMAND"
        break

    move_robot(model, data, wl, wr, DT)

    if not np.all(np.isfinite(data.qpos[:9])):
        status = "INVALID ROBOT STATE"
        break

    wheels = data.qpos[7:9].copy()
    dl, dr = (wheels - previous_wheels) * WHEEL_RADIUS
    previous_wheels = wheels

    ekf.predict(dl, dr)
    trajectory.append(data.qpos[:2].copy())

    if step % 100 == 0:
        print(
            f"Step {step}: "
            f"range={measured_range:.3f}m, "
            f"bearing={np.degrees(measured_bearing):.1f}deg"
        )

renderer.close()

trajectory = np.array(trajectory)

if len(trajectory):
    plt.figure(figsize=(7, 6))
    plt.plot(trajectory[:, 0], trajectory[:, 1], label="Docking path")
    plt.scatter(*landmark, marker="*", s=180, label="ArUco station")
    plt.scatter(0.9, 1.8, label="Docking start")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title("Camera-Guided Autonomous Docking")
    plt.savefig("results/docking_trajectory.png", dpi=160)
    plt.close()

print("\nDOCKING RESULTS")
print("Status:", status)
print("Docked:", docked)
print("Marker detections:", detections)
print("Final position:", np.round(data.qpos[:2], 3))
print("Distance to marker:", round(np.linalg.norm(landmark - data.qpos[:2]), 3), "m")
