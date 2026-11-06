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
mujoco.mj_forward(model, data)

ekf = RobotEKF(
    initial_state=(-1.8, -1.8, 0.0),
    wheel_base=WHEEL_BASE
)

waypoints = np.array([
    [-0.8, -1.6],
    [0.1, -1.4],
    [1.5, -1.4],
    [1.7, -0.4],
    [1.7, 0.8],
    [0.9, 1.8]
])

landmark = np.array([1.9, 1.8])
target_range = 0.55

renderer = mujoco.Renderer(model, width=640, height=480)

video_renderer = mujoco.Renderer(
    model, width=960, height=540
)

video_options = mujoco.MjvOption()
video_options.flags[
    mujoco.mjtVisFlag.mjVIS_RANGEFINDER
] = 0

video_path = "results/integrated_demo.mp4"

video_writer = cv2.VideoWriter(
    video_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    25,
    (1280, 720)
)

if not video_writer.isOpened():
    raise RuntimeError("Could not open MP4 video writer.")

def capture_frame(label):
    video_renderer.update_scene(
        data, camera="overview",
        scene_option=video_options
    )
    overview = video_renderer.render()
    frame = cv2.cvtColor(
        overview, cv2.COLOR_RGB2BGR
    )

    frame = cv2.resize(
        frame, (1280, 720),
        interpolation=cv2.INTER_LINEAR
    )

    overlay = frame.copy()
    cv2.rectangle(
        overlay, (0, 0), (1280, 110),
        (15, 23, 35), -1
    )
    frame = cv2.addWeighted(
        overlay, 0.85, frame, 0.15, 0
    )

    cv2.putText(
        frame, "TURTLEBOT3 | AUTONOMOUS DOCKING",
        (30, 43), cv2.FONT_HERSHEY_SIMPLEX,
        0.9, (255, 255, 255), 2
    )
    cv2.putText(
        frame,
        f"Phase: {label}  |  Waypoints: {globals().get('waypoint_index', 0)}/6",
        (30, 85), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (120, 230, 255), 2
    )

    if label in ("DOCKING", "DOCKED"):
        renderer.update_scene(
            data, camera="robot_camera",
            scene_option=options
        )
        camera_rgb = renderer.render()
        camera_bgr = cv2.cvtColor(
            camera_rgb, cv2.COLOR_RGB2BGR
        )
        camera_bgr = cv2.resize(
            camera_bgr, (360, 270)
        )

        x0, y0 = 900, 425
        frame[y0:y0+270, x0:x0+360] = camera_bgr

        cv2.rectangle(
            frame, (x0, y0),
            (x0+360, y0+270),
            (255, 255, 255), 2
        )
        cv2.putText(
            frame, "ROBOT CAMERA",
            (x0+10, y0+25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 2
        )

    if label == "DOCKED":
        cv2.rectangle(
            frame, (20, 625), (500, 700),
            (15, 80, 35), -1
        )
        cv2.putText(
            frame, "DOCKING SUCCESSFUL",
            (40, 675),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9, (255, 255, 255), 2
        )

    video_writer.write(frame)


# Camera rendering and ArUco detection configuration.
options = mujoco.MjvOption()
options.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)
detector = cv2.aruco.ArucoDetector(dictionary)

height, width = 480, 640
fovy = np.deg2rad(70)
f = (height / 2) / np.tan(fovy / 2)

previous_wheels = data.qpos[7:9].copy()
trajectory = []
estimated = []

waypoint_index = 0
phase = "NAVIGATION"
detections = 0
safety_stops = 0
missed_frames = 0
status = "TIMEOUT"

def wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))

for _ in range(25):
    capture_frame("START")

for step in range(12000):
    pose = ekf.x
    lidar = read_lidar(model, data)

    if phase == "NAVIGATION":
        goal = waypoints[waypoint_index]

        dx = goal[0] - pose[0]
        dy = goal[1] - pose[1]
        distance = np.hypot(dx, dy)

        if distance < 0.10:
            print(f"Waypoint {waypoint_index + 1} reached")
            waypoint_index += 1

            if waypoint_index == len(waypoints):
                phase = "ALIGNMENT"
                print("Navigation complete. Aligning with ArUco marker.")

            continue

        heading = np.arctan2(dy, dx)
        error = wrap_angle(heading - pose[2])

        v = min(0.18, 0.8 * distance)
        omega = np.clip(2.0 * error, -1.2, 1.2)

        if abs(error) > np.deg2rad(45):
            v = 0.0

        front = lidar[
            np.abs(wrap_angle(
                np.linspace(0, 2*np.pi, len(lidar), endpoint=False)
            )) < np.deg2rad(30)
        ]

        if len(front) and np.min(front) < 0.28:
            v = 0.0
            safety_stops += 1

    elif phase == "ALIGNMENT":
        heading = np.arctan2(
            landmark[1] - pose[1],
            landmark[0] - pose[0]
        )
        error = wrap_angle(heading - pose[2])

        v = 0.0
        omega = np.clip(2.0 * error, -0.6, 0.6)

        if abs(error) < np.deg2rad(3):
            phase = "DOCKING"
            print("Aligned. Starting camera-guided docking.")
            continue

    elif phase == "DOCKING":
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
                camera_distance = f * 0.20 / pixel_width
                bearing = -np.arctan2(
                    centre_x - width/2, f
                )
                measured_range = (
                    camera_distance + 0.10
                ) / max(np.cos(bearing), 0.1)

                if np.all(np.isfinite([
                    measured_range, bearing
                ])):
                    measurement = (
                        measured_range, bearing
                    )

        if measurement is None:
            missed_frames += 1

            if missed_frames > 10:
                status = "CAMERA TRACKING LOST"
                break

            continue

        missed_frames = 0
        measured_range, bearing = measurement
        detections += 1

        ekf.update_landmark(
            measured_range, bearing, landmark
        )

        if measured_range <= target_range + 0.02:
            status = "DOCKED"
            break

        v = np.clip(
            0.5 * (measured_range - target_range),
            0.0, 0.08
        )
        omega = np.clip(
            1.5 * bearing, -0.5, 0.5
        )

        if abs(bearing) > np.deg2rad(15):
            v = 0.0

        if np.min(lidar) < 0.20:
            status = "LIDAR SAFETY STOP"
            break

    wl = (v - omega * WHEEL_BASE/2) / WHEEL_RADIUS
    wr = (v + omega * WHEEL_BASE/2) / WHEEL_RADIUS

    if not np.all(np.isfinite([wl, wr])):
        status = "INVALID CONTROL"
        break

    move_robot(model, data, wl, wr, DT)

    if step % 4 == 0:
        capture_frame(phase)

    wheels = data.qpos[7:9].copy()
    dl, dr = (wheels - previous_wheels) * WHEEL_RADIUS
    previous_wheels = wheels

    ekf.predict(dl, dr)

    trajectory.append(data.qpos[:2].copy())
    estimated.append(ekf.x[:2].copy())

    if not np.all(np.isfinite(ekf.x)):
        status = "INVALID EKF STATE"
        break

    if step % 1000 == 0:
        print(
            f"Step {step}: {phase}, "
            f"position={np.round(data.qpos[:2], 2)}"
        )

for _ in range(50):
    capture_frame(status)

video_writer.release()
video_renderer.close()
renderer.close()

print("Video saved:", video_path)

trajectory = np.array(trajectory)
estimated = np.array(estimated)

if len(trajectory):
    plt.figure(figsize=(9, 7))
    plt.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        label="Robot trajectory"
    )
    plt.plot(
        estimated[:, 0],
        estimated[:, 1],
        "--",
        label="EKF estimate"
    )
    plt.plot(
        waypoints[:, 0],
        waypoints[:, 1],
        "o--",
        label="Waypoints"
    )
    plt.scatter(
        *landmark, marker="*", s=180,
        label="Charging station"
    )
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title("41014 Integrated Autonomous Navigation and Docking")
    plt.tight_layout()
    plt.savefig(
        "results/integrated_trajectory.png",
        dpi=160
    )
    plt.close()

print("\nFINAL INTEGRATED PROJECT RESULTS")
print("Status:", status)
print("Navigation waypoints:", waypoint_index, "/", len(waypoints))
print("Camera detections:", detections)
print("LiDAR safety stops:", safety_stops)
print("Final position:", np.round(data.qpos[:2], 3))
print("Distance to station:", round(
    np.linalg.norm(landmark - data.qpos[:2]), 3
), "m")
print("Saved results/integrated_trajectory.png")
