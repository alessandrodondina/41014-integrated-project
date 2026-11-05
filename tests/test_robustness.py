import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import csv
import mujoco
import cv2
import numpy as np
import matplotlib.pyplot as plt

from main import move_robot, WHEEL_RADIUS, WHEEL_BASE, DT
from estimation.ekf import RobotEKF
from sensors.lidar import read_lidar

Path("results").mkdir(exist_ok=True)

WAYPOINTS = np.array([
    [-0.8, -1.6],
    [0.1, -1.4],
    [1.5, -1.4],
    [1.7, -0.4],
    [1.7, 0.8],
    [0.9, 1.8]
])

LANDMARK = np.array([1.9, 1.8])
NOISE_LEVELS = [0.0, 0.0005, 0.002]

def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))

def run_trial(noise_std, seed):
    rng = np.random.default_rng(seed)

    model = mujoco.MjModel.from_xml_path(
        "models/turtlebot3.xml"
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    ekf = RobotEKF(
        initial_state=(-1.8, -1.8, 0.0),
        wheel_base=WHEEL_BASE
    )

    renderer = mujoco.Renderer(
        model, width=640, height=480
    )

    options = mujoco.MjvOption()
    options.flags[
        mujoco.mjtVisFlag.mjVIS_RANGEFINDER
    ] = 0

    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_4X4_50
    )
    detector = cv2.aruco.ArucoDetector(dictionary)

    f = (480 / 2) / np.tan(np.deg2rad(70) / 2)

    previous_wheels = data.qpos[7:9].copy()
    waypoint_index = 0
    phase = "NAVIGATION"
    status = "TIMEOUT"
    missed_frames = 0
    detections = 0
    errors = []
    uncertainties = []
    trajectory = []

    try:
        for step in range(12000):
            pose = ekf.x
            lidar = read_lidar(model, data)

            if phase == "NAVIGATION":
                goal = WAYPOINTS[waypoint_index]

                dx = goal[0] - pose[0]
                dy = goal[1] - pose[1]
                distance = np.hypot(dx, dy)

                if distance < 0.10:
                    waypoint_index += 1

                    if waypoint_index == len(WAYPOINTS):
                        phase = "ALIGNMENT"

                    continue

                heading = np.arctan2(dy, dx)
                error = wrap(heading - pose[2])

                v = min(0.18, 0.8 * distance)
                omega = np.clip(
                    2.0 * error, -1.2, 1.2
                )

                if abs(error) > np.deg2rad(45):
                    v = 0.0

                angles = wrap(
                    np.linspace(
                        0, 2*np.pi,
                        len(lidar),
                        endpoint=False
                    )
                )

                front = lidar[
                    np.abs(angles) < np.deg2rad(30)
                ]

                if len(front) and np.min(front) < 0.28:
                    v = 0.0

            elif phase == "ALIGNMENT":
                heading = np.arctan2(
                    LANDMARK[1] - pose[1],
                    LANDMARK[0] - pose[0]
                )

                error = wrap(heading - pose[2])
                v = 0.0
                omega = np.clip(
                    2.0 * error, -0.6, 0.6
                )

                if abs(error) < np.deg2rad(3):
                    phase = "DOCKING"
                    continue

            elif phase == "DOCKING":
                renderer.update_scene(
                    data,
                    camera="robot_camera",
                    scene_option=options
                )

                image = renderer.render()
                gray = cv2.cvtColor(
                    image, cv2.COLOR_RGB2GRAY
                )

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
                        camera_distance = (
                            f * 0.20 / pixel_width
                        )

                        bearing = -np.arctan2(
                            centre_x - 320, f
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
                        status = "CAMERA LOST"
                        break

                    continue

                missed_frames = 0
                measured_range, bearing = measurement
                detections += 1

                ekf.update_landmark(
                    measured_range,
                    bearing,
                    LANDMARK
                )

                if measured_range <= 0.57:
                    status = "DOCKED"
                    break

                v = np.clip(
                    0.5 * (measured_range - 0.55),
                    0, 0.08
                )

                omega = np.clip(
                    1.5 * bearing, -0.5, 0.5
                )

                if abs(bearing) > np.deg2rad(15):
                    v = 0.0

                if np.min(lidar) < 0.20:
                    status = "SAFETY STOP"
                    break

            wl = (
                v - omega * WHEEL_BASE / 2
            ) / WHEEL_RADIUS

            wr = (
                v + omega * WHEEL_BASE / 2
            ) / WHEEL_RADIUS

            if not np.all(np.isfinite([wl, wr])):
                status = "INVALID CONTROL"
                break

            move_robot(model, data, wl, wr, DT)

            wheels = data.qpos[7:9].copy()
            dl, dr = (
                wheels - previous_wheels
            ) * WHEEL_RADIUS
            previous_wheels = wheels

            # Inject reproducible encoder measurement noise.
            dl += rng.normal(0, noise_std)
            dr += rng.normal(0, noise_std)

            ekf.predict(dl, dr)

            if not np.all(np.isfinite(ekf.x)):
                status = "INVALID EKF"
                break

            true_xy = data.qpos[:2].copy()

            errors.append(
                np.linalg.norm(ekf.x[:2] - true_xy)
            )
            uncertainties.append(
                np.sqrt(
                    np.trace(ekf.P[:2, :2])
                )
            )
            trajectory.append(true_xy)

    finally:
        renderer.close()

    final_error = np.linalg.norm(
        data.qpos[:2] - LANDMARK
    )

    return {
        "noise_std_m": noise_std,
        "status": status,
        "waypoints": waypoint_index,
        "detections": detections,
        "dock_distance_m": final_error,
        "localisation_rmse_m": (
            np.sqrt(np.mean(np.square(errors)))
            if errors else float("nan")
        ),
        "final_uncertainty_m": (
            uncertainties[-1]
            if uncertainties else float("nan")
        ),
        "trajectory": np.array(trajectory)
    }

results = []

for i, noise in enumerate(NOISE_LEVELS):
    print(f"\nRunning trial {i+1}/3: noise={noise} m")

    result = run_trial(noise, seed=42+i)
    results.append(result)

    print(
        f"Status={result['status']}, "
        f"waypoints={result['waypoints']}/6, "
        f"distance={result['dock_distance_m']:.3f}m, "
        f"RMSE={result['localisation_rmse_m']:.3f}m"
    )

with open(
    "results/robustness_results.csv",
    "w",
    newline=""
) as file:
    writer = csv.writer(file)

    writer.writerow([
        "encoder_noise_std_m",
        "status",
        "waypoints_completed",
        "marker_detections",
        "dock_distance_m",
        "localisation_rmse_m",
        "final_position_uncertainty_m"
    ])

    for r in results:
        writer.writerow([
            r["noise_std_m"],
            r["status"],
            r["waypoints"],
            r["detections"],
            r["dock_distance_m"],
            r["localisation_rmse_m"],
            r["final_uncertainty_m"]
        ])

plt.figure(figsize=(9, 7))

for r in results:
    t = r["trajectory"]

    if len(t):
        plt.plot(
            t[:, 0],
            t[:, 1],
            label=f"Noise std={r['noise_std_m']}m"
        )

plt.scatter(
    *LANDMARK,
    marker="*",
    s=180,
    label="Charging station"
)

plt.axis("equal")
plt.grid(True)
plt.legend()
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Robustness Under Encoder Measurement Noise")
plt.tight_layout()
plt.savefig(
    "results/robustness_comparison.png",
    dpi=160
)
plt.close()

print("\nROBUSTNESS TEST COMPLETE")
print("Saved results/robustness_results.csv")
print("Saved results/robustness_comparison.png")
