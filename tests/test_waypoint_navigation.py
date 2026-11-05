import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
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
    [1.4, 1.8]
])

previous_wheels = data.qpos[7:9].copy()
trajectory = []
estimated = []

waypoint_index = 0
reached = False
safety_stops = 0

def wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))

for step in range(10000):
    pose = ekf.x
    goal = waypoints[waypoint_index]

    dx = goal[0] - pose[0]
    dy = goal[1] - pose[1]
    distance = np.hypot(dx, dy)

    if distance < 0.15:
        print(f"Waypoint {waypoint_index + 1} reached")
        waypoint_index += 1

        if waypoint_index == len(waypoints):
            reached = True
            break

        continue

    heading = np.arctan2(dy, dx)
    error = wrap_angle(heading - pose[2])

    v = min(0.18, 0.8 * distance)
    omega = np.clip(2.0 * error, -1.2, 1.2)

    if abs(error) > np.deg2rad(45):
        v = 0.0

    lidar = read_lidar(model, data)

    angles = wrap_angle(
        np.linspace(0, 2*np.pi, len(lidar), endpoint=False)
    )

    front = lidar[np.abs(angles) < np.deg2rad(30)]

    if len(front) and np.min(front) < 0.28:
        v = 0.0
        safety_stops += 1

    wl = (v - omega * WHEEL_BASE / 2) / WHEEL_RADIUS
    wr = (v + omega * WHEEL_BASE / 2) / WHEEL_RADIUS

    move_robot(model, data, wl, wr, DT)

    wheels = data.qpos[7:9].copy()
    dl, dr = (wheels - previous_wheels) * WHEEL_RADIUS
    previous_wheels = wheels

    ekf.predict(dl, dr)

    trajectory.append(data.qpos[:2].copy())
    estimated.append(ekf.x[:2].copy())

    if step % 1000 == 0:
        print(
            f"Step {step}: "
            f"waypoint={waypoint_index + 1}, "
            f"position={np.round(data.qpos[:2], 2)}"
        )

trajectory = np.array(trajectory)
estimated = np.array(estimated)

plt.figure(figsize=(8, 7))
plt.plot(trajectory[:, 0], trajectory[:, 1], label="Robot path")
plt.plot(estimated[:, 0], estimated[:, 1], "--", label="EKF estimate")
plt.plot(waypoints[:, 0], waypoints[:, 1], "o--", label="Waypoints")
plt.scatter(-1.8, -1.8, marker="s", label="Start")
plt.scatter(1.9, 1.8, marker="*", s=180, label="Charging station")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Waypoint Navigation with EKF and LiDAR Safety")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("results/waypoint_navigation.png", dpi=160)
plt.close()

print("\nWAYPOINT NAVIGATION RESULTS")
print("Reached final waypoint:", reached)
print("Waypoints completed:", waypoint_index, "/", len(waypoints))
print("LiDAR safety stops:", safety_stops)
print("Final position:", np.round(data.qpos[:2], 3))
print("Saved results/waypoint_navigation.png")
