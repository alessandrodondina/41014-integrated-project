import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
import matplotlib.pyplot as plt

from estimation.ekf import RobotEKF
from main import move_robot, WHEEL_RADIUS, WHEEL_BASE, DT

Path("results").mkdir(exist_ok=True)

model = mujoco.MjModel.from_xml_path("models/turtlebot3.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

initial_pose = np.array([-1.8, -1.8, 0.0])

ekf = RobotEKF(
    initial_state=initial_pose,
    wheel_base=WHEEL_BASE
)

true_history = []
estimated_history = []
odometry_history = []

odometry = initial_pose.copy()
previous_wheels = data.qpos[7:9].copy()

rng = np.random.default_rng(42)

for step in range(500):

    if step < 150:
        wl, wr = 4.0, 4.0
    elif step < 250:
        wl, wr = 2.0, 4.0
    elif step < 400:
        wl, wr = 4.0, 4.0
    else:
        wl, wr = 4.0, 2.0

    move_robot(model, data, wl, wr, DT)

    wheels = data.qpos[7:9].copy()
    dl, dr = (wheels - previous_wheels) * WHEEL_RADIUS
    previous_wheels = wheels

    noisy_dl = dl + rng.normal(0, 0.0005)
    noisy_dr = dr + rng.normal(0, 0.0005)

    ekf.predict(noisy_dl, noisy_dr)

    ds = (noisy_dl + noisy_dr) / 2
    dtheta = (noisy_dr - noisy_dl) / WHEEL_BASE
    theta_mid = odometry[2] + dtheta / 2

    odometry[0] += ds * np.cos(theta_mid)
    odometry[1] += ds * np.sin(theta_mid)
    odometry[2] += dtheta

    true_pose = np.array([
        data.qpos[0],
        data.qpos[1],
        2 * np.arctan2(data.qpos[6], data.qpos[3])
    ])

    true_history.append(true_pose.copy())
    estimated_history.append(ekf.x.copy())
    odometry_history.append(odometry.copy())

true_history = np.array(true_history)
estimated_history = np.array(estimated_history)
odometry_history = np.array(odometry_history)

ekf_error = np.linalg.norm(
    true_history[:, :2] - estimated_history[:, :2], axis=1
)

odom_error = np.linalg.norm(
    true_history[:, :2] - odometry_history[:, :2], axis=1
)

plt.figure(figsize=(8, 6))
plt.plot(true_history[:, 0], true_history[:, 1], label="Ground truth")
plt.plot(estimated_history[:, 0], estimated_history[:, 1], "--", label="EKF estimate")
plt.plot(odometry_history[:, 0], odometry_history[:, 1], ":", label="Wheel odometry")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("EKF Localization in MuJoCo")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("results/ekf_mujoco_trajectory.png", dpi=160)
plt.close()

plt.figure(figsize=(8, 4))
plt.plot(ekf_error, label="EKF position error")
plt.plot(odom_error, label="Odometry position error")
plt.xlabel("Simulation step")
plt.ylabel("Position error (m)")
plt.title("Localization Error")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("results/ekf_mujoco_error.png", dpi=160)
plt.close()

print("EKF integrated with MuJoCo wheel encoders")
print(f"Final EKF error: {ekf_error[-1]:.4f} m")
print(f"Final odometry error: {odom_error[-1]:.4f} m")
print("Saved plots to results/")
