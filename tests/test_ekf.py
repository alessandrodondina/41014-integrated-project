import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1])
)

import numpy as np
import matplotlib.pyplot as plt

from estimation.ekf import RobotEKF, wrap_angle


def main():
    rng = np.random.default_rng(42)

    ekf = RobotEKF(
        initial_state=(0.0, 0.0, 0.0),
        wheel_base=0.16
    )

    true_state = np.array([0.0, 0.0, 0.0])
    odom_state = np.array([0.0, 0.0, 0.0])

    landmark = np.array([2.0, 1.0])

    true_history = []
    ekf_history = []
    odom_history = []

    for step in range(120):
        # True motion for synthetic test.
        dl_true = 0.012
        dr_true = 0.014

        ds_true = (dl_true + dr_true) / 2
        dtheta_true = (
            dr_true - dl_true
        ) / 0.16

        theta_mid = true_state[2] + dtheta_true / 2

        true_state[0] += ds_true * np.cos(theta_mid)
        true_state[1] += ds_true * np.sin(theta_mid)
        true_state[2] = wrap_angle(
            true_state[2] + dtheta_true
        )

        # Noisy wheel encoder readings.
        dl = dl_true + rng.normal(0, 0.0015)
        dr = dr_true + rng.normal(0, 0.0015)

        # Raw odometry for comparison.
        ds = (dl + dr) / 2
        dtheta = (dr - dl) / 0.16
        mid = odom_state[2] + dtheta / 2

        odom_state[0] += ds * np.cos(mid)
        odom_state[1] += ds * np.sin(mid)
        odom_state[2] = wrap_angle(
            odom_state[2] + dtheta
        )

        # EKF prediction uses the same noisy encoders.
        ekf.predict(dl, dr)

        # Simulated camera observation every 5 steps.
        if step % 5 == 0:
            dx = landmark[0] - true_state[0]
            dy = landmark[1] - true_state[1]

            true_range = np.hypot(dx, dy)
            true_bearing = wrap_angle(
                np.arctan2(dy, dx) - true_state[2]
            )

            measured_range = (
                true_range + rng.normal(0, 0.03)
            )

            measured_bearing = wrap_angle(
                true_bearing
                + rng.normal(0, np.deg2rad(2))
            )

            ekf.update_landmark(
                measured_range,
                measured_bearing,
                landmark
            )

        true_history.append(true_state[:2].copy())
        ekf_history.append(ekf.x[:2].copy())
        odom_history.append(odom_state[:2].copy())

    true_history = np.array(true_history)
    ekf_history = np.array(ekf_history)
    odom_history = np.array(odom_history)

    ekf_errors = np.linalg.norm(
        ekf_history - true_history, axis=1
    )

    odom_errors = np.linalg.norm(
        odom_history - true_history, axis=1
    )

    print("\nEKF VALIDATION")
    print("--------------------")
    print(
        "Final EKF error:",
        round(ekf_errors[-1], 4), "m"
    )
    print(
        "Final odometry error:",
        round(odom_errors[-1], 4), "m"
    )
    print(
        "EKF RMSE:",
        round(np.sqrt(np.mean(ekf_errors ** 2)), 4),
        "m"
    )
    print(
        "Odometry RMSE:",
        round(np.sqrt(np.mean(odom_errors ** 2)), 4),
        "m"
    )

    plt.figure(figsize=(8, 6))
    plt.plot(
        true_history[:, 0],
        true_history[:, 1],
        label="Ground truth",
        linewidth=2
    )
    plt.plot(
        odom_history[:, 0],
        odom_history[:, 1],
        "--",
        label="Wheel odometry"
    )
    plt.plot(
        ekf_history[:, 0],
        ekf_history[:, 1],
        label="EKF estimate"
    )
    plt.scatter(
        landmark[0],
        landmark[1],
        marker="*",
        s=180,
        label="Landmark"
    )

    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.title("EKF State Estimation Validation")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "results/ekf_trajectory.png",
        dpi=180
    )
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(
        ekf_errors,
        label="EKF position error"
    )
    plt.plot(
        odom_errors,
        label="Odometry position error"
    )

    plt.xlabel("Simulation step")
    plt.ylabel("Position error (m)")
    plt.title("EKF vs Odometry Error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "results/ekf_error.png",
        dpi=180
    )
    plt.close()

    print("\nSaved results:")
    print("results/ekf_trajectory.png")
    print("results/ekf_error.png")


if __name__ == "__main__":
    main()
