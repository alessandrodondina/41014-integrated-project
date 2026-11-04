import csv
from pathlib import Path

import mujoco
import numpy as np
import matplotlib.pyplot as plt

from camera_calibration import (
    camera_matrix_from_fovy,
    estimate_marker_pose,
)

MODEL_PATH = "models/turtlebot3.xml"
WIDTH, HEIGHT = 640, 480
MARKER_SIZE = 0.20

# Robot positions along the marker's viewing direction.
# The marker is located at x=1.9, y=1.8.
TEST_X = [1.3, 1.1, 0.9, 0.7, 0.5]

def main():
    Path("results").mkdir(exist_ok=True)

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    camera_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, "robot_camera"
    )

    K = camera_matrix_from_fovy(
        model.cam_fovy[camera_id], WIDTH, HEIGHT
    )

    renderer = mujoco.Renderer(
        model, width=WIDTH, height=HEIGHT
    )

    options = mujoco.MjvOption()
    options.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0

    results = []

    for x in TEST_X:
        mujoco.mj_resetData(model, data)

        data.qpos[0] = x
        data.qpos[1] = 1.8
        data.qpos[2] = 0.09

        mujoco.mj_forward(model, data)

        renderer.update_scene(
            data,
            camera="robot_camera",
            scene_option=options
        )

        image = renderer.render()
        estimate = estimate_marker_pose(
            image, K, MARKER_SIZE
        )

        camera_pos = data.cam_xpos[camera_id].copy()
        marker_pos = np.array([1.9, 1.8, 0.35])
        truth = np.linalg.norm(marker_pos - camera_pos)

        if estimate is None:
            print(f"x={x:.2f}: Marker NOT detected")
            results.append([x, truth, np.nan, np.nan])
            continue

        measured = estimate["distance"]
        error = abs(measured - truth)

        print(
            f"x={x:.2f} | "
            f"True={truth:.3f} m | "
            f"Estimated={measured:.3f} m | "
            f"Error={error:.3f} m"
        )

        results.append([x, truth, measured, error])

    renderer.close()

    with open(
        "results/camera_validation.csv",
        "w",
        newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "robot_x",
            "true_distance_m",
            "estimated_distance_m",
            "absolute_error_m"
        ])
        writer.writerows(results)

    results = np.array(results, dtype=float)
    valid = np.isfinite(results[:, 2])

    if np.any(valid):
        plt.figure(figsize=(8, 5))
        plt.plot(
            results[valid, 1],
            results[valid, 2],
            "o-",
            label="Estimated"
        )

        limits = [
            np.min(results[valid, 1]),
            np.max(results[valid, 1])
        ]

        plt.plot(
            limits, limits, "--",
            label="Ideal"
        )

        plt.xlabel("Ground-truth distance (m)")
        plt.ylabel("Estimated distance (m)")
        plt.title("ArUco Camera Distance Validation")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(
            "results/camera_validation.png",
            dpi=180
        )
        plt.close()

        errors = results[valid, 3]
        print("\nVALIDATION SUMMARY")
        print("Successful detections:", np.sum(valid), "/", len(TEST_X))
        print("Mean absolute error:", round(np.mean(errors), 4), "m")
        print("Maximum error:", round(np.max(errors), 4), "m")
    else:
        print("No successful detections; inspect marker visibility.")

if __name__ == "__main__":
    main()
