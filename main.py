import mujoco
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

MODEL_PATH = "models/turtlebot3.xml"
WHEEL_RADIUS = 0.045
WHEEL_BASE = 0.210
DT = 0.02

def wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))

def set_pose(model, data, x, y, theta):
    data.qpos[0] = x
    data.qpos[1] = y
    data.qpos[2] = 0.10
    data.qpos[3:7] = [
        np.cos(theta / 2),
        0,
        0,
        np.sin(theta / 2)
    ]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

def move_robot(model, data, wl, wr, dt):
    x, y = data.qpos[:2]
    q = data.qpos[3:7]
    theta = 2 * np.arctan2(q[3], q[0])

    v = WHEEL_RADIUS * (wr + wl) / 2
    omega = WHEEL_RADIUS * (wr - wl) / WHEEL_BASE

    theta_mid = theta + omega * dt / 2

    x += v * np.cos(theta_mid) * dt
    y += v * np.sin(theta_mid) * dt
    theta = wrap_angle(theta + omega * dt)

    data.qpos[7] += wl * dt
    data.qpos[8] += wr * dt

    set_pose(model, data, x, y, theta)

def main():
    Path("results").mkdir(exist_ok=True)

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, width=960, height=540)

    renderer.update_scene(data, camera="overview")
    plt.imsave("results/turtlebot3_environment.png", renderer.render())

    start = data.qpos[:2].copy()

    for _ in range(100):
        move_robot(model, data, 4.0, 4.0, DT)

    end = data.qpos[:2].copy()

    renderer.update_scene(data, camera="overview")
    plt.imsave("results/turtlebot3_after_movement.png", renderer.render())

    print("Starting position:", start)
    print("Final position:", end)
    print("Distance travelled:", np.linalg.norm(end - start))
    print("Expected distance: 0.36 m")

    renderer.close()

if __name__ == "__main__":
    main()
