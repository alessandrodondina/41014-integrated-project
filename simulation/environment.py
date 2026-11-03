import mujoco
import numpy as np
import matplotlib.pyplot as plt

MODEL_PATH = "models/turtlebot3.xml"

def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, width=960, height=540)

    renderer.update_scene(data, camera="overview")
    image = renderer.render()
    plt.imsave("results/turtlebot3_environment.png", image)

    print("Robot environment loaded successfully")
    print("Bodies:", model.nbody)
    print("Actuators:", model.nu)
    print("Initial robot position:", data.qpos[:3])

    # Basic motor test
    data.ctrl[0] = 0.12
    data.ctrl[1] = 0.12

    for _ in range(2000):
        mujoco.mj_step(model, data)

    print("Final robot position:", data.qpos[:3])
    print("Movement distance:",
          np.linalg.norm(data.qpos[:2] - np.array([-1.8, -1.8])))

    renderer.update_scene(data, camera="overview")
    image = renderer.render()
    plt.imsave("results/turtlebot3_after_movement.png", image)

    renderer.close()
    print("Images saved to results/")

if __name__ == "__main__":
    main()
