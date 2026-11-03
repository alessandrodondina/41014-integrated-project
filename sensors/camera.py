import mujoco
import matplotlib.pyplot as plt

model = mujoco.MjModel.from_xml_path("models/turtlebot3.xml")
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, width=640, height=480)

# Hide yellow LiDAR rangefinder rays
options = mujoco.MjvOption()
options.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0

renderer.update_scene(
    data,
    camera="robot_camera",
    scene_option=options
)

image = renderer.render()

plt.imsave("results/robot_camera_clean.png", image)
renderer.close()

print("SUCCESS: Clean camera image saved!")
