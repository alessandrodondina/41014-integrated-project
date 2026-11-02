import mujoco
import matplotlib.pyplot as plt

model = mujoco.MjModel.from_xml_path("models/test_scene.xml")
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, width=640, height=480)
renderer.update_scene(data)

image = renderer.render()

plt.imsave("results/test_render.png", image)
renderer.close()

print("SUCCESS: Saved results/test_render.png")
