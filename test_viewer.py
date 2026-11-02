import mujoco
import mujoco.viewer

print("Loading simulation...", flush=True)

model = mujoco.MjModel.from_xml_path("models/test_scene.xml")
data = mujoco.MjData(model)

print("Model loaded successfully.", flush=True)
print("Opening MuJoCo viewer...", flush=True)

mujoco.viewer.launch(model, data)
