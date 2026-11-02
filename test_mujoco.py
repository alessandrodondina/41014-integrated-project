import mujoco

xml = """
<mujoco model="41014 Test">
  <worldbody>
    <light pos="0 0 3"/>
    <geom name="floor" type="plane" size="3 3 0.1"/>
    <body name="test_robot" pos="0 0 0.2">
      <freejoint/>
      <geom type="box" size="0.15 0.1 0.1"
            rgba="0.2 0.5 0.9 1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

print("Simulation loaded successfully!")
print("Bodies:", model.nbody)
