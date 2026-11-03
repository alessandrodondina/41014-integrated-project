from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

path = Path("models/turtlebot3.xml")
tree = ET.parse(path)
root = tree.getroot()

worldbody = root.find("worldbody")
robot = worldbody.find(".//body[@name='turtlebot3']")

# Remove previously generated LiDAR sites, if any.
for site in list(robot.findall("site")):
    if site.get("name", "").startswith("lidar_ray_"):
        robot.remove(site)

sensor_element = root.find("sensor")
if sensor_element is None:
    sensor_element = ET.SubElement(root, "sensor")

for sensor in list(sensor_element):
    if sensor.get("name", "").startswith("lidar_"):
        sensor_element.remove(sensor)

num_rays = 36

for i in range(num_rays):
    angle = 2 * np.pi * i / num_rays

    # MuJoCo rangefinders measure along the site's local +Z axis.
    # Rotate the local Z axis into the horizontal XY plane.
    direction = np.array([
        np.cos(angle),
        np.sin(angle),
        0.0
    ])

    x_axis = np.array([0.0, 0.0, 1.0])
    y_axis = np.cross(direction, x_axis)

    rotation = np.column_stack((x_axis, y_axis, direction))

    from scipy.spatial.transform import Rotation
    quat = Rotation.from_matrix(rotation).as_quat()
    quat = [quat[3], quat[0], quat[1], quat[2]]

    site_name = f"lidar_ray_{i:02d}"

    ET.SubElement(
        robot,
        "site",
        name=site_name,
        pos="0 0 0.09",
        size="0.005",
        quat=" ".join(f"{v:.8f}" for v in quat),
        rgba="1 0 0 0.3"
    )

    ET.SubElement(
        sensor_element,
        "rangefinder",
        name=f"lidar_{i:02d}",
        site=site_name,
        cutoff="4.0"
    )

ET.indent(tree, space="  ")
tree.write(path, encoding="unicode")

print(f"Added {num_rays} LiDAR rangefinders to {path}")
