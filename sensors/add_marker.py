from pathlib import Path
import xml.etree.ElementTree as ET

path = Path("models/turtlebot3.xml")
tree = ET.parse(path)
root = tree.getroot()

asset = root.find("asset")
worldbody = root.find("worldbody")

# Remove previous marker definitions if rerunning
for element in list(asset):
    if element.get("name") in ("aruco_texture", "aruco_material"):
        asset.remove(element)

for body in list(worldbody.findall("body")):
    if body.get("name") == "aruco_station":
        worldbody.remove(body)

ET.SubElement(
    asset, "texture",
    name="aruco_texture",
    type="2d",
    file="../assets/aruco_marker_0.png"
)

ET.SubElement(
    asset, "material",
    name="aruco_material",
    texture="aruco_texture",
    texuniform="false",
    specular="0",
    shininess="0"
)

station = ET.SubElement(
    worldbody, "body",
    name="aruco_station",
    pos="1.9 1.8 0.35"
)

# Vertical sign facing toward negative X
ET.SubElement(
    station, "geom",
    name="aruco_sign",
    type="box",
    size="0.005 0.12 0.12",
    material="aruco_material",
    contype="0",
    conaffinity="0"
)

ET.indent(tree, space="  ")
tree.write(path, encoding="unicode")

print("SUCCESS: ArUco station added")
