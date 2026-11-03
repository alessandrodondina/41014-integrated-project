import mujoco
import cv2
import numpy as np

model = mujoco.MjModel.from_xml_path(
    "models/turtlebot3.xml"
)
data = mujoco.MjData(model)

# Position robot near station, facing +X.
# This is a controlled sensor test only.
data.qpos[0] = 0.9
data.qpos[1] = 1.8
data.qpos[2] = 0.09

mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(
    model, width=640, height=480
)

options = mujoco.MjvOption()
options.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0

renderer.update_scene(
    data,
    camera="robot_camera",
    scene_option=options
)

image = renderer.render()
renderer.close()

cv2.imwrite(
    "results/aruco_camera.png",
    cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
)

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

detector = cv2.aruco.ArucoDetector(dictionary)

corners, ids, rejected = detector.detectMarkers(gray)

if ids is not None:
    print("SUCCESS: ArUco marker detected!")
    print("Marker IDs:", ids.flatten().tolist())

    annotated = image.copy()
    cv2.aruco.drawDetectedMarkers(
        annotated, corners, ids
    )

    cv2.imwrite(
        "results/aruco_detected.png",
        cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    )
else:
    print("Marker not detected yet.")
    print("Check the camera image and marker orientation.")

print("Saved results/aruco_camera.png")
