import cv2
import mujoco
import numpy as np

WIDTH = 640
HEIGHT = 480
MARKER_SIZE = 0.20

MODEL_PATH = "models/turtlebot3.xml"

def camera_matrix_from_fovy(fovy_deg, width, height):
    fy = (height / 2) / np.tan(np.deg2rad(fovy_deg) / 2)
    fx = fy

    return np.array([
        [fx, 0, width / 2],
        [0, fy, height / 2],
        [0, 0, 1]
    ], dtype=np.float64)

def estimate_marker_pose(image, camera_matrix, marker_size):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_4X4_50
    )

    detector = cv2.aruco.ArucoDetector(dictionary)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return None

    index = np.where(ids.flatten() == 0)[0]

    if len(index) == 0:
        return None

    points = corners[index[0]][0].astype(np.float64)

    half = marker_size / 2

    object_points = np.array([
        [-half, half, 0],
        [half, half, 0],
        [half, -half, 0],
        [-half, -half, 0]
    ], dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        points,
        camera_matrix,
        np.zeros(5),
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )

    if not success:
        return None

    x, y, z = tvec.flatten()

    distance = np.linalg.norm(tvec)
    bearing = np.rad2deg(np.arctan2(x, z))

    return {
        "distance": float(distance),
        "bearing_deg": float(bearing),
        "translation": tvec.flatten(),
        "rotation": rvec.flatten(),
        "corners": points
    }

def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # Controlled test position, facing the marker
    data.qpos[0] = 0.9
    data.qpos[1] = 1.8
    data.qpos[2] = 0.09

    mujoco.mj_forward(model, data)

    camera_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, "robot_camera"
    )

    fovy = model.cam_fovy[camera_id]
    K = camera_matrix_from_fovy(fovy, WIDTH, HEIGHT)

    renderer = mujoco.Renderer(
        model, width=WIDTH, height=HEIGHT
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

    result = estimate_marker_pose(image, K, MARKER_SIZE)

    print("\nCAMERA CALIBRATION")
    print("------------------")
    print("Resolution:", WIDTH, "x", HEIGHT)
    print("Vertical FOV:", round(fovy, 2), "degrees")
    print("Camera matrix:\n", K)

    if result is None:
        print("\nERROR: Marker not detected")
        return

    print("\nARUCO POSE ESTIMATION")
    print("---------------------")
    print("Marker ID: 0")
    print("Estimated distance:",
          round(result["distance"], 3), "m")
    print("Estimated bearing:",
          round(result["bearing_deg"], 3), "degrees")
    print("Camera-frame translation:",
          np.round(result["translation"], 3))

    # Evaluation only: ground truth is not used by solvePnP.
    camera_pos = data.cam_xpos[camera_id]
    marker_pos = np.array([1.9, 1.8, 0.35])

    true_distance = np.linalg.norm(marker_pos - camera_pos)

    print("\nVALIDATION")
    print("Ground-truth distance:",
          round(true_distance, 3), "m")
    print("Absolute distance error:",
          round(abs(result["distance"] - true_distance), 3), "m")

if __name__ == "__main__":
    main()
