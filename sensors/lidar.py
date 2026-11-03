import mujoco
import numpy as np
import matplotlib.pyplot as plt

MODEL_PATH = "models/turtlebot3.xml"
NUM_RAYS = 36
MAX_RANGE = 4.0

def read_lidar(model, data):
    distances = []

    for i in range(NUM_RAYS):
        sensor_name = f"lidar_{i:02d}"
        sensor_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name
        )
        if sensor_id < 0:
            raise RuntimeError(f"Missing sensor: {sensor_name}")

        start = model.sensor_adr[sensor_id]
        value = float(data.sensordata[start])

        # MuJoCo returns -1 for no detected intersection.
        if value < 0:
            value = MAX_RANGE

        distances.append(min(value, MAX_RANGE))

    return np.array(distances)

def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    distances = read_lidar(model, data)
    angles = np.linspace(0, 2*np.pi, NUM_RAYS, endpoint=False)

    print("LiDAR scan completed")
    print("Number of rays:", len(distances))
    print("Minimum distance:", round(distances.min(), 3), "m")
    print("Maximum distance:", round(distances.max(), 3), "m")

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="polar")

    ax.plot(angles, distances, marker="o", linewidth=1.5)
    ax.set_ylim(0, MAX_RANGE)
    ax.set_title("TurtleBot3 — Simulated 2D LiDAR Scan")

    plt.savefig("results/lidar_scan.png", dpi=160)
    plt.close()

    print("Saved results/lidar_scan.png")

if __name__ == "__main__":
    main()
