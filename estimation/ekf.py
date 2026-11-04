import numpy as np


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class RobotEKF:
    """
    EKF state: [x, y, theta]

    Prediction inputs:
        dl = left wheel travel (metres)
        dr = right wheel travel (metres)

    Camera measurement:
        range and bearing to a known landmark
    """

    def __init__(
        self,
        initial_state=(0.0, 0.0, 0.0),
        wheel_base=0.16,
    ):
        self.x = np.array(initial_state, dtype=float)

        self.P = np.diag([
            0.10 ** 2,
            0.10 ** 2,
            np.deg2rad(10) ** 2
        ])

        self.wheel_base = wheel_base

        # Initial tuning values; validate later.
        self.Q = np.diag([
            0.01 ** 2,
            0.01 ** 2,
            np.deg2rad(1.0) ** 2
        ])

        self.R = np.diag([
            0.03 ** 2,
            np.deg2rad(2.0) ** 2
        ])

    def predict(self, dl, dr):
        x, y, theta = self.x

        ds = (dr + dl) / 2.0
        dtheta = (dr - dl) / self.wheel_base

        theta_mid = theta + dtheta / 2.0

        dx = ds * np.cos(theta_mid)
        dy = ds * np.sin(theta_mid)

        self.x = np.array([
            x + dx,
            y + dy,
            wrap_angle(theta + dtheta)
        ])

        # Jacobian of motion model with respect to state.
        F = np.eye(3)

        F[0, 2] = -ds * np.sin(theta_mid)
        F[1, 2] = ds * np.cos(theta_mid)

        self.P = F @ self.P @ F.T + self.Q
        self.P = (self.P + self.P.T) / 2

        return self.x.copy()

    def update_landmark(
        self,
        measured_range,
        measured_bearing,
        landmark_xy
    ):
        landmark_xy = np.asarray(
            landmark_xy, dtype=float
        )

        dx = landmark_xy[0] - self.x[0]
        dy = landmark_xy[1] - self.x[1]

        q = dx ** 2 + dy ** 2

        if q < 1e-10:
            return self.x.copy()

        predicted_range = np.sqrt(q)

        predicted_bearing = wrap_angle(
            np.arctan2(dy, dx) - self.x[2]
        )

        z = np.array([
            measured_range,
            measured_bearing
        ])

        h = np.array([
            predicted_range,
            predicted_bearing
        ])

        innovation = z - h
        innovation[1] = wrap_angle(innovation[1])

        # Measurement Jacobian.
        H = np.array([
            [-dx / predicted_range,
             -dy / predicted_range,
             0.0],

            [dy / q,
             -dx / q,
             -1.0]
        ])

        S = H @ self.P @ H.T + self.R

        K = np.linalg.solve(
            S, H @ self.P
        ).T

        self.x += K @ innovation
        self.x[2] = wrap_angle(self.x[2])

        # Joseph covariance update for stability.
        I = np.eye(3)
        A = I - K @ H

        self.P = (
            A @ self.P @ A.T
            + K @ self.R @ K.T
        )

        self.P = (self.P + self.P.T) / 2

        return self.x.copy()
