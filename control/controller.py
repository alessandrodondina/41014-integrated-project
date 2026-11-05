import numpy as np

WHEEL_RADIUS = 0.045
WHEEL_BASE = 0.210

def wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))

class NavigationController:
    def __init__(self):
        self.goal_tolerance = 0.15
        self.safe_distance = 0.40
        self.clear_distance = 0.65
        self.max_speed = 0.18
        self.max_turn = 1.2
        self.avoiding = False
        self.turn_direction = 1
        self.avoid_steps = 0

    def compute(self, pose, goal, lidar):
        x, y, theta = pose

        dx = goal[0] - x
        dy = goal[1] - y

        distance = np.hypot(dx, dy)
        heading = np.arctan2(dy, dx)
        error = wrap_angle(heading - theta)

        if distance < self.goal_tolerance:
            return 0.0, 0.0, True

        angles = wrap_angle(
            np.linspace(0, 2*np.pi, len(lidar), endpoint=False)
        )

        front = lidar[np.abs(angles) < np.deg2rad(35)]
        left = lidar[(angles > 0) & (angles < np.pi/2)]
        right = lidar[(angles < 0) & (angles > -np.pi/2)]

        front_min = np.min(front) if len(front) else 4.0
        left_clear = np.mean(left) if len(left) else 4.0
        right_clear = np.mean(right) if len(right) else 4.0

        if not self.avoiding and front_min < self.safe_distance:
            self.avoiding = True
            self.turn_direction = (
                1 if left_clear > right_clear else -1
            )
            self.avoid_steps = 0

        if self.avoiding:
            self.avoid_steps += 1

            if front_min < self.clear_distance:
                v = 0.0
                omega = self.turn_direction * self.max_turn
            else:
                v = 0.12
                omega = self.turn_direction * 0.35

            if (
                self.avoid_steps > 80
                and front_min > self.clear_distance
                and abs(error) < np.deg2rad(35)
            ):
                self.avoiding = False

        else:
            v = min(self.max_speed, 0.8 * distance)
            omega = np.clip(
                2.0 * error,
                -self.max_turn,
                self.max_turn
            )

            if abs(error) > np.deg2rad(45):
                v = 0.0

        wl = (v - omega * WHEEL_BASE / 2) / WHEEL_RADIUS
        wr = (v + omega * WHEEL_BASE / 2) / WHEEL_RADIUS

        return wl, wr, False
