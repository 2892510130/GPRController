import numpy as np
import scipy.special
from typing import Any, Tuple

from .quintic_polynomials import QuinticPolynomial
from .resample import resample
from utils import TrajectoryConfig
from utils.angle import convert_angle, normalize_angle

class BezierPath:
    def __init__(self, config: TrajectoryConfig, setup_speed_profile: bool = True, create_method: str = "start-end") -> None:
        self.config = config
        if create_method == "start-end":
            self.path, self.control_points_derivatives = BezierPath.calc_4points_bezier_path(
                config.start_x, config.start_y, config.start_yaw, config.end_x, config.end_y, config.end_yaw, config.offset, config.samples)
        elif create_method == "control_points":
            control_points = config.control_points
            self.control_points_derivatives = BezierPath.bezier_derivatives_control_points(control_points, 3)
            self.path = BezierPath.calc_bezier_path(control_points, config.samples)
        
        self.max_vel = config.max_vel
        self.vel = BezierPath.calc_bezier_path(self.control_points_derivatives[1], config.samples)
        dist = np.cumsum(np.hypot(self.vel[:, 0], self.vel[:, 1])) / (config.samples - 1)
        self.s = np.concatenate(([0.0], dist[:-1]))
        yaw = np.arctan2(self.vel[:, 1], self.vel[:, 0])
        self.path = np.concatenate((self.path, yaw.reshape(-1, 1)), axis=1)

        self.acc = BezierPath.calc_bezier_path(self.control_points_derivatives[2], config.samples)
        curvature = BezierPath.calc_curvature(self.vel[:, 0], self.vel[:, 1], self.acc[:, 0], self.acc[:, 1])
        self.path = np.concatenate((self.path, curvature.reshape(-1, 1)), axis=1)

        if setup_speed_profile:
            self.get_speed_profile()

    def get_speed_profile(self) -> None:
        start_v, start_a, end_v, end_a, total_time, sample_dt = (
            self.config.start_v, self.config.start_a, self.config.end_v, self.config.end_a, self.config.total_time, self.config.sample_dt)
        
        q = QuinticPolynomial(0, start_v, start_a, self.s[-1], end_v, end_a, total_time)
        time_list = np.arange(0, total_time + 1e-6, sample_dt)

        speed_profile = np.ndarray((len(time_list), 6))
        new_path = np.ndarray((len(time_list), 4))
        speed_profile[:, 0] = time_list # time
        speed_profile[:, 1] = q.calc_point(time_list) # s
        speed_profile[:, 2] = q.calc_first_derivative(time_list) # v
        speed_profile[:, 3] = q.calc_second_derivative(time_list) # a

        convert_angle(self.path[:, 2])
        new_path[:, 0] = resample(self.path[:, 0], self.s, speed_profile[:, 1])
        new_path[:, 1] = resample(self.path[:, 1], self.s, speed_profile[:, 1])
        new_path[:, 2] = normalize_angle(resample(self.path[:, 2], self.s, speed_profile[:, 1]))
        new_path[:, 3] = resample(self.path[:, 3], self.s, speed_profile[:, 1])

        speed_profile[:, 4] = speed_profile[:, 2] * new_path[:, 3] # w
        delta_w = (speed_profile[1:, 4] - speed_profile[:-1, 4]) / sample_dt
        speed_profile[:, 5] = np.concatenate((np.array([start_a * new_path[0, 3] + start_v * float(new_path[1, 3] - new_path[0, 3]) / sample_dt]), delta_w)) # a_w
        
        self.path = new_path
        self.speed_profile = speed_profile

    def calc_4points_bezier_path(
        sx: float,
        sy: float,
        syaw: float,
        ex: float,
        ey: float,
        eyaw: float,
        offset: float,
        n_points: int
    ) -> Tuple[np.ndarray | dict]:
        """
        Compute control points and path given start and end position.

        :param sx: (float) x-coordinate of the starting point
        :param sy: (float) y-coordinate of the starting point
        :param syaw: (float) yaw angle at start
        :param ex: (float) x-coordinate of the ending point
        :param ey: (float) y-coordinate of the ending point
        :param eyaw: (float) yaw angle at the end
        :param offset: (float)
        :return: (numpy array, numpy array)
        """
        dist = np.hypot(sx - ex, sy - ey) / offset
        control_points = np.array(
            [[sx, sy],
            [sx + dist * np.cos(syaw), sy + dist * np.sin(syaw)],
            [ex - dist * np.cos(eyaw), ey - dist * np.sin(eyaw)],
            [ex, ey]])
        control_points_derivatives = BezierPath.bezier_derivatives_control_points(control_points, 3)
        path = BezierPath.calc_bezier_path(control_points, n_points)

        return path, control_points_derivatives

    @staticmethod
    def calc_bezier_path(control_points: np.ndarray, n_points: int = 100) -> np.ndarray:
        """
        Compute bezier path (trajectory) given control points.

        :param control_points: (numpy array)
        :param n_points: (int) number of points in the trajectory
        :return: (numpy array)
        """
        traj = []
        for t in np.linspace(0, 1, n_points):
            traj.append(BezierPath.bezier(t, control_points))

        return np.array(traj)

    @staticmethod
    def bernstein_poly(n: int, i: int, t: float) -> (Any | np.ndarray | float):
        """
        Bernstein polynom.

        :param n: (int) polynom degree
        :param i: (int)
        :param t: (float)
        :return: (float)
        """
        return scipy.special.comb(n, i) * t ** i * (1 - t) ** (n - i)

    @staticmethod
    def bezier(t: float, control_points: np.ndarray) -> np.ndarray:
        """
        Return one point on the bezier curve.

        :param t: (float) number in [0, 1]
        :param control_points: (numpy array)
        :return: (numpy array) Coordinates of the point
        """
        n = len(control_points) - 1
        return np.sum([BezierPath.bernstein_poly(n, i, t) * control_points[i] for i in range(n + 1)], axis=0)

    @staticmethod
    def bezier_derivatives_control_points(control_points: np.ndarray, n_derivatives: int) -> dict[int, np.ndarray]:
        """
        Compute control points of the successive derivatives of a given bezier curve.

        A derivative of a bezier curve is a bezier curve.
        See https://pomax.github.io/bezierinfo/#derivatives
        for detailed explanations

        :param control_points: (numpy array)
        :param n_derivatives: (int)
        e.g., n_derivatives=2 -> compute control points for first and second derivatives
        :return: ([numpy array])
        """
        w = {0: control_points}
        for i in range(n_derivatives):
            n = len(w[i])
            w[i + 1] = np.array([(n - 1) * (w[i][j + 1] - w[i][j])
                                for j in range(n - 1)])
        return w

    @staticmethod
    def calc_curvature(dx: float, dy: float, ddx: float, ddy: float) -> float:
        """
        Compute curvature at one point given first and second derivatives.

        :param dx: (float) First derivative along x axis
        :param dy: (float)
        :param ddx: (float) Second derivative along x axis
        :param ddy: (float)
        :return: (float)
        """
        return (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** (3 / 2)
