import yaml
import numpy as np

from utils import TrajectoryConfig
from .resample import resample
from .quintic_polynomials import QuinticPolynomial
from utils.angle import convert_angle, normalize_angle

class RacePath:
    def __init__(self, config: TrajectoryConfig, track_dir: str) -> None:
        self.config = config

        with open(track_dir) as file:
            self.track = yaml.load(file, Loader=yaml.FullLoader)["track"]
            split = 2
            length = len(self.track["arcLength"]) // split
            self.s = self.track["arcLength"][:length]
            self.x = self.track["xCoords"][:length]
            self.y = self.track["yCoords"][:length]
            self.curvature = self.track["curvature"][:length]
            self.yaw = self.track["tangentAngle"][:length]

        self.resample()

    def resample(self) -> None:
        start_v, start_a, end_v, end_a, total_time, sample_dt = (
            self.config.start_v, self.config.start_a, self.config.end_v, self.config.end_a, self.config.total_time, self.config.sample_dt)
        
        q = QuinticPolynomial(0, start_v, start_a, self.s[-1], end_v, end_a, total_time)
        time_list = np.arange(0, total_time + sample_dt - 1e-6, sample_dt)

        speed_profile = np.ndarray((len(time_list), 6))
        new_path = np.ndarray((len(time_list), 4))
        speed_profile[:, 0] = time_list # time
        speed_profile[:, 1] = q.calc_point(time_list) # s
        speed_profile[:, 2] = q.calc_first_derivative(time_list) # v
        speed_profile[:, 3] = q.calc_second_derivative(time_list) # a

        convert_angle(self.yaw)
        new_path[:, 0] = resample(self.x, self.s, speed_profile[:, 1])
        new_path[:, 1] = resample(self.y, self.s, speed_profile[:, 1])
        new_path[:, 2] = normalize_angle(resample(self.yaw, self.s, speed_profile[:, 1]))
        new_path[:, 3] = resample(self.curvature, self.s, speed_profile[:, 1])

        speed_profile[:, 4] = speed_profile[:, 2] * new_path[:, 3] # w
        delta_w = (speed_profile[1:, 4] - speed_profile[:-1, 4]) / sample_dt
        speed_profile[:, 5] = np.concatenate((np.array([start_a * new_path[0, 3] + start_v * float(new_path[1, 3] - new_path[0, 3]) / sample_dt]), delta_w)) # a_w
        
        self.path = new_path
        self.speed_profile = speed_profile