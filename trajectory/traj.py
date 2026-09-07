import numpy as np
import yaml

class Trajectory:
    def __init__(self, data_path: str) -> None:
        with open(data_path) as file:
            self.traj = yaml.load(file, Loader=yaml.FullLoader)
        
        self.length = len(self.traj["x"])
        self.path = np.zeros((self.length, 4))
        self.speed_profile = np.zeros((self.length, 6))

        self.path[:, 0] = np.array(self.traj["x"])
        self.path[:, 1] = np.array(self.traj["y"])
        self.path[:, 2] = np.array(self.traj["yaw"])
        self.path[:, 3] = np.array(self.traj["curvature"])

        self.speed_profile[:, 0] = np.array(self.traj["time"])
        self.speed_profile[:, 1] = np.array(self.traj["arcLength"])
        self.speed_profile[:, 2] = np.array(self.traj["vel"])
        self.speed_profile[:, 3] = np.array(self.traj["acc"])
        self.speed_profile[:, 4] = np.array(self.traj["omega"])
        self.speed_profile[:, 5] = np.array(self.traj["acc_omega"])