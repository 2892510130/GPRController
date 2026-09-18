import numpy as np
import casadi as ca
from typing import Any

from utils import SpeedOptimizerConfig, TrajectoryConfig
from .resample import resample

class NonlinearTrajectoryOptimizer:
    def __init__(self, config: SpeedOptimizerConfig, traj: np.ndarray | None = None, traj_config: TrajectoryConfig = None) -> None:
        """
        Optimize the trajectory to better fit vehicle dynamics
            traj: (n, 4) shape, each row is (arclength, vel, acc, curvature)
        """
        self.config = config
        self.traj = traj
        self.traj_config = traj_config
        self.dt = self.config.dt
        self.initialized = False

    def initialize(self) -> None:
        if self.initialized is False:
            self.length = self.traj.shape[0]
            self.x = ca.MX.sym("x", 3, self.length) # each one is [s, v, a]
            self.curv_func = ca.interpolant('k_s', 'bspline', [self.traj[:, 0]], self.traj[:, 3])

    def setup_constraints(self) -> None:
        # self.lbg = np.zeros((self.length * 4 - 3,))
        # self.ubg = np.zeros((self.length * 4 - 3,))
        self.lbg = np.zeros((self.length * 3 - 3,))
        self.ubg = np.zeros((self.length * 3 - 3,))
        s_e, v_e, a_e = self.traj[-1, 0], self.traj_config.end_v, self.traj_config.end_a
        lower_bound = np.array([0.0, 0.0, -1.0 * self.config.a_m]) # -1.0 * self.config.v_m
        upper_bound = np.array([s_e, self.config.v_m, self.config.a_m])
        self.lbx = np.tile(lower_bound, self.length)
        self.ubx = np.tile(upper_bound, self.length)

        # The dynamic between s, v, a and jerk
        self.g = []
        for i in range(self.length - 1):
            self.g.append(self.x[:, i+1] - self.dynamic(self.x[:, i], self.x[:, i+1]))
            self.lbg[i * 3 + 2] = -1.0 * self.config.j_m * self.dt
            self.ubg[i * 3 + 2] = self.config.j_m * self.dt
        
        # bound on the w = v * k
        # for i in range(self.length):
        #     self.g.append(self.x[1, i] * self.curv_func(self.x[0, i]))
        #     self.lbg[3 * self.length - 3 + i] = -1.0 * self.config.w_m
        #     self.ubg[3 * self.length - 3 + i] = self.config.w_m
        
        # Initial point constraint
        self.lbx[0] = self.ubx[0] = self.traj[0, 0]
        self.lbx[1] = self.ubx[1] = self.traj[0, 1]
        self.lbx[2] = self.ubx[2] = self.traj[0, 2]

        # End point constraint
        self.lbx[self.length * 3 - 3] = self.ubx[self.length * 3 - 3] = s_e
        self.lbx[self.length * 3 - 2] = self.ubx[self.length * 3 - 2] = v_e
        self.lbx[self.length * 3 - 1] = self.ubx[self.length * 3 - 1] = a_e

        self.g = ca.vertcat(*self.g)
    
    def setup_cost(self) -> None:
        self.cost = 0
        dt_square = self.dt * self.dt
        s_e, v_e, a_e = self.traj[-1, 0], 0, 0
        w_s, w_v, w_a, w_j, w_k = self.config.w_s, self.config.w_v, self.config.w_a, self.config.w_j, self.config.w_cur
        w_s_e, w_v_e, w_a_e = self.config.w_s_e, self.config.w_v_e, self.config.w_a_e
        for i in range(self.length):
            self.cost += w_s * (self.x[0, i] - self.traj[i, 0]) ** 2
            self.cost += w_v * (self.x[1, i] - self.traj[i, 1]) ** 2
            self.cost += w_a * (self.x[2, i] - self.traj[i, 2]) ** 2
            self.cost += w_k * ca.fabs(self.curv_func(self.x[0, i])) * self.x[1, i] ** 2
            # self.cost += w_k * np.abs(self.traj[i, 3]) * self.x[1, i] ** 2
            if i != 0:
                self.cost += w_j * (self.x[2, i] - self.x[2, i-1]) ** 2 / dt_square
        
        # We may not need this as it is a hard constraint above
        # self.cost += w_s_e * (self.x[0, self.length-1] - s_e) ** 2
        # self.cost += w_v_e * (self.x[1, self.length-1] - v_e) ** 2
        # self.cost += w_a_e * (self.x[2, self.length-1] - a_e) ** 2
    
    def solve(self) -> np.ndarray:
        self.initialize()
        self.setup_constraints()
        self.setup_cost()

        nlp = {'x': ca.vec(self.x), 'f': self.cost, 'g': self.g}
        solver_opts = {
            'verbose': None,
            'print_time': 1,
            'ipopt' : {
                'print_level': 0,  # Optional: suppress output
            }
        }

        solver = ca.nlpsol('solver', 'ipopt', nlp, solver_opts)
        sol = solver(lbx = self.lbx, ubx = self.ubx, lbg = self.lbg, ubg = self.ubg)
        return np.array(sol['x']).reshape(-1, 3)

    def dynamic(self, x: ca.MX, x_next: ca.MX) -> (ca.DM | Any):
        dt_square = self.dt * self.dt
        s_next = x[0] + x[1] * self.dt + x[2] * dt_square / 3 + x_next[2] * dt_square / 6
        v_next = x[1] + x[2] * self.dt / 2 + x_next[2] * self.dt / 2
        a_next = x[2]
        return ca.vertcat(s_next, v_next, a_next)

    def set_traj_with_bezier(self, path: np.ndarray, speed_profile: np.ndarray) -> None:
        self.traj = np.zeros((path.shape[0], 4))
        self.traj[:, 0] = speed_profile[:, 1]
        self.traj[:, 1] = speed_profile[:, 2]
        self.traj[:, 2] = speed_profile[:, 3]
        self.traj[:, 3] = path[:, 3]

def test_nonliner_speed_optimize():
    import yaml
    from .bezier import BezierPath
    import matplotlib.pyplot as plt
    from utils.angle import convert_angle, normalize_angle

    config = TrajectoryConfig.from_yaml("config/track/bezier_path.yaml")
    rc_path = BezierPath(config)

    # config = TrajectoryConfig.from_yaml("config/track/full_track_config.yaml")
    # rc_path = Trajectory("config/track/full_track.yaml")
    sp = SpeedOptimizerConfig.from_yaml("config/speed_optimizer/bezier.yaml")

    optimizer = NonlinearTrajectoryOptimizer(sp, traj_config=config)
    optimizer.set_traj_with_bezier(rc_path.path, rc_path.speed_profile)
    sol = optimizer.solve()
    dt = sp.dt

    k_s = np.array(optimizer.curv_func(sol[:, 0]))
    sampled_k = resample(rc_path.path[:, 3], rc_path.speed_profile[:, 1], sol[:, 0])
    w_opt = sol[:, 1] * k_s.squeeze()
    dw = (w_opt[1:] - w_opt[:-1]) / sp.dt
    print(dw.shape)
    dk = (k_s[1:] - k_s[:-1]) / config.sample_dt
    dw_opt = np.concatenate((np.array([config.start_a * float(k_s[0]) + config.start_v * float(dk[0])]), dw))
    dw_2 = sol[:, 2] * k_s.squeeze() + sol[:, 1] * np.concatenate((dk.squeeze(), np.array([float(dk[-1])])))

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(rc_path.speed_profile[:, 0], rc_path.speed_profile[:, 2], label="v")
    ax[0, 0].plot(rc_path.speed_profile[:, 0], sol[:, 1], label="v_opt")
    ax[0, 0].plot(rc_path.speed_profile[:, 0], rc_path.speed_profile[:, 4], label="w")
    ax[0, 0].plot(rc_path.speed_profile[:, 0], w_opt, label="w_opt")
    ax[0, 0].plot(rc_path.speed_profile[:, 0], k_s / np.max(k_s), label="k")
    ax[0, 0].legend()
    ax[0, 0].set_title("Speed Profile")

    ax[1, 0].plot(rc_path.speed_profile[:, 0], rc_path.speed_profile[:, 3], label="a")
    ax[1, 0].plot(rc_path.speed_profile[:, 0], rc_path.speed_profile[:, 5], label="a_w")
    ax[1, 0].plot(rc_path.speed_profile[:, 0], sol[:, 2], label="a_opt")
    ax[1, 0].plot(rc_path.speed_profile[:, 0], dw_opt, label="a_w_opt")
    ax[1, 0].plot(rc_path.speed_profile[:, 0], dw_2, label="a_w_opt_2")
    ax[1, 0].legend()
    ax[1, 0].set_title("Acc Profile")

    ax[0, 1].plot(rc_path.speed_profile[:, 0], rc_path.speed_profile[:, 1], label="s")
    ax[0, 1].plot(rc_path.speed_profile[:, 0], sol[:, 0], label="s_opt")
    ax[0, 1].legend()
    ax[0, 1].set_title("Arc Profile")

    ax[1, 1].plot(rc_path.speed_profile[:-1, 0], (rc_path.speed_profile[1:, 3] - rc_path.speed_profile[:-1, 3]) / dt, label="j")
    ax[1, 1].plot(rc_path.speed_profile[:-1, 0], (sol[1:, 2] - sol[:-1, 2]) / dt, label="j_opt")
    ax[1, 1].legend()
    ax[1, 1].set_title("Jerk Profile")

    fig2, ax2 = plt.subplots(1, 2)
    ax2[0].plot(rc_path.speed_profile[:, 0], rc_path.speed_profile[:, 1], label="s")
    ax2[0].plot(rc_path.speed_profile[:, 0], rc_path.speed_profile[:, 2], label="v")
    ax2[0].plot(rc_path.speed_profile[:, 0], rc_path.speed_profile[:, 3], label="a")
    ax2[0].plot(rc_path.speed_profile[:, 0], rc_path.path[:, 3] / np.max(rc_path.path[:, 3]), label="k")
    ax2[0].legend()
    ax2[0].set_title("Speed Profile")

    ax2[1].plot(rc_path.speed_profile[:, 0], sol[:, 0], label="s")
    ax2[1].plot(rc_path.speed_profile[:, 0], sol[:, 1], label="v")
    ax2[1].plot(rc_path.speed_profile[:, 0], sol[:, 2], label="a")
    ax2[1].plot(rc_path.speed_profile[:, 0], k_s / np.max(k_s), label="k")
    ax2[1].plot(rc_path.speed_profile[:, 0], sampled_k / np.max(sampled_k), label="sampled_k")
    ax2[1].legend()
    ax2[1].set_title("Speed Profile Optimized")

    plt.show()

    convert_angle(rc_path.path[:, 2])
    optimized_data = {}
    optimized_data["x"]         = resample(rc_path.path[:, 0], rc_path.speed_profile[:, 1], sol[:, 0]).tolist()
    optimized_data["y"]         = resample(rc_path.path[:, 1], rc_path.speed_profile[:, 1], sol[:, 0]).tolist()
    optimized_data["yaw"]       = resample(rc_path.path[:, 2], rc_path.speed_profile[:, 1], sol[:, 0]).tolist()
    optimized_data["curvature"] = np.array(k_s).squeeze().tolist()
    optimized_data["arcLength"] = sol[:, 0].tolist()
    optimized_data["vel"]       = sol[:, 1].tolist()
    optimized_data["acc"]       = sol[:, 2].tolist()
    optimized_data["time"]      = rc_path.speed_profile[:, 0].tolist()
    optimized_data["omega"]     = w_opt.tolist()
    optimized_data["acc_omega"] = dw_2.tolist()

    with open('config/track/opt_track.yaml', 'w') as f:
        yaml.dump(optimized_data, f)