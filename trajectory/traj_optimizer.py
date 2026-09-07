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

class TrajectoryOptimizer:
    def __init__(self, config: SpeedOptimizerConfig, traj: np.ndarray = None) -> None:
        """
        Optimize the trajectory to better fit vehicle dynamics
            traj: (n, 4) shape, each row is (arclength, vel, acc, curvature)
        """
        self.config = config
        self.traj = traj

    def initialize(self) -> None:
        self.dt = self.config.dt
        self.number_of_points = self.traj.shape[0]
        self.number_of_constraints = 3 * self.number_of_points - 3
        self.gradient = np.zeros((self.number_of_points * 3,))
        self.Hessian = np.zeros((self.number_of_points * 3, self.number_of_points * 3))
        self.constraint = np.zeros((self.number_of_constraints, self.number_of_points * 3))
        self.lower_bound_x = np.zeros((self.number_of_points * 3,))
        self.upper_bound_x = np.zeros((self.number_of_points * 3,))
        self.lower_bound_a = np.zeros((self.number_of_constraints,))
        self.upper_bound_a = np.zeros((self.number_of_constraints,))

        self.penalty_curv = np.abs(self.traj[:, 3]) * self.config.w_cur

    def setup_problem(self) -> None:
        scale = self.config.scale
        dt, dt_square = self.dt, self.dt ** 2
        w_s, w_v, w_a, w_j = self.config.w_s, self.config.w_v, self.config.w_a, self.config.w_j
        v_m, a_m, j_m = self.config.v_m, self.config.a_m, self.config.j_m
        w_s_e, w_v_e, w_a_e = self.config.w_s_e, self.config.w_v_e, self.config.w_a_e
        s_e, v_e, a_e = self.traj[-1, 0], 0, 0

        for i in range(self.number_of_points):
            self.gradient[i] = -1 * w_s * self.traj[i, 0] / scale[0]
            self.gradient[self.number_of_points + i] = -1 * w_v * self.traj[i, 1] / scale[1]
            # self.gradient[self.number_of_points * 2 + i] = -1 * w_a * self.traj[i, 2] / scale[2] # original paper do not minimize between a and a_ref

            self.Hessian[i, i] = self.config.w_s / (scale[0] * scale[0]) 
            self.Hessian[self.number_of_points + i, self.number_of_points + i] = (w_v + self.penalty_curv[i]) / (scale[1] * scale[1]) 
            self.Hessian[2 * self.number_of_points + i, 2 * self.number_of_points + i] = (w_a + 2 * w_j / dt_square) / (scale[2] * scale[2]) 

            self.lower_bound_x[i] = 0 * scale[0]
            self.lower_bound_x[self.number_of_points + i] = 0 * scale[1] # -v_m * scale[1]
            self.lower_bound_x[self.number_of_points * 2 + i] = -a_m * scale[2]

            self.upper_bound_x[i] = s_e * scale[0]
            self.upper_bound_x[self.number_of_points + i] = v_m * scale[1]
            self.upper_bound_x[self.number_of_points * 2 + i] = a_m * scale[2]

            # all about linear constraints
            if i < self.number_of_points - 1:
                # Dynamic of arclength: [s(i+1) - s(i)] / dt = Vel * dt + Acc * dt^2 / 2 + Jerk * dt^3 / 6
                # where Jerk is given by [s(i+1)'' - s(i)''] / dt
                self.constraint[i, i] = -1 * scale[1] * scale[2]
                self.constraint[i, i + 1] = 1 * scale[1] * scale[2]
                self.constraint[i, self.number_of_points + i] = -dt * scale[0] * scale[2]
                self.constraint[i, self.number_of_points * 2 + i] = -dt_square * scale[0] * scale[1] / 3
                self.constraint[i, self.number_of_points * 2 + 1 + i] = -dt_square * scale[0] * scale[1] / 6

                # Dynamic between speed and acceleration: [s(i+1)' - s(i)'] / dt = Acc * dt + Jerk * dt^2 / 2
                # where Jerk is given by [s(i+1)'' - s(i)''] / dt
                self.constraint[self.number_of_points - 1 + i, self.number_of_points + i] = -1 * scale[2]
                self.constraint[self.number_of_points - 1 + i, self.number_of_points + 1 + i] = 1 * scale[2]
                self.constraint[self.number_of_points - 1 + i, self.number_of_points * 2 + i] = -dt * scale[1] / 2
                self.constraint[self.number_of_points - 1 + i, self.number_of_points * 2 + 1 + i] = -dt * scale[1] / 2

                self.Hessian[2 * self.number_of_points + i, 2 * self.number_of_points + i + 1] = (-w_j / dt_square) / (scale[2] * scale[2])
                self.Hessian[2 * self.number_of_points + i + 1, 2 * self.number_of_points + i] = (-w_j / dt_square) / (scale[2] * scale[2])

                # Constraint on the jerk: -Jerk_max * dt <= s(i+1)'' - s(i)'' <= Jerk_max * dt
                self.constraint[self.number_of_points * 2 - 2 + i, self.number_of_points * 2 + i] = -1
                self.constraint[self.number_of_points * 2 - 2 + i, self.number_of_points * 2 + 1 + i] = 1
                self.lower_bound_a[self.number_of_points * 2 - 2 + i] = -j_m * dt * scale[2]
                self.upper_bound_a[self.number_of_points * 2 - 2 + i] = j_m * dt * scale[2]

        self.gradient[self.number_of_points - 1] -= 1 * w_s_e * s_e / scale[0]
        self.gradient[2 * self.number_of_points - 1] -= 1 * w_v_e * v_e / scale[1]
        self.gradient[3 * self.number_of_points - 1] -= 1 * w_a_e * a_e / scale[2]

        self.Hessian[2 * self.number_of_points, 2 * self.number_of_points] -= (w_j / dt_square) / (scale[2] * scale[2])

        # we may not need this as it is hard constraint below
        # self.Hessian[self.number_of_points - 1, self.number_of_points - 1] += w_s_e / (scale[0] * scale[0])
        # self.Hessian[2 * self.number_of_points - 1, 2 * self.number_of_points - 1] += w_v_e / (scale[1] * scale[1])
        # self.Hessian[3 * self.number_of_points - 1, 3 * self.number_of_points - 1] += (w_a_e -  w_j / dt_square) / (scale[2] * scale[2])

        self.upper_bound_x[0] = 0
        self.lower_bound_x[0] = 0
        self.lower_bound_x[self.number_of_points] = self.traj[0, 1] * scale[1]
        self.upper_bound_x[self.number_of_points] = self.traj[0, 1] * scale[1]
        self.lower_bound_x[2 * self.number_of_points] = self.traj[0, 2] * scale[2]
        self.upper_bound_x[2 * self.number_of_points] = self.traj[0, 2] * scale[2]

        self.upper_bound_x[self.number_of_points - 1] = s_e * scale[0]
        self.lower_bound_x[self.number_of_points - 1] = s_e * scale[0]
        self.lower_bound_x[2 * self.number_of_points - 1] = v_e * scale[1]
        self.upper_bound_x[2 * self.number_of_points - 1] = v_e * scale[1]
        self.lower_bound_x[3 * self.number_of_points - 1] = a_e * scale[2]
        self.upper_bound_x[3 * self.number_of_points - 1] = a_e * scale[2]

    def solve(self) -> np.ndarray:
        self.initialize()
        self.setup_problem()

        self.H_casadi = ca.DM(self.Hessian)
        self.linearMatrix_casadi = ca.DM(self.constraint)

        qp = {"h" : self.H_casadi.sparsity(), "a" : self.linearMatrix_casadi.sparsity()}
        opts = {"error_on_fail": True, "osqp.verbose": 0, "print_time": True}
        solver = ca.conic("SpeedOptimizer", "osqp", qp, opts) # osqp, qpoases
        args = {
            "h"   : self.H_casadi,
            "g"   : self.gradient,
            "lbx" : self.lower_bound_x,
            "ubx" : self.upper_bound_x,
            "a"   : self.constraint,
            "lba" : self.lower_bound_a,
            "uba" : self.upper_bound_a,
        }
        solution = solver(**args)
        return np.array(solution["x"]).reshape(-1, 3, order="F")

    def set_traj_with_bezier(self, path: np.ndarray, speed_profile: np.ndarray) -> None:
        self.traj = np.zeros((path.shape[0], 4))
        self.traj[:, 0] = speed_profile[:, 1]
        self.traj[:, 1] = speed_profile[:, 2]
        self.traj[:, 2] = speed_profile[:, 3]
        self.traj[:, 3] = path[:, 3]

    def solve_iteratively(self, iters: int = 3) -> np.ndarray:
        sol = self.solve()
        for i in range(iters - 1):
            self.traj[:, 3] = resample(self.traj[:, 3], self.traj[:, 0], sol[:, 0])
            self.traj[:, 0] = sol[:, 0]
            self.traj[:, 1] = sol[:, 1]
            self.traj[:, 2] = sol[:, 2]
            sol = self.solve()
        return sol