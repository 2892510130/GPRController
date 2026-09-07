import numpy as np
import casadi as ca

from utils import Config, Logger, normalize_angle
from model import ModelLists

class LinearMPC:
    def __init__(self, opts:Config, path, speed_profile):
        self.opts = opts.mpc
        self.gaussian_dict = opts.gaussian
        self.logger = Logger("log/linear_mpc_log.txt")

        self.model = ModelLists[self.opts.model](
            integral_dt=self.opts.mpc_dt,
            casadi_model=False, 
            discrete_method=self.opts.discrete_method, 
            gaussian_opts=self.gaussian_dict
        )

        if "Gaussian" in self.model.name:
            self.gpr_input = np.zeros((1, self.model.feature_size))
            self.gpr_per_feature_size = 4
            self.x_guess_gpr = np.zeros((self.model.nx, 1))
            self.augmented_x0 = np.zeros((self.model.YAWV + 1 + (self.model.input_window - 1) * 2, 1))
            self.augmented_u0 = np.zeros(((self.model.input_window - 1) * 3, 1))
            if self.opts.variance_aware:
                self.A_sigma = np.zeros((1, self.model.nx * self.opts.mpcWindow))
                self.B_sigma = np.zeros((1, self.model.nu * self.opts.mpcWindow))

        self.path = path
        self.speed_profile = speed_profile
        self.u_guess = np.zeros((self.model.nu * self.opts.mpcWindow, 1))
        self.x_guess = np.zeros((self.model.nx * self.opts.mpcWindow, 1))
        self.state_prediction_nominal = np.zeros((self.model.nx, 1))
        self.last_command = np.zeros((self.model.nu, 1))
        self.linear_method = self.opts.linear_method # 0 for ref, 1 for last solution
        self.setup_weight_matrices()

        self.first_iter = True
        self.record_start = False

        if self.opts.record_data == True:
            self.gpr_file = open(self.opts.record_data_path, 'w')

    def initialize(self, cur_state, dt):
        self.mpc_dt = dt
        self.cur_state = cur_state
        self.get_initial_state(cur_state)
        # self.x_guess[:self.model.nx] = self.x0 # important change? we want the newest x0 as the guess
        self.setup_ref(self.index)
        if self.linear_method == 1 and self.first_iter:
            self.prepare(dt)
        self.setup_matrices(self.index, dt)
        self.setup_bound()
        self.setup_qp_problem()

    def get_initial_state(self, cur_state):
        self.lat_error, self.head_error, self.index = self.compute_error(cur_state)
        x0 = np.zeros((self.model.YAWV + 1, 1))

        if "Cartesian" in self.model.name:
            x0 = cur_state
        elif "Frenet" in self.model.name:
            x0[0], x0[1], x0[2], x0[3] = self.lat_error, self.head_error, cur_state[3], cur_state[4]

        if "Gaussian" in self.model.name:
            self.gpr_input_tmp = self.gpr_input.copy()
            self.gpr_input_step_1 = self.gpr_input.copy()
            self.gpr_input[:, self.gpr_per_feature_size:] = self.gpr_input[:, :-self.gpr_per_feature_size]

        if "Augmented" in self.model.name:
            self.augmented_x0[self.model.LINV+2:] = self.augmented_x0[self.model.LINV:-2]
            self.augmented_x0[:x0.shape[0]] = x0
            self.augmented_u0[:2] = self.u_guess[:2]
            self.augmented_u0[2:] = self.augmented_u0[:-2]
            if self.first_iter:
                for i in range(self.model.input_window - 1):
                    history_index = max(0, self.index - 1 - i)
                    self.augmented_x0[self.model.LINV + 2 + 2 * i] = self.speed_profile[history_index, 2]
                    self.augmented_x0[self.model.YAWV + 2 + 2 * i] = self.speed_profile[history_index, 4]
                    self.augmented_u0[2 + 2 * i] = self.speed_profile[history_index, 3]
                    self.augmented_u0[2 + 2 * i + 1] = self.speed_profile[history_index, 5]
                self.augmented_u0[0] = self.speed_profile[self.index, 3]
                self.augmented_u0[1] = self.speed_profile[self.index, 5]
            x0, _ = self.model.change_vector_to_model2(self.augmented_x0, self.augmented_u0)
        
        self.u_guess[:-self.model.nu, :] = self.u_guess[self.model.nu:, :] # shift the guess vectors
        self.x0 = x0.copy()

    def compute_error(self, cur_state):
        distances = np.linalg.norm(self.path[:, :2] - cur_state[:2].reshape(-1, 2), axis=1)
        nearest_index = np.argmin(distances)
        dx = cur_state[0] - self.path[nearest_index, 0]
        dy = cur_state[1] - self.path[nearest_index, 1]
        dyaw = cur_state[2] - self.path[nearest_index, 2]
        lat_error = np.cos(self.path[nearest_index, 2]) * dy - np.sin(self.path[nearest_index, 2]) * dx
        head_error = normalize_angle(dyaw)
        return lat_error, head_error, nearest_index
    
    def setup_weight_matrices(self):
        N = self.opts.mpcWindow
        nx, nu = self.model.nx, self.model.nu
        self.Q = np.zeros((nx * N, nx * N))
        self.R = np.zeros((nu * N, nu * N))
        
        for i in range(N):
            if i < N - 1 : self.Q[i * nx : (i+1) * nx, i * nx : (i+1) * nx] = self.opts.Q
            else : self.Q[i * nx : (i+1) * nx, i * nx : (i+1) * nx] = self.opts.Q_final
            self.R[i * nu : (i+1) * nu, i * nu : (i+1) * nu] = self.opts.R

    def first_step_nominal_matrices(self, index, dt):
        nx, nu = self.model.nx, self.model.nu
        self.model.set_reference(self.xr[0 : nx], self.ur[0 : nu])
        self.model.set_guess(self.x_guess[0 : nx], self.u_guess[0 : nu])
        if self.model.name == "Kinematic Cartesian":
            self.model.create_discrete_model(dt, self.linear_method)
        elif self.model.name == "Kinematic Frenet":
            self.model.create_discrete_model(dt, self.path[index, 3], self.linear_method) # Give the curvature
        elif "Gaussian" in self.model.name:
            self.model.create_discrete_model(
                dt, 
                self.gpr_input_tmp, 
                linear_method = self.linear_method, 
                gpr_method = -1, 
                m_curvature = self.path[index, 3]
            )
        
        self.state_prediction_nominal = self.model.A @ self.x0 + self.model.B @ self.u_guess[0 : nu] + self.model.C

    def setup_matrices(self, index, dt):
        N = self.opts.mpcWindow
        nx, nu = self.model.nx, self.model.nu
        self.Aex = np.zeros((nx * N, nx))
        self.Bex = np.zeros((nx * N, nu * N))
        self.Cex = np.zeros((nx * N, 1))
        if self.opts.propagete_cov:
            self.x_var = np.zeros((nx, nx))
            self.x_var_list = []
            self.var_list = []
        x_pred = self.x0.copy()
        self.gpr_list = []

        for i in range(N):
            self.model.set_reference(self.xr[i * nx : (i+1) * nx], self.ur[i * nu : (i+1) * nu])
            self.model.set_guess(self.x_guess[i * nx : (i+1) * nx], self.u_guess[i * nu : (i+1) * nu])
            if self.model.name == "Kinematic Cartesian":
                self.model.create_discrete_model(dt, self.linear_method)
            elif self.model.name == "Kinematic Frenet":
                self.model.create_discrete_model(dt, self.path[index, 3], self.linear_method) # Give the curvature
            elif "Gaussian" in self.model.name:
                if self.gaussian_dict.lagacy:
                    x_pred = self.update_gpr_lagacy(i, nx, nu, x_pred)
                else:
                    self.update_gpr(i, nx, nu)

                if self.opts.propagete_cov:
                    self.x_var = self.model.create_discrete_model(
                        dt, 
                        self.gpr_input_tmp, 
                        linear_method = self.linear_method, 
                        x_var = self.x_var, 
                        gpr_method = self.gaussian_dict.gpr_method, 
                        m_curvature = self.path[index, 3],
                        compute_dvar_dx = self.opts.variance_aware
                    )
                    self.x_var_list.append(self.x_var)
                    self.var_list.append([self.model.var_v, self.model.var_w])
                else:
                    self.model.create_discrete_model(
                        dt, 
                        self.gpr_input_tmp, 
                        linear_method = self.linear_method, 
                        gpr_method = self.gaussian_dict.gpr_method, 
                        m_curvature = self.path[index, 3],
                        compute_dvar_dx = self.opts.variance_aware
                    )
                self.gpr_list.append(self.gpr_input_tmp.copy())

                if self.opts.variance_aware and i < self.opts.aware_len:
                    self.A_sigma[0, i * nx : (i+1) * nx] = self.opts.lambda_var[0] * self.model.A_sigma_v + self.opts.lambda_var[1] * self.model.A_sigma_w
                    self.B_sigma[0, i * nu : (i+1) * nu] = self.opts.lambda_var[0] * self.model.B_sigma_v + self.opts.lambda_var[1] * self.model.B_sigma_w

            if i == 0:
                self.Aex[i * nx : (i + 1) * nx, :] = self.model.A
                self.Cex[i * nx : (i + 1) * nx, :] = self.model.C
            else:
                self.Aex[i * nx : (i + 1) * nx, :] = self.model.A @ self.Aex[(i - 1) * nx : i * nx, :]
                self.Cex[i * nx : (i + 1) * nx, :] = self.model.A @ self.Cex[(i - 1) * nx : i * nx, :] + self.model.C

            self.Bex[i * nx : (i + 1) * nx, i * nu : (i + 1) * nu] = self.model.B
            for j in range(i):
                self.Bex[i * nx : (i + 1) * nx, j * nu : (j + 1) * nu] = self.model.A @ \
                    self.Bex[(i - 1) * nx : i * nx, j * nu : (j + 1) * nu]

            index = index + 1 if (index < self.path.shape[0] - 1) else index
            # index = (index + 1) % self.path.shape[0]

    def setup_bound(self):
        N = self.opts.mpcWindow
        nx, nu = self.model.nx, self.model.nu
        nc = N * nu - nu
        # nc = N * 2 - 2 # sub constraints
        max_u = self.opts.max_input
        max_delta_u = self.opts.max_delta_input
        self.lbx = np.zeros((nu * N, 1))
        self.ubx = np.zeros((nu * N, 1))
        self.lba = np.zeros((nc, 1))
        self.uba = np.zeros((nc, 1))
        self.linearMatrix = np.zeros((nc, nu * N))

        for i in range(N):
            self.lbx[i * nu] = -1 * max_u[0]
            self.lbx[i * nu + 1] = -1 * max_u[1]
            self.ubx[i * nu] = max_u[0]
            self.ubx[i * nu + 1] = max_u[1]

            # if i * nu < nc:
            #     for k in range(nu):
            #         self.lba[i * nu + k] = -1 * max_delta_u[k % 2] * self.mpc_dt
            #         self.uba[i * nu + k] = max_delta_u[k % 2] * self.mpc_dt
            #         self.linearMatrix[i * nu + k, i * nu + k] = -1
            #         self.linearMatrix[i * nu + k, i * nu + nu + k] = 1

            if i != N - 1: # we may not constraint all the u, as some u are duplicated. (in test all cons and sub cons they are the same)
                for k in range(2):
                    self.lba[i * nu + k] = -1 * max_delta_u[k % 2] * self.mpc_dt
                    self.uba[i * nu + k] = max_delta_u[k % 2] * self.mpc_dt
                    self.linearMatrix[i * nu + k, i * nu + k] = -1
                    self.linearMatrix[i * nu + k, i * nu + nu + k] = 1
                
                if nu > 2: # for augmented model 1, but seems like failed
                    for k in range(2, 2 * self.model.input_window):
                        self.linearMatrix[i * nu + k, i * nu + k - 2] = -1
                        self.linearMatrix[i * nu + k, i * nu + nu + k] = 1

        
        control_dt = self.opts.control_dt
        self.lbx[0] = np.maximum(self.last_command[0] - max_delta_u[0] * control_dt, -1 * max_u[0])
        self.ubx[0] = np.minimum(self.last_command[0] + max_delta_u[0] * control_dt, max_u[0])
        self.lbx[1] = np.maximum(self.last_command[1] - max_delta_u[1] * control_dt, -1 * max_u[1])
        self.ubx[1] = np.minimum(self.last_command[1] + max_delta_u[1] * control_dt, max_u[1])
        # print(f"[Bound]: {float(self.last_command[1]):+.3f}, "
        #       f"{float(self.last_command[1] - max_delta_u[1] * control_dt):+.3f}, "
        #       f"{float(self.last_command[1] + max_delta_u[1] * control_dt):+.3f}")

    def setup_ref(self, index):
        N = self.opts.mpcWindow
        nx, nu = self.model.nx, self.model.nu
        self.xr = np.zeros((nx * (N + 1), 1))
        self.ur = np.zeros((nu * N, 1))
        self.pred_traj = np.zeros((N, 5))

        for i in range(N + 1):
            if i != N:
                self.ur[nu * i]     = self.speed_profile[index, 3]
                self.ur[nu * i + 1] = self.speed_profile[index, 5]
                self.pred_traj[i, 0] = self.path[index, 0]
                self.pred_traj[i, 1] = self.path[index, 1]
                self.pred_traj[i, 2] = self.path[index, 2]

            if "Cartesian" in self.model.name:
                self.xr[i * nx + 0] = self.path[index, 0]
                self.xr[i * nx + 1] = self.path[index, 1]
                self.xr[i * nx + 2] = self.path[index, 2]
                self.xr[i * nx + 3] = self.speed_profile[index, 2]
                self.xr[i * nx + 4] = self.speed_profile[index, 4]
            elif "Frenet" in self.model.name:
                self.xr[i * nx + 2] = self.speed_profile[index, 2]
                self.xr[i * nx + 3] = self.speed_profile[index, 4]
            
            if "Augmented" in self.model.name: # if not added, the delta_w will be larger
                for k in range(self.model.input_window - 1):
                    tmp_index = max(0, index - 1 - k)
                    self.xr[i * nx + self.model.LINV + 2 + 2 * k] = self.speed_profile[tmp_index, 2]
                    self.xr[i * nx + self.model.YAWV + 2 + 2 * k] = self.speed_profile[tmp_index, 4]
                    self.xr[i * nx + self.model.LINV + 2 * self.model.input_window + 2 * k] = self.speed_profile[tmp_index, 3]
                    self.xr[i * nx + self.model.YAWV + 2 * self.model.input_window + 2 * k] = self.speed_profile[tmp_index, 5]
            
            index = index + 1 if (index < self.path.shape[0] - 1) else index
            # index = (index + 1) % self.path.shape[0]
            
    def setup_qp_problem(self):
        x_delta = self.Aex @ self.x0 + self.Cex - self.xr[self.model.nx:]
        if "Cartesian" in self.model.name:
            x_delta[2::self.model.nx] = normalize_angle(x_delta[2::self.model.nx])

        H = self.Bex.transpose() @ self.Q @ self.Bex + self.R
        self.gradient = (x_delta.transpose() @ self.Q @ self.Bex).transpose()
        # self.gradient = (x_delta.transpose() @ self.Q @ self.Bex - self.ur.transpose() @ self.R).transpose()
        if self.opts.variance_aware:
            self.gradient += (self.A_sigma @ self.Bex + self.B_sigma).transpose()

        self.H_casadi = ca.DM(H)
        self.H_casadi_sparse = self.H_casadi.sparsity()
        self.linearMatrix_casadi = ca.DM(self.linearMatrix)
        self.linearMatrix_casadi_sparse = self.linearMatrix_casadi.sparsity()

    def update_gpr_lagacy(self, i, nx, nu, x_pred):
        self.gpr_input_tmp[:, self.gpr_per_feature_size:] = self.gpr_input_tmp[:, :-self.gpr_per_feature_size]

        if i != 0:
            # x_pred[:] = self.model.A @ x_pred + self.model.B @ self.gpr_input_tmp[:, 2:4].T + self.model.C
            x_pred[:] = self.model.A @ x_pred + self.model.B @ self.u_guess[(i-1) * nu : i * nu] + self.model.C

        self.gpr_input_tmp[0, 0] = x_pred[self.model.LINV]
        self.gpr_input_tmp[0, 1] = x_pred[self.model.YAWV]
        self.gpr_input_tmp[0, 2] = self.u_guess[i * nu]
        self.gpr_input_tmp[0, 3] = self.u_guess[i * nu + 1]

        return x_pred
    
    def update_gpr(self, i, nx, nu):
        self.gpr_input_tmp[:, self.gpr_per_feature_size:] = self.gpr_input_tmp[:, :-self.gpr_per_feature_size]
        if self.linear_method == 0:
            self.gpr_input_tmp[0, 0] = self.xr[self.model.LINV + i * nx]
            self.gpr_input_tmp[0, 1] = self.xr[self.model.YAWV + i * nx]
            self.gpr_input_tmp[0, 2] = self.ur[i * nu]
            self.gpr_input_tmp[0, 3] = self.ur[i * nu + 1]
        elif self.linear_method == 1:
            self.gpr_input_tmp[0, 0] = self.x_guess[self.model.LINV + i * nx]
            self.gpr_input_tmp[0, 1] = self.x_guess[self.model.YAWV + i * nx]
            self.gpr_input_tmp[0, 2] = self.u_guess[i * nu]
            self.gpr_input_tmp[0, 3] = self.u_guess[i * nu + 1]
        
        # if i == 0:
        #     self.gpr_input_tmp[0, 0] = self.x0[self.model.LINV]
        #     self.gpr_input_tmp[0, 1] = self.x0[self.model.YAWV]

    def solve(self):
        # 1. setup the solver
        if self.opts.add_delta_constraints:
            qp = {"h" : self.H_casadi_sparse, "a" : self.linearMatrix_casadi_sparse}
        else:
            qp = {"h" : self.H_casadi_sparse}
        
        opts = {"error_on_fail": False, "print_time": self.opts.print_time}
        if self.opts.solver == "osqp":
            opts["osqp.verbose"] = 0
        solver = ca.conic("LinearMPC", self.opts.solver, qp, opts)
        
        # 2. first iter, solve and get initial guess for warm start
        if self.first_iter and self.linear_method == 0:
            args = {"h"   : self.H_casadi,
                    "g"   : self.gradient,
                    "lbx" : self.lbx,
                    "ubx" : self.ubx,
            }
            args_append = {
                "a"   : self.linearMatrix,
                "lba" : self.lba,
                "uba" : self.uba,
            }
            if self.opts.add_delta_constraints:
                args.update(args_append)
            solution = solver(**args)
            self.u_guess = np.array(solution["x"])
            x_guess_tmp = self.Aex @ self.x0 + self.Bex @ self.u_guess + self.Cex
            self.x_guess[self.model.nx:] = x_guess_tmp[:-self.model.nx] # still in this time step, first x should still be x0_guess
            self.lam_x = solution["lam_x"]
            self.lam_a = solution["lam_a"]
        
        # 3. solve and get the control
        args = {"h"      : self.H_casadi,
                "g"      : self.gradient,
                "lbx"    : self.lbx,
                "ubx"    : self.ubx,
                "x0"     : self.u_guess,
                "lam_x0" : self.lam_x,
                "lam_a0" : self.lam_a}
        args_append = {
            "a"   : self.linearMatrix,
            "lba" : self.lba,
            "uba" : self.uba,
        }
        if self.opts.add_delta_constraints:
            args.update(args_append)
        solution = solver(**args)
        self.u_guess = np.array(solution["x"])
        self.lam_x = solution["lam_x"]
        self.lam_a = solution["lam_a"]

        self.delta_v = (self.x0[self.model.LINV] - self.x_guess_gpr[self.model.LINV]) / self.opts.control_dt
        self.delta_w = (self.x0[self.model.YAWV] - self.x_guess_gpr[self.model.YAWV]) / self.opts.control_dt
        self.delta_v_nominal = (self.x0[self.model.LINV] - self.state_prediction_nominal[self.model.LINV]) / self.opts.control_dt
        self.delta_w_nominal = (self.x0[self.model.YAWV] - self.state_prediction_nominal[self.model.YAWV]) / self.opts.control_dt
        self.first_step_nominal_matrices(self.index, self.opts.control_dt)

        # 4. record gpr data
        if self.opts.record_data and self.first_iter is False:
            self.record_gpr_data()

        # 5. update gpr input
        if "Gaussian" in self.model.name:
            self.update_gp_model()

        self.x_guess = self.Aex @ self.x0 + self.Bex @ self.u_guess + self.Cex
        self.ref_for_visual = self.pred_traj.copy()
        self.pred_traj = self.model.set_predict_traj(self.x_guess, self.pred_traj)
        self.x_guess_gpr = self.x_guess[:self.model.nx]
        self.last_command = self.u_guess[:self.model.nu].copy()
        self.first_iter = False

        # For the logging system
        self.log_data()

        return self.u_guess

    def prepare(self, dt):
        print("[LinearMPC]: First round, use reference to prepare.")
        tmp = self.linear_method
        tmp_lagacy = self.gaussian_dict.lagacy
        self.gaussian_dict.lagacy = False
        self.linear_method = 0
        self.setup_matrices(self.index, dt)
        self.setup_bound()
        self.setup_qp_problem()
        self.linear_method = tmp
        self.gaussian_dict.lagacy = tmp_lagacy

        if self.opts.add_delta_constraints:
            qp = {"h" : self.H_casadi_sparse, "a" : self.linearMatrix_casadi_sparse}
        else:
            qp = {"h" : self.H_casadi_sparse}
        opts = {"osqp.verbose": 0, "print_time": self.opts.print_time}
        solver = ca.conic("LinearMPC", "osqp", qp, opts)
        args = {
            "h"   : self.H_casadi,
            "g"   : self.gradient,
            "lbx" : self.lbx,
            "ubx" : self.ubx,
        }
        args_append = {
            "a"   : self.linearMatrix,
            "lba" : self.lba,
            "uba" : self.uba,
        }
        if self.opts.add_delta_constraints:
            args.update(args_append)
        solution = solver(**args)
        self.u_guess = np.array(solution["x"])
        self.lam_x = solution["lam_x"]
        self.lam_a = solution["lam_a"]
        x_guess_tmp = self.Aex @ self.x0 + self.Bex @ self.u_guess + self.Cex
        self.x_guess[self.model.nx:] = x_guess_tmp[:-self.model.nx] # still in this time step, first x should still be x0
        if "Gaussian" in self.model.name: self.gpr_input_tmp = self.gpr_input.copy() # should not change the gpr input

    def update_gp_model(self):
        self.gpr_input[:, 0] = self.x0[self.model.LINV]
        self.gpr_input[:, 1] = self.x0[self.model.YAWV]
        self.gpr_input[:, 2] = self.u_guess[0]
        self.gpr_input[:, 3] = self.u_guess[1]
        if self.gaussian_dict.update_mu_su is True and self.first_iter is False:
            self.model.gaussian_model.gps[0].update(self.gpr_input_step_1.copy(), self.delta_v_nominal)
            self.model.gaussian_model.gps[1].update(self.gpr_input_step_1.copy(), self.delta_w_nominal)
        if self.gaussian_dict.add_mu_su is True:
            for gp in self.model.gaussian_model.gps:
                var, aa = gp.check_new_inducing_var(self.gpr_input.copy())
                if var > aa * 0.01 and gp.M <= 10 * self.gaussian_dict.inducing_data_number:
                    gp.add_new_inducing_quick(self.gpr_input.copy())

    def record_gpr_data(self):
        state_v, state_w = self.x0[self.model.LINV], self.x0[self.model.YAWV]
        control_v, control_w = self.u_guess[0], self.u_guess[1]
        self.gpr_file.write(str(float(self.delta_v_nominal)) + ' ' + str(float(self.delta_w_nominal)) + ' ' + str(float(state_v)) + ' ' + str(float(state_w)) + 
                            ' ' + str(float(control_v)) + ' ' + str(float(control_w)) + '\n')
    
    def log_data(self):
        self.logger.debug("LinearMPC] Current State: ", self.index)
        self.logger.debug("LinearMPC] Delta State Nominal: ", [float(self.delta_v_nominal), float(self.delta_w_nominal)])
        self.logger.debug("LinearMPC] Delta State GPR: ", [float(self.delta_v), float(self.delta_w)])
        self.logger.debug("LinearMPC] GPR Input: ", [float(i) for i in self.gpr_input_step_1.squeeze()])
        self.logger.debug("LinearMPC] Matrix C:\n", self.Cex[:self.model.nx].squeeze())
        self.logger.debug("LinearMPC] Last Command: ", self.last_command.squeeze())

        if self.gaussian_dict.sparse_method == 2 or self.gaussian_dict.sparse_method == 3:
            for i, gp in enumerate(self.model.gaussian_model.gps):
                self.logger.debug(f"LinearMPC] GPR {i} Inducing Size: ", gp.U.shape)
                self.logger.debug(f"LinearMPC] GPR {i} predict test: ", gp.predict_noiseless(gp.X[0].reshape(1, -1), full_cov = True))

        if self.opts.variance_aware:
            pass
        
        self.logger.empty_line(3)
    
    def save_model(self):
        if self.gaussian_dict.sparse_method == 2 or self.gaussian_dict.sparse_method == 3:
            for i, gp in enumerate(self.model.gaussian_model.gps):
                gp.save_model("_" + str(i))