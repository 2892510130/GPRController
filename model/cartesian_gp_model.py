import numpy as np
import casadi as ca
from typing import Tuple

from . import DiscreteMethods
from .noise_model import GaussianNoiseModel
from .model_interface import ModelInterface
from .gp_model import GaussianModel, test_gaussian_model

from utils import GaussianConfig

class CartesianGaussianModel(ModelInterface):
    def __init__(
        self,
        integral_dt: int = 0.1,
        discrete_method: str = 'Euler',
        gaussian_opts: GaussianConfig | None = None,
        Q: np.ndarray | None = None,
        **kwargs
    ) -> None:
        self.name = "Kinematic Cartesian Gaussian"
        self.nx = 5
        self.nu = 2
        self.LINV = 3
        self.YAWV = 4
        self.discrete_method = DiscreteMethods[discrete_method](integral_dt)
        self.create_noise_model(Q, integral_dt)
        self.build_guassian_model(gaussian_opts, test = gaussian_opts.enable_test)
    
    def build_guassian_model(self, gaussian_opts: GaussianConfig, test: bool = False):
        self.gaussian_model = GaussianModel(gaussian_opts)
        self.feature_size = self.gaussian_model.feature_size

        self.gaussian_model_v = self.gaussian_model.gps[self.gaussian_model.LINV]
        self.kernel_v = self.gaussian_model_v.kern

        self.gaussian_model_w = self.gaussian_model.gps[self.gaussian_model.YAWV]
        self.kernel_w = self.gaussian_model_w.kern

        if test:
            test_gaussian_model(self.gaussian_model)
    
    def create_discrete_model(
        self,
        dt: float,
        gpr_input: np.ndarray,
        linear_method: int = 0,
        x_var: np.ndarray | None = None,
        gpr_method: int = 0,
        m_curvature: float = 0.0,
        compute_dvar_dx: bool = False
    ) -> np.ndarray:
        # m_curvature is accepted for interface compatibility with the Frenet
        # models (and with LinearMPC); the Cartesian models do not use it.
        self.A = np.zeros((self.nx, self.nx))
        self.B = np.zeros((self.nx, self.nu))
        self.C = np.zeros((self.nx, 1))

        self.set_linearize_point_type(linear_method)

        self.A[0, 2] = -1.0 * self.x_lin[3] * np.sin(self.x_lin[2])
        self.A[0, 3] = np.cos(self.x_lin[2])
        self.A[1, 2] = self.x_lin[3] * np.cos(self.x_lin[2])
        self.A[1, 3] = np.sin(self.x_lin[2])
        self.A[2, 4] = 1.0

        self.B[3, 0] = 1.0
        self.B[4, 1] = 1.0

        self.C[0, 0] = -1.0 * self.A[0, 2] * self.x_lin[2]
        self.C[1, 0] = -1.0 * self.A[1, 2] * self.x_lin[2]

        if gpr_method >= 0:
            self.mean_v, self.var_v = self.gaussian_model_v.predict_noiseless(gpr_input, full_cov=True, compute_dvar_dx=compute_dvar_dx)
            self.mean_w, self.var_w = self.gaussian_model_w.predict_noiseless(gpr_input, full_cov=True, compute_dvar_dx=compute_dvar_dx)

            self.C[3] += self.mean_v.squeeze()
            self.C[4] += self.mean_w.squeeze()

            if x_var is not None:
                x_var = self.A @ x_var @ self.A.transpose()
                x_var[self.LINV, self.LINV] += self.var_v * dt ** 2
                x_var[self.YAWV, self.YAWV] += self.var_w * dt ** 2
        
        self.A, self.B, self.C = self.discrete_method.discretize(self.A, self.B, self.C)

        return x_var
    
    def create_noise_model(self, Q: np.ndarray | None, integral_dt: float) -> None:
        if Q is None:
            Q = np.diag([0.01, 0.01, 0.01, 0.1, 0.1])
        self.noise_model = GaussianNoiseModel(Q, integral_dt)
    
    def set_predict_traj(self, x_guess: np.ndarray, pred_traj: np.ndarray) -> np.ndarray:
        # x_guess is the flattened predicted state sequence (nx * N); the first
        # five entries of every state are [x, y, yaw, v, w].
        return np.asarray(x_guess).reshape(-1, self.nx)[:, :5].copy()

class AugmentedCartesianGaussianModel(ModelInterface):
    def __init__(
        self,
        integral_dt: int = 0.1,
        discrete_method: str = 'Euler',
        gaussian_opts: GaussianConfig | None = None,
        Q: np.ndarray | None = None,
        **kwargs
    ) -> None:
        self.name = "Kinematic Augmented Cartesian Gaussian"
        self.nx = 5 + (gaussian_opts.input_window - 1) * 4
        self.nu = 2
        self.input_window = gaussian_opts.input_window
        self.LINV = 3
        self.YAWV = 4
        self.discrete_method = DiscreteMethods[discrete_method](integral_dt)
        self.create_noise_model(Q, integral_dt)
        self.build_guassian_model(gaussian_opts, test = gaussian_opts.enable_test)

    def change_vector_to_model2(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray]:
        new_x = np.zeros((self.nx, 1))
        new_u = np.zeros((2, 1))
        new_x[:x.shape[0]] = x
        new_x[x.shape[0]:] = u[2:]
        new_u = u[:2]
        return new_x, new_u
    
    def build_guassian_model(self, gaussian_opts: GaussianConfig, test: bool = False) -> None:
        self.gaussian_model = GaussianModel(gaussian_opts)
        self.feature_size = self.gaussian_model.feature_size

        self.gaussian_model_v = self.gaussian_model.gps[self.gaussian_model.LINV]
        self.kernel_v = self.gaussian_model_v.kern

        self.gaussian_model_w = self.gaussian_model.gps[self.gaussian_model.YAWV]
        self.kernel_w = self.gaussian_model_w.kern

        if test:
            test_gaussian_model(self.gaussian_model)

    def create_discrete_model(
        self,
        dt: float,
        gpr_input: np.ndarray,
        linear_method: int = 0,
        x_var: np.ndarray | None = None,
        gpr_method: int = 0,
        m_curvature: float = 0.0,
        compute_dvar_dx: bool = False
    ) -> np.ndarray:
        # m_curvature is accepted for interface compatibility with the Frenet
        # models (and with LinearMPC); the Cartesian models do not use it.
        self.A = np.zeros((self.nx, self.nx))
        self.B = np.zeros((self.nx, self.nu))
        self.C = np.zeros((self.nx, 1))

        self.set_linearize_point_type(linear_method)

        self.A[0, 2] = -1.0 * self.x_lin[3] * np.sin(self.x_lin[2])
        self.A[0, 3] = np.cos(self.x_lin[2])
        self.A[1, 2] = self.x_lin[3] * np.cos(self.x_lin[2])
        self.A[1, 3] = np.sin(self.x_lin[2])
        self.A[2, 4] = 1.0

        self.B[3, 0] = 1.0
        self.B[4, 1] = 1.0

        self.C[0, 0] = -1.0 * self.A[0, 2] * self.x_lin[2]
        self.C[1, 0] = -1.0 * self.A[1, 2] * self.x_lin[2]


        if gpr_method >= 0: # add the noise or not?
            self.mean_v, self.var_v = self.gaussian_model_v.predict_noiseless(gpr_input, full_cov=True, compute_dvar_dx=compute_dvar_dx)
            self.mean_w, self.var_w = self.gaussian_model_w.predict_noiseless(gpr_input, full_cov=True, compute_dvar_dx=compute_dvar_dx)

            self.C[3] += self.mean_v.squeeze()
            self.C[4] += self.mean_w.squeeze()

        
        if gpr_method >= 1:
            self.get_gradient(gpr_input)
            mask = np.zeros(self.input_window * 4, dtype = bool)
            for i in range(0, self.input_window * 4, 4):
                mask[i:i+2] = True
            mask2 = mask.copy()
            mask2[2:4] = True
            self.A[3, 3:3+2*self.input_window] += self.dmu_dv[mask].squeeze()
            self.A[4, 3:3+2*self.input_window] += self.dmu_dw[mask].squeeze()
            self.A[3, 3+2*self.input_window:]  += self.dmu_dv[~mask2].squeeze()
            self.A[4, 3+2*self.input_window:]  += self.dmu_dw[~mask2].squeeze()
            self.B[3, :] += self.dmu_dv[2:4].squeeze()
            self.B[4, :] += self.dmu_dw[2:4].squeeze()
            self.C[3] -= (gpr_input @ self.dmu_dv).squeeze()
            self.C[4] -= (gpr_input @ self.dmu_dw).squeeze()

            if compute_dvar_dx is True:
                self.compute_variance_aware(mask, mask2)

        self.A, self.B, self.C = self.discrete_method.discretize(self.A, self.B, self.C)
        self.A[5:, :] = 0

        for i in range(self.input_window - 1):
            self.A[5 + 2 * i, 3 + 2 * i] = 1
            self.A[6 + 2 * i, 4 + 2 * i] = 1

        for i in range(self.input_window - 2):
            self.A[5 + 2 * (i + self.input_window), 3 + 2 * (i + self.input_window)] = 1
            self.A[6 + 2 * (i + self.input_window), 4 + 2 * (i + self.input_window)] = 1

        self.B[3 + 2 * self.input_window, 0] = 1
        self.B[4 + 2 * self.input_window, 1] = 1

        if gpr_method >= 0 and x_var is not None:
            # white noise of measurement is not added
            x_var = self.A @ x_var @ self.A.transpose()
            x_var[self.LINV, self.LINV] += self.var_v * dt ** 2
            x_var[self.YAWV, self.YAWV] += self.var_w * dt ** 2

        return x_var

    def get_gradient(self, gpr_input: np.ndarray) -> None:
        self.dmu_dv, self.dmu_dw = self.gaussian_model.dmu_dx(gpr_input)

    def compute_variance_aware(self, mask: np.ndarray, mask2: np.ndarray) -> None:
        self.A_sigma_v = np.zeros((1, self.nx))
        self.B_sigma_v = np.zeros((1, self.nu))
        self.A_sigma_w = np.zeros((1, self.nx))
        self.B_sigma_w = np.zeros((1, self.nu))
        self.A_sigma_v[0, 3:3+2*self.input_window] += self.gaussian_model_v.dvar_dx_data[mask].squeeze()
        self.A_sigma_v[0, 3+2*self.input_window:] += self.gaussian_model_v.dvar_dx_data[~mask2].squeeze()
        self.A_sigma_w[0, 3:3+2*self.input_window] += self.gaussian_model_w.dvar_dx_data[mask].squeeze()
        self.A_sigma_w[0, 3+2*self.input_window:] += self.gaussian_model_w.dvar_dx_data[~mask2].squeeze()

        self.B_sigma_v[0, :] += self.gaussian_model_v.dvar_dx_data[2:4].squeeze()
        self.B_sigma_w[0, :] += self.gaussian_model_w.dvar_dx_data[2:4].squeeze()

    def create_noise_model(self, Q: np.ndarray, integral_dt: float) -> None:
        if Q is None:
            Q = np.diag([0.01, 0.01, 0.01, 0.1, 0.1])
        self.noise_model = GaussianNoiseModel(Q, integral_dt)
    
    def set_predict_traj(self, x_guess: np.ndarray, pred_traj: np.ndarray) -> np.ndarray:
        pred_traj = np.asarray(x_guess).reshape(-1, self.nx)
        return pred_traj[:, :5].copy()