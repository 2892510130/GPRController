import numpy as np
import casadi as ca

from . import DiscreteMethods
from .noise_model import GaussianNoiseModel
from .model_interface import ModelInterface

class FrenetModel(ModelInterface):
    def __init__(
        self,
        integral_dt: int = 0.1,
        discrete_method: str = 'Euler',
        Q: np.ndarray | None = None,
        **kwargs
    ) -> None:
        self.name = "Kinematic Frenet"
        self.nx = 4
        self.nu = 2
        self.np = 1
        self.LINV = 2
        self.YAWV = 3
        self.discrete_method = DiscreteMethods[discrete_method](integral_dt)
        self.create_model()
        self.create_integrator(integral_dt)
        self.create_noise_model(Q, integral_dt)

    def create_model(self) -> None:
        self.x = ca.MX.sym("x", self.nx, 1)
        self.u = ca.MX.sym("u", self.nu, 1)
        self.kappa = ca.MX.sym("kappa")
        self.p = ca.vertcat(self.u, self.kappa)

        self.x_dot = ca.vertcat(
            self.x[2] * ca.sin(self.x[1]),
            self.x[3] - self.x[2] * ca.cos(self.x[1]) * self.kappa / (1 - self.x[0] * self.kappa),
            self.u[0],
            self.u[1]
        )

    def create_integrator(self, tf: float, solver: str = 'rk') -> None:
        dae = {'x': self.x, 'p': self.p, 'ode': self.x_dot}
        opts = {}
        self.integrator = ca.integrator("I", solver, dae, 0, tf, opts)

    def integrate(self, x0: np.ndarray, input: np.ndarray, kappa: np.ndarray) -> np.ndarray:
            return np.array(self.integrator(x0 = x0, p = ca.vertcat(input, kappa))['xf'])

    def create_discrete_model(self, dt: float, m_curvature: float, linear_method: int = 0) -> None:
        self.A = np.zeros((self.nx, self.nx))
        self.B = np.zeros((self.nx, self.nu))
        self.C = np.zeros((self.nx, 1))

        self.set_linearize_point_type(linear_method)

        self.A[0, 1] = self.x_lin[2]
        self.A[1, 0] = -1.0 * self.x_lin[2] * m_curvature ** 2
        self.A[1, 2] = -1.0 * m_curvature
        self.A[1, 3] = 1.0

        self.B[2, 0] = 1.0
        self.B[3, 1] = 1.0

        self.A, self.B, self.C = self.discrete_method.discretize(self.A, self.B, self.C)

    def create_noise_model(self, Q: np.ndarray, integral_dt: float) -> None:
        if Q is None:
            Q = np.diag([0.01, 0.01, 0.1, 0.1])
        self.noise_model = GaussianNoiseModel(Q, integral_dt)

    def set_predict_traj(self, x_guess: np.ndarray, pred_traj: np.ndarray) -> np.ndarray:
        x_guess = np.asarray(x_guess).reshape(-1)
        for i in range(len(x_guess) // self.nx):
            lat_error, yaw_error = x_guess[i * self.nx], x_guess[i * self.nx + 1]
            pred_traj[i, 0] -= np.sin(pred_traj[i, 2]) * lat_error
            pred_traj[i, 1] += np.cos(pred_traj[i, 2]) * lat_error
            pred_traj[i, 2] += yaw_error
            pred_traj[i, 3] = x_guess[i * self.nx + 2]
            pred_traj[i, 4] = x_guess[i * self.nx + 3]
        return pred_traj