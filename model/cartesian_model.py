import numpy as np
import casadi as ca

from . import DiscreteMethods
from .noise_model import GaussianNoiseModel
from .model_interface import ModelInterface

class CartesianModel(ModelInterface):
    def __init__(
        self,
        integral_dt: int = 0.1,
        discrete_method: str = 'Euler',
        Q: np.ndarray | None = None,
        **kwargs
    ) -> None:
        self.name = 'Kinematic Cartesian'
        self.nx = 5
        self.nu = 2
        self.np = self.nx * (self.nx + self.nu + 1)
        self.LINV = 3
        self.YAWV = 4
        self.discrete_method = DiscreteMethods[discrete_method](integral_dt)
        self.create_model()
        self.create_integrator(integral_dt)
        self.create_noise_model(Q, integral_dt)

    def create_model(self) -> None:
        self.x = ca.MX.sym('x', self.nx, 1)
        self.u = ca.MX.sym('u', self.nu, 1)
        self.p = ca.MX.sym('p', 0, 0)

        self.x_dot = ca.vertcat(
            ca.cos(self.x[2]) * self.x[3],
            ca.sin(self.x[2]) * self.x[3],
            self.x[4],
            self.u[0],
            self.u[1]
        )

    def create_integrator(self, tf: float, solver: str = 'rk') -> None:
        dae = {'x': self.x, 'u': self.u, 'ode': self.x_dot}
        self.integrator = ca.integrator('I', solver, dae, 0, tf, {})

    def integrate(self, x0: np.ndarray, input: np.ndarray) -> np.ndarray:
        return np.array(self.integrator(x0 = x0, u = input)['xf'])

    def create_discrete_model(self, dt: float, linear_method: int = 0) -> None:
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

        self.A, self.B, self.C = self.discrete_method.discretize(self.A, self.B, self.C)
    
    def create_noise_model(self, Q: np.ndarray, integral_dt: float) -> None:
        if Q is None:
            Q = np.diag([0.01, 0.01, 0.01, 0.1, 0.1])
        self.noise_model = GaussianNoiseModel(Q, integral_dt)

    def set_predict_traj(self, x_guess: np.ndarray, pred_traj: np.ndarray) -> np.ndarray:
        # x_guess is the flattened predicted state sequence (nx * N); the first
        # five entries of every state are [x, y, yaw, v, w].
        return np.asarray(x_guess).reshape(-1, self.nx)[:, :5].copy()

class CartesianDelayModel(CartesianModel):
    def __init__(
        self,
        integral_dt: float = 0.1,
        tau_v: float = 0.1,
        tau_w: float = 0.1,
        discrete_method:str = 'Euler',
        Q: np.ndarray | None = None,
        **kwargs
    ) -> None:
        self.tau_v = tau_v
        self.tau_w = tau_w
        self.LINV = 3
        self.YAWV = 4
        self.discrete_method = DiscreteMethods[discrete_method](integral_dt)
        self.name = 'Kinematic Cartesian Delay'
        self.nx = 5
        self.nu = 2
        self.np = self.nx * (self.nx + self.nu + 1)
        self.create_model()
        self.create_integrator(integral_dt)
        self.create_noise_model(Q, integral_dt)
    
    def create_model(self) -> None:
        self.u = ca.MX.sym("u", self.nu, 1)
        self.x = ca.MX.sym("x", self.nx, 1)

        self.x_dot = ca.vertcat(
            ca.cos(self.x[2]) * self.x[3],
            ca.sin(self.x[2]) * self.x[3],
            self.x[4],
            (self.u[0] - self.x[3]) / self.tau_v,
            (self.u[1] - self.x[4]) / self.tau_w
        )

    def create_discrete_model(self, dt: float, linear_method: int = 0) -> None:
        self.A = np.zeros((self.nx, self.nx))
        self.B = np.zeros((self.nx, self.nu))
        self.C = np.zeros((self.nx, 1))

        self.set_linearize_point_type(linear_method)

        self.A[0, 2] = -1.0 * self.x_lin[3] * np.sin(self.x_lin[2])
        self.A[0, 3] = np.cos(self.x_lin[2])
        self.A[1, 2] = self.x_lin[3] * np.cos(self.x_lin[2])
        self.A[1, 3] = np.sin(self.x_lin[2])
        self.A[2, 4] = 1.0
        self.A[3, 3] = -1.0 / self.tau_v
        self.A[4, 4] = -1.0 / self.tau_w

        self.B[3, 0] = 1.0 / self.tau_v
        self.B[4, 1] = 1.0 / self.tau_w

        self.C[0, 0] = -1.0 * self.A[0, 2] * self.x_lin[2]
        self.C[1, 0] = -1.0 * self.A[1, 2] * self.x_lin[2]

        self.A, self.B, self.C = self.discrete_method.discretize(self.A, self.B, self.C)