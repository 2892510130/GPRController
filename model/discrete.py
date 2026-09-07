import numpy as np

from abc import abstractmethod
from typing import Tuple

class DiscreteBase:
    def __init__(self, dt: float, n_steps: int = 1) -> None:
        self.dt = dt
        self.n_steps = n_steps

    @abstractmethod
    def discretize(self, A: np.ndarray, B: np.ndarray, C: np.ndarray):
        pass
    
class EulerDiscrete(DiscreteBase):
    def __init__(self, dt: float) -> None:
        super().__init__(dt)

    def discretize(self, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> Tuple[np.ndarray, ...]:
        nx, nu = B.shape
        identity = np.eye(nx)

        A = identity + A * self.dt
        B = B * self.dt
        C = C * self.dt
        return A, B, C

class CrankNicolsonDiscrete(DiscreteBase):
    def __init__(self, dt: float) -> None:
        super().__init__(dt)

    def discretize(self, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> Tuple[np.ndarray, ...]:
        nx, nu = B.shape
        identity = np.eye(nx)

        temp = np.linalg.inv(identity - A * self.dt * 0.5)
        A = temp @ (identity + A * self.dt * 0.5)
        B = temp @ B * self.dt
        C = temp @ C * self.dt
        return A, B, C


class RK4Discrete(DiscreteBase):
    """
    x_dot = A @ x + B @ u + C
    k1 = x_dot
    k2 = A @ (x + dt / 2 * k1) + B @ u + C
    k3 = A @ (x + dt / 2 * k2) + B @ u + C
    k4 = A @ (x + dt * k3) + B @ u + C
    x_next = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    """
    def __init__(self, dt: float) -> None:
        super().__init__(dt)

    def discretize(self, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> Tuple[np.ndarray, ...]:
        nx, nu = B.shape
        identity = np.eye(nx)

        t1 = A * self.dt / 2 + identity
        t2 = A * self.dt / 2 @ t1 + identity
        t3 = A * self.dt @ t2 + identity

        M = self.dt / 6 * (identity + 2 * t1 + 2 * t2 + t3)

        A = identity + M @ A
        B = M @ B
        C = M @ C
        return A, B, C

DiscreteMethods = {
    'Euler': EulerDiscrete, 
    'CrankNicolson': CrankNicolsonDiscrete, 
    'RK4': RK4Discrete
}