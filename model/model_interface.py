import numpy as np

from abc import abstractmethod

class ModelInterface:
    # set reference data
    def set_reference(self, x_ref: np.ndarray, u_ref: np.ndarray) -> None:
        self.x_ref = x_ref
        self.u_ref = u_ref

    # set guess data
    def set_guess(self, x_guess: np.ndarray, u_guess: np.ndarray) -> None:
        self.x_guess = x_guess
        self.u_guess = u_guess

    # linearize around reference data or guess data
    def set_linearize_point_type(self, linear_method: int) -> None:
        if linear_method == 0:
            self.x_lin = self.x_ref
            self.u_lin = self.u_ref
        elif linear_method == 1:
            self.x_lin = self.x_guess
            self.u_lin = self.u_guess

    @abstractmethod
    def create_discrete_model(self, dt: float, linear_method: int = 0) -> None:
        pass