import numpy as np

class GaussianNoiseModel:
    def __init__(self, Q: np.ndarray, dt: float):
        self.Q = Q / dt

    def create_noise(self) -> np.ndarray:
        random_vec = np.random.randn(self.Q.shape[0], 1)
        return self.Q @ random_vec