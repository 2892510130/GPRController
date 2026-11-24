import numpy as np


class GaussianNoiseModel:
    """Process noise sampled from a continuous-time covariance rate.

    ``Q`` is the continuous-time covariance rate (variance per second) and
    ``dt`` is the integration step, so one step adds noise with covariance
    ``Q * dt`` (standard deviation ``sqrt(Q * dt)``).
    """

    def __init__(self, Q: np.ndarray, dt: float):
        if dt <= 0.0:
            raise ValueError(f"GaussianNoiseModel: dt must be positive, got {dt}")
        Q = np.atleast_2d(np.asarray(Q, dtype=float))
        if Q.shape[0] != Q.shape[1]:
            raise ValueError(f"GaussianNoiseModel: Q must be square, got shape {Q.shape}")

        covariance = 0.5 * (Q + Q.T) * dt
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        if np.any(eigenvalues < -1e-12):
            raise ValueError("GaussianNoiseModel: Q must be positive semi-definite")

        self.covariance = covariance
        self.sqrt_covariance = eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))

    def create_noise(self) -> np.ndarray:
        return self.sqrt_covariance @ np.random.randn(self.sqrt_covariance.shape[0], 1)
