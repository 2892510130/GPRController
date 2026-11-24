import numpy as np
import pytest

from model.noise_model import GaussianNoiseModel


def test_noise_covariance_is_q_times_dt():
    Q = np.diag([4e-4, 9e-4])
    dt = 0.5
    model = GaussianNoiseModel(Q, dt)

    np.random.seed(0)
    samples = np.hstack([model.create_noise() for _ in range(20000)])
    assert samples.shape == (2, 20000)

    empirical = np.cov(samples)
    np.testing.assert_allclose(np.diag(empirical), np.diag(Q * dt), rtol=0.1)
    off_diagonal = empirical - np.diag(np.diag(empirical))
    assert np.max(np.abs(off_diagonal)) < 5e-5


def test_zero_q_returns_zero_noise():
    model = GaussianNoiseModel(np.zeros((2, 2)), 0.1)
    np.testing.assert_array_equal(model.create_noise(), np.zeros((2, 1)))


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        GaussianNoiseModel(np.eye(2), 0.0)
    with pytest.raises(ValueError):
        GaussianNoiseModel(np.diag([1.0, -1.0]), 0.1)
    with pytest.raises(ValueError):
        GaussianNoiseModel(np.ones((2, 3)), 0.1)
