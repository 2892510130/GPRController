import numpy as np

from model import (
    CartesianGaussianModel,
    AugmentedCartesianGaussianModel,
)

from utils import Config

CONFIG_PATH = "config/mpc/linear_mpc_frenet.yaml"


def _config():
    config = Config.from_yaml(CONFIG_PATH)
    config.gaussian.read_model = False
    return config


def _gp_input(model):
    return np.zeros((1, model.feature_size))


def test_cartesian_gp_model_integration():
    config = _config()
    model = CartesianGaussianModel(integral_dt=0.1, discrete_method='RK4', gaussian_opts=config.gaussian)
    x0 = np.zeros((model.nx, 1))
    u0 = np.array([0.05, 0.1])
    gp_input = _gp_input(model)

    # Far reference point
    ref_x = np.array([0.1, 0.1, 0.1, 0.01, 0.01])
    ref_u = np.array([0.1, 0.15])
    model.set_reference(ref_x, ref_u)
    model.set_linearize_point_type(0)
    model.create_discrete_model(dt=0.1, gpr_input=gp_input, linear_method=0, m_curvature=0.2)
    state_predict_far = model.A @ x0 + model.B @ u0.reshape(2, 1) + model.C
    assert state_predict_far.shape == (model.nx, 1)
    assert np.all(np.isfinite(state_predict_far))

    # Near reference point
    ref_x = np.array([0, 0, 0, 0, 0])
    ref_u = np.array([0.05, 0.1])
    model.set_reference(ref_x, ref_u)
    model.create_discrete_model(dt=0.1, gpr_input=gp_input, linear_method=0, m_curvature=0.2)
    state_predict_near = model.A @ x0 + model.B @ u0.reshape(2, 1) + model.C
    assert state_predict_near.shape == (model.nx, 1)
    assert np.all(np.isfinite(state_predict_near))

    # The predicted trajectory keeps the (N, 5) layout used by the controller.
    pred_traj = model.set_predict_traj(np.zeros((model.nx * 4, 1)), np.zeros((4, 5)))
    assert pred_traj.shape == (4, 5)


def test_cartesian_gp_augment_model_integration():
    config = _config()
    model = AugmentedCartesianGaussianModel(integral_dt=0.1, discrete_method='RK4', gaussian_opts=config.gaussian)
    x0 = np.zeros((model.nx, 1))
    u0 = np.array([0.05, 0.1])
    gp_input = _gp_input(model)

    # Far reference point
    ref_x = np.array([0.1, 0.1, 0.1, 0.01, 0.01])
    ref_u = np.array([0.1, 0.15])
    model.set_reference(ref_x, ref_u)
    model.set_linearize_point_type(0)
    model.create_discrete_model(dt=0.1, gpr_input=gp_input, linear_method=0, m_curvature=0.2)
    state_predict_far = model.A @ x0 + model.B @ u0.reshape(2, 1) + model.C
    assert state_predict_far.shape == (model.nx, 1)
    assert np.all(np.isfinite(state_predict_far))

    # Near reference point
    ref_x = np.array([0, 0, 0, 0, 0])
    ref_u = np.array([0.05, 0.1])
    model.set_reference(ref_x, ref_u)
    model.create_discrete_model(dt=0.1, gpr_input=gp_input, linear_method=0, m_curvature=0.2)
    state_predict_near = model.A @ x0 + model.B @ u0.reshape(2, 1) + model.C
    assert state_predict_near.shape == (model.nx, 1)
    assert np.all(np.isfinite(state_predict_near))

    pred_traj = model.set_predict_traj(np.zeros((model.nx * 4, 1)), np.zeros((4, 5)))
    assert pred_traj.shape == (4, 5)


def test_compare_cartesian_gp_and_augment():
    config = _config()
    u0 = np.array([0.05, 0.1])
    ref_x = np.array([0.1, 0.1, 0.1, 0.01, 0.01])
    ref_u = np.array([0.1, 0.15])

    model = CartesianGaussianModel(integral_dt=0.1, discrete_method='RK4', gaussian_opts=config.gaussian)
    x0 = np.zeros((model.nx, 1))
    model.set_reference(ref_x, ref_u)
    model.set_linearize_point_type(0)
    model.create_discrete_model(dt=0.1, gpr_input=_gp_input(model), linear_method=0, m_curvature=0.2)
    state_predict = model.A @ x0 + model.B @ u0.reshape(2, 1) + model.C

    model_aug = AugmentedCartesianGaussianModel(integral_dt=0.1, discrete_method='RK4', gaussian_opts=config.gaussian)
    x0 = np.zeros((model_aug.nx, 1))
    model_aug.set_reference(ref_x, ref_u)
    model_aug.set_linearize_point_type(0)
    model_aug.create_discrete_model(dt=0.1, gpr_input=_gp_input(model_aug), linear_method=0, m_curvature=0.2)
    state_predict_aug = model_aug.A @ x0 + model_aug.B @ u0.reshape(2, 1) + model_aug.C

    assert np.mean(np.abs(state_predict - state_predict_aug[:5])) < 1e-6
