import numpy as np

from model import (
    FrenetGaussianModel,
    AugmentedFrenetGaussianModel
)

from utils import Config

def test_frenet_gp_model_integration():
    config = Config.from_yaml("config/mpc/linear_frenet_modest.yaml")
    model = FrenetGaussianModel(integral_dt = 0.1, discrete_method = 'RK4', gaussian_opts = config.gaussian)
    x0 = np.array([0, 0, 0, 0])
    u0 = np.array([0.05, 0.1])
    curvature = 0.2
    gp_input = np.zeros((1, 12))

    # Far reference point
    ref_x = np.array([0.1, 0.1, 0.01, 0.01])
    ref_u = np.array([0.1, 0.15])
    model.set_reference(ref_x, ref_u)
    model.set_linearize_point_type(0)
    model.create_discrete_model(dt = 0.1, m_curvature = curvature, gpr_input = gp_input, linear_method = 0)
    state_predict_far = model.A @ x0.reshape(4, 1) + model.B @ u0.reshape(2, 1) + model.C
    print(state_predict_far.squeeze())

    # Near reference point
    ref_x = np.array([0, 0, 0, 0])
    ref_u = np.array([0.05, 0.1])
    model.set_reference(ref_x, ref_u)
    model.create_discrete_model(dt = 0.1, m_curvature = curvature, gpr_input = gp_input, linear_method = 0)
    state_predict_near = model.A @ x0.reshape(4, 1) + model.B @ u0.reshape(2, 1) + model.C
    print(state_predict_near.squeeze())

def test_frenet_gp_augmented_model_integration():
    config = Config.from_yaml("config/mpc/linear_frenet_modest.yaml")
    model = AugmentedFrenetGaussianModel(integral_dt = 0.1, discrete_method = 'RK4', gaussian_opts = config.gaussian)
    x0 = np.zeros((12, 1))
    u0 = np.array([0.05, 0.1])
    curvature = 0.2
    gp_input = np.zeros((1, 12))

    # Far reference point
    ref_x = np.array([0.1, 0.1, 0.01, 0.01])
    ref_u = np.array([0.1, 0.15])
    model.set_reference(ref_x, ref_u)
    model.set_linearize_point_type(0)
    model.create_discrete_model(dt = 0.1, m_curvature = curvature, gpr_input = gp_input, linear_method = 0)
    state_predict_far = model.A @ x0.reshape(12, 1) + model.B @ u0.reshape(2, 1) + model.C
    print(state_predict_far.squeeze())

    # Near reference point
    ref_x = np.array([0, 0, 0, 0])
    ref_u = np.array([0.05, 0.1])
    model.set_reference(ref_x, ref_u)
    model.create_discrete_model(dt = 0.1, m_curvature = curvature, gpr_input = gp_input, linear_method = 0)
    state_predict_near = model.A @ x0.reshape(12, 1) + model.B @ u0.reshape(2, 1) + model.C
    print(state_predict_near.squeeze())

def test_compare_frenet_gp_and_augment():
    config = Config.from_yaml("config/mpc/linear_frenet_modest.yaml")
    u0 = np.array([0.05, 0.1])
    gp_input = np.zeros((1, 12))
    ref_x = np.array([0.1, 0.1, 0.01, 0.01])
    ref_u = np.array([0.1, 0.15])
    curvature = 0.2

    model = FrenetGaussianModel(integral_dt = 0.1, discrete_method = 'RK4', gaussian_opts = config.gaussian)
    x0 = np.array([0, 0, 0, 0])
    model.set_reference(ref_x, ref_u)
    model.set_linearize_point_type(0)
    model.create_discrete_model(dt = 0.1, m_curvature = curvature, gpr_input = gp_input, linear_method = 0)
    state_predict = model.A @ x0.reshape(4, 1) + model.B @ u0.reshape(2, 1) + model.C

    model_aug = AugmentedFrenetGaussianModel(integral_dt = 0.1, discrete_method = 'RK4', gaussian_opts = config.gaussian)
    x0 = np.zeros((12, 1))
    model_aug.set_reference(ref_x, ref_u)
    model_aug.set_linearize_point_type(0)
    model_aug.create_discrete_model(dt = 0.1, m_curvature = curvature, gpr_input = gp_input, linear_method = 0)
    state_predict_aug = model_aug.A @ x0.reshape(12, 1) + model_aug.B @ u0.reshape(2, 1) + model_aug.C

    assert(np.mean(np.abs(state_predict - state_predict_aug[:4])) < 1e-6)