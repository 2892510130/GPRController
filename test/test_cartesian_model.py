import numpy as np

from model import (
    CartesianModel,
)

def test_cartesian_model_integration():
    model = CartesianModel(integral_dt = 0.1, discrete_method = 'RK4')

    # Casadi integration
    x0 = np.array([0, 0, 0, 0, 0])
    u0 = np.array([0.05, 0.1])
    state = model.integrate(x0, u0)
    print(state.squeeze())

    # Far reference point
    ref_x = np.array([0.1, 0.1, 0.1, 0.01, 0.01])
    ref_u = np.array([0.1, 0.15])
    model.set_reference(ref_x, ref_u)
    model.set_linearize_point_type(0)
    model.create_discrete_model(dt = 0.1, linear_method = 0)
    state_predict_far = model.A @ x0.reshape(5, 1) + model.B @ u0.reshape(2, 1) + model.C
    print(state_predict_far.squeeze())

    # Near reference point
    ref_x = np.array([0, 0, 0, 0, 0])
    ref_u = np.array([0.05, 0.1])
    model.set_reference(ref_x, ref_u)
    model.create_discrete_model(dt = 0.1, linear_method = 0)
    state_predict_near = model.A @ x0.reshape(5, 1) + model.B @ u0.reshape(2, 1) + model.C
    print(state_predict_near.squeeze())

    assert(np.sum(np.power(state - state_predict_far, 2)) > np.sum(np.power(state - state_predict_near, 2)))