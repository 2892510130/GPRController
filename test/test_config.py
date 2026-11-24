import pytest

from utils import Config
from utils.config import MPCConfig, SimulationConfig


def _mpc_kwargs(**overrides):
    kwargs = dict(
        mpcWindow=3,
        linear_method=0,
        model="Kinematic Frenet",
        mpc_name="LinearMPC",
        discrete_method="Euler",
        solver="osqp",
        control_dt=0.05,
        mpc_dt=0.05,
        print_time=False,
        propagete_cov=False,
        variance_aware=False,
        lambda_var=[1.0, 1.0],
        aware_len=3,
        Q=[1.0, 1.0, 1.0, 1.0],
        R=[1.0, 1.0],
        Q_final=[1.0, 1.0, 1.0, 1.0],
        max_input=[1.5, 1.5],
        max_delta_input=[5.0, 5.0],
        record_data=False,
        record_data_path="log/gpr_log/test.txt",
    )
    kwargs.update(overrides)
    return kwargs


def _sim_kwargs(**overrides):
    kwargs = dict(
        sim_model="Kinematic Cartesian",
        SIM_TIMES=10,
        noise_scale=1,
        delay_window=1,
        start_index=0,
        tau_scale=1.0,
        control_dt=0.05,
        filter_feq_v=5.0,
        filter_feq_w=5.0,
        max_lateral_error=1.0,
        terminal_tolerance=0.1,
        add_noise=False,
        add_filter=False,
        add_clamp=False,
        sim_x0=[0, 0, 0, 0, 0],
        sim_noise=[0, 0, 0, 0, 0],
        bad_wheel=False,
        wheel_left=1.0,
        wheel_right=1.0,
        L=0.5,
    )
    kwargs.update(overrides)
    return kwargs


def test_config_allows_missing_sections():
    config = Config(mpc=MPCConfig(**_mpc_kwargs()))
    assert config.mpc is not None
    assert config.sim is None
    assert config.gaussian is None

    config = Config(sim=SimulationConfig(**_sim_kwargs()))
    assert config.mpc is None
    assert config.sim is not None


def test_config_matching_control_dt_is_accepted():
    config = Config(
        mpc=MPCConfig(**_mpc_kwargs(control_dt=0.05)),
        sim=SimulationConfig(**_sim_kwargs(control_dt=0.05)),
    )
    assert config.mpc.control_dt == pytest.approx(config.sim.control_dt)


def test_config_rejects_control_dt_mismatch():
    with pytest.raises(ValueError, match="control dt"):
        Config(
            mpc=MPCConfig(**_mpc_kwargs(control_dt=0.05)),
            sim=SimulationConfig(**_sim_kwargs(control_dt=0.06)),
        )
