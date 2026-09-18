import yaml
import numpy as np
from typing import Optional, List
from dataclasses import dataclass

@dataclass
class MPCConfig:
    mpcWindow:             int
    linear_method:         int # 0 for reference, 1 for guess from last time step solution shifted
    model:                 str
    mpc_name:              str
    discrete_method:       str
    solver:                str
    control_dt:            float
    mpc_dt:                float
    print_time:            bool
    propagete_cov:         bool
    variance_aware:        bool
    lambda_var:            List
    aware_len:             int

    # cost weights
    Q:                     np.ndarray
    R:                     np.ndarray
    Q_final:               np.ndarray
     
    # constraints     
    max_input:             List
    max_delta_input:       List
     
    # for the gpr data     
    record_data:           bool
    record_data_path:      str

    # for acados mpc, if true, will use discrete linearlized model
    linearized:            Optional[bool] = None
    add_delta_constraints: Optional[bool] = False


    def __post_init__(self):
        # Convert lists to numpy arrays if they aren't already
        if not isinstance(self.Q, np.ndarray):
            self.Q = np.diag(self.Q)
        if not isinstance(self.R, np.ndarray):
            self.R = np.diag(self.R)
        if not isinstance(self.Q_final, np.ndarray):
            self.Q_final = np.diag(self.Q_final)

@dataclass
class GaussianConfig:
    path:                 str
    save_path:            str
    load_path:            str
    total_data_number:    int
    inducing_data_number: int
    input_window:         int
    gpr_method:           int # 0 for zero-order linear (no gradient of gaussian), 1 for first-order linear (with gradient), -1 for no gaussian
    sparse_method:        int # 0, 1 for DTC, FITC, 2, 3 for Variational DTC, FITC

    threshold:            float

    normalize_y:          bool
    lagacy:               bool # whether use lagacy gpr update method
    enable_test:          bool # whether do a test at mpc init
    update_mu_su:         bool
    add_mu_su:            bool
    read_model:           bool

    params:               List

    @classmethod
    def from_yaml(cls, file_path: str):
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)

        return cls(**data)

@dataclass
class SimulationConfig:
    sim_model:          str
      
    SIM_TIMES:          int
    noise_scale:        int
    delay_window:       int
    start_index:        int
      
    tau_scale:          float
    control_dt:         float
    filter_feq_v:       float
    filter_feq_w:       float
    max_lateral_error:  float
    terminal_tolerance: float

    add_noise:          bool
    add_filter:         bool
    add_clamp:          bool
       
    sim_x0:             np.ndarray
    sim_noise:          np.ndarray

    bad_wheel:          bool
    wheel_left:         float
    wheel_right:        float
    L:                  float

    def __post_init__(self):
        # Convert lists to numpy arrays if they aren't already
        if not isinstance(self.sim_x0, np.ndarray):
            self.sim_x0 = np.array(self.sim_x0, dtype=np.float32).reshape(-1, 1)
        if not isinstance(self.sim_noise, np.ndarray):
            self.sim_noise = np.diag(self.sim_noise)

@dataclass
class TrajectoryConfig:
    start_x:    float
    start_y:    float
    start_yaw:  float
    end_x:      float
    end_y:      float
    end_yaw:    float
    offset:     float
    max_vel:    float
    samples:    int

    start_v:    float
    start_a:    float
    end_v:      float
    end_a:      float
    total_time: float
    sample_dt:  float

    def __post_init__(self):
        self.start_yaw = np.radians(self.start_yaw)
        self.end_yaw = np.radians(self.end_yaw)
    
    @classmethod
    def from_yaml(cls, file_path: str):
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)

        return cls(**data)

@dataclass
class SpeedOptimizerConfig:
    w_s:   float # weight on s(t) - s_r(t)
    w_v:   float # weight on v(t) - v_r(t)
    w_a:   float # weight on a(t) - a_r(t)
    w_j:   float # weight on jerk(t)
    w_s_e: float # weight on s_end - s_r_end
    w_v_e: float # weight on v_end - v_r_end
    w_a_e: float # weight on a_end - a_r_end
    w_cur: float # weight on curvature
    v_m:   float # max vel
    a_m:   float # max acc
    j_m:   float # max jerk
    w_m:   float # max anguler speed
    dt:    float # delta t between 2 points, should be the same for all pairs!
    scale: list[float]  # scale between s, v and a to get similar scale

    @classmethod
    def from_yaml(cls, file_path: str):
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)

        return cls(**data)

@dataclass
class Config:
    mpc:      MPCConfig = None
    gaussian: GaussianConfig = None
    traj:     TrajectoryConfig = None
    sim:      SimulationConfig = None

    @classmethod
    def from_yaml(cls, file_path: str):
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)

        _fields = {
            "mpc":      MPCConfig,
            "gaussian": GaussianConfig,
            "traj":     TrajectoryConfig,
            "sim":      SimulationConfig
        }
        
        config_data = {}
        
        for _name, _class in _fields.items():
            if _name in data:
                config_data[_name] = _class(**data[_name])
        
        # Add any top-level fields
        # for key, value in data.items():
        #     if key not in ['mpc', 'gaussian', 'traj']:
        #         config_data[key] = value
                
        return cls(**config_data)
    
    def __post_init__(self):
        if hasattr(self.mpc, "control_dt") and np.abs(self.mpc.control_dt - self.sim.control_dt) > 1e-6:
            raise ValueError("control dt should be same in sim and mpc!")

@dataclass
class TrainConfig:
    plot:              bool
    print:             bool
    write_data:        bool
    optimize:          bool
    normalize_y:       bool
    train_data_number: int
    test_data_number:  int
    input_window:      int
    gap_window:        int
    num_inducing:      int
    method:            str
    train_file_path:   str
    test_file_path:    str
    fixed_data_path:   str

    @classmethod
    def from_yaml(cls, file_path: str):
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)

        return cls(**data)