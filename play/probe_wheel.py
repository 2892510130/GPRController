"""Probe: how the nominal controller's error scales with the wheel-efficiency loss."""

import numpy as np

from controller import MPCList
from play.mpc_closed_loop import SimulationRunner
from trajectory import Trajectory
from utils import Config

GP0 = [6.31033195e-27, 7.102742367, 0.501616042, 5.12330577, 6.276647649,
       6.870772865, 3.988981479, 4.597690728, 8.372639431, 6.085131514,
       0.4243633695, 3.393323193, 9.385186209, 9.149415148]
GP1 = [1.579816364e-33, 5.582491513, 3.140257759, 0.6423122812, 10.13865966,
       4.127419998, 4.403624426, 4.131115671, 9.401297721, 9.385516953,
       2.792767986, 0.4882394274, 9.266471496, 9.70348758]

TRACK = Trajectory("config/track/full_track_opt.yaml")

for wheel_left in (0.975, 0.95, 0.925, 0.90):
    cfg = Config.from_yaml("config/mpc/linear_mpc_frenet.yaml")
    cfg.sim.delay_window = 1
    cfg.sim.bad_wheel = True
    cfg.sim.wheel_left = wheel_left
    cfg.sim.wheel_right = 1.0
    cfg.mpc.R = np.diag([0.1, 0.1])
    cfg.mpc.max_input = [1.0, 1.5]
    cfg.mpc.record_data = False
    cfg.mpc.print_time = False
    cfg.mpc.log_path = f"log/sweep/probe_{wheel_left}.txt"
    cfg.gaussian.gpr_method = -1
    cfg.gaussian.path = "log/processed_data/bad_wheel.txt"
    cfg.gaussian.total_data_number = 394
    cfg.gaussian.params = [GP0, GP1]
    cfg.gaussian.read_model = False
    np.random.seed(0)

    mpc = MPCList[cfg.mpc.mpc_name](cfg, TRACK.path, TRACK.speed_profile)
    runner = SimulationRunner(mpc, cfg)
    runner.run_full_simulation()
    mpc.logger.close_log()

    err = np.abs(np.asarray(runner.state, dtype=float))
    print(
        f"wheel_left={wheel_left:.3f}  mean|lat|={err[:,0].mean():.5f}  "
        f"max|lat|={err[:,0].max():.5f}  mean|head|={err[:,1].mean():.5f}  "
        f"max|head|={err[:,1].max():.5f}  steps={err.shape[0]}"
    )