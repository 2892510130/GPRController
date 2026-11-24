## How to run?

### Create a trajectory
- Using `BezierPath` to generate a bezier path, it will create a `QuinticPolynomial` for it.
- Then using `NonlinearTrajectoryOptimizer` to optimize the trajectory, see `test_nonliner_speed_optimize`.
- Or you can just use pre-generated trajectory like `full_track_opt.yaml` using `Trajectory`.

### Run with default model/controller
- Set `gpr_method = -1` in your mpc config file, then run `test_mpc_closed_loop`.
- This will generate a log file in `log/gpr_log/`.

### Train the Gaussian process model
- Modify the `train.yaml`, then run `test_train_gp`.

### Run with the advanced model/controller
- Modify the mpc config, the gaussian part (`path`, `total_data_number` and more)
  - `gpr_method` with 0 is zero-order GP controller, with 1 is first-order GP controller
- With `add_mu_su: true` new inducing points are added online through the fast `mu_su`
  path. That path only keeps the mean/covariance prediction state up to date; the
  Cholesky-based predictor is marked as stale until `sync_cholesky()` is called
  explicitly. `predict_noiseless()` (`mu_su`) is the intended online predictor.
