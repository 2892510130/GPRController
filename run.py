from play import test_mpc_closed_loop, plot_delta_state
from utils.log import Logger

if __name__ == "__main__":
    # Training can be run separately via train.test_train_gp().
    config = None
    log = Logger("log/update.txt")
    for i in range(10):
        config = test_mpc_closed_loop(visualize=0, config=config)
        if i == 0:
            config.gaussian.read_model = True
        with open(config.mpc.log_path) as f:
            inc_dv, inc_dw = plot_delta_state(f)
        log.info("Update Result: ", [inc_dv, inc_dw])
    log.close_log()
