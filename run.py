from play import test_mpc_closed_loop, test_mpc, plot_prediction_offline, plot_delta_state
from train import test_train_gp
from utils import Logger

if __name__ == "__main__":
    # test_train_gp()
    
    config = None
    log = Logger("log/update.txt")
    for i in range(10):
        config = test_mpc_closed_loop(visualize = 0, config = config)
        if i == 0: config.gaussian.read_model = True
        with open("log/linear_mpc_log.txt") as f:
            inc_dv, inc_dw = plot_delta_state(f)
            log.info("Update Result] inc_dv, inc_dw: ", [inc_dv, inc_dw])
    log.close_log()