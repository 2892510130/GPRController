from play import test_mpc_closed_loop, test_mpc, plot_prediction_offline, plot_delta_state
from train import test_train_gp

if __name__ == "__main__":
    # test_train_gp()

    test_mpc_closed_loop(visualize = 0)
    with open("log/linear_mpc_log.txt") as f:
        plot_delta_state(f)