import re
import numpy as np
import matplotlib.pyplot as plt

from utils import Config
from model.gp_model import GaussianModel

def plot_delta_state(lines):
    delta_state_nominal, delta_state = [], []

    for line in lines:
        if re.search(r"Delta State Nominal", line):
            result = re.findall(r'-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?', line)
            delta_state_nominal.append([float(x) for x in result])
        if re.search(r"Delta State GPR", line):
            result = re.findall(r'-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?', line)
            delta_state.append([float(x) for x in result])

    delta_state_nominal = np.array(delta_state_nominal)
    delta_state = np.array(delta_state)

    mae_nom_dv, mae_nom_dw = np.mean(np.abs(delta_state_nominal[:, 0])), np.mean(np.abs(delta_state_nominal[:, 1]))
    mae_gpr_dv, mae_gpr_dw = np.mean(np.abs(delta_state[:, 0])), np.mean(np.abs(delta_state[:, 1]))
    increase_dv, increase_dw = (mae_nom_dv - mae_gpr_dv) / mae_nom_dv * 100, (mae_nom_dw - mae_gpr_dw) / mae_nom_dw * 100
    print(f"Mean MAE nom dv and dw:          {np.mean(np.abs(delta_state_nominal[:, 0])):+.6f}, {np.mean(np.abs(delta_state_nominal[:, 1])):+.6f} |")
    print(f"Mean MAE gpr dv and dw:          {np.mean(np.abs(delta_state[:, 0])):+.6f}, {np.mean(np.abs(delta_state[:, 1])):+.6f} |")
    print(f"Mean MAE inc dv and dw:          {increase_dv:+06.2f}%,   {increase_dw:+06.2f}%   |")
    
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(delta_state_nominal[:, 0], label = "nominal dv")
    axs[0].plot(delta_state[:, 0], label = "gpr dv")
    axs[0].set_ylim(-2, 2)
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(delta_state_nominal[:, 1], label = "nominal dw")
    axs[1].plot(delta_state[:, 1], label = "gpr dw")
    axs[1].set_ylim(-2, 2)
    axs[1].legend()
    axs[1].grid()

    # plt.show()

    return increase_dv, increase_dw

def plot_prediction_offline(lines):
    delta_state_nominal, delta_state, gpr_input = [], [], []

    for line in lines:
        if re.search(r"Delta State Nominal", line):
            result = re.findall(r'-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?', line)
            delta_state_nominal.append([float(x) for x in result])
        if re.search(r"Delta State GPR", line):
            result = re.findall(r'-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?', line)
            delta_state.append([float(x) for x in result])
        if re.search(r"GPR Input", line):
            result = re.findall(r'-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?', line)
            gpr_input.append([float(x) for x in result])
    
    delta_state_nominal = np.array(delta_state_nominal)
    delta_state = np.array(delta_state)
    gpr_input = np.array(gpr_input)

    config_dir = "config/mpc/linear_mpc_frenet.yaml"
    config = Config.from_yaml(config_dir)
    gp_config = config.gaussian
    gp_config.read_model = False

    gp = GaussianModel(gp_config)

    predict_v, predict_w, var_vs, var_ws = [], [], [], []

    for data in gpr_input:
        mu_v, var_v = gp.gps[0].predict_noiseless(data, full_cov = True)
        mu_w, var_w = gp.gps[1].predict_noiseless(data, full_cov = True)
        var_vs.append(var_v ** 0.5 * 2)
        var_ws.append(var_w ** 0.5 * 2)
        predict_v.append(mu_v)
        predict_w.append(mu_w)
    
    var_vs = np.array(var_vs).squeeze()
    var_ws = np.array(var_ws).squeeze()
    predict_v = np.array(predict_v).squeeze()
    predict_w = np.array(predict_w).squeeze()

    error_v = np.mean(np.abs(delta_state_nominal[:, 0] - predict_v))
    error_w = np.mean(np.abs(delta_state_nominal[:, 1] - predict_w))
    error_v2 = np.mean(np.abs(delta_state[:, 0] - predict_v))
    error_w2 = np.mean(np.abs(delta_state[:, 1] - predict_w))

    print(f"Error between gpr update and nominal: {error_v:+.6f} {error_w:+.6f}")
    print(f"Error between gpr update and gpr:     {error_v2:+.6f} {error_w2:+.6f}")

    x_axis = np.arange(0, len(predict_v))
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(delta_state_nominal[:, 0], label = "nominal dv")
    axs[0].plot(predict_v, label = "gpr dv")
    axs[0].fill_between(x_axis, 
        (predict_v - 2 * var_vs), 
        (predict_v + 2 * var_vs), 
        color='red', alpha=0.2, label='Confidence Interval')
    axs[0].set_ylim(-2, 2)
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(delta_state_nominal[:, 1], label = "nominal dw")
    axs[1].plot(predict_w, label = "gpr dw")
    axs[1].fill_between(x_axis, 
        (predict_w - 2 * var_ws), 
        (predict_w + 2 * var_ws), 
        color='red', alpha=0.2, label='Confidence Interval')
    axs[1].set_ylim(-2, 2)
    axs[1].legend()
    axs[1].grid()

    plt.show()