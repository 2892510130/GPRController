import os
import numpy as np
import matplotlib.pyplot as plt
from model import ModelLists
from collections import deque
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle, Ellipse
from utils.filter import LPFilter
from utils import Config, TrajectoryConfig
from controller import MPCList
from trajectory import BezierPath, RacePath, Trajectory

class MPCPlotter:
    """Handles all plotting functionality for MPC simulation"""
    
    def __init__(self, bezier_path):
        self.bezier_path = bezier_path
        self.x_min = self.bezier_path.path[:, 0].min() - 0.25
        self.x_max = self.bezier_path.path[:, 0].max() + 0.25
        self.y_min = self.bezier_path.path[:, 1].min() - 0.25
        self.y_max = self.bezier_path.path[:, 1].max() + 0.25
        self.square_height = 0.08
        self.square_width = 0.06
        self.car = Rectangle(
            (0, 0), self.square_height, self.square_width, 
            angle=0, fc='yellow', ec='black', label='Car'
        )
        
    def setup_animation_plot(self):
        """Initialize the real-time animation plot"""
        self.fig_visual, self.ax_visual = plt.subplots()
        self.fig_control, self.ax_control = plt.subplots()
        self.fig_state, self.ax_state = plt.subplots()
    
    def update_animation(self, simulation_runner):
        """Update the animation with new simulation data"""
        state, control = simulation_runner.state, simulation_runner.control
        pred_traj, xr, nx, x_guess = simulation_runner.mpc.pred_traj, simulation_runner.mpc.ref_for_visual, simulation_runner.mpc.model.nx, simulation_runner.mpc.x_guess.reshape(-1, simulation_runner.mpc.model.nx)

        x, y, theta = state[0, 0], state[0, 1], state[0, 2]
        lb_x = x - self.square_height * np.cos(theta) / 2 + self.square_width * np.sin(theta) / 2
        lb_y = y - self.square_height * np.sin(theta) / 2 - self.square_width * np.cos(theta) / 2
        self.car.set_xy((lb_x, lb_y))
        self.car.angle = np.degrees(theta)

        # Clear previous plots
        self.ax_visual.cla()
        self.ax_control.cla()
        self.ax_state.cla()

        # Plot the reference path and current state
        self.ax_visual.set_xlim(self.x_min, self.x_max)
        self.ax_visual.set_ylim(self.y_min, self.y_max)
        self.ax_visual.plot(self.bezier_path.path[:, 0], self.bezier_path.path[:, 1], label='ref')
        self.ax_visual.scatter(self.bezier_path.path[:, 0], self.bezier_path.path[:, 1], s=10)
        self.ax_visual.plot(state[:, 0], state[:, 1], label="real_traj", linewidth = 3)
        self.ax_visual.plot(pred_traj[:, 0], pred_traj[:, 1], label="pred_traj", linewidth = 3)
        self.ax_visual.plot(xr[:, 0], xr[:, 1], label="ref_traj", linewidth = 3)
        self.ax_visual.add_patch(self.car)

        # Plot the prediction of the state
        self.ax_state.plot(pred_traj[:, 0], label="pred_state_0")
        self.ax_state.plot(pred_traj[:, 1], label="pred_state_1")
        self.ax_state.plot(state[:, 0], label="state_0")
        self.ax_state.plot(state[:, 1], label="tate_1")
        self.ax_state.legend()

        if len(simulation_runner.mpc.x_var_list) > 0 and simulation_runner.mpc.opts.propagete_cov:
            var_list = simulation_runner.mpc.x_var_list
            for i in range(simulation_runner.mpc.opts.mpcWindow):
                var = var_list[i]
                r = 2 * np.sqrt(var[0, 0] + var[1, 1]) * 10 # average std on x and y -> (std_x^2 + std_y^2) ^ 0.5
                circle = Circle((pred_traj[i, 0], pred_traj[i, 1]), radius=r, edgecolor='blue', facecolor='none', linewidth=2, alpha=0.6)
                
                # seperate the x and y for plot, the visual is not good as one std is small and another is large
                # x, y = np.sqrt(var[0, 0]), np.sqrt(var[1, 1])
                # circle = Ellipse((x_guess[i*nx], x_guess[i*nx+1]), width=x, height=y, edgecolor='blue', facecolor='none', linewidth=2, alpha=0.6)
                self.ax_visual.add_patch(circle)

        self.ax_visual.legend()
        self.ax_visual.set_aspect('equal')
        self.ax_control.plot(control[:, 0], label='contorl_dv')
        self.ax_control.plot(control[:, 1], label='contorl_dw')
        self.ax_control.plot(state[:, 3], label='v')
        self.ax_control.plot(state[:, 4], label='w')
        self.ax_control.plot(pred_traj[:, 3], label='pred_v')
        self.ax_control.plot(pred_traj[:, 4], label='pred_w')
        self.ax_control.legend()

        self.fig_control.canvas.draw()
        self.fig_control.canvas.flush_events()
        self.fig_visual.canvas.draw()
        self.fig_visual.canvas.flush_events()
        self.fig_state.canvas.draw()
        self.fig_state.canvas.flush_events()
        plt.pause(0.05)
    
    def create_analysis_plots(self, time, sim_path, ref, control, state, mpc_model):
        """Create post-simulation analysis plots"""
    
    def print_performance_summary(self, simulation_runner):
        """Print simulation performance metrics"""

        print("\n" + "="*55)
        print("SIMULATION PERFORMANCE SUMMARY")
        delta_v = np.array(simulation_runner.delta_v).squeeze()
        delta_w = np.array(simulation_runner.delta_w).squeeze()
        error = np.array(simulation_runner.error).reshape(-1, 2)
        print("Mean delta v: ", f"{np.mean(np.abs(delta_v)):.6f}")
        print("Mean delta w: ", f"{np.mean(np.abs(delta_w)):.6f}")
        print("Mean delta v at k = 0: ", f"{np.mean(np.abs(delta_v[:, 0])):.6f}")
        print("Mean delta w at k = 0: ", f"{np.mean(np.abs(delta_w[:, 0])):.6f}")
        print("Mean lateral error: ", f"{np.mean(np.abs(error[:, 0])):.6f}")
        print("Mean heading error: ", f"{np.mean(np.abs(error[:, 1])):.6f}")
        print("Mean delta x0: ", f"{np.mean(np.abs(simulation_runner.delta_x0)):.6f}")
        print("Mean delta x1: ", f"{np.mean(np.abs(simulation_runner.delta_x1)):.6f}")
        print("="*55)

    def show(self):
        plt.show()

class SimulationRunner:
    """Handles the core simulation logic and configuration"""
    
    def __init__(self, mpc, config):
        self.mpc = mpc
        self.config = config
        self.sim_config = config.sim
        self.nx, self.nu = mpc.model.nx, mpc.model.nu
        
        # Initialize simulation parameters
        self.dt = self.sim_config.control_dt
        self.nu = mpc.model.nu

        # Create simulation model with all parameters
        sim_noise = self.sim_config.sim_noise * self.sim_config.noise_scale
        tau_scale = self.sim_config.tau_scale
        tau_v, tau_w = tau_scale * self.dt, tau_scale * self.dt
        
        self.sim_model = ModelLists[self.sim_config.sim_model](
            integral_dt=self.dt, 
            tau_v=tau_v, 
            tau_w=tau_w, 
            Q=sim_noise
        )
        self.sim_x0 = self.sim_config.sim_x0[:self.sim_model.nx]
        start_index = self.sim_config.start_index
        if start_index != 0:
            self.sim_x0[0, 0], self.sim_x0[1, 0], self.sim_x0[2, 0], self.sim_x0[3, 0], self.sim_x0[4, 0] = self.mpc.path[start_index, 0], self.mpc.path[start_index, 1], self.mpc.path[start_index, 2], self.mpc.speed_profile[start_index, 2], self.mpc.speed_profile[start_index, 4]

        
        if "Delay" in self.sim_model.name: 
            print("Using delay model")
        
        # Initialize filters
        self.lp_v = LPFilter(self.sim_config.filter_feq_v, self.dt)
        self.lp_w = LPFilter(self.sim_config.filter_feq_w, self.dt)
        
        # Initialize control buffer
        delay_window = self.sim_config.delay_window
        self.control_buffer = deque(
            [np.array([0.0, 0.0]) for _ in range(delay_window)], 
            maxlen=delay_window
        )
        
        # Initialize simulation data storage
        self.delta_v, self.delta_w, self.delta_x0, self.delta_x1, self.error = [], [], [], [], []
        self.state, self.control = None, None

        self.max_lateral_error = self.sim_config.max_lateral_error
        self.terminal_tolerance = self.sim_config.terminal_tolerance
        self.simulation_finished = False
        self.termination_reason = None

        self.user_input = 1
        self.index = 0
        self.pre_run_steps = 10000
        self.interrupt_mode = False

    def check_termination_conditions(self, current_state):
        """Check if simulation should terminate"""
        # Check if reached end of path
        if self.mpc.index >= len(self.mpc.path) - 1:
            x, y = current_state[0, 0], current_state[1, 0]
            distance_to_end = np.hypot(x - self.mpc.path[-1, 0], y - self.mpc.path[-1, 1])
            if distance_to_end < self.terminal_tolerance:
                self.simulation_finished = True
                self.termination_reason = "Reached end of path"
                return True
        
        # Check lateral error (divergence)
        if len(self.state) > 0:  # Make sure we have at least one state recorded
            if self.mpc.lat_error > self.max_lateral_error:
                self.simulation_finished = True
                self.termination_reason = f"Diverged too far (lat_error: {float(self.mpc.lat_error):.3f})"
                return True
        
        return False
    
    def bad_wheel_simulation(self, control):
        tmp1 = self.sim_config.wheel_right - self.sim_config.wheel_left
        tmp2 = self.sim_config.wheel_right + self.sim_config.wheel_left
        v = (2 * control[0] * tmp2 + self.sim_config.L * control[1] * tmp1) / 4
        w = (2 * control[0] * tmp1 + self.sim_config.L * control[1] * tmp2) / 2 / self.sim_config.L
        control[0] = v
        control[1] = w
    
    def run_step(self):
        """Execute one simulation step - even cleaner interface!"""
        if self.index > self.pre_run_steps and self.interrupt_mode is False:
            self.interrupt_mode = True

        if self.index != 0 and self.interrupt_mode:
            self.user_input = int(input())

        if self.simulation_finished or self.user_input != 1 or self.index > self.sim_config.SIM_TIMES:
            return False
        
        # Solve MPC
        save_npy = False
        self.mpc.initialize(self.sim_x0[:5], self.dt)
        if save_npy:
            f_npy = open("log/test.npy", 'wb')
            np.save(f_npy, self.mpc.cur_state)
            np.save(f_npy, self.mpc.x0)
            np.save(f_npy, self.mpc.x_guess)
            np.save(f_npy, self.mpc.u_guess)
            np.save(f_npy, np.array(self.mpc.gpr_list))
            np.save(f_npy, np.array(self.control_buffer))
            np.save(f_npy, self.mpc.pred_traj)
        # print("-" * 50) # I use this find an error
        # print(self.mpc.x0.squeeze())
        # print(self.mpc.x_guess[:self.mpc.model.nx].squeeze())
        # print(self.mpc.x0.squeeze() - self.mpc.x_guess[:self.mpc.model.nx].squeeze())
        self.mpc.solve()

        control_buffer_tmp = self.control_buffer.copy()
        x0_tmp = self.sim_x0.copy()
        self.state = []
        self.control = []
        self.index += 1

        for i in range(self.config.mpc.mpcWindow):
            sol = self.mpc.u_guess[i * self.nu:(i+1) * self.nu].copy().squeeze()
            # Handle control delays
            if "Delay" in self.sim_model.name and "Acc" not in self.sim_model.name:
                control_tmp = control_buffer_tmp[-1].copy()
                control_tmp += sol * self.dt
                control_buffer_tmp.append(control_tmp)
            else:
                control_buffer_tmp.append(sol)
            
            # Apply filtering
            if self.sim_config.add_filter:
                control_buffer_tmp[0][0] = self.lp_v.filter(control_buffer_tmp[0][0])
                control_buffer_tmp[0][1] = self.lp_w.filter(control_buffer_tmp[0][1])
            
            # Apply clamping
            if self.sim_config.add_clamp:
                control_buffer_tmp[0][0] = max(0, control_buffer_tmp[0][0])

            # Apply bad wheel simulation
            if self.sim_config.bad_wheel:
                self.bad_wheel_simulation(control_buffer_tmp[0]) # Note that the i==0 record maybe not correct
            
            # Integrate dynamics
            x0_tmp = self.sim_model.integrate(x0_tmp, control_buffer_tmp[0])
            
            # Add noise
            if self.sim_config.add_noise:
                x0_tmp += self.sim_model.noise_model.create_noise()
            
            if i == 0: 
                self.sim_x0 = x0_tmp.copy()
                self.control_buffer = control_buffer_tmp.copy()
                self.error.append([self.mpc.lat_error, self.mpc.head_error])

            self.state.append(x0_tmp)
            self.control.append(sol)

            # print("-" * 50) # The gpr_method 1 error is growning real quick
            # print(i, x0_tmp.squeeze()[3:] - self.mpc.x_guess[i * self.mpc.model.nx:(i + 1) * self.mpc.model.nx].squeeze()[3:5])

        # Store simulation data
        self.control = np.array(self.control)
        self.state = np.array(self.state)
        mpc_delta_v = self.state[:, 3] - self.mpc.x_guess[self.mpc.model.LINV::self.mpc.model.nx]
        mpc_delta_w = self.state[:, 4] - self.mpc.x_guess[self.mpc.model.YAWV::self.mpc.model.nx]
        mpc_delta_x0 = self.state[:, 0] - self.mpc.x_guess[0::self.mpc.model.nx]
        mpc_delta_x1 = self.state[:, 1] - self.mpc.x_guess[1::self.mpc.model.nx]
        self.delta_v.append(mpc_delta_v)
        self.delta_w.append(mpc_delta_w)
        self.delta_x0.append(mpc_delta_x0)
        self.delta_x1.append(mpc_delta_x1)

        if self.interrupt_mode:
            print("delta v: ", f"{np.mean(np.abs(mpc_delta_v)):.6f}")
            print("delta w: ", f"{np.mean(np.abs(mpc_delta_w)):.6f}")
        
        ## we can save some array for isolate test
        if save_npy:
            np.save(f_npy, self.mpc.u_guess)

        self.check_termination_conditions(self.sim_x0)

        return True
    
def test_mpc():
    enable_animation = True
    # --------------- Read the config from yaml --------------- #
    for (root, dirs, files) in os.walk("config/mpc"):
        config_list = files

    print("Choose one mpc config to run:")
    for index, name in enumerate(config_list):
        print(f" {index:2d}: {name}")
    
    user_select = int(input())
    config_dir = "config/mpc/" + config_list[user_select]
    config = Config.from_yaml(config_dir)
    # path_config = TrajectoryConfig.from_yaml("config/track/bezier_path.yaml")
    # path_config = TrajectoryConfig.from_yaml("config/track/l4acados_config.yaml")

    # --------------- Create trajectory and mpc controller --------------- #
    # bezier_path = BezierPath(path_config)
    # bezier_path = RacePath(path_config.traj, "config/track/l4acados_track.yaml")
    bezier_path = Trajectory("config/track/full_track_opt.yaml")
    # bezier_path = Trajectory("config/track/1-10.yaml")
    mpc = MPCList[config.mpc.mpc_name](
        config, 
        bezier_path.path, 
        bezier_path.speed_profile, 
    )

    # --------------- Setup simulation and plot --------------- #
    simulation_runner = SimulationRunner(mpc, config)
    plotter = MPCPlotter(bezier_path)
    plotter.setup_animation_plot()

    while True:
        status = simulation_runner.run_step()
        if enable_animation:
            plotter.update_animation(simulation_runner)
        elif simulation_runner.index % 100 == 0:
            print(f"Simulation step: {simulation_runner.index}/{config.sim.SIM_TIMES}")

        if status is False: 
            print("Terminated with reason: ", simulation_runner.termination_reason)
            break
    
    if enable_animation:
        plt.show()

    plotter.print_performance_summary(simulation_runner)