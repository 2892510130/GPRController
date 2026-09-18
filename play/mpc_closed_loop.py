import os
import numpy as np
import matplotlib.pyplot as plt
from model import ModelLists
from collections import deque
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle
from utils.filter import LPFilter
from utils import Config, TrajectoryConfig
from controller import MPCList
from trajectory import BezierPath, RacePath, Trajectory

class MPCPlotter:
    """Handles all plotting functionality for MPC simulation"""
    
    def __init__(self, bezier_path):
        self.bezier_path = bezier_path
        self.fig = None
        self.ax = None
        self.static_line = None
        self.static_scatter = None
        self.dynamic_scatter = None
        self.dynamic_square = None
        self.square_height = 0.08
        self.square_width = 0.06
        
    def setup_animation_plot(self, sim_runner):
        """Initialize the real-time animation plot"""

        self.fig, self.ax = plt.subplots()
        
        # Static reference path
        self.static_line, = self.ax.plot(
            self.bezier_path.path[:, 0], 
            self.bezier_path.path[:, 1], 
            'b-', 
            label='Reference Path'
        )
        self.static_scatter = self.ax.scatter(
            self.bezier_path.path[:, 0], 
            self.bezier_path.path[:, 1], 
            s=10, c='blue', alpha=0.5
        )
        
        # Dynamic simulation elements
        self.dynamic_scatter = self.ax.scatter(
            [], [], s=50, c='red', label='Simulated Path'
        )
        self.dynamic_square = Rectangle(
            (0, 0), self.square_height, self.square_width, 
            angle=0, fc='yellow', ec='black', label='Car'
        )
        self.ax.add_patch(self.dynamic_square)

        self.pred_path, = self.ax.plot([], [], 'green', label="Pred Path")

        self.var_fig = []

        if sim_runner.mpc.opts.propagete_cov:
            for i in range(sim_runner.mpc.opts.mpcWindow):
                circle = Circle((0, 0), radius=0.0, edgecolor='green', facecolor='none', linewidth=2, alpha=0.6, label="Uncertainty" if i == 0 else None)
                self.var_fig.append(circle)
                self.ax.add_patch(circle)
        
        # Configure plot appearance
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.legend(fontsize=12) #28
        self.ax.grid(True)
        
        # Set plot limits with padding
        x_min = self.bezier_path.path[:, 0].min() - 0.25
        x_max = self.bezier_path.path[:, 0].max() + 0.25
        y_min = self.bezier_path.path[:, 1].min() - 0.25
        y_max = self.bezier_path.path[:, 1].max() + 0.25
        
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_aspect('equal')

        # ax = plt.gca()  # 获取当前轴对象
        self.ax.spines['top'].set_linewidth(4)  # 上边框
        self.ax.spines['right'].set_linewidth(4)  # 右边框
        self.ax.spines['bottom'].set_linewidth(4)  # 下边框
        self.ax.spines['left'].set_linewidth(4)  # 左边框
        
        return self.fig, self.ax
    
    def update_animation(self, sim_runner, current_state):
        """Update the animation with new simulation data"""
        sim_path_array = np.array(sim_runner.sim_path)[:, [0, 1]]
        self.dynamic_scatter.set_offsets(sim_path_array)
        self.pred_path.set_data(sim_runner.mpc.pred_traj[:, 0], sim_runner.mpc.pred_traj[:, 1])
        
        # Update car position and orientation
        x, y, theta = current_state[0, 0], current_state[1, 0], current_state[2, 0]
        lb_x = x - self.square_height * np.cos(theta) / 2 + self.square_width * np.sin(theta) / 2
        lb_y = y - self.square_height * np.sin(theta) / 2 - self.square_width * np.cos(theta) / 2
        
        self.dynamic_square.set_xy((lb_x, lb_y))
        self.dynamic_square.angle = np.degrees(theta)

        if sim_runner.mpc.opts.propagete_cov and sim_runner.mpc.gaussian_dict.gpr_method >= 0:
            var_list = sim_runner.mpc.x_var_list
            for i in range(sim_runner.mpc.opts.mpcWindow):
                var = var_list[i]
                r = 2 * np.sqrt(var[0, 0] + var[1, 1])
                self.var_fig[i].set_center((sim_runner.mpc.pred_traj[i, 0], sim_runner.mpc.pred_traj[i, 1]))
                self.var_fig[i].set_radius(r)
        
        return self.static_line, self.static_scatter, self.dynamic_scatter, self.dynamic_square, self.pred_path, *self.var_fig
    
    def create_analysis_plots(self, time, sim_path, ref, control, state, mpc_model, sim_model):
        """Create post-simulation analysis plots"""
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        
        # Control and state tracking plot
        axs[0].plot(time, sim_path[:, sim_model.LINV], label="v", linewidth=2)
        axs[0].plot(time, sim_path[:, sim_model.YAWV], label="w", linewidth=2)
        axs[0].plot(time, ref[:, mpc_model.LINV], label="ref_v", linestyle='--', alpha=0.7)
        axs[0].plot(time, ref[:, mpc_model.YAWV], label="ref_w", linestyle='--', alpha=0.7)
        axs[0].plot(time, control[:, 0], label="control_dv", alpha=0.8)
        axs[0].plot(time, control[:, 1], label="control_dw", alpha=0.8)

        # axs[0].plot(time, np.cumsum(ref[:, mpc_model.YAWV]).squeeze() * 0.05 * 180 / 3.1415926, label="cum_ref")
        # axs[0].plot(time, np.cumsum(sim_path[:, mpc_model.YAWV]).squeeze() * 0.05 * 180 / 3.1415926, label="cum_sim")
        # axs[0].plot(time, ref[:, 2] * 180 / 3.1415926, label="real")

        # axs[0].plot(time, np.cumsum(ref[:, mpc_model.LINV]).squeeze() * 0.05, label="cum_ref")
        # axs[0].plot(time, np.cumsum(sim_path[:, mpc_model.LINV]).squeeze() * 0.05, label="cum_sim")
        # axs[0].plot(time, ref[:, 2], label="real")
        
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Velocity / Control')
        axs[0].set_title('Velocity Tracking and Control Inputs')
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)
        
        # Error tracking plot
        axs[1].plot(time, state[:, 0], label="lat_error", linewidth=2, color='red')
        axs[1].plot(time, state[:, 1], label="head_error", linewidth=2, color='orange')
        
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Error')
        axs[1].set_title('Tracking Errors')
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, axs
    
    def print_performance_summary(self, state):
        """Print simulation performance metrics"""
        print("\n" + "="*55)
        print(" " * 11 + "SIMULATION PERFORMANCE SUMMARY")
        print("="*55)
        print(f"Final lateral, heading error:    {float(state[-1, 0]):+.6f}, {float(state[-1, 1]):+.6f} |")
        print(f"Max   lateral, heading error:    {np.max(np.abs(state[:, 0])):+.6f}, {np.max(np.abs(state[:, 1])):+.6f} |")
        print(f"Mean  lateral, heading error:    {np.mean(np.abs(state[:, 0])):+.6f}, {np.mean(np.abs(state[:, 1])):+.6f} |")
        print("="*55)


class SimulationRunner:
    """Handles the core simulation logic and configuration"""
    
    def __init__(self, mpc, config):
        self.mpc = mpc
        self.config = config
        self.sim_config = config.sim
        self.frame_cnt = 0
        
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
        
        if "Delay" in self.sim_model.name: 
            print("[Simulation]: Using delay model")
        
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
        self.sim_path = []
        self.state = []
        self.ref = []
        self.control = []

        self.max_lateral_error = self.sim_config.max_lateral_error
        self.terminal_tolerance = self.sim_config.terminal_tolerance
        self.simulation_finished = False
        self.termination_reason = None

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
            lat_error = abs(self.state[-1][0])  # Latest lateral error
            if lat_error > self.max_lateral_error:
                self.simulation_finished = True
                self.termination_reason = f"Diverged too far (lat_error: {float(lat_error):.3f})"
                return True
        
        return False
    
    def bad_wheel_simulation(self, control):
        tmp1 = self.sim_config.wheel_right - self.sim_config.wheel_left
        tmp2 = self.sim_config.wheel_right + self.sim_config.wheel_left
        v = (2 * control[0] * tmp2 + self.sim_config.L * control[1] * tmp1) / 4
        w = (2 * control[0] * tmp1 + self.sim_config.L * control[1] * tmp2) / 2 / self.sim_config.L
        control[0] = v
        control[1] = w

    def run_step(self, current_state):
        """Execute one simulation step - even cleaner interface!"""
        if self.simulation_finished or self.frame_cnt > self.sim_config.SIM_TIMES:
            return current_state
        # Solve MPC
        self.mpc.initialize(current_state[:5], self.dt)
        sol = self.mpc.solve()[:self.nu].copy().squeeze()
        
        # Handle control delays
        if "Delay" in self.sim_model.name and "Acc" not in self.sim_model.name:
            control_tmp = self.control_buffer[-1].copy()
            control_tmp += sol * self.dt
            self.control_buffer.append(control_tmp)
        else:
            self.control_buffer.append(sol)
        
        # Apply filtering
        if self.sim_config.add_filter:
            self.control_buffer[0][0] = self.lp_v.filter(self.control_buffer[0][0])
            self.control_buffer[0][1] = self.lp_w.filter(self.control_buffer[0][1])
        
        # Apply clamping
        if self.sim_config.add_clamp:
            self.control_buffer[0][0] = max(0, self.control_buffer[0][0])
        
        # Apply bad wheel simulation
        if self.sim_config.bad_wheel:
            self.bad_wheel_simulation(self.control_buffer[0])

        # Integrate dynamics
        new_state = self.sim_model.integrate(current_state, self.control_buffer[0])
        
        # Add noise
        if self.sim_config.add_noise:
            new_state += self.sim_model.noise_model.create_noise()
        
        # Store simulation data
        self.sim_path.append(new_state)
        self.state.append([self.mpc.lat_error, self.mpc.head_error])
        self.ref.append(self.mpc.xr[:self.mpc.model.nx])
        self.control.append(np.array(sol))
        
        self.check_termination_conditions(new_state)
        self.frame_cnt += 1

        return new_state

    def run_full_simulation(self):
        """Headless: 只跑仿真，不做任何绘图"""
        current_state = self.get_initial_state()
        for _ in range(self.get_simulation_frames()):
            if self.simulation_finished:
                print(f"[Simulation] Simulation terminated: {self.termination_reason}")
                break
            current_state = self.run_step(current_state)
        self.mpc.logger.close_log()
        return current_state
    
    def get_simulation_frames(self):
        """Get the number of simulation frames"""
        return self.sim_config.SIM_TIMES
    
    def get_initial_state(self):
        """Get the initial simulation state"""
        return self.sim_x0
    
def test_mpc_closed_loop(visualize: int = 0): 
    plt.rcParams.update({
        "lines.linewidth": 2,
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "legend.frameon": True,
        "legend.framealpha": 1,
        "legend.edgecolor": "black",
        "font.size": 13,
        "savefig.dpi": 300
    })
    
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
    # bezier_path = RacePath("config/track/race_opt.yaml")
    # bezier_path = Trajectory("config/track/1-10.yaml")
    bezier_path = Trajectory("config/track/full_track_opt.yaml")
    mpc = MPCList[config.mpc.mpc_name](
        config, 
        bezier_path.path, 
        bezier_path.speed_profile, 
    )

    # --------------- Initialize simulation and plotting --------------- #
    sim_runner = SimulationRunner(mpc, config)
    plotter = MPCPlotter(bezier_path)

    if visualize >= 2:
        fig, ax = plotter.setup_animation_plot(sim_runner)

        # Get initial state from the runner
        current_state = sim_runner.get_initial_state()

        # --------------- Animation update function --------------- #
        def update(frame):
            nonlocal current_state
            if sim_runner.simulation_finished:
                print(f"[Simulation] Simulation terminated: {sim_runner.termination_reason}")
                ani.event_source.stop()
                sim_runner.mpc.logger.close_log()
                return plotter.update_animation(sim_runner, current_state)
            current_state = sim_runner.run_step(current_state)
            return plotter.update_animation(sim_runner, current_state)
        
        # --------------- Run animation --------------- #
        ani = FuncAnimation(
            fig, update, 
            frames=sim_runner.get_simulation_frames(),
            interval=20, blit=True, repeat=False
        )
        plt.show()
    else:
        sim_runner.run_full_simulation()

    # --------------- Generate analysis plots --------------- #
    # Convert lists to arrays
    state, ref, sim_path, control = (
        np.array(sim_runner.state), np.array(sim_runner.ref), np.array(sim_runner.sim_path), np.array(sim_runner.control))
    
    # Print performance summary
    plotter.print_performance_summary(state)
    
    # Create time vector and analysis plots
    if visualize >= 1:
        time = np.linspace(0, sim_path.shape[0] * sim_runner.dt, sim_path.shape[0])
        analysis_fig, analysis_axs = plotter.create_analysis_plots(
            time, sim_path, ref, control, state, mpc.model, sim_runner.sim_model
        )
        plt.show()

    sim_runner.mpc.save_model()