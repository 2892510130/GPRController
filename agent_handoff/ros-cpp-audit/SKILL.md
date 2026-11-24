---
name: ros-cpp-audit
description: Use when auditing or optimizing a C++/ROS (ROS 1 or ROS 2) robotics, control, estimation, MPC, or simulation stack for bugs, real-time issues, integration problems, or performance regressions; use when asked to review a ROS repository before changing controller, solver, parameter, TF, QoS, launch, or safety behavior.
---

# ROS / C++ control stack audit

Use this procedure to find, verify, and fix bugs in a C++/ROS repository without
turning guesses into conclusions. If the repository has `docs/agent/FROM_PYTHON.md`
(a handoff from a sibling implementation), read it first; it is a checklist, not
ground truth. The repository's own `AGENTS.md`, architecture, and runbook win.

## 1. Bootstrap before changing code

1. Record git SHA, branch, ROS distribution, build type, and workspace layout.
2. Clean build; count warnings. Run the full test suite; list failures and skips.
3. Find or record one golden rosbag; replay it; confirm the baseline behavior.
4. Record baseline metrics: tracking error, control-loop period/jitter, solver
   time p50/p99, solver failure count, overrun count, CPU/memory, topic rates.
5. Read the repository's `FINDINGS.md` / `HANDOFF.md` if present; list the top
   three risks.
6. Treat real-vehicle launch files as read-only unless the user explicitly asks.

## 2. Static audit

- Enable `-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wdouble-promotion`
  and `-Werror` only after agreeing on a warning baseline.
- `clang-tidy -p build` / `run-clang-tidy -p build` on changed packages.
- Build the simulation target with ASan/UBSan; never run sanitizer binaries on
  real hardware.
- Inspect uninitialized members, fixed-size arrays, Eigen dimensions/alignment/
  aliasing, integer division, signed/unsigned, unit conversions.
- Search for cached factorizations/matrices that incremental updates may
  invalidate; search for solver calls that ignore status codes.
- Compare parameter defaults in code, launch files, and YAML.

## 3. Dynamic audit

- Replay a bag with `/clock` and `use_sim_time:=true`; compare with wall time.
- Check topic types/QoS/rates and TF frames/timestamps.
- Add temporary instrumentation (timing, NaN/bounds checks, solver status) rather
  than guessing; remove or gate it before merging.
- Compare simulation and real logs for the same trajectory and parameters.
- For performance: `perf record --call-graph dwarf`, `ros2_tracing`, loop overrun
  counters, solver timing histograms; look for allocation/locks/logging inside
  the control loop.

## 4. High-yield checks

1. Filters/smoothers: history buffers initialized? step response gain 1? cutoff
   below Nyquist?
2. History/horizon dimensions: assert the dimension identity; sweep
   `input_window`/horizon sizes instead of testing only the default.
3. Model variants: no-GP / GP variants / augmented / Frenet / Cartesian all
   initialize and solve.
4. Parameters: required values validated; `ros2 param dump` matches launch/YAML.
5. Model/calibration files: version, feature window, normalization, solver/GP
   type, checksum; reject mismatches instead of silently predicting.
6. Online updates: cache invalidation flags; no mixing stale/new predictors;
   save only consistent state.
7. Solver: check status, timeout, iteration limit, NaN/Inf, bounds; define a
   fail-safe action and monitor failure rate.
8. Time/units/frames: sim vs wall time, rad/deg, m/mm, TF direction/timestamp.
9. Control loop: allocations, locks, logging, exceptions; watchdog behavior.
10. Nearest-point/path index: local window + global fallback, self-intersecting
    tracks, lap wrap, reinitialization.

## 5. Evidence format

Every finding records:

```text
ID/severity/area
Symptom / expected / actual
Repro: command + launch args + bag + git SHA
Evidence: log line, topic sample, timing, screenshot
Root cause:
Fix + regression test:
Status: open / investigating / fixed / wontfix
```

Label confidence: E0 code reading, E1 static, E2 simulation/bag, E3 real.
Only E2/E3 supports "confirmed".

## 6. Commands (ROS 2; note ROS 1 equivalents)

```bash
colcon build --symlink-install \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo \
               -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
colcon test --packages-select <pkg> --event-handlers console_direct+
colcon test-result --verbose

ros2 node list
ros2 node info /<node>
ros2 topic list -t
ros2 topic hz /<topic>
ros2 topic bw /<topic>
ros2 param dump /<node>
ros2 run tf2_ros tf2_echo map base_link
ros2 bag record -o <bag> <topics>
ros2 bag play <bag> --clock
ros2 launch <pkg> <launch.py> use_sim_time:=true
```

ROS 1: `catkin build`/`catkin_make`, `catkin_make run_tests`/`rostest`,
`rostopic hz`, `rosparam dump`, `rosrun tf tf_echo`, `rosbag record/play`.

## 7. Fix loop

1. Reproduce with the smallest scenario.
2. Fix one root cause at a time.
3. Add a regression test: gtest for units, launch_testing for integration, bag
   replay for scenarios.
4. Re-run the baseline; compare metrics before/after.
5. Update `FINDINGS.md` / `HANDOFF.md`; note any compatibility change (topics,
   frames, parameters, units, control period).

## 8. Safety

- Do not enable real hardware motion, disable e-stops, or change safety limits
  without explicit user approval.
- Prefer simulation/bag evidence; state what remains unverified.
- Keep a rollback path (git branch, parameter backup, known-good bag).
