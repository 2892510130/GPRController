## Cross-project prior experience

Before starting, read `docs/agent/FROM_PYTHON.md`. It contains cross-project
lessons from the Python GPRController implementation (bug classes, invariants,
verification recipes, C++/ROS checks). Treat it as a checklist, not as ground
truth for this repository.

- If it conflicts with this repository's code or docs, this repository wins.
- Reproduce every suspected issue in Gazebo or a rosbag before claiming it.
- Do not run real-vehicle launch files or change safety limits without explicit
  approval.
- Every finding needs: command/launch args/bag, expected vs actual, root cause,
  regression test, and before/after metric.
- Update `docs/agent/FINDINGS.md` and `docs/agent/HANDOFF.md` as work progresses.

For the first session, follow the bootstrap/runbook in `FROM_PYTHON.md`
(clean build → tests → golden bag replay → baseline → top risks).
