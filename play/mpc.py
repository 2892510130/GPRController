"""Legacy interactive entry point.

The duplicated ``SimulationRunner`` / ``MPCPlotter`` implementation used to
live here. It has been removed in favour of the single implementation in
``play.mpc_closed_loop``; ``test_mpc`` is kept as a thin animated wrapper.
"""

from .mpc_closed_loop import test_mpc_closed_loop


def test_mpc():
    """Deprecated alias for :func:`test_mpc_closed_loop` with animation."""
    return test_mpc_closed_loop(visualize=2)
