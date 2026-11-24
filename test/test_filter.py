import numpy as np
import pytest

from scipy.signal import butter, lfilter

from utils.filter import LPFilter

DT = 0.05
CUTOFF = 5.0


def test_matches_scipy_butterworth():
    lp_filter = LPFilter(CUTOFF, DT)
    y = np.array([lp_filter.filter(1.0) for _ in range(200)])

    b, a = butter(2, CUTOFF, btype="lowpass", fs=1.0 / DT)
    y_ref = lfilter(b, a, np.ones(200))

    # Start-up samples as well as the steady state must match the reference
    # biquad: a filter that never stores its history converges to `b0` only.
    np.testing.assert_allclose(y, y_ref, atol=1e-10)
    assert y[-1] == pytest.approx(1.0, abs=1e-6)


def test_negative_and_zero_cutoff_rejected():
    with pytest.raises(ValueError):
        LPFilter(0.0, DT)
    with pytest.raises(ValueError):
        LPFilter(-1.0, DT)


def test_cutoff_above_nyquist_rejected():
    # 25 Hz is above the 10 Hz Nyquist frequency of a 20 Hz control loop.
    with pytest.raises(ValueError):
        LPFilter(25.0, DT)


def test_non_positive_dt_rejected():
    with pytest.raises(ValueError):
        LPFilter(CUTOFF, 0.0)
