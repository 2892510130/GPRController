import numpy as np
from collections import deque
from typing import Iterable, Sequence


class LPFilter:
    """Second-order Butterworth low-pass filter (bilinear transform).

    Keeps the last three input samples and the last two output samples so the
    biquad difference equation

        y[k] = (b0*x[k] + b1*x[k-1] + b2*x[k-2]
                - a1*y[k-1] - a2*y[k-2]) / a0

    can be evaluated with the full history. The deques must start filled with
    zeros: an empty deque never grows past one element because every call pops
    the oldest sample before inserting the new one.
    """

    def __init__(self, cutoff_freq: float, dt: float) -> None:
        if dt <= 0.0:
            raise ValueError(f"LPFilter: dt must be positive, got {dt}")
        nyquist = 0.5 / dt
        if not 0.0 < cutoff_freq < nyquist:
            raise ValueError(
                f"LPFilter: cutoff_freq ({cutoff_freq} Hz) must be in "
                f"(0, {nyquist} Hz) for dt = {dt} s"
            )

        self.x_values: deque[float] = deque([0.0] * 3, maxlen=3)
        self.y_values: deque[float] = deque([0.0] * 3, maxlen=3)

        wa = 2.0 * np.pi * cutoff_freq
        # Bilinear transform with frequency pre-warping so that the digital
        # -3 dB point matches `cutoff_freq` exactly (tan x ~ x for small x).
        alpha = np.tan(wa * dt / 2.0)
        alpha_sqr = alpha * alpha
        tmp_term = np.sqrt(2.0) * alpha + alpha_sqr
        gain = alpha_sqr / (1.0 + tmp_term)

        self.denominator = [
            1.0,
            2.0 * (alpha_sqr - 1.0) / (1.0 + tmp_term),
            (1.0 - np.sqrt(2.0) * alpha + alpha_sqr) / (1.0 + tmp_term),
        ]

        self.numerator = [gain, 2.0 * gain, gain]

    def filter(self, value: float) -> float:
        self.x_values.pop()
        self.x_values.appendleft(value)
        xside = self.compute(self.x_values, self.numerator, 0)

        self.y_values.pop()
        yside = self.compute(self.y_values, self.denominator, 1)

        y_insert = 0.0
        if np.fabs(self.denominator[0]) > 1e-6:
            y_insert = (xside - yside) / self.denominator[0]

        self.y_values.appendleft(y_insert)

        return y_insert

    @staticmethod
    def compute(
        values: Iterable[float],
        coefficients: Sequence[float],
        coeff_start: int = 0,
    ) -> float:
        total = 0.0
        for value, coeff in zip(values, coefficients[coeff_start:]):
            total += value * coeff
        return total
