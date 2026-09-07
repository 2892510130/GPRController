import numpy as np
from collections import deque

class LPFilter:
    def __init__(self, cutoff_freq: float, dt: float) -> None:
        self.x_values = deque([] * 3, maxlen = 3)
        self.y_values = deque([] * 3, maxlen = 3)

        wa = 2.0 * np.pi * cutoff_freq
        alpha = wa * dt / 2.0;   
        alpha_sqr = alpha * alpha
        tmp_term = np.sqrt(2.0) * alpha + alpha_sqr
        gain = alpha_sqr / (1.0 + tmp_term)

        self.denominator = [
            1.0, 
            2.0 * (alpha_sqr - 1.0) / (1.0 + tmp_term), 
            (1.0 - np.sqrt(2.0) * alpha + alpha_sqr) / (1.0 + tmp_term)
        ]

        self.numerator = [gain, 2.0 * gain, gain]
    
    def filter(self, input: float) -> float:
        if len(self.x_values) != 0:
            self.x_values.pop()
        self.x_values.appendleft(input)
        xside = self.compute(self.x_values, self.numerator, 0)

        if len(self.y_values) != 0:
            self.y_values.pop()
        yside = self.compute(self.y_values, self.denominator, 1)

        y_insert = 0.0
        if np.fabs(self.denominator[0]) > 1e-6:
            y_insert = (xside - yside) / self.denominator[0]

        self.y_values.appendleft(y_insert)

        return y_insert
    
    def compute(self, values: float, coefficients: float, coeff_start: float) -> float:
        sum = 0.0
        for value, coeff in zip(values, coefficients[coeff_start:]):
            sum += value * coeff
        return sum