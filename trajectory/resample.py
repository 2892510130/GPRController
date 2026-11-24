import numpy as np
from scipy.interpolate import CubicSpline
from utils.math import lerp

def resample(base_value: np.ndarray, base_key: np.ndarray, query_key: np.ndarray, method: str = "linear") ->np.ndarray:
    if method == "linear":
        return resample_linear(base_value, base_key, query_key)
    elif method == "cubic":
        return resample_with_scipy(base_value, base_key, query_key)

def resample_with_scipy(base_value: np.ndarray, base_key: np.ndarray, query_key: np.ndarray) ->np.ndarray:
    spline = CubicSpline(base_key, base_value, extrapolate=True)
    return spline(query_key)

def resample_linear(base_value: np.ndarray, base_key: np.ndarray, query_key: np.ndarray) ->np.ndarray:
    key_index = 0
    key_length = len(base_key)
    query_values = []
    for query in query_key:
        while key_index + 2 < key_length and base_key[key_index + 1] < query:
            key_index += 1
        
        ratio = (query - base_key[key_index]) / (base_key[key_index + 1] - base_key[key_index])
        query_values.append(lerp(base_value[key_index], base_value[key_index + 1], ratio))
    return np.array(query_values)