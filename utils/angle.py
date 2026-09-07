import numpy as np

def normalize_angle(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi

def convert_angle(angle_vec: list[float]) -> None:
    for i in range(1, len(angle_vec)):
        da = angle_vec[i] - angle_vec[i-1]
        angle_vec[i] = angle_vec[i-1] + normalize_angle(da)