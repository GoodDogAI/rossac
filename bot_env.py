from gym.spaces import Box, Tuple
import numpy as np
from math import inf, ceil
from typing import Optional

PAN_LOW = -35
PAN_HIGH = 35
TILT_LOW = -35
TILT_HIGH = 35


def normalize_output(val, low, high):
    return (val - low) * 2 / (high - low) - 1

def output_from_normalized(val_normalized, low, high):
    return (val_normalized + 1) * (high - low) / 2 + low

YOLO_OBSERVATION_SPACE = Box(low=-inf, high=inf, shape=(512 * 15 * 20,), dtype=np.float32)

class RobotEnvironment:
    # Observation space is the yolo intermediate layer output
    observation_space = YOLO_OBSERVATION_SPACE

    # Action space is the forward speed, angular rate, camera pan, and camera tilt
    # Constants taken from randomwalk.cpp in the mainbot brain code
    action_space = Box(low=np.array([-0.5, -0.5, PAN_LOW, TILT_LOW]),
                       high=np.array([0.5, 0.5, PAN_HIGH, TILT_HIGH]), dtype=np.float32)


class SlicedRobotEnvironment:
    # Observation space is the yolo intermediate layer output, but sliced down every X elements
    observation_space = None

    # Action space is the forward speed, angular rate, camera pan, and camera tilt
    # Constants taken from randomwalk.cpp in the mainbot brain code
    action_space = Box(low=np.array([-0.5, -0.5, PAN_LOW, TILT_LOW]),
                       high=np.array([0.5, 0.5, PAN_HIGH, TILT_HIGH]), dtype=np.float32)

    def __init__(self, slice: Optional[int]=None):
        yolo_output_size = 512 * 15 * 20 if slice is None else ceil(512 * 15 * 20 / slice)
        pantilt_current_size = 2
        head_gyro_size = 3
        head_accel_size = 3
        odrive_feedback_size = 2
        vbus_size = 1
        total_size = (pantilt_current_size + head_gyro_size + head_accel_size
                    + odrive_feedback_size + vbus_size + yolo_output_size)
        self.observation_space = Box(low=-inf, high=inf, shape=(total_size,), dtype=np.float32)


class NormalizedRobotEnvironment(SlicedRobotEnvironment):
    action_space = Box(low=np.array([-0.5, -0.5, -1, -1]),
                       high=np.array([+0.5, +0.5, +1, +1]), dtype=np.float32)

    @staticmethod
    def normalize_pan(pan): return normalize_output(pan, low=PAN_LOW, high=PAN_HIGH)
    @staticmethod
    def pan_from_normalized(normalized_pan):
        return output_from_normalized(normalized_pan, low=PAN_LOW, high=PAN_HIGH)
    @staticmethod
    def normalize_tilt(tilt): return normalize_output(tilt, low=TILT_LOW, high=TILT_HIGH)
    @staticmethod
    def tilt_from_normalized(normalized_tilt):
        return output_from_normalized(normalized_tilt, low=TILT_LOW, high=TILT_HIGH)