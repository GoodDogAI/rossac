from gym.spaces import Box, Tuple
import numpy as np
from math import inf, ceil
from typing import Optional


def normalize_output(val, low, high):
    return (val - low) * 2 / (high - low) - 1

def output_from_normalized(val_normalized, low, high):
    return (val_normalized + 1) * (high - low) / 2 + low

YOLO_OBSERVATION_SPACE = Box(low=-inf, high=inf, shape=(512 * 15 * 20,), dtype=np.float32)


class RobotEnvironment:
    # Observation space is the yolo intermediate layer output
    observation_space = YOLO_OBSERVATION_SPACE

    def __init__(self, pan_low=-35, pan_high=35, tilt_low=-35, tilt_high=35):
        # Action space is the forward speed, angular rate, camera pan, and camera tilt
        # Constants taken from randomwalk.cpp in the mainbot brain code
        self.action_space = Box(low=np.array([-0.5, -0.5, pan_low, tilt_low]),
                                high=np.array([0.5, 0.5, pan_high, tilt_high]), dtype=np.float32)

    @property
    def pan_low(self):
        return self.action_space.low[2]

    @property
    def pan_high(self):
        return self.action_space.high[2]

    @property
    def tilt_low(self):
        return self.action_space.low[3]

    @property
    def tilt_high(self):
        return self.action_space.high[3]


class SlicedRobotEnvironment(RobotEnvironment):
    # Observation space is the yolo intermediate layer output, but sliced down every X elements
    observation_space = None

    def __init__(self, slice: Optional[int]=None, **kwargs):
        super().__init__(**kwargs)

        yolo_output_size = 1024 * 15 * 20 if slice is None else ceil(1024 * 15 * 20 / slice)
        pantilt_current_size = 4
        head_gyro_size = 3
        head_accel_size = 3
        odrive_feedback_size = 6
        vbus_size = 1
        total_size = (pantilt_current_size + head_gyro_size + head_accel_size
                     + odrive_feedback_size + vbus_size + yolo_output_size)
        self.observation_space = Box(low=-inf, high=inf, shape=(total_size,), dtype=np.float32)


class NormalizedRobotEnvironment(RobotEnvironment):
    def __init__(self, original_env: RobotEnvironment):
        super().__init__()
        self.original_env = original_env

        self.observation_space = original_env.observation_space

        self.action_space = Box(low=np.array([-0.5, -0.5, -1, -1]),
                                high=np.array([+0.5, +0.5, +1, +1]), dtype=np.float32)

    def normalize_pan(self, pan):
        return normalize_output(pan, low=self.original_env.pan_low, high=self.original_env.pan_high)

    def pan_from_normalized(self, normalized_pan):
        return output_from_normalized(normalized_pan, low=self.original_env.pan_low, high=self.original_env.pan_high)

    def normalize_tilt(self, tilt):
        return normalize_output(tilt, low=self.original_env.tilt_low, high=self.original_env.tilt_high)

    def tilt_from_normalized(self, normalized_tilt):
        return output_from_normalized(normalized_tilt, low=self.original_env.tilt_low, high=self.original_env.tilt_high)