from gym.spaces import Box
import numpy as np
from math import inf

PAN_LOW = 350
PAN_HIGH = 700
TILT_LOW = 475
TILT_HIGH = 725

class RobotEnvironment:
    # Observation space is the yolo intermediate layer output
    observation_space = Box(low=-inf, high=inf, shape=(512 * 15 * 20,), dtype=np.float32)

    # Action space is the forward speed, angular rate, camera pan, and camera tilt
    # Constants taken from randomwalk.cpp in the mainbot brain code
    action_space = Box(low=np.array([-0.5, -0.5, PAN_LOW, TILT_LOW]),
                       high=np.array([0.5, 0.5, PAN_HIGH, TILT_HIGH]), dtype=np.float32)

def _normalize(val, low, high):
    return (val - low) * 2 / (high - low) - 1
def _from_normalized(val_normalized, low, high):
    return (val_normalized + 1) * (high - low) / 2 + low

class NormalizedRobotEnvironment(RobotEnvironment):
    action_space = Box(low=np.array([-0.5, -0.5, -1, -1]),
                       high=np.array([+0.5, +0.5, +1, +1]), dtype=np.float32)

    @staticmethod
    def normalize_pan(pan): return _normalize(pan, low=PAN_LOW, high=PAN_HIGH)
    @staticmethod
    def pan_from_normalized(normalized_pan):
        return _from_normalized(normalized_pan, low=PAN_LOW, high=PAN_HIGH)
    @staticmethod
    def normalize_tilt(tilt): return _normalize(tilt, low=TILT_LOW, high=TILT_HIGH)
    @staticmethod
    def tilt_from_normalized(normalized_tilt):
        return _from_normalized(normalized_tilt, low=TILT_LOW, high=TILT_HIGH)