from gym.spaces import Box
import numpy as np
from math import inf

class RobotEnvironment:
    # Observation space is the yolo intermediate layer output
    observation_space = Box(low=-inf, high=inf, shape=(512 * 15 * 20,), dtype=np.float32)

    # Action space is the forward speed, angular rate, camera pan, and camera tilt
    # Constants taken from randomwalk.cpp in the mainbot brain code
    action_space = Box(low=np.array([-0.5, -0.5, 350, 475]),
                       high=np.array([0.5, 0.5, 700, 725]), dtype=np.float32)