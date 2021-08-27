import unittest
import torch
import numpy as np
import os

import png

from bot_env import NormalizedRobotEnvironment
from dump_onnx import export
from sac import SoftActorCritic
from train import get_onnx_sess
from yolo_reward import get_prediction, get_intermediate_layer
from yolo_reward import detect_yolo_bboxes, yolo1, yolo2, yolo3


class TestONNXExport(unittest.TestCase):
    def testBasicActorNetwork(self):
        env_fn = lambda: NormalizedRobotEnvironment(slice=157)
        sac = SoftActorCritic(env_fn)
        export(sac.ac, None, "test.onnx", sac.env)