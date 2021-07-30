import unittest
import torch
import numpy as np
import os

import png

from train import get_onnx_sess
from yolo_reward import get_prediction, get_intermediate_layer
from yolo_reward import detect_yolo_bboxes, yolo1, yolo2, yolo3


class TestYoloExport(unittest.TestCase):
    def testChairPerson(self):
        pngdata = png.Reader(filename=os.path.join(os.path.dirname(__file__), "test_data", "chair_person.png")).asRGB8()
        image_np = np.vstack(pngdata[2])
        image_np = image_np.reshape((image_np.shape[0], image_np.shape[1] // 3, 3))
        image_np = image_np[..., 0]

        pred = get_prediction(get_onnx_sess(os.path.join(os.path.dirname(__file__), "..", "yolov5s.onnx")), image_np)
        intermediate = get_intermediate_layer(pred)
        boxes = detect_yolo_bboxes(pred[0], yolo1)

        # There should be at least one person in this image
        # TODO: Latest yolo builds detect 1 person and 1 chair, but we are not using those checkpoints
        self.assertTrue(any(b.class_name == "person" for b in boxes))