import unittest
from math import sqrt

import torch
import numpy as np
import os

import png
import onnxruntime as rt


from train import get_onnx_sess
from yolo_reward import get_onnx_prediction, get_intermediate_layer, convert_wh_to_nchw, get_pt_gpu_prediction
from yolo_reward import detect_yolo_bboxes


class TestYoloExport(unittest.TestCase):
    onnx_path = os.path.join(os.path.dirname(__file__), "..", "yolov5s_trt.onnx")
    pt_path = os.path.join(os.path.dirname(__file__), "..", "yolov5s.torchscript.pt")

    def setUp(self) -> None:
        pngdata = png.Reader(filename=os.path.join(os.path.dirname(__file__), "test_data", "chair_person.png")).asRGB8()
        image_np = np.vstack(pngdata[2])
        image_np = image_np.reshape((image_np.shape[0], image_np.shape[1] // 3, 3))
        image_np = image_np[..., 0]
        self.image_np = image_np

    def testOnnxAndPTLineUp(self):
        onnx_sess = rt.InferenceSession(self.onnx_path)
        onnx_pred = get_onnx_prediction(onnx_sess, self.image_np)

        pt_sess = torch.jit.load(self.pt_path)
        pt_pred = pt_sess(torch.from_numpy(convert_wh_to_nchw(self.image_np)))

        gpu_device = torch.device("cuda")
        pt_gpu_sess = torch.jit.load(self.pt_path, gpu_device)

        pt_gpu_pred = get_pt_gpu_prediction(pt_gpu_sess, self.image_np)

        np.testing.assert_almost_equal(onnx_pred[0], pt_pred[0].numpy(), decimal=2)
        np.testing.assert_almost_equal(onnx_pred[0], pt_gpu_pred, decimal=2)

    def testChairPerson(self):
        onnx_sess = rt.InferenceSession(self.onnx_path)
        pred = get_onnx_prediction(onnx_sess, self.image_np)

        boxes = detect_yolo_bboxes(pred[0])

        # There should be at least one person and one chair in this image
        self.assertEqual(set(b.class_name for b in boxes), {"person", "chair"})

        # All detections should be in lower left corner
        for box in boxes:
            self.assertLess(sqrt((box.x - 50) ** 2 + (box.y - 356) ** 2), 100)
