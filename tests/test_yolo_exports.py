import unittest
from collections import Counter
from math import sqrt

import torch
import numpy as np
import os

import onnxruntime as rt


from yolo_reward import get_onnx_prediction, convert_hwc_to_nchw, non_max_supression, load_png
from yolo_reward import detect_yolo_bboxes


def get_pt_gpu_prediction(pt: torch.ScriptModule, image_np: np.ndarray) -> np.ndarray:
    image_np = torch.from_numpy(convert_hwc_to_nchw(image_np))

    device = torch.device("cuda")
    pred = pt(image_np.to(device))
    pred = pred[0].cpu().numpy()

    return pred


class TestYoloExport(unittest.TestCase):
    onnx_path = os.path.join(os.path.dirname(__file__), "..", "yolov5s_v5_0_op11_rossac.onnx")
    pt_path = os.path.join(os.path.dirname(__file__), "..", "yolov5s.torchscript.pt")

    def setUp(self) -> None:
        self.image_np = load_png(os.path.join(os.path.dirname(__file__), "test_data", "chair_person.png"))

    def testOnnxAndPTMatch(self):
        onnx_sess = rt.InferenceSession(self.onnx_path)
        onnx_pred = get_onnx_prediction(onnx_sess, self.image_np)

        pt_sess = torch.jit.load(self.pt_path)
        pt_pred = pt_sess(torch.from_numpy(convert_hwc_to_nchw(self.image_np)))

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

    def testColorBasic(self):
        onnx_sess = rt.InferenceSession(self.onnx_path)
        image_np = load_png(os.path.join(os.path.dirname(__file__), "test_data", "frame0429.png"))

        bboxes, intermediate = get_onnx_prediction(onnx_sess, image_np)
        nms_boxes = non_max_supression(bboxes)
        detections = detect_yolo_bboxes(nms_boxes, threshold=0.25)

        # Check that it matches the official rendering at least in number of detections of each kind
        c = Counter(d.class_name for d in detections)

        self.assertEqual(c, {
            "tv": 1,
            "potted plant": 1,
            "book": 6,
            "chair": 1,
        })
