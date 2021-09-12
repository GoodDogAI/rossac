import unittest
import torch
import numpy as np
import os

import png
import onnxruntime as rt


from train import get_onnx_sess
from yolo_reward import get_onnx_prediction, get_intermediate_layer, convert_wh_to_nchw
from yolo_reward import detect_yolo_bboxes, yolo1, yolo2, yolo3


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

        np.testing.assert_almost_equal(onnx_pred[0], pt_pred[0].numpy(), decimal=2)

    # def testChairPerson(self):
    #     pred = get_prediction(get_onnx_sess(os.path.join(os.path.dirname(__file__), "..", "yolov5s_trt.onnx")), self.image_np)
    #     intermediate = get_intermediate_layer(pred)
    #     boxes = detect_yolo_bboxes(pred[0], yolo1)
    #
    #     # There should be at least one person in this image
    #     # TODO: Latest yolo builds detect 1 person and 1 chair, but we are not using those checkpoints
    #     self.assertTrue(any(b.class_name == "person" for b in boxes))