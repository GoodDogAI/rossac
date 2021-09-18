import unittest
import numpy as np
import os

import png
import onnxruntime as rt


from yolo_reward import get_onnx_prediction, sum_centered_objects_present, prioritize_centered_objects, non_max_supression
from yolo_reward import detect_yolo_bboxes


class TestYoloReward(unittest.TestCase):
    onnx_path = os.path.join(os.path.dirname(__file__), "..", "yolov5s_trt.onnx")

    def setUp(self) -> None:
        self.onnx_sess = rt.InferenceSession(self.onnx_path)

    def _load_image_np(self, path) -> np.ndarray:
        pngdata = png.Reader(filename=path).asRGB8()
        image_np = np.vstack(pngdata[2])
        image_np = image_np.reshape((image_np.shape[0], image_np.shape[1] // 3, 3))
        image_np = image_np[..., 0]
        return image_np

    def _get_sum_centered_objects_present(self, image_np) -> float:
        pred = get_onnx_prediction(self.onnx_sess, image_np)
        return sum_centered_objects_present(pred)

    def test_sum_centered_objects_present(self):
        reward = self._get_sum_centered_objects_present(self._load_image_np(
            os.path.join(os.path.dirname(__file__), "test_data", "chair_person.png")))

        # The image has a person and chair in the lower left corner, so rolling the image
        # to the right by 50 pixels, should increase the reward
        reward_roll_right = self._get_sum_centered_objects_present(self._load_image_np(
            os.path.join(os.path.dirname(__file__), "test_data", "chair_person_shift_right1.png")))
        self.assertGreater(reward_roll_right, reward)

        # Rolling the graphic so it's almost in the center
        reward_shift_center = self._get_sum_centered_objects_present(self._load_image_np(
            os.path.join(os.path.dirname(__file__), "test_data", "chair_person_shift_center.png")))
        self.assertGreater(reward_shift_center, reward_roll_right)

        # Rolling the graphic so it's moving past into the right side of the screen should be less
        reward_shift_offcenter = self._get_sum_centered_objects_present(self._load_image_np(
            os.path.join(os.path.dirname(__file__), "test_data", "chair_person_shift_offcenter.png")))
        self.assertGreater(reward_shift_offcenter, reward)
        self.assertLess(reward_shift_offcenter, reward_shift_center)

        reward_shift_center_scaledup = self._get_sum_centered_objects_present(self._load_image_np(
            os.path.join(os.path.dirname(__file__), "test_data", "chair_person_shift_center_scaledup.png")))
        self.assertGreater(reward_shift_center_scaledup, reward_shift_offcenter)

        # Roll the image up
        reward_roll_up = self._get_sum_centered_objects_present(np.roll(self._load_image_np(
            os.path.join(os.path.dirname(__file__), "test_data", "chair_person.png")), -50, axis=0))
        self.assertGreater(reward_roll_up, reward)

        # Flipping the image left/right should be around the same reward
        reward_flip_lr = self._get_sum_centered_objects_present(np.flip(self._load_image_np(
            os.path.join(os.path.dirname(__file__), "test_data", "chair_person.png")), axis=1))
        self.assertLess(abs(reward - reward_flip_lr), 1.0)

    def test_prioritize_centered_objects(self):
        image_np = self._load_image_np(
            os.path.join(os.path.dirname(__file__), "test_data", "chair_person.png"))
        pred = get_onnx_prediction(self.onnx_sess, image_np)
        reward = sum_centered_objects_present(pred)
        reward_scaled = prioritize_centered_objects(pred, {"person": 10,
                                                           "chair": 10})

        self.assertGreater(reward_scaled, reward)

    def test_nms(self):
        image_np = self._load_image_np(
            os.path.join(os.path.dirname(__file__), "test_data", "chair_person.png"))
        bboxes, intermediate = get_onnx_prediction(self.onnx_sess, image_np)

        nms_boxes = non_max_supression(bboxes)
        detections = detect_yolo_bboxes(nms_boxes)

        self.assertEqual(len(detections), 2)

        self.assertEqual(detections[0].class_name, "chair")
        self.assertAlmostEqual(detections[0].x, 5, delta=1.0)
        self.assertAlmostEqual(detections[0].y, 395, delta=1.0)
        self.assertAlmostEqual(detections[0].width, 103, delta=1.0)
        self.assertAlmostEqual(detections[0].height, 85, delta=1.0)

        self.assertEqual(detections[1].class_name, "person")
        self.assertAlmostEqual(detections[1].x, 1, delta=1.0)
        self.assertAlmostEqual(detections[1].y, 276, delta=1.0)
        self.assertAlmostEqual(detections[1].width, 82, delta=1.0)
        self.assertAlmostEqual(detections[1].height, 174, delta=1.0)


