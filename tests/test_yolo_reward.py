import unittest
import numpy as np
import os

import png
import onnxruntime as rt


from yolo_reward import get_onnx_prediction, sum_centered_objects_present, prioritize_centered_objects, \
    non_max_supression, load_png
from yolo_reward import detect_yolo_bboxes


class TestYoloReward(unittest.TestCase):
    onnx_path = os.path.join(os.path.dirname(__file__), "..", "yolov5l_6_0_op11_rossac.onnx")

    def setUp(self) -> None:
        self.onnx_sess = rt.InferenceSession(self.onnx_path)

    def _get_sum_centered_objects_present(self, image_np) -> float:
        bboxes, intermediate = get_onnx_prediction(self.onnx_sess, image_np)
        bboxes = non_max_supression(bboxes)
        return sum_centered_objects_present(bboxes)

    def test_sum_centered_objects_present(self):
        reward = self._get_sum_centered_objects_present(load_png(
            os.path.join(os.path.dirname(__file__), "test_data", "chair_person.png")))

        # The image has a person and chair in the lower left corner, so rolling the image
        # to the right by 50 pixels, should increase the reward
        reward_roll_right = self._get_sum_centered_objects_present(load_png(
            os.path.join(os.path.dirname(__file__), "test_data", "chair_person_shift_right1.png")))
        self.assertGreater(reward_roll_right, reward)

        # Rolling the graphic so it's almost in the center
        reward_shift_center = self._get_sum_centered_objects_present(load_png(
            os.path.join(os.path.dirname(__file__), "test_data", "chair_person_shift_center.png")))
        self.assertGreater(reward_shift_center, reward_roll_right)

        # Rolling the graphic so it's moving past into the right side of the screen should be less
        reward_shift_offcenter = self._get_sum_centered_objects_present(load_png(
            os.path.join(os.path.dirname(__file__), "test_data", "chair_person_shift_offcenter.png")))
        self.assertGreater(reward_shift_offcenter, reward)
        self.assertLess(reward_shift_offcenter, reward_shift_center)

        reward_shift_center_scaledup = self._get_sum_centered_objects_present(load_png(
            os.path.join(os.path.dirname(__file__), "test_data", "chair_person_shift_center_scaledup.png")))
        self.assertGreater(reward_shift_center_scaledup, reward_shift_offcenter)

        # Roll the image up
        reward_roll_up = self._get_sum_centered_objects_present(np.roll(load_png(
            os.path.join(os.path.dirname(__file__), "test_data", "chair_person.png")), -50, axis=0))
        self.assertGreater(reward_roll_up, reward)

        # Flipping the image left/right should be around the same reward
        reward_flip_lr = self._get_sum_centered_objects_present(np.flip(load_png(
            os.path.join(os.path.dirname(__file__), "test_data", "chair_person.png")), axis=1))
        self.assertLess(abs(reward - reward_flip_lr), 1.0)

    def test_prioritize_centered_objects(self):
        image_np = load_png(
            os.path.join(os.path.dirname(__file__), "test_data", "chair_person.png"))
        bboxes, intermediate = get_onnx_prediction(self.onnx_sess, image_np)
        bboxes = non_max_supression(bboxes)

        reward = sum_centered_objects_present(bboxes)
        reward_scaled = prioritize_centered_objects(bboxes, {"person": 10,
                                                             "chair": 10})

        self.assertAlmostEqual(reward_scaled, reward * 10, places=5)

    def test_nms(self):
        image_np = load_png(
            os.path.join(os.path.dirname(__file__), "test_data", "chair_person.png"))
        bboxes, intermediate = get_onnx_prediction(self.onnx_sess, image_np)

        nms_boxes = non_max_supression(bboxes)
        detections = detect_yolo_bboxes(nms_boxes)

        self.assertEqual(len(detections), 2)

        self.assertEqual(detections[0].class_name, "chair")
        self.assertAlmostEqual(detections[0].x, 0, delta=1.0)
        self.assertAlmostEqual(detections[0].y, 396, delta=1.0)
        self.assertAlmostEqual(detections[0].width, 107, delta=1.0)
        self.assertAlmostEqual(detections[0].height, 83, delta=1.0)

        self.assertEqual(detections[1].class_name, "person")
        self.assertAlmostEqual(detections[1].x, 1, delta=1.0)
        self.assertAlmostEqual(detections[1].y, 277, delta=1.0)
        self.assertAlmostEqual(detections[1].width, 81, delta=1.0)
        self.assertAlmostEqual(detections[1].height, 165, delta=1.0)

    def test_indoor_image(self):
        image_np = load_png(
            os.path.join(os.path.dirname(__file__), "test_data", "person_indoor.png"))
        bboxes, intermediate = get_onnx_prediction(self.onnx_sess, image_np)

        nms_boxes = non_max_supression(bboxes)
        detections = detect_yolo_bboxes(nms_boxes)

        self.assertEqual(detections[0].class_name, "person")
        self.assertAlmostEqual(detections[0].x, 314, delta=1)
        self.assertAlmostEqual(detections[0].y, 381, delta=1)

