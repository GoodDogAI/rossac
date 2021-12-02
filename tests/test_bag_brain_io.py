import rosbag
import os
import unittest
import numpy as np
import onnxruntime as rt

from train import read_bag_into_numpy, create_dataset, get_timed_entry, TimedEntryNotFound
from yolo_reward import get_onnx_prediction, non_max_supression, detect_yolo_bboxes


class TestGetTimedEntry(unittest.TestCase):
    def test_basic(self):
        entries = {
            0: 1.0,
            10: 2.0,
            20: 3.0
        }

        self.assertEqual(get_timed_entry(entries, 0, 0), 1.0)
        self.assertEqual(get_timed_entry(entries, 0, 1), 2.0)
        self.assertEqual(get_timed_entry(entries, 0, 2), 3.0)

        with self.assertRaises(TimedEntryNotFound):
            get_timed_entry(entries, 0, 3)

        with self.assertRaises(TimedEntryNotFound):
            get_timed_entry(entries, 0, -1)

        self.assertEqual(get_timed_entry(entries, 10, 0), 2.0)
        self.assertEqual(get_timed_entry(entries, 10, -1), 1.0)
        self.assertEqual(get_timed_entry(entries, 10, 1), 3.0)

        with self.assertRaises(TimedEntryNotFound):
            get_timed_entry(entries, 10, 2)


class TestBagYoloIntermediate(unittest.TestCase):
    test_bag = "/home/jake/bagfiles/test_bags/yolov5s_intermediate_color_test.bag"
    onnx_path = os.path.join(os.path.dirname(__file__), "..", "yolov5s_trt.onnx")

    def test_yolo_color(self):
        bag = rosbag.Bag(self.test_bag, "r")
        onnx_sess = rt.InferenceSession(self.onnx_path)

        inputs = []
        outputs = []

        for topic, msg, ts in bag.read_messages(["/processed_img"]):
            img = []
            for i in range(0, len(msg.data), msg.step):
                img.append(np.frombuffer(msg.data[i:i + msg.step], dtype=np.uint8))
            image_np = np.array(img)
            image_np = image_np.reshape((msg.height, msg.width, -1))
            inputs.append(image_np)

        for topic, msg, ts in bag.read_messages(["/yolo_intermediate"]):
            outputs.append(np.array(msg.data))

        print(f"Loaded {len(inputs)} inputs, {len(outputs)} outputs")

        # Check that inputs match outputs through
        for inp, out in zip(inputs, outputs):
            bboxes, intermediate = get_onnx_prediction(onnx_sess, inp)
            intermediate = intermediate.reshape(-1)
            nms_boxes = non_max_supression(bboxes)
            detections = detect_yolo_bboxes(nms_boxes, threshold=0.25)

            print("Diff: ", np.linalg.norm(out - intermediate))
            np.testing.assert_almost_equal(out, intermediate, decimal=4)

class TestBagBrainInputs(unittest.TestCase):
    test_bag = "/home/jake/bagfiles/test_bags/test_intermediate_values1.bag"
    onnx_path = os.path.join(os.path.dirname(__file__), "..", "yolov5s_trt.onnx")

    def test_yolo_brain_inputs(self):
        bag = rosbag.Bag(self.test_bag, "r")
        onnx_sess = rt.InferenceSession(self.onnx_path)

        inputs = {}

        for topic, msg, ts in bag.read_messages(["/brain_inputs"]):
            full_ts = ts.nsecs + ts.secs * 1000000000
            inputs[full_ts] = np.array(msg.data)

        bag_entries = read_bag_into_numpy(self.test_bag,
                                          reward_delay_ms=0,
                                          punish_backtrack_ms=0)

        dataset_entries = create_dataset(bag_entries,
                                         backbone_onnx_path=self.onnx_path,
                                         reward_func_name="prioritize_centered_spoons_with_nms",
                                         interpolation_slice=157)

        print(f"Loaded {len(inputs)} inputs")

        for ts, inputs in inputs.items():
            matching_ds = sorted([ds for ds in dataset_entries if ds.ts < ts], key=lambda ds: -ds.ts)

            if not matching_ds:
                continue

            ds = matching_ds[0]
            last_inputs = inputs[-990:]

            diffs = last_inputs - ds.observation

        # Check that inputs match outputs through
        # for inp, out in zip(inputs, outputs):
        #     bboxes, intermediate = get_onnx_prediction(onnx_sess, inp)
        #     intermediate = intermediate.reshape(-1)
        #     nms_boxes = non_max_supression(bboxes)
        #     detections = detect_yolo_bboxes(nms_boxes, threshold=0.25)
        #
        #     print("Diff: ", np.linalg.norm(out - intermediate))
        #     np.testing.assert_almost_equal(out, intermediate, decimal=4)


class TestBagBrainIO(unittest.TestCase):
    test_bag = "/home/jake/bagfiles/test_bags/record-brain-sac-pleasant-frog-450-02048-samp0.0_2021-09-06-13-24-24_3.bag"
    test_onnx = "/home/jake/hipow-checkpoints/sac-pleasant-frog-450-02048.onnx"
    test_yolo_onnx = os.path.join(os.path.dirname(__file__), "..", "yolov5s.onnx")

    def test_lstm_sample(self):
        bag = rosbag.Bag(self.test_bag, "r")
        onnx = rt.InferenceSession(self.test_onnx)
        history_size = 240

        inputs = []
        outputs = []

        for topic, msg, ts in bag.read_messages(["/brain_inputs"]):
            inputs.append(np.array(msg.data))

        for topic, msg, ts in bag.read_messages(["/brain_outputs"]):
            outputs.append(np.array(msg.data))

        print("Loaded")
        print([len(x) for x in inputs])

        # Check all outputs the same size
        self.assertTrue(all(len(x) == len(outputs[0]) for x in outputs))

        # Check that inputs match outputs through
        for inp, out in zip(inputs, outputs):
            inp = np.reshape(inp, (1, history_size, len(inp) // history_size))
            inp = inp.astype(np.float32)

            pred = onnx.run(["actions"], {
                "yolo_intermediate": inp
            })
            pred = pred[0][0] # First index to get the actions array as output, second to get first batch only

            print(np.linalg.norm(out - pred))
            np.testing.assert_almost_equal(out, pred, decimal=4)