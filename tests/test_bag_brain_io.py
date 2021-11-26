import rosbag
import os
import unittest
import numpy as np
import onnxruntime as rt

from yolo_reward import get_onnx_prediction, non_max_supression, detect_yolo_bboxes


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
            image_np = image_np.reshape((480, 640, 3))
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