import rosbag
import unittest
import numpy as np
import onnxruntime as rt


class TestBagBrainIO(unittest.TestCase):
    def test_lstm_sample(self):
        bag = rosbag.Bag("/home/jake/bagfiles/test_bags/record-brain-test-lstm-samp0.0_2021-08-20-13-13-06_0.bag", "r")
        onnx = rt.InferenceSession("/home/jake/bagfiles/test_bags/test-lstm.onnx")
        history_size = 8

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
            np.testing.assert_almost_equal(out, pred, decimal=5)