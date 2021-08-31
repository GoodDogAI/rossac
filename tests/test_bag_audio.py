import rosbag
import os
import unittest
import numpy as np
import onnxruntime as rt
import soundfile
import tempfile


class TestBagAudio(unittest.TestCase):
    sample_rate = 48_000
    test_bag = "/home/jake/bagfiles/test_bags/record-brain-sac-upbeat-donkey-432-18944-samp0.0_2021-08-31-15-30-24_1.bag"

    def test_lstm_sample(self):
        bag = rosbag.Bag(self.test_bag, "r")

        inputs = []

        for topic, msg, ts in bag.read_messages(["/audio"]):
            inputs.append(np.frombuffer(msg.data, dtype=np.int32))

        print(len(inputs))

        audio = np.concatenate(inputs)

        with tempfile.NamedTemporaryFile("wb", suffix=".wav") as tf:
            soundfile.write(tf, audio, self.sample_rate)
            print(tf.name)