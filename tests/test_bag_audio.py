import rosbag
import os
import unittest
import numpy as np
import onnxruntime as rt
import soundfile
import tempfile


class TestBagAudio(unittest.TestCase):
    sample_rate = 48_000
    test_bag = "/home/jake/bagfiles/newbot_nov21/record-brain-sac-unique-paper-476-09600-samp0.0_2021-12-16-21-30-33_21.bag"

    def test_bag_audio(self):
        bag = rosbag.Bag(self.test_bag, "r")

        inputs = []

        for topic, msg, ts in bag.read_messages(["/audio"]):
            inputs.append(np.frombuffer(msg.data, dtype=np.float32))

        print(len(inputs))

        audio = np.concatenate(inputs)

       # with tempfile.NamedTemporaryFile("wb", suffix=".wav") as tf:
        with open("test.wav", "wb") as tf:
            soundfile.write(tf, audio, self.sample_rate)
            print(tf.name)