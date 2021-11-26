import rosbag
import os
import unittest
import numpy as np
import onnxruntime as rt
import soundfile
import tempfile


class TestBagAudio(unittest.TestCase):
    sample_rate = 48_000
    test_bag = "/home/jake/Downloads/record-brain-sac-young-hippogriff-465-15872-samp0.0_2021-11-26-11-44-20_0.bag"

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