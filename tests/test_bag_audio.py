import rosbag
import os
import unittest
import numpy as np
import onnxruntime as rt
import soundfile
import tempfile


class TestBagAudio(unittest.TestCase):
    sample_rate = 48_000
    test_bag = "/home/jake/bagfiles/test_bags/record-brain-sac-pleasant-frog-450-02048-samp0.2_2021-09-10-14-44-04_16.bag"

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