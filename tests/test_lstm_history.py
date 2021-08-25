import rosbag
import os
import unittest
import numpy as np
import onnxruntime as rt

from bag_utils import read_bag
from replay_buffer import TorchLSTMReplayBuffer


class TestBagBrainIO(unittest.TestCase):
    def test_history_buffer_too_many_entries(self):
        replay = TorchLSTMReplayBuffer(obs_dim=2, act_dim=1, size=10, mode="right_aligned")

        for i in range(10):
            replay.store(obs=np.random.rand(2),
                         act=np.random.rand(1),
                         rew=0.0,
                         next_obs=np.random.rand(2),
                         lstm_history_count=1,
                         done=False)

        # Storing one more entry should crash, because the history buffer is full
        with self.assertRaises(Exception):
            replay.store(obs=np.random.rand(2),
                         act=np.random.rand(1),
                         rew=0.0,
                         next_obs=np.random.rand(2),
                         lstm_history_count=1,
                         done=False)

    def test_history_buffer_too_large_history(self):
        replay = TorchLSTMReplayBuffer(obs_dim=2, act_dim=1, size=10, mode="right_aligned")

        # Storing one more entry should crash, because the history buffer is not filled yet
        with self.assertRaises(Exception):
            replay.store(obs=np.random.rand(2),
                         act=np.random.rand(1),
                         rew=0.0,
                         next_obs=np.random.rand(2),
                         lstm_history_count=2,
                         done=False)

    def test_single_history_count(self):
        replay = TorchLSTMReplayBuffer(obs_dim=2, act_dim=1, size=10, mode="right_aligned")

        for i in range(10):
            replay.store(obs=np.random.rand(2),
                         act=np.random.rand(1),
                         rew=0.0,
                         next_obs=np.random.rand(2),
                         lstm_history_count=1,
                         done=False)

        batch = replay.sample_batch(10)

        for i in range(10):
            np.testing.assert_almost_equal(batch["obs"][i].numpy(), batch["lstm_history"][i, 0].numpy())

    def test_single_history_count(self):
        replay = TorchLSTMReplayBuffer(obs_dim=2, act_dim=1, size=10, mode="right_aligned")

        for i in range(10):
            replay.store(obs=np.random.rand(2),
                         act=np.random.rand(1),
                         rew=0.0,
                         next_obs=np.random.rand(2),
                         lstm_history_count=1,
                         done=False)

        batch = replay.sample_batch(100)

        for i in range(100):
            np.testing.assert_almost_equal(batch["obs"][i].numpy(), batch["lstm_history"][i, 0].numpy())

    def test_double_history_count(self):
        replay = TorchLSTMReplayBuffer(obs_dim=2, act_dim=1, size=10, mode="right_aligned")

        for i in range(10):
            replay.store(obs=np.random.rand(2),
                         act=np.random.rand(1),
                         rew=float(i),
                         next_obs=np.random.rand(2),
                         lstm_history_count=min(i + 1, 3),
                         done=False)

        batch = replay.sample_batch(100)

        for i in range(100):
            np.testing.assert_almost_equal(batch["obs"][i].numpy(), batch["lstm_history"][i, -1].numpy())


