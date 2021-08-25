import unittest
import random
import numpy as np

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

        batch = replay.sample_batch(100)

        for i in range(100):
            np.testing.assert_almost_equal(batch["obs"][i].numpy(), batch["lstm_history"][i, 0].numpy())

    def test_double_history_count(self):
        replay = TorchLSTMReplayBuffer(obs_dim=20, act_dim=1, size=100, device="cuda:0", mode="right_aligned")

        for i in range(100):
            replay.store(obs=np.random.rand(20),
                         act=np.random.rand(1),
                         rew=float(i),
                         next_obs=np.random.rand(20),
                         lstm_history_count=min(i + 1, random.randint(1, 10)),
                         done=False)

        for iteration in range(100):
            batch = replay.sample_batch(100)

            for i in range(100):
                self.assertGreater(np.sum(np.abs(batch["obs"][i].cpu().numpy())), 1.0)
                np.testing.assert_almost_equal(batch["obs"][i].cpu().numpy(), batch["lstm_history"][i, -1].cpu().numpy())


