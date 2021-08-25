import numpy as np
import torch

from actor_critic import core as core


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


class TorchReplayBuffer:
    def __init__(self, obs_dim, act_dim, size, device=None):
        self.device = device
        self.obs_buf = torch.zeros(core.combined_shape(size, obs_dim), device=device, dtype=torch.float32)
        self.obs2_buf = torch.zeros(core.combined_shape(size, obs_dim), device=device, dtype=torch.float32)
        self.act_buf = torch.zeros(core.combined_shape(size, act_dim), device=device, dtype=torch.float32)
        self.rew_buf = torch.zeros(size, device=device, dtype=torch.float32)
        self.done_buf = torch.zeros(size, device=device, dtype=torch.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        with torch.no_grad():
            # replace random entry when full
            if self.size == self.max_size:
                self.ptr = torch.randint(0, self.max_size, (), dtype=torch.int64, device=self.device)
            self.obs_buf[self.ptr] = torch.as_tensor(obs, device=self.device)
            self.obs2_buf[self.ptr] = torch.as_tensor(next_obs, device=self.device)
            self.act_buf[self.ptr] = torch.as_tensor(act, device=self.device)
            self.rew_buf[self.ptr] = torch.as_tensor(rew, device=self.device)
            self.done_buf[self.ptr] = torch.as_tensor(done, device=self.device)
            self.ptr += 1
            self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = torch.randint(0, self.size, (batch_size,), dtype=torch.int64, device=self.device)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return batch


class TorchLSTMReplayBuffer:
    def __init__(self, obs_dim, act_dim, size, device=None, mode="right_aligned"):
        self.device = device
        self.mode = mode
        self.obs_buf = torch.zeros(core.combined_shape(size, obs_dim), device=device, dtype=torch.float32)
        self.obs2_buf = torch.zeros(core.combined_shape(size, obs_dim), device=device, dtype=torch.float32)
        self.act_buf = torch.zeros(core.combined_shape(size, act_dim), device=device, dtype=torch.float32)
        self.rew_buf = torch.zeros(size, device=device, dtype=torch.float32)
        self.done_buf = torch.zeros(size, device=device, dtype=torch.float32)
        self.lstm_history_lens = torch.zeros(size, device="cpu", dtype=torch.int64)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, lstm_history_count, done):
        with torch.no_grad():
            # replace random entry when full
            if self.size == self.max_size:
                raise RuntimeError("In LSTM Mode, you can't exceed the size of your replay buffer")

            self.obs_buf[self.ptr] = torch.as_tensor(obs, device=self.device)
            self.obs2_buf[self.ptr] = torch.as_tensor(next_obs, device=self.device)
            self.act_buf[self.ptr] = torch.as_tensor(act, device=self.device)
            self.rew_buf[self.ptr] = torch.as_tensor(rew, device=self.device)
            self.done_buf[self.ptr] = torch.as_tensor(done, device=self.device)

            self.lstm_history_lens[self.ptr] = lstm_history_count
            self.ptr += 1
            self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = torch.randint(0, self.size, (batch_size,), dtype=torch.int64, device=self.device)
        lstm_indexes = torch.arange(0, torch.max(self.lstm_history_lens[idxs]), dtype=torch.int64)
        lstm_indexes = lstm_indexes.repeat(batch_size, 1)
        lstm_pads = torch.clone(lstm_indexes)

        if self.mode == "right_aligned":
            # Data is right aligned if the sequence is shorter than the max, and padded to zeros
            # If you want to just access the last N'th element, then you want to right align sequences
            lstm_indexes = lstm_indexes + (idxs.cpu() + 1 - torch.max(self.lstm_history_lens[idxs])).unsqueeze(-1)
            lstm_pads = (lstm_pads + torch.unsqueeze(self.lstm_history_lens[idxs] - torch.max(self.lstm_history_lens[idxs]), -1)) < 0
            lstm_history = self.obs_buf[lstm_indexes]
            lstm_history[lstm_pads] = 0
        elif self.mode == "padded_seq":
            # Data is packed left-aligned into a torch Padded Sequence for use in their LSTMs API
            lstm_indexes = lstm_indexes + (idxs.cpu() - self.lstm_history_lens[idxs] + 1).unsqueeze(-1)
            lstm_history = torch.nn.utils.rnn.pack_padded_sequence(self.obs_buf[lstm_indexes],
                                                                   lengths=self.lstm_history_lens[idxs],
                                                                   batch_first=True,
                                                                   enforce_sorted=False)

        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     lstm_history=lstm_history,
                     done=self.done_buf[idxs])
        return batch