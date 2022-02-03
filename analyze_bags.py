import csv
import glob
import sys
import os
import torch

from typing import Dict

from tqdm import tqdm

import numpy as np
from bot_env import NormalizedRobotEnvironment, SlicedRobotEnvironment
from sac import TorchLSTMReplayBuffer, TorchReplayBuffer, SoftActorCritic
from split_dropout import SplitDropout
from train import read_bag


def analyze_bags(opt: Dict):
    bagcsv = csv.writer(open(os.path.join(opt.cache_dir, "bag_analysis.csv"), "w"), delimiter=",")
    device = torch.device("cuda")
    backbone_slice = opt.backbone_slice

    for bag_path in tqdm(glob.glob(os.path.join(opt.bag_dir, "*.bag"))):
        # Load entries into memory
        entries = read_bag(bag_path, opt.cache_dir, opt.onnx, opt.reward,
                           env=NormalizedRobotEnvironment(SlicedRobotEnvironment(slice=backbone_slice)),
                           interpolation_slice=backbone_slice,
                           reward_delay_ms=opt.reward_delay_ms,
                           base_reward_scale=opt.base_reward_scale,
                           punish_backtrack_ms=opt.punish_backtrack_ms)

        if len(entries) == 0:
            continue

        # Now, load that into a replay buffer
        env_fn = lambda: NormalizedRobotEnvironment(SlicedRobotEnvironment(slice=backbone_slice))


        actor_hidden_sizes = [int(s) for s in opt.actor_hidden_sizes.split(',')]
        critic_hidden_sizes = [int(s) for s in opt.critic_hidden_sizes.split(',')]
        history_indexes = [int(s) for s in opt.history_indexes.split(',')]

        assert history_indexes[
                   0] == -1, "First history index needs to be -1, and will be replaced with extra_obs during SAC bellman step"

        history_size = max(map(abs, history_indexes))
        if history_size > 1:
            replay_buffer_factory = lambda obs_dim, act_dim, size: TorchLSTMReplayBuffer(obs_dim=obs_dim,
                                                                                         act_dim=act_dim,
                                                                                         size=size,
                                                                                         device="gpu",
                                                                                         history_size=history_size)
        else:
            replay_buffer_factory = lambda obs_dim, act_dim, size: TorchReplayBuffer(obs_dim=obs_dim, act_dim=act_dim,
                                                                                     size=size,
                                                                                     device="gpu")

        actor_critic_args = {
            'actor_hidden_sizes': actor_hidden_sizes,
            'critic_hidden_sizes': critic_hidden_sizes,
            'history_indexes': history_indexes,
        }

        env = env_fn()

        example_entry = entries.iloc[0]
        backbone_data_size = example_entry.observation_yolo_intermediate_size
        example_observation = example_entry.observation
        dropout = SplitDropout([example_observation.shape[0] - backbone_data_size, backbone_data_size],
                               [0.05, opt.dropout])
        sac = SoftActorCritic(env_fn, replay_size=len(entries), device=device, dropout=dropout,
                              lr=opt.lr,
                              ac_kwargs=actor_critic_args,
                              replay_buffer_factory=replay_buffer_factory)

        resume_dict = sac.load(opt.checkpoint_path) if os.path.exists(opt.checkpoint_path) else None
        pretrained_dict = None
        if opt.pretrained_path:
            if resume_dict:
                raise 'if pretrained-path is specified, checkpoint-path must not already exist'
            pretrained_dict = sac.load(opt.pretrained_path)

        num_samples = min(len(entries) - 1, opt.max_samples)

        def ts_from_seconds(seconds):
            return int(seconds * 1000000000)

        MAX_TS_DIFF = ts_from_seconds(0.20)

        nans = 0
        oobs = 0
        dones = 0
        lstm_history_count = 0

        for i in tqdm(range(num_samples)):
            entry = entries.iloc[i]
            next_entry = entries.iloc[i + 1]

            obs = entry.observation
            future_obs = next_entry.observation

            if np.isnan(obs).any() or np.isnan(future_obs).any() or np.isnan(entry.reward).any():
                nans += 1
                continue

            if obs.max() > 1000 or future_obs.max() > 1000:
                oobs += 1
                continue

            if abs(next_entry.name - entry.name) > MAX_TS_DIFF:
                lstm_history_count = 0
                continue

            if entry.done:
                lstm_history_count = 0
                continue

            if next_entry.done:
                dones += 1

            if lstm_history_count >= history_size:
                lstm_history_count -= 1

            lstm_history_count += 1
            sac.replay_buffer.store(obs=obs,
                                    act=entry.action,
                                    rew=entry.reward,
                                    next_obs=future_obs,
                                    lstm_history_count=lstm_history_count,
                                    done=next_entry.done)

        bagcsv.writerow([bag_path])