import argparse
import os.path
import sys

import numpy as np
import torch

from bot_env import RobotEnvironment
from actor_critic.core import MLPActorCritic
from sac import SoftActorCritic
from dump_onnx import export

# requires https://github.com/ArmyOfRobots/yolov5 to be cloned in ..\YOLOv5
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'yolov5'))

import models

DEFAULT_MAX_GAP_SECONDS = 5

def interpolate(pre_ts, pre_data, ts, post_ts, post_data):
    interval_len = post_ts - pre_ts
    if interval_len == 0: return pre_data
    target_offset = ts - pre_ts
    interval_diff = post_data - pre_data
    return pre_data + (interval_diff * target_offset / interval_len)

assert(interpolate(1, 1, 5, 11, 11) == 5)

def interpolate_events(primary, secondaries, max_gap_ns):
    primary = sorted(primary.items())
    secondaries = list(map(lambda dict: list(sorted(dict.items())), secondaries))

    result = []
    for ts, event in primary:
        interpolated = []
        for i in range(len(secondaries)):
            while len(secondaries[i]) > 0:
                if len(secondaries[i]) == 1:
                    # only one potentially correlated reading
                    secondary_ts, data = secondaries[i][0]
                    if abs(secondary_ts - ts) < max_gap_ns:
                        interpolated.append(data)
                    break
                elif secondaries[i][1][0] < ts:
                    # next and next+1 readings are both before "now"
                    # means we can drop next and advance to next+1
                    secondaries[i].pop(0)
                elif secondaries[i][0][0] >= ts:
                    # all readings are after "now", so we may only use the first
                    secondary_ts, data = secondaries[i][0]
                    if abs(secondary_ts - ts) < max_gap_ns:
                        interpolated.append(data)
                    break
                else:
                    # at this point next is before "now", and next+1 is after "now"
                    pre_ts, pre_data = secondaries[i][0]
                    post_ts, post_data = secondaries[i][1]
                    if abs(pre_ts - ts) < max_gap_ns and abs(post_ts - ts) < max_gap_ns:
                        data = interpolate(pre_ts, pre_data, ts, post_ts, post_data)
                        interpolated.append(data)
                    elif abs(pre_ts - ts) < max_gap_ns:
                        interpolated.append(pre_data)
                    elif abs(post_ts - ts) < max_gap_ns:
                        interpolated.append(post_data)
                    break
            if len(interpolated) != i + 1:
                break
        if len(interpolated) != len(secondaries):
            continue
        result.append((ts, event, interpolated))

    return result

def _simulate_trace(ticks, primary_interval, secondary_probs):
    import random

    primary = dict()
    secondaries = list(map(lambda x: dict(), secondary_probs))
    for tick in range(ticks):
        if tick % primary_interval == 0:
            primary[tick] = tick
        for i, prob in enumerate(secondary_probs):
            if prob > random.random():
                secondaries[i][tick] = tick
    return primary, secondaries

# test_primary, test_secondaries = _simulate_trace(100, 5, [0.3, 0.5])
# test_interpolated = interpolate_events(test_primary, test_secondaries, max_gap_ns=3)

def load_cmdvels(dir_path):
    import json

    cmdvels = dict()
    for name in os.listdir(dir_path):
        full_name = os.path.join(dir_path, name)
        _, ext = os.path.splitext(name)
        if os.path.isfile(full_name) and ext == '.cmd_vels':
            with open(full_name) as file:
                for line in file.readlines():
                    if len(line.strip()) == 0:
                        continue
                    cmdvel = json.loads(line)
                    ts = int(cmdvel['ts'])
                    reading = np.asarray([cmdvel['linear'][0]] + [cmdvel['angular'][2]])
                    cmdvels[ts] = reading
    
    return cmdvels

def load_backbone_outputs(dir_path):
    outputs = dict()

    for name in os.listdir(dir_path):
        full_name = os.path.join(dir_path, name)
        no_ext, ext = os.path.splitext(name)
        no_ext2, extra_ext = os.path.splitext(no_ext)
        if os.path.isfile(full_name) and ext == '.npy':
            ts = int(no_ext2 or no_ext)
            outputs[ts] = np.load(full_name)
    
    return outputs

def _flatten(arr):
    return np.reshape(arr, -1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir', type=str, help='directory with backbone outputs')
    parser.add_argument('--cmdvel-dir', type=str, help='directory with cmd_vels')
    parser.add_argument('--max-gap', type=int, default=DEFAULT_MAX_GAP_SECONDS, help='max gap in seconds')
    opt = parser.parse_args()

    print("CUDA: " + str(torch.cuda.is_available()))

    cmdvels = load_cmdvels(opt.cmdvel_dir)
    print("cmdvels: " + str(len(cmdvels)))

    backbone_outputs = load_backbone_outputs(opt.img_dir)
    print("backbone outputs: " + str(len(backbone_outputs)))

    interpolated = interpolate_events(cmdvels, [backbone_outputs], max_gap_ns=1000*1000*1000)
    print("matching events: " + str(len(interpolated)))

    # every 1000 entries in replay are ~500MB
    sac = SoftActorCritic(RobotEnvironment, replay_size=20000)

    for i in range(len(interpolated)-1):
        ts, act, observations = interpolated[i]
        _, _, future_observations = interpolated[i+1]
        reward = np.average(future_observations[0])
        end_of_episode = i%100 == 99
        sac.replay_buffer.store(_flatten(observations[0]),
            np.concatenate([act, np.array([700, 700])]),
            rew=reward,
            next_obs=_flatten(future_observations[0]),
            done=end_of_episode)       

    print("filled in replay buffer")
    
    for i in range(1000*1000*1000):
        sac.train(batch_size=32, batch_count=32)
        print('LossQ: ' + str(sac.logger.epoch_dict['LossQ'][-1]) +
              '  LossPi: ' + str(sac.logger.epoch_dict['LossPi'][-1]))
        model_name = f"sac-{i:05d}.onnx"
        export(sac.ac, model_name)
        print("saved " + model_name)
