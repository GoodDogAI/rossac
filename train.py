import argparse
import os.path
import sys
import json
from typing import Dict, Any, Callable

import numpy as np
import torch

from bot_env import RobotEnvironment
from actor_critic.core import MLPActorCritic
from sac import SoftActorCritic
from dump_onnx import export

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

def load_json_data(dir_path:str, extension:str, function: Callable[[Dict], Any]) -> Dict[int, Any]:
    result = dict()
    for name in os.listdir(dir_path):
        full_name = os.path.join(dir_path, name)
        _, ext = os.path.splitext(name)
        if os.path.isfile(full_name) and ext == extension:
            with open(full_name) as file:
                for line in file.readlines():
                    if len(line.strip()) == 0:
                        continue
                    data = json.loads(line)
                    ts = int(data['ts'])
                    reading = function(data)
                    result[ts] = reading

    return result

def load_backbone_outputs(dir_path):
    outputs = dict()

    for name in os.listdir(dir_path):
        full_name = os.path.join(dir_path, name)
        no_ext, ext = os.path.splitext(name)
        no_ext2, extra_ext = os.path.splitext(no_ext)
        if os.path.isfile(full_name) and ext == '.npy':
            ts = int(no_ext2 or no_ext)
            outputs[ts] = np.load(full_name, mmap_mode='r')
    
    return outputs

def _flatten(arr):
    return np.reshape(arr, -1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--read-dir', type=str, help='directory with images, backbones, and json files')
    parser.add_argument('--max-gap', type=int, default=DEFAULT_MAX_GAP_SECONDS, help='max gap in seconds')
    parser.add_argument('--batch-size', type=int, default=32, help='number of samples per training step')
    parser.add_argument('--max-samples', type=int, default=20000, help='max number of training samples to load at once')
    opt = parser.parse_args()

    print("CUDA: " + str(torch.cuda.is_available()))

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    cmdvels = load_json_data(opt.read_dir, '.cmd_vels', lambda json: np.asarray([json['linear'][0]] + [json['angular'][2]]))
    print("cmdvels: " + str(len(cmdvels)))

    dynamixel = load_json_data(opt.read_dir, '.dynamixel', lambda json: np.asarray([json['pan_state']] + [json['tilt_state']], dtype=np.float32))
    print("dynamixel: " + str(len(dynamixel)))

    rewards = load_json_data(opt.read_dir, '.rewards', lambda json: np.asarray([json['reward']]))
    print("rewards: " + str(len(rewards)))

    backbone_outputs = load_backbone_outputs(opt.read_dir)
    print("backbone outputs: " + str(len(backbone_outputs)))

    interpolated = interpolate_events(backbone_outputs, [cmdvels, dynamixel, rewards], max_gap_ns=1000*1000*1000)
    print("matching events: " + str(len(interpolated)))

    # every 1000 entries in replay are ~500MB
    sac = SoftActorCritic(RobotEnvironment, replay_size=opt.max_samples, device=device)

    for i in range(min(len(interpolated)-1, opt.max_samples)):
        ts, backbone, (cmdvel, pantilt, reward) = interpolated[i]
        _, future_backbone, (future_cmdvel, future_pantilt, future_reward) = future_observations = interpolated[i+1]

        sac.replay_buffer.store(obs=_flatten(backbone),
            act=np.concatenate([cmdvel, pantilt]),
            rew=reward,
            next_obs=_flatten(future_backbone),
            done=False)

    print("filled in replay buffer")

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    np.set_printoptions(precision=2)

    for i in range(1000*1000*1000):
        sac.train(batch_size=opt.batch_size, batch_count=32)
        print(f"  LossQ: {sum(sac.logger.epoch_dict['LossQ'][-opt.batch_size:])/opt.batch_size}", end=None)
        print(f"  LossPi: {sum(sac.logger.epoch_dict['LossPi'][-opt.batch_size:])/opt.batch_size}", end=None)
        sample_action = sac.logger.epoch_dict['Pi'][-1][0]
        print(f"  Sample Action: velocity {sample_action[0]:.2f}  angle {sample_action[1]:.2f}  pan {sample_action[2]:.1f}  tilt {sample_action[3]:.1f}")
        model_name = f"checkpoints/sac-{i:05d}.onnx"

        if i % 20 == 0:
            export(sac.ac, device, model_name)
            print("saved " + model_name)
