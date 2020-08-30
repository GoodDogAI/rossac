import argparse
import os.path
import sys

import numpy as np
import torch

from bot_env import RobotEnvironment
from sac import SoftActorCritic

# requires https://github.com/ArmyOfRobots/yolov5 to be cloned in ..\YOLOv5
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'YOLOv5'))

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
    secondaries = list(map(lambda dict: list(sorted(dict)), secondaries))

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir', type=str, help='directory with backbone outputs')
    parser.add_argument('--cmdvel-dir', type=str, help='directory with cmd_vels')
    parser.add_argument('--max-gap', type=int, default=DEFAULT_MAX_GAP_SECONDS, help='max gap in seconds')
    opt = parser.parse_args()

    # every 1000 entries in replay are ~500MB
    sac = SoftActorCritic(RobotEnvironment, replay_size=20000)
    print(sac.replay_buffer)
