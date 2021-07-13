import argparse
import os.path
import time

import wandb
import json
import png
import glob
import rosbag
import dataclasses

from typing import Dict, Any, Callable, List
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import onnxruntime as rt

import tensorflow.compat.v1 as tf

from bot_env import RobotEnvironment
from actor_critic.core import MLPActorCritic
from sac import SoftActorCritic
import yolo_reward
from yolo_reward import get_prediction, get_intermediate_layer
from dump_onnx import export

DEFAULT_MAX_GAP_SECONDS = 5

tf.disable_v2_behavior()
wandb.init(project="sac-series1", entity="armyofrobots")


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


@dataclass
class BagEntries:
    # yolo_intermediate layers, as the base prediction
    yolo_intermediate: Dict[int, np.ndarray] = field(default_factory=dict)
    reward: Dict[int, float] = field(default_factory=dict)

    # pan, tilt in steps, (1024 steps = 300 degrees)
    dynamixel_cur_state: Dict[int, np.ndarray] = field(default_factory=dict)
    dynamixel_command_state: Dict[int, np.ndarray] = field(default_factory=dict)

    # commanded forward speed, rotational speed
    cmd_vel: Dict[int, np.ndarray] = field(default_factory=dict)

    # Each wheels actual speed in rotations per second
    odrive_feedback: Dict[int, np.ndarray] = field(default_factory=dict)

    # head gyro rate, in radians per second
    head_gyro: Dict[int, np.ndarray] = field(default_factory=dict)

    # head acceleration, in meters per second
    head_accel: Dict[int, np.ndarray] = field(default_factory=dict)

    # robot bus voltage, in Volts
    vbus: Dict[int, np.ndarray] = field(default_factory=dict)


def read_bag(bag_file: str, reward_func_name: str) -> BagEntries:
    print(f"Opening {bag_file}")
    bag = rosbag.Bag(bag_file, 'r')
    entries = BagEntries()
    reward_func = getattr(yolo_reward, reward_func_name)
    onnx_sess = None

    # TODO, Don't write any entries until you get at least one message from each channel

    for topic, msg, ts in bag.read_messages([opt.camera_topic,
                                             '/dynamixel_workbench/dynamixel_state',
                                             '/camera/accel/sample',
                                             '/camera/gyro/sample',
                                             '/head_feedback',
                                             '/cmd_vel',
                                             '/odrive_feedback',
                                             '/vbus']):
        full_ts = ts.nsecs + ts.secs * 1000000000

        if topic == opt.camera_topic:
            # Save off the image, the YOLO Intermediate data, and calculate the reward
            img_name = os.path.join(opt.bag_dir, "_cache", f"{full_ts}.png")

            if not os.path.isfile(img_name):
                img = []
                for i in range(0, len(msg.data), msg.step):
                    img.append(msg.data[i:i + msg.step])

                img_mode = 'L' if "infra" in opt.camera_topic else 'RGB'
                png.from_array(img, mode=img_mode).save(img_name)

            intermediate_name = os.path.join(opt.bag_dir, "_cache", f"{full_ts}.intermediate.npy")
            reward_name = os.path.join(opt.bag_dir, "_cache", f"{full_ts}.reward_{reward_func_name}.npy")

            if not os.path.isfile(intermediate_name) or not os.path.isfile(reward_name):
                if not onnx_sess:
                    onnx_sess = rt.InferenceSession(opt.onnx)

                pred = get_prediction(onnx_sess, img_name)
                intermediate = get_intermediate_layer(pred)
                np.save(intermediate_name, intermediate, allow_pickle=False)

                reward = reward_func(pred)
                np.save(reward_name, reward, allow_pickle=False)

            entries.yolo_intermediate[full_ts] = np.load(intermediate_name, allow_pickle=False, mmap_mode='r')
            entries.reward[full_ts] = np.load(reward_name, allow_pickle=False)
        elif topic == '/dynamixel_workbench/dynamixel_state':
            entries.dynamixel_cur_state[full_ts] = np.array([msg.dynamixel_state[0].present_position,
                                                             msg.dynamixel_state[1].present_position])
        elif topic == "/head_feedback":
            entries.dynamixel_command_state[full_ts] = np.array([msg.pan_command,
                                                                 msg.tilt_command])
        elif topic == "/cmd_vel":
            entries.cmd_vel[full_ts] = np.array([msg.linear.x,
                                                 msg.angular.z])
        elif topic == "/camera/accel/sample":
            entries.head_accel[full_ts] = np.array([msg.linear_acceleration.x,
                                                    msg.linear_acceleration.y,
                                                    msg.linear_acceleration.z])
        elif topic == "/camera/gyro/sample":
            entries.head_gyro[full_ts] = np.array([msg.angular_velocity.x,
                                                   msg.angular_velocity.y,
                                                   msg.angular_velocity.z])
        elif topic == "/odrive_feedback":
            entries.odrive_feedback[full_ts] = np.array([msg.motor_vel_actual_0,
                                                         msg.motor_vel_actual_1])
        elif topic == "/vbus":
            entries.vbus[full_ts] = np.array([msg.data])
        else:
            raise KeyError("Unexpected rosbag topic")

    return entries


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag-dir', type=str, help='directory with bag files to use for training data')
    parser.add_argument("--onnx", type=str, default='./yolov5s.onnx', help='onnx weights path for intermediate stage')
    parser.add_argument("--camera_topic", default='/camera/infra2/image_rect_raw')
    parser.add_argument("--reward", default='sum_centered_objects_present')
    parser.add_argument('--max-gap', type=int, default=DEFAULT_MAX_GAP_SECONDS, help='max gap in seconds')
    parser.add_argument('--batch-size', type=int, default=32, help='number of samples per training step')
    parser.add_argument('--max-samples', type=int, default=20000, help='max number of training samples to load at once')
    parser.add_argument('--cpu', default=False, action="store_true", help='run training on CPU only')
    opt = parser.parse_args()

    if torch.cuda.is_available() and not opt.cpu:
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = None
        print("Using CPU")

    cache_dir = os.path.join(opt.bag_dir, "_cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    start_load = time.perf_counter()
    all_entries = BagEntries()

    with ThreadPoolExecutor() as executor:
        entry_futures = executor.map(lambda bag_path: read_bag(bag_path, opt.reward),
                                     glob.glob(os.path.join(opt.bag_dir, "*.bag")))

        for entries in entry_futures:
            print(f"Adding {len(entries.reward)} states to entries")

            # Merge all of the fields into one datastructure
            for field in dataclasses.fields(entries):
                getattr(all_entries, field.name).update(getattr(entries, field.name))

    print(f"Loaded {len(all_entries.yolo_intermediate)} backbone outputs")
    print(f"Loaded {len(all_entries.reward)} rewards")
    print(f"Loaded {len(all_entries.cmd_vel)} cmd_vels")
    print(f"Loaded {len(all_entries.dynamixel_command_state)} dynamixel commands")
    print(f"Loaded {len(all_entries.dynamixel_cur_state)} dynamixel states")
    print(f"Loaded {len(all_entries.head_gyro)} head gyros")
    print(f"Loaded {len(all_entries.head_accel)} head accels")
    print(f"Loaded {len(all_entries.odrive_feedback)} odrive feedbacks")
    print(f"Loaded {len(all_entries.vbus)} vbus")
    print(f"Took {time.perf_counter() - start_load}")

    interpolated = interpolate_events(all_entries.yolo_intermediate, [all_entries.reward,
                                                                      all_entries.cmd_vel,
                                                                      all_entries.dynamixel_command_state,
                                                                      all_entries.dynamixel_cur_state,
                                                                      all_entries.head_gyro,
                                                                      all_entries.head_accel,
                                                                      all_entries.odrive_feedback,
                                                                      all_entries.vbus], max_gap_ns=1000*1000*1000)
    print("matching events: " + str(len(interpolated)))

    # every 1000 entries in replay are ~500MB
    sac = SoftActorCritic(RobotEnvironment, replay_size=opt.max_samples, device=device)

    # Save basic params to wandb configuration
    wandb.config.read_dir = opt.bag_dir
    wandb.config.reward_func_name = opt.reward
    wandb.config.num_samples = min(len(interpolated)-1, opt.max_samples)
    wandb.config.batch_size = opt.batch_size
    wandb.config.device = str(device)

    wandb.watch(sac.ac, log="gradients", log_freq=100)  # Log gradients periodically

    for i in range(wandb.config.num_samples):
        ts, backbone, (reward, cmd_vel, pantilt_command, pantilt_current, head_gyro, head_accel, odrive_feedback, vbus) = interpolated[i]
        _, future_backbone, _ = future_observations = interpolated[i+1]

        sac.replay_buffer.store(obs=_flatten(backbone),
            act=np.concatenate([cmd_vel, pantilt_command]),
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

        wandb.log({
            "LossQ": sum(sac.logger.epoch_dict['LossQ'][-opt.batch_size:])/opt.batch_size,
            "LossPi": sum(sac.logger.epoch_dict['LossPi'][-opt.batch_size:])/opt.batch_size,
        })

        sample_action = sac.logger.epoch_dict['Pi'][-1][0]
        print(f"  Sample Action: velocity {sample_action[0]:.2f}  angle {sample_action[1]:.2f}  pan {sample_action[2]:.1f}  tilt {sample_action[3]:.1f}")
        model_name = f"checkpoints/sac-{i:05d}.onnx"

        if i % 20 == 0:
            action_samples = sac.sample_actions(8).cpu()
            print(action_samples)
            export(sac.ac, device, model_name)
            print("saved " + model_name)
