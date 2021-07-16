import argparse
import os.path
import time

import wandb
import png
import glob
import rosbag
import dataclasses

from typing import Dict, Any, Callable, List
from dataclasses import dataclass, field
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import onnxruntime as rt

import tensorflow.compat.v1 as tf

from bot_env import RobotEnvironment, NormalizedRobotEnvironment
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

    # If you are the first bag in a series, don't output any entries until you received one message from each channel
    # This is because it can take some time for everything to fully initialize (ex. if building tensorrt models),
    # and we don't want bogus data points.
    wait_for_each_msg = bag_file.endswith("_0.bag")
    received_topic = defaultdict(bool)
    ros_topics = [opt.camera_topic,
                  '/dynamixel_workbench/dynamixel_state',
                  '/camera/accel/sample',
                  '/camera/gyro/sample',
                  '/head_feedback',
                  '/cmd_vel',
                  '/odrive_feedback',
                  '/vbus']

    for topic, msg, ts in bag.read_messages(ros_topics):
        full_ts = ts.nsecs + ts.secs * 1000000000

        received_topic[topic] = True
        if wait_for_each_msg and not all(received_topic[topic] for topic in ros_topics):
            continue

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

                try:
                    pred = get_prediction(onnx_sess, img_name)
                except png.FormatError:
                    # Clear the png file so you can retry if something went wrong
                    os.remove(img_name)
                    raise

                intermediate = get_intermediate_layer(pred)
                np.save(intermediate_name, intermediate, allow_pickle=False)

                reward = reward_func(pred)
                np.save(reward_name, reward, allow_pickle=False)

            if opt.backbone_slice:
                entries.yolo_intermediate[full_ts] = _flatten(np.load(intermediate_name, allow_pickle=False))[::opt.backbone_slice]
            else:
                # If you need the whole intermediate array, you can mmap it, but there is a 65k file limit
                entries.yolo_intermediate[full_ts] = _flatten(np.load(intermediate_name, allow_pickle=False, mmap_mode='r'))

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
    parser.add_argument('--batch-size', type=int, default=1024, help='number of samples per training step')
    parser.add_argument('--max-samples', type=int, default=20000, help='max number of training samples to load at once')
    parser.add_argument('--cpu', default=False, action="store_true", help='run training on CPU only')
    parser.add_argument('--reward-delay-ms', type=int, default=0, help='delay reward from action by the specified amount of milliseconds')
    # default rate for dropout assumes small inputs (order of 1024 elements)
    parser.add_argument('--dropout', type=float, default=0.88, help='input dropout rate for training')
    parser.add_argument('--backbone-slice', type=int, default=None, help='use every nth datapoint of the backbone')
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

    if opt.reward_delay_ms > 0:
        all_entries.reward = { ts + opt.reward_delay_ms * 1000000: reward for ts, reward in all_entries.reward.items() }

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
    backbone_slice = opt.backbone_slice
    env_fn = lambda: NormalizedRobotEnvironment(slice=backbone_slice)
    sac = SoftActorCritic(env_fn, replay_size=opt.max_samples, device=device, dropout=opt.dropout)

    # Save basic params to wandb configuration
    wandb.config.read_dir = opt.bag_dir
    wandb.config.reward_func_name = opt.reward
    wandb.config.num_samples = min(len(interpolated)-1, opt.max_samples)
    wandb.config.batch_size = opt.batch_size
    wandb.config.gamma = sac.gamma
    wandb.config.polyak = sac.polyak
    wandb.config.lr = sac.lr
    wandb.config.alpha = sac.alpha
    wandb.config.dropout = sac.dropout
    wandb.config.device = str(device)
    wandb.config.reward_delay_ms = opt.reward_delay_ms
    wandb.config.backbone_slice = backbone_slice

    wandb.watch(sac.ac, log="gradients", log_freq=100)  # Log gradients periodically

    env = env_fn()

    def normalize_pantilt(pantilt):
        return env.normalize_pan(pantilt[0, np.newaxis]), env.normalize_tilt(pantilt[1, np.newaxis])

    for i in range(wandb.config.num_samples):
        ts, backbone, (reward, cmd_vel, pantilt_command, pantilt_current, head_gyro, head_accel, odrive_feedback, vbus) = interpolated[i]
        _, future_backbone, _ = future_observations = interpolated[i+1]

        pan_command, tilt_command = normalize_pantilt(pantilt_command)
        pan_curr, tilt_curr = normalize_pantilt(pantilt_current)

        move_penalty = abs(cmd_vel).mean() * 0.02
        pantilt_penalty = float((abs(pan_command - pan_curr) + abs(tilt_command - tilt_curr)) * 0.01)
        reward -= move_penalty + pantilt_penalty

        sac.replay_buffer.store(obs=backbone,
            act=np.concatenate([cmd_vel, pan_command, tilt_command]),
            rew=reward,
            next_obs=future_backbone,
            done=False)

    print("filled in replay buffer")

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    np.set_printoptions(precision=2)

    for i in range(1000*1000*1000):
        sac.train(batch_size=opt.batch_size, batch_count=32)
        lossQ = sum(sac.logger.epoch_dict['LossQ'][-opt.batch_size:])/opt.batch_size
        lossPi = sum(sac.logger.epoch_dict['LossPi'][-opt.batch_size:])/opt.batch_size
        print(f"  LossQ: {lossQ}", end=None)
        print(f"  LossPi: {lossPi}", end=None)

        wandb.log({
            "LossQ": lossQ,
            "LossPi": lossPi,
        })

        sample_action = sac.logger.epoch_dict['Pi'][-1][0]
        print(f"  Sample Action: velocity {sample_action[0]:.2f}  angle {sample_action[1]:.2f}  pan {sample_action[2]:.1f}  tilt {sample_action[3]:.1f}")
        model_name = f"checkpoints/sac-{i:05d}.onnx"

        if i % 20 == 0:
            action_samples = sac.sample_actions(8).cpu()
            print(action_samples)
            samples_name = model_name + ".samples"
            with open(samples_name, 'w') as samples_file:
                print(f"  LossQ: {lossQ}", file=samples_file)
                print(f"  LossPi: {lossPi}", file=samples_file)
                print(action_samples, file=samples_file)
            export(sac.ac, device, model_name, sac.env)
            print("saved " + model_name)
